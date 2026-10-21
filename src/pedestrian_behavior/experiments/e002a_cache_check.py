"""Bounded native/GPU acceptance for production E002a caches; no fitting or held-out access."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import FeatureCache, gb_bytes
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.inspection.feature_probe import box_membership, relative_xyz
from pedestrian_behavior.inspection.native_features import native_scene
from .e002a_features import (FeatureCatalog, FeatureSupply, FrozenEncoders, add_arguments,
                             extraction_identity, load_configuration, selected_tracks, validate_roots)
from .e002a_probe import write_json


def rng_fingerprint():
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode())
    state = np.random.get_state()
    digest.update(repr((state[0],state[2:])).encode()); digest.update(state[1].tobytes())
    digest.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for value in torch.cuda.get_rng_state_all():
            digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_same(left, right):
    if set(left) != set(right):
        raise AssertionError("Payload fields changed")
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def run(args):
    validate_roots(args,{})
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    record = {"status":"incomplete","started_utc":datetime.now(timezone.utc).isoformat(),
        "config":{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        "command":sys.orig_argv,"revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "diff":subprocess.check_output(["git","diff","HEAD"],text=True),"checks":[],"failures":[],
        "scope":"bounded frozen source train/validation features only; no behavior fitting, test inference or W&B"}
    write_json(output/"acceptance.json",record)
    begin=time.perf_counter();protected={};encoders=None;telemetry=log=None
    try:
        if args.selection is None or args.frames_per_track < 1:
            raise ValueError("Explicit development selection and positive bounded frame count required")
        config,transform=load_configuration(args.probe)
        validate_roots(args,config)
        datasets=("loki","road-waymo") if args.dataset=="both" else (args.dataset,)
        catalogs={d:FeatureCatalog(args.outputs_root,d,config,transform) for d in datasets}
        for c in catalogs.values():protected.update(c.protected)
        for path in (args.selection,args.probe/"probe.json",transform,Path(config["dino_weights"]),Path(config["ditr_weights"]),Path(config["ditr_config"])):
            protected[str(path.resolve())]=checksum(path)
        selected=selected_tracks(catalogs,args.selection)
        if len(selected)>10:
            raise ValueError("Cache acceptance is limited to ten preselected development tracks")
        identity=extraction_identity(args.device,config)
        record["extraction"]=identity
        encoders=FrozenEncoders(config,args.device)
        random.seed(47);np.random.seed(47);torch.manual_seed(47)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        if args.device=="cuda":
            torch.cuda.reset_peak_memory_stats()
            log=(output/"system.csv").open("w")
            telemetry=subprocess.Popen(["nvidia-smi","--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu","--format=csv","-l","1"],stdout=log,stderr=subprocess.STDOUT)
        state=rng_fingerprint()
        references={};point_refs=[]
        with FeatureCache(args.rgb_cache_root,min_free_bytes=gb_bytes(args.min_free_gb)) as rgb, \
             FeatureCache(args.point_cache_root,capacity_bytes=gb_bytes(args.point_cache_gb),min_free_bytes=gb_bytes(args.min_free_gb)) as points:
            if rgb.initial["entries"] or points.initial["entries"]:
                raise ValueError("Acceptance requires empty dedicated caches; previous evidence is never overwritten")
            supply=FeatureSupply(catalogs,rgb,points,encoders,identity)
            record["initial"]={"rgb":rgb.initial,"points":points.initial}
            # Full RGB tracks, including all missing/unlabeled context; no crop pixels retained.
            for d,k in selected:
                t=catalogs[d].tracks[k]
                values=supply.rgb(d,k)
                references[d,k]=values
                record["checks"].append({"check":"rgb-cold","dataset":d,"locator":k,"slots":t["slots"],
                    "valid":int(values["valid"].sum()),"missing":int((~values["valid"]).sum())})
                usable=np.flatnonzero(t["point_usable"])
                indices=np.linspace(0,len(usable)-1,min(args.frames_per_track,len(usable)),dtype=int) if len(usable) else []
                point_refs.extend((d,k,int(usable[i])) for i in indices)
            record["rgb_cold"]=rgb.snapshot()
            if rng_fingerprint()!=state:raise AssertionError("RGB miss/init changed training RNG")
            for d,k in selected:assert_same(references[d,k],supply.rgb(d,k))
            record["rgb_warm"]=rgb.snapshot()
            # A bounded point request list, not full-population warming.
            point_values={}
            for d,k,slot in point_refs:
                values=supply.points(d,k,slot)
                point_values[d,k,slot]=values
                record["checks"].append({"check":"point-cold","dataset":d,"locator":k,"slot":slot,
                    "tokens":len(values["features"]),"inside_box":int(values["inside_box"].sum())})
                print("Point cold",d,slot,len(values["features"]),flush=True)
            record["point_cold"]=points.snapshot()
            if rng_fingerprint()!=state:raise AssertionError("Point miss/init changed training RNG")
            calls=dict(encoders.calls)
            encoders.release()
            for d,k,slot in point_refs:assert_same(point_values[d,k,slot],supply.points(d,k,slot))
            if encoders.calls!=calls or encoders.encoder is not None:
                raise AssertionError("A hit loaded/called a frozen encoder")
            record["point_warm"]=points.snapshot()
            # Independently decode/re-encode one native frame per source, bypassing cache construction.
            for d in catalogs:
                d_,k,slot=next(ref for ref in point_refs if ref[0]==d)
                catalog=catalogs[d];track=catalog.tracks[k];group=track["group"]
                reader=catalog.native_reader(group)
                frame,=native_scene(reader,group,[track["source_keys"][slot]],[track["track_id"]],catalog.manifest["scenes"][group],include_rgb=False)
                for p in frame["files"].values():
                    if p and Path(p).is_file():protected[str(Path(p).resolve())]=checksum(p)
                features,_,selected_indices,inverse=encoders("points",frame["points"])
                xyz=frame["points"][selected_indices,:3]
                box=frame["boxes"][track["track_id"]]
                offsets=relative_xyz(xyz,box[0],frame["world_from_native"],catalog.anchors[k])
                keep=np.linalg.norm(offsets[:,:2],axis=1)<=20
                expected={"features":features[keep],"coordinates":xyz[keep],"native_indices":selected_indices[keep],
                    "relative_xyz":offsets[keep],"inside_box":box_membership(xyz[keep],*box)}
                assert_same(expected,point_values[d,k,slot])
                np.testing.assert_array_equal(inverse[selected_indices],np.arange(len(selected_indices)))
                np.savez(output/(d+"-native-reference.npz"),**expected)
                record["checks"].append({"check":"independent-full-sweep-native-reencode","dataset":d,
                    "native_points":len(frame["points"]),"encoder_voxels":len(selected_indices),"crop_voxels":int(keep.sum())})
                # Another pedestrian in that exact frame must reuse the same union entry.
                members=catalog.frames[group,str(track["source_keys"][slot])]
                others=[m for m in members if m!=k]
                if others:
                    other=others[0];ot=catalog.tracks[other]
                    oslot=next(i for i,key in enumerate(ot["source_keys"]) if key==track["source_keys"][slot] and ot["point_usable"][i])
                    before=dict(encoders.calls)
                    shared=supply.points(d,other,oslot)
                    if before!=encoders.calls:raise AssertionError("Shared frame was encoded again")
                    record["checks"].append({"check":"shared-pedestrian-union-hit","dataset":d,"tokens":len(shared["features"])})
            # Persisted LRU capacity reduction, then a miss must reproduce the same full crop.
            measured_costs={ref:supply.point_cost(*ref) for ref in point_refs}
            points.trim(0)
            if any(supply.point_cost(*ref)!=value for ref,value in measured_costs.items()):
                raise AssertionError("Eviction lost measured input-only costs")
            record["checks"].append({"check":"point-costs-survive-eviction","references":len(measured_costs)})
            if points.snapshot()["resident_bytes"]:raise AssertionError("Lowered capacity did not trim")
            points.trim(gb_bytes(args.point_cache_gb))
            d,k,slot=point_refs[0]
            before=dict(encoders.calls)
            assert_same(point_values[d,k,slot],supply.points(d,k,slot))
            if encoders.calls["points"]!=before["points"]+1:raise AssertionError("Eviction did not cause native encoding")
            record["after_capacity_reduction"]=points.snapshot()
            record["checks"].append({"check":"evicted-entry-reencode-identical","passed":True})
            if rng_fingerprint()!=state:raise AssertionError("Cache activity changed training RNG")
        # New owner/process-equivalent reopen: hits must not initialize any encoder/native reader.
        encoders.release()
        calls=dict(encoders.calls)
        with FeatureCache(args.rgb_cache_root,min_free_bytes=gb_bytes(args.min_free_gb)) as rgb, \
             FeatureCache(args.point_cache_root,capacity_bytes=gb_bytes(args.point_cache_gb),min_free_bytes=gb_bytes(args.min_free_gb)) as points:
            supply=FeatureSupply(catalogs,rgb,points,encoders,identity)
            for d,k in selected:assert_same(references[d,k],supply.rgb(d,k))
            d,k,slot=point_refs[0]
            assert_same(point_values[d,k,slot],supply.points(d,k,slot))
            if encoders.calls!=calls or encoders.encoder is not None:raise AssertionError("Reopen hit invoked encoder")
            record["reopened"]={"rgb":rgb.snapshot(),"points":points.snapshot()}
        record["checks"].append({"check":"RNG-and-deterministic-settings-preserved","passed":torch.are_deterministic_algorithms_enabled()})
        if not torch.are_deterministic_algorithms_enabled():raise AssertionError("Determinism was not restored")
        record["status"]="complete"
    except BaseException:
        record["status"]="failed";record["failures"].append(traceback.format_exc());raise
    finally:
        if telemetry is not None:telemetry.terminate();telemetry.wait()
        if log is not None:log.close()
        if encoders is not None:
            encoders.release()
            record["encoders"]={"calls":encoders.calls,"loads":encoders.loads,"seconds":encoders.seconds}
        changed=[p for p,h in protected.items() if not Path(p).is_file() or checksum(p)!=h]
        record.update(protected_files=len(protected),changed_files=changed,elapsed_seconds=time.perf_counter()-begin,
            peak_host_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else 0,
            finished_utc=datetime.now(timezone.utc).isoformat())
        if changed:record["status"]="failed"
        write_json(output/"protected.json",protected);write_json(output/"acceptance.json",record)
        if changed:raise ValueError("Protected input changed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--frames-per-track",type=int,default=2)
    run(parser.parse_args())
