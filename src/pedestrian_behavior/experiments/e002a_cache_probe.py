"""Replay frozen source train/validation references through a hypothetical LRU."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.inspection.cache_simulation import ByteLRU
from .e002a_probe import write_json


GB = 10**9


def point_slots(metadata, position_valid, pose_valid):
    """Input-availability references only; no behavior labels or GT mask."""
    keys = metadata["source_frames"]
    if len(keys) != len(position_valid) or len(keys) != len(pose_valid) or len(keys) != len(metadata["availability"]):
        raise ValueError("Misaligned saved track")
    return [str(k) for i,k in enumerate(keys) if k is not None and metadata["anchor_valid"]
            and position_valid[i] and pose_valid[i] and metadata["availability"][i].get("native_3d",False)]


def load_trace(root, dataset, protected):
    setup_path=root/"experiments/E001/data-setup"/dataset/"setup.json"
    collection=root/"experiments/E001/reader-preparation"/dataset
    manifest=collection/"manifest.json"
    setup=json.loads(setup_path.read_text())
    if setup["status"] != "complete" or setup["collection"]["manifest_sha256"] != checksum(manifest):
        raise ValueError("Frozen source setup/collection mismatch")
    protected[str(setup_path)]=checksum(setup_path);protected[str(manifest)]=checksum(manifest)
    catalog=[];lookup={};tracks={s:[] for s in ("training","validation")};locators={s:[] for s in tracks};slots={s:0 for s in tracks}
    for entry in setup["tracks"]:
        split=entry["split"]
        if entry["exclusions"] or split not in tracks:continue
        archive=collection/"tracks"/entry["archive"]
        protected[str(archive)]=checksum(archive)
        with np.load(archive,allow_pickle=False) as a:
            m=json.loads(str(a["metadata"]))
            if m["track_locator"] != entry["locator"] or setup["groups"][entry["group"]] != split:
                raise ValueError("Track identity/split mismatch")
            keys=point_slots(m,a["ped_position_valid"],a["ego_pose_valid"])
        if len(m["source_frames"]) != entry["slots"]:raise ValueError("Track length mismatch")
        requests=[]
        for key in keys:
            identity=(entry["group"],key)
            if identity not in lookup:
                lookup[identity]=len(catalog)
                catalog.append({"group":identity[0],"key":identity[1],"split":split})
            frame=lookup[identity]
            if catalog[frame]["split"] != split:raise ValueError("Native frame crosses source splits")
            requests.append(frame)
        tracks[split].append(requests);locators[split].append(entry["locator"]);slots[split]+=entry["slots"]
    return {"catalog":catalog,"tracks":tracks,"locators":locators,"slots":slots}


def replay_orders(count, epochs, seed=0):
    # Same persistent generator/base-seed/randperm consumption as track_batches;
    # default integer collation changes no RNG state or sampling order.
    loader=DataLoader(range(count),batch_size=64,shuffle=True,drop_last=False,
                      generator=torch.Generator().manual_seed(seed),num_workers=0)
    return [[i for batch in loader for i in batch.tolist()] for _ in range(epochs)]


def frame_sizes(trace, probe, dataset, model, scale):
    catalog=trace["catalog"];sizes=np.zeros(len(catalog),np.int64)
    for split in ("training","validation"):
        indexes=[i for i,f in enumerate(catalog) if f["split"]==split]
        footprint=next(e["union_bytes"] for e in probe["estimates"] if e["dataset"]==dataset and e["split"]==split)*scale
        if not indexes:raise ValueError("Empty source split trace")
        weights=np.ones(len(indexes),np.float64)
        if model=="group-varying":
            measured=defaultdict(list)
            for f in probe["frames"]:
                if f["dataset"]==dataset and f["split"]==split and "random" in f["reasons"] and "voxels" in f:
                    measured[f["group"]].append(f["union_tokens"]*67*4)
            means={g:float(np.mean(v)) for g,v in measured.items() if np.mean(v)>0}
            observed=[means[g] for g in sorted(means)]
            if not observed:raise ValueError("No positive sampled group sizes")
            for j,i in enumerate(indexes):
                group=catalog[i]["group"]
                pick=int.from_bytes(hashlib.sha256(f"{dataset}/{group}".encode()).digest()[:8],"big")%len(observed)
                weights[j]=means.get(group,observed[pick])
        elif model != "uniform":raise ValueError("Unknown size scenario")
        sizes[indexes]=np.maximum(1,np.rint(footprint*weights/weights.sum())).astype(np.int64)
    return sizes.tolist()


def simulate(trace, orders, sizes, capacity):
    cache=ByteLRU(capacity);records=[]
    for epoch,order in enumerate(orders,1):
        for split in ("training","validation"):
            cache.reset_counts()
            indexes=order if split=="training" else range(len(trace["tracks"][split]))
            for i in indexes:
                for frame in trace["tracks"][split][i]:cache.access(frame,sizes[frame],epoch)
            records.append({"epoch":epoch,"split":split,**cache.snapshot()})
    return records


def summary(rows):
    total=lambda field:sum(r[field] for r in rows)
    requests=total("requests");hits=total("hits")
    return {"hit_rate":hits/requests if requests else None,"requests_per_epoch":requests/len({r['epoch'] for r in rows}),
        "misses_per_epoch":total("misses")/len({r['epoch'] for r in rows}),
        "read_GB_per_epoch":total("read_bytes")/GB/len({r['epoch'] for r in rows}),
        "write_GB_per_epoch":total("write_bytes")/GB/len({r['epoch'] for r in rows}),
        "evicted_GB_per_epoch":total("evicted_bytes")/GB/len({r['epoch'] for r in rows}),
        "prior_epoch_hits_per_epoch":total("prior_epoch_hits")/len({r['epoch'] for r in rows})}


def report(output,record):
    lines=["# E002a bounded output-cache simulation","",f"Status: **{record['status']}**.","",
        "Metadata replay only. Complete frozen eligible source training/validation tracks; no test archives, "
        "native decoding, encoder calls, actual feature-cache construction or training. One immutable shared "
        "entry per source scene/native frame, containing the required voxel-feature union. Misses mean "
        "hypothetical extraction/write; LRU eviction is byte-based.400 GB means400,000,000,000 bytes (372.5 GiB).", "",
        "Each source is simulated separately from an empty cache. Seed0 persistent PyTorch whole-track "
        "shuffle matches existing E001 loader reference; source validation runs in setup order after every "
        "epoch through the same cache. E002a memory-bucketing/frame-prefetch order is not implemented and "
        "can change locality. One feature request per usable point slot; GT/unlabeled status never selects "
        "requests. Active recomputation assumes its CPU inputs remain available until backward.", "",
        "Frame identities/accesses come from saved input-validity masks, not full native audits. Sizes "
        "are hypothetical, calibrated separately to earlier15 m source/split union-byte extrapolations; "
        "new20 m per-entry sizes are unknown. Uniform entries and group-varying entries bracket one "
        "locality sensitivity: observed random-group means where available, deterministic imputation "
        "for other groups, then normalization to the same split totals. Scale.75/1/1.5/2 is sensitivity, "
        "not a confidence interval. Cross-group density/popularity correlations remain unmeasured.", "",
        "Hit rates count avoided encoder requests relative to literal track-sequential on-demand supply. "
        "An optimized loader may already share repeated frames within RAM/microbatches; this simulation "
        "does not attribute those hits uniquely to disk caching. Byte totals exclude headers, indexes, "
        "atomic-write headroom and compression; cache lookup, actual read/write latency, GPU overlap and "
        "epoch wall time are not measured. Intra-epoch and prior-epoch hits are retained separately.", "",
        "[Numerical results and provenance](simulation.json) · [Protected inputs](protected.json)", "",
        "| Source | Tracks train/val | Requests train/val | Unique entries train/val | Nominal footprint train/val GB |", "|---|---:|---:|---:|---:|"]
    for d,v in record.get("traces",{}).items():
        lines.append(f"| {d} | {v['tracks']['training']}/{v['tracks']['validation']} | {v['requests']['training']:,}/{v['requests']['validation']:,} | {v['entries']['training']:,}/{v['entries']['validation']:,} | {v['nominal_GB']['training']:.1f}/{v['nominal_GB']['validation']:.1f} |")
    lines += ["","Mean across epochs2–6 (epoch1 cold results and per-split/epoch records retained numerically):","",
        "| Source | Size model | Size multiplier | Capacity GB | Train hit % | Val hit % | Combined hit % | Writes GB/epoch | Evicted GB/epoch |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in record["scenarios"]:
        if r["capacity_GB"]!=400:continue
        a=r["steady"]
        lines.append(f"| {r['dataset']} | {r['size_model']} | {r['size_scale']:g} |400| {100*a['training']['hit_rate']:.1f} | {100*a['validation']['hit_rate']:.1f} | {100*a['combined']['hit_rate']:.1f} | {a['combined']['write_GB_per_epoch']:.1f} | {a['combined']['evicted_GB_per_epoch']:.1f} |")
    lines += ["", "Capacity controls at nominal size (uniform entries), epochs2–6:", "",
        "| Source | Capacity GB | Train hit % | Val hit % | Combined hit % | Writes GB/epoch |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in record["scenarios"]:
        if r["size_model"] != "uniform" or r["size_scale"] != 1:continue
        a=r["steady"];cap="unbounded" if r["capacity_GB"] is None else str(r["capacity_GB"])
        lines.append(f"| {r['dataset']} | {cap} | {100*a['training']['hit_rate']:.1f} | {100*a['validation']['hit_rate']:.1f} | {100*a['combined']['hit_rate']:.1f} | {a['combined']['write_GB_per_epoch']:.1f} |")
    (output/"report.md").write_text("\n".join(lines)+"\n")


def run(args):
    root=args.outputs_root.resolve();output=args.output.resolve()
    if not output.is_relative_to(root/"experiments/E002a"):raise ValueError("Output must be under E002a")
    output.mkdir(parents=True,exist_ok=False)
    probe_path=args.probe.resolve()/"probe.json";probe=json.loads(probe_path.read_text())
    if probe["status"]!="complete":raise ValueError("Complete reference probe required")
    protected={str(probe_path):checksum(probe_path)}
    record={"status":"incomplete","started_utc":datetime.now(timezone.utc).isoformat(),"seed":0,"epochs":6,
        "torch":torch.__version__,"numpy":np.__version__,"command":__import__('sys').orig_argv,
        "revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "git_diff":subprocess.check_output(["git","diff","HEAD"],text=True),
        "source_sha256":{str(p):checksum(p) for p in (Path(__file__),Path(__file__).parents[1]/"inspection/cache_simulation.py",Path("scripts/simulate-e002a-cache.py"))},
        "traces":{},"scenarios":[],"failures":[]}
    started=time.perf_counter()
    write_json(output/"simulation.json",record)
    try:
        for dataset in ("loki","road-waymo"):
            trace=load_trace(root,dataset,protected)
            orders=replay_orders(len(trace["tracks"]["training"]),6)
            arrays={"training_order":np.array(orders,np.int32)}
            for split in ("training","validation"):
                arrays[split+"_offsets"]=np.r_[0,np.cumsum([len(t) for t in trace["tracks"][split]])]
                arrays[split+"_requests"]=np.array([i for t in trace["tracks"][split] for i in t],np.int32)
                arrays[split+"_locators"]=np.array(trace["locators"][split])
            np.savez_compressed(output/(dataset+"-trace.npz"),**arrays)
            write_json(output/(dataset+"-catalog.json"),trace["catalog"])
            info={"tracks":{s:len(t) for s,t in trace["tracks"].items()},"slots":trace["slots"],
                "requests":{s:sum(map(len,t)) for s,t in trace["tracks"].items()},
                "entries":{s:sum(f["split"]==s for f in trace["catalog"]) for s in trace["tracks"]},
                "nominal_GB":{s:next(e["union_bytes"] for e in probe["estimates"] if e["dataset"]==dataset and e["split"]==s)/GB for s in trace["tracks"]}}
            record["traces"][dataset]=info
            cases=[("uniform",1,0),("uniform",1,None)]
            cases += [(model,scale,400) for model in ("uniform","group-varying") for scale in (.75,1,1.5,2)]
            cases += [(model,1,capacity) for model in ("uniform","group-varying") for capacity in (8,200,600)]
            for model,scale,capacity in cases:
                sizes=frame_sizes(trace,probe,dataset,model,scale)
                rows=simulate(trace,orders,sizes,None if capacity is None else capacity*GB)
                steady=[r for r in rows if r["epoch"]>1]
                item={"dataset":dataset,"size_model":model,"size_scale":scale,"capacity_GB":capacity,
                    "modeled_working_set_bytes":sum(sizes),"max_entry_bytes":max(sizes),"epochs":rows,
                    "cold":summary([r for r in rows if r["epoch"]==1]),
                    "steady":{**{s:summary([r for r in steady if r["split"]==s]) for s in ("training","validation")},"combined":summary(steady)}}
                if capacity is not None:assert all(r["resident_bytes"]<=capacity*GB for r in rows)
                assert all(r["requests"]==r["hits"]+r["misses"] for r in rows)
                record["scenarios"].append(item);write_json(output/"simulation.json",record);report(output,record)
                print(dataset,model,scale,capacity,"steady hit",round(item["steady"]["combined"]["hit_rate"],4),flush=True)
        record["status"]="complete"
    except BaseException:
        record["status"]="failed";record["failures"].append(traceback.format_exc());raise
    finally:
        changed=[p for p,d in protected.items() if checksum(p)!=d]
        record.update(protected_files=len(protected),changed_files=changed,elapsed_seconds=time.perf_counter()-started,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        if changed:record["status"]="failed"
        write_json(output/"protected.json",protected);write_json(output/"simulation.json",record);report(output,record)
        if changed:raise ValueError("Protected inputs changed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root",type=Path,default=Path("outputs"))
    parser.add_argument("--probe",type=Path,default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
