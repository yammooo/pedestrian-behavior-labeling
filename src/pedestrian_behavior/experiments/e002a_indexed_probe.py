"""Bounded shared coarse/fine storage, complete-track loading and CUDA readout probe."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import identity_digest
from pedestrian_behavior.data.indexed_points import ShardWriter, IndexedShard, PROTOCOL, build_frame
from pedestrian_behavior.data.multimodal import collate_multimodal
from pedestrian_behavior.data.preparation import checksum, native_float, valid_pose
from pedestrian_behavior.datasets import loki
from pedestrian_behavior.evaluation import track_cross_entropy
from pedestrian_behavior.inspection.feature_probe import box_membership, relative_xyz
from pedestrian_behavior.models.multimodal import MultimodalClassifier
from pedestrian_behavior.training import seed_run
from .e002a_calibrate import fixture_samples, measure
from .e002a_point_check import context
from .e002a_pooling import fixture_union
from .e002a_probe import write_json


def source_arrays(path, protected):
    protected[str(path)] = checksum(path)
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files if k != "__identity__"}


def verify_independent(arrays, root, sha, frame):
    """Compare supplied values with fresh in-memory construction, including geometric flags."""
    payload, meta = build_frame(arrays)
    reader = IndexedShard(root, sha)
    try:
        for selection in meta["selections"]:
            start, count = selection["index_start"], selection["count"]
            rows = payload["indices"][start:start+count]
            actual = reader.crop(frame, selection["locator"])
            np.testing.assert_array_equal(actual[:, :64], payload["features"][rows].astype(np.float32))
            box = np.array(selection["box"])
            expected = relative_xyz(payload["coordinates"][rows],box[:3],arrays["world_from_native"],np.array(selection["anchor"]))
            np.testing.assert_allclose(actual[:,64:67],expected/20,atol=1e-7)
            np.testing.assert_array_equal(actual[:,67].astype(bool),box_membership(payload["coordinates"][rows],box[:3],box[3:6],box[6]))
    finally:
        reader.close()


def prepare_population(args, record, protected):
    previous = json.loads(args.measurement.read_text())
    if previous["status"] != "complete" or not previous["protected_unchanged"]:
        raise ValueError("Completed pooling measurement required")
    source = json.loads((args.probe/"probe.json").read_text())
    outputs = args.probe.resolve().parents[2]
    contexts, grouped = {}, defaultdict(list)
    for item in previous["frames"]:
        grouped[item["dataset"],item["group"]].append(item)
    for (dataset,group), items in grouped.items():
        path = args.output/"population"/f"{dataset}-{group}"
        writer = ShardWriter(path,{"dataset":dataset,"group":group,"scope":"bounded-population-frames"})
        try:
            for item in items:
                if dataset == "loki":
                    original_frame = next(f for f in source["frames"] if f["dataset"]==dataset and f["group"]==group and str(f["key"])==str(item["key"]))
                    arrays = fixture_union(original_frame,args.probe,source,outputs,protected,contexts)
                else:
                    key = identity_digest(item["cache_identity"])
                    original = args.point_cache/"entries"/(key+".npz")
                    if checksum(original) != previous["protected"][str(original.resolve())]:
                        raise ValueError("Selected original cache changed")
                    arrays = source_arrays(original,protected)
                before = time.perf_counter()
                meta = writer.add(item["key"],arrays)
                record["population"].append({k:item[k] for k in ("dataset","group","key","split","reasons")} | {
                    "shard":str(path.relative_to(args.output)),"prepare_seconds":time.perf_counter()-before,
                    "original_fp32_diagnostic_bytes":item["modes"]["original"]["storage"]["fp32"]["bytes"],
                    "original_crop_tokens":sum(c[0] for c in item["modes"]["original"]["crop_counts"]),
                    "metadata":meta})
            sha = writer.finish()
            verify_independent(arrays,path,sha,items[-1]["key"])
            record["shards"][str(path.relative_to(args.output))] = sha
        finally:
            writer.close()
        print("Population",dataset,group,len(items),flush=True)
        write_json(args.output/"prototype.json",record)


def native_box(reader, group, key, track, frozen, paths):
    if reader.dataset == "loki":
        frame = loki.read_frame(reader.root/group,key,paths[key])
        row = frame.pedestrians[track].label3d
        center,size = (np.array([native_float(row[p+a]) for a in "xyz"]) for p in ("pos_","dim_"))
        return np.r_[center,size,native_float(row["yaw"])]
    row = reader.lidar[frozen["association_ids"][track],key]
    p = "[LiDARBoxComponent].box."
    return np.array([row[p+field+a] for field in ("center.","size.") for a in "xyz"]+[row[p+"heading"]])


def prepare_tracks(args,record,protected):
    execution = json.loads((args.fixtures/"execution.json").read_text())
    source = json.loads((args.probe/"probe.json").read_text())
    outputs = args.probe.resolve().parents[2]
    grouped = defaultdict(list)
    for item in execution["tracks"]:
        if item["split"] not in ("training","validation"):
            raise ValueError("Held-out track forbidden")
        rgb = (args.fixtures/item["rgb_cache"]).resolve()
        protected[str(rgb)] = checksum(rgb)
        grouped[item["dataset"],item["group"]].append(item)
    for (dataset,group), items in grouped.items():
        collection,manifest,setup,reader = context(dataset,outputs,source["config"],args.probe,protected)
        reader.index_scene(group)
        frozen = manifest["scenes"][group]
        actual = reader.index_scene(group)["provenance"]
        fields = ("annotation_odometry_sha256",) if dataset=="loki" else ("component_sha256","association_ids")
        if any(actual[k]!=frozen[k] for k in fields):
            raise ValueError("Native inputs differ from frozen scene")
        frames = defaultdict(list)
        for item in items:
            matches = [e for e in setup["tracks"] if e["locator"]==item["locator"] and not e["exclusions"] and e["split"]==item["split"]]
            if len(matches)!=1:
                raise ValueError("Resource track outside frozen source population")
            protected[str(collection/"tracks"/item["archive"])] = checksum(collection/"tracks"/item["archive"])
            target = {k:item[k] for k in ("name","dataset","group","locator","slots","split")}
            target["refs"] = [None]*item["slots"]
            record["tracks"].append(target)
            for slot,ref in enumerate(item["point_frames"]):
                if ref is not None:
                    frames[item["source_frames"][slot]].append((item,target,slot,ref))
        keys = set(frames)|{item["source_frames"][0] for item in items}
        poses = reader.read_frames(group,keys)
        paths = dict(loki.frame_files(reader.root/group)) if dataset=="loki" else None
        path = args.output/"tracks"/f"{dataset}-{group}"
        writer = ShardWriter(path,{"dataset":dataset,"group":group,"scope":"six-complete-development-tracks"})
        try:
            for key,participants in sorted(frames.items(),key=lambda p:int(p[0])):
                original = args.fixtures/participants[0][3]["cache"]
                raw = source_arrays(original.resolve(),protected)
                locators,boxes,anchors = [],[],[]
                for item,target,slot,ref in participants:
                    if (args.fixtures/ref["cache"]).resolve()!=original.resolve():
                        raise ValueError("Conflicting shared-frame resource fixtures")
                    locators.append(item["locator"])
                    boxes.append(native_box(reader,group,key,item["track_id"],frozen,paths))
                    anchors.append(poses[item["source_frames"][0]]["ego_pose"])
                    target["refs"][slot] = {"shard":str(path.relative_to(args.output)),"frame":str(key)}
                pose = poses[key]["ego_pose"]
                world = pose@reader.extrinsic if dataset=="loki" else pose
                if not valid_pose(world):
                    raise ValueError("Missing native transform")
                arrays = {k:raw[k] for k in ("features","coordinates","native_indices")}
                arrays.update(locators=np.array(locators),boxes=np.array(boxes,np.float64),anchors=np.array(anchors,np.float64),world_from_native=np.array(world,np.float64))
                for box,anchor,(_,_,_,ref) in zip(boxes,anchors,participants):
                    rows=raw[ref["prefix"]+"rows"]
                    np.testing.assert_allclose(relative_xyz(raw["coordinates"][rows],box[:3],world,anchor),raw[ref["prefix"]+"offsets"],atol=1e-4)
                    np.testing.assert_array_equal(box_membership(raw["coordinates"][rows],box[:3],box[3:6],box[6]),raw[ref["prefix"]+"inside"])
                writer.add(key,arrays)
            sha = writer.finish()
            record["shards"][str(path.relative_to(args.output))] = sha
            verify_independent(arrays,path,sha,key)
        finally:
            writer.close()
        print("Complete tracks",dataset,group,len(frames),flush=True)
        write_json(args.output/"prototype.json",record)


def open_shards(args,record,scope):
    return {name:IndexedShard(args.output/name,sha) for name,sha in record["shards"].items() if name.startswith(scope+"/")}


def load_track_points(item,readers):
    return [torch.from_numpy(readers[ref["shard"]].crop(ref["frame"],item["locator"]))
            if ref else torch.zeros(0,68) for ref in item["refs"]]


def benchmark_loading(args,record):
    order = np.random.default_rng(0).permutation(len(record["tracks"])).tolist()
    for phase in ("warm-files-fresh-verification","warm-files-memoized","advisory-cold-fresh-verification"):
        if phase != "warm-files-memoized":
            readers = open_shards(args,record,"tracks")
        if phase.startswith("advisory"):
            record["advisory_eviction_requested"] = all([r.evict_pages() for r in readers.values()])
            readers = open_shards(args,record,"tracks")
        snapshots = {name:dict(r.stats) for name,r in readers.items()}
        for i in order:
            item = record["tracks"][i]
            before = time.perf_counter()
            points = load_track_points(item,readers)
            seconds = time.perf_counter()-before
            record["loads"].append({"phase":phase,"name":item["name"],"dataset":item["dataset"],
                "slots":len(points),"tokens":sum(len(f) for f in points),"seconds":seconds})
            del points
        delta = {k:sum(r.stats[k]-snapshots[name][k] for name,r in readers.items()) for k in next(iter(readers.values())).stats}
        record["load_traffic"][phase] = delta
        if phase != "warm-files-fresh-verification":
            for reader in readers.values():reader.close()
    print("Indexed loading complete",flush=True)


def benchmark_model(args,record):
    if args.device=="cuda" and torch.cuda.mem_get_info()[0] < 10*2**30:
        raise RuntimeError("Prototype needs10GiB free VRAM; no active process is stopped")
    profile = json.loads(args.profile.read_text())
    if profile["status"]!="complete":
        raise ValueError("Completed resource profile required")
    readers = open_shards(args,record,"tracks")
    samples = dict(fixture_samples(args.fixtures))
    try:
        for item in record["tracks"]:
            sample = samples.pop(item["name"])
            sample["points"] = load_track_points(item,readers)
            sample["flags"][:,2] = torch.tensor([len(f)>0 for f in sample["points"]],dtype=torch.float32)
            for count in (1,2):
                row = measure(sample,"g-rgb-dditr",count,profile["profile"],"ordinary",args.device)
                if row["peak_allocated"] and row["peak_allocated"]>profile["allocated_ceiling"]:
                    raise ValueError("Prototype exceeded calibrated VRAM headroom")
                record["model"].append(row|{"name":item["name"],"dataset":item["dataset"],"phase":"train-update"})
            seed_run(0)
            model = MultimodalClassifier("g-rgb-dditr",profile["profile"]).to(args.device).eval()
            batch = collate_multimodal([sample])
            timings=[]
            with torch.no_grad():
                for repeat in range(3):
                    if args.device=="cuda":torch.cuda.synchronize()
                    before=time.perf_counter()
                    logits=model(batch)
                    loss=track_cross_entropy(logits,batch["targets"].to(args.device),batch["gt_valid"].to(args.device),batch["padding_mask"].to(args.device)).mean()
                    if not torch.isfinite(loss):raise ValueError("Nonfinite prototype loss")
                    if args.device=="cuda":torch.cuda.synchronize()
                    if repeat:timings.append(time.perf_counter()-before)
            record["model"].append({"name":item["name"],"dataset":item["dataset"],"phase":"validation-forward",
                "tokens":sum(len(f) for f in sample["points"]),"slots":len(sample["points"]),"tracks":1,"seconds":float(np.median(timings))})
            del model,batch,sample,logits,loss
            gc.collect()
            if args.device=="cuda":torch.cuda.empty_cache()
            print("Model",item["name"],flush=True)
            write_json(args.output/"prototype.json",record)
    finally:
        for reader in readers.values():reader.close()


def run(args):
    args.output = args.output.resolve()
    outputs = args.probe.resolve().parents[2]
    if not args.output.is_relative_to(outputs/"experiments/E002a") or args.output.is_relative_to(args.point_cache.resolve()):
        raise ValueError("Use a fresh dedicated E002a diagnostic output")
    if shutil.disk_usage(args.output.parent).free < 25*10**9:
        raise OSError("Keep at least25GB free before bounded preparation")
    args.output.mkdir(parents=True,exist_ok=False)
    seed_run(0);torch.set_num_threads(2)
    protected={str(p.resolve()):checksum(p) for p in (args.measurement,args.probe/"probe.json",args.fixtures/"execution.json",args.profile)}
    record={"status":"running","started_utc":datetime.now(timezone.utc).isoformat(),"command":sys.orig_argv,
        "revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"protocol":PROTOCOL,
        "torch":str(torch.__version__),"numpy":np.__version__,"device":args.device,
        "population":[],"tracks":[],"shards":{},"loads":[],"load_traffic":{},"model":[],"protected":protected,
        "scope":"Bounded prototype only. Original36 frames for sizing, six complete source development tracks for loading/model timings. Tracks' saved unions contain selected fixture participants, not the full population. No encoder inference, production writes, comparison training or W&B. Advisory eviction only own immutable shard files; no global drop_caches.512 memoized verified chunks per scene."}
    begin=time.perf_counter()
    try:
        prepare_population(args,record,protected)
        prepare_tracks(args,record,protected)
        benchmark_loading(args,record)
        benchmark_model(args,record)
        record["status"]="complete"
    except BaseException:
        record["status"]="failed"
        (args.output/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        record["protected_unchanged"]=all(p.exists() and checksum(p)==sha for name,sha in protected.items() for p in [Path(name)])
        record["elapsed_seconds"]=time.perf_counter()-begin
        record["bytes"]=sum(p.stat().st_size for p in args.output.rglob("*.bin"))
        record["finished_utc"]=datetime.now(timezone.utc).isoformat()
        write_json(args.output/"prototype.json",record)
        if not record["protected_unchanged"]:raise ValueError("Protected inputs changed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path("outputs/experiments/E002a")
    parser.add_argument("--probe",type=Path,default=root/"native-probe-07")
    parser.add_argument("--measurement",type=Path,default=root/"output-pooling-20261022/measurement.json")
    parser.add_argument("--point-cache",type=Path,default=root/"point-cache")
    parser.add_argument("--fixtures",type=Path,default=root/"execution-probe-04")
    parser.add_argument("--profile",type=Path,default=root/"geometry-revision-calibration/profile.json")
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())
