"""Bounded RGB-CLS caching and D-DITR feature-supply/memory measurements."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum, load_track, valid_pose
from pedestrian_behavior.inspection.execution_probe import ExecutionGraph, MODES, compare_gradients, crop_tokens, select_tracks
from pedestrian_behavior.inspection.feature_probe import box_membership, normalized_crop, relative_xyz, square_crop
from pedestrian_behavior.inspection.native_features import native_camera, native_scene
from pedestrian_behavior.inspection.pretrained import ImageEncoder, PointEncoder
from .e002a_point_check import context
from .e002a_probe import write_json


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def loaded_track(item, output):
    frames = []
    for ref in item["point_frames"]:
        if ref is None:
            frames.append(torch.zeros(0, 68))
            continue
        with np.load(output/ref["cache"], allow_pickle=False) as a:
            rows = a[ref["prefix"]+"rows"]
            value = np.c_[a["features"][rows], a[ref["prefix"]+"offsets"]/20, a[ref["prefix"]+"inside"]].astype(np.float32)
            frames.append(torch.from_numpy(value))
    with np.load(output/item["rgb_cache"], allow_pickle=False) as a:
        rgb = torch.from_numpy(a["cls"].copy())
        valid = torch.from_numpy(a["valid"].astype(np.float32))
    return frames, rgb, valid


def report(output, record):
    lines = ["# E002a execution and feature-supply probe", "", f"Status: **{record['status']}**.", "",
        "Six complete development tracks, minimum/median/maximum slot lengths in previously inspected groups; "
        "no GT/error-based ranking, held-out access or behavior fitting. RGB:3x centered square,256², original "
        "DINOv3 CLS768, float32 cache with explicit missing flags. LiDAR:every encoder voxel in20 m, "
        "D-DITR64; shared scene/voxel features. Earlier1.5x/15 m artifacts are unchanged.", "",
        "GPU workload uses the agreed dimensions, one spatial query and temporal RoPE. Geometry is zero, "
        "RGB is real cached CLS, and squared logits supply synthetic gradients; timings are not behavior "
        "training throughput. Microbatch2/4 repeats a complete selected track as an explicitly labeled shape "
        "test. Spatial dropout0, temporal dropout.1, GELU and final LayerNorm are probe settings. "
        "Recompute stages CPU inputs, offload saves spatial backward tensors on CPU, ordinary retains them on GPU.", "",
        "Native passes are warmed-filesystem traversals, including LiDAR decoding but excluding RGB. "
        "Encoder initialization is separate; full native-file checksum verification/reader indexing is included "
        "in traversal time where invoked. Disk reads are OS-warm; no machine-wide page-cache eviction. "
        "Native supply and readout are measured separately, not an overlapped training pipeline.", "",
        "| Track | Slots | Total point tokens | Max tokens/frame | RGB crops | Cache read s |", "|---|---:|---:|---:|---:|---:|"]
    for t in record["tracks"]:
        lines.append(f"| {t['name']} | {t['slots']} | {sum(t.get('point_counts', [])):,} | {max(t.get('point_counts', [0])):,} | {t.get('rgb_valid', 0)} | {t.get('cache_read_seconds',0):.3f} |")
    lines += ["", "| Native group | Frames | Native+encode repeated pass s | Point cache MiB |", "|---|---:|---:|---:|"]
    for r in record["supply"]:
        lines.append(f"| {r['dataset']}/{r['group']} | {r['verified_repeat_frames']} | {r['repeat_native_encode_seconds']:.3f} | {sum(f['bytes'] for f in r['frames'])/2**20:.2f} |")
    lines += ["", "| Track | Mode | Tracks/microbatch | Status | Measured seconds | Peak allocated GiB | Peak reserved GiB |", "|---|---|---:|---|---:|---:|---:|"]
    for r in record["workloads"]:
        lines.append(f"| {r['track']} | {r['mode']} | {r['tracks']} | {r['status']} | {r.get('median_seconds',0):.3f} | {r.get('peak_allocated',0)/2**30:.3f} | {r.get('peak_reserved',0)/2**30:.3f} |")
    lines += ["", "[Numerical evidence, native timings, compression and provenance](execution.json) · [GPU system samples](system.csv)", "",
        "Only a bounded six-track RGB/point cache is built. Population-wide RGB caching and point-supply choice "
        "remain separate work; these timings do not prove a full cache fits. Peak host RSS includes retained "
        "CPU inputs and may include earlier phases. CUDA peaks exclude the unloaded frozen encoders during "
        "cached-readout measurements; live on-demand encoding would add encoder residency/workspaces."]
    (output/"report.md").write_text("\n".join(lines)+"\n")


def run(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(2)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe, output = args.probe.resolve(), args.output.resolve()
    root = probe.parents[2]
    if not output.is_relative_to(root/"experiments/E002a"):
        raise ValueError("Output must be inside outputs/experiments/E002a")
    source = json.loads((probe/"probe.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Completed development probe required")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("rgb", "points", "compressed"):
        (output/folder).mkdir()
    config = source["config"]
    protected = dict(source["protected"])
    for path in [probe/"probe.json", Path(config["ditr_config"]), Path(config["ditr_weights"]), Path(config["dino_weights"])]:
        protected[str(path)] = checksum(path)
    record = {"status":"incomplete", "started_utc":datetime.now(timezone.utc).isoformat(),
        "command":sys.orig_argv, "revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "git_diff":subprocess.check_output(["git","diff","HEAD"],text=True),
        "torch":torch.__version__, "cuda":torch.version.cuda, "device":args.device,
        "crop_expansion":3, "radius_m":20, "tracks":[], "supply":[], "workloads":[], "failures":[]}
    started = time.perf_counter()
    telemetry = log = None
    write_json(output/"execution.json", record)
    try:
        for path, digest in protected.items():
            if checksum(path) != digest:
                raise ValueError(f"Protected source changed: {path}")
        if args.device == "cuda":
            record["hardware"] = {"name":torch.cuda.get_device_name(), "total_bytes":torch.cuda.get_device_properties(0).total_memory}
            log = (output/"system.csv").open("w")
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu", "--format=csv", "-l", "1"], stdout=log, stderr=subprocess.STDOUT)
        contexts = {}
        for dataset in ("loki", "road-waymo"):
            collection, manifest, setup, reader = context(dataset, root, config, probe, protected)
            contexts[dataset] = (collection, manifest, reader)
            groups = source["samples"][dataset]
            candidates = [e for e in setup["tracks"] if not e["exclusions"] and e["group"] in groups and e["split"] in ("training", "validation")]
            for category, entry in select_tracks(candidates):
                if setup["groups"][entry["group"]] != entry["split"] or groups[entry["group"]]["split"] != entry["split"]:
                    raise ValueError("Development split mismatch")
                path = collection/"tracks"/entry["archive"]
                protected[str(path)] = checksum(path)
                m, _ = load_track(path)
                if m["track_locator"] != entry["locator"]:
                    raise ValueError("Archive/setup identity mismatch")
                identity = json.loads(entry["locator"])
                item = {"name":dataset+"-"+category, "dataset":dataset, "category":category, "group":entry["group"],
                    "split":entry["split"], "locator":entry["locator"], "archive":entry["archive"], "slots":entry["slots"],
                    "track_id":identity.get("track_id",identity.get("tube_uid")), "source_frames":m["source_frames"],
                    "point_frames":[None]*entry["slots"], "point_counts":[0]*entry["slots"]}
                record["tracks"].append(item)
        image_start = time.perf_counter()
        image_encoder = ImageEncoder(config["dino_code"], config["dino_weights"], args.device)
        record["rgb_load_seconds"] = time.perf_counter()-image_start
        for item in record["tracks"]:
            _, manifest, reader = contexts[item["dataset"]]
            keys = [k for k in item["source_frames"] if k is not None]
            if len(keys) != len(set(keys)):
                raise ValueError("Repeated native timestamps require explicit slot mapping")
            reader.index_scene(item["group"])
            slots = {k:i for i,k in enumerate(item["source_frames"]) if k is not None}
            values = np.zeros((item["slots"],768), np.float32)
            valid = np.zeros(item["slots"], bool)
            crops, indices, crop_info = [], [], []
            begin = time.perf_counter()
            for key, image, boxes in native_camera(reader, item["group"], keys, [item["track_id"]], manifest["scenes"][item["group"]]):
                crop, info = square_crop(image, boxes.get(item["track_id"]), expansion=3)
                crop_info.append({"slot":slots[key], "key":key, **info})
                if crop is not None:
                    crops.append(normalized_crop(crop));indices.append(slots[key])
            for start in range(0,len(crops),16):
                values[indices[start:start+16]] = image_encoder(torch.stack(crops[start:start+16]))
            valid[indices] = True
            if crops:
                repeat = image_encoder(torch.stack(crops[:min(16,len(crops))]))
                np.testing.assert_array_equal(repeat, values[indices[:len(repeat)]])
            item.update(rgb_valid=int(valid.sum()), rgb_seconds=time.perf_counter()-begin, crop_info=crop_info,
                        rgb_cache="rgb/"+item["name"]+".npz")
            np.savez(output/item["rgb_cache"], cls=values, valid=valid, source_key=np.array([k or "" for k in item["source_frames"]]))
            with np.load(output/item["rgb_cache"],allow_pickle=False) as a:
                np.testing.assert_array_equal(a["cls"], values)
                np.testing.assert_array_equal(a["valid"], valid)
            print("RGB cached", item["name"], int(valid.sum()), "of",item["slots"],flush=True)
            write_json(output/"execution.json",record)
        del image_encoder, crops
        gc.collect()
        if args.device == "cuda":torch.cuda.empty_cache()
        begin = time.perf_counter()
        encoder = PointEncoder(config["ditr_code"],config["ditr_config"],config["ditr_weights"],args.device)
        record["point_load_seconds"] = time.perf_counter()-begin
        grouped = defaultdict(list)
        for item in record["tracks"]:grouped[item["dataset"],item["group"]].append(item)
        for (dataset,group), items in grouped.items():
            _, manifest, reader = contexts[dataset]
            ids = [i["track_id"] for i in items]
            keys = sorted({k for i in items for k in i["source_frames"] if k is not None},key=int)
            anchors = {}
            begin = time.perf_counter()
            reader.index_scene(group)
            for item in items:
                first = item["source_frames"][0]
                anchor = reader.read_frames(group,[first]).get(first,{}).get("ego_pose") if first is not None else None
                if not valid_pose(anchor):raise ValueError("Track initial ego anchor unavailable")
                anchors[item["name"]] = anchor
            slots = {i["name"]:{k:j for j,k in enumerate(i["source_frames"]) if k is not None} for i in items}
            stream = iter(native_scene(reader,group,keys,ids,manifest["scenes"][group],include_rgb=False))
            frames_record = []
            for key in keys:
                before = time.perf_counter();frame = next(stream);native_seconds = time.perf_counter()-before
                if frame["key"] != key:raise ValueError("Native timestamp mismatch")
                points = frame["points"]
                if points is None or not len(points):continue
                sync(args.device)
                if args.device == "cuda":torch.cuda.reset_peak_memory_stats()
                before = time.perf_counter()
                features, _, selected, _ = encoder(points)
                sync(args.device);encoder_seconds = time.perf_counter()-before
                xyz = points[selected,:3]
                union = np.zeros(len(xyz),bool);parts = {}
                for number,item in enumerate(items):
                    if key not in slots[item["name"]]:continue
                    box = frame["boxes"].get(item["track_id"])
                    if box is None or not valid_pose(frame["world_from_native"]):continue
                    offsets = relative_xyz(xyz,box[0],frame["world_from_native"],anchors[item["name"]])
                    inside = box_membership(xyz,*box)
                    _, keep = crop_tokens(features,xyz,offsets,inside)
                    union |= keep;parts[number] = (keep,offsets,inside)
                cache = {"features":features[union],"coordinates":xyz[union],"native_indices":selected[union]}
                path = "points/"+dataset+"-"+group+"-"+key+".npz"
                for number,(keep,offsets,inside) in parts.items():
                    prefix = f"t{number}_";item = items[number];slot = slots[item["name"]][key]
                    cache[prefix+"rows"] = np.flatnonzero(keep[union])
                    cache[prefix+"offsets"] = offsets[keep].astype(np.float32)
                    cache[prefix+"inside"] = inside[keep]
                    item["point_frames"][slot] = {"cache":path,"prefix":prefix}
                    item["point_counts"][slot] = int(keep.sum())
                before=time.perf_counter();np.savez(output/path,**cache);write_seconds=time.perf_counter()-before
                r={"key":key,"cache":path,"native_seconds":native_seconds,"encoder_seconds":encoder_seconds,
                    "native_points":len(points),"scene_voxels":len(selected),"union_tokens":int(union.sum()),
                    "bytes":(output/path).stat().st_size,"write_seconds":write_seconds,
                    "encoder_peak_allocated":torch.cuda.max_memory_allocated() if args.device == "cuda" else 0}
                if not frames_record:
                    before=time.perf_counter()
                    with np.load(output/path,allow_pickle=False) as a:
                        for k,v in cache.items():np.testing.assert_array_equal(a[k],v)
                    r["uncompressed_read_verify_seconds"]=time.perf_counter()-before
                    compressed=output/"compressed"/Path(path).name
                    before=time.perf_counter();np.savez_compressed(compressed,**cache);r["compressed_write_seconds"]=time.perf_counter()-before
                    r["compressed_bytes"]=compressed.stat().st_size
                    before=time.perf_counter()
                    with np.load(compressed,allow_pickle=False) as a:
                        for k,v in cache.items():np.testing.assert_array_equal(a[k],v)
                    r["compressed_read_verify_seconds"]=time.perf_counter()-before
                frames_record.append(r)
                if len(frames_record)%20==0:print("Points",dataset,group,len(frames_record),"/",len(keys),flush=True)
            elapsed=time.perf_counter()-begin
            # A separate native traversal: no persistent feature cache is read to supply inference.
            begin=time.perf_counter();repeats=0;pure_supply=0.
            stream=iter(native_scene(reader,group,keys,ids,manifest["scenes"][group],include_rgb=False))
            for key in keys:
                before=time.perf_counter();frame=next(stream)
                if frame["points"] is None or not len(frame["points"]):
                    pure_supply+=time.perf_counter()-before
                    continue
                features,_,_,inverse=encoder(frame["points"]);sync(args.device)
                pure_supply+=time.perf_counter()-before
                ref=next(r for r in frames_record if r["key"]==frame["key"])
                with np.load(output/ref["cache"],allow_pickle=False) as a:
                    np.testing.assert_array_equal(a["features"],features[inverse[a["native_indices"]]])
                    np.testing.assert_array_equal(a["coordinates"],frame["points"][a["native_indices"],:3])
                repeats+=1
            repeat_seconds=time.perf_counter()-begin
            record["supply"].append({"dataset":dataset,"group":group,"keys":len(keys),"frames":frames_record,
                "initial_extract_crop_write_seconds":elapsed,"repeat_native_encode_verify_seconds":repeat_seconds,
                "verified_repeat_frames":repeats,"repeat_native_encode_seconds":pure_supply})
            write_json(output/"execution.json",record)
            print("Native passes complete",dataset,group,len(keys),flush=True)
        del encoder, features, cache, points
        gc.collect()
        if args.device == "cuda":torch.cuda.empty_cache()
        # Tiny native subset checks algebra/gradients without a full-track optimization.
        first=next(t for t in record["tracks"] if max(t["point_counts"])>0)
        fixture=loaded_track(first,output)
        indices=[i for i,f in enumerate(fixture[0]) if len(f)][:3]
        small=[fixture[0][i][:128].clone() for i in indices]
        record["gradient_equivalence"]=compare_gradients(small,fixture[1][indices],fixture[2][indices],args.device)
        del fixture
        for item in record["tracks"]:
            before=time.perf_counter();frames,rgb,valid=loaded_track(item,output)
            item["cache_read_seconds"]=time.perf_counter()-before
            for mode in MODES:
                for batch in (1,2,4):
                    r={"track":item["name"],"mode":mode,"tracks":batch,"status":"incomplete","timings":[],
                        "replicated_track_shape":batch>1}
                    graph=None
                    try:
                        gc.collect()
                        if args.device=="cuda":torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
                        torch.manual_seed(0);graph=ExecutionGraph().to(args.device)
                        # Independent conditions. OOM ends this condition; no retry/cap/truncation.
                        for repeat in range(3):
                            graph.zero_grad(set_to_none=True);sync(args.device);before=time.perf_counter()
                            value=graph(frames,rgb,valid,mode=mode,tracks=batch)
                            value.square().mean().backward();sync(args.device)
                            r["timings"].append(time.perf_counter()-before)
                            if not torch.isfinite(value).all() or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in graph.parameters()):
                                raise ValueError("Nonfinite workload")
                            del value
                        r.update(status="complete",median_seconds=float(np.median(r["timings"][1:])),
                            track_slots_per_second=batch*item["slots"]/float(np.median(r["timings"][1:])))
                    except torch.OutOfMemoryError as error:
                        r.update(status="out-of-memory",error=str(error))
                    finally:
                        if args.device=="cuda":r.update(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
                        if "value" in locals():del value
                        del graph
                        gc.collect()
                        if args.device=="cuda":torch.cuda.empty_cache()
                    r["peak_host_rss_bytes"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                    record["workloads"].append(r);write_json(output/"execution.json",record);report(output,record)
                    print("Workload",item["name"],mode,batch,r["status"],r.get("median_seconds"),flush=True)
        record["status"]="complete"
    except BaseException:
        record["status"]="failed";record["failures"].append(traceback.format_exc());raise
    finally:
        if telemetry is not None:telemetry.terminate();telemetry.wait()
        if log is not None:log.close()
        changed=[p for p,d in protected.items() if not Path(p).is_file() or checksum(p)!=d]
        record.update(protected=protected,protected_files=len(protected),changed_files=changed,
            elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat(),
            cache_bytes=sum(p.stat().st_size for p in output.rglob("*.npz")))
        if changed:record["status"]="failed"
        write_json(output/"execution.json",record);report(output,record)
        if changed:raise ValueError("Protected sources changed")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),required=True)
    run(parser.parse_args())
