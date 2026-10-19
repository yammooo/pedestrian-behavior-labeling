"""Bounded native-feature feasibility evidence; no behavior training/test access."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import faulthandler
import hashlib
import html
import json
from importlib.metadata import distributions
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from pedestrian_behavior.data.preparation import checksum, load_track, valid_pose
from pedestrian_behavior.datasets import loki, road_waymo
from pedestrian_behavior.inspection.native_features import native_scene
from pedestrian_behavior.inspection.feature_probe import (ProbeReadout, bev_sheet, box_membership,
    crop_sheet, normalized_crop, relative_xyz, square_crop, temporal_probe)
from pedestrian_behavior.inspection.pretrained import PointEncoder, ImageEncoder, DITR_SHA256, DITR_REVISION, DINO_REVISION


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def sample_groups(setup):
    rng = np.random.default_rng(0)
    entries = [e for e in setup["tracks"] if not e["exclusions"] and e["split"] in ("training", "validation")]
    grouped = defaultdict(list)
    for entry in entries:
        if setup["groups"][entry["group"]] != entry["split"]:
            raise ValueError("Setup split mismatch")
        grouped[entry["group"]].append(entry)
    selected = {}
    for split, count in (("training", 8), ("validation", 2)):
        groups = sorted(g for g in grouped if grouped[g][0]["split"] == split)
        if len(groups) < count:
            raise ValueError("Insufficient development groups for declared probe sample")
        for g in rng.choice(groups, count, replace=False):
            selected[str(g)] = {"split": split, "random": True, "stress": []}
    training = [e for e in entries if e["split"] == "training"]
    longest = max(training, key=lambda e: (e["slots"], e["locator"]))
    crowded = max((g for g in grouped if grouped[g][0]["split"] == "training"), key=lambda g: (len(grouped[g]), g))
    for g, reason in ((longest["group"], "longest-track"), (crowded, "crowded-group")):
        selected.setdefault(g, {"split": "training", "random": False, "stress": []})["stress"].append(reason)
    return grouped, selected, longest


def selected_frames(metadata, random_sample, stress):
    keys = sorted({k for m in metadata.values() for k in m["source_frames"] if k is not None}, key=int)
    rng = np.random.default_rng(0)
    random = set(rng.choice(len(keys), min(3, len(keys)), replace=False).tolist()) if random_sample else set()
    reasons = {keys[i]: ["random"] for i in random}
    if stress:
        counts = {k: sum(k in m["source_frames"] for m in metadata.values()) for k in keys}
        crowded = max(keys, key=lambda k: (counts[k], int(k)))
        reasons.setdefault(crowded, []).extend(stress)
        longest = max(metadata.values(), key=lambda m: len(m["source_frames"]))
        available = [k for k in longest["source_frames"] if k is not None]
        for k in (available[0], available[len(available)//2], available[-1]):
            reasons.setdefault(k, []).append("longest-track-context")
    return reasons, len(keys)


def benchmark(tokens, length, device, output):
    """All-token conservative shape workload, not measured end-to-end training."""
    records = []
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for width in (128, 256):
        for seeds in (1, 4):
            for batch in (1, 2, 4, 8):
                item = {"width": width, "seeds": seeds, "batch_tracks": batch, "slots": length,
                        "tokens_per_slot": len(tokens), "workload": "repeat-largest-observed-neighborhood-at-every-slot"}
                readout = temporal = fusion = geometry = rgb_projection = classifier = x = result = loss = lidar = pe = None
                try:
                    gc.collect()
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                    torch.manual_seed(0)
                    readout = ProbeReadout(width, seeds).to(device)
                    fusion = nn.Sequential(nn.Linear(579, 256), nn.GELU()).to(device)
                    geometry = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 64)).to(device)
                    rgb_projection = nn.Linear(768, 256).to(device)
                    classifier = nn.Linear(256, 4).to(device)
                    temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 4, 1024, .1,
                        batch_first=True, norm_first=True), 2, enable_nested_tensor=False).to(device)
                    # Real observed token values; copy per slot to include input storage.
                    started = time.perf_counter()
                    features = []
                    for _ in range(batch*length):
                        x = torch.tensor(tokens, device=device)
                        lidar = readout(x)
                        g = geometry(torch.zeros(20, device=device))
                        r = rgb_projection(torch.zeros(768, device=device))
                        features.append(torch.cat((g, r, lidar, torch.ones(3, device=device))))
                    result = fusion(torch.stack(features).reshape(batch, length, 579))
                    result = classifier(temporal_probe(temporal, result, torch.full((batch,), length, device=device)))
                    loss = result.square().mean()
                    loss.backward()
                    if not torch.isfinite(result).all() or any(p.grad is not None and not torch.isfinite(p.grad).all()
                            for module in (readout, fusion, temporal, geometry, rgb_projection, classifier) for p in module.parameters()):
                        raise ValueError("Nonfinite workload forward/backward")
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    item.update(status="complete", seconds=time.perf_counter()-started)
                except torch.OutOfMemoryError as error:
                    item.update(status="out-of-memory", error=str(error))
                finally:
                    if str(device).startswith("cuda"):
                        item["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    readout = temporal = fusion = geometry = rgb_projection = classifier = x = result = loss = lidar = pe = g = r = None
                    if "features" in locals():
                        features.clear()
                    gc.collect()
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                records.append(item)
                write_json(output/"workloads.json", records)
                print("workload", width, seeds, batch, item["status"], flush=True)
    return records


def summarize(record, output):
    lines = ["# E002a native-feature feasibility", "", f"Status: **{record['status']}**.", "",
             "Development observations only. No behavior training, held-out inference, W&B runs or full caches.", "",
             "Semantic outputs are the pretrained 16-class nuScenes head; no segmentation GT or accuracy claim.", "",
             "## Measurements", "", "| Dataset | Frames | Native points/frame (median) | Voxels/frame (median) | Point GPU seconds/frame (median) | Peak CUDA GiB |",
             "|---|---:|---:|---:|---:|---:|"]
    for dataset in ("loki", "road-waymo"):
        rows = [r for r in record["frames"] if r["dataset"] == dataset and "voxels" in r]
        if rows:
            lines.append(f"| {dataset} | {len(rows)} | {np.median([r['native_points'] for r in rows]):.0f} | "
                         f"{np.median([r['voxels'] for r in rows]):.0f} | {np.median([r['gpu_seconds'] for r in rows]):.3f} | "
                         f"{max(r['peak_cuda_allocated_bytes'] for r in rows)/2**30:.2f} |")
    lines += ["", "## Random-sample extrapolation", "", "Rough group-scaled estimates, not a storage commitment. Float32 payload only; excludes indexes/compression and test preparation. Stress cases excluded.", "",
              "| Dataset / split | Scene features GiB | Neighborhood union GiB | Duplicated neighborhoods GiB | RGB CLS GiB | Point GPU hours |",
              "|---|---:|---:|---:|---:|---:|"]
    for item in record.get("estimates", []):
        lines.append(f"| {item['dataset']} / {item['split']} | {item['scene_bytes']/2**30:.2f} | {item['union_bytes']/2**30:.2f} | "
                     f"{item['duplicated_bytes']/2**30:.2f} | {item['rgb_bytes']/2**30:.3f} | {item['gpu_seconds']/3600:.2f} |")
    lines += ["", "## Limits and next decision", "",
        "- Inspect the linked panels for native alignment and semantic domain failures. Good segmentation is not proof of behavior recoverability.",
        "- Point input uses a declared ego-to-nuScenes axis rotation, without inventing a roof-sensor translation. LOKI retains already normalized strength; Waymo uses tanh, matching the author's Waymo preprocessing.",
        "- Deterministic first-return 5 cm voxels and fixed serialization orders differ from randomized author validation; no segmentation TTA. Native inverse mappings are retained.",
        "- Workloads repeat the largest measured neighborhood for a complete longest-track shape. They are conservative synthetic resource checks, not measured training throughput or actual full-track inference.",
        "- Decide cache layout, spatial width and any resource limits after review. No automatic token cap, reduced-precision storage or model substitution.",
        "", "[Inspection panels](index.html) · [Numerical/provenance record](probe.json) · [Workloads](workloads.json)", ""]
    if record.get("failures"):
        lines += ["## Failures / unavailable checks", ""]+["- "+e for e in record["failures"]]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    panels = sorted((output/"panels").glob("*.png"))
    (output/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>E002a native checks</title>'
        '<style>body{font:16px system-ui;background:#202020;color:white;margin:24px}img{max-width:100%}article{margin-bottom:40px}</style>'
        '<h1>E002a qualitative checks</h1><p>Development frames; semantic predictions without segmentation GT.</p>'+
        "".join(f'<article><h2>{html.escape(p.stem)}</h2><a href="panels/{p.name}"><img loading="lazy" src="panels/{p.name}"></a></article>' for p in panels)+"</html>")


def run(args):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output = args.output.resolve()
    root = args.outputs_root.resolve()
    if not output.is_relative_to(root/"experiments/E002a"):
        raise ValueError("Probe output must be under outputs-root/experiments/E002a")
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("panels", "fixtures"):
        (output/folder).mkdir()
    record = {"schema_version": 1, "status": "incomplete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.orig_argv, "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "provenance": {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
            "python": platform.python_version(), "host": platform.node(), "torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "seed": 0,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "point_revision": DITR_REVISION, "rgb_revision": DINO_REVISION, "point_sha256": DITR_SHA256,
            "point_precision": "FP32 sparse/linear, author-internal FP16 FlashAttention", "rgb_precision": "FP16 CUDA autocast"},
        "protected": {}, "samples": {}, "frames": [], "failures": [], "estimates": []}
    protected = {}
    largest = None
    max_slots = 0
    started = time.perf_counter()
    write_json(output/"probe.json", record)
    write_json(output/"dependencies.json", {d.metadata["Name"]: d.version for d in distributions()})
    def protect(path):
        protected[str(path.resolve())] = checksum(path)
    try:
        print("Loading pinned point checkpoint", flush=True)
        point_encoder = PointEncoder(args.ditr_code, args.ditr_config, args.ditr_weights, args.device)
        print("Point checkpoint loaded strictly", flush=True)
        image_encoder = ImageEncoder(args.dino_code, args.dino_weights, args.device) if args.dino_weights else None
        if image_encoder:
            record["provenance"]["rgb_sha256"] = checksum(args.dino_weights)
        else:
            record["failures"].append("DINOv3 weights unavailable: RGB crops checked, frozen RGB descriptors NOT executed.")
        for dataset in ("loki", "road-waymo"):
            collection = root/"experiments/E001/reader-preparation"/dataset
            setup_path = root/"experiments/E001/data-setup"/dataset/"setup.json"
            manifest_path = collection/"manifest.json"
            setup, manifest = (json.loads(p.read_text()) for p in (setup_path, manifest_path))
            protect(setup_path)
            protect(manifest_path)
            if setup["status"] != "complete" or setup["collection"]["manifest_sha256"] != checksum(manifest_path):
                raise ValueError("Invalid frozen E001 collection/setup")
            grouped, groups, longest = sample_groups(setup)
            max_slots = max(max_slots, longest["slots"])
            if dataset == "loki":
                transform_path = output/"loki-transform.json"
                write_json(transform_path, manifest["sources"]["transform"])
                reader = loki.TrackReader(args.loki_root or manifest["sources"]["root"], transform_path)
            else:
                reader = road_waymo.TrackReader(args.road_index or manifest["sources"]["index"], args.waymo_root)
                if reader.sources["sha256"] != manifest["sources"]["sha256"]:
                    raise ValueError("Frozen ROAD annotation/index checksum mismatch")
            record["samples"][dataset] = groups
            for group, sample in sorted(groups.items()):
                print("Reading", dataset, group, flush=True)
                entries = grouped[group]
                saved = {}
                for entry in entries:
                    archive = collection/"tracks"/entry["archive"]
                    protect(archive)
                    metadata, _ = load_track(archive)
                    if metadata["track_locator"] != entry["locator"]:
                        raise ValueError("Frozen archive/setup identity mismatch")
                    saved[entry["locator"]] = metadata
                reasons, scene_slots = selected_frames(saved, sample["random"], sample["stress"])
                sample.update(frames={str(k): v for k, v in reasons.items()}, scene_slots=scene_slots,
                              archives=[e["archive"] for e in entries])
                track_ids = {json.loads(e["locator"]).get("track_id", json.loads(e["locator"]).get("tube_uid")): e for e in entries}
                anchors = {}
                first_keys = {m["source_frames"][0] for m in saved.values() if m["source_frames"][0] is not None}
                reader.index_scene(group)
                anchor_records = reader.read_frames(group, first_keys)
                for identity, metadata in saved.items():
                    key = metadata["source_frames"][0]
                    anchors[identity] = anchor_records.get(key, {}).get("ego_pose")
                for frame in native_scene(reader, group, reasons, track_ids, manifest["scenes"][group]):
                    print("Extracting", dataset, group, frame["key"], flush=True)
                    frame_start = time.perf_counter()
                    key = frame["key"]
                    label = f"{dataset}-{group}-{key}"
                    item = {"dataset": dataset, "group": group, "split": sample["split"], "key": key,
                            "reasons": reasons[key], "files": frame["files"], "tracks": []}
                    points = frame["points"]
                    if points is None or not len(points):
                        item["missing_points"] = True
                        record["frames"].append(item)
                        continue
                    if str(args.device).startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                    gpu_start = time.perf_counter()
                    features, logits, selected, inverse = point_encoder(points)
                    if str(args.device).startswith("cuda"):
                        torch.cuda.synchronize()
                    item.update(gpu_seconds=time.perf_counter()-gpu_start, native_points=len(points), voxels=len(selected),
                                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                                strength_quantiles=np.quantile(points[:, 3], [0, .01, .5, .99, 1]).tolist(),
                                feature_mean=float(features.mean()), feature_std=float(features.std()),
                                semantic_counts=np.bincount(logits.argmax(1), minlength=16).tolist())
                    if not any(r["dataset"] == dataset for r in record["frames"]):
                        repeat, *_ = point_encoder(points)
                        item["repeat_max_absolute_error"] = float(np.max(np.abs(features-repeat)))
                        if not np.allclose(features, repeat, rtol=1e-4, atol=1e-4):
                            write_json(output/"repeatability-failure.json", item)
                            raise ValueError("Frozen extraction repeatability check failed")
                    spatial = points[selected, :3]
                    union = np.zeros(len(spatial), dtype=bool)
                    rgb_crops, rgb_tracks = [], []
                    render = "random" in reasons[key] and sum(r["dataset"] == dataset and r.get("rendered") for r in record["frames"]) < 6
                    render = render or "random" not in reasons[key]
                    for track, entry in track_ids.items():
                        metadata = saved[entry["locator"]]
                        if key not in metadata["source_frames"]:
                            continue
                        crop, crop_info = square_crop(frame["image"], frame["camera"].get(track))
                        info = {"locator": entry["locator"], "archive": entry["archive"],
                                "slot": metadata["source_frames"].index(key), "crop": crop_info}
                        if crop is not None:
                            rgb_crops.append(normalized_crop(crop))
                            rgb_tracks.append(entry["locator"])
                        box = frame["boxes"].get(track)
                        if box is not None and valid_pose(frame["world_from_native"]) and valid_pose(anchors[entry["locator"]]):
                            center, size, yaw = box
                            all_offsets = relative_xyz(spatial, center, frame["world_from_native"], anchors[entry["locator"]])
                            neighborhood = np.linalg.norm(all_offsets[:, :2], axis=1) <= 15
                            inside = box_membership(spatial, center, size, yaw)
                            union |= neighborhood
                            offsets = all_offsets[neighborhood]
                            tokens = np.c_[features[neighborhood], offsets, inside[neighborhood]].astype(np.float32)
                            info.update(neighborhood_tokens=int(neighborhood.sum()), target_tokens=int(inside.sum()),
                                        native_target_returns=int(box_membership(points[:, :3], center, size, yaw).sum()))
                            if len(tokens) and (largest is None or len(tokens) > len(largest)):
                                largest = tokens
                        else:
                            info.update(neighborhood_tokens=0, target_tokens=0, missing_box_or_pose=True)
                        if render and len(item["tracks"]) < 3:
                            crop_sheet(frame["image"], frame["camera"].get(track), crop, crop_info,
                                       output/"panels"/f"{label}-crop-{len(item['tracks'])}.png")
                        item["tracks"].append(info)
                    if rgb_crops and image_encoder:
                        rgb_start = time.perf_counter()
                        rgb = np.concatenate([image_encoder(torch.stack(rgb_crops[i:i+16])) for i in range(0, len(rgb_crops), 16)])
                        item["rgb_seconds"] = time.perf_counter()-rgb_start
                        np.savez(output/"fixtures"/f"{label}-rgb.npz", cls=rgb,
                                 locators=np.array(rgb_tracks), key=np.array(str(key)))
                    item.update(union_tokens=int(union.sum()), rgb_crops=len(rgb_crops), rendered=render,
                                total_seconds=time.perf_counter()-frame_start)
                    if render:
                        bev_sheet(points, logits.argmax(1)[inverse], list(frame["boxes"].values()), point_encoder.names,
                                  output/"panels"/f"{label}-bev.png")
                        np.savez(output/"fixtures"/f"{label}-spatial.npz", coord=spatial.astype(np.float32),
                                 feature=features.astype(np.float32), logits=logits.astype(np.float32),
                                 selected_native_index=selected, native_to_voxel=inverse,
                                 native_xyz=points[:, :3], world_from_native=frame["world_from_native"])
                    record["frames"].append(item)
                    write_json(output/"probe.json", record)
                    print(dataset, group, key, "points", len(points), "voxels", len(features), flush=True)
            for split in ("training", "validation"):
                random_groups = [g for g, s in groups.items() if s["random"] and s["split"] == split]
                group_estimates = []
                for g in random_groups:
                    rows = [r for r in record["frames"] if r["dataset"] == dataset and r["group"] == g and "random" in r["reasons"] and "voxels" in r]
                    factor = groups[g]["scene_slots"]/3
                    group_estimates.append({"scene_bytes": sum(r["voxels"]*67*4 for r in rows)*factor,
                        "union_bytes": sum(r["union_tokens"]*67*4 for r in rows)*factor,
                        "duplicated_bytes": sum(sum(t["neighborhood_tokens"] for t in r["tracks"])*68*4 for r in rows)*factor,
                        "rgb_bytes": sum(r["rgb_crops"]*768*4 for r in rows)*factor,
                        "gpu_seconds": sum(r["gpu_seconds"] for r in rows)*factor})
                count = len({e["group"] for rows in grouped.values() for e in rows if e["split"] == split})
                record["estimates"].append({"dataset": dataset, "split": split, "groups": count,
                    **{k: float(np.mean([v[k] for v in group_estimates])*count) for k in group_estimates[0]}})
        del point_encoder, image_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if largest is not None:
            benchmark(largest, max_slots, args.device, output)
        record["status"] = "complete" if args.dino_weights else "partial-rgb-weights-unavailable"
    except BaseException as error:
        record["status"] = "failed"
        record["failures"].append(f"{type(error).__name__}: {error}")
        (output/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        record["protected"] = protected
        record["protected_unchanged"] = all(checksum(Path(p)) == value for p, value in protected.items())
        record["elapsed_seconds"] = time.perf_counter()-started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output/"probe.json", record)
        summarize(record, output)
        if not record["protected_unchanged"]:
            raise ValueError("Protected source files changed during probe")
    return record


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    for option in ("ditr-code", "ditr-config", "ditr-weights", "dino-code"):
        parser.add_argument("--"+option, type=Path, required=True)
    for option in ("dino-weights", "loki-root", "road-index", "waymo-root"):
        parser.add_argument("--"+option, type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
