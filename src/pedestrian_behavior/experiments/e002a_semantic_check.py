"""Bounded direct semantic transfer of the released frozen nuScenes head."""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.datasets import road_waymo as road
from pedestrian_behavior.inspection.native_features import native_scene
from pedestrian_behavior.inspection.pretrained import PointEncoder, DITR_REVISION, DITR_SHA256
from pedestrian_behavior.inspection.segmentation import semantic_bev, semantic_confusion, semantic_metrics
from .e002a_point_check import context
from .e002a_probe import write_json

NAMES = ("vehicle", "pedestrian", "bicycle", "motorcycle", "cone", "drivable", "sidewalk", "manmade", "vegetation")
WAYMO_NAMES = ("undefined", "car", "truck", "bus", "other_vehicle", "motorcyclist", "bicyclist", "pedestrian",
    "sign", "traffic_light", "pole", "construction_cone", "bicycle", "motorcycle", "building", "vegetation",
    "tree_trunk", "curb", "road", "lane_marker", "other_ground", "walkable", "sidewalk")
NUSCENES_NAMES = ("barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian",
    "traffic_cone", "trailer", "truck", "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation")
GT_MAP = np.array([-1,0,0,0,0,-1,-1,1,7,7,7,4,2,3,7,8,8,-1,5,5,-1,-1,6], dtype=np.int16)
PRED_MAP = np.array([9,2,0,0,0,3,1,4,0,0,5,9,6,9,7,8], dtype=np.int16)
PALETTE = np.array([[70,145,245],[255,65,75],[245,170,55],[185,95,235],[255,215,70],
                    [90,100,125],[75,220,215],[170,150,120],[85,190,75]], dtype=np.uint8)
SOURCES = ["https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/segmentation.proto",
    "https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/segmentation.py",
    "https://huggingface.co/rwthvision/ditr/blob/a656b4385aa28060b30152676023903a923b4a17/nuscenes/semseg-ptv3_weight=distill-ptv3_nuscenes_dino-L/config.py"]


def aligned_semantics(lidar_rows, segmentation_rows):
    """Same laser/return concatenation as native_scene; retain unannotated points."""
    labels, parts, seen = [], [], set()
    annotation = {r["key.laser_name"]: r for r in segmentation_rows}
    if len(annotation) != len(segmentation_rows) or set(annotation)-{1}:
        raise ValueError("Expected unique TOP segmentation row")
    offset = 0
    for row in lidar_rows:
        laser = row["key.laser_name"]
        if laser in seen:
            raise ValueError("Duplicate native laser row")
        seen.add(laser)
        for ret in (1, 2):
            prefix = f"[LiDARComponent].range_image_return{ret}."
            if row[prefix+"values"] is None:
                continue
            shape, values = row[prefix+"shape"], row[prefix+"values"]
            ri = np.asarray(values).reshape(shape)
            valid = np.isfinite(ri[..., 0]) & (ri[..., 0] > 0)
            count = int(valid.sum())
            gt = np.full(count, -1, dtype=np.int16)
            if laser in annotation:
                sp = f"[LiDARSegmentationLabelComponent].range_image_return{ret}."
                sr = annotation[laser]
                if sr[sp+"values"] is not None:
                    gt = road.range_semantics(sr[sp+"values"], sr[sp+"shape"], values, shape)
                    if (gt > 22).any():
                        raise ValueError("Unknown Waymo semantic ID")
                    gt = gt.astype(np.int16)
            labels.append(gt)
            parts.append({"laser": int(laser), "return": ret, "start": offset, "stop": offset+count,
                "range_shape": list(shape), "valid_flat_pixels": np.flatnonzero(valid).astype(np.int32)})
            offset += count
    if set(annotation)-seen:
        raise ValueError("Segmentation laser missing from scan")
    return np.concatenate(labels), parts


def mapped_labels(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    if (gt.shape != pred.shape or gt.ndim != 1 or not np.issubdtype(gt.dtype, np.integer)
            or not np.issubdtype(pred.dtype, np.integer) or (gt < -1).any() or (gt > 22).any()
            or (pred < 0).any() or (pred > 15).any()):
        raise ValueError("Invalid native semantic IDs")
    target = np.full(gt.shape, -1, dtype=np.int16)
    valid = gt >= 0
    target[valid] = GT_MAP[gt[valid]]
    return target, PRED_MAP[pred]


def publish(output, record):
    metrics = record.get("metrics")
    lines = ["# E002a frozen D-DITR semantic transfer", "", f"Status: **{record['status']}**.", "",
        "Seven exact timestamp matches from the existing development probe, selected before inference. "
        "No training, held-out scoring or model selection. Full multi-sensor/two-return scan as encoder input; "
        "only TOP points with supported semantic GT scored, at all ranges. Native 16-class argmax is mapped after prediction.", "",
        "Coarse shared taxonomy, not official Waymo mIoU or reproduction of nuScenes results. "
        "Excluded GT points never contribute false positives. Supported-class macro-IoU averages classes with GT support only. "
        "Unmatched predictions on scored GT remain errors. No confidence intervals: seven correlated frames/five clips.", "",
        "## Taxonomy fixed before inference", "", "| Shared class | Waymo GT IDs | nuScenes predictions |", "|---|---|---|"]
    for i, name in enumerate(NAMES):
        lines.append(f"| {name} | {', '.join(f'{j}:{WAYMO_NAMES[j]}' for j in np.flatnonzero(GT_MAP == i))} | "
                     f"{', '.join(NUSCENES_NAMES[j] for j in np.flatnonzero(PRED_MAP == i))} |")
    lines += ["", "Excluded Waymo GT: undefined, riders, curb, other_ground and walkable. "
        "The latter ground categories straddle nuScenes classes. Vehicle is deliberately merged; "
        "manmade combines buildings/signs/lights/poles, vegetation includes trunks. Cone/bicycle/motorcycle "
        "boundaries remain approximate across datasets. Unmatched predictions: barrier, other_flat, terrain. "
        "This diagnostic is restricted to the declared common support.", ""]
    if metrics:
        lines += ["## Pooled native-point results", "", f"Scored points: **{metrics['points']:,}**. "
            f"Accuracy: **{100*metrics['accuracy']:.2f}%**. Supported-class macro-IoU: "
            f"**{100*metrics['supported_macro_iou']:.2f}%** ({metrics['supported_classes']}/{len(NAMES)} supported classes).", "",
            "| Class | GT points | Predicted points | IoU % | Precision % | Recall % | F1 % |", "|---|---:|---:|---:|---:|---:|---:|"]
        fmt = lambda v: "—" if v is None else f"{100*v:.2f}"
        for name, row in zip(NAMES, metrics["classes"]):
            lines.append(f"| {name} | {row['support']:,} | {row['predicted']:,} | {fmt(row['iou'])} | "
                         f"{fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} |")
        lines += ["", "## Frames", "", "| Clip | Split | Timestamp | Scored points | Pedestrian GT | Accuracy % |", "|---|---|---|---:|---:|---:|"]
        for item in record["frames"]:
            m = item["metrics"]
            lines.append(f"| {item['group']} | {item['split']} | {item['key']} | {m['points']:,} | {m['classes'][1]['support']:,} | {fmt(m['accuracy'])} |")
    lines += ["", "## Limits and artifacts", "", "Scores test this frozen head and our single-pass 5 cm preprocessing. "
        "They do not establish linear separability of the features, behavior usefulness or LOKI transfer. "
        "LiDAR input retains the existing axis/intensity adapters and TOP motion compensation; no TTA. "
        "Both returns count as distinct native returns, not unique physical surfaces.", "",
        "[Gallery](index.html) · [Numerical record](semantic-check.json). Per-frame pickle-free NPZ files preserve "
        "native XYZ/strength, raw GT, native nuScenes argmax, TOP pixel identities and inverse voxel association; "
        "metrics can be reconstructed without inference.", "", "Sources:", "", *[f"- {s}" for s in SOURCES]]
    (output/"report.md").write_text("\n".join(lines)+"\n")
    body = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>D-DITR semantic transfer</title>",
        "<style>body{font:16px system-ui;background:#171b22;color:white;margin:24px}img{width:100%;max-width:1980px}a{color:#8cf}</style>",
        "<h1>Frozen D-DITR: Waymo semantic transfer</h1>", f"<p>Status: {html.escape(record['status'])}. "
        "Seven development frames; coarse common taxonomy. No training or official benchmark score.</p>",
        "<p><a href='report.md'>Results and mapping</a> · <a href='semantic-check.json'>Numerical evidence</a></p>"]
    if (output/"confusion.png").exists():
        body.append("<img src='confusion.png' alt='Raw and row-normalized semantic confusion'>")
    for item in record["frames"]:
        body.append(f"<h2>{html.escape(item['name'])}</h2><img src='{item['name']}.png' alt='Ground truth, prediction and errors'>")
    (output/"index.html").write_text("\n".join(body)+"</html>\n")


def plot_confusion(output, confusion, *, title="Frozen nuScenes D-DITR: seven Waymo development frames; shared taxonomy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(19, 8), constrained_layout=True)
    for ax, norm in zip(axes, (False, True)):
        data = confusion/np.maximum(confusion.sum(1, keepdims=True), 1) if norm else confusion
        image = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if norm else None)
        for y, x in np.ndindex(data.shape):
            if confusion[y].sum():
                ax.text(x, y, f"{data[y,x]:.2f}" if norm else f"{data[y,x]:,}", ha="center", va="center", fontsize=7,
                        color="white" if data[y,x] > (0.5 if norm else data.max()/2) else "black")
        ax.set(xticks=range(len(NAMES)+1), xticklabels=[*NAMES,"unmatched"], yticks=range(len(NAMES)),
               yticklabels=NAMES, xlabel="Prediction", ylabel="Waymo GT", title="Row fraction" if norm else "Native return counts")
        ax.tick_params(axis="x", rotation=65)
        fig.colorbar(image, ax=ax, shrink=.7)
    fig.suptitle(title)
    for ext in ("png", "svg"):
        fig.savefig(output/f"confusion.{ext}", dpi=160)
    plt.close(fig)


def run(args):
    import pyarrow.parquet as pq
    torch.manual_seed(0)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    probe, output = args.probe.resolve(), args.output.resolve()
    root = probe.parents[2]
    if not output.is_relative_to(root/"experiments/E002a"):
        raise ValueError("Output must be under outputs/experiments/E002a")
    source = json.loads((probe/"probe.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Completed native probe required")
    output.mkdir(parents=True, exist_ok=False)
    config = source["config"]
    protected = source["protected"].copy()
    for path in (probe/"probe.json", Path(config["ditr_weights"]), Path(config["ditr_config"])):
        protected[str(path)] = checksum(path)
    record = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
        "revision": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
        "git_diff": subprocess.check_output(["git","diff","HEAD"], text=True),
        "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device,
        "checkpoint_sha256": DITR_SHA256, "author_revision": DITR_REVISION, "sources": SOURCES,
        "names": NAMES, "waymo_names": WAYMO_NAMES, "nuscenes_names": NUSCENES_NAMES,
        "gt_map": GT_MAP.tolist(), "prediction_map": PRED_MAP.tolist(), "seed": 0,
        "policy": "existing seven matching dev frames; full scene input, labeled TOP both returns scored at all ranges; native argmax then mapping; unmatched prediction errors retained; supported-GT-class macro IoU",
        "frames": [], "selection": [], "failures": []}
    started = time.monotonic()
    write_json(output/"semantic-check.json", record)
    try:
        for path, digest in protected.items():
            if checksum(path) != digest:
                raise ValueError(f"Protected source changed: {path}")
        collection, manifest, setup, reader = context("road-waymo", root, config, probe, protected)
        available = {}
        for item in source["frames"]:
            if item["dataset"] != "road-waymo":
                continue
            group, key = item["group"], item["key"]
            if setup["groups"][group] not in ("training","validation"):
                raise ValueError("Held-out group in development diagnostic")
            lidar_path = Path(item["files"]["lidar"])
            path = lidar_path.parent.parent/"lidar_segmentation"/lidar_path.name
            if path not in available:
                available[path] = set(pq.read_table(path, columns=["key.frame_timestamp_micros"])
                    .column(0).to_pylist()) if path.exists() else set()
            selected = int(key) in available[path]
            record["selection"].append({"group": group, "key": key, "selected": selected, "path": str(path)})
        chosen = [i for i in record["selection"] if i["selected"]]
        if len(chosen) != 7 or len({i["group"] for i in chosen}) != 5:
            raise ValueError("Expected the audited seven frames/five development groups")
        for item in chosen:
            protected[item["path"]] = checksum(item["path"])
        write_json(output/"semantic-check.json", record)
        encoder = PointEncoder(config["ditr_code"], config["ditr_config"], config["ditr_weights"], args.device)
        if tuple(encoder.names) != NUSCENES_NAMES:
            raise ValueError("Unexpected native head taxonomy")
        record["hardware"] = torch.cuda.get_device_name() if args.device == "cuda" else "CPU"
        total = np.zeros((len(NAMES),len(NAMES)+1), dtype=np.int64)
        for group in dict.fromkeys(i["group"] for i in chosen):
            keys = [i["key"] for i in chosen if i["group"] == group]
            scene = reader.scenes[group]
            frozen = manifest["scenes"][group]
            identities = [json.loads(e["locator"]) for e in setup["tracks"] if e["group"] == group and not e["exclusions"]]
            tracks = [i["tube_uid"] for i in identities]
            reader.index_scene(group)
            scene_name = scene["segment_context_name"]
            lidar_path = road.component_path(scene,"lidar",reader.waymo_root)
            seg_path = Path(next(i["path"] for i in chosen if i["group"] == group))
            lidar_iter = iter(road.frame_groups(road.component_rows(lidar_path,scene_name)))
            seg_iter = iter(road.frame_groups(road.component_rows(seg_path,scene_name)))
            from pedestrian_behavior.inspection.road_waymo import aligned_frames
            scans = iter(aligned_frames(lidar_iter,sorted(keys,key=int)))
            labels = iter(aligned_frames(seg_iter,sorted(keys,key=int)))
            for frame in native_scene(reader,group,keys,tracks,frozen):
                lidar_rows = next(scans)
                gt, parts = aligned_semantics(lidar_rows, next(labels))
                points = frame["points"]
                if points is None or len(gt) != len(points):
                    raise ValueError("Native point/semantic count mismatch")
                by_laser = {r["key.laser_name"]: r for r in lidar_rows}
                strengths = []
                for part in parts:
                    prefix = f"[LiDARComponent].range_image_return{part['return']}."
                    ri = np.asarray(by_laser[part["laser"]][prefix+"values"], dtype=np.float64).reshape(part["range_shape"])
                    strengths.append(np.tanh(ri.reshape(-1,4)[part["valid_flat_pixels"],1]).astype(np.float32))
                if not np.array_equal(points[:,3],np.concatenate(strengths)):
                    raise ValueError("Native laser/return/pixel order differs from label alignment")
                torch.cuda.reset_peak_memory_stats() if args.device == "cuda" else None
                tick = time.monotonic()
                _, logits, selected, inverse = encoder(points)
                pred = logits.argmax(1)[inverse].astype(np.int16)
                elapsed = time.monotonic()-tick
                target, prediction = mapped_labels(gt,pred)
                cm = semantic_confusion(target,prediction,len(NAMES))
                total += cm
                name = f"{group}-{frame['key']}"
                top = np.zeros(len(points), dtype=bool)
                arrays = {}
                part_records = []
                for part in parts:
                    part = dict(part)
                    pixels = part.pop("valid_flat_pixels")
                    arrays[f"laser{part['laser']}_return{part['return']}_pixels"] = pixels
                    if part["laser"] == 1:
                        top[part["start"]:part["stop"]] = True
                    part_records.append(part)
                np.savez_compressed(output/f"{name}.npz", native_points=points, gt_waymo=gt, pred_nuscenes=pred,
                    voxel_selected=selected, voxel_inverse=inverse, **arrays)
                semantic_bev(points[top,:3],target[top],prediction[top],NAMES,PALETTE,
                    f"{group} | {frame['key']} | frozen D-DITR / Waymo GT").save(output/f"{name}.png")
                item = {"name": name, "group": group, "key": frame["key"], "split": setup["groups"][group],
                    "segment": scene_name, "segmentation_path": str(seg_path), "parts": part_records,
                    "native_strength_order_exact": True,
                    "native_points": len(points), "annotated_points": int((gt>0).sum()),
                    "excluded_gt_counts": {WAYMO_NAMES[i]: int((gt==i).sum()) for i in np.flatnonzero(GT_MAP<0)},
                    "unannotated_sensor_points": int((gt<0).sum()), "metrics": semantic_metrics(cm),
                    "encoder_seconds": elapsed, "peak_cuda_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                    "fixture_sha256": checksum(output/f"{name}.npz")}
                record["frames"].append(item)
                write_json(output/"semantic-check.json",record)
                print("Completed",name,"scored points",int(cm.sum()),flush=True)
        record["metrics"] = semantic_metrics(total)
        for i, row in enumerate(record["metrics"]["classes"]):
            row["supporting_frames"] = sum(f["metrics"]["classes"][i]["support"]>0 for f in record["frames"])
            row["supporting_groups"] = len({f["group"] for f in record["frames"] if f["metrics"]["classes"][i]["support"]>0})
        plot_confusion(output,total)
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        record["failures"].append(traceback.format_exc())
        raise
    finally:
        changed = [p for p,digest in protected.items() if not Path(p).is_file() or checksum(p)!=digest]
        record.update(protected=protected,changed_files=changed,protected_files=len(protected),seconds=time.monotonic()-started,
            finished_utc=datetime.now(timezone.utc).isoformat())
        if changed:
            record["status"] = "failed"
        write_json(output/"semantic-check.json",record)
        publish(output,record)
        if changed:
            raise ValueError("Protected sources changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu","cuda"), required=True)
    run(parser.parse_args())
