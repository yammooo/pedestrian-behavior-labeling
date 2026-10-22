"""Bounded CPU storage/read diagnostic on existing frozen D-DITR outputs."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

import numpy as np

from pedestrian_behavior.data.frozen_features import point_union, validate_points
from pedestrian_behavior.data.preparation import checksum, load_track, native_float
from pedestrian_behavior.datasets import loki
from pedestrian_behavior.inspection.feature_probe import box_membership, relative_xyz
from pedestrian_behavior.inspection.output_pooling import pool_outputs
from .e002a_probe import write_json


def cache_sample(root, setup):
    """One random frame/group, up to 20 training/8 validation groups; two size stresses."""
    db = sqlite3.connect((root / "index.sqlite3").resolve().as_uri() + "?mode=ro", uri=True)
    grouped = defaultdict(list)
    try:
        for key, identity, size, sha in db.execute("SELECT key,identity,bytes,sha256 FROM entries"):
            identity = json.loads(identity)
            if identity.get("dataset") == "road-waymo":
                if identity["setup_sha256"] != setup["sha256"]:
                    raise ValueError("Retained ROAD cache references another setup")
                group = identity["group"]
                grouped[group].append((key, identity, size, sha))
    finally:
        db.close()
    rng, chosen = np.random.default_rng(0), {}
    for split, limit in (("training", 20), ("validation", 8)):
        groups = sorted(g for g in grouped if setup["groups"][g] == split)
        for g in rng.choice(groups, min(limit, len(groups)), replace=False):
            rows = sorted(grouped[g])
            row = rows[int(rng.integers(len(rows)))]
            chosen[row[0]] = (*row, "group-balanced")
    all_rows = [r for g, rows in grouped.items() if setup["groups"][g] in ("training", "validation") for r in rows]
    for row in sorted(all_rows, key=lambda r: (r[2], r[0]))[-2:]:
        chosen.setdefault(row[0], (*row, "largest-retained"))
    return sorted(chosen.values(), key=lambda r: (r[1]["group"], r[1]["key"]))


def fixture_union(frame, probe, source, outputs, protected, contexts):
    """Rebuild LOKI's current 20 m union from full saved outputs and native cuboids only."""
    group = frame["group"]
    if group not in contexts:
        collection = outputs / "experiments/E001/reader-preparation/loki"
        manifest_path = collection / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        reader = loki.TrackReader(source["config"]["loki_root"], probe / "loki-transform.json")
        actual = reader.index_scene(group)["provenance"]["annotation_odometry_sha256"]
        if actual != manifest["scenes"][group]["annotation_odometry_sha256"]:
            raise ValueError("LOKI native annotations differ from frozen collection")
        protected[str(manifest_path)] = checksum(manifest_path)
        contexts[group] = collection, reader, {}
    collection, reader, saved = contexts[group]
    paths = {k: Path(v) if v else None for k, v in frame["files"].items()}
    native = loki.read_frame(reader.root / group, frame["key"], paths)
    for path in (paths["label3d"], paths["odometry"]):
        if path:
            protected[str(path)] = checksum(path)
    boxes, anchors = {}, {}
    for track in frame["tracks"]:
        if track.get("missing_box_or_pose"):
            continue
        locator = track["locator"]
        if locator not in saved:
            path = collection / "tracks" / track["archive"]
            m, a = load_track(path)
            protected[str(path)] = checksum(path)
            saved[locator] = m, a
        m, a = saved[locator]
        slot = m["source_frames"].index(frame["key"])
        if not (m["anchor_valid"] and a["ped_position_valid"][slot] and a["ego_pose_valid"][slot]
                and m["availability"][slot].get("native_3d", False)):
            continue
        row = native.pedestrians[json.loads(locator)["track_id"]].label3d
        center, size = (np.array([native_float(row[p + axis]) for axis in "xyz"]) for p in ("pos_", "dim_"))
        boxes[locator] = center, size, native_float(row["yaw"])
        first = m["source_frames"][0]
        anchors[locator] = reader.read_frames(group, [first])[first]["ego_pose"]
    path = probe / "fixtures" / f"loki-{group}-{frame['key']}-spatial.npz"
    protected[str(path)] = checksum(path)
    with np.load(path, allow_pickle=False) as a:
        points = np.c_[a["coord"], np.zeros(len(a["coord"]), np.float32)]
        union = point_union(points, a["feature"], np.arange(len(points), dtype=np.int64),
                            boxes, a["world_from_native"], anchors)
        union["native_indices"] = a["selected_native_index"][union["native_indices"]]
    validate_points(union)
    return union


def diagnostic_payload(arrays, mode):
    xyz, features = arrays["coordinates"], arrays["features"]
    before = time.perf_counter()
    if mode == "original":
        pooled_xyz, pooled, inverse, counts = xyz, features, np.arange(len(xyz)), np.ones(len(xyz), np.int64)
    else:
        preserve = np.zeros(len(xyz), bool)
        if mode == "adaptive":
            for box in arrays["boxes"]:
                preserve |= box_membership(xyz, box[:3], box[3:6] + 1., box[6])
        pooled_xyz, pooled, inverse, counts = pool_outputs(xyz, features, .1 if mode == "10cm" else .2, preserve)
    pooling_seconds = time.perf_counter() - before
    if len(pooled) >= 2**32:
        raise ValueError("Crop indices exceed uint32")
    representatives = np.full(len(pooled), np.iinfo(np.int64).max, np.int64)
    np.minimum.at(representatives, inverse, arrays["native_indices"])
    indices, offsets, crop_counts, target_rows = [], [0], [], []
    for box, anchor in zip(arrays["boxes"], arrays["anchors"]):
        relative = relative_xyz(pooled_xyz, box[:3], arrays["world_from_native"], anchor)
        rows = np.flatnonzero(np.linalg.norm(relative[:, :2], axis=1) <= 20).astype(np.uint32)
        indices.append(rows)
        offsets.append(offsets[-1] + len(rows))
        original_relative = relative_xyz(xyz, box[:3], arrays["world_from_native"], anchor)
        original_crop = np.linalg.norm(original_relative[:, :2], axis=1) <= 20
        crop_counts.append([int(original_crop.sum()), len(rows)])
        inside = box_membership(xyz, box[:3], box[3:6], box[6])
        inside_counts = np.bincount(inverse, weights=inside, minlength=len(pooled))
        mixed = (inside_counts > 0) & (inside_counts < counts)
        target_rows.append({"original_inside": int(inside.sum()),
            "pooled_centroids_inside": int(box_membership(pooled_xyz, box[:3], box[3:6], box[6]).sum()),
            "inside_tokens_mixed_with_outside": int(inside_counts[mixed].sum())})
    error = features.astype(np.float64) - pooled[inverse]
    feature_energy = float(np.square(features.astype(np.float64)).sum())
    displacement = np.linalg.norm(xyz - pooled_xyz[inverse], axis=1)
    half = pooled.astype(np.float16)
    if not np.isfinite(half).all():
        raise ValueError("FP16 feature overflow")
    payload = {k: arrays[k] for k in ("locators", "boxes", "anchors", "world_from_native")}
    payload.update(features=pooled, coordinates=pooled_xyz, representative_native_index=representatives,
                   member_counts=counts.astype(np.uint32), crop_indices=np.concatenate(indices) if indices else np.zeros(0, np.uint32),
                   crop_offsets=np.array(offsets, np.int64))
    metrics = {"tokens": len(pooled), "original_tokens": len(xyz), "pooling_seconds": pooling_seconds,
        "crop_counts": crop_counts, "targets": target_rows,
        "feature_squared_error": float(np.square(error).sum()), "feature_energy": feature_energy,
        "centroid_displacement_mean_m": float(displacement.mean()) if len(xyz) else 0.,
        "centroid_displacement_max_m": float(displacement.max()) if len(xyz) else 0.,
        "fp16_squared_error": float(np.square(pooled.astype(np.float64) - half).sum()),
        "pooled_feature_energy": float(np.square(pooled.astype(np.float64)).sum())}
    return payload, metrics


def storage_measure(payload, path):
    """Warm-file read+SHA+NPZ, then all crop gathers/geometry; no GPU or cold-read claim."""
    measurements = {}
    for precision in ("fp32", "fp16"):
        stored = dict(payload, features=payload["features"].astype(np.float32 if precision == "fp32" else np.float16))
        if shutil.disk_usage(path.parent).free < 20 * 10**9 + sum(a.nbytes for a in stored.values()) * 2:
            raise OSError("Insufficient diagnostic temporary-file headroom")
        before = time.perf_counter()
        np.savez(path, **stored)
        write_seconds = time.perf_counter() - before
        sha = checksum(path)
        reads, gathers = [], []
        for _ in range(3):
            before = time.perf_counter()
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Diagnostic file checksum mismatch")
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                loaded = {k: archive[k] for k in archive.files}
            loaded["features"] = loaded["features"].astype(np.float32, copy=False)
            reads.append(time.perf_counter() - before)
            for name, expected in stored.items():
                np.testing.assert_array_equal(loaded[name], expected.astype(np.float32) if name == "features" else expected)
            before = time.perf_counter()
            for i, (box, anchor) in enumerate(zip(loaded["boxes"], loaded["anchors"])):
                rows = loaded["crop_indices"][loaded["crop_offsets"][i]:loaded["crop_offsets"][i+1]]
                features = loaded["features"][rows]
                coordinates = loaded["coordinates"][rows]
                relative = relative_xyz(coordinates, box[:3], loaded["world_from_native"], anchor)
                inside = box_membership(coordinates, box[:3], box[3:6], box[6])
                assert features.shape == (len(rows), 64) and len(relative) == len(inside) == len(rows)
            gathers.append(time.perf_counter() - before)
        measurements[precision] = {"bytes": path.stat().st_size, "write_seconds": write_seconds,
            "warm_read_check_load_seconds": float(np.median(reads)), "all_crop_gather_geometry_seconds": float(np.median(gathers))}
        path.unlink()
    return measurements


def summarize(record):
    result = []
    for dataset in ("loki", "road-waymo"):
        rows = [r for r in record["frames"] if r["dataset"] == dataset]
        base = [r["modes"]["original"] for r in rows]
        for mode in ("original", "10cm", "20cm", "adaptive"):
            values = [r["modes"][mode] for r in rows]
            total = sum(v["original_tokens"] for v in values)
            targets = [t for v in values for t in v["targets"]]
            out = {"dataset": dataset, "mode": mode, "frames": len(rows), "groups": len({r["group"] for r in rows}),
                "tokens_retained": sum(v["tokens"] for v in values) / total,
                "crop_tokens_retained": sum(c[1] for v in values for c in v["crop_counts"]) / sum(c[0] for v in values for c in v["crop_counts"]),
                "feature_relative_l2": (sum(v["feature_squared_error"] for v in values) / sum(v["feature_energy"] for v in values)) ** .5,
                "target_inside_mixed_fraction": sum(t["inside_tokens_mixed_with_outside"] for t in targets) / max(1, sum(t["original_inside"] for t in targets)),
                "storage": {}}
            for precision in ("fp32", "fp16"):
                out["storage"][precision] = {"sample_bytes": sum(v["storage"][precision]["bytes"] for v in values),
                    "ratio_to_original_fp32": sum(v["storage"][precision]["bytes"] for v in values) / sum(v["storage"]["fp32"]["bytes"] for v in base),
                    "mean_warm_read_ms": 1000 * np.mean([v["storage"][precision]["warm_read_check_load_seconds"] for v in values]),
                    "mean_all_crop_gather_ms": 1000 * np.mean([v["storage"][precision]["all_crop_gather_geometry_seconds"] for v in values])}
            result.append(out)
    return result


def run(args):
    probe, output = args.probe.resolve(), args.output.resolve()
    outputs = probe.parents[2]
    if not output.is_relative_to(outputs / "experiments/E002a"):
        raise ValueError("Diagnostic output must be inside E002a outputs")
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((probe / "probe.json").read_text())
    if source["status"] != "complete":
        raise ValueError("Completed frozen probe required")
    protected = {str(probe / "probe.json"): checksum(probe / "probe.json")}
    record = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_sha256": checksum(__file__), "numpy": np.__version__, "frames": [],
        "policy": "Native-origin grid, arithmetic mean features/centroid. Adaptive preserves every original token within any represented box +0.5m each side, context20cm. 20m union/crops. No native inverse map persisted; representative native ref +member count only. FP16 features only. CPU, 3 warm reads with SHA/NPZ/cast; gathers separate. No training, encoder inference or production cache writes.",
        "sampling": "ROAD seed0 one random retained frame/group, up to20 train/8 validation groups plus2 largest; retained LRU is biased. LOKI all14 saved spatial fixtures (4 training groups, random+stress); no validation fixtures. No held-out test access.",
        "protected": protected}
    started = time.perf_counter()
    try:
        contexts = {}
        for frame in source["frames"]:
            path = probe / "fixtures" / f"loki-{frame['group']}-{frame['key']}-spatial.npz"
            if frame["dataset"] != "loki" or not path.exists():
                continue
            arrays = fixture_union(frame, probe, source, outputs, protected, contexts)
            measure_frame(record, arrays, {k: frame[k] for k in ("dataset", "group", "key", "split", "reasons")}, output)
        setup_path = outputs / "experiments/E001/data-setup/road-waymo/setup.json"
        protected[str(setup_path)] = checksum(setup_path)
        setup = json.loads(setup_path.read_text()) | {"sha256": protected[str(setup_path)]}
        sample = cache_sample(args.point_cache.resolve(), setup)
        for key, identity, size, sha, reason in sample:
            path = args.point_cache.resolve() / "entries" / (key + ".npz")
            raw = path.read_bytes()
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError("Selected cache payload changed or is corrupt")
            with np.load(BytesIO(raw), allow_pickle=False) as a:
                embedded = json.loads(a["__identity__"].item())
                if embedded != identity:
                    raise ValueError("Selected cache identity mismatch")
                arrays = {k: a[k] for k in a.files if k != "__identity__"}
            validate_points(arrays)
            protected[str(path)] = sha
            measure_frame(record, arrays, {"dataset": "road-waymo", "group": identity["group"], "key": identity["key"],
                "split": setup["groups"][identity["group"]], "reasons": [reason], "cache_identity": identity}, output)
        record["summary"] = summarize(record)
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        (output / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        record["protected_unchanged"] = all(Path(p).is_file() and checksum(p) == sha for p, sha in protected.items())
        record["elapsed_seconds"] = time.perf_counter() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "measurement.json", record)
        if not record["protected_unchanged"]:
            raise ValueError("A selected protected source changed during measurement")
    print(json.dumps(record["summary"], indent=2), flush=True)


def measure_frame(record, arrays, item, output):
    item["modes"] = {}
    for mode in ("original", "10cm", "20cm", "adaptive"):
        payload, metrics = diagnostic_payload(arrays, mode)
        metrics["storage"] = storage_measure(payload, output / "temporary-frame.npz")
        item["modes"][mode] = metrics
    record["frames"].append(item)
    write_json(output / "measurement.json", record)
    print(item["dataset"], item["group"], item["key"], {m: v["tokens"] for m, v in item["modes"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--point-cache", type=Path, default=Path("outputs/experiments/E002a/point-cache"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
