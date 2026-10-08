"""Additive E001b bbox evidence on the frozen E001 population and time grid."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from pedestrian_behavior.data.bbox import COLUMNS, SOURCES, bbox_inputs, box_geometry, load_bbox, normalization_sample
from pedestrian_behavior.data.features import fit_normalization
from pedestrian_behavior.data.loading import TrackDataset
from pedestrian_behavior.data.preparation import checksum, load_manifest, load_track, source_times
from pedestrian_behavior.datasets.bbox import read_loki_boxes, read_road_boxes, read_waymo_boxes
from .e001 import CLASSES, dataset_from_setup, targets

POLICY = {"baseline": "K+T+R", "configurations": ["availability", "geometry"],
          "numeric_columns": COLUMNS, "flags": ["bbox_valid", "bbox_velocity_valid"], "sources": SOURCES,
          "road_source_rule": "ROAD when present; native FRONT camera box only when ROAD is absent",
          "coordinates": "image-relative xyxy; retain native extents without clipping",
          "invalid_box": "nonfinite or nonpositive width/height: masked and reported; no fallback substitution",
          "derivatives": "E001 immediate-neighbor rule using selected source times; no gaps or source switches",
          "normalization": "valid source-training context only; population std; missing numeric zero after standardization",
          "population": "exact eligible E001 tracks/splits/targets; no restriction by 2D or 3D availability"}


def _support():
    return {"context_frames": 0, "gt_frames": 0, "class_frames": [0]*4,
            "groups": set(), "tracks": set(), "gt_groups": set(), "gt_tracks": set(),
            "class_groups": [set() for _ in CLASSES], "class_tracks": [set() for _ in CLASSES]}


def _add_support(record, entry, y, gt, mask):
    record["context_frames"] += int(mask.sum())
    accepted = gt & mask
    record["gt_frames"] += int(accepted.sum())
    for field, condition in (("", mask.any()), ("gt_", accepted.any())):
        if condition:
            record[field+"groups"].add(entry["group"])
            record[field+"tracks"].add(entry["locator"])
    for label in range(4):
        count = int((accepted & (y == label)).sum())
        record["class_frames"][label] += count
        if count:
            record["class_groups"][label].add(entry["group"])
            record["class_tracks"][label].add(entry["locator"])


def prepare_bbox_extension(collection, setup_path, native_input, output):
    collection, setup_path, native_input, output = map(Path, (collection, setup_path, native_input, output))
    if any(output.resolve().is_relative_to(p.resolve()) for p in (collection, setup_path.parent, native_input)):
        raise ValueError("Extension output must be separate from frozen/native data")
    original = load_manifest(collection)
    setup = json.loads(setup_path.read_text())
    if (setup.get("status") != "complete" or setup.get("experiment") != "E001" or setup.get("schema_version") != 1
            or setup["dataset"] != original["dataset"] or checksum(collection/"manifest.json") != setup["collection"]["manifest_sha256"]):
        raise ValueError("Incomplete or mismatched E001 collection/setup")
    entries = [e for e in setup["tracks"] if not e["exclusions"]]
    if (not entries or len({e["locator"] for e in entries}) != len(entries)
            or len({e["archive"] for e in entries}) != len(entries)
            or any(e["archive"] not in original["tracks"] or e["split"] not in ("training", "validation", "test")
                   or setup["groups"][e["group"]] != e["split"] for e in entries)):
        raise ValueError("Invalid frozen E001 population")
    # Include excluded archives too when proving that preparation left E001 untouched.
    protected = [collection/"manifest.json", collection/"audit.json", setup_path]
    protected += [collection/"tracks"/name for name in original["tracks"]]
    before = {str(p.resolve()): checksum(p) for p in protected}
    output.mkdir(parents=True, exist_ok=False)
    (output/"tracks").mkdir()
    started = time.monotonic()
    record = {"schema_version": 1, "experiment": "E001b", "dataset": original["dataset"], "status": "incomplete",
              "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
              "policy": POLICY, "clock": original["clock"], "classes": CLASSES,
              "collection": {"path": str(collection.resolve()), "manifest_sha256": checksum(collection/"manifest.json")},
              "setup": {"path": str(setup_path.resolve()), "sha256": checksum(setup_path)},
              "native_input": str(native_input.resolve()), "groups": setup["groups"], "population": setup["population"],
              "baseline_normalization": setup["normalization"]["K+T+R"], "scenes": {}, "tracks": [], "failures": [],
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "host": platform.node(),
                  "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
                  "code_sha256": {str(p): checksum(p) for p in sorted(Path("src").rglob("*.py"))}}}
    def save():
        (output/"manifest.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    save()
    coverage = {split: defaultdict(_support) for split in ("training", "validation", "test")}
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["group"]].append(entry)
    try:
        road = None
        if original["dataset"] == "road-waymo":
            native_hashes = {f: checksum(native_input/f) for f in original["sources"]["sha256"]}
            if native_hashes != original["sources"]["sha256"]:
                raise ValueError("Native ROAD index differs from frozen E001")
            record["native_index_sha256"] = native_hashes
            road = read_road_boxes(native_input, {g: {json.loads(e["locator"])["tube_uid"] for e in rows} for g, rows in grouped.items()})
        for scene_number, (scene, rows) in enumerate(sorted(grouped.items()), 1):
            saved = {e["archive"]: load_track(collection/"tracks"/e["archive"]) for e in rows}
            requested = {f for metadata, _ in saved.values() for f in metadata["source_frames"] if f is not None}
            if road is None:
                native, provenance = read_loki_boxes(native_input, scene, requested, original["scenes"][scene]["annotation_odometry_sha256"])
            else:
                waymo, size, provenance = read_waymo_boxes(original["scenes"][scene], {json.loads(e["locator"])["tube_uid"] for e in rows})
            record["scenes"][scene] = provenance
            for entry in rows:
                metadata, arrays3d = saved[entry["archive"]]
                if metadata["track_locator"] != entry["locator"]:
                    raise ValueError("Frozen setup/archive identity mismatch")
                n = len(metadata["source_frames"])
                corners, dimensions = np.full((n, 4), np.nan), np.zeros((n, 2), dtype=np.int64)
                source = np.zeros(n, dtype=np.uint8)
                presence = {key: np.array([bool(a.get(key, False)) for a in metadata["availability"]])
                            for key in ("loki_2d", "road_2d", "waymo_2d")}
                identity = json.loads(entry["locator"])
                dual_count, dual_error, issues = 0, 0., []
                for i, frame in enumerate(metadata["source_frames"]):
                    if frame is None:
                        continue
                    if road is None:
                        boxes = native[frame]["boxes"]
                        dimensions[i] = native[frame]["image_size"]
                        box = boxes.get(identity["track_id"])
                        if (box is not None) != presence["loki_2d"][i]:
                            raise ValueError("Native LOKI presence differs from E001")
                        if box is not None:
                            corners[i], source[i] = box, 1
                    else:
                        key = identity["tube_uid"], frame
                        road_box, waymo_box = road[scene].get(key), waymo.get(key)
                        dimensions[i] = size
                        if (road_box is not None) != presence["road_2d"][i] or (waymo_box is not None) != presence["waymo_2d"][i]:
                            raise ValueError("Native ROAD/Waymo presence differs from E001")
                        if road_box is not None:
                            corners[i], source[i] = road_box, 2
                        elif waymo_box is not None:
                            corners[i], source[i] = waymo_box, 3
                        if road_box is not None and waymo_box is not None:
                            delta = np.abs(np.asarray(road_box)-waymo_box)
                            if np.isfinite(delta).all():
                                dual_count += 1
                                dual_error = max(dual_error, float(delta.max()))
                arrays = box_geometry(corners, source_times(metadata), source) | presence | {"image_size": dimensions}
                invalid = (source != 0) & ~arrays["bbox_valid"]
                for i in np.flatnonzero(invalid):
                    issues.append({"slot": int(i), "reason": "nonfinite-box" if not np.isfinite(corners[i]).all() else "nonpositive-box-size"})
                reference = {"track_locator": metadata["track_locator"], "source_frames": metadata["source_frames"],
                             "source_time_us": source_times(metadata), "track_sha256": before[str((collection/"tracks"/entry["archive"]).resolve())],
                             "issues": issues}
                archive = output/"tracks"/entry["archive"]
                np.savez_compressed(archive, metadata=np.array(json.dumps(reference, sort_keys=True, allow_nan=False)), **arrays)
                _, reloaded = load_bbox(archive, metadata, reference["track_sha256"])
                if any(not np.array_equal(v, reloaded[k], equal_nan=True) for k, v in arrays.items()):
                    raise ValueError("BBox persistence changed values")
                y, gt = targets(metadata, original["dataset"], setup["native_audit"]["verified_overrides"])
                if n != entry["slots"] or int(gt.sum()) != entry["gt_frames"] or [int((gt & (y == j)).sum()) for j in range(4)] != entry["class_frames"]:
                    raise ValueError("Frozen GT/support differs from E001")
                valid = arrays["bbox_valid"]
                track_slice = "never" if not valid.any() else "complete" if valid.all() else "partial"
                slices = {"full": np.ones(n, dtype=bool), "bbox-valid": valid, "bbox-missing": ~valid,
                          "bbox-velocity-valid": arrays["bbox_velocity_valid"],
                          "bbox-valid-without-3d": valid & ~arrays3d["ped_position_valid"],
                          "track-"+track_slice: np.ones(n, dtype=bool)}
                for name, mask in slices.items():
                    _add_support(coverage[entry["split"]][name], entry, y, gt, mask)
                record["tracks"].append(entry | {"e001_sha256": reference["track_sha256"], "bbox_sha256": checksum(archive),
                    "bbox_frames": int(valid.sum()), "bbox_velocity_frames": int(arrays["bbox_velocity_valid"].sum()),
                    "invalid_box_frames": int(invalid.sum()), "outside_image_frames": int((valid & ((corners < 0) | (corners > 1)).any(axis=1)).sum()),
                    "source_frames": {SOURCES[k]: int((source == k).sum()) for k in SOURCES},
                    "dual_source_frames": dual_count, "dual_source_max_coordinate_difference": dual_error if dual_count else None})
            if road is not None:
                del road[scene]
            if scene_number % 20 == 0:
                save()
                print(f"{original['dataset']}: {scene_number}/{len(grouped)} groups, {len(record['tracks'])} tracks", flush=True)
        def training_samples():
            for entry in record["tracks"]:
                if entry["split"] == "training":
                    metadata, _ = load_track(collection/"tracks"/entry["archive"])
                    _, arrays = load_bbox(output/"tracks"/entry["archive"], metadata, entry["e001_sha256"])
                    yield normalization_sample(arrays)
        record["normalization"] = fit_normalization(training_samples())
        for split in coverage:
            for name in ("full", "bbox-valid", "bbox-missing", "bbox-velocity-valid", "bbox-valid-without-3d", "track-never", "track-partial", "track-complete"):
                coverage[split][name]
        record["coverage"] = {split: {name: {k: len(v) if isinstance(v, set) else [len(s) for s in v] if k in ("class_groups", "class_tracks") else v
            for k, v in support.items()} for name, support in slices.items()} for split, slices in coverage.items()}
        if road is not None and {f: checksum(native_input/f) for f in native_hashes} != native_hashes:
            raise ValueError("Native ROAD index changed during preparation")
        record["status"] = "complete"
        record["training_gate"] = "Review measured coverage and association qualifications; E001b training not run"
    except Exception as error:
        record["status"] = "failed"
        record["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        unchanged = all(checksum(Path(p)) == digest for p, digest in before.items())
        record["e001_unchanged"] = {"files_checked": len(before), "all_sha256_match": unchanged}
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["elapsed_seconds"] = time.monotonic()-started
        if not unchanged:
            record["status"] = "failed"
            record["failures"].append("E001 files changed during bbox preparation")
        save()
        if not unchanged:
            raise ValueError("E001 files changed during bbox preparation")
    return record


class BboxTrackDataset(TrackDataset):
    """Append bbox evidence while retaining E001 sample/batch interfaces."""

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        entry = self.entries[index]
        metadata, _ = load_track(sample["archive"])
        if checksum(sample["archive"]) != entry["e001_sha256"]:
            raise ValueError("E001 archive changed after bbox preparation")
        archive = self.extension/"tracks"/entry["archive"]
        if checksum(archive) != entry["bbox_sha256"]:
            raise ValueError("BBox archive changed after preparation")
        _, arrays = load_bbox(archive, metadata, entry["e001_sha256"])
        sample["inputs"] = torch.cat((sample["inputs"], torch.from_numpy(bbox_inputs(arrays, self.bbox_configuration, self.bbox_statistics))), dim=1)
        return sample


def dataset_from_extension(collection, setup_path, extension, split, configuration, baseline_statistics, bbox_statistics):
    """Pass both frozen SOURCE statistics explicitly, including for transfer."""
    if configuration not in POLICY["configurations"]:
        raise ValueError("Unknown E001b configuration")
    extension = Path(extension)
    record = json.loads((extension/"manifest.json").read_text())
    if (record.get("schema_version") != 1 or record.get("experiment") != "E001b" or record.get("status") != "complete"
            or record["policy"] != json.loads(json.dumps(POLICY)) or checksum(setup_path) != record["setup"]["sha256"]
            or checksum(Path(collection)/"manifest.json") != record["collection"]["manifest_sha256"]):
        raise ValueError("Incomplete or mismatched bbox extension")
    baseline = dataset_from_setup(collection, setup_path, split, "K+T+R", baseline_statistics)
    extension_entries = {e["locator"]: e for e in record["tracks"] if e["split"] == split}
    if set(extension_entries) != {e["locator"] for e in baseline.entries}:
        raise ValueError("BBox extension changes the E001 population")
    entries = [extension_entries[e["locator"]] for e in baseline.entries]
    if any(e["archive"] != b["archive"] for e, b in zip(entries, baseline.entries)):
        raise ValueError("BBox extension archive identity mismatch")
    dataset = BboxTrackDataset(collection, entries, "K+T+R", baseline_statistics, baseline.target_policy)
    dataset.extension, dataset.bbox_configuration, dataset.bbox_statistics = extension, configuration, bbox_statistics
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--native-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = prepare_bbox_extension(args.collection, args.setup, args.native_input, args.output)
    print(json.dumps({"status": record["status"], "tracks": len(record["tracks"]), "elapsed_seconds": record["elapsed_seconds"],
                      "e001_unchanged": record["e001_unchanged"], "coverage": record["coverage"]}, indent=2))
