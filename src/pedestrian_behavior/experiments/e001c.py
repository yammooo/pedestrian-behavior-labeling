"""Recompute E001c bbox inputs from native data on the frozen E001 population."""

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

from pedestrian_behavior.data.boxes import COLUMNS, CONFIGURATIONS, FLAGS, box_features, box_inputs, column_validity, load_boxes
from pedestrian_behavior.data.features import fit_normalization
from pedestrian_behavior.data.preparation import checksum, load_manifest, load_track, source_times
from pedestrian_behavior.datasets.boxes import read_loki_geometry, read_waymo_geometry
from pedestrian_behavior.datasets.loki import TrackReader as LokiReader
from .e001 import CLASSES, targets

POLICY = {"numeric_columns": COLUMNS, "flags": FLAGS, "configurations": CONFIGURATIONS,
          "coordinates3d": "native box position transformed to world, initial ego xyz subtracted, initial yaw removed",
          "dimensions3d": "native dim_x/dim_y/dim_z (LOKI); size.x/size.y/size.z (Waymo); no sorting or heading input",
          "loki_anchor_qualification": "native position retained without height correction; center supported by cloud samples, not explicitly defined by release documentation",
          "coordinates2d": "native geometric center u/v and width/height divided by camera W/H; no clipping",
          "sources2d": "LOKI native boxes; Waymo FRONT only, no ROAD fallback; independent of GT",
          "velocity": "E001 immediate-neighbor unequal-time centered/one-sided differences per second; no interpolation or gap bridging",
          "validity": "four shared flags; fail if 3D center/size usability differs; individual evidence retained",
          "normalization": "all valid source-training context only; population std; missing zero after standardization",
          "population": "frozen eligible E001 identities, associations, source frames/times, targets and splits"}


def _support():
    return {"tracks": set(), "groups": set(), "context_frames": 0, "gt_frames": 0, "class_frames": np.zeros(4, dtype=np.int64)}


def prepare_boxes(collection, setup_path, native_input, output, waymo_root=None, loki_transform=None):
    collection, setup_path, native_input, output = map(Path, (collection, setup_path, native_input, output))
    original, setup = load_manifest(collection), json.loads(setup_path.read_text())
    roots = [collection, setup_path.parent, native_input] + ([Path(waymo_root)] if waymo_root else [])
    if any(output.resolve().is_relative_to(root.resolve()) or root.resolve().is_relative_to(output.resolve()) for root in roots):
        raise ValueError("Output must be separate from frozen/native data")
    if (setup.get("status") != "complete" or setup.get("experiment") != "E001" or setup.get("schema_version") != 1
            or setup["dataset"] != original["dataset"] or checksum(collection/"manifest.json") != setup["collection"]["manifest_sha256"]):
        raise ValueError("Incomplete or mismatched frozen E001 setup")
    entries = [e for e in setup["tracks"] if not e["exclusions"]]
    if (not entries or len({e["locator"] for e in entries}) != len(entries)
            or len({e["archive"] for e in entries}) != len(entries)
            or any(e["archive"] not in original["tracks"] or e["split"] not in ("training", "validation", "test")
                   or setup["groups"][e["group"]] != e["split"] for e in entries)):
        raise ValueError("Invalid frozen population/splits")
    protected = [collection/"manifest.json", collection/"audit.json", setup_path]
    protected += [collection/"tracks"/name for name in original["tracks"]]
    before = {str(p.resolve()): checksum(p) for p in protected}
    output.mkdir(parents=True, exist_ok=False)
    (output/"tracks").mkdir()
    started = time.monotonic()
    record = {"schema_version": 1, "experiment": "E001c", "status": "incomplete", "dataset": original["dataset"],
              "created_utc": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
              "policy": POLICY, "classes": CLASSES, "clock": original["clock"], "groups": setup["groups"],
              "population": setup["population"], "collection": {"path": str(collection.resolve()), "manifest_sha256": checksum(collection/"manifest.json")},
              "setup": {"path": str(setup_path.resolve()), "sha256": checksum(setup_path)},
              "native_input": str(native_input.resolve()), "waymo_root": str(Path(waymo_root).resolve()) if waymo_root else None,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "host": platform.node(),
                  "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
                  "code_sha256": {str(p): checksum(p) for p in sorted(Path("src").rglob("*.py"))}},
              "scenes": {}, "tracks": [], "failures": []}
    def save():
        temporary = output/"manifest.tmp"
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
        temporary.replace(output/"manifest.json")
    save()
    grouped = defaultdict(list)
    coverage = {split: defaultdict(_support) for split in ("training", "validation", "test")}
    for entry in entries:
        grouped[entry["group"]].append(entry)
    try:
        reader = None
        if original["dataset"] == "loki":
            if waymo_root is not None or loki_transform is None:
                raise ValueError("LOKI requires --loki-transform and no --waymo-root")
            reader = LokiReader(native_input, loki_transform)
            if reader.contract != original["sources"]["transform"]:
                raise ValueError("LOKI transform differs from frozen E001")
            record["transform_sha256"] = checksum(loki_transform)
        elif original["dataset"] == "road-waymo":
            if loki_transform is not None:
                raise ValueError("LOKI transform cannot apply to ROAD-Waymo")
            hashes = {name: checksum(native_input/name) for name in original["sources"]["sha256"]}
            if hashes != original["sources"]["sha256"]:
                raise ValueError("Native ROAD index differs from frozen E001")
            record["native_index_sha256"] = hashes
        else:
            raise ValueError("Unsupported dataset")
        for scene_number, (scene, rows) in enumerate(sorted(grouped.items()), 1):
            saved = {e["archive"]: load_track(collection/"tracks"/e["archive"]) for e in rows}
            requested = {f for metadata, _ in saved.values() for f in metadata["source_frames"] if f is not None}
            identities = {e["locator"]: json.loads(e["locator"]).get("track_id", json.loads(e["locator"]).get("tube_uid")) for e in rows}
            native, evidence = (read_loki_geometry(reader, scene, requested, original["scenes"][scene]) if reader else
                read_waymo_geometry(original["scenes"][scene], set(identities.values()), requested, waymo_root))
            record["scenes"][scene] = evidence
            for entry in rows:
                metadata, old = saved[entry["archive"]]
                if metadata["track_locator"] != entry["locator"]:
                    raise ValueError("Frozen setup/archive identity mismatch")
                n, track = len(metadata["source_frames"]), identities[entry["locator"]]
                centers, sizes, corners = np.full((n, 3), np.nan), np.full((n, 3), np.nan), np.full((n, 4), np.nan)
                image_size = np.zeros((n, 2), dtype=np.int64)
                presence = {k: np.array([bool(a.get(k, False)) for a in metadata["availability"]])
                            for k in ("native_3d", "loki_2d", "road_2d", "waymo_2d")}
                for i, frame in enumerate(metadata["source_frames"]):
                    if frame is None:
                        continue
                    person, camera = native[frame]["pedestrians"].get(track, {}), native[frame]["camera"]
                    center, size, box = person.get("world_position"), person.get("size"), camera["boxes"].get(track)
                    if (size is not None) != presence["native_3d"][i] or (box is not None) != presence["loki_2d" if reader else "waymo_2d"][i]:
                        raise ValueError("Native box presence differs from frozen E001")
                    if center is not None:
                        centers[i] = center
                    if size is not None:
                        sizes[i] = size
                    if box is not None:
                        corners[i] = box
                    image_size[i] = camera["image_size"]
                anchor = native[metadata["source_frames"][0]]["ego_pose"] if metadata["source_frames"][0] is not None else None
                arrays = box_features(centers, sizes, corners, source_times(metadata), anchor)
                # Check independently rebuilt xy against frozen E001, never use its numbers as inputs.
                if (not np.array_equal(arrays["box3d_valid"], old["ped_position_valid"])
                        or not np.allclose(arrays["values"][:, :2], old["ped_position"], rtol=0, atol=1e-8, equal_nan=True)):
                    raise ValueError("Fresh 3D geometry differs from frozen coordinates")
                y, gt = targets(metadata, original["dataset"], setup["native_audit"]["verified_overrides"])
                if n != entry["slots"] or int(gt.sum()) != entry["gt_frames"] or [int((gt & (y == j)).sum()) for j in range(4)] != entry["class_frames"]:
                    raise ValueError("Frozen GT/support differs from E001")
                arrays.update(presence | {"image_size": image_size, "targets": y, "gt_valid": gt})
                reference = {"track_locator": metadata["track_locator"], "source_frames": metadata["source_frames"],
                    "source_time_us": source_times(metadata), "native_extent_us": metadata["native_extent_us"],
                    "track_sha256": before[str((collection/"tracks"/entry["archive"]).resolve())]}
                path = output/"tracks"/entry["archive"]
                np.savez_compressed(path, metadata=np.array(json.dumps(reference, sort_keys=True, allow_nan=False)), **arrays)
                _, actual = load_boxes(path, metadata, reference["track_sha256"])
                if any(not np.array_equal(value, actual[key], equal_nan=True) for key, value in arrays.items()):
                    raise ValueError("E001c persistence changed values")
                counts = {key: int(arrays[key].sum()) for key in FLAGS}
                record["tracks"].append(entry | {"e001_sha256": reference["track_sha256"], "boxes_sha256": checksum(path), "valid_frames": counts,
                    "invalid_native_2d_frames": int(((presence["loki_2d"] | presence["waymo_2d"]) & ~arrays["box2d_valid"]).sum())})
                slices = {"full": np.ones(n, dtype=bool)} | {key: arrays[key] for key in FLAGS}
                for name, mask in slices.items():
                    support = coverage[entry["split"]][name]
                    support["context_frames"] += int(mask.sum())
                    support["gt_frames"] += int((mask & gt).sum())
                    support["class_frames"] += np.array([int((mask & gt & (y == j)).sum()) for j in range(4)])
                    if mask.any():
                        support["tracks"].add(entry["locator"])
                        support["groups"].add(entry["group"])
            if scene_number % 50 == 0:
                save()
                print(f"{original['dataset']}: {scene_number}/{len(grouped)} groups; {len(record['tracks'])} tracks", flush=True)
        def samples():
            for entry in record["tracks"]:
                if entry["split"] == "training":
                    metadata, _ = load_track(collection/"tracks"/entry["archive"])
                    _, arrays = load_boxes(output/"tracks"/entry["archive"], metadata, entry["e001_sha256"])
                    yield arrays["values"], column_validity(arrays)
        record["normalization"] = fit_normalization(samples())
        record["coverage"] = {split: {name: {key: len(value) if isinstance(value, set) else value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in support.items()} for name, support in slices.items()} for split, slices in coverage.items()}
        for entry in record["tracks"]:
            metadata, _ = load_track(collection/"tracks"/entry["archive"])
            _, arrays = load_boxes(output/"tracks"/entry["archive"], metadata, entry["e001_sha256"])
            for configuration in CONFIGURATIONS:
                if not np.isfinite(box_inputs(arrays, configuration, record["normalization"])).all():
                    raise ValueError("Nonfinite standardized inputs")
        if reader is None and {name: checksum(native_input/name) for name in hashes} != hashes:
            raise ValueError("Native ROAD index changed during preparation")
        record["mask_audit"] = {"center_size_disagreements": 0, "four_flags_supported": True}
        record["status"] = "complete"
    except Exception as error:
        record["status"] = "failed"
        record["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        unchanged = all(checksum(Path(path)) == digest for path, digest in before.items())
        record["e001_unchanged"] = {"files_checked": len(before), "all_sha256_match": unchanged}
        record["finished_utc"], record["elapsed_seconds"] = datetime.now(timezone.utc).isoformat(), time.monotonic()-started
        if not unchanged:
            record["status"] = "failed"
            record["failures"].append("Frozen E001 files changed during preparation")
        save()
        if not unchanged:
            raise ValueError("Frozen E001 files changed during preparation")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("collection", "setup", "native-input", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--waymo-root", type=Path)
    parser.add_argument("--loki-transform", type=Path)
    args = parser.parse_args()
    result = prepare_boxes(args.collection, args.setup, args.native_input, args.output, args.waymo_root, args.loki_transform)
    print(json.dumps({key: result[key] for key in ("status", "dataset", "elapsed_seconds", "mask_audit", "e001_unchanged", "coverage")}, indent=2))
