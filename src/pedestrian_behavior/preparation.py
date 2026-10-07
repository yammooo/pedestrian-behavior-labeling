"""Complete-track 5 Hz preparation; native supervision stays native."""

import argparse
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

PERIOD_US = 200_000
TOLERANCE_US = 75_000
ARRAYS = {"ped_position": (2,), "ped_velocity": (2,), "ego_position": (2,),
          "ego_velocity": (2,), "ego_yaw": ()}
MASKS = ("ped_position_valid", "ped_velocity_valid", "ego_pose_valid", "ego_velocity_valid")


def locator(**fields):
    return json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def native_float(value):
    return float(value) if value not in (None, "", "None") else float("nan")


def valid_pose(value):
    if value is None:
        return False
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("Pose must be a 4x4 transform")
    return (np.isfinite(pose).all() and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-8)
            and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            and np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-6)
            and np.linalg.norm(pose[:2, 0]) > 1e-8)


def world_position(center, pose):
    """Unused box dimensions/heading cannot invalidate an otherwise usable center."""
    if center is None or not valid_pose(pose):
        return None
    xyz = np.asarray(center, dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError("Pedestrian center must have three coordinates")
    if not np.isfinite(xyz).all():
        return None
    return pose[:3, :3] @ xyz + pose[:3, 3]


def select_frames(frames, start, end):
    times = [f["time_us"] for f in frames]
    if any(not isinstance(t, int) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Scene times must be unique, increasing integer microseconds")
    keys = [f["key"] for f in frames]
    if len(set(keys)) != len(keys) or not times or start > end:
        raise ValueError("Invalid scene catalogue or track extent")
    selected = []
    for tick in range(start, end + 1, PERIOD_US):
        at = bisect_left(times, tick)
        choices = [i for i in (at - 1, at) if 0 <= i < len(times) and start <= times[i] <= end]
        best = min(choices, key=lambda i: (abs(times[i] - tick), times[i])) if choices else None
        selected.append(frames[best] if best is not None and abs(times[best] - tick) <= TOLERANCE_US else None)
    return selected


def velocities(positions, valid, times):
    result = np.full_like(positions, np.nan)
    mask = np.zeros(len(valid), dtype=bool)
    observed_times = [t for t in times if t is not None]
    if any(b <= a for a, b in zip(observed_times, observed_times[1:])):
        raise ValueError("Selected source times must be strictly increasing")
    for i in np.flatnonzero(valid):
        terms = []
        if i > 0 and valid[i - 1]:
            h = (times[i] - times[i - 1]) / 1e6
            terms.append(((positions[i] - positions[i - 1]) / h, h))
        if i + 1 < len(valid) and valid[i + 1]:
            h = (times[i + 1] - times[i]) / 1e6
            terms.append(((positions[i + 1] - positions[i]) / h, h))
        if len(terms) == 2:
            (left, hm), (right, hp) = terms
            result[i] = (hp * left + hm * right) / (hm + hp)
        elif terms:
            result[i] = terms[0][0]
        mask[i] = bool(terms)
    return result, mask


def prepare_track(track_id, candidate, selected, records):
    n = len(selected)
    arrays = {name: np.full((n, *shape), np.nan, dtype=np.float64) for name, shape in ARRAYS.items()}
    arrays.update({name: np.zeros(n, dtype=bool) for name in MASKS})
    times = [f["time_us"] if f else None for f in selected]
    first = records[selected[0]["key"]] if selected[0] else {}
    anchor = first.get("ego_pose")
    anchored = valid_pose(anchor)
    if anchored:
        yaw0 = np.arctan2(anchor[1, 0], anchor[0, 0])
        c, s = np.cos(yaw0), np.sin(yaw0)
        rotation = np.array([[c, s], [-s, c]])
        origin = anchor[:2, 3]
    annotations, availability, issues, native_metadata = [], [], [], []
    for i, frame in enumerate(selected):
        record = records[frame["key"]] if frame else {}
        person = record.get("pedestrians", {}).get(track_id, {})
        annotations.append(person.get("annotations", {}))
        native_metadata.append({k: person[k] for k in ("native_type", "export_flags") if k in person})
        availability.append(person.get("availability", {}) | {"rgb": frame.get("rgb") if frame else None})
        slot_issues = list(person.get("issues", []))
        if not frame:
            slot_issues.append("no-qualifying-scene-frame")
        if not anchored:
            slot_issues.append("invalid-initial-anchor")
        pose = record.get("ego_pose")
        if anchored and valid_pose(pose):
            arrays["ego_position"][i] = rotation @ (pose[:2, 3] - origin)
            arrays["ego_yaw"][i] = np.arctan2(pose[1, 0], pose[0, 0]) - yaw0
            arrays["ego_pose_valid"][i] = True
        elif frame:
            slot_issues.append("unusable-ego-pose")
        xyz = person.get("world_position")
        if xyz is not None and np.asarray(xyz).shape != (3,):
            raise ValueError("World pedestrian position must have three coordinates")
        if anchored and xyz is not None and np.isfinite(xyz).all():
            arrays["ped_position"][i] = rotation @ (np.asarray(xyz)[:2] - origin)
            arrays["ped_position_valid"][i] = True
        elif frame:
            slot_issues.append("unusable-pedestrian-position")
        issues.append(slot_issues)
    for prefix, mask in (("ped", "ped_position_valid"), ("ego", "ego_pose_valid")):
        arrays[prefix + "_velocity"], arrays[prefix + "_velocity_valid"] = velocities(
            arrays[prefix + "_position"], arrays[mask], times)
    metadata = {"track_locator": candidate["locator"], "native_extent_us": candidate["extent_us"],
                "source_frames": [f["key"] if f else None for f in selected],
                "native_annotations": annotations, "availability": availability, "issues": issues,
                "native_metadata": native_metadata,
                "anchor_valid": bool(anchored), "native_counts": candidate["counts"]}
    # Waymo's integer frame key IS the timestamp; do not store it twice.
    if any(f and f["key"] != f["time_us"] for f in selected):
        metadata["source_time_us"] = times
    return metadata, arrays


def source_times(metadata):
    return metadata.get("source_time_us", metadata["source_frames"])


def validate_track(metadata, arrays):
    n = (metadata["native_extent_us"][1] - metadata["native_extent_us"][0]) // PERIOD_US + 1
    for name, shape in ARRAYS.items():
        if arrays[name].shape != (n, *shape) or arrays[name].dtype != np.float64:
            raise ValueError(f"Invalid {name} shape/dtype")
    for name in MASKS:
        if arrays[name].shape != (n,) or arrays[name].dtype != bool:
            raise ValueError(f"Invalid {name} mask")
    for field, mask in (("ped_position", "ped_position_valid"), ("ped_velocity", "ped_velocity_valid"),
                        ("ego_position", "ego_pose_valid"), ("ego_yaw", "ego_pose_valid"),
                        ("ego_velocity", "ego_velocity_valid")):
        values, valid = arrays[field], arrays[mask]
        if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
            raise ValueError(f"Invalid {field} missingness")
    for field in ("source_frames", "native_annotations", "availability", "native_metadata", "issues"):
        if len(metadata[field]) != n:
            raise ValueError(f"Invalid {field} length")
    times = source_times(metadata)
    if len(times) != n or any((a is None) != (b is None) for a, b in zip(times, metadata["source_frames"])):
        raise ValueError("Source keys and times disagree")
    observed = [t for t in times if t is not None]
    if any(not isinstance(t, int) for t in observed) or any(b <= a for a, b in zip(observed, observed[1:])):
        raise ValueError("Invalid selected source times")
    start, end = metadata["native_extent_us"]
    if any(t is not None and (not start <= t <= end or abs(t-(start+i*PERIOD_US)) > TOLERANCE_US)
           for i, t in enumerate(times)):
        raise ValueError("Selected source frame violates temporal bounds")
    missing = np.array([k is None for k in metadata["source_frames"]])
    if any(arrays[k][missing].any() for k in MASKS):
        raise ValueError("Missing source frame has usable input")
    for prefix, position_mask in (("ped", "ped_position_valid"), ("ego", "ego_pose_valid")):
        if (arrays[prefix+"_velocity_valid"] & ~arrays[position_mask]).any():
            raise ValueError("Usable velocity requires a usable position")
    locator_fields = json.loads(metadata["track_locator"])
    if locator(**locator_fields) != metadata["track_locator"]:
        raise ValueError("Noncanonical track locator")


def save_track(path, metadata, arrays):
    validate_track(metadata, arrays)
    np.savez_compressed(path, metadata=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)), **arrays)


def load_track(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        arrays = {name: saved[name] for name in (*ARRAYS, *MASKS)}
    validate_track(metadata, arrays)
    return metadata, arrays


def prepare_collection(reader, output, scenes=None):
    """Write a fresh collection; keep partial/failing runs explicitly incomplete."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "tracks").mkdir()
    manifest = {"schema_version": 1, "dataset": reader.dataset, "sources": reader.sources,
                "clock": reader.clock, "preparation": {"period_us": PERIOD_US, "tolerance_us": TOLERANCE_US,
                "coordinate_frame": "initial-ego-heading, x forward/y left", "units": {"position": "metres", "velocity": "metres/second", "yaw": "radians"},
                "python": platform.python_version(), "numpy": np.__version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "diff_sha256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest()},
                "status": "incomplete", "tracks": [], "scenes": {}}
    manifest["preparation"]["code_sha256"] = {str(p): checksum(p) for p in sorted(Path("src").rglob("*.py"))}
    effective = {k: v for k, v in manifest["preparation"].items() if k != "created_utc"}
    manifest["preparation_id"] = hashlib.sha256(json.dumps({k: manifest[k] for k in
        ("schema_version", "dataset", "sources", "clock")} | {"preparation": effective}, sort_keys=True).encode()).hexdigest()
    audit = {"tracks": [], "failures": [], "native_candidates": 0, "saved_tracks": 0}
    def write_progress():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        (output / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    write_progress()
    try:
        for scene in scenes if scenes is not None else reader.scene_ids:
            index = reader.index_scene(scene)
            manifest["scenes"][scene] = index.get("provenance", {})
            audit["native_candidates"] += len(index["tracks"])
            selections = {key: select_frames(index["frames"], *c["extent_us"]) for key, c in index["tracks"].items()}
            requested = {f["key"] for selected in selections.values() for f in selected if f}
            records = reader.read_frames(scene, requested)
            if set(records) != requested:
                raise ValueError(f"{scene}: reader did not return exactly requested frames")
            for track_id, candidate in sorted(index["tracks"].items()):
                metadata, arrays = prepare_track(track_id, candidate, selections[track_id], records)
                filename = hashlib.sha256(metadata["track_locator"].encode()).hexdigest() + ".npz"
                save_track(output / "tracks" / filename, metadata, arrays)
                reloaded, actual = load_track(output / "tracks" / filename)
                if reloaded != metadata or any(not np.array_equal(actual[k], v, equal_nan=True) for k, v in arrays.items()):
                    raise ValueError("Persistence changed prepared data")
                manifest["tracks"].append(filename)
                audit["tracks"].append({"locator": metadata["track_locator"], "native_extent_us": candidate["extent_us"],
                    "native_counts": candidate["counts"], "selected_slots": len(selections[track_id]),
                    "selected_frames": sum(f is not None for f in selections[track_id]),
                    "selected_annotations": sum(bool(x) for x in metadata["native_annotations"]),
                    "validity_counts": {k: int(v.sum()) for k, v in arrays.items() if k in MASKS},
                    "selected_availability": {key: sum(a.get(key) is True for a in metadata["availability"])
                                              for key in sorted({k for a in metadata["availability"] for k in a})},
                    "anchor_valid": metadata["anchor_valid"]})
            audit["saved_tracks"] = len(manifest["tracks"])
            write_progress()
            print(f"{scene}: {len(index['tracks'])} candidates; {len(manifest['tracks'])} saved", flush=True)
        if audit["native_candidates"] != audit["saved_tracks"]:
            raise ValueError("Native inventory and persisted candidate counts disagree")
        manifest["status"] = "complete"
    except Exception as error:
        audit["failures"].append({"scene": scene, "error": str(error)})
        raise
    finally:
        write_progress()
    return manifest, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("loki", "road-waymo"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waymo-root", type=Path)
    parser.add_argument("--loki-transform", type=Path, help="Evidence-backed transform contract; no default calibration")
    parser.add_argument("--scene", action="append")
    args = parser.parse_args()
    if args.dataset == "loki":
        from pedestrian_behavior.datasets.loki import TrackReader
        reader = TrackReader(args.input, args.loki_transform)
    else:
        from pedestrian_behavior.datasets.road_waymo import TrackReader
        reader = TrackReader(args.input, args.waymo_root)
    prepare_collection(reader, args.output, args.scene)


if __name__ == "__main__":
    main()
