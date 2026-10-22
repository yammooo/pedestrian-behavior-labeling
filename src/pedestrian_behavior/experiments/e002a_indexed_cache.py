"""Prepare a source's immutable indexed D-DITR cache directly from native sweeps."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
from pedestrian_behavior.data.feature_cache import gb_bytes, identity_digest
from pedestrian_behavior.data.frozen_features import point_union
from pedestrian_behavior.data.indexed_points import IndexedShard, PROTOCOL, ShardWriter, build_frame
from pedestrian_behavior.data.preparation import checksum, valid_pose
from pedestrian_behavior.inspection.native_features import native_scene
from pedestrian_behavior.inspection.pretrained import DITR_CONFIG_SHA256, DITR_SHA256, DITR_REVISION
from .e002a_features import FeatureCatalog, FrozenEncoders, backend_identity
from .e002a_probe import write_json


FORMAT = "e002a-indexed-source-v1"


def publish_json(path, record):
    """Publish durable completion evidence only after the referenced files are durable."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def check_root(args, config):
    root = args.output.resolve()
    base = args.outputs_root.resolve() / "experiments"
    if not root.is_relative_to(base / "E002a") or root == base / "E002a":
        raise ValueError("Choose a fresh directory below outputs/experiments/E002a")
    protected = [base / name for name in ("E001", "E001b", "E001c")]
    protected += [base / "E002a" / name for name in ("weights", "extractor-env", "rgb-cache", "point-cache")]
    protected += [args.probe.resolve()]
    protected += [Path(config[k]).resolve() for k in ("loki_root", "waymo_root", "road_index", "ditr_code") if config.get(k)]
    if any(root.is_relative_to(p) or p.is_relative_to(root) for p in protected):
        raise ValueError("Output overlaps protected inputs or another feature cache")
    gb_bytes(args.min_free_gb)
    for limit in (args.max_scenes, args.max_frames_per_scene):
        if limit is not None and limit <= 0:
            raise ValueError("Bounded-check limits must be positive")


def point_configuration(probe):
    record = json.loads((probe / "probe.json").read_text())
    if record["status"] != "complete":
        raise ValueError("Completed pinned native probe required")
    config = dict(record["config"])
    if checksum(config["ditr_config"]) != DITR_CONFIG_SHA256 or checksum(config["ditr_weights"]) != DITR_SHA256:
        raise ValueError("Changed D-DITR assets")
    return config, probe / "loki-transform.json"


def source_stamp(catalog, group):
    catalog.source_snapshots.pop(group, None)
    return catalog.source_identity(group)


def scene_anchors(catalog, group, keys):
    participants = sorted({k for key in keys for k in catalog.frames[group, key]})
    first = {catalog.tracks[k]["source_keys"][0] for k in participants}
    if None in first:
        raise ValueError("Usable point track has no initial native anchor")
    reader = catalog.native_reader(group)
    records = reader.read_frames(group, first) if first else {}
    anchors = {k: records[catalog.tracks[k]["source_keys"][0]]["ego_pose"] for k in participants}
    if any(not valid_pose(pose) for pose in anchors.values()):
        raise ValueError("Invalid initial native anchor")
    return reader, participants, anchors


def verify_scene(root, sha, expected):
    """Fresh-reader every-selection readback plus complete binary-file checksum checks."""
    reader = IndexedShard(root, sha)
    try:
        for name, spec in reader.record["arrays"].items():
            if checksum(root / (name + ".bin")) != spec["sha256"]:
                raise ValueError("Indexed scene file checksum mismatch")
        requests, tokens = 0, 0
        for key, meta in reader.record["frames"].items():
            if sorted(s["locator"] for s in meta["selections"]) != sorted(expected[key]):
                raise ValueError("Indexed frame participants differ from frozen catalog")
            for selection in meta["selections"]:
                crop = reader.crop(key, selection["locator"])
                if crop.shape != (selection["count"], 68):
                    raise ValueError("Indexed crop shape mismatch")
                requests += 1
                tokens += len(crop)
        if set(reader.record["frames"]) != set(expected):
            raise ValueError("Indexed source timestamps missing or duplicated")
        return {"requests": requests, "tokens": tokens, "files_sha256_verified": True}
    finally:
        reader.close()


def prepare_scene(catalog, group, keys, encoders, output, min_free_bytes, progress):
    """Single ordered native traversal and one encoder forward per nonempty requested sweep."""
    before = source_stamp(catalog, group)
    reader, participants, anchors = scene_anchors(catalog, group, keys)
    writer = ShardWriter(output, {"dataset": catalog.dataset, "group": group,
        "population": catalog.identity, "native_source_stat_sha256": before}, min_free_bytes=min_free_bytes)
    seen, counts = set(), {"original_tokens": 0, "stored_tokens": 0, "target_requests": 0}
    last_expected = None
    try:
        raw_keys = {str(key): key for k in participants for key in catalog.tracks[k]["source_keys"]
                    if key is not None and str(key) in keys}
        native = native_scene(reader, group, [raw_keys[k] for k in keys], [catalog.tracks[k]["track_id"] for k in participants],
                              catalog.manifest["scenes"][group], include_rgb=False)
        for frame in native:
            key = str(frame["key"])
            if key not in keys or key in seen:
                raise ValueError("Unexpected/duplicate native sweep")
            seen.add(key)
            boxes = {}
            for locator in catalog.frames[group, key]:
                box = frame["boxes"].get(catalog.tracks[locator]["track_id"])
                if box is None:
                    raise ValueError("Frozen usable observation lost native cuboid")
                boxes[locator] = box
            points = frame["points"]
            if points is None or not len(points):
                points = np.zeros((0, 4), np.float32)
                features, selected = np.zeros((0, 64), np.float32), np.zeros(0, np.int64)
            else:
                features, _, selected, _ = encoders("points", points)
            arrays = point_union(points, features, selected, boxes, frame["world_from_native"], anchors)
            # Keep only the last frame's reference for an independent serialized-value check.
            if key == keys[-1]:
                payload, meta = build_frame(arrays)
                last_expected = (key, payload, meta)
            added = writer.add(key, arrays)
            counts["original_tokens"] += added["original_tokens"]
            counts["stored_tokens"] += added["count"]
            counts["target_requests"] += len(added["selections"])
            progress(key, added)
        if seen != set(keys):
            raise ValueError("Missing native sweep")
        sha = writer.finish()
    finally:
        writer.close()
    if source_stamp(catalog, group) != before:
        raise ValueError("Native source changed during scene preparation")
    expected = {key: catalog.frames[group, key] for key in keys}
    verification = verify_scene(output, sha, expected)
    if last_expected is not None:
        key, payload, meta = last_expected
        fresh = IndexedShard(output, sha)
        try:
            for selection in meta["selections"]:
                start, count = selection["index_start"], selection["count"]
                rows = payload["indices"][start:start+count]
                actual = fresh.crop(key, selection["locator"])
                np.testing.assert_array_equal(actual[:, :64], payload["features"][rows].astype(np.float32))
        finally:
            fresh.close()
    return {"manifest_sha256": sha, "native_source_stat_sha256": before, "frames": len(keys),
        "bytes": sum(p.stat().st_size for p in output.iterdir() if p.is_file()),
        **counts, "verification": verification}


def track_records(catalog, groups):
    records = []
    for locator, track in sorted(catalog.tracks.items()):
        if track["group"] not in groups:
            continue
        keys = groups[track["group"]]
        frames = [str(key) if usable and str(key) in keys else None
                  for key, usable in zip(track["source_keys"], track["point_usable"])]
        records.append({k: track[k] for k in ("locator", "group", "archive", "split", "slots", "sha256")} | {
            "source_keys": track["source_keys"], "times_us": track["times_us"], "point_frames": frames,
            "point_usable": track["point_usable"].tolist()})
    return records


def run(args):
    check_root(args, {})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    bounded = args.max_scenes is not None or args.max_frames_per_scene is not None
    record = {"format": FORMAT, "protocol": PROTOCOL, "status": "incomplete", "dataset": args.dataset,
        "scope": "bounded-native-check" if bounded else "full-frozen-source-train-validation",
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff": subprocess.check_output(["git", "diff", "HEAD"], text=True), "host": platform.node(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scenes": {}, "completed_frames": 0, "bytes": 0, "failures": []}
    write_json(output / "cache.json", record)
    encoders, protected = None, {}
    try:
        config, transform = point_configuration(args.probe)
        check_root(args, config)
        floor = gb_bytes(args.min_free_gb)
        if shutil.disk_usage(output).free <= floor:
            raise OSError("Insufficient free space before extraction")
        catalog = FeatureCatalog(args.outputs_root, args.dataset, config, transform)
        protected.update(catalog.protected)
        for path in (args.probe / "probe.json", transform, Path(config["ditr_weights"]), Path(config["ditr_config"])):
            protected[str(path.resolve())] = checksum(path)
        from pedestrian_behavior.data import indexed_points, frozen_features
        from pedestrian_behavior.inspection import pretrained, native_features, feature_probe
        record.update(population=catalog.identity, extraction={"backend": backend_identity(args.device),
            "implementation": {m.__name__: checksum(m.__file__) for m in
                (indexed_points, frozen_features, pretrained, native_features, feature_probe)},
            "builder_sha256": checksum(__file__)},
            encoder={"revision": DITR_REVISION, "checkpoint": DITR_SHA256, "configuration": DITR_CONFIG_SHA256,
                "preprocessing": "full-sweep;rotated-xyz-strength;5cm-first-native;fixed-serialization;float32"})
        groups = defaultdict(set)
        for t in catalog.tracks.values():
            groups[t["group"]]  # Preserve tracks with no usable LiDAR observations.
        for group, key in catalog.frames:
            groups[group].add(key)
        groups = {g: set(sorted(groups[g], key=int)[:args.max_frames_per_scene])
                  for g in sorted(groups)[:args.max_scenes]}
        record["expected_scenes"] = len(groups)
        record["expected_frames"] = sum(map(len, groups.values()))
        record["tracks"] = len(track_records(catalog, groups))
        write_json(output / "cache.json", record)
        encoders = FrozenEncoders(config, args.device)
        for group, keys in groups.items():
            shard = "scenes/" + identity_digest({"dataset": args.dataset, "group": group})
            def progress(key, meta):
                record["completed_frames"] += 1
                record["elapsed_seconds"] = time.perf_counter() - begin
                write_json(output / "cache.json", record)
                if record["completed_frames"] == 1 or record["completed_frames"] % 25 == 0:
                    print(f'{args.dataset}: {record["completed_frames"]}/{record["expected_frames"]} sweeps; '
                          f'{record["elapsed_seconds"]/60:.1f} min; '
                          f'{shutil.disk_usage(output).free/1e9:.1f} GB free', flush=True)
            result = prepare_scene(catalog, group, sorted(keys, key=int), encoders, output / shard, floor, progress)
            record["scenes"][group] = {"path": shard, **result}
            record["bytes"] += result["bytes"]
            write_json(output / "cache.json", record)
            print(f'{args.dataset}: scene {len(record["scenes"])}/{len(groups)} verified; '
                  f'{record["bytes"]/1e9:.3f} GB written', flush=True)
        if any(checksum(p) != h for p, h in protected.items()):
            raise ValueError("Protected frozen input changed")
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        record["failures"].append(traceback.format_exc())
        raise
    finally:
        if encoders is not None:
            encoders.release()
            record["encoders"] = {"calls": encoders.calls, "loads": encoders.loads, "seconds": encoders.seconds}
        changed = [p for p, h in protected.items() if not Path(p).is_file() or checksum(p) != h]
        if changed:
            record["status"] = "failed"
        record.update(changed_files=changed, protected_files=len(protected), elapsed_seconds=time.perf_counter()-begin,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(output / "protected.json", protected)
        write_json(output / "cache.json", record)
        if changed:
            raise ValueError("Protected frozen input changed")
    manifest = {**record, "tracks": track_records(catalog, groups)}
    try:
        publish_json(output / "manifest.json", manifest)
    except BaseException:
        record["status"] = "failed"
        record["failures"].append(traceback.format_exc())
        write_json(output / "cache.json", record)
        raise
    print(f'Complete {record["scope"]}: {output}/manifest.json', flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--dataset", choices=("loki", "road-waymo"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--min-free-gb", type=float, default=20)
    parser.add_argument("--max-scenes", type=int, help="Bounded acceptance only; manifest is explicitly partial")
    parser.add_argument("--max-frames-per-scene", type=int, help="Bounded acceptance only; manifest is explicitly partial")
    parser.add_argument("--output", type=Path, required=True, help="Fresh source-specific cache directory")
    run(parser.parse_args())
