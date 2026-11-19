"""Write E005 spatial features for frozen ROAD training and validation clips."""

import argparse
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import isolated_extraction
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.datasets import road_waymo
from pedestrian_behavior.inspection.feature_probe import relative_xyz
from pedestrian_behavior.inspection.pretrained import DINO_REVISION, ImageEncoder, UTONIA_REVISION

from .e002a_features import FeatureCatalog, backend_identity
from .e005_cache import FORMAT, TINY_REVISION, TINY_SHA256, E005Cache, E005Writer, TinyUtoniaEncoder, atomic_json
from .e005_transfer import DEFAULT_PROBE, DEFAULT_TINY_CHECKPOINT, DEFAULT_UTONIA_CODE, ROAD_POPULATION, RoadSpatialFeatures


SOURCE = "road-waymo"
MIN_FREE_BYTES = 20_000_000_000
TRACK_FIELDS = ("locator", "group", "archive", "split", "slots", "source_keys", "times_us", "sha256")


def bounded_groups(clips):
    training = [group for group, rows in clips.items() if rows[0]["split"] == "training"]
    validation = [group for group, rows in clips.items() if rows[0]["split"] == "validation"]

    def by_slots(groups):
        return sorted(groups, key=lambda group: (sum(row["slots"] for row in clips[group]), group))

    ranked = by_slots(training)
    largest = ranked[-1]
    populous = max((group for group in training if group != largest),
                   key=lambda group: (len(clips[group]), group))
    selected = {ranked[len(ranked) * q // 4] for q in (1, 2, 3)}
    return sorted(selected | {largest, populous, by_slots(validation)[len(validation) // 2]})


def _write_clip(writer, online, locators, rgb_boxes):
    """Keep each sweep union once and use increasing row indices per target."""
    locators_by_id = {online.tracks[locator]["id"]: locator for locator in locators}
    rgb_by_frame = defaultdict(list)
    for (key, track_id), box in rgb_boxes.items():
        rgb_by_frame[key].append((locators_by_id[track_id], box, online.rgb[key, track_id][:, :768]))
    for key, rows in rgb_by_frame.items():
        writer.add_rgb_batch(key, [(locator, box) for locator, box, _ in rows],
                             np.stack([patches for _, _, patches in rows]))

    targets_by_frame = defaultdict(list)
    for locator in locators:
        track = online.tracks[locator]
        keys = track["keys"]
        writer.summaries["rgb_candidates"] += sum(key is not None for key in keys)
        for key, usable in zip(keys, track["point_usable"]):
            if usable:
                targets_by_frame[key].append(locator)

    for key, (stages, world, _) in online.sweeps.items():
        targets = {locator: dict(zip(("box", "anchor"), online.lidar_target(locator, key)))
                   for locator in targets_by_frame[key]}
        united = []
        for level, (features, coords) in zip((3, 5), stages):
            masks = {locator: np.linalg.norm(relative_xyz(
                coords, np.asarray(target["box"][:3]), world, target["anchor"])[:, :2], axis=1) <= 20.
                for locator, target in targets.items()}
            union = np.flatnonzero(np.logical_or.reduce(list(masks.values())))
            united.append((features[union], coords[union]))
            row_map = np.empty(len(coords), dtype=np.int64)
            row_map[union] = np.arange(len(union))
            for locator, mask in masks.items():
                targets[locator][f"indices{level}"] = row_map[mask]
        writer.add_points(key, *united, targets, world)


def _capture_parity(writer, online, locators):
    """Sample a present and missing slot per modality from each clip when available."""
    checked = set()
    samples = {}
    for locator in locators:
        for slot, key in enumerate(online.tracks[locator]["keys"]):
            frame = writer.frames.get(str(key), {})
            target = frame.get("targets", {}).get(locator, {})
            rgb = locator in frame.get("rgb", {})
            lidar = any(target.get(f"stage{level}_indices", [0, 0])[1] for level in (3, 5))
            states = {("rgb", rgb), ("lidar", lidar)}
            if not states <= checked:
                samples[locator, slot] = (online.rgb_frame(locator, slot), *online.lidar_frame(locator, slot))
                checked.update(states)
            if len(checked) == 4:
                return samples
    return samples


def _verify_parity(cache, samples):
    counts = dict(rgb=0, lidar=0, rgb_missing=0, lidar_missing=0)
    for (locator, slot), expected in samples.items():
        rgb = cache.rgb_frame(locator, slot)
        lidar = cache.lidar_frame(locator, slot)
        for value, reference in zip((rgb, *lidar), expected):
            if value.dtype != reference.dtype or not np.array_equal(value, reference):
                raise ValueError(f"Cached and online ROAD features differ for {locator} slot {slot}")
        counts["rgb" if len(rgb) else "rgb_missing"] += 1
        counts["lidar" if any(map(len, lidar)) else "lidar_missing"] += 1
    if not counts["rgb"] or not counts["lidar"]:
        raise ValueError("ROAD cache parity requires actual RGB and LiDAR features")
    return counts


def _project_bytes(writer, arrays, catalog):
    full = {"rgb_candidates": sum(key is not None for row in catalog.tracks.values() for key in row["source_keys"]),
            "point_sweeps": len(catalog.frames), "point_targets": sum(map(len, catalog.frames.values()))}
    estimates = {}
    for name, record in arrays.items():
        unit = "rgb_candidates" if name.startswith("rgb") else "point_targets" if name.endswith("_indices") else "point_sweeps"
        estimates[name] = round(record["bytes"] * full[unit] / writer.summaries[unit])
    return {"expected_full_units": full, "projected_arrays_bytes": estimates,
            "projected_total_bytes": sum(estimates.values()),
            "qualification": "Linear projection from six bounded clips, including dense clips."}


def run(args):
    output, root, native_root = (path.resolve() for path in (args.output, args.outputs_root, args.waymo_root))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite E005 cache: {output}")
    if not native_root.is_dir() or output.is_relative_to(native_root):
        raise ValueError("Choose an existing Waymo root and a cache location outside the native data")
    probe = args.probe or root / DEFAULT_PROBE
    probe_path = probe / "probe.json"
    probe_record = json.loads(probe_path.read_text())
    if probe_record["status"] != "complete":
        raise ValueError("Completed pinned native probe required")
    native = dict(probe_record["config"], waymo_root=str(native_root))
    torch.set_num_threads(2)
    started = perf_counter()
    catalog = FeatureCatalog(root, SOURCE, native, probe / "loki-transform.json")
    if any(output.is_relative_to(path) for path in (catalog.collection, root / "experiments/E001/data-setup",
                                                    root / "experiments/E001b")):
        raise ValueError("Use a fresh location outside the frozen preparations")
    clips = defaultdict(list)
    for row in catalog.tracks.values():
        clips[row["group"]].append(row)
    if any(sum(len(rows) for rows in clips.values() if rows[0]["split"] == split) != expected["tracks"]
           or sum(rows[0]["split"] == split for rows in clips.values()) != expected["groups"]
           for split, expected in ROAD_POPULATION.items()):
        raise ValueError("Frozen ROAD training/validation population changed")
    groups = sorted(clips) if args.scope == "full" else bounded_groups(clips)
    records = [row for group in groups for row in clips[group]]

    code = args.utonia_code or root / DEFAULT_UTONIA_CODE
    tiny_path = args.tiny_checkpoint or root / DEFAULT_TINY_CHECKPOINT
    dino_path = native["dino_weights"]
    dino_sha = checksum(dino_path)
    required_free = MIN_FREE_BYTES
    if args.scope == "full":
        if args.estimate is None:
            raise ValueError("Full ROAD cache requires a matching verified bounded manifest")
        estimate = json.loads(args.estimate.read_text())
        if (estimate["status"] != "bounded-native-check" or not estimate.get("verified")
                or estimate["population"] != catalog.identity
                or estimate["provenance"]["DINO"]["checkpoint_sha256"] != dino_sha
                or estimate["provenance"]["Tiny-Utonia"]["checkpoint_sha256"] != TINY_SHA256):
            raise ValueError("Full ROAD cache sizing evidence does not match the frozen inputs")
        required_free += estimate["projection"]["projected_total_bytes"]

    output.mkdir(parents=True, exist_ok=False)
    progress = {"format": FORMAT, "status": "incomplete", "dataset": SOURCE, "scope": args.scope,
                "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
                "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "population": catalog.identity, "selected_groups": groups, "completed_scenes": 0,
                "split_tracks": {split: sum(row["split"] == split for row in records) for split in ROAD_POPULATION},
                "waymo_root": str(native_root), "minimum_free_bytes": MIN_FREE_BYTES}
    atomic_json(output / "cache.json", progress)
    manifest_path = output / "manifest.json"
    try:
        if shutil.disk_usage(output).free < required_free:
            raise OSError(f"ROAD cache needs {required_free} available bytes including its free-space floor")
        reader = road_waymo.TrackReader(native.get("road_index") or catalog.manifest["sources"]["index"], native_root)
        if reader.sources["sha256"] != catalog.manifest["sources"]["sha256"]:
            raise ValueError("Frozen ROAD index mismatch")
        protected = catalog.protected | {str(probe_path): checksum(probe_path),
                    str(dino_path): dino_sha, str(tiny_path): TINY_SHA256}
        samples = {}
        with isolated_extraction():
            dino = ImageEncoder(native["dino_code"], dino_path, args.device)
            tiny = TinyUtoniaEncoder(code, tiny_path, args.device)
        tracks = {row["locator"]: {"id": row["track_id"], "keys": row["source_keys"],
                                   "point_usable": row["point_usable"]} for row in records}
        online = RoadSpatialFeatures(tracks, reader, catalog.manifest["scenes"], dino, tiny)
        with closing(E005Writer(output, MIN_FREE_BYTES)) as writer:
            for index, group in enumerate(groups, 1):
                locators = [row["locator"] for row in clips[group]]
                before = catalog.source_identity(group)
                rgb_boxes = {}
                online.prepare(group, locators, rgb_boxes=rgb_boxes)
                writer.begin_scene(group)
                _write_clip(writer, online, locators, rgb_boxes)
                if args.scope == "bounded" or index in (1, len(groups)):
                    samples.update(_capture_parity(writer, online, locators))
                catalog.source_snapshots.pop(group)
                if catalog.source_identity(group) != before:
                    raise ValueError(f"Native ROAD clip changed during extraction: {group}")
                writer.finish_scene(before)
                progress.update(completed_scenes=index, last_scene=group, summary=dict(writer.summaries),
                                elapsed_seconds=perf_counter() - started)
                atomic_json(output / "cache.json", progress)
                print(f"ROAD E005: {index}/{len(groups)} clips, {group}, "
                      f"{sum(writer.byte_counts.values()) / 1e9:.2f} GB written", flush=True)
            arrays, payload_bytes = writer.finish()
            projection = _project_bytes(writer, arrays, catalog)

        if any(checksum(path) != digest for path, digest in protected.items()):
            raise ValueError("Protected E005 ROAD source or encoder asset changed")
        status = "complete" if args.scope == "full" else "bounded-native-check"
        tracks_path = output / "tracks.json"
        atomic_json(tracks_path, {"format": FORMAT, "population": catalog.identity,
                    "tracks": [{field: row[field] for field in TRACK_FIELDS}
                               for row in sorted(records, key=lambda item: item["locator"])]})
        metadata = progress | {"status": status, "tracks_sha256": checksum(tracks_path),
            "arrays": arrays, "scenes": writer.scenes, "payload_bytes": payload_bytes, "projection": projection,
            "provenance": {"DINO": {"revision": DINO_REVISION, "checkpoint_sha256": dino_sha},
                "Tiny-Utonia": {"code_revision": UTONIA_REVISION, "checkpoint_revision": TINY_REVISION,
                                "checkpoint_sha256": TINY_SHA256},
                "backend": backend_identity(args.device), "protected_files": len(protected)}}
        atomic_json(manifest_path, metadata)
        with E005Cache(output, allow_partial=True, verify_checksums=True) as cache:
            parity = _verify_parity(cache, samples)
        metadata.update(parity=parity, verified=True)
        atomic_json(manifest_path, metadata)
        progress.update(status=status, payload_bytes=payload_bytes, projection=projection, parity=parity,
                        manifest_sha256=checksum(manifest_path))
    except BaseException as error:
        manifest_path.unlink(missing_ok=True)
        progress.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        progress.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=perf_counter() - started)
        atomic_json(output / "cache.json", progress)
    print(f"ROAD E005 {progress['status']}: {output}", flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--utonia-code", type=Path)
    parser.add_argument("--tiny-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--scope", choices=("bounded", "full"), default="bounded")
    parser.add_argument("--estimate", type=Path, help="Verified bounded manifest for full builds")
    run(parser.parse_args())
