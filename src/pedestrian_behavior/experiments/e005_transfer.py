"""Evaluate a completed LOKI E005 checkpoint on frozen ROAD-Waymo validation."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import isolated_extraction
from pedestrian_behavior.data.preparation import checksum, load_track, valid_pose
from pedestrian_behavior.datasets import road_waymo
from pedestrian_behavior.evaluation import predict, prediction_metrics, save_predictions, stratified_metrics
from pedestrian_behavior.inspection.feature_probe import normalized_crop, relative_xyz, square_crop
from pedestrian_behavior.inspection.native_features import native_camera, native_scene
from pedestrian_behavior.inspection.pretrained import DINO_REVISION, ImageEncoder, UTONIA_REVISION
from pedestrian_behavior.models import e005
from pedestrian_behavior.training import seed_run

from . import e001b, e001b_run, e005_run
from .e002_transfer import _source_normalization
from .e002b_group import _decorate
from .e005_cache import (DINO_BATCH, TINY_REVISION, TINY_SHA256, E005FeatureDataset,
                         TinyUtoniaEncoder, collate_e005, lidar_tokens, rgb_crop_box, rgb_tokens)
from .run import write_json


SOURCE, TARGET, SPLIT = "loki", "road-waymo", "validation"
ARM = "g-rgb-utonia"
DEFAULT_PROBE = Path("experiments/E002a/native-probe-07")
DEFAULT_UTONIA_CODE = Path("experiments/E002a/dependencies/utonia")
DEFAULT_TINY_CHECKPOINT = Path("experiments/E005/online-probe-20261117/utonia_tiny.pth")
COMPONENTS = ("camera_image", "camera_box", "camera_calibration", "vehicle_pose",
              "lidar", "lidar_pose", "lidar_calibration", "lidar_box", "camera_to_lidar_box_association")
EMPTY_RGB = np.empty((0, 772), dtype=np.float16)
EMPTY_LIDAR = np.empty((0, 77), dtype=np.float16), np.empty((0, 221), dtype=np.float16)


def _load_source(run):
    status = json.loads((run / "status.json").read_text())
    config = json.loads((run / "config.json").read_text())
    if (status.get("status") != "complete" or status.get("kind") != "exploratory-source-validation"
            or config.get("experiment") != "E005" or config.get("source") != SOURCE
            or config.get("configuration") != ARM or config.get("variant") != e005_run.VARIANT
            or config.get("held_out_tests") is not False or config.get("smoke_limits") is not None):
        raise ValueError("A completed full LOKI E005 geometry + DINO + Tiny-Utonia source run is required")
    path = run / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    provenance = json.loads((run / "provenance.json").read_text())
    selected = checkpoint["selection"]
    if (json.loads(json.dumps(checkpoint["config"])) != config
            or checkpoint["epoch"] != selected["epoch"]
            or selected != provenance["selection"]
            or checkpoint["epoch"] != provenance["selected_checkpoint_epoch"]):
        raise ValueError("E005 best.pt differs from the recorded LOKI selection")
    return path, config, checkpoint, provenance


def _index_validation(baseline):
    tracks = {}
    for entry in baseline.entries:
        locator = entry["locator"]
        identity = json.loads(locator)
        metadata, arrays = load_track(baseline.collection / "tracks" / entry["archive"])
        keys = metadata["source_frames"]
        if (entry["split"] != SPLIT or identity["clip"] != entry["group"] or locator in tracks
                or metadata["track_locator"] != locator or len(keys) != entry["slots"]):
            raise ValueError("ROAD online track does not match frozen validation")
        point_usable = [
            key is not None and metadata["anchor_valid"] and arrays["ped_position_valid"][i]
            and arrays["ego_pose_valid"][i] and metadata["availability"][i].get("native_3d", False)
            for i, key in enumerate(keys)
        ]
        tracks[locator] = {"id": identity["tube_uid"], "keys": keys, "point_usable": point_usable}
    return tracks


class RoadSpatialFeatures:
    """Match E005's cache reader interface with one clip of in-memory features."""

    def __init__(self, tracks, reader, scenes, dino, tiny):
        self.tracks, self.reader, self.scenes, self.dino, self.tiny = tracks, reader, scenes, dino, tiny
        self.rgb, self.sweeps, self.anchors = {}, {}, {}
        self.rgb_crops, self.lidar_sweeps = 0, 0

    def prepare(self, group, locators):
        self.rgb.clear()
        self.sweeps.clear()
        self.anchors.clear()
        tracks = [self.tracks[locator] for locator in locators]
        ids = [track["id"] for track in tracks]
        rgb_targets = defaultdict(list)
        point_keys = set()
        for track in tracks:
            for key in set(track["keys"]) - {None}:
                rgb_targets[key].append(track["id"])
            point_keys.update(key for key, usable in zip(track["keys"], track["point_usable"]) if usable)
        scene = self.scenes[group]
        for component in COMPONENTS:
            path = road_waymo.component_path(scene, component, self.reader.waymo_root)
            if path is None or not path.is_file():
                raise ValueError(f"Missing ROAD component {component} for {group}: {path}")

        # The native point reader verifies the frozen box associations before using them.
        for frame in native_scene(self.reader, group, point_keys, ids, scene, include_rgb=False):
            key, world = frame["key"], frame["world_from_native"]
            if key in self.sweeps or key not in point_keys or not valid_pose(world):
                raise ValueError(f"Invalid ROAD point timestamp or pose for {group}/{key}")
            points = frame["points"]
            with isolated_extraction():
                stages = self.tiny.stages(points if points is not None else np.empty((0, 4), np.float32))
            if any(not np.isfinite(features).all() for features, _ in stages):
                raise ValueError("Nonfinite FP16 Tiny-Utonia features")
            self.sweeps[key] = stages, world, frame["boxes"]
            self.lidar_sweeps += 1
        if self.sweeps.keys() != point_keys:
            raise ValueError(f"Missing native ROAD point timestamps for {group}")
        first_keys = {track["keys"][0] for track in tracks if any(track["point_usable"])}
        if first_keys:
            records = self.reader.read_frames(group, first_keys)
            self.anchors.update({key: record["ego_pose"] for key, record in records.items()})
            if self.anchors.keys() != first_keys or not all(map(valid_pose, self.anchors.values())):
                raise ValueError(f"Invalid initial ROAD ego pose for {group}")

        crops, crop_refs = [], []

        def flush():
            if not crops:
                return
            with isolated_extraction():
                _, patches = self.dino.descriptors(torch.stack(crops), patches=True)
            for (key, track, box), patch in zip(crop_refs, patches.reshape(len(crops), 256, 768)):
                tokens = rgb_tokens(patch, box)
                if not np.isfinite(tokens).all():
                    raise ValueError("Nonfinite FP16 DINO features")
                self.rgb[key, track] = tokens
            self.rgb_crops += len(crops)
            crops.clear()
            crop_refs.clear()

        seen = set()
        for key, image, boxes in native_camera(self.reader, group, rgb_targets, ids, scene):
            if key in seen or key not in rgb_targets:
                raise ValueError(f"Unexpected native ROAD camera frame for {group}/{key}")
            seen.add(key)
            if image is None:
                continue
            for track in rgb_targets[key]:
                box = boxes.get(track)
                pixels, detail = square_crop(image, box, expansion=3.)
                if pixels is not None:
                    crops.append(normalized_crop(pixels))
                    crop_refs.append((key, track, rgb_crop_box(box, image.size, detail["bounds_pixels"])))
                    if len(crops) == DINO_BATCH:
                        flush()
        flush()
        if seen != rgb_targets.keys():
            raise ValueError(f"Missing native ROAD camera timestamps for {group}")

    def rgb_frame(self, locator, slot):
        track = self.tracks[locator]
        return self.rgb.get((track["keys"][slot], track["id"]), EMPTY_RGB)

    def lidar_frame(self, locator, slot):
        track = self.tracks[locator]
        key = track["keys"][slot]
        if not track["point_usable"][slot]:
            return EMPTY_LIDAR
        stages, world, boxes = self.sweeps[key]
        if track["id"] not in boxes:
            raise ValueError(f"Frozen usable ROAD observation lost its cuboid: {locator}/{key}")
        center, size, yaw = boxes[track["id"]]
        box = [*center, *size, yaw]
        anchor = self.anchors[track["keys"][0]]
        result = []
        for features, coords in stages:
            relative = relative_xyz(coords, center, world, anchor)
            selected = np.linalg.norm(relative[:, :2], axis=1) <= 20.
            result.append(lidar_tokens(features[selected], coords[selected], box, world, anchor))
        return tuple(result)


def _batches(dataset, features):
    groups = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        groups[entry["group"]].append(index)
    for step, (group, indices) in enumerate(sorted(groups.items()), 1):
        features.prepare(group, [dataset.entries[i]["locator"] for i in indices])
        for index in indices:
            yield collate_e005([dataset[index]])
        if step == 1 or step % 5 == 0 or step == len(groups):
            print(f"ROAD validation clips: {step}/{len(groups)}", flush=True)


def run(args):
    root, output = args.outputs_root.resolve(), args.output.resolve()
    source_run = args.source_run.resolve()
    output_root = root / "experiments/E005/transfer"
    if output == output_root or not output.is_relative_to(output_root) or output.is_relative_to(source_run):
        raise ValueError("Use a fresh output directory under outputs/experiments/E005/transfer")
    if output.exists():
        raise FileExistsError(output)
    path, config, checkpoint, source_provenance = _load_source(source_run)
    e001_root, e001b_root = root / "experiments/E001", root / "experiments/E001b"
    frozen = e001b_run.assemble(SOURCE, "geometry", "B", e001_root=e001_root, e001b_root=e001b_root)
    reference = frozen["datasets"][SOURCE]
    if any(source_provenance["datasets"][SOURCE][key] != reference[key]
           for key in ("setup_sha256", "collection_sha256")) or (
            source_provenance["datasets"][SOURCE]["bbox"]["manifest_sha256"]
            != reference["bbox"]["manifest_sha256"]):
        raise ValueError("Source run differs from the frozen LOKI preparation")
    stats = _source_normalization(checkpoint, frozen["normalization"]["baseline"], frozen["normalization"]["bbox"])
    target = frozen["datasets"][TARGET]
    baseline = e001b.dataset_from_extension(target["collection_path"], target["setup_path"],
        e001b_root / "bbox-native" / TARGET, SPLIT, "geometry", stats["baseline"], stats["bbox"])
    index = _index_validation(baseline)
    manifest = json.loads((Path(target["collection_path"]) / "manifest.json").read_text())
    probe = (args.probe or root / DEFAULT_PROBE) / "probe.json"
    probe_record = json.loads(probe.read_text())
    if probe_record["status"] != "complete":
        raise ValueError("Completed pinned native probe required")
    native = probe_record["config"]
    tiny_checkpoint = args.tiny_checkpoint or root / DEFAULT_TINY_CHECKPOINT
    dino_sha = checksum(native["dino_weights"])
    encoders = config["cache"]["provenance"]
    if ((encoders["DINO"]["revision"], encoders["DINO"]["checkpoint_sha256"]) != (DINO_REVISION, dino_sha)
            or (encoders["Tiny-Utonia"]["code_revision"], encoders["Tiny-Utonia"]["checkpoint_revision"],
                encoders["Tiny-Utonia"]["checkpoint_sha256"]) != (UTONIA_REVISION, TINY_REVISION, TINY_SHA256)):
        raise ValueError("Online encoder identities differ from the frozen E005 cache")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        torch.set_num_threads(2)
        determinism = seed_run(config["settings"]["seed"])
        waymo_root = args.waymo_root or native.get("waymo_root")
        reader = road_waymo.TrackReader(native.get("road_index") or manifest["sources"]["index"],
                                       Path(waymo_root) if waymo_root else None)
        if reader.sources["sha256"] != manifest["sources"]["sha256"]:
            raise ValueError("Frozen ROAD index mismatch")
        provenance = {"command": sys.orig_argv, "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), "device": str(device), "torch": str(torch.__version__),
            "cuda": torch.version.cuda, "determinism": determinism, "source": SOURCE, "target": TARGET,
            "split": SPLIT, "held_out_tests": False, "selection": checkpoint["selection"],
            "source_checkpoint": str(path), "checkpoint_sha256": checksum(path), "normalization_source": SOURCE,
            "normalization": stats, "datasets": {SOURCE: reference, TARGET: target}, "encoders": encoders,
            "waymo_root": str(reader.waymo_root), "probe_sha256": checksum(probe)}
        write_json(output / "provenance.json", provenance)
        with isolated_extraction():
            dino = ImageEncoder(native["dino_code"], native["dino_weights"], device)
            tiny = TinyUtoniaEncoder(args.utonia_code or root / DEFAULT_UTONIA_CODE, tiny_checkpoint, device)
        features = RoadSpatialFeatures(index, reader, manifest["scenes"], dino, tiny)
        dataset = E005FeatureDataset(baseline, features, ARM)
        model = e005.build_model(ARM, config["profile"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        tracks = predict(model, _batches(dataset, features), device, forward_batch=lambda model, batch, _: model(batch))
        if len(tracks) != len(index) or {track["locator"] for track in tracks} != index.keys():
            raise ValueError("Predictions do not cover the complete frozen ROAD validation split")

        def conditions(metadata, arrays, track, source):
            membership, flags = frozen["conditions"](metadata, arrays, track, source)
            membership.update({axis: track["source_flags"][flag].astype("int64")
                               for axis, flag in e005_run.OBSERVATION_FLAGS.items()})
            return membership, flags

        for track in tracks:
            track["group"] = json.loads(track["locator"])["clip"]
        _decorate(tracks, {"conditions": conditions}, {}, TARGET)
        cell = output / f"{TARGET}-{SPLIT}"
        cell.mkdir()
        save_predictions(cell / "predictions.npz", tracks)
        classes = len(e005_run.CLASSES)
        primary = prediction_metrics(tracks, classes)
        write_json(cell / "metrics.json", {"split": SPLIT, "training_source": SOURCE, "target_source": TARGET,
            "selected_epoch": checkpoint["epoch"], "checkpoint_sha256": provenance["checkpoint_sha256"],
            "primary": primary, "strata": stratified_metrics(tracks, classes, e005_run.axes(ARM)),
            "rgb_crops": features.rgb_crops, "lidar_sweeps": features.lidar_sweeps})
        status.update(status="complete")
        print(f"ROAD validation track-weighted macro-F1: {100 * primary['track_weighted']['macro_f1']:.2f}%", flush=True)
    except BaseException as error:
        status.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--waymo-root", type=Path)
    parser.add_argument("--utonia-code", type=Path)
    parser.add_argument("--tiny-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    run(parser.parse_args())
