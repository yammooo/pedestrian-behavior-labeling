"""Evaluate a completed LOKI E005 checkpoint and extract frozen ROAD tracks."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import hashlib
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
from pedestrian_behavior.evaluation import (load_predictions, predict, prediction_metrics,
                                            save_predictions, stratified_metrics)
from pedestrian_behavior.inspection.feature_probe import normalized_crop, relative_xyz, square_crop
from pedestrian_behavior.inspection.native_features import native_camera, native_scene
from pedestrian_behavior.inspection.pretrained import DINO_REVISION, ImageEncoder, UTONIA_REVISION
from pedestrian_behavior.models import e005
from pedestrian_behavior.training import seed_run

from . import e001b, e001b_run, e003b_run, e005_run
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
SOURCE_CHECKPOINT_SHA256 = "7d630cedb0e2e80f3132be3af43270160c3c3144be15da13fdc7d152e69f3d6b"
ROAD_VALIDATION_SHA256 = "263d8d3d7bc1767f0e1b4ec4ff284ab92c43a42c110730b0a7f41181f4104ef4"
ROAD_PREPARATION_SHA256 = (
    "68d30161dcb104f2e96c06b02889118fc57ecff950b5c04a1978e2561423f2e1",
    "4225fc21eedc1ec7e4805639f5a2328daa5829536ba4492aae850f83f5d62b66",
    "688308d0d5d736bbe85ab8027fddc0aed030f5a543c16aa396c8f2605ad09abb",
)
ROAD_POPULATION = {
    "training": {"groups": 358, "tracks": 4380, "context_slots": 332940, "gt_timesteps": 167280},
    "validation": {"groups": 78, "tracks": 1110, "context_slots": 87824, "gt_timesteps": 45388},
}
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


def _index_road_split(baseline, split):
    tracks = {}
    for entry in baseline.entries:
        locator = entry["locator"]
        identity = json.loads(locator)
        metadata, arrays = load_track(baseline.collection / "tracks" / entry["archive"])
        keys = metadata["source_frames"]
        if (entry["split"] != split or identity["clip"] != entry["group"] or locator in tracks
                or metadata["track_locator"] != locator or len(keys) != entry["slots"]):
            raise ValueError(f"ROAD online track does not match frozen {split}")
        point_usable = [
            key is not None and metadata["anchor_valid"] and arrays["ped_position_valid"][i]
            and arrays["ego_pose_valid"][i] and metadata["availability"][i].get("native_3d", False)
            for i, key in enumerate(keys)
        ]
        tracks[locator] = {"id": identity["tube_uid"], "keys": keys, "point_usable": point_usable}
    return tracks


def _index_validation(baseline):
    return _index_road_split(baseline, SPLIT)


class RoadSpatialFeatures:
    """Match E005's cache reader interface with one clip of in-memory features."""

    def __init__(self, tracks, reader, scenes, dino, tiny, *, prefetch=True):
        self.tracks, self.reader, self.scenes, self.dino, self.tiny = tracks, reader, scenes, dino, tiny
        self.rgb, self.sweeps, self.anchors = {}, {}, {}
        self.rgb_crops, self.lidar_sweeps = 0, 0
        self.prefetch = prefetch

    def prepare(self, group, locators, *, rgb_boxes=None):
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
        frames = native_scene(self.reader, group, point_keys, ids, scene, include_rgb=False)
        with closing(_one_ahead(frames) if self.prefetch else frames) as decoded:
            for frame in decoded:
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
                if rgb_boxes is not None:
                    rgb_boxes[key, track] = box
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

    def lidar_target(self, locator, key):
        track = self.tracks[locator]
        boxes = self.sweeps[key][2]
        if track["id"] not in boxes:
            raise ValueError(f"Frozen usable ROAD observation lost its cuboid: {locator}/{key}")
        center, size, yaw = boxes[track["id"]]
        return [*center, *size, float(yaw)], self.anchors[track["keys"][0]]

    def lidar_frame(self, locator, slot):
        track = self.tracks[locator]
        key = track["keys"][slot]
        if not track["point_usable"][slot]:
            return EMPTY_LIDAR
        stages, world, _ = self.sweeps[key]
        box, anchor = self.lidar_target(locator, key)
        result = []
        for features, coords in stages:
            relative = relative_xyz(coords, np.asarray(box[:3]), world, anchor)
            selected = np.linalg.norm(relative[:, :2], axis=1) <= 20.
            result.append(lidar_tokens(features[selected], coords[selected], box, world, anchor))
        return tuple(result)


def _one_ahead(iterator):
    """Overlap CPU-only preparation with the current GPU work and retain one pending item."""
    with closing(iterator), ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(next, iterator, None)
        try:
            while (item := pending.result()) is not None:
                pending = pool.submit(next, iterator, None)
                yield item
        finally:
            pending.cancel()


def _batches(dataset, features, *, prefetch=True, split=SPLIT):
    groups = defaultdict(list)
    for index, entry in enumerate(dataset.entries):
        groups[entry["group"]].append(index)
    for step, (group, indices) in enumerate(sorted(groups.items()), 1):
        features.prepare(group, [dataset.entries[i]["locator"] for i in indices])
        batches = (collate_e005([dataset[index]]) for index in indices)
        with closing(_one_ahead(batches) if prefetch else batches) as prepared:
            yield from prepared
        if step == 1 or step % 5 == 0 or step == len(groups):
            print(f"ROAD {split} clips: {step}/{len(groups)}", flush=True)


def _transfer_inputs(args):
    root = args.outputs_root.resolve()
    source_run = args.source_run.resolve()
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
    return {"source_checkpoint": path, "config": config, "checkpoint": checkpoint, "frozen": frozen,
            "normalization": stats, "target": target, "manifest": manifest, "probe": probe, "native": native,
            "encoders": encoders, "device": device, "bbox": e001b_root / "bbox-native" / TARGET,
            "utonia_code": args.utonia_code or root / DEFAULT_UTONIA_CODE, "tiny_checkpoint": tiny_checkpoint}


def _baseline(inputs, split):
    target, stats = inputs["target"], inputs["normalization"]
    return e001b.dataset_from_extension(target["collection_path"], target["setup_path"],
        inputs["bbox"], split, "geometry", stats["baseline"], stats["bbox"])


def _transfer_runtime(args, inputs):
    config, native, manifest = (inputs[key] for key in ("config", "native", "manifest"))
    device = inputs["device"]
    torch.set_num_threads(2)
    determinism = seed_run(config["settings"]["seed"])
    waymo_root = args.waymo_root or native.get("waymo_root")
    reader = road_waymo.TrackReader(native.get("road_index") or manifest["sources"]["index"],
                                   Path(waymo_root) if waymo_root else None)
    if reader.sources["sha256"] != manifest["sources"]["sha256"]:
        raise ValueError("Frozen ROAD index mismatch")
    profile = (e005_run.AALTO_FAST_PROFILE
               if device.type == "cuda" and torch.cuda.get_device_name(device) == e005_run.AALTO_GPU
               else config["profile"])
    path = inputs["source_checkpoint"]
    provenance = {"command": sys.orig_argv, "revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(), "device": str(device), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "determinism": determinism, "source": SOURCE, "target": TARGET,
        "held_out_tests": False, "selection": inputs["checkpoint"]["selection"],
        "source_checkpoint": str(path), "checkpoint_sha256": checksum(path), "normalization_source": SOURCE,
        "normalization": inputs["normalization"], "datasets": inputs["frozen"]["datasets"], "encoders": inputs["encoders"],
        "waymo_root": str(reader.waymo_root), "probe_sha256": checksum(inputs["probe"]),
        "execution": {"readout_profile": profile, "readout_mode": "ordinary",
                      "cpu_lookahead_sweeps": 1, "cpu_lookahead_tracks": 1}}
    return reader, profile, provenance


def _load_online_models(inputs, profile):
    native, device = inputs["native"], inputs["device"]
    with isolated_extraction():
        dino = ImageEncoder(native["dino_code"], native["dino_weights"], device)
        tiny = TinyUtoniaEncoder(inputs["utonia_code"], inputs["tiny_checkpoint"], device)
    model = e005.build_model(ARM, profile).to(device)
    model.load_state_dict(inputs["checkpoint"]["model_state"], strict=True)
    return model, dino, tiny


def run(args):
    root, output = args.outputs_root.resolve(), args.output.resolve()
    source_run = args.source_run.resolve()
    output_root = root / "experiments/E005/transfer"
    if output == output_root or not output.is_relative_to(output_root) or output.is_relative_to(source_run):
        raise ValueError("Use a fresh output directory under outputs/experiments/E005/transfer")
    if output.exists():
        raise FileExistsError(output)
    inputs = _transfer_inputs(args)
    baseline = _baseline(inputs, SPLIT)
    index = _index_validation(baseline)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        reader, profile, provenance = _transfer_runtime(args, inputs)
        write_json(output / "provenance.json", provenance | {"split": SPLIT})
        model, dino, tiny = _load_online_models(inputs, profile)
        features = RoadSpatialFeatures(index, reader, inputs["manifest"]["scenes"], dino, tiny)
        dataset = E005FeatureDataset(baseline, features, ARM)
        tracks = predict(model, _batches(dataset, features), inputs["device"],
                         forward_batch=lambda model, batch, _: model(batch, mode="ordinary"))
        if len(tracks) != len(index) or {track["locator"] for track in tracks} != index.keys():
            raise ValueError("Predictions do not cover the complete frozen ROAD validation split")

        def conditions(metadata, arrays, track, source):
            membership, flags = inputs["frozen"]["conditions"](metadata, arrays, track, source)
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
            "selected_epoch": inputs["checkpoint"]["epoch"], "checkpoint_sha256": provenance["checkpoint_sha256"],
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


def _extract_embedding_records(model, batches, device):
    captured = []

    def forward(model, batch, _):
        logits = model(batch, mode="ordinary")
        features, = captured
        captured.clear()
        if (features.shape != (*logits.shape[:2], 128) or features.dtype != torch.float32
                or not torch.isfinite(features).all()):
            raise ValueError("Expected finite float32 post-LayerNorm E005 features")
        features = features.cpu().numpy()
        for i, length in enumerate(batch["lengths"].tolist()):
            batch["prediction_metadata"][i]["features"] = features[i, :length].copy()
        return logits

    with model.final_norm.register_forward_hook(lambda _module, _inputs, value: captured.append(value)):
        records = predict(model, batches, device, forward_batch=forward)
    for row in records:
        row["source_logits"] = row.pop("logits")
        row["group"] = json.loads(row["locator"])["clip"]
    return records


def _select_smoke_clips(entries, split):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["group"]].append(entry)
    minimum = 13 if split == "training" else 1
    candidates = sorted((sum(row["slots"] for row in rows), group)
                        for group, rows in groups.items() if len(rows) >= minimum)
    if len(candidates) < 2:
        raise ValueError(f"Need two eligible clips for ROAD {split} smoke extraction")
    selected = {group for _, group in candidates[:2]}
    return [row for row in entries if row["group"] in selected]


def _embedding_population(records):
    return {"tracks": len(records), "groups": len({row["group"] for row in records}),
            "context_slots": sum(len(row["targets"]) for row in records),
            "gt_timesteps": sum(int(row["gt_valid"].sum()) for row in records)}


def _check_embedding_predictions(records, reference, *, smoke):
    expected = {row["locator"]: row for row in reference}
    locators = {row["locator"] for row in records}
    if len(locators) != len(records) or not locators <= expected.keys() or (not smoke and locators != expected.keys()):
        raise ValueError("Extracted ROAD validation locators differ from the saved predictions")
    max_delta = 0.
    for row in records:
        prior = expected[row["locator"]]
        logits = row["source_logits"]
        if (row["group"] != prior["group"] or Path(row["archive"]).name != Path(prior["archive"]).name
                or not np.array_equal(row["targets"], prior["targets"])
                or not np.array_equal(row["gt_valid"], prior["gt_valid"])
                or any(not np.array_equal(value, prior["source_flags"][name])
                       for name, value in row["source_flags"].items())
                or logits.shape != prior["logits"].shape
                or not np.allclose(logits, prior["logits"], rtol=1e-4, atol=1e-5)):
            raise ValueError(f"Frozen ROAD validation replay differs for {row['locator']}")
        max_delta = max(max_delta, float(np.abs(logits - prior["logits"]).max()))
    return [row for row in reference if row["locator"] in locators], max_delta


def _head_digest(head):
    return hashlib.sha256(head["weight"].tobytes() + head["bias"].tobytes()).hexdigest()


def _write_embedding_cache(args, root, smoke):
    reference_path = args.reference_predictions.resolve()
    if checksum(reference_path) != ROAD_VALIDATION_SHA256:
        raise ValueError("Expected the pinned E005 direct ROAD validation predictions")
    inputs = _transfer_inputs(args)
    target = inputs["target"]
    if (target["setup_sha256"], target["collection_sha256"], target["bbox"]["manifest_sha256"]) != ROAD_PREPARATION_SHA256:
        raise ValueError("Frozen ROAD preparation differs from the original E005 validation")
    reader, profile, provenance = _transfer_runtime(args, inputs)
    if provenance["checkpoint_sha256"] != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("Expected the original epoch-14 LOKI E005 checkpoint")
    write_json(root / "provenance.json", provenance | {"splits": list(ROAD_POPULATION), "smoke": smoke,
        "reference_predictions": str(reference_path), "reference_predictions_sha256": ROAD_VALIDATION_SHA256})
    model, dino, tiny = _load_online_models(inputs, profile)
    model.requires_grad_(False)
    head = {key: value.detach().cpu().numpy() for key, value in model.head.state_dict().items()}
    head_path = root / "source-head.npz"
    np.savez_compressed(head_path, **head)
    entries = {}
    predictions_path = root / "source-validation-predictions.npz"

    for split in ("validation", "training"):
        baseline = _baseline(inputs, split)
        if smoke:
            baseline.entries = _select_smoke_clips(baseline.entries, split)
        index = _index_road_split(baseline, split)
        features = RoadSpatialFeatures(index, reader, inputs["manifest"]["scenes"], dino, tiny)
        dataset = E005FeatureDataset(baseline, features, ARM)
        records = _extract_embedding_records(model, _batches(dataset, features, split=split), inputs["device"])
        counts = _embedding_population(records)
        if (len(records) != len(index) or {row["locator"] for row in records} != index.keys()
                or (not smoke and counts != ROAD_POPULATION[split])):
            raise ValueError(f"Embedding extraction changed the frozen ROAD {split} population")
        if split == "validation":
            anchor, max_delta = _check_embedding_predictions(records, load_predictions(reference_path), smoke=smoke)
            save_predictions(predictions_path, anchor)
            print(f"ROAD validation replay checked; maximum absolute logit difference {max_delta:.3g}", flush=True)
        identity = {"stage": "post-bilstm-final-layernorm", "source": SOURCE, "target": TARGET, "split": split,
                    "checkpoint_sha256": SOURCE_CHECKPOINT_SHA256, "smoke": smoke}
        path = root / f"features-{split}.npz"
        e003b_run._save_cache(path, records, identity)
        entries[split] = {"path": path.name, "sha256": checksum(path), "identity": identity, **counts,
                          "rgb_crops": features.rgb_crops, "lidar_sweeps": features.lidar_sweeps}
        del records, dataset, features

    manifest = {"status": "complete", "experiment": "E005", "architecture": e005_run.VARIANT,
        "source": SOURCE, "target": TARGET, "configuration": ARM, "held_out_tests": False, "smoke": smoke,
        "source_checkpoint": provenance["source_checkpoint"], "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "normalization_source": SOURCE, "normalization": inputs["normalization"],
        "derived_features": entries, "population": {split: entry["tracks"] for split, entry in entries.items()},
        "source_head": {"path": head_path.name, "sha256": checksum(head_path), "state_sha256": _head_digest(head)},
        "source_validation_predictions": {"path": predictions_path.name, "sha256": checksum(predictions_path),
            "reference_sha256": ROAD_VALIDATION_SHA256, "max_absolute_logit_difference": max_delta,
            "relative_tolerance": 1e-4, "absolute_tolerance": 1e-5}}
    write_json(root / "cache.json", manifest)


def _read_embedding_cache(root, smoke):
    manifest = json.loads((root / "cache.json").read_text())
    if (manifest.get("status") != "complete" or manifest.get("experiment") != "E005"
            or manifest.get("configuration") != ARM or manifest.get("architecture") != e005_run.VARIANT
            or manifest.get("smoke") is not smoke or manifest.get("held_out_tests") is not False
            or manifest.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_SHA256
            or manifest["source_validation_predictions"]["reference_sha256"] != ROAD_VALIDATION_SHA256):
        raise ValueError("ROAD embedding cache identity or scope mismatch")

    def verified_file(name, entry):
        path = root / name
        if entry["path"] != name or checksum(path) != entry["sha256"]:
            raise ValueError(f"ROAD embedding cache checksum mismatch for {name}")
        return path

    records = {}
    for split, full_counts in ROAD_POPULATION.items():
        entry = manifest["derived_features"][split]
        path = verified_file(f"features-{split}.npz", entry)
        records[split] = e003b_run._load_cache(path, entry["identity"])
        counts = _embedding_population(records[split])
        if (any(entry[key] != value for key, value in counts.items())
                or (not smoke and counts != full_counts)):
            raise ValueError(f"ROAD embedding cache population mismatch for {split}")
    head_entry = manifest["source_head"]
    with np.load(verified_file("source-head.npz", head_entry), allow_pickle=False) as archive:
        head = {key: archive[key] for key in ("weight", "bias")}
    if (head["weight"].shape != (4, 128) or head["bias"].shape != (4,)
            or any(value.dtype != np.float32 or not np.isfinite(value).all() for value in head.values())
            or _head_digest(head) != head_entry["state_sha256"]):
        raise ValueError("ROAD embedding cache source head must be the frozen float32 Linear(128, 4)")
    source_validation = load_predictions(verified_file("source-validation-predictions.npz",
                                                       manifest["source_validation_predictions"]))
    return {"root": root, "manifest": manifest, "records": records, "head": head,
            "source_validation": source_validation}


def load_or_extract_embeddings(args, cache_root):
    """Read a complete ROAD cache or publish a new cache after validation replay."""
    root = Path(cache_root).resolve()
    smoke = bool(getattr(args, "smoke", False))
    status_path = root / "status.json"
    if root.exists():
        if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "complete":
            raise ValueError(f"Incomplete ROAD embedding cache at {root}; move it aside before rerunning")
        return _read_embedding_cache(root, smoke)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat(),
              "held_out_tests": False, "smoke": smoke}
    write_json(status_path, status)
    try:
        _write_embedding_cache(args, root, smoke)
        result = _read_embedding_cache(root, smoke)
        status["status"] = "complete"
        return result
    except BaseException as error:
        status.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        status.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(status_path, status)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a completed LOKI E005 checkpoint on ROAD validation.")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--waymo-root", type=Path)
    parser.add_argument("--utonia-code", type=Path)
    parser.add_argument("--tiny-checkpoint", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    run(parser.parse_args())
