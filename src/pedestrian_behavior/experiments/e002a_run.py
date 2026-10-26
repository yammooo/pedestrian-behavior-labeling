"""E002a source-validation pilot: frozen features and the agreed multimodal readout."""

import argparse
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import resource
import subprocess

import numpy as np
import torch

from pedestrian_behavior.accumulation import train_accumulated_epoch
from pedestrian_behavior.data.prepared_rgb import PreparedRGB
from pedestrian_behavior.data.prepared_points import PreparedPoints
from pedestrian_behavior.data.indexed_points import PROTOCOL as POINT_PROTOCOL
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.models.multimodal import ARM_FLAGS, MultimodalClassifier
from . import e001b_run
from .e001 import CLASSES
from .e002a_features import (FeatureCatalog, FeatureSupply, extraction_identity,
                             load_configuration, validate_roots)
from pedestrian_behavior.inspection.pretrained import DITR_REVISION, DITR_SHA256, DITR_CONFIG_SHA256
from .run import run_attempt as execute_attempt, write_json


ARMS = ("g", "g-rgb", "g-rgb-dditr")  # Keep the original six-run pilot unchanged.
SUPPORTED_ARMS = tuple(ARM_FLAGS)
RGB_ARMS = ("g-rgb", "g-rgb-dditr")
POINT_ARMS = ("g-dditr", "g-rgb-dditr")
DEFAULT_POINT_WORKERS = 4
FUSION_PROTOCOL = "enabled-branches-v2"
STRATA = e001b_run.STRATA | {"rgb-observed": ("missing", "valid"), "lidar-observed": ("missing", "valid")}
SETTINGS = {"seed": 0, "batch_size": 64, "epochs": 100, "patience": 15, "clip_norm": 1.,
            "precision": "float32", "warmup_epochs": 3,
            "optimizer": {"lr": .0003, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8}}
MODEL_SPEC = {"geometry_mlp": [20, 64, 64],
              "temporal": {"width": 256, "heads": 4, "layers": 2, "ffn": 1024, "attention_bias": True,
                           "rope_base": 10000, "slot_seconds": .2, "bidirectional": True},
              "classifier": [256, 4], "activation": "GELU", "dropout": .1, "layer_norm_eps": 1e-5,
              "trainable_precision": "float32", "attention_backend": "torch-SDPA-math"}


def model_spec(arm):
    flags = ARM_FLAGS[arm]
    spec = MODEL_SPEC | {"availability": list(flags), "fusion": [64 + 128 * (len(flags) - 1) + len(flags), 256]}
    if "rgb_valid" in flags:
        spec["rgb_projection"] = [768, 128]
    if "lidar_valid" in flags:
        spec.update(point_projection=[68, 128], spatial={"width": 128, "heads": 4, "seeds": 1,
                                                       "layers": 1, "ffn": 512, "attention_bias": False})
    return spec


def resource_code():
    from pedestrian_behavior.models import multimodal
    from pedestrian_behavior.data import multimodal as loading
    from pedestrian_behavior import accumulation, evaluation, training
    from pedestrian_behavior.data import indexed_points, prepared_points, prepared_rgb
    from . import run as shared_run
    return {m.__name__: checksum(m.__file__) for m in
            (multimodal, loading, accumulation, evaluation, training, shared_run,
             indexed_points, prepared_points, prepared_rgb)}


class PreparedSupply:
    def __init__(self, catalog, rgb, points, identity):
        self.catalogs = {catalog.dataset: catalog}
        self.points = points
        self.rgb_supply = FeatureSupply(self.catalogs, rgb, None, None, identity) if rgb is not None else None
        self.identity = {"rgb": identity if rgb is not None else None,
                         "points": points.identity if points is not None else None}

    def rgb(self, source, locator):
        return self.rgb_supply.rgb(source, locator)

    def track_cost(self, source, locator):
        return self.points.track_cost(locator)

    def point_tensors(self, source, locator):
        return self.points.iter_track(locator)


def load_point_configuration(probe):
    """Load pinned point assets without validating or depending on RGB weights."""
    probe = Path(probe).resolve()
    record = json.loads((probe / "probe.json").read_text())
    if record.get("status") != "complete":
        raise ValueError("Completed pinned native probe required")
    config = record["config"]
    if checksum(config["ditr_config"]) != DITR_CONFIG_SHA256 or checksum(config["ditr_weights"]) != DITR_SHA256:
        raise ValueError("Changed point encoder assets")
    return config, probe / "loki-transform.json"


def indexed_path(args):
    return args.indexed_point_cache if args.indexed_point_cache is not None else (
        args.outputs_root / "experiments/E002a" / ("indexed-" + args.source))


def validate_profile(profile):
    required = ("max_tracks", "temporal_cells", "microbatch_tokens", "ordinary_tokens", "frame_chunk", "frame_tokens")
    if any(type(profile.get(k)) is not int or profile[k] < 1 for k in required):
        raise ValueError("Positive calibrated packing/frame limits required")
    return {k: profile[k] for k in required}


class FeatureDataset:
    def __init__(self, baseline, supply, source, arm):
        self.baseline, self.supply, self.source, self.arm = baseline, supply, source, arm
        self.entries = baseline.entries if supply is None else [supply.catalogs[source].tracks[e["locator"]] for e in baseline.entries]
        if len(self.entries) != len(baseline):
            raise ValueError("Feature supply changed frozen population")

    def __len__(self):
        return len(self.entries)

    def cost(self, index):
        if self.arm not in POINT_ARMS:
            return 0
        entry = self.entries[index]
        return self.supply.track_cost(self.source, entry["locator"])

    def __getitem__(self, index):
        sample = self.baseline[index]
        locator = sample["locator"]
        n = len(sample["targets"])
        geometry = sample.pop("inputs")
        # E001b layout: eight numeric + four group flags, six bbox numeric + two flags.
        geometry_valid = geometry[:, [8, 9, 10, 11, 18, 19]].bool().any(-1)
        flags, source_flags = [geometry_valid], {}
        sample["geometry"] = geometry
        if self.arm in RGB_ARMS:
            rgb = self.supply.rgb(self.source, locator)
            if len(rgb["valid"]) != n:
                raise ValueError("RGB/geometry complete-track mismatch")
            sample["rgb"] = torch.from_numpy(rgb["cls"])
            flags.append(torch.from_numpy(rgb["valid"]))
            source_flags["rgb_valid"] = rgb["valid"]
        if self.arm in POINT_ARMS:
            frames = [torch.from_numpy(value) for value in self.supply.point_tensors(self.source, locator)]
            if len(frames) != n:
                raise ValueError("LiDAR/geometry complete-track mismatch")
            counts = np.array([len(f) for f in frames], np.int64)
            sample["points"] = frames
            flags.append(torch.from_numpy(counts > 0))
            source_flags.update(lidar_valid=counts > 0, point_tokens=counts)
        metadata = {"source_flags": source_flags}
        if self.arm != "g":
            metadata["frozen_feature_identity"] = self.supply.identity
        return sample | {"flags": torch.stack(flags, -1).float(), "prediction_metadata": metadata}


def assembly(args, *, policy=None):
    """Open only the source train/validation archives; own resources until attempt exit."""
    policy = policy or {}
    experiment = policy.get("experiment", "E002a")
    resource_identity = policy.get("resource_code", resource_code)()
    root_validator = policy.get("validate_roots", validate_roots)
    model_spec_for = policy.get("model_spec", lambda arm, variant: model_spec(arm))
    model_factory = policy.get("model_factory", lambda arm, variant, profile: MultimodalClassifier(arm, profile))
    model_variant = getattr(args, "model_variant", None)
    if args.arm in RGB_ARMS:
        config, transform = load_configuration(args.probe)
    elif args.arm in POINT_ARMS:
        config, transform = load_point_configuration(args.probe)
    else:
        config, transform = {}, None
    point_root = indexed_path(args)
    root_validator(argparse.Namespace(**(vars(args) | {"point_cache_root": point_root})), config)
    profile_record = json.loads(args.profile.read_text())
    if profile_record.get("status") != "complete" or profile_record.get("device") != args.device:
        raise ValueError("Completed calibration on the requested device required")
    if profile_record.get("code") != resource_identity:
        raise ValueError("Execution code changed since resource calibration")
    if profile_record.get("torch") != str(torch.__version__) or profile_record.get("cuda") != torch.version.cuda:
        raise ValueError("PyTorch/CUDA backend differs from resource calibration")
    profile = validate_profile(profile_record["profile"])
    if args.device == "cuda" and profile_record.get("gpu") != torch.cuda.get_device_name():
        raise ValueError("Resource profile was calibrated on different GPU hardware")
    stack = ExitStack()
    telemetry = log = None
    try:
        baseline = e001b_run.assemble(args.source, "geometry", "B", e001_root=args.outputs_root / "experiments/E001",
                                     e001b_root=args.outputs_root / "experiments/E001b")
        catalog = FeatureCatalog(args.outputs_root, args.source, config, transform) if args.arm != "g" else None
        rgb = points = supply = identity = None
        protected = {}
        if catalog is not None:
            if args.arm in RGB_ARMS:
                rgb = PreparedRGB(args.rgb_cache_root)
                stack.callback(rgb.close)
            if args.arm in POINT_ARMS:
                points = PreparedPoints(point_root, args.source, catalog.identity, catalog.tracks,
                    {"revision": DITR_REVISION, "checkpoint": DITR_SHA256, "configuration": DITR_CONFIG_SHA256},
                    allow_partial=args.smoke, frame_workers=getattr(args, "point_workers", DEFAULT_POINT_WORKERS))
                stack.callback(points.close)
                if profile_record.get("point_storage_protocol") != POINT_PROTOCOL:
                    raise ValueError("Resource calibration for indexed point storage required")
            identity = extraction_identity(args.device, config) if args.arm in RGB_ARMS else None
            supply = PreparedSupply(catalog, rgb, points, identity)
            protected = catalog.protected
            if points is not None:
                protected.update(points.protected)
            if rgb is not None:
                protected[str(args.rgb_preparation.resolve())] = checksum(args.rgb_preparation)
            assets = [args.probe / "probe.json", transform]
            if rgb is not None:
                assets.append(Path(config["dino_weights"]))
            if points is not None:
                assets += [Path(config[k]) for k in ("ditr_weights", "ditr_config")]
            for path in assets:
                protected[str(path.resolve())] = checksum(path)
        if catalog is not None and args.arm in RGB_ARMS:
            preparation = json.loads(args.rgb_preparation.read_text())
            expected = {k for k in catalog.tracks}
            actual = {t["locator"] for t in preparation.get("tracks", []) if t["dataset"] == args.source}
            if (preparation.get("status") != "complete" or preparation.get("extraction") != identity
                    or expected != actual or Path(preparation["config"]["rgb_cache_root"]).resolve() != args.rgb_cache_root.resolve()):
                raise ValueError("Matching full source train/validation RGB cache preparation required")
        datasets = {split: baseline["dataset"](args.source, split) for split in ("training", "validation")}
        full_entries = {e["locator"]: e for d in datasets.values() for e in d.entries}
        references = baseline["datasets"][args.source]
        collection = Path(references["collection_path"])
        for path in (Path(references["setup_path"]), collection / "manifest.json"):
            protected[str(path.resolve())] = checksum(path)
        for entry in full_entries.values():
            path = collection / "tracks" / entry["archive"]
            protected[str(path.resolve())] = checksum(path)
        bbox_manifest = Path(baseline["datasets"][args.source]["bbox"]["manifest_path"])
        protected[str(bbox_manifest.resolve())] = checksum(bbox_manifest)
        for entry in json.loads(bbox_manifest.read_text())["tracks"]:
            if entry["locator"] in full_entries:
                protected[str((bbox_manifest.parent / "tracks" / entry["archive"]).resolve())] = entry["bbox_sha256"]
        selected = None
        if args.smoke:
            selected = {t["locator"] for t in json.loads(args.smoke_selection.read_text())["tracks"] if t["dataset"] == args.source}
            if not selected or len(selected) > 10 or not selected.issubset(full_entries):
                raise ValueError("Smoke selection outside frozen source train/validation")
        for split in ("training", "validation"):
            d = datasets[split]
            if selected is not None:
                d.entries = [e for e in d.entries if e["locator"] in selected]
                if not d.entries:
                    raise ValueError("Smoke needs explicit source training and validation tracks")
            for entry in d.entries:
                if entry["slots"] ** 2 > profile["temporal_cells"]:
                    raise ValueError("Complete track exceeds calibrated temporal ceiling")
                if points is not None:
                    points.require_track(entry["locator"])
                    if points.counts[entry["locator"]].max(initial=0) > profile["frame_tokens"]:
                        raise ValueError("Prepared frame exceeds calibrated token ceiling; recalibrate")
            datasets[split] = FeatureDataset(d, supply, args.source, args.arm)
        full_training_count = sum(t["split"] == "training" for t in full_entries.values())
        warmup = 3 * ((full_training_count + 63) // 64)
        if args.device == "cuda":
            log = (args.output / "system.csv").open("w")
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
                                          "--format=csv", "-l", "1"], stdout=log, stderr=subprocess.STDOUT)

        def snapshot():
            caches = {k: v for k, v in (("rgb", rgb), ("points", points)) if v is not None}
            return {**{k: v.snapshot() for k, v in caches.items()}, "encoder_loads": {},
                    "initial": {k: v.initial for k, v in caches.items()},
                    "encoder_calls": {}, "encoder_seconds": {},
                    "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated() if args.device == "cuda" else None,
                    "cuda_reserved_bytes": torch.cuda.memory_reserved() if args.device == "cuda" else None}

        def close():
            try:
                stack.close()
            finally:
                if telemetry is not None:
                    telemetry.terminate(); telemetry.wait(timeout=10)
                    log.close()
                changed = [p for p, h in protected.items() if checksum(p) != h]
                write_json(args.output / "protected.json", {"hashes": protected, "changed": changed})
                if changed:
                    raise ValueError("Frozen source files changed")

        def batches(d, training=False, seed=0):
            def packing(row):
                with (args.output / "packing.jsonl").open("a") as stream:
                    stream.write(json.dumps(row | {"phase": "training" if training else "validation"}, allow_nan=False) + "\n")
            logical = LogicalBatches(d, profile, training, seed, max_updates=2 if args.smoke and training else None,
                                     on_pack=packing, prefetch=args.arm in POINT_ARMS)
            return logical if training else PredictionBatches(logical)

        def conditions(metadata, arrays, track, dataset):
            c, flags = baseline["conditions"](metadata, arrays, track, dataset)
            observed = {axis: track["source_flags"][flag].astype(np.int64)
                        for axis, flag in (("rgb-observed", "rgb_valid"), ("lidar-observed", "lidar_valid"))
                        if flag in track["source_flags"]}
            return c | observed, flags

        return {"config": {"experiment": experiment, "arm": args.arm, "fusion_protocol": FUSION_PROTOCOL,
                    "model": model_spec_for(args.arm, model_variant), "geometry": "E001b-native-geometry",
                    "frozen_features": supply.identity if identity is None and supply is not None else identity,
                    "profile": profile, "profile_sha256": checksum(args.profile),
                    "indexed_point_cache": points.identity if points is not None else None,
                    "point_storage_protocol": POINT_PROTOCOL if points is not None else None,
                    "point_loading": {"prefetch_physical_microbatches": 1, "open_scenes": points.limit,
                        "frame_workers": points.frame_workers,
                        "metadata_limit_bytes": points.metadata_limit,
                        "verification": "source-process-file-stamps-chunk-flags"} if points is not None else None,
                    "feature_supply": "prepared-read-only-no-inference",
                    "rgb_cache_root": str(args.rgb_cache_root.resolve()) if rgb is not None else None, "warmup_updates": warmup,
                    "held_out_tests": False, "exploratory": True},
                "normalization": baseline["normalization"], "datasets": {args.source: baseline["datasets"][args.source]},
                "build_model": lambda: model_factory(args.arm, model_variant, profile),
                "dataset": lambda source, split: datasets[split] if source == args.source else None,
                "batches": batches, "forward_batch": lambda m, b, device: m(b),
                "epoch_operation": partial(train_accumulated_epoch, peak_lr=.0003, warmup_steps=warmup),
                "evaluation_splits": [(args.source, "validation-smoke" if args.smoke else "validation")],
                "conditions": conditions, "kind": "integration-smoke" if args.smoke else "exploratory-source-validation",
                "custom_smoke": True, "smoke_limits": {"selected_tracks": sorted(selected), "max_training_updates": 2,
                    "held_out_tests": False} if selected is not None else None,
                "run_name": policy.get("run_name", f"e002a-{args.source}-{args.arm}-s0" +
                    ("-smoke-" + args.output.name if args.smoke else "")),
                "log_learning_rate": True, "runtime_snapshot": snapshot, "close": close}
    except BaseException:
        stack.close()
        if telemetry is not None:
            telemetry.terminate(); telemetry.wait(timeout=10); log.close()
        raise


def add_arguments(parser):
    parser.add_argument("--source", choices=("loki", "road-waymo"), required=True)
    parser.add_argument("--arm", choices=SUPPORTED_ARMS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path, help="Required for full RGB-enabled attempts: matching RGB preparation JSON")
    parser.add_argument("--indexed-point-cache", type=Path,
                        help="Completed source cache; default outputs-root/experiments/E002a/indexed-<source>")
    parser.add_argument("--point-workers", type=int, default=DEFAULT_POINT_WORKERS, help="Bounded ordered CPU frame-preparation workers")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path, help="Explicit bounded track selection; required for smoke")


def run(args):
    if args.smoke and args.smoke_selection is None:
        raise ValueError("Smoke requires an explicit bounded source train/validation selection")
    if args.arm in RGB_ARMS and args.rgb_preparation is None:
        raise ValueError("RGB-enabled attempt requires --rgb-preparation with completed matching RGB preparation")
    torch.set_num_threads(2)
    return execute_attempt(args.source, args.arm, "Transformer", args.output, args.device, args.smoke,
                           settings=SETTINGS, assemble=lambda *_: assembly(args), classes=CLASSES,
                           axes={k: v for k, v in STRATA.items() if k not in ("rgb-observed", "lidar-observed")
                                 or (k == "rgb-observed" and args.arm in RGB_ARMS)
                                 or (k == "lidar-observed" and args.arm in POINT_ARMS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())
