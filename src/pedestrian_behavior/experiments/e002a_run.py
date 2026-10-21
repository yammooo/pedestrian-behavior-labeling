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
from pedestrian_behavior.data.feature_cache import FeatureCache, gb_bytes
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.models.multimodal import MultimodalClassifier
from . import e001b_run
from .e001 import CLASSES
from .e002a_features import (FeatureCatalog, FeatureSupply, FrozenEncoders, extraction_identity,
                             load_configuration, validate_roots)
from .run import run_attempt as execute_attempt, write_json


ARMS = ("g", "g-rgb", "g-rgb-dditr")
STRATA = e001b_run.STRATA | {"rgb-observed": ("missing", "valid"), "lidar-observed": ("missing", "valid")}
SETTINGS = {"seed": 0, "batch_size": 64, "epochs": 100, "patience": 15, "clip_norm": 1.,
            "precision": "float32", "warmup_epochs": 3,
            "optimizer": {"lr": .0003, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8}}
MODEL_SPEC = {"geometry_mlp": [20, 64, 64], "rgb_projection": [768, 128], "point_projection": [68, 128],
              "spatial": {"width": 128, "heads": 4, "seeds": 1, "layers": 1, "ffn": 512, "attention_bias": False},
              "availability": ["geometry_valid", "rgb_valid", "lidar_valid"], "fusion": [323, 256],
              "temporal": {"width": 256, "heads": 4, "layers": 2, "ffn": 1024, "attention_bias": True,
                           "rope_base": 10000, "slot_seconds": .2, "bidirectional": True},
              "classifier": [256, 4], "activation": "GELU", "dropout": .1, "layer_norm_eps": 1e-5,
              "trainable_precision": "float32", "attention_backend": "torch-SDPA-math"}


def resource_code():
    from pedestrian_behavior.models import multimodal
    from pedestrian_behavior.data import multimodal as loading
    from pedestrian_behavior import accumulation
    return {m.__name__: checksum(m.__file__) for m in (multimodal, loading, accumulation)}


class TrainingEncoders(FrozenEncoders):
    def __init__(self, config, device, require_rgb_cache):
        super().__init__(config, device)
        self.require_rgb_cache = require_rgb_cache

    def __call__(self, kind, inputs):
        if self.require_rgb_cache and kind == "rgb":
            raise ValueError("RGB cache miss: complete the matching full RGB preparation before the pilot")
        return super().__call__(kind, inputs)


def validate_profile(profile):
    required = ("max_tracks", "temporal_cells", "microbatch_tokens", "ordinary_tokens", "frame_chunk", "frame_tokens")
    if any(type(profile.get(k)) is not int or profile[k] < 1 for k in required):
        raise ValueError("Positive calibrated packing/frame limits required")
    return {k: profile[k] for k in required}


class FeatureDataset:
    def __init__(self, baseline, supply, source, arm):
        self.baseline, self.supply, self.source, self.arm = baseline, supply, source, arm
        catalog = supply.catalogs[source]
        self.entries = [catalog.tracks[e["locator"]] for e in baseline.entries]
        if len(self.entries) != len(baseline):
            raise ValueError("Feature supply changed frozen population")

    def __len__(self):
        return len(self.entries)

    def cost(self, index):
        if self.arm != "g-rgb-dditr":
            return 0
        entry = self.entries[index]
        values = [self.supply.point_cost(self.source, entry["locator"], i) for i in range(entry["slots"])]
        return None if any(v is None for v in values) else sum(v["tokens"] for v in values)

    def __getitem__(self, index):
        sample = self.baseline[index]
        locator = sample["locator"]
        rgb = self.supply.rgb(self.source, locator)
        n = len(sample["targets"])
        if len(rgb["valid"]) != n:
            raise ValueError("RGB/geometry complete-track mismatch")
        counts = []
        frames = []
        known = [self.supply.point_cost(self.source, locator, i) for i in range(n)]
        if self.arm == "g-rgb-dditr" or any(v is None for v in known):
            for crop in self.supply.iter_points(self.source, locator):
                counts.append(len(crop["features"]))
                if self.arm == "g-rgb-dditr":
                    frames.append(torch.from_numpy(np.c_[crop["features"], crop["relative_xyz"] / 20,
                                                          crop["inside_box"]].astype(np.float32)))
        else:
            counts = [v["tokens"] for v in known]
        if not frames:
            frames = [torch.zeros(0, 68) for _ in range(n)]
        geometry = sample.pop("inputs")
        # E001b layout: eight numeric + four group flags, six bbox numeric + two flags.
        geometry_valid = geometry[:, [8, 9, 10, 11, 18, 19]].bool().any(-1)
        flags = torch.stack((geometry_valid, torch.from_numpy(rgb["valid"]), torch.tensor(counts) > 0), -1).float()
        return sample | {"geometry": geometry, "rgb": torch.from_numpy(rgb["cls"]), "flags": flags, "points": frames,
            "prediction_metadata": {"source_flags": {"rgb_valid": rgb["valid"], "lidar_valid": np.array(counts) > 0,
                                                     "point_tokens": np.array(counts, np.int64)},
                "frozen_feature_identity": self.supply.identity}}


def assembly(args):
    """Open only the source train/validation archives; own resources until attempt exit."""
    config, transform = load_configuration(args.probe)
    validate_roots(args, config)
    profile_record = json.loads(args.profile.read_text())
    if profile_record.get("status") != "complete" or profile_record.get("device") != args.device:
        raise ValueError("Completed calibration on the requested device required")
    if profile_record.get("code") != resource_code():
        raise ValueError("Execution code changed since resource calibration")
    if profile_record.get("torch") != str(torch.__version__) or profile_record.get("cuda") != torch.version.cuda:
        raise ValueError("PyTorch/CUDA backend differs from resource calibration")
    profile = validate_profile(profile_record["profile"])
    if args.device == "cuda" and profile_record.get("gpu") != torch.cuda.get_device_name():
        raise ValueError("Resource profile was calibrated on different GPU hardware")
    stack = ExitStack()
    telemetry = log = None
    try:
        catalog = FeatureCatalog(args.outputs_root, args.source, config, transform)
        rgb = stack.enter_context(FeatureCache(args.rgb_cache_root, min_free_bytes=gb_bytes(args.min_free_gb)))
        points = stack.enter_context(FeatureCache(args.point_cache_root, capacity_bytes=gb_bytes(args.point_cache_gb),
                                                 min_free_bytes=gb_bytes(args.min_free_gb)))
        encoders = TrainingEncoders(config, args.device, require_rgb_cache=not args.smoke)
        stack.callback(encoders.release)
        identity = extraction_identity(args.device, config)
        if not args.smoke:
            preparation = json.loads(args.rgb_preparation.read_text())
            expected = {k for k in catalog.tracks}
            actual = {t["locator"] for t in preparation.get("tracks", []) if t["dataset"] == args.source}
            if (preparation.get("status") != "complete" or preparation.get("extraction") != identity
                    or expected != actual or Path(preparation["config"]["rgb_cache_root"]).resolve() != args.rgb_cache_root.resolve()):
                raise ValueError("Matching full source train/validation RGB cache preparation required")
        supply = FeatureSupply({args.source: catalog}, rgb, points, encoders, identity)
        baseline = e001b_run.assemble(args.source, "geometry", "B", e001_root=args.outputs_root / "experiments/E001",
                                     e001b_root=args.outputs_root / "experiments/E001b")
        assets = [args.probe / "probe.json", transform] + [Path(config[k]) for k in ("dino_weights", "ditr_weights", "ditr_config")]
        bbox_manifest = Path(baseline["datasets"][args.source]["bbox"]["manifest_path"])
        for path in assets + [bbox_manifest]:
            catalog.protected[str(path.resolve())] = checksum(path)
        for entry in json.loads(bbox_manifest.read_text())["tracks"]:
            if entry["locator"] in catalog.tracks:
                catalog.protected[str((bbox_manifest.parent / "tracks" / entry["archive"]).resolve())] = entry["bbox_sha256"]
        selected = None
        if args.smoke:
            selected = {t["locator"] for t in json.loads(args.smoke_selection.read_text())["tracks"] if t["dataset"] == args.source}
            if not selected or len(selected) > 10 or not selected.issubset(catalog.tracks):
                raise ValueError("Smoke selection outside frozen source train/validation")
        datasets = {}
        for split in ("training", "validation"):
            d = baseline["dataset"](args.source, split)
            if selected is not None:
                d.entries = [e for e in d.entries if e["locator"] in selected]
                if not d.entries:
                    raise ValueError("Smoke needs explicit source training and validation tracks")
            datasets[split] = FeatureDataset(d, supply, args.source, args.arm)
        full_training_count = sum(t["split"] == "training" for t in catalog.tracks.values())
        warmup = 3 * ((full_training_count + 63) // 64)
        if args.device == "cuda":
            log = (args.output / "system.csv").open("w")
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
                                          "--format=csv", "-l", "1"], stdout=log, stderr=subprocess.STDOUT)

        def snapshot():
            return {"rgb": rgb.snapshot(), "points": points.snapshot(), "encoder_loads": encoders.loads,
                    "initial": {"rgb": rgb.initial, "points": points.initial},
                    "encoder_calls": encoders.calls, "encoder_seconds": encoders.seconds,
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
                changed = [p for p, h in catalog.protected.items() if checksum(p) != h]
                write_json(args.output / "protected.json", {"hashes": catalog.protected, "changed": changed})
                if changed:
                    raise ValueError("Frozen source files changed")

        def batches(d, training=False, seed=0):
            def packing(row):
                with (args.output / "packing.jsonl").open("a") as stream:
                    stream.write(json.dumps(row | {"phase": "training" if training else "validation"}, allow_nan=False) + "\n")
            logical = LogicalBatches(d, profile, training, seed, max_updates=2 if args.smoke and training else None, on_pack=packing)
            return logical if training else PredictionBatches(logical)

        def conditions(metadata, arrays, track, dataset):
            c, flags = baseline["conditions"](metadata, arrays, track, dataset)
            return c | {"rgb-observed": track["source_flags"]["rgb_valid"].astype(np.int64),
                        "lidar-observed": track["source_flags"]["lidar_valid"].astype(np.int64)}, flags

        return {"config": {"experiment": "E002a", "arm": args.arm, "model": MODEL_SPEC, "geometry": "E001b-native-geometry",
                    "frozen_features": identity, "profile": profile, "profile_sha256": checksum(args.profile),
                    "point_cache_root": str(args.point_cache_root.resolve()), "point_cache_gb": args.point_cache_gb,
                    "frozen_encoder_residency": "one encoder retained across misses; calibration reserves 2 GiB",
                    "rgb_cache_root": str(args.rgb_cache_root.resolve()), "warmup_updates": warmup,
                    "held_out_tests": False, "exploratory": True},
                "normalization": baseline["normalization"], "datasets": {args.source: baseline["datasets"][args.source]},
                "build_model": lambda: MultimodalClassifier(args.arm, profile),
                "dataset": lambda source, split: datasets[split] if source == args.source else None,
                "batches": batches, "forward_batch": lambda m, b, device: m(b),
                "epoch_operation": partial(train_accumulated_epoch, peak_lr=.0003, warmup_steps=warmup),
                "evaluation_splits": [(args.source, "validation-smoke" if args.smoke else "validation")],
                "conditions": conditions, "kind": "integration-smoke" if args.smoke else "exploratory-source-validation",
                "custom_smoke": True, "smoke_limits": {"selected_tracks": sorted(selected), "max_training_updates": 2,
                    "held_out_tests": False} if selected is not None else None,
                "run_name": f"e002a-{args.source}-{args.arm}-s0" + ("-smoke-" + args.output.name if args.smoke else ""),
                "log_learning_rate": True, "runtime_snapshot": snapshot, "close": close}
    except BaseException:
        stack.close()
        if telemetry is not None:
            telemetry.terminate(); telemetry.wait(timeout=10); log.close()
        raise


def add_arguments(parser):
    parser.add_argument("--source", choices=("loki", "road-waymo"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path, default=Path("outputs/experiments/E002a/rgb-cache-preparation/cache.json"))
    parser.add_argument("--point-cache-root", type=Path, default=Path("outputs/experiments/E002a/point-cache"))
    parser.add_argument("--point-cache-gb", type=float, default=450)
    parser.add_argument("--min-free-gb", type=float, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path, help="Explicit bounded track selection; required for smoke")


def run(args):
    if args.smoke and args.smoke_selection is None:
        raise ValueError("Smoke requires an explicit bounded source train/validation selection")
    torch.set_num_threads(2)
    return execute_attempt(args.source, args.arm, "Transformer", args.output, args.device, args.smoke,
                           settings=SETTINGS, assemble=lambda *_: assembly(args), classes=CLASSES, axes=STRATA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())
