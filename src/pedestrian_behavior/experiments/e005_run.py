"""Train one E005 spatial model on the frozen LOKI training split."""

import argparse
from functools import partial
from pathlib import Path

import torch

from pedestrian_behavior.accumulation import train_accumulated_epoch
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches
from pedestrian_behavior.models import e005

from . import e001b_run
from .e001 import CLASSES
from .e002a_run import SETTINGS, STRATA
from .e005_cache import E005Cache, E005FeatureDataset, collate_e005
from .run import run_attempt


SOURCE, VARIANT = "loki", "bilstm-96"
DEFAULT_ARM = "g-rgb-utonia"
DEFAULT_CACHE = Path("experiments/E005/loki-cache-full-20261117")
PROFILE = {"frame_chunk": 2, "frame_tokens": 65_536, "ordinary_tokens": 65_536}
SPLITS = ("training", "validation")
OBSERVATION_FLAGS = {"rgb-observed": "rgb_valid", "lidar-observed": "lidar_valid"}


class SpatialBatches(LogicalBatches):
    """Accumulate 64 tracks without materializing all of their spatial tokens."""

    def _microbatches(self, indices):
        size = self.profile["max_tracks"]
        for start in range(0, len(indices), size):
            yield collate_e005([self.dataset[i] for i in indices[start:start + size]])


def axes(arm):
    return {key: labels for key, labels in STRATA.items()
            if key not in OBSERVATION_FLAGS or OBSERVATION_FLAGS[key] in e005.BRANCH_FLAGS[arm]}


def assembly(args):
    root = args.outputs_root.resolve()
    baseline = e001b_run.assemble(SOURCE, "geometry", "B", e001_root=root / "experiments/E001",
                                 e001b_root=root / "experiments/E001b")
    reference = baseline["datasets"][SOURCE]
    cache = E005Cache(args.cache or root / DEFAULT_CACHE)
    population = cache.record["population"]
    if (population["dataset"], population["setup_sha256"], population["manifest_sha256"]) != (
            SOURCE, reference["setup_sha256"], reference["collection_sha256"]):
        raise ValueError("E005 cache and frozen LOKI setup/collection differ")

    datasets = {split: E005FeatureDataset(baseline["dataset"](SOURCE, split), cache, args.arm)
                for split in SPLITS}
    entries = {entry["locator"]: entry for dataset in datasets.values() for entry in dataset.entries}
    if entries.keys() != cache.tracks.keys() or any(
            any(entry[key] != cache.tracks[locator][key] for key in ("archive", "group", "split", "slots"))
            or entry["e001_sha256"] != cache.tracks[locator]["sha256"]
            for locator, entry in entries.items()):
        raise ValueError("E005 cached track references differ from the frozen LOKI population")

    lidar = args.arm == DEFAULT_ARM
    max_tracks = 1 if lidar else 4
    warmup = SETTINGS["warmup_epochs"] * ((len(datasets["training"]) + 63) // 64)

    def batches(dataset, training=False, seed=0):
        logical = SpatialBatches(dataset, {"max_tracks": max_tracks}, training, seed, prefetch=lidar)
        return logical if training else PredictionBatches(logical)

    def conditions(metadata, arrays, track, dataset):
        memberships, flags = baseline["conditions"](metadata, arrays, track, dataset)
        return memberships | {axis: track["source_flags"][flag].astype("int64")
                              for axis, flag in OBSERVATION_FLAGS.items()
                              if flag in track["source_flags"]}, flags

    return {
        "config": {"experiment": "E005", "arm": args.arm, "architecture": VARIANT,
                   "geometry": "E001b-native-geometry", "profile": PROFILE, "warmup_updates": warmup,
                   "cache": {"path": str(cache.root), "manifest_sha256": cache.manifest_sha256,
                             "population": population, "provenance": cache.record["provenance"]},
                   "batching": {"max_tracks": max_tracks, "prefetch_microbatches": int(lidar)},
                   "held_out_tests": False, "exploratory": True},
        "normalization": baseline["normalization"], "datasets": {SOURCE: reference},
        "build_model": lambda: e005.build_model(args.arm, PROFILE),
        "dataset": lambda source, split: datasets[split] if source == SOURCE else None,
        "batches": batches, "forward_batch": lambda model, batch, _: model(batch),
        "epoch_operation": partial(train_accumulated_epoch, peak_lr=SETTINGS["optimizer"]["lr"],
                                   warmup_steps=warmup),
        "evaluation_splits": [(SOURCE, "validation")], "conditions": conditions,
        "kind": "exploratory-source-validation", "log_learning_rate": True, "close": cache.close,
    }


def run(args):
    output = args.output.resolve()
    root = args.outputs_root.resolve()
    runs = (root / "experiments/E005/runs").resolve()
    cache = (args.cache or root / DEFAULT_CACHE).resolve()
    if output == runs or not output.is_relative_to(runs) or output.is_relative_to(cache):
        raise ValueError("Use a fresh output directory under outputs/experiments/E005/runs")
    torch.set_num_threads(2)
    return run_attempt(SOURCE, args.arm, VARIANT, output, args.device,
                       settings=SETTINGS, assemble=lambda *_: assembly(args),
                       classes=CLASSES, axes=axes(args.arm))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=e005.ARMS, default=DEFAULT_ARM)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--cache", type=Path, help=f"Completed frozen cache under outputs root (default: {DEFAULT_CACHE})")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
