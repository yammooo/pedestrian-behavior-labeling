"""Train one E005 spatial model on the frozen LOKI training split."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
from pathlib import Path
from time import perf_counter

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
AALTO_FAST_PROFILE = {"frame_chunk": 8, "frame_tokens": 262_144, "ordinary_tokens": 3_274_888, "fold_single_query": True}
AALTO_FAST_TRACKS, AALTO_FAST_READERS = 2, 2
SPLITS = ("training", "validation")
OBSERVATION_FLAGS = {"rgb-observed": "rgb_valid", "lidar-observed": "lidar_valid"}


class SpatialBatches(LogicalBatches):
    """Accumulate 64 tracks without materializing all of their spatial tokens."""

    def __init__(self, dataset, profile, training=False, seed=0, *, prefetch=False, prefetch_workers=1):
        super().__init__(dataset, profile, training, seed, prefetch=prefetch)
        self.prefetch_workers = prefetch_workers

    def _collate_group(self, group):
        return collate_e005([self.dataset[i] for i in group])

    def _microbatches(self, indices):
        size = self.profile["max_tracks"]
        for start in range(0, len(indices), size):
            yield self._collate_group(indices[start:start + size])

    def microbatches(self, indices):
        if not self.prefetch or self.prefetch_workers == 1:
            yield from super().microbatches(indices)
            return
        size = self.profile["max_tracks"]
        groups = iter(indices[i:i+size] for i in range(0, len(indices), size))
        cache = self.dataset.cache
        for name in cache.record["arrays"]:
            cache._map(name)
        with ThreadPoolExecutor(max_workers=self.prefetch_workers) as pool:
            def submit(group):
                # Resolve mutable lazy cache entries on the main thread before the parallel reads.
                before = perf_counter()
                for i in group:
                    cache._scene(self.dataset.entries[i]["group"])
                self.stats["wait_seconds"] += perf_counter() - before
                return pool.submit(self._collate_group, group)

            pending = [submit(group) for group in islice(groups, self.prefetch_workers)]
            while pending:
                before = perf_counter()
                batch = pending.pop(0).result()
                self.stats["wait_seconds"] += perf_counter() - before
                next_group = next(groups, None)
                if next_group is not None:
                    pending.append(submit(next_group))
                yield batch
                del batch


def enable_aalto_fast(arm, device):
    if arm != DEFAULT_ARM or device != "cuda" or not torch.cuda.is_available():
        raise ValueError("--aalto-fast requires the geometry + RGB + Utonia arm on CUDA")
    free, total = torch.cuda.mem_get_info()
    cap = min(int(total * .89), free - 768 * 1024**2)
    if cap < 11_000_000_000:
        raise ValueError("--aalto-fast requires at least 11 GB of free allocator capacity")
    torch.cuda.set_per_process_memory_fraction(cap / total)
    return cap


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
    fast = getattr(args, "aalto_fast", False)
    profile = AALTO_FAST_PROFILE if fast else PROFILE
    max_tracks = AALTO_FAST_TRACKS if fast else 1 if lidar else 4
    workers = AALTO_FAST_READERS if fast else 1
    warmup = SETTINGS["warmup_epochs"] * ((len(datasets["training"]) + 63) // 64)

    def batches(dataset, training=False, seed=0):
        logical = SpatialBatches(dataset, {"max_tracks": max_tracks}, training, seed, prefetch=lidar,
                                 prefetch_workers=workers)
        return logical if training else PredictionBatches(logical)

    def conditions(metadata, arrays, track, dataset):
        memberships, flags = baseline["conditions"](metadata, arrays, track, dataset)
        return memberships | {axis: track["source_flags"][flag].astype("int64")
                              for axis, flag in OBSERVATION_FLAGS.items()
                              if flag in track["source_flags"]}, flags

    return {
        "config": {"experiment": "E005", "arm": args.arm, "architecture": VARIANT,
                   "geometry": "E001b-native-geometry", "profile": profile, "warmup_updates": warmup,
                   "execution_profile": "aalto-fast" if fast else "original",
                   "allocator_cap_bytes": getattr(args, "allocator_cap_bytes", None),
                   "cache": {"path": str(cache.root), "manifest_sha256": cache.manifest_sha256,
                             "population": population, "provenance": cache.record["provenance"]},
                   "batching": {"max_tracks": max_tracks, "prefetch_microbatches": workers if lidar else 0},
                   "held_out_tests": False, "exploratory": True},
        "normalization": baseline["normalization"], "datasets": {SOURCE: reference},
        "build_model": lambda: e005.build_model(args.arm, profile),
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
    if getattr(args, "aalto_fast", False):
        args.allocator_cap_bytes = enable_aalto_fast(args.arm, args.device)
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
    parser.add_argument("--aalto-fast", action="store_true", help="Use the calibrated Aalto CUDA execution profile")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
