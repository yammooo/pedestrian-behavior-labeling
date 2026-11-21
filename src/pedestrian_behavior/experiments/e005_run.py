"""Train one E005 spatial model on a frozen source training split."""

import argparse
import json
import math
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
AALTO_ROAD_FAST_PROFILE = AALTO_FAST_PROFILE | {"sort_frames_by_tokens": True}
AALTO_FAST_TRACKS, AALTO_FAST_READERS = 2, 2
AALTO_GPU = "NVIDIA GeForce RTX 4080 Laptop GPU"
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
        caches = getattr(self.dataset, "caches", None)
        if caches is None:
            caches = {SOURCE: self.dataset.cache}
        for cache in caches.values():
            for name in cache.record["arrays"]:
                cache._map(name)
        with ThreadPoolExecutor(max_workers=self.prefetch_workers) as pool:
            def submit(group):
                # Resolve mutable lazy cache entries on the main thread before the parallel reads.
                before = perf_counter()
                for i in group:
                    if hasattr(self.dataset, "prepare_scene"):
                        self.dataset.prepare_scene(i)
                    else:
                        self.dataset.cache._scene(self.dataset.entries[i]["group"])
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


class JointFeatureDataset:
    """A single index over the two frozen E005 populations."""

    def __init__(self, datasets, caches):
        self.datasets, self.caches = datasets, caches
        self.source_indices, self.source_ranges, self.entries = {}, {}, []
        for source, dataset in datasets.items():
            start = len(self.entries)
            self.entries.extend(entry | {"source": source} for entry in dataset.entries)
            indices = list(range(start, len(self.entries)))
            self.source_indices[source] = indices
            self.source_ranges[source] = (start, len(self.entries))
        self._routes = [(source, index) for source, dataset in datasets.items() for index in range(len(dataset))]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        source, local = self._routes[index]
        return self.datasets[source][local]

    def prepare_scene(self, index):
        source, local = self._routes[index]
        self.caches[source]._scene(self.datasets[source].entries[local]["group"])

    def close(self):
        for cache in self.caches.values():
            cache.close()


class JointSpatialBatches(SpatialBatches):
    """Balanced 32+32 track updates with source queues carried across epochs."""

    def __init__(self, dataset, profile, training=False, seed=0, *, prefetch=False,
                 prefetch_workers=1, source_batch_size=32):
        super().__init__(dataset, profile, training, seed, prefetch=prefetch,
                         prefetch_workers=prefetch_workers)
        self.source_batch_size = source_batch_size
        self.queues = {source: [] for source in dataset.source_indices}
        self.positions = {source: 0 for source in dataset.source_indices}

    def __len__(self):
        visits = (len(self.dataset) + 1) // 2
        return math.ceil(visits / self.source_batch_size)

    def _take(self, source, count):
        indices, queue = self.dataset.source_indices[source], self.queues[source]
        taken = []
        while len(taken) < count:
            if self.positions[source] == len(queue):
                order = torch.randperm(len(indices), generator=self.generator).tolist()
                self.queues[source] = queue = [indices[i] for i in order]
                self.positions[source] = 0
            available = min(count - len(taken), len(queue) - self.positions[source])
            start = self.positions[source]
            taken.extend(queue[start:start + available])
            self.positions[source] += available
        return taken

    def __iter__(self):
        visits = (len(self.dataset) + 1) // 2
        for start in range(0, visits, self.source_batch_size):
            count = min(self.source_batch_size, visits - start)
            indices = self._take("road-waymo", count) + self._take("loki", count)
            order = torch.randperm(len(indices), generator=self.generator).tolist()
            yield [indices[i] for i in order]


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


def _cache_path(args, root, source):
    if args.cache is not None:
        return args.cache
    if source != SOURCE:
        raise ValueError("ROAD E005 requires an explicit spatial cache path")
    return root / DEFAULT_CACHE


def _merge_normalization(statistics):
    counts = torch.tensor([row["count"] for row in statistics], dtype=torch.float64)
    count = counts.sum(0)
    means = torch.tensor([row["mean"] for row in statistics], dtype=torch.float64)
    stds = torch.tensor([row["std"] for row in statistics], dtype=torch.float64)
    mean = (means * counts).sum(0) / count.clamp_min(1)
    variance = sum((row_count * (row_std.square() + (row_mean - mean).square())
                    for row_count, row_std, row_mean in zip(counts, stds, means))) / count.clamp_min(1)
    std = variance.clamp_min(0).sqrt()
    scale = torch.where(std <= statistics[0]["threshold"], 1., std)
    guarded = [{"column": i, "reason": "entirely-missing" if count[i] == 0 else "near-zero",
                "std": float(std[i])} for i in range(len(count))
               if count[i] == 0 or std[i] <= statistics[0]["threshold"]]
    return {"mean": mean.tolist(), "scale": scale.tolist(), "count": count.to(torch.int64).tolist(),
            "std": std.tolist(), "threshold": statistics[0]["threshold"], "guarded_columns": guarded}


def _joint_cache_paths(args, root):
    loki = args.loki_cache or root / DEFAULT_CACHE
    if args.road_cache is None:
        raise ValueError("Joint E005 requires --road-cache")
    return {"loki": loki, "road-waymo": args.road_cache}


def _joint_assembly(args, *, allow_partial=False):
    root = args.outputs_root.resolve()
    cache_paths = _joint_cache_paths(args, root)
    sources = ("road-waymo", "loki")
    baselines, caches = {}, {}
    try:
        for source in sources:
            baselines[source] = e001b_run.assemble(source, "geometry", "B", e001_root=root / "experiments/E001",
                                                   e001b_root=root / "experiments/E001b")
            caches[source] = E005Cache(cache_paths[source], allow_partial=allow_partial)
            reference = baselines[source]["datasets"][source]
            population = caches[source].record["population"]
            if (population["dataset"], population["setup_sha256"], population["manifest_sha256"]) != (
                    source, reference["setup_sha256"], reference["collection_sha256"]):
                raise ValueError(f"E005 cache and frozen {source} setup/collection differ")

        normalization = {
            "baseline": _merge_normalization([baselines[s]["normalization"]["baseline"] for s in sources]),
            "bbox": _merge_normalization([baselines[s]["normalization"]["bbox"] for s in sources]),
        }
        datasets, cache_by_source = {}, {}
        for source in sources:
            baseline = baselines[source]
            source_datasets = []
            for split in SPLITS:
                track_dataset = baseline["dataset"](source, split)
                track_dataset.statistics = normalization["baseline"]
                track_dataset.bbox_statistics = normalization["bbox"]
                if caches[source].record["status"] == "bounded-native-check":
                    track_dataset.entries = [e for e in track_dataset.entries if e["locator"] in caches[source].tracks]
                if not track_dataset.entries:
                    raise ValueError(f"E005 {source} cache has no frozen {split} tracks")
                dataset = E005FeatureDataset(track_dataset, caches[source], args.arm)
                source_datasets.append(dataset)
                datasets[source, split] = dataset
                cache_by_source[source] = caches[source]
            entries = {e["locator"]: e for dataset in source_datasets for e in dataset.entries}
            if entries.keys() != caches[source].tracks.keys() or any(
                    any(e[key] != caches[source].tracks[locator][key] for key in ("archive", "group", "split", "slots"))
                    or e["e001_sha256"] != caches[source].tracks[locator]["sha256"]
                    for locator, e in entries.items()):
                raise ValueError(f"E005 cached track references differ from the frozen {source} population")

        training = JointFeatureDataset({s: datasets[s, "training"] for s in sources}, cache_by_source)
    except BaseException:
        for cache in caches.values():
            cache.close()
        raise

    fast = getattr(args, "aalto_fast", False)
    profile = AALTO_FAST_PROFILE if fast else PROFILE
    max_tracks = AALTO_FAST_TRACKS if fast else 1 if args.arm == DEFAULT_ARM else 4
    workers = AALTO_FAST_READERS if fast and args.arm == DEFAULT_ARM else 1
    joint_profile = getattr(args, "joint_profile", None)
    if joint_profile is not None:
        record = json.loads(Path(joint_profile).read_text())
        if record.get("status") != "complete" or not isinstance(record.get("winner"), dict):
            raise ValueError("Joint profile must be a completed calibration with a winner")
        winner = record["winner"]
        profile = winner["profile"]
        max_tracks = winner["max_tracks"]
        workers = winner["workers"]
        if not isinstance(profile, dict) or max_tracks < 1 or workers < 1:
            raise ValueError("Invalid selected joint training profile")

    warmup = SETTINGS["warmup_epochs"] * math.ceil(len(training) / (2 * 32))

    def batches(dataset, training=False, seed=0):
        cls = JointSpatialBatches if getattr(dataset, "source_indices", None) else SpatialBatches
        logical = cls(dataset, {"max_tracks": max_tracks}, training, seed,
                      prefetch=args.arm == DEFAULT_ARM, prefetch_workers=workers)
        return logical if training else PredictionBatches(logical)

    def validate(model):
        from pedestrian_behavior.evaluation import predict, prediction_metrics
        metrics = {}
        for source in sources:
            validation = batches(datasets[source, "validation"], seed=SETTINGS["seed"])
            tracks = predict(model, validation, torch.device(args.device), forward_batch=lambda m, b, _: m(b))
            metrics[source] = prediction_metrics(tracks, len(CLASSES))
        return {"track_weighted": {"macro_f1": sum(m["track_weighted"]["macro_f1"] for m in metrics.values()) / len(metrics)},
                "track_ce": sum(m["track_ce"] for m in metrics.values()) / len(metrics),
                **{f"val/{s}/macro_f1": m["track_weighted"]["macro_f1"] for s, m in metrics.items()},
                **{f"val/{s}/track_ce": m["track_ce"] for s, m in metrics.items()}}

    def conditions(metadata, arrays, track, source):
        memberships, flags = baselines[source]["conditions"](metadata, arrays, track, source)
        return memberships | {axis: track["source_flags"][flag].astype("int64")
                              for axis, flag in OBSERVATION_FLAGS.items()
                              if flag in track["source_flags"]}, flags

    return {
        "config": {"experiment": "E005", "arm": args.arm, "architecture": VARIANT,
                   "geometry": "E001b-native-geometry", "profile": profile, "warmup_updates": warmup,
                   "execution_profile": "joint-calibrated" if joint_profile else "aalto-fast" if fast else "original",
                   "allocator_cap_bytes": getattr(args, "allocator_cap_bytes", None),
                   "joint_sampling": {"tracks_per_source_per_update": 32, "tracks_per_source_per_epoch": len(training) // 2,
                                      "sources": list(sources), "epoch_updates": len(JointSpatialBatches(training, profile, source_batch_size=32))},
                   "joint_profile": str(joint_profile) if joint_profile else None,
                   "caches": {s: {"path": str(caches[s].root), "manifest_sha256": caches[s].manifest_sha256,
                                  "population": caches[s].record["population"],
                                  "provenance": caches[s].record["provenance"]} for s in sources},
                   "batching": {"max_tracks": max_tracks, "prefetch_microbatches": workers if args.arm == DEFAULT_ARM else 0},
                   "held_out_tests": False, "exploratory": True},
        "normalization": normalization,
        "datasets": {s: baselines[s]["datasets"][s] for s in sources},
        "build_model": lambda: e005.build_model(args.arm, profile),
        "dataset": lambda source, split: training if source == "joint" and split == "training" else datasets.get((source, split)),
        "batches": batches, "forward_batch": lambda model, batch, _: model(batch),
        "epoch_operation": partial(train_accumulated_epoch, peak_lr=SETTINGS["optimizer"]["lr"],
                                   warmup_steps=warmup),
        "validation_metrics": validate,
        "evaluation_splits": [(s, "validation") for s in sources], "conditions": conditions,
        "kind": "exploratory-joint-validation", "log_learning_rate": True,
        "close": training.close,
    }


def assembly(args, *, allow_partial=False):
    root = args.outputs_root.resolve()
    source = getattr(args, "source", SOURCE)
    if source == "joint":
        return _joint_assembly(args, allow_partial=allow_partial)
    baseline = e001b_run.assemble(source, "geometry", "B", e001_root=root / "experiments/E001",
                                 e001b_root=root / "experiments/E001b")
    reference = baseline["datasets"][source]
    cache = E005Cache(_cache_path(args, root, source), allow_partial=allow_partial)
    population = cache.record["population"]
    if (population["dataset"], population["setup_sha256"], population["manifest_sha256"]) != (
            source, reference["setup_sha256"], reference["collection_sha256"]):
        raise ValueError(f"E005 cache and frozen {source} setup/collection differ")

    datasets = {}
    for split in SPLITS:
        dataset = baseline["dataset"](source, split)
        if cache.record["status"] == "bounded-native-check":
            dataset.entries = [entry for entry in dataset.entries if entry["locator"] in cache.tracks]
        if not dataset.entries:
            raise ValueError(f"E005 cache has no frozen {split} tracks")
        datasets[split] = E005FeatureDataset(dataset, cache, args.arm)
    entries = {entry["locator"]: entry for dataset in datasets.values() for entry in dataset.entries}
    if entries.keys() != cache.tracks.keys() or any(
            any(entry[key] != cache.tracks[locator][key] for key in ("archive", "group", "split", "slots"))
            or entry["e001_sha256"] != cache.tracks[locator]["sha256"]
            for locator, entry in entries.items()):
        raise ValueError(f"E005 cached track references differ from the frozen {source} population")

    lidar = args.arm == DEFAULT_ARM
    fast = getattr(args, "aalto_fast", False)
    profile = AALTO_FAST_PROFILE if fast else PROFILE
    if fast and source == "road-waymo":
        profile = AALTO_ROAD_FAST_PROFILE
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
        "normalization": baseline["normalization"], "datasets": {source: reference},
        "build_model": lambda: e005.build_model(args.arm, profile),
        "dataset": lambda dataset_source, split: datasets[split] if dataset_source == source else None,
        "batches": batches, "forward_batch": lambda model, batch, _: model(batch),
        "epoch_operation": partial(train_accumulated_epoch, peak_lr=SETTINGS["optimizer"]["lr"],
                                   warmup_steps=warmup),
        "evaluation_splits": [(source, "validation")], "conditions": conditions,
        "kind": "exploratory-source-validation", "log_learning_rate": True, "close": cache.close,
    }


def run(args):
    source = getattr(args, "source", SOURCE)
    output = args.output.resolve()
    root = args.outputs_root.resolve()
    runs = (root / "experiments/E005/runs").resolve()
    cache_paths = ([path.resolve() for path in _joint_cache_paths(args, root).values()] if source == "joint"
                   else [_cache_path(args, root, source).resolve()])
    if output == runs or not output.is_relative_to(runs) or any(output.is_relative_to(path) for path in cache_paths):
        raise ValueError("Use a fresh output directory under outputs/experiments/E005/runs")
    if getattr(args, "aalto_fast", None) is None:
        args.aalto_fast = (args.arm == DEFAULT_ARM and args.device == "cuda" and torch.cuda.is_available()
                           and torch.cuda.get_device_name() == AALTO_GPU)
    if args.aalto_fast:
        args.allocator_cap_bytes = enable_aalto_fast(args.arm, args.device)
    torch.set_num_threads(2)
    return run_attempt(source, args.arm, VARIANT, output, args.device,
                       settings=SETTINGS, assemble=lambda *_: assembly(args),
                       classes=CLASSES, axes=axes(args.arm))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=(SOURCE, "road-waymo", "joint"), default=SOURCE)
    parser.add_argument("--arm", choices=e005.ARMS, default=DEFAULT_ARM)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--cache", type=Path, help=f"Completed frozen cache under outputs root (default: {DEFAULT_CACHE})")
    parser.add_argument("--loki-cache", type=Path, help=f"LOKI cache for joint training (default: {DEFAULT_CACHE})")
    parser.add_argument("--road-cache", type=Path, help="Completed ROAD cache for joint training")
    parser.add_argument("--joint-profile", type=Path, help="Completed joint training calibration JSON")
    parser.add_argument("--aalto-fast", action=argparse.BooleanOptionalAction,
                        help="Use the calibrated profile (default for g-rgb-utonia on the RTX 4080 Laptop GPU)")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
