"""Compare E005 execution settings on one matched, warmed-cache LOKI update."""

import argparse
import itertools
import json
from pathlib import Path
from statistics import median
import subprocess
from types import SimpleNamespace

import torch

from pedestrian_behavior.experiments.e005_cache import atomic_json
from pedestrian_behavior.experiments.e005_run import DEFAULT_ARM, PROFILE, SETTINGS, SOURCE, SpatialBatches, assembly
from pedestrian_behavior.models.e005 import build_model, _frame_chunks
from pedestrian_behavior.training import seed_run


VARIANTS = {
    "baseline": (PROFILE, 1),
    "ordinary-limit": (PROFILE | {"ordinary_tokens": 1_000_000}, 1),
    "eight-frames": ({"frame_chunk": 8, "frame_tokens": 262_144, "ordinary_tokens": 1_000_000}, 1),
    "four-tracks": ({"frame_chunk": 8, "frame_tokens": 262_144, "ordinary_tokens": 1_000_000}, 4),
}
DEVICE = torch.device("cuda")


class OneUpdate(SpatialBatches):
    def __init__(self, dataset, indices, max_tracks, prefetch):
        super().__init__(dataset, {"max_tracks": max_tracks}, prefetch=prefetch)
        self.indices = indices

    def __iter__(self):
        yield self.indices


def timed(prepared, state, dataset, indices, variant, *, prefetch):
    profile, max_tracks = VARIANTS[variant]
    seed_run(0)
    torch.cuda.empty_cache()
    model = build_model(DEFAULT_ARM, profile).to(DEVICE)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    batches = OneUpdate(dataset, indices, max_tracks, prefetch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        row = prepared["epoch_operation"](model, batches, optimizer, DEVICE,
                                           SETTINGS["clip_norm"], step_offset=402)
    except torch.cuda.OutOfMemoryError:
        return {"status": "oom", "variant": variant, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    assert row["processed_tracks"] == len(indices) and row["optimizer_steps"] == 1
    return row | {
        "status": "complete",
        "variant": variant,
        "loader_wait_seconds": batches.stats["wait_seconds"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def track_costs(dataset):
    """Read only scene metadata to find the densest tracks for the memory probe."""
    cache = dataset.cache
    groups = {}
    for i, entry in enumerate(dataset.entries):
        groups.setdefault(entry["group"], []).append(i)
    result = {}
    for group, indices in groups.items():
        scene_path = cache.root / cache.record["scenes"][group]["path"]
        frames = json.loads(scene_path.read_text())["frames"]
        for i in indices:
            locator = dataset.entries[i]["locator"]
            rgb, lidar = [], []
            for key in cache.tracks[locator]["source_keys"]:
                frame = frames.get(str(key), {})
                target = frame.get("targets", {}).get(locator, {})
                rgb.append(256 if locator in frame.get("rgb", {}) else 0)
                lidar.append(sum(target.get(f"stage{stage}_indices", (0, 0))[1] for stage in (3, 5)))
            result[i] = rgb, lidar
    return result


def padded_count(costs, indices, profile):
    total = 0
    for levels in zip(*(costs[i] for i in indices)):
        counts = list(itertools.chain.from_iterable(levels))
        chunks = _frame_chunks(counts, profile["frame_chunk"], profile["frame_tokens"])
        total += sum(len(chunk) * max(counts[i] for i in chunk) for chunk in chunks)
    return total


def run(args):
    output = args.output.resolve()
    root = args.outputs_root.resolve()
    allowed = root / "experiments/E005/benchmarks"
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("Use a fresh directory inside outputs/experiments/E005/benchmarks")
    if not torch.cuda.is_available():
        raise ValueError("The E005 benchmark requires CUDA")
    torch.set_num_threads(2)
    seed_run(0)
    torch.cuda.set_per_process_memory_fraction(.75)
    output.mkdir(parents=True, exist_ok=False)
    destinations = output / "benchmark.json"
    record = {"status": "incomplete", "variants": VARIANTS, "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "seed": 0, "rows": [],
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    prepared = assembly(SimpleNamespace(outputs_root=root, arm=DEFAULT_ARM, cache=None))
    try:
        dataset = prepared["dataset"](SOURCE, "training")
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0)).tolist()
        indices = order[:64]
        record["locators"] = [dataset.entries[i]["locator"] for i in indices]
        record["cache_manifest_sha256"] = prepared["config"]["cache"]["manifest_sha256"]
        state = build_model(DEFAULT_ARM, PROFILE).state_dict()
        # Warm the shared file pages once and retain CPU batches for the no-loader comparison.
        samples = {i: dataset[i] for i in indices}

        def measure(variant, source, selected, workload, *, prefetch, **extra):
            row = timed(prepared, state, source, selected, variant, prefetch=prefetch)
            record["rows"].append(row | {"workload": workload} | extra)
            atomic_json(destinations, record)
            print(record["rows"][-1], flush=True)
            return row

        eligible = [name for name in VARIANTS if measure(name, samples, indices[:8], "warmup",
                                                           prefetch=False)["status"] == "complete"]
        for repeat, names in enumerate((eligible, list(reversed(eligible))), start=1):
            for name in names:
                measure(name, dataset, indices, "streamed", prefetch=True, repeat=repeat)
        streamed = {name: [row["seconds"] for row in record["rows"] if row["workload"] == "streamed"
                           and row["variant"] == name and row["status"] == "complete"] for name in eligible}
        winner = min((name for name in eligible if len(streamed[name]) == 2), key=lambda name: median(streamed[name]))
        record["winner"] = winner
        for name in dict.fromkeys(("baseline", winner)):
            measure(name, samples, indices, "preloaded", prefetch=False)
        del samples

        profile, tracks = VARIANTS[winner]
        costs = track_costs(dataset)
        groups = [[i] for i in order]
        if tracks > 1:
            groups += [order[i:i+tracks] for start in range(0, len(order), 64)
                       for i in range(start, min(start+64, len(order)), tracks)]
        by_size = [(padded_count(costs, group, profile), group) for group in groups]
        for name, candidates in (("densest", by_size),
                                 ("largest-ordinary", [row for row in by_size if row[0] <= profile["ordinary_tokens"]])):
            if candidates:
                count, stress_indices = max(candidates)
                measure(winner, dataset, stress_indices, name, prefetch=True, padded_tokens=count,
                        locators=[dataset.entries[i]["locator"] for i in stress_indices])
        record["status"] = "complete"
        atomic_json(destinations, record)
    finally:
        prepared["close"]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
