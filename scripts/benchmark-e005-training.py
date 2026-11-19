"""Compare E005 execution settings on one matched, warmed-cache LOKI update."""

import argparse
from contextlib import nullcontext
import itertools
import json
from pathlib import Path
import runpy
from statistics import median
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import torch

from pedestrian_behavior.experiments.e005_cache import atomic_json
from pedestrian_behavior.experiments.e005_run import DEFAULT_ARM, SETTINGS, SOURCE, SpatialBatches, assembly, enable_aalto_fast
from pedestrian_behavior.models.e005 import SpatialReadout, build_model, _frame_chunks
from pedestrian_behavior.training import seed_run


ORIGINAL_PROFILE = {"frame_chunk": 2, "frame_tokens": 65_536, "ordinary_tokens": 65_536}
VARIANTS = {
    "baseline": (ORIGINAL_PROFILE, 1),
    "ordinary-limit": (ORIGINAL_PROFILE | {"ordinary_tokens": 1_000_000}, 1),
    "eight-frames": ({"frame_chunk": 8, "frame_tokens": 262_144, "ordinary_tokens": 1_000_000}, 1),
    "four-tracks": ({"frame_chunk": 8, "frame_tokens": 262_144, "ordinary_tokens": 1_000_000}, 4),
}
DEVICE = torch.device("cuda")
TUNING_LAYOUTS = ((1, 2), (2, 8), (4, 8), (4, 16), (8, 16))


class OneUpdate(SpatialBatches):
    def __init__(self, dataset, indices, max_tracks, prefetch, workers):
        super().__init__(dataset, {"max_tracks": max_tracks}, prefetch=prefetch, prefetch_workers=workers)
        self.indices = indices

    def __iter__(self):
        yield self.indices


def timed(prepared, state, dataset, indices, variant, *, prefetch, workers=1):
    profile, max_tracks = VARIANTS[variant]
    seed_run(0)
    torch.cuda.empty_cache()
    model = build_model(DEFAULT_ARM, profile).to(DEVICE)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    batches = OneUpdate(dataset, indices, max_tracks, prefetch, workers)
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
        "workers": workers,
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


def tune(dataset, order, measure, cap_bytes, record):
    """Find fast layouts with ordinary ceilings derived from measured reserve peaks."""
    costs = track_costs(dataset)
    previous_observation = json.loads(
        (dataset.cache.root.parent / "benchmarks/e005-chunks-20261119/benchmark.json").read_text())
    previous = next(row for row in previous_observation["rows"] if row["workload"] == "largest-ordinary")
    reserved_per_token = previous["peak_reserved_bytes"] / previous["padded_tokens"]
    record["prior_reserved_bytes_per_padded_token"] = reserved_per_token
    ranked = {}
    for tracks, frames in TUNING_LAYOUTS:
        name = f"{tracks}track-{frames}frame"
        profile = {"frame_chunk": frames, "frame_tokens": 32_768 * frames, "ordinary_tokens": 1}
        groups = [order[i:i+tracks] for i in range(0, len(order), tracks)]
        by_size = sorted((padded_count(costs, group, profile), group) for group in groups)
        upper = int(cap_bytes * .88 / reserved_per_token)
        approved = None
        for attempt in range(3):
            candidates = [case for case in by_size if case[0] <= upper]
            if not candidates:
                break
            tokens, indices = candidates[-1]
            if approved is not None and tokens == approved:
                break
            VARIANTS[name] = (profile | {"ordinary_tokens": tokens}, tracks)
            row = measure(name, dataset, indices, "calibration", prefetch=True,
                          padded_tokens=tokens, attempt=attempt,
                          locators=[dataset.entries[i]["locator"] for i in indices])
            if row["status"] != "complete":
                upper = min(int(tokens * .78), approved or tokens)
                if approved is not None:
                    break
                continue
            if row["peak_reserved_bytes"] > cap_bytes * .94:
                upper = int(tokens * .86)
                continue
            approved = tokens
            upper = int(tokens * min(1.4, cap_bytes * .90 / row["peak_reserved_bytes"]))
            if upper <= tokens or tokens == by_size[-1][0]:
                break
        if approved is not None:
            VARIANTS[name] = (profile | {"ordinary_tokens": approved}, tracks)
            ranked[name] = by_size
        else:
            VARIANTS.pop(name, None)

    profiles = ["baseline", *ranked]
    scores = {}
    cohorts = [order[i:i+64] for i in (0, 1280)]
    for cohort, indices in enumerate(cohorts):
        # Warm exactly this cohort for every layout, then reverse layout order on the second cohort.
        for i in indices:
            dataset[i]
        for name in profiles[::1 if cohort == 0 else -1]:
            row = measure(name, dataset, indices, "layouts", prefetch=True, cohort=cohort)
            if row["status"] == "complete":
                scores.setdefault(name, []).append(row["seconds"])
    finalists = sorted((name for name in scores if len(scores[name]) == len(cohorts)),
                       key=lambda name: median(scores[name]))
    winner = finalists[0]
    record["layout_winner"] = winner

    # Compare bounded concurrent loading on new cohorts, rotating who sees the first file reads.
    shortlist = list(dict.fromkeys(("baseline", *finalists[:2])))
    worker_cases = list(dict.fromkeys([(name, 1) for name in shortlist]
                                     + [(winner, 2), (winner, 4)]))
    ratios = {}
    for cohort, start in enumerate((2560, 4160)):
        indices = order[start:start+64]
        for name, workers in worker_cases[::1 if cohort == 0 else -1]:
            row = measure(name, dataset, indices, "confirm", prefetch=True, workers=workers, cohort=cohort)
            if row["status"] == "complete":
                ratios.setdefault((name, workers), []).append(row["seconds"])
    winner, workers = min((case for case in ratios if len(ratios[case]) == 2),
                          key=lambda case: median(ratios[case]))
    record["winner"], record["winner_workers"] = winner, workers
    if winner in ranked:
        tokens, indices = ranked[winner][-1]
        measure(winner, dataset, indices, "densest", prefetch=True, workers=workers, padded_tokens=tokens,
                locators=[dataset.entries[i]["locator"] for i in indices])

    indices = cohorts[0]
    samples = {i: dataset[i] for i in indices}
    for name in dict.fromkeys(("baseline", winner)):
        measure(name, samples, indices, "preloaded", prefetch=False, cohort=0)

    paired_checks(dataset, order, measure)


def paired_checks(dataset, order, measure):
    old_reader = {}
    source = "src/pedestrian_behavior/experiments/e005_cache.py"
    original = subprocess.check_output(["git", "show", f"bf222a7:{source}"], text=True)
    exec(compile(original, source, "exec"), old_reader)
    cache_type = type(dataset.cache)
    methods = {name: getattr(old_reader["E005Cache"], name) for name in ("rgb_frame", "lidar_frame")}
    folded = runpy.run_path(str(Path(__file__).with_name("benchmark-e005-single-query.py")))["folded_forward"]
    ordinary = SpatialReadout.forward
    cases = (
        ("original-pipeline", "baseline", 1, True, ordinary),
        ("two-track-one-reader", "2track-8frame", 1, False, ordinary),
        ("two-track-two-readers", "2track-8frame", 2, False, ordinary),
        ("two-track-original-reader", "2track-8frame", 2, True, ordinary),
        ("two-track-folded", "2track-8frame", 2, False, folded),
    )
    for cohort, start in enumerate((2560, 4160)):
        indices = order[start:start+64]
        for tag, variant, workers, old, forward in cases[::1 if cohort == 0 else -1]:
            reader = patch.multiple(cache_type, **methods) if old else nullcontext()
            with reader, patch.object(SpatialReadout, "forward", forward):
                measure(variant, dataset, indices, "paired-pipeline", prefetch=True, workers=workers,
                        cohort=cohort, case=tag)


def validate_fast(prepared, dataset, order, record, destinations):
    """Run yielded batches through an unchanged model/optimizer across three real updates."""
    profile = prepared["config"]["profile"]
    costs = track_costs(dataset)
    count = prepared["config"]["batching"]["max_tracks"]
    cases = [(padded_count(costs, order[i:i+count], profile), order[i:i+count])
             for i in range(0, len(order), count)]
    ordinary = max(row for row in cases if row[0] <= profile["ordinary_tokens"])
    dense_indices = sorted(order, key=lambda i: padded_count(costs, [i], profile), reverse=True)[:count]
    dense = padded_count(costs, dense_indices, profile), dense_indices
    model = prepared["build_model"]().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    batches = prepared["batches"](dataset, training=True, seed=0)

    def update(indices, label, offset, stress_tokens=None):
        class Selection:
            def __iter__(self):
                yield indices

            microbatches = batches.microbatches

        torch.cuda.reset_peak_memory_stats()
        wait = batches.stats["wait_seconds"]
        row = prepared["epoch_operation"](model, Selection(), optimizer, DEVICE,
                                           SETTINGS["clip_norm"], step_offset=402+offset)
        assert row["processed_tracks"] == 64 and row["optimizer_steps"] == 1
        row.update(profile=profile, workload=label, loader_wait_seconds=batches.stats["wait_seconds"]-wait,
                   stress_padded_tokens=stress_tokens, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
        record["rows"].append(row)
        atomic_json(destinations, record)
        print(row, flush=True)

    update(order[:64], "persistent-representative", 0)
    for offset, (label, (tokens, prefix)) in enumerate((("persistent-ordinary", ordinary), ("persistent-dense", dense)), start=1):
        rest = [i for i in order[offset*128:offset*128+64] if i not in prefix]
        update(prefix + rest[:64-len(prefix)], label, offset, tokens)
    return costs


def validate_folded_cuda():
    folded = runpy.run_path(str(Path(__file__).with_name("benchmark-e005-single-query.py")))["folded_forward"]
    ordinary = SpatialReadout.forward
    seed_run(0)
    layer = SpatialReadout().to(DEVICE).train()
    frames = [torch.randn(n, 128, device=DEVICE, requires_grad=True) for n in (17, 4093, 21_859)]
    args = frames + list(layer.parameters())
    seed_run(0)
    reference = ordinary(layer, frames)
    reference_grad = torch.autograd.grad(reference.square().mean(), args)
    seed_run(0)
    result = folded(layer, frames)
    result_grad = torch.autograd.grad(result.square().mean(), args)
    torch.testing.assert_close(result, reference, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(result_grad, reference_grad):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    return {"dropout_output_max_diff": (result-reference).abs().max().item(),
            "gradient_max_diff": max((actual-expected).abs().max().item()
                                      for actual, expected in zip(result_grad, reference_grad))}


def check_four_track_folded(dataset, order, costs, profile, measure):
    groups = [order[i:i+4] for i in range(0, len(order), 4)]
    tokens, dense = max((padded_count(costs, group, profile), group) for group in groups)
    VARIANTS["folded-four-tracks"] = profile | {"ordinary_tokens": tokens}, 4
    stress = measure("folded-four-tracks", dataset, dense, "four-track-folded-stress", prefetch=True, workers=2,
                     padded_tokens=tokens, locators=[dataset.entries[i]["locator"] for i in dense])
    if stress["status"] != "complete":
        return
    options = ("2track-8frame", "folded-four-tracks")
    indices = order[:64]
    for i in indices:
        dataset[i]
    for repeat, pair in enumerate((options, options[::-1])):
        for name in pair:
            measure(name, dataset, indices, "folded-track-pair", prefetch=True, workers=2, repeat=repeat)


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
    free, total = torch.cuda.mem_get_info()
    cap = (enable_aalto_fast(DEFAULT_ARM, "cuda") if args.verify_fast else
           min(int(total * .89), free - 768 * 1024**2) if args.tune or args.pairs_from else int(total * .75))
    if cap < total * .75:
        raise ValueError("GPU memory is occupied")
    if not args.verify_fast:
        torch.cuda.set_per_process_memory_fraction(cap / total)
    output.mkdir(parents=True, exist_ok=False)
    destinations = output / "benchmark.json"
    record = {"status": "incomplete", "variants": VARIANTS, "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "seed": 0, "rows": [],
              "gpu_total_bytes": total, "gpu_free_bytes": free, "allocator_cap_bytes": cap,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    prepared = assembly(SimpleNamespace(outputs_root=root, arm=DEFAULT_ARM, cache=None, aalto_fast=args.verify_fast,
                                        allocator_cap_bytes=cap))
    try:
        dataset = prepared["dataset"](SOURCE, "training")
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0)).tolist()
        indices = order[:64]
        record["locators"] = [dataset.entries[i]["locator"] for i in indices]
        record["cache_manifest_sha256"] = prepared["config"]["cache"]["manifest_sha256"]
        state = build_model(DEFAULT_ARM, ORIGINAL_PROFILE).state_dict()

        def measure(variant, source, selected, workload, *, prefetch, workers=1, **extra):
            row = timed(prepared, state, source, selected, variant, prefetch=prefetch, workers=workers)
            extra |= {"profile": VARIANTS[variant][0], "max_tracks": VARIANTS[variant][1]}
            record["rows"].append(row | {"workload": workload} | extra)
            atomic_json(destinations, record)
            print(record["rows"][-1], flush=True)
            return row

        if args.verify_fast:
            record["execution"] = {key: prepared["config"][key] for key in ("execution_profile", "profile", "batching")}
            costs = validate_fast(prepared, dataset, order, record, destinations)
            VARIANTS["2track-8frame"] = prepared["config"]["profile"], prepared["config"]["batching"]["max_tracks"]
            try:
                record["folded_cuda_parity"] = validate_folded_cuda()
            except AssertionError as error:
                record["folded_cuda_parity"] = {"status": "failed", "message": str(error)}
            check_four_track_folded(dataset, order, costs, prepared["config"]["profile"], measure)
            record["status"] = "complete"
            atomic_json(destinations, record)
            return

        if args.pairs_from:
            previous = json.loads(args.pairs_from.read_text())
            assert previous["cache_manifest_sha256"] == record["cache_manifest_sha256"]
            assert previous["locators"] == record["locators"]
            VARIANTS.update(previous["variants"])
            record["prior_calibration"] = str(args.pairs_from)
            paired_checks(dataset, order, measure)
            record["status"] = "complete"
            atomic_json(destinations, record)
            return

        if args.tune:
            record["cohort_starts"] = [0, 1280, 2560, 4160]
            record["cohort_locators"] = [[dataset.entries[j]["locator"] for j in order[i:i+64]]
                                          for i in record["cohort_starts"]]
            tune(dataset, order, measure, cap, record)
            record["status"] = "complete"
            atomic_json(destinations, record)
            return

        # Warm the shared file pages once and retain CPU batches for the no-loader comparison.
        samples = {i: dataset[i] for i in indices}

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
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tune", action="store_true", help="Calibrate memory and compare multiple held batches")
    modes.add_argument("--pairs-from", type=Path, help="Replay paired reader/attention controls from a calibration JSON")
    modes.add_argument("--verify-fast", action="store_true", help="Check the production profile across three updates")
    run(parser.parse_args())
