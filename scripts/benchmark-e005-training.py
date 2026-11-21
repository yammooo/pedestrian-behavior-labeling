"""Compare E005 execution settings on matched complete-track updates."""

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

from pedestrian_behavior.experiments.e005_cache import atomic_json, collate_e005
from pedestrian_behavior.experiments.e005_run import (AALTO_FAST_PROFILE, AALTO_ROAD_FAST_PROFILE, DEFAULT_ARM, SETTINGS,
                                                     SOURCE, SpatialBatches, assembly, enable_aalto_fast)
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


class OneUpdate:
    def __init__(self, dataset, indices, max_tracks, prefetch, workers, batch_type=SpatialBatches):
        batches = batch_type(dataset, {"max_tracks": max_tracks}, prefetch=prefetch, prefetch_workers=workers)
        self.microbatches, self.stats = batches.microbatches, batches.stats
        self.indices = indices

    def __iter__(self):
        yield self.indices


def timed(prepared, state, dataset, indices, variant, *, prefetch, workers=1, batch_type=SpatialBatches):
    profile, max_tracks = VARIANTS[variant]
    seed_run(0)
    torch.cuda.empty_cache()
    model = build_model(DEFAULT_ARM, profile).to(DEVICE)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    batches = OneUpdate(dataset, indices, max_tracks, prefetch, workers, batch_type)
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
        chunks = _frame_chunks(counts, profile["frame_chunk"], profile["frame_tokens"],
                               sort_by_size=profile.get("sort_frames_by_tokens", False))
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


def check_execution_parity(dataset, indices, state, candidate, *, training):
    batch = collate_e005([dataset[i] for i in indices])
    observations = []
    for profile in (AALTO_FAST_PROFILE, candidate):
        model = build_model(DEFAULT_ARM, profile).to(DEVICE).train(training)
        # cuDNN needs a training-mode LSTM for backward; this one-layer LSTM has no dropout.
        model.temporal.lstm.train()
        model.load_state_dict(state)
        seed_run(0)
        logits = model(batch)
        gradients = torch.autograd.grad(logits.square().mean(), tuple(model.parameters()))
        observations.append((logits.detach().cpu(), tuple(g.detach().cpu() for g in gradients)))
    (reference, reference_grads), (actual, actual_grads) = observations
    torch.testing.assert_close(actual, reference, atol=3e-5, rtol=5e-4)
    for value, expected in zip(actual_grads, reference_grads):
        torch.testing.assert_close(value, expected, atol=3e-5, rtol=5e-4)
    return {"logit_max_abs_diff": (actual-reference).abs().max().item(),
            "gradient_max_abs_diff": max((x-y).abs().max().item() for x, y in zip(actual_grads, reference_grads))}


def road_checks(prepared, dataset, order, state, record, measure, destination, *, tune_chunks):
    if len(order) < 128:
        raise ValueError("The ROAD benchmark requires two disjoint 64-track updates")
    costs = track_costs(dataset)
    VARIANTS["loki-fast"] = AALTO_FAST_PROFILE, 2
    if tune_chunks:
        larger = AALTO_FAST_PROFILE | {"frame_chunk": 16, "frame_tokens": 524_288}
        bucketed = larger | {"sort_frames_by_tokens": True}
        wide_bucketed = bucketed | {"frame_chunk": 32}
        VARIANTS.update({"road-16": (larger, 2), "road-16-bucketed": (bucketed, 2),
                         "road-32-bucketed": (wide_bucketed, 2),
                         "road-32-bucketed-packed": (wide_bucketed | {"pack_lidar_projection": True}, 2)})
        cases = (("loki-fast", 2), ("road-16", 2), ("road-16-bucketed", 2),
                 ("road-32-bucketed", 2), ("road-32-bucketed-packed", 2))
        parity_name, parity_candidate = "size_order_parity", VARIANTS["road-32-bucketed-packed"][0]
    else:
        packed = AALTO_FAST_PROFILE | {"pack_lidar_projection": True}
        wide = packed | {"ordinary_tokens": 4_194_304}
        VARIANTS.update({"road-packed": (packed, 2), "road-wide": (wide, 4)})
        cases = (("loki-fast", 2), ("road-packed", 2), ("road-wide", 2), ("road-packed", 4))
        parity_name, parity_candidate = "packed_projection_parity", packed
    cohorts = (order[:64], order[64:128])
    record["cohort_locators"] = [[dataset.entries[i]["locator"] for i in cohort] for cohort in cohorts]
    measure("loki-fast", dataset, cohorts[0], "road-first-pass", prefetch=True, workers=2)
    parity_indices = [i for i in order if all(any(level) for level in costs[i])][:2]
    record[parity_name] = check_execution_parity(dataset, parity_indices, state, parity_candidate,
                                                 training=not tune_chunks)

    def tokens_per_batch(indices, profile, max_tracks):
        return [padded_count(costs, indices[i:i+max_tracks], profile) for i in range(0, len(indices), max_tracks)]

    scores = {}
    for n, indices in enumerate(cohorts):
        for i in indices:
            dataset[i]
        for variant, workers in cases[::1 if n == 0 else -1]:
            profile, max_tracks = VARIANTS[variant]
            tokens = tokens_per_batch(indices, profile, max_tracks)
            row = measure(variant, dataset, indices, "road-warmed", prefetch=True, workers=workers, cohort=n,
                          padded_tokens=sum(tokens), max_padded_tokens=max(tokens),
                          recompute_microbatches=sum(x > profile["ordinary_tokens"] for x in tokens))
            if row["status"] == "complete":
                scores.setdefault((variant, workers), []).append(row["seconds"])
    eligible = {case: median(times) for case, times in scores.items() if len(times) == len(cohorts)}
    if ("loki-fast", 2) not in eligible:
        raise RuntimeError("The reference LOKI fast profile failed on a ROAD representative update")
    winner = min(eligible, key=eligible.get)
    variant, workers = winner
    profile, max_tracks = VARIANTS[variant]
    dense = sorted(order, key=lambda i: padded_count(costs, [i], profile), reverse=True)[:max_tracks]
    stress = (dense + [i for i in order[128:] + order[:128] if i not in dense])[:64]
    model = build_model(DEFAULT_ARM, profile).to(DEVICE)
    model.load_state_dict(state)
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    torch.cuda.empty_cache()
    seed_run(0)
    groups = [order[i:i+max_tracks] for i in range(0, len(order), max_tracks)]
    counts = [(padded_count(costs, group, profile), group) for group in groups]
    ordinary_count, ordinary_group = max((row for row in counts if row[0] <= profile["ordinary_tokens"]),
                                         default=(0, []))
    updates = [("representative", cohorts[0])]
    if ordinary_group:
        rest = [i for i in order[64:] + order[:64] if i not in ordinary_group]
        updates.append(("ordinary-boundary", (ordinary_group + rest)[:64]))
    updates.append(("dense", stress))
    record["largest_probed_ordinary_group_tokens"] = ordinary_count
    for offset, (label, indices) in enumerate(updates):
        batches = OneUpdate(dataset, indices, max_tracks, True, workers)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        row = prepared["epoch_operation"](model, batches, optimizer, DEVICE,
                                           SETTINGS["clip_norm"], step_offset=402+offset)
        assert row["processed_tracks"] == 64 and row["optimizer_steps"] == 1
        tokens = tokens_per_batch(indices, profile, max_tracks)
        row.update(workload=f"road-persistent-{label}", status="complete", variant=variant, workers=workers,
                   max_padded_tokens=max(tokens), recompute_microbatches=sum(x > profile["ordinary_tokens"] for x in tokens),
                   loader_wait_seconds=batches.stats["wait_seconds"], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   locators=[dataset.entries[i]["locator"] for i in indices])
        record["rows"].append(row)
        atomic_json(destination, record)
        print({key: value for key, value in row.items() if key != "locators"}, flush=True)
    record["winner"] = {"variant": variant, "profile": profile, "max_tracks": max_tracks, "workers": workers,
                        "median_seconds": eligible[winner], "loki_fast_median_seconds": eligible[("loki-fast", 2)],
                        "speedup": eligible[("loki-fast", 2)] / eligible[winner]}


def run(args):
    output = args.output.resolve()
    root = args.outputs_root.resolve()
    allowed = root / "experiments/E005/benchmarks"
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("Use a fresh directory inside outputs/experiments/E005/benchmarks")
    if not torch.cuda.is_available():
        raise ValueError("The E005 benchmark requires CUDA")
    road = args.road or args.road_chunks
    torch.set_num_threads(2)
    seed_run(0)
    free, total = torch.cuda.mem_get_info()
    cap = (enable_aalto_fast(DEFAULT_ARM, "cuda") if args.verify_fast or road or args.joint else
           min(int(total * .89), free - 768 * 1024**2) if args.tune or args.pairs_from else int(total * .75))
    if cap < total * .75:
        raise ValueError("GPU memory is occupied")
    if not (args.verify_fast or road or args.joint):
        torch.cuda.set_per_process_memory_fraction(cap / total)
    output.mkdir(parents=True, exist_ok=False)
    destinations = output / "benchmark.json"
    record = {"status": "incomplete", "variants": VARIANTS, "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "seed": 0, "rows": [],
              "gpu_total_bytes": total, "gpu_free_bytes": free, "allocator_cap_bytes": cap,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if args.joint:
        joint_checks(args, root, cap, record, destinations)
        return
    source = "road-waymo" if road else SOURCE
    prepared = assembly(SimpleNamespace(outputs_root=root, arm=DEFAULT_ARM, source=source, cache=args.cache,
                                        aalto_fast=args.verify_fast or road, allocator_cap_bytes=cap),
                        allow_partial=road)
    try:
        dataset = prepared["dataset"](source, "training")
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

        if road:
            road_checks(prepared, dataset, order, state, record, measure, destinations, tune_chunks=args.road_chunks)
            record["status"] = "complete"
            atomic_json(destinations, record)
            return

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


def joint_checks(args, root, cap, record, destination):
    """Compare complete balanced training updates and confirm the winner under a longer-track workload."""
    prepared = assembly(SimpleNamespace(outputs_root=root, arm=DEFAULT_ARM, source="joint", cache=None,
                                        loki_cache=args.loki_cache, road_cache=args.road_cache, joint_profile=None,
                                        aalto_fast=True, allocator_cap_bytes=cap))
    try:
        dataset = prepared["dataset"]("joint", "training")
        logical = prepared["batches"](dataset, training=True, seed=0)
        cohorts = list(itertools.islice(logical, 2))
        state = build_model(DEFAULT_ARM, ORIGINAL_PROFILE).state_dict()
        base = AALTO_ROAD_FAST_PROFILE
        VARIANTS.update({
            "joint-reference": (base, 2),
            "joint-16-frames": (base | {"frame_chunk": 16, "frame_tokens": 524_288}, 2),
            "joint-32-frames": (base | {"frame_chunk": 32, "frame_tokens": 524_288}, 2),
            "joint-four-tracks": (base | {"frame_chunk": 16, "frame_tokens": 524_288,
                                          "ordinary_tokens": 4_194_304}, 4),
            "joint-eight-tracks": (base | {"frame_chunk": 32, "frame_tokens": 524_288}, 8),
        })
        cases = (("joint-reference", 2), ("joint-16-frames", 2), ("joint-32-frames", 2),
                 ("joint-four-tracks", 2), ("joint-eight-tracks", 2), ("joint-reference", 4),
                 ("joint-four-tracks", 4))
        record["source"] = "joint"
        record["cohort_locators"] = [[dataset.entries[i]["locator"] for i in cohort] for cohort in cohorts]

        def measure(name, workers, indices, workload):
            row = timed(prepared, state, dataset, indices, name, prefetch=True, workers=workers,
                        batch_type=type(logical))
            row.update(workload=workload, workers=workers)
            record["rows"].append(row)
            atomic_json(destination, record)
            print(row, flush=True)
            return row

        measure(*cases[0], cohorts[0], "warmup")
        scores = {}
        for i, cohort in enumerate(cohorts):
            for name, workers in cases[::1 if i == 0 else -1]:
                row = measure(name, workers, cohort, f"matched-{i}")
                if row["status"] == "complete" and row["peak_reserved_bytes"] < cap * .95:
                    scores.setdefault((name, workers), []).append(row["seconds"])
        ranked = sorted((case for case in scores if len(scores[case]) == len(cohorts)),
                        key=lambda case: median(scores[case]))
        # Put long tracks from both sources together and keep a full balanced update.
        long_tracks = []
        for source_indices in dataset.source_indices.values():
            longest = sorted(source_indices, key=lambda i: dataset.entries[i]["slots"], reverse=True)[:4]
            same_source = set(source_indices)
            rest = [i for i in cohorts[1] if i in same_source and i not in longest]
            long_tracks.append((longest + rest)[:32])
        longest_first = list(itertools.chain.from_iterable(zip(*long_tracks, strict=True)))
        for name, workers in ranked:
            row = measure(name, workers, longest_first, "long-tracks")
            if row["status"] == "complete" and row["peak_reserved_bytes"] < cap * .95:
                profile, tracks = VARIANTS[name]
                record["winner"] = {"variant": name, "profile": profile, "max_tracks": tracks,
                                    "workers": workers, "median_seconds": median(scores[name, workers])}
                record["status"] = "complete"
                atomic_json(destination, record)
                print("Selected", record["winner"], flush=True)
                return
        raise RuntimeError("No joint execution layout passed the CUDA calibration")
    finally:
        prepared["close"]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="Explicit E005 cache path")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--tune", action="store_true", help="Calibrate memory and compare multiple held batches")
    modes.add_argument("--pairs-from", type=Path, help="Replay paired reader/attention controls from a calibration JSON")
    modes.add_argument("--verify-fast", action="store_true", help="Check the production profile across three updates")
    modes.add_argument("--road", action="store_true", help="Tune and stress the ROAD source on a bounded cache")
    modes.add_argument("--road-chunks", action="store_true", help="Compare larger and size-ordered ROAD chunks")
    modes.add_argument("--joint", action="store_true", help="Compare execution layouts on balanced ROAD + LOKI updates")
    parser.add_argument("--loki-cache", type=Path, help="Complete LOKI cache for the joint benchmark")
    parser.add_argument("--road-cache", type=Path, help="Complete ROAD cache for the joint benchmark")
    run(parser.parse_args())
