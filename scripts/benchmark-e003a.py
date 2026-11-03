#!/usr/bin/env python3
"""Matched warm-cache update timings on three frozen E003a training fixtures."""

import json
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import torch

from pedestrian_behavior.experiments import e003a_run
from pedestrian_behavior.experiments.e003a import FeatureDataset
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.accumulation import train_accumulated_epoch
from pedestrian_behavior.training import seed_run
from pedestrian_behavior.data.preparation import checksum


class FixedUpdate:
    def __init__(self, logical, indices):
        if len(indices) != 3 or not callable(getattr(logical, "microbatches", None)):
            raise ValueError("FixedUpdate needs three indices and production microbatch packing")
        self.logical, self.indices = logical, indices

    def __iter__(self):
        self.logical.iteration += 1
        yield self.indices

    def microbatches(self, indices):
        return self.logical.microbatches(indices)


def _cache_delta(before, after):
    result = {}
    for cache in ("rgb", "points"):
        left, right = before.get(cache, {}), after.get(cache, {})
        result[cache] = {key: right[key] - value for key, value in left.items()
                         if isinstance(value, (int, float)) and not isinstance(value, bool)
                         and isinstance(right.get(key), (int, float)) and not isinstance(right[key], bool)}
    return result


def _cache_snapshot(snapshot):
    return {cache: snapshot[cache] for cache in ("rgb", "points") if cache in snapshot}


def _update(prepared, data, indices, device):
    logical = prepared["batches"](data, training=True, seed=0)
    if data.mode == "training":
        data.epoch = lambda: 1
    seed_run(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    setup_start = time.perf_counter()
    model = prepared["build_model"]().to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  **e003a_run.SETTINGS["optimizer"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    before = prepared["runtime_snapshot"]()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    row = train_accumulated_epoch(model, FixedUpdate(logical, indices), optimizer, device,
        e003a_run.SETTINGS["clip_norm"], step_offset=0, peak_lr=.0003,
        warmup_steps=prepared["config"]["warmup_updates"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    after = prepared["runtime_snapshot"]()
    result = {"model_optimizer_setup_seconds": setup_seconds, "update_wall_seconds": elapsed,
        "update_seconds": row["seconds"], "optimizer_steps": row["optimizer_steps"],
        "processed_tracks": row["processed_tracks"], "context_slots": row["context_slots"],
        "padding_slots": row["padding_slots"], "batch_slots": row["batch_slots"],
        "padding_fraction": row["padding_fraction"], "physical_microbatches": row["physical_microbatches"],
        "point_tokens": row["point_tokens"], "cache_before": _cache_snapshot(before),
        "cache_after": _cache_snapshot(after), "cache_delta": _cache_delta(before, after),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args):
    if args.smoke or args.device not in ("cpu", "cuda"):
        raise ValueError("Benchmark needs full prepared source populations and an explicit CPU/CUDA device")
    output, outputs_root = args.output.resolve(), args.outputs_root.resolve()
    if not output.is_relative_to(outputs_root / "experiments/E003a") or output.exists():
        raise ValueError("Choose a fresh benchmark directory under outputs/experiments/E003a")
    torch.set_num_threads(2)
    fixtures = json.loads(args.fixtures.read_text())
    fixture_rows = {row["name"]: row for row in fixtures["tracks"]}
    names = [f"{args.source}-{category}" for category in ("short", "typical", "long")]
    if any(name not in fixture_rows or fixture_rows[name].get("split") != "training"
           or fixture_rows[name].get("dataset") != args.source for name in names):
        raise ValueError("Execution-probe-04 must contain the three named source training fixtures")
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.perf_counter()
    prepared = e003a_run.assembly(args)
    assembly_seconds = time.perf_counter() - started
    try:
        masked = prepared["dataset"](args.source, "training")
        base = masked.base
        catalog = base.supply.catalogs[args.source]
        fixture_ids = [fixture_rows[name]["locator"] for name in names]
        available = {entry["locator"]: entry for entry in base.baseline.entries}
        if not set(fixture_ids).issubset(available):
            raise ValueError("Prepared training population does not contain all execution fixtures")
        base.baseline.entries = [available[locator] for locator in fixture_ids]
        base.entries = [catalog.tracks[locator] for locator in fixture_ids]
        if [entry["slots"] for entry in base.entries] != [fixture_rows[name]["slots"] for name in names]:
            raise ValueError("Fixture lengths differ from the prepared frozen tracks")
        natural = FeatureDataset(base, args.source, e003a_run.ARM, mode="natural")
        indices = list(range(3))
        dataset_by_view = {"natural": natural, "e003a-mask": masked}
        profile = prepared["config"]["profile"]
        point_counts = base.supply.points.counts
        natural_tokens = {locator: int(point_counts[locator].sum()) for locator in fixture_ids}

        record = {"status": "incomplete", "experiment": "E003a", "source": args.source,
            "device": str(device), "profile": profile, "assembly_seconds": assembly_seconds,
            "profile_sha256": checksum(args.profile), "fixture_sha256": checksum(args.fixtures),
            "fixtures": [{"name": name, "locator": locator, "slots": fixture_rows[name]["slots"],
                          "natural_point_tokens": natural_tokens[locator]}
                         for name, locator in zip(names, fixture_ids)],
            "scope": "one warmup and two timed accumulated updates per view on three frozen training fixtures; fresh seed-0 model/optimizer each update, weights discarded; no validation/test inference, W&B, or full training",
            "views": {}}
        for view, data in dataset_by_view.items():
            record["views"][view] = {"warmup": _update(prepared, data, indices, device), "measured": []}
        for repetition in range(2):
            for view, data in dataset_by_view.items():
                row = _update(prepared, data, indices, device)
                row["repetition"] = repetition + 1
                record["views"][view]["measured"].append(row)

        summary, details = masked.finish_epoch()
        if summary["tracks"] != len(fixture_ids):
            raise ValueError("Masked benchmark update did not visit each fixture exactly once")
        original_gt = {entry["locator"]: entry["gt_frames"] for entry in base.baseline.entries}
        if any(detail["posthoc_gt"]["gt_slots"] != original_gt[detail["locator"]] for detail in details):
            raise ValueError("Masking changed the fixture GT coverage")
        lookup = {detail["locator"]: detail for detail in details}
        for index, locator in enumerate(fixture_ids):
            natural_sample, masked_sample = natural[index], masked[index]
            if (not torch.equal(natural_sample["targets"], masked_sample["targets"])
                    or not torch.equal(natural_sample["gt_valid"], masked_sample["gt_valid"])):
                raise ValueError("Natural and masked views changed fixture GT")
        record["mask_summary"] = summary
        record["mask_details"] = lookup
        record["point_tokens"] = {"natural": sum(natural_tokens.values()),
            "effective": sum(int(point_counts[locator][np.fromiter(
                (value == "1" for value in lookup[locator]["effective"]["lidar"]), dtype=bool)].sum())
                for locator in fixture_ids)}
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        record["status"] = "complete"
        write_json(output / "benchmark.json", record)
        return record
    except BaseException as error:
        write_json(output / "benchmark.json", {"status": "failed", "source": args.source,
            "assembly_seconds": assembly_seconds, "failure": f"{type(error).__name__}: {error}"})
        raise
    finally:
        try:
            prepared["close"]()
        except BaseException as error:
            path = output / "benchmark.json"
            if path.exists():
                failed = json.loads(path.read_text())
                failed.update(status="failed", cleanup_failure=f"{type(error).__name__}: {error}")
                write_json(path, failed)
            raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    e003a_run.add_arguments(parser)
    parser.add_argument("--fixtures", type=Path,
        default=Path("outputs/experiments/E002a/execution-probe-04/execution.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
