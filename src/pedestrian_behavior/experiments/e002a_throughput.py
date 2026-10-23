"""Bounded complete-update throughput on prepared source caches; no comparison fitting."""

import argparse
import gc
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.accumulation import train_accumulated_epoch
from pedestrian_behavior.data.indexed_points import DTYPES, IndexedShard
from pedestrian_behavior.data.prepared_rgb import PreparedRGB
from contextlib import ExitStack, contextmanager
from unittest.mock import patch
from pedestrian_behavior.evaluation import track_cross_entropy
from pedestrian_behavior.training import seed_run
from .e002a_run import add_arguments, assembly, SETTINGS
from .run import write_json


def cases(dataset):
    """Input-only selection: seeded shuffle, highest point costs, longest tracks."""
    n = len(dataset)
    if n < 1:
        raise ValueError("Empty throughput population")
    lengths = [e["slots"] for e in dataset.entries]
    costs = [dataset.cost(i) for i in range(n)]
    if any(c is None for c in costs):
        raise ValueError("Prepared input costs must all be known")
    size = min(64, n)
    return {"shuffled": torch.randperm(n, generator=torch.Generator().manual_seed(0))[:size].tolist(),
            "dense": sorted(range(n), key=lambda i: (-costs[i], i))[:size],
            "long": sorted(range(n), key=lambda i: (-lengths[i], -costs[i], i))[:size]}


class OneUpdate:
    def __init__(self, logical, indices):
        self.logical, self.indices = logical, indices
        self.loading_seconds = 0.

    def __iter__(self):
        yield self.indices

    def microbatches(self, indices):
        iterator = iter(self.logical.microbatches(indices))
        while True:
            begin = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                self.loading_seconds += time.perf_counter() - begin
                return
            self.loading_seconds += time.perf_counter() - begin
            yield batch


@contextmanager
def breakdown_timers(enabled):
    totals = {}
    with ExitStack() as stack:
        if enabled:
            for cls, name, label in ((IndexedShard, "take", "point_read_verify_gather_seconds"),
                                     (IndexedShard, "crop", "point_crop_total_seconds"),
                                     (PreparedRGB, "get", "rgb_read_seconds")):
                original = getattr(cls, name)
                def timed(*args, _original=original, _label=label, **kwargs):
                    begin = time.perf_counter()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        totals[_label] = totals.get(_label, 0.) + time.perf_counter()-begin
                stack.enter_context(patch.object(cls, name, timed))
        yield totals


def advisory_cold(dataset, indices):
    """Evict only requested point-shard pages; no global OS cache controls."""
    supply = dataset.supply
    points = supply.points if supply is not None else None
    if points is not None:
        points.close()
        groups = {dataset.entries[i]["group"] for i in indices}
        if hasattr(os, "posix_fadvise"):
            for group in groups:
                root = points.root / points.record["scenes"][group]["path"]
                for name in DTYPES:
                    with (root / (name + ".bin")).open("rb") as stream:
                        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def forecast(rows, training_tracks, validation_tracks, overhead_seconds=60.):
    """Stress envelope, not a statistical interval or a measured full epoch."""
    rates = {phase: max(r["seconds"] / r["tracks"] for r in rows if r["operation"] == phase)
             for phase in ("training", "validation")}
    seconds = training_tracks*rates["training"] + validation_tracks*rates["validation"] + overhead_seconds
    return {"training_tracks": training_tracks, "validation_tracks": validation_tracks,
            "max_measured_seconds_per_track": rates, "assumed_epoch_overhead_seconds": overhead_seconds,
            "stress_envelope_seconds": seconds, "under_30_minutes": seconds < 1800,
            "interpretation": "Maximum sampled cold/repeat random/dense/long seconds per track applied to every track; "
                              "not a measured full epoch, guaranteed worst case or physical-cold I/O guarantee"}


def run(args):
    if args.smoke:
        raise ValueError("Throughput uses full source catalogs, not smoke selection")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"status": "incomplete", "scope": "bounded source-only operational timing; discard weights; no W&B/test scoring",
              "source": args.source, "arm": args.arm, "rows": [], "failures": [],
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    prepared = model = optimizer = None
    try:
        seed_run(0)
        prepared = assembly(args)
        record["config"] = prepared["config"]
        populations = {p: prepared["dataset"](args.source, p) for p in ("training", "validation")}
        for operation, dataset in populations.items():
            if args.breakdown and operation != "training":
                continue
            chosen = cases(dataset)
            if args.breakdown:
                chosen = {"shuffled": chosen["shuffled"]}
            record.setdefault("selections", {})[operation] = {k: [dataset.entries[i]["locator"] for i in v]
                                                           for k,v in chosen.items()}
            logical = prepared["batches"](dataset, training=True, seed=0)
            for name, indices in chosen.items():
                for phase in ("advisory-cold", "repeat-files"):
                    seed_run(0)
                    model = prepared["build_model"]().to(args.device)
                    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
                    torch.cuda.reset_peak_memory_stats()
                    before = prepared["runtime_snapshot"]()
                    if phase == "advisory-cold":
                        advisory_cold(dataset, indices)
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    if operation == "training":
                        timed_update = OneUpdate(logical, indices)
                        with breakdown_timers(args.breakdown) as detail:
                            result = train_accumulated_epoch(model, timed_update, optimizer,
                                torch.device(args.device), 1., peak_lr=.0003,
                                warmup_steps=prepared["config"]["warmup_updates"])
                        tracks, tokens = result["processed_tracks"], result["point_tokens"]
                        physical = result["physical_microbatches"]
                    else:
                        tracks = tokens = physical = 0
                        model.eval()
                        with torch.no_grad():
                            for batch in logical.microbatches(indices):
                                logits = model(batch)
                                losses = track_cross_entropy(logits, batch["targets"].to(args.device),
                                    batch["gt_valid"].to(args.device), batch["padding_mask"].to(args.device))
                                if not torch.isfinite(losses).all():
                                    raise ValueError("Nonfinite throughput validation")
                                tracks += len(losses)
                                tokens += sum(len(f) for f in batch.get("points", ()))
                                physical += 1
                                del batch, logits, losses
                    torch.cuda.synchronize()
                    row = {"operation": operation, "case": name, "phase": phase, "seconds": time.perf_counter()-begin,
                           "tracks": tracks, "tokens": tokens, "physical_microbatches": physical,
                           "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                           "runtime_before": before, "runtime_after": prepared["runtime_snapshot"]()}
                    if operation == "training":
                        row["input_preparation_seconds"] = timed_update.loading_seconds
                        row["model_transfer_step_seconds"] = row["seconds"] - timed_update.loading_seconds
                        row["input_detail"] = dict(detail)
                    record["rows"].append(row)
                    write_json(args.output/"throughput.json", record)
                    print(args.source, args.arm, operation, name, phase, tracks, round(row["seconds"],2), flush=True)
                    del model, optimizer
                    model = optimizer = None
                    gc.collect(); torch.cuda.empty_cache()
        if not args.breakdown:
            record["forecast"] = forecast(record["rows"],len(populations["training"]),len(populations["validation"]))
        record["breakdown_only"] = args.breakdown
        record["status"] = "complete"
        return record
    except BaseException:
        record["status"] = "failed"; record["failures"].append(traceback.format_exc()); raise
    finally:
        try:
            if prepared is not None:
                prepared["close"]()
        except BaseException:
            record["status"] = "failed"
            record["failures"].append(traceback.format_exc())
            raise
        finally:
            write_json(args.output/"throughput.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--breakdown", action="store_true", help="Only the same shuffled training update twice, with load/model timers")
    args = parser.parse_args()
    if args.device != "cuda":
        raise ValueError("GPU throughput readiness requires CUDA")
    run(args)
