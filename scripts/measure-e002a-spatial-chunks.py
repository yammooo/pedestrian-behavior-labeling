"""Matched warmed-update sweep for E002a spatial chunk sizes and CPU checks."""

import argparse
import contextlib
import gc
import json
import os
import subprocess
import time
import traceback
from unittest.mock import patch

import torch

from pedestrian_behavior.accumulation import train_accumulated_epoch
from pedestrian_behavior.experiments.e002a_run import add_arguments, assembly, SETTINGS
from pedestrian_behavior.experiments.e002a_throughput import TimedBatches, cases
from pedestrian_behavior.models import multimodal
from pedestrian_behavior.training import seed_run
from pedestrian_behavior.experiments.run import write_json


def chunk_stats(batch, profile):
    chunks = list(multimodal.frame_chunks(batch["points"], profile["frame_chunk"], profile["frame_tokens"]))
    padded = [len(chunk) * max(len(frame) for _, frame in chunk) for chunk in chunks]
    return {"chunks": len(chunks), "padded_tokens": sum(padded),
            "max_chunk_padded_tokens": max(padded, default=0),
            "mode": "ordinary" if sum(padded) <= profile["ordinary_tokens"] else "recompute"}


def preflight(prepared, dataset, indices, profile, *, include_locators=False):
    logical = prepared["batches"](dataset, training=True, seed=0)
    rows = []
    for batch in logical.microbatches(indices):
        item = chunk_stats(batch, profile)
        if item["max_chunk_padded_tokens"] > profile["frame_tokens"]:
            raise ValueError("Chunk exceeds the unchanged padded-token ceiling")
        item["tokens"] = sum(map(len, batch["points"]))
        if include_locators:
            item["locators"] = batch["locators"]
        rows.append(item)
    return rows


def one_update(prepared, dataset, indices, profile, finite_check, phase, trace_path=None):
    seed_run(0)
    model = prepared["build_model"]().to("cuda")
    model.profile.update(profile)
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    logical = prepared["batches"](dataset, training=True, seed=0)
    batches = TimedBatches(logical, indices)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    begin = time.perf_counter()
    check = (contextlib.nullcontext() if finite_check == "numpy" else
             patch.object(multimodal, "_all_finite_point_tokens", lambda value: torch.isfinite(value).all()))
    try:
        with check:
            if trace_path is None:
                result = train_accumulated_epoch(model, batches, optimizer, torch.device("cuda"), 1.,
                    peak_lr=.0003, warmup_steps=prepared["config"]["warmup_updates"])
            else:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=False, with_stack=False, profile_memory=False) as profiler:
                    result = train_accumulated_epoch(model, batches, optimizer, torch.device("cuda"), 1.,
                        peak_lr=.0003, warmup_steps=prepared["config"]["warmup_updates"])
                profiler.export_chrome_trace(str(trace_path))
        torch.cuda.synchronize()
        row = {"phase": phase, "finite_check": finite_check, "frame_chunk": profile["frame_chunk"],
               "seconds": time.perf_counter() - begin, "input_wait_seconds": batches.loading_seconds,
               "processed_tracks": result["processed_tracks"], "point_tokens": result["point_tokens"],
               "physical_microbatches": result["physical_microbatches"], "optimizer_steps": result["optimizer_steps"],
               "training_ce": result["training_ce"], "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_cuda_delta_bytes": torch.cuda.max_memory_allocated() - before,
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
               "loader": dict(logical.stats)}
        del model, optimizer, batches, logical
        gc.collect()
        torch.cuda.empty_cache()
        return row
    except BaseException:
        del model, optimizer, batches, logical
        gc.collect()
        torch.cuda.empty_cache()
        raise


def run(args):
    if args.device != "cuda" or args.arm != "g-rgb-dditr" or args.smoke:
        raise ValueError("Sweep requires full-source ROAD/LOKI LiDAR arm on CUDA")
    if not 1 <= args.repeats <= 5:
        raise ValueError("Use one to five bounded measured repeats")
    if not args.frame_chunks or any(size < 1 for size in args.frame_chunks):
        raise ValueError("Frame chunk candidates must be positive")
    if args.ordinary_token_threshold is not None and args.ordinary_token_threshold < 1:
        raise ValueError("Ordinary-token override must be positive")
    if args.trace_one_update and (not args.largest_microbatch_only or not args.async_spatial_transfer
                                  or args.repeats != 1):
        raise ValueError("Trace requires one async largest-microbatch measurement")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    record = {"status": "incomplete", "source": args.source, "arm": args.arm, "device": args.device,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "workers": args.point_workers, "repeats": args.repeats, "seed": 0,
              "finite_check_comparison": ["torch", "numpy"], "rows": [], "failures": [],
              "async_spatial_transfer": args.async_spatial_transfer,
              "cpu_thread_limits": {k: os.environ.get(k) for k in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    prepared = None
    try:
        seed_run(0)
        prepared = assembly(args)
        dataset = prepared["dataset"](args.source, "training")
        indices = cases(dataset)["shuffled"]
        if len(indices) != 64 and not args.largest_microbatch_only:
            raise ValueError("Expected the representative shuffled64-track update")
        base_profile = dict(prepared["config"]["profile"])
        frame_chunks = args.frame_chunks
        variants = ([(2, "torch"), (2, "numpy")] if 2 in frame_chunks else []) + [
            (size, "numpy") for size in frame_chunks if size != 2]
        record["finite_check_comparison"] = ["torch", "numpy"] if 2 in frame_chunks else ["numpy"]
        selected_indices = indices
        if args.largest_microbatch_only:
            selection_profile = base_profile | {"frame_chunk": max(frame_chunks)}
            groups = preflight(prepared, dataset, indices, selection_profile, include_locators=True)
            largest = max(groups, key=lambda row: row["padded_tokens"])
            by_locator = {entry["locator"]: i for i, entry in enumerate(dataset.entries)}
            selected_indices = [by_locator[locator] for locator in largest["locators"]]
            record["largest_microbatch_selection"] = {
                "selection_chunk": max(frame_chunks), "locators": largest["locators"],
                "tokens": largest["tokens"], "padded_tokens": largest["padded_tokens"],
                "tracks": len(selected_indices),
            }
            if args.ordinary_token_threshold is not None:
                calibration = json.loads(args.profile.read_text())
                reference_workload = max(
                    (row for row in calibration["workloads"]
                     if row.get("arm") == "g-rgb-dditr" and row.get("mode") == "ordinary"
                     and row.get("frame_chunk") == base_profile["frame_chunk"]
                     and row.get("peak_allocated") is not None),
                    key=lambda row: row["padded_tokens"],
                )
                scale = max(largest["tokens"] / reference_workload["tokens"],
                            largest["padded_tokens"] / reference_workload["padded_tokens"])
                estimate = int(reference_workload["peak_allocated"] * scale * 1.25)
                ceiling = calibration["allocated_ceiling"]
                record["ordinary_probe_memory_estimate"] = {
                    "reference_workload": reference_workload["name"],
                    "reference_peak_allocated_bytes": reference_workload["peak_allocated"],
                    "reference_tokens": reference_workload["tokens"],
                    "reference_padded_tokens": reference_workload["padded_tokens"],
                    "scale": scale, "safety_factor": 1.25,
                    "estimated_peak_allocated_bytes": estimate,
                    "calibrated_78pct_ceiling_bytes": ceiling,
                }
                if estimate >= ceiling * 0.70:
                    raise ValueError("Conservative ordinary-mode estimate exceeds 70% of the calibrated ceiling")
        record.update(scope="one warmed pass plus matched shuffled64 updates; weights discarded; no W&B/validation/tests",
                      model_config=prepared["config"],
                      selections=[dataset.entries[i]["locator"] for i in indices],
                      frame_token_ceiling=base_profile["frame_tokens"],
                      ordinary_token_threshold=base_profile["ordinary_tokens"])
        if args.largest_microbatch_only:
            record["scope"] = "one warmed pass plus matched largest physical microbatch only; weights discarded"
        if args.ordinary_token_threshold is not None:
            record["ordinary_token_threshold_override"] = args.ordinary_token_threshold
        reference = None
        for size, finite_check in variants:
            profile = base_profile | {"frame_chunk": size}
            if args.ordinary_token_threshold is not None:
                profile["ordinary_tokens"] = args.ordinary_token_threshold
            if args.async_spatial_transfer:
                profile["async_spatial_transfer"] = True
            candidate = "all-under-token-ceiling" if size > 1000 else size
            plan = preflight(prepared, dataset, selected_indices, profile)
            if args.largest_microbatch_only and len(plan) != 1:
                raise ValueError("Selected tracks did not reproduce one physical microbatch")
            totals = {"physical_microbatches": len(plan),
                      "point_tokens": sum(row["tokens"] for row in plan),
                      "total_spatial_chunks": sum(row["chunks"] for row in plan),
                      "total_padded_tokens": sum(row["padded_tokens"] for row in plan),
                      "max_chunk_padded_tokens": max((row["max_chunk_padded_tokens"] for row in plan), default=0),
                      "predicted_modes": {mode: sum(row["mode"] == mode for row in plan) for mode in ("ordinary", "recompute")},
                      "per_microbatch": plan}
            if totals["max_chunk_padded_tokens"] > base_profile["frame_tokens"]:
                raise ValueError("Preflight exceeded the unchanged point-token ceiling")
            if args.largest_microbatch_only and args.ordinary_token_threshold is not None and totals["predicted_modes"]["recompute"]:
                raise ValueError("Ordinary-token override does not cover the selected microbatch")
            record.setdefault("variants", []).append({"frame_chunk": candidate, "finite_check": finite_check,
                                                        "preflight": totals})
            warm = one_update(prepared, dataset, selected_indices, profile, finite_check, "warmup")
            record["variants"][-1]["warmup"] = warm
            write_json(args.output / "throughput.json", record)
            rows = []
            for repeat in range(args.repeats):
                trace_path = (args.output / "async-single-batch-trace.json"
                              if args.trace_one_update and repeat == 0 else None)
                row = one_update(prepared, dataset, selected_indices, profile, finite_check,
                                 f"warm-{repeat+1}", trace_path=trace_path)
                row["frame_chunk_variant"] = candidate
                rows.append(row)
                record["rows"].append(row)
                write_json(args.output / "throughput.json", record)
                print(args.source, finite_check, candidate, row["phase"], round(row["seconds"], 3),
                      "wait", round(row["input_wait_seconds"], 3), "peak_GiB",
                      round(row["peak_cuda_allocated_bytes"] / 2**30, 2), flush=True)
            if any(r["point_tokens"] != totals["point_tokens"] or r["physical_microbatches"] != totals["physical_microbatches"]
                   for r in rows):
                raise ValueError("Chunk/check variant changed complete-track membership or packing")
            if any(r["training_ce"] != warm["training_ce"] for r in rows):
                raise ValueError("Fixed-seed repeated update changed loss within one variant")
            if reference is None:
                reference = {"tokens": rows[0]["point_tokens"], "physical": rows[0]["physical_microbatches"]}
            if any(r["point_tokens"] != reference["tokens"] or r["physical_microbatches"] != reference["physical"]
                   for r in rows):
                raise ValueError("Variant changed the matched shuffled64 update")
            record["variants"][-1]["median_seconds"] = sorted(r["seconds"] for r in rows)[len(rows)//2]
            if size == 2 and finite_check == "numpy":
                torch_rows = [r for r in record["rows"] if r["frame_chunk_variant"] == 2 and r["finite_check"] == "torch"]
                if [r["training_ce"] for r in torch_rows] != [r["training_ce"] for r in rows]:
                    raise ValueError("CPU finite-check implementation changed seeded training CE")
        record.update(status="complete", runtime_after=prepared["runtime_snapshot"](),
                      interpretation="Input/packing/token counts are matched. Different frame chunks can change dropout RNG use and CE; compare timing and memory, not cross-chunk dropout losses.")
        write_json(args.output / "throughput.json", record)
    except BaseException:
        record["status"] = "failed"
        record["failures"].append(traceback.format_exc())
        write_json(args.output / "throughput.json", record)
        raise
    finally:
        if prepared is not None:
            prepared["close"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--frame-chunks", type=int, nargs="+", default=[2, 4, 8, 16, 2**31 - 1])
    parser.add_argument("--ordinary-token-threshold", type=int,
                        help="Temporary measured threshold override; never updates a saved profile")
    parser.add_argument("--largest-microbatch-only", action="store_true",
                        help="Restrict the probe to the matched physical batch with greatest padded spatial cost")
    parser.add_argument("--async-spatial-transfer", action="store_true",
                        help="Prototype pinned host staging and an explicit copy stream in ordinary CUDA mode")
    parser.add_argument("--trace-one-update", action="store_true",
                        help="Export a bounded CPU/CUDA trace for the one async largest-microbatch update")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
