"""Bounded CPU/CUDA trace of one warmed, seeded E002a ROAD update."""

import argparse
from collections import defaultdict
from contextvars import ContextVar
import json
import os
import subprocess
import time
from unittest.mock import patch

import torch

from pedestrian_behavior.accumulation import train_accumulated_epoch
from pedestrian_behavior.experiments.e002a_run import add_arguments, assembly, SETTINGS
from pedestrian_behavior.experiments.e002a_throughput import cases, TimedBatches
from pedestrian_behavior.training import seed_run


IN_SPATIAL = ContextVar("e002a_in_spatial", default=False)


def timed_scopes(model, optimizer):
    original_forward = model.forward

    def forward(batch, *args, **kwargs):
        with torch.profiler.record_function("e002a.model_forward"):
            return original_forward(batch, *args, **kwargs)

    model.forward = forward
    original_spatial = model.spatial.forward

    def spatial(inputs, *args, **kwargs):
        token = IN_SPATIAL.set(True)
        try:
            with torch.profiler.record_function("e002a.spatial_forward"):
                return original_spatial(inputs, *args, **kwargs)
        finally:
            IN_SPATIAL.reset(token)

    model.spatial.forward = spatial
    original_step = optimizer.step

    def step(*args, **kwargs):
        with torch.profiler.record_function("e002a.optimizer_step"):
            return original_step(*args, **kwargs)

    optimizer.step = step


def trace_summary(profiler, output):
    trace_path = output / "trace.json"
    profiler.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    categories = defaultdict(int)
    kernels, copies = [], []
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        categories[category] += 1
        if category == "kernel":
            kernels.append(event)
        elif category == "gpu_memcpy":
            copies.append(event)

    def interval_union(events):
        intervals = sorted((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0))) for e in events)
        end = total = 0.
        for begin, stop in intervals:
            total += max(0., stop - max(begin, end))
            end = max(end, stop)
        return total

    def row(event):
        return {"name": event.key, "calls": event.count,
                "self_cpu_ms": event.self_cpu_time_total / 1000,
                "self_cuda_ms": getattr(event, "self_cuda_time_total", getattr(event, "self_device_time_total", 0.)) / 1000}

    averages = list(profiler.key_averages())
    top_cpu = [row(e) for e in sorted(averages, key=lambda e: e.self_cpu_time_total, reverse=True)[:30]]
    focused_terms = ("aten::cat", "aten::to", "aten::_to_copy", "aten::copy_", "aten::item",
                     "_local_scalar_dense", "is_nonzero", "aten::equal", "aten::all", "aten::isfinite")
    focused = [row(e) for e in averages if any(term in e.key for term in focused_terms)]
    scopes = defaultdict(lambda: {"calls": 0, "cpu_total_ms": 0., "cpu_self_ms": 0., "cuda_total_ms": 0.})
    for event in profiler.events():
        if not event.name.startswith("e002a."):
            continue
        value = scopes[event.name]
        value["calls"] += 1
        value["cpu_total_ms"] += event.cpu_time_total / 1000
        value["cpu_self_ms"] += event.self_cpu_time_total / 1000
        value["cuda_total_ms"] += getattr(event, "cuda_time_total", getattr(event, "device_time_total", 0.)) / 1000

    h2d = [e for e in copies if any(tag in (e.get("name", "") + json.dumps(e.get("args", {}))).lower()
                                     for tag in ("htod", "h2d", "host to device", "hostto"))]
    return {"trace": str(trace_path), "trace_event_categories": dict(categories),
            "cuda_kernel_calls": len(kernels), "cuda_kernel_elapsed_ms_sum_may_overlap": sum(float(e.get("dur", 0)) for e in kernels) / 1000,
            "cuda_kernel_active_union_ms": interval_union(kernels) / 1000,
            "gpu_memcpy_calls": len(copies), "h2d_calls": len(h2d),
            "h2d_bytes_reported": sum(int(e.get("args", {}).get("bytes", 0) or 0) for e in h2d),
            "h2d_elapsed_ms_sum_may_overlap": sum(float(e.get("dur", 0)) for e in h2d) / 1000,
            "h2d_active_union_ms": interval_union(h2d) / 1000,
            "gpu_memcpy_event_samples": [{"name": e.get("name"), "duration_us": e.get("dur"),
                                           "args": e.get("args", {})} for e in copies[:5]],
            "scopes_nested_do_not_sum": dict(scopes), "top_cpu_self_ops": top_cpu,
            "focused_ops": focused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    if (args.source, args.arm, args.device) != ("road-waymo", "g-rgb-dditr", "cuda") or args.smoke:
        raise ValueError("Profile requires full-source ROAD g-rgb-dditr on CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    record = {"status": "incomplete", "scope": "one warmed plus one profiled shuffled64-track update; discard weights",
              "source": args.source, "arm": args.arm, "workers": args.point_workers,
              "cpu_thread_limits": {k: os.environ.get(k) for k in
                                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
    prepared = None
    try:
        seed_run(0)
        prepared = assembly(args)
        dataset = prepared["dataset"](args.source, "training")
        indices = cases(dataset)["shuffled"]
        if len(indices) != 64:
            raise ValueError("Expected the representative complete64-track update")
        logical = prepared["batches"](dataset, training=True, seed=0)
        batches = TimedBatches(logical, indices)
        record.update(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      selections=[dataset.entries[i]["locator"] for i in indices],
                      runtime_before=prepared["runtime_snapshot"]())

        def warmup_update():
            seed_run(0)
            model = prepared["build_model"]().to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = train_accumulated_epoch(model, batches, optimizer, torch.device("cuda"), 1.,
                peak_lr=.0003, warmup_steps=prepared["config"]["warmup_updates"])
            torch.cuda.synchronize()
            row = {"seconds": time.perf_counter() - begin, "input_wait_seconds": batches.loading_seconds,
                   "tracks": result["processed_tracks"], "tokens": result["point_tokens"],
                   "physical_microbatches": result["physical_microbatches"], "training_ce": result["training_ce"]}
            del optimizer, model
            return row, result

        record["warmup"] = warmup_update()[0]
        torch.cuda.empty_cache()
        batches.loading_seconds = 0.
        # The profiled pass resets seed/model/optimizer while retaining verified mmap/cache state.
        profiler, row, backward_calls = profiled_run(prepared, batches)
        record["profiled_update"] = row
        record["backward_calls"] = backward_calls
        record["profile"] = trace_summary(profiler, args.output)
        if row["tracks"] != 64 or row["tokens"] != record["warmup"]["tokens"] or row["training_ce"] != record["warmup"]["training_ce"]:
            raise ValueError("Warm and profiled updates changed inputs or seeded loss")
        record.update(status="complete", runtime_after=prepared["runtime_snapshot"](),
                      interpretation="Profiler wall time is instrumented; CUDA busy uses a union of kernel intervals. Nested scope times are not additive.")
        (args.output / "profile.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    finally:
        if prepared is not None:
            prepared["close"]()


def profiled_run(prepared, batches):
    seed_run(0)
    model = prepared["build_model"]().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    timed_scopes(model, optimizer)
    original_microbatches = TimedBatches.microbatches

    def input_scoped(self, selected):
        iterator = original_microbatches(self, selected)
        try:
            while True:
                with torch.profiler.record_function("e002a.input_next"):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        return
                yield batch
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()

    from pedestrian_behavior.accumulation import track_cross_entropy as original_loss

    def loss_scoped(*args, **kwargs):
        with torch.profiler.record_function("e002a.loss_checks"):
            return original_loss(*args, **kwargs)

    original_backward = torch.autograd.backward
    backward_calls = 0

    def backward_scoped(*args, **kwargs):
        nonlocal backward_calls
        backward_calls += 1
        with torch.profiler.record_function("e002a.backward"):
            return original_backward(*args, **kwargs)

    original_cat = torch.cat

    def cat_scoped(*args, **kwargs):
        if IN_SPATIAL.get():
            with torch.profiler.record_function("e002a.spatial_cat"):
                return original_cat(*args, **kwargs)
        return original_cat(*args, **kwargs)

    begin = time.perf_counter()
    kwargs = {"activities": [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
              "record_shapes": False, "profile_memory": False, "with_stack": False}
    torch.cuda.synchronize()
    with patch.object(TimedBatches, "microbatches", input_scoped), \
         patch("pedestrian_behavior.accumulation.track_cross_entropy", loss_scoped), \
         patch("torch.autograd.backward", backward_scoped), \
         patch.object(torch, "cat", cat_scoped), \
         torch.profiler.profile(**kwargs) as profiler:
        with torch.profiler.record_function("e002a.optimizer_update"):
            result = train_accumulated_epoch(model, batches, optimizer, torch.device("cuda"), 1.,
                peak_lr=.0003, warmup_steps=prepared["config"]["warmup_updates"])
        torch.cuda.synchronize()
    row = {"seconds": time.perf_counter() - begin, "input_wait_seconds": batches.loading_seconds,
           "tracks": result["processed_tracks"], "tokens": result["point_tokens"],
           "physical_microbatches": result["physical_microbatches"], "training_ce": result["training_ce"]}
    del optimizer, model
    return profiler, row, backward_calls


if __name__ == "__main__":
    main()
