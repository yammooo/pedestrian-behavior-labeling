"""Bounded real-model calibration for the independent E002b geometry/RGB path."""

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.multimodal import collate_multimodal
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import track_cross_entropy
from pedestrian_behavior.training import seed_run
from . import e002a_calibrate, e002b, e002b_run
from .run import write_json


ARMS = ("g", "g-rgb")


def _model_sample(sample, arm):
    result = {key: sample[key] for key in ("geometry", "targets", "gt_valid", "locator", "archive")}
    indices = [0] if arm == "g" else [0, 1]
    result["flags"] = sample["flags"][:, indices].clone()
    if arm == "g-rgb":
        result["rgb"] = sample["rgb"]
    return result


def _stress_sample(slots, arm):
    torch.manual_seed(73)
    result = {"geometry": torch.randn(slots, 20), "flags": torch.ones(slots, 1 if arm == "g" else 2),
              "targets": torch.arange(slots, dtype=torch.long) % 4,
              "gt_valid": torch.ones(slots, dtype=torch.bool), "locator": "shape-stress",
              "archive": "shape-stress.npz"}
    if arm == "g-rgb":
        result["rgb"] = torch.randn(slots, 768)
    return result


def _measure(sample, arm, variant, profile, device, *, tracks=64, shape_stress=False):
    seed_run(0)
    batch = collate_multimodal([sample] * tracks)
    model = e002b.build_model(arm, variant, profile).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), **e002a_calibrate.SETTINGS["optimizer"])
    timings = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(3):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device),
                                   batch["padding_mask"].to(device)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite calibration loss")
        if repeat:
            timings.append(time.perf_counter() - start)
        del logits, loss
    result = {"source": sample.get("_resource_source", "shape-stress"), "arm": arm, "variant": variant,
              "tracks": tracks, "slots_per_track": int(batch["lengths"].max()),
              "temporal_cells": int(batch["lengths"].max()) ** 2 * tracks,
              "seconds_median": float(np.median(timings)), "tracks_per_second": tracks / float(np.median(timings)),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "shape_stress": shape_stress, "status": "complete"}
    del model, optimizer, batch
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args):
    torch.set_num_threads(2)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("E002b resource profiles require CUDA calibration")
    outputs = Path("outputs").resolve()
    output = args.output.resolve()
    fixture_root = args.fixtures.resolve()
    reference_path = args.reference_profile.resolve()
    if not output.is_relative_to(outputs / "experiments/E002b"):
        raise ValueError("Calibration evidence must be under outputs-root/experiments/E002b")
    if output == fixture_root or output.is_relative_to(fixture_root) or fixture_root.is_relative_to(output):
        raise ValueError("Calibration output and read-only fixtures must be separate")
    reference = json.loads(reference_path.read_text())
    if reference.get("status") != "complete":
        raise ValueError("A completed reference profile is required for packing limits")
    profile = e002a_calibrate.validate_profile(reference["profile"])
    if profile["max_tracks"] != 64:
        raise ValueError("E002b retains the agreed effective 64-track logical batch")

    total_vram = torch.cuda.get_device_properties(0).total_memory
    allocated_ceiling = int(total_vram * .78)
    record = {"status": "incomplete", "experiment": "E002b", "device": args.device,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "command": sys.orig_argv, "code": e002b_run.resource_code(), "profile": profile,
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "total_vram_bytes": total_vram,
              "allocated_ceiling_bytes": allocated_ceiling,
              "reference_profile": {"path": str(reference_path), "sha256": checksum(reference_path),
                                    "status": reference.get("status"), "code": reference.get("code")},
              "fixtures": str(fixture_root / "execution.json"),
              "fixture_sha256": checksum(fixture_root / "execution.json"),
              "settings": e002a_calibrate.SETTINGS, "model_specs": {}, "workloads": [],
              "failures": [], "held_out_tests": False,
              "interpretation": "CUDA CE/backward/AdamW resource checks only; no behavior result or training run"}
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        for arm in ARMS:
            for variant in e002b.VARIANTS:
                record["model_specs"][f"{arm}/{variant}"] = e002b.model_spec(arm, variant)

        samples = e002a_calibrate.fixture_samples(fixture_root)
        representatives = {}
        for name, sample in samples:
            source = sample["_resource_source"]
            sample = _model_sample(sample, "g-rgb")
            sample["_resource_source"] = source
            if source not in representatives or len(sample["targets"]) > len(representatives[source][1]["targets"]):
                representatives[source] = (name, sample)
        if set(representatives) != {"road-waymo", "loki"}:
            raise ValueError("Calibration fixtures must include both frozen source populations")
        max_tracks = profile["max_tracks"]
        slots_bound = math.isqrt(profile["temporal_cells"] // max_tracks)
        if slots_bound < max(len(sample["targets"]) for _, sample in representatives.values()):
            raise ValueError("Reference packing profile does not cover the longest calibration fixture")
        record["selected_fixtures"] = {source: {"name": name, "locator": sample["locator"],
            "slots": len(sample["targets"])} for source, (name, sample) in representatives.items()}
        record["shape_bound_slots"] = slots_bound
        reference_rows = [row for row in reference.get("workloads", [])
                          if row.get("status") == "complete" and row.get("arm") in ARMS
                          and row.get("tracks") == max_tracks and row.get("mode") == "ordinary"
                          and row.get("peak_allocated") and row.get("slots")]
        if not reference_rows:
            raise ValueError("Reference profile lacks measured 64-track geometry/RGB workloads for safe preflight")
        reference_slots = max(row["slots"] // row["tracks"] for row in reference_rows)
        reference_peak = max(row["peak_allocated"] for row in reference_rows)
        preflight_slots = max(slots_bound, max(len(sample["targets"]) for _, sample in representatives.values()))
        estimated_peak = int(reference_peak * (preflight_slots / reference_slots) ** 2 * 1.25)
        record["shape_preflight"] = {"reference_max_slots": reference_slots,
            "reference_peak_allocated_bytes": reference_peak, "target_slots": preflight_slots,
            "margin": 1.25, "estimated_peak_allocated_bytes": estimated_peak,
            "allocated_ceiling_bytes": allocated_ceiling}
        if estimated_peak > allocated_ceiling:
            raise ValueError("Conservative E002b shape estimate exceeds the 78% VRAM ceiling")

        for source in ("road-waymo", "loki"):
            _, rgb_sample = representatives[source]
            for arm in ARMS:
                sample = _model_sample(rgb_sample, arm)
                sample["_resource_source"] = source
                for variant in e002b.VARIANTS:
                    row = _measure(sample, arm, variant, profile, torch.device(args.device), tracks=max_tracks)
                    if row["peak_allocated_bytes"] > allocated_ceiling:
                        raise ValueError(f"Measured allocation exceeded 78% VRAM ceiling: {source}/{arm}/{variant}")
                    row["name"] = f"{source}-longest-fixture"
                    record["workloads"].append(row)
                    write_json(args.output / "profile.json", record)
                    print(row["name"], arm, variant, row["slots_per_track"],
                          round(row["seconds_median"], 3), row["peak_allocated_bytes"], flush=True)

        for arm in ARMS:
            sample = _stress_sample(slots_bound, arm)
            for variant in e002b.VARIANTS:
                row = _measure(sample, arm, variant, profile, torch.device(args.device),
                               tracks=max_tracks, shape_stress=True)
                if row["peak_allocated_bytes"] > allocated_ceiling:
                    raise ValueError(f"Shape stress exceeded 78% VRAM ceiling: {arm}/{variant}")
                row["source"] = "both-source-shape-bound"
                row["name"] = "64-track-temporal-cell-bound"
                record["workloads"].append(row)
                write_json(args.output / "profile.json", record)
                print(row["name"], arm, variant, row["slots_per_track"],
                      round(row["seconds_median"], 3), row["peak_allocated_bytes"], flush=True)

        expected = {f"{arm}/{variant}" for arm in ARMS for variant in e002b.VARIANTS}
        observed = {f"{row['arm']}/{row['variant']}" for row in record["workloads"]}
        if not expected.issubset(observed):
            raise ValueError("Calibration did not cover every independent E002b model")
        record["status"] = "complete"
        record["model_coverage"] = sorted(expected)
        return record
    except BaseException:
        record["status"] = "failed"
        record["failures"].append(traceback.format_exc())
        raise
    finally:
        write_json(args.output / "profile.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, required=True,
                        help="Completed profile used only for existing packing limits; model code is recalibrated here")
    parser.add_argument("--fixtures", type=Path, default=Path("outputs/experiments/E002a/execution-probe-04"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
