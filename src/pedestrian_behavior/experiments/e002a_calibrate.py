"""Bounded real-model CE/backward/AdamW resource calibration, never a comparison run."""

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.multimodal import collate_multimodal
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import track_cross_entropy
from pedestrian_behavior.models.multimodal import MultimodalClassifier
from pedestrian_behavior.training import seed_run
from . import e001b_run
from .e002a_execution_probe import loaded_track
from .e002a_run import resource_code, SETTINGS
from .run import write_json


def fixture_samples(fixtures):
    record = json.loads((fixtures / "execution.json").read_text())
    if record["status"] != "complete" or len(record["tracks"]) != 6:
        raise ValueError("Completed six-track verified resource fixture required")
    datasets = {}
    result = []
    for item in record["tracks"]:
        d = item["dataset"]
        if d not in datasets:
            baseline = e001b_run.assemble(d, "geometry", "B")
            datasets[d] = {split: baseline["dataset"](d, split) for split in ("training", "validation")}
        matches = [(dataset, i) for dataset in datasets[d].values() for i, e in enumerate(dataset.entries) if e["locator"] == item["locator"]]
        if len(matches) != 1:
            raise ValueError("Calibration fixture not in frozen source development population")
        dataset, index = matches[0]
        sample = dataset[index]
        frames, rgb, valid = loaded_track(item, fixtures)
        geometry = sample.pop("inputs")
        if len(frames) != len(geometry):
            raise ValueError("Fixture/geometry slots differ")
        flags = torch.stack((geometry[:, [8, 9, 10, 11, 18, 19]].bool().any(-1), valid.bool(),
                             torch.tensor([len(f) > 0 for f in frames])), -1).float()
        result.append((item["name"], sample | {"geometry": geometry, "rgb": rgb, "flags": flags, "points": frames}))
    return result


def measure(sample, arm, count, profile, mode, device):
    seed_run(0)
    batch = collate_multimodal([sample] * count)
    model = MultimodalClassifier(arm, profile).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), **SETTINGS["optimizer"])
    timings = []
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for repeat in range(3):
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch, mode=mode)
        loss = track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device), batch["padding_mask"].to(device)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if device == "cuda":
            torch.cuda.synchronize()
        if repeat:
            timings.append(time.perf_counter() - begin)
        del logits, loss
    seconds = float(np.median(timings))
    result = {"arm": arm, "mode": mode, "tracks": count, "slots": int(batch["lengths"].sum()),
              "tokens": sum(len(f) for f in batch["points"]) if arm == "g-rgb-dditr" else 0,
              "seconds": seconds, "tracks_per_second": count / seconds,
              "peak_allocated": torch.cuda.max_memory_allocated() if device == "cuda" else None,
              "peak_reserved": torch.cuda.max_memory_reserved() if device == "cuda" else None,
              "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    del model, optimizer, batch
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def run(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    record = {"status": "incomplete", "device": args.device, "code": resource_code(),
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "command": sys.orig_argv, "workloads": [], "failures": [], "held_out_tests": False,
              "fixture_sha256": checksum(args.fixtures / "execution.json"), "settings": SETTINGS}
    telemetry = log = None
    try:
        seed_run(0)
        total = torch.cuda.get_device_properties(0).total_memory if args.device == "cuda" else 12 * 2**30
        record.update(gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                      total_vram=total, allocated_ceiling=int(total * .78), frozen_miss_reserve_bytes=2 * 2**30,
                      interpretation="Real model/GT CE updates on development fixtures; optimizer resets per shape; no selected behavior result")
        if args.device == "cuda":
            log = (args.output / "system.csv").open("w")
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
                                          "--format=csv", "-l", "1"], stdout=log, stderr=subprocess.STDOUT)
        with (args.output / "equivalence-tests.log").open("w") as stream:
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_e002a_training.py"],
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        samples = fixture_samples(args.fixtures)
        max_length = max(len(s["targets"]) for _, s in samples)
        profile = {"max_tracks": 64, "temporal_cells": 64 * max_length**2, "microbatch_tokens": 1,
                   "ordinary_tokens": 1, "frame_chunk": 2, "frame_tokens": 524288}
        record["selection"] = [{"name": n, "locator": s["locator"], "slots": len(s["targets"]),
                                "tokens": sum(len(f) for f in s["points"])} for n, s in samples]
        # A separately labelled shape stress tiles features; it is never a training population sample.
        largest = max((f for _, s in samples for f in s["points"]), key=len)
        stress = deepcopy(samples[0][1])
        for key in ("geometry", "rgb", "flags", "targets", "gt_valid"):
            stress[key] = stress[key][:1].clone()
        stress["flags"][:, 2] = 1
        stress["points"] = [largest.repeat((524288 + len(largest) - 1) // len(largest), 1)[:524288]]
        workloads = []
        for name, sample in samples:
            cost = sum(len(f) for f in sample["points"])
            for count in (1, 2, 4):
                for mode in ("ordinary", "recompute"):
                    # Conservative preflight avoids probing known unsafe allocations; no OOM retry.
                    estimate = count * cost * 4000 + 2**30
                    if mode == "ordinary" and estimate > record["allocated_ceiling"]:
                        record["workloads"].append({"name": name, "tracks": count, "mode": mode, "status": "not-run-shape-estimate", "estimate": estimate})
                    else:
                        workloads.append((name, sample, "g-rgb-dditr", count, mode))
        workloads.append(("single-frame-524288-token-shape-stress", stress, "g-rgb-dditr", 1, "recompute"))
        typical = max(samples, key=lambda row: len(row[1]["targets"]))
        for arm in ("g", "g-rgb"):
            for count in (8, 16, 32, 64):
                workloads.append((typical[0], typical[1], arm, count, "ordinary"))
        for name, sample, arm, count, mode in workloads:
            row = measure(sample, arm, count, profile, mode, args.device) | {"name": name, "status": "complete"}
            if row["peak_allocated"] is not None and row["peak_allocated"] > record["allocated_ceiling"]:
                raise ValueError("Measured memory exceeds predeclared calibration headroom")
            record["workloads"].append(row)
            write_json(args.output / "profile.json", record)
            print(name, arm, count, mode, round(row["seconds"], 3), flush=True)
        fitting = [w for w in record["workloads"] if w.get("status") == "complete" and w.get("mode") == "ordinary"
                   and w.get("arm") == "g-rgb-dditr"]
        if not fitting:
            raise ValueError("No ordinary spatial calibration condition fit")
        # Leave ten percent token headroom below the largest measured ordinary shape.
        ordinary = int(max(w["tokens"] for w in fitting) * .9)
        profile.update(ordinary_tokens=ordinary, microbatch_tokens=ordinary)
        record.update(status="complete", profile=profile)
        return record
    except BaseException:
        record["status"] = "failed"; record["failures"].append(traceback.format_exc()); raise
    finally:
        if telemetry is not None:
            telemetry.terminate(); telemetry.wait(timeout=10); log.close()
        write_json(args.output / "profile.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=Path("outputs/experiments/E002a/execution-probe-04"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run(parser.parse_args())
