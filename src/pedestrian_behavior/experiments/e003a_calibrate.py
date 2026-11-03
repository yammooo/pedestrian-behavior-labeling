"""Fresh bounded CUDA resource checks for E003a; no behavior training."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from pedestrian_behavior.data.indexed_points import PROTOCOL as POINT_PROTOCOL
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.experiments import e002a_calibrate, e002a_run, e002b, e002b_calibrate
from pedestrian_behavior.experiments.e003a_run import (ARM, RGB_DROPOUT, VARIANT, calibration_code,
                                                       model_spec, resource_code)
from pedestrian_behavior.experiments.run import write_json


def run(args):
    torch.set_num_threads(2)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("E003a resource profiles require CUDA")
    output, fixtures, reference_path = args.output.resolve(), args.fixtures.resolve(), args.reference_profile.resolve()
    outputs_root = Path("outputs").resolve()
    if not output.is_relative_to(outputs_root / "experiments/E003a"):
        raise ValueError("Calibration evidence must be under outputs/experiments/E003a")
    if output.exists() or any(output == p or output.is_relative_to(p) or p.is_relative_to(output)
                              for p in (fixtures, reference_path.parent)):
        raise ValueError("Calibration output must be fresh and separate from read-only inputs")
    reference = json.loads(reference_path.read_text())
    if (reference.get("status") != "complete" or reference.get("experiment") != "E002d"
            or reference.get("point_storage_protocol") != POINT_PROTOCOL):
        raise ValueError("Completed E002d indexed-point calibration required as a packing-limit reference")
    profile = e002a_run.validate_profile(reference.get("profile", {}))
    if profile["max_tracks"] != 64:
        raise ValueError("E003a keeps the effective 64-track logical batch")

    started = time.monotonic()
    record = {"status": "incomplete", "experiment": "E003a", "device": args.device,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.orig_argv, "code": resource_code(), "profile": profile,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "total_vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "allocated_ceiling_bytes": int(torch.cuda.get_device_properties(0).total_memory * .78),
        "point_storage_protocol": POINT_PROTOCOL,
        "model_specs": {f"{ARM}/{VARIANT}": model_spec()}, "model_coverage": [],
        "reference_profile": {"path": str(reference_path), "sha256": checksum(reference_path),
                              "status": reference["status"], "code": reference.get("code")},
        "fixtures": str(fixtures / "execution.json"),
        "fixture_sha256": checksum(fixtures / "execution.json"),
        "calibration_code_sha256": calibration_code(),
        "settings": e002a_calibrate.SETTINGS, "rgb_branch_dropout": RGB_DROPOUT,
        "workloads": [], "failures": [], "held_out_tests": False,
        "scope": "fresh readout/update capacity checks on frozen training fixtures and bounded mask shapes; no behavior training, validation scoring, held-out access or W&B",
        "started_utc": datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True, exist_ok=False)
    try:
        reference_rows = [row for row in reference.get("workloads", [])
                          if row.get("status") == "complete" and row.get("arm") == ARM
                          and row.get("tracks") and row.get("peak_allocated_bytes")]
        if not reference_rows:
            raise ValueError("E002d reference lacks completed combined-arm CUDA workloads")
        reference_peak = max(row["peak_allocated_bytes"] for row in reference_rows)
        if reference_peak > record["allocated_ceiling_bytes"]:
            raise ValueError("Existing combined-arm preflight exceeds the current GPU safety ceiling")
        record["shape_preflight"] = {"reference_peak_allocated_bytes": reference_peak,
            "allocated_ceiling_bytes": record["allocated_ceiling_bytes"],
            "method": "reuse the accepted temporal/token limits, then remeasure E003a code, both-source real point payloads and the 64-track bound"}
        e002a_run.validate_profile(profile)
        samples = e002a_calibrate.fixture_samples(fixtures)
        representatives = e002b_calibrate._representatives(samples)
        if set(representatives) != {"road-waymo", "loki"}:
            raise ValueError("Calibration fixtures must represent both frozen source populations")
        slots_bound = math.isqrt(profile["temporal_cells"] // profile["max_tracks"])
        if slots_bound < max(len(sample["targets"]) for _, sample in representatives.values()):
            raise ValueError("Reference temporal profile does not cover the longest source fixture")
        record["shape_bound_slots"] = slots_bound
        for source in ("road-waymo", "loki"):
            name, sample = representatives[source]
            sample = e002b_calibrate._model_sample(sample, ARM)
            row = e002b_calibrate._measure(sample, ARM, VARIANT, profile, torch.device("cuda"),
                tracks=1, rgb_branch_dropout=RGB_DROPOUT)
            row["source"], row["fixture"] = source, name
            if row["peak_allocated_bytes"] > record["allocated_ceiling_bytes"]:
                raise ValueError(f"Measured allocation exceeded 78% of VRAM: {source}")
            record["workloads"].append(row)
            write_json(output / "profile.json", record)
        stress = e002b_calibrate._stress_sample(slots_bound, ARM)
        middle = slots_bound // 2
        stress["flags"][middle] = 0.
        stress["geometry"][middle] = 0.
        stress["rgb"][middle] = 0.
        row = e002b_calibrate._measure(stress, ARM, VARIANT, profile, torch.device("cuda"),
            tracks=profile["max_tracks"], shape_stress=True, rgb_branch_dropout=RGB_DROPOUT)
        row.update(source="both-source-shape-bound", masked_empty_slot=middle)
        if row["peak_allocated_bytes"] > record["allocated_ceiling_bytes"]:
            raise ValueError("64-track masked shape check exceeded 78% of VRAM")
        record["workloads"].append(row)
        record.update(status="complete", model_coverage=[f"{ARM}/{VARIANT}"],
                      finished_utc=datetime.now(timezone.utc).isoformat())
        return record
    except BaseException:
        record["status"] = "failed"
        record["failures"].append(traceback.format_exc())
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "profile.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-profile", type=Path, required=True,
        help="Completed E002d profile used only as the accepted packing-limit reference")
    parser.add_argument("--fixtures", type=Path, default=Path("outputs/experiments/E002a/execution-probe-04"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
