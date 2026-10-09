"""The frozen E001b BiLSTM-only, two-input, two-source, five-seed matrix."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from .e001b_run import INPUT_DIMENSIONS
from .sweep import run_sweep as execute_sweep


def attempts(seeds):
    return [{"source": source, "configuration": configuration, "variant": "B", "seed": seed,
             "name": f"{source}-{configuration}-BiLSTM-seed{seed}"}
            for seed in seeds for source in ("road-waymo", "loki") for configuration in INPUT_DIMENSIONS]


def attempt_command(attempt, output, device):
    return [sys.executable, "scripts/train-e001b.py", "--source", attempt["source"],
            "--configuration", attempt["configuration"], "--seed", str(attempt["seed"]),
            "--epochs", "75", "--patience", "8", "--device", device,
            "--output", str(output/attempt["name"])]


def run_sweep(output, jobs=2, seeds=(0, 1, 2, 3, 4), device="cuda"):
    return execute_sweep(output, attempts(seeds), attempt_command, jobs, seeds, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/E001b/runs")/("seeds-"+timestamp))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run_sweep(**vars(parser.parse_args()))
