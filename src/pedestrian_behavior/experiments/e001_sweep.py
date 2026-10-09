"""Run the agreed E001 repetitions in separate, bounded concurrent processes."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from .e001 import FEATURE_SETS
from .sweep import run_sweep as execute_sweep


def attempts(seeds):
    return [{"source": source, "configuration": features, "variant": variant, "seed": seed,
             "name": f"{source}-{features}-{'MLP' if variant == 'A' else 'BiLSTM'}-seed{seed}"}
            for seed in seeds for source in ("road-waymo", "loki")
            for features in FEATURE_SETS for variant in ("A", "B")]


def attempt_command(attempt, output, device):
    return [sys.executable, "scripts/train-e001.py", "--source", attempt["source"],
            "--configuration", attempt["configuration"], "--variant", attempt["variant"],
            "--seed", str(attempt["seed"]), "--epochs", "75", "--patience", "8",
            "--device", device, "--output", str(output/attempt["name"])]


def run_sweep(output, jobs=2, seeds=(0, 1, 2, 3, 4), device="cuda"):
    return execute_sweep(output, attempts(seeds), attempt_command, jobs, seeds, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/E001/runs")/("seeds-"+timestamp))
    parser.add_argument("--jobs", type=int, default=2, help="Maximum simultaneous attempts on the selected device")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run_sweep(**vars(parser.parse_args()))
