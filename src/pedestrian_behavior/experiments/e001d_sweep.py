"""Launch the ten E001d matched-capacity attempts."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

from .e001d_run import SETTINGS
from .sweep import run_sweep as execute_sweep

SEEDS = (0, 1, 2, 3, 4)
SOURCES = ("road-waymo", "loki")


def attempts():
    return [{"source": source, "seed": seed,
             "name": f"e001d-{source}-ktr-mlp-matched-s{seed}"}
            for seed in SEEDS for source in SOURCES]


def attempt_command(attempt, output, device):
    return [sys.executable, "scripts/train-e001d.py", "--source", attempt["source"],
            "--seed", str(attempt["seed"]), "--device", device,
            "--output", str(output/attempt["name"])]


def run_sweep(output, jobs=2, device="cuda"):
    return execute_sweep(output, attempts(), attempt_command, jobs, SEEDS, device,
                         settings={"experiment": "E001d", "configuration": "K+T+R",
                                   "variant": "mlp-matched", "training": SETTINGS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/experiments/E001d/runs")/("matched-capacity-"+timestamp))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run_sweep(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
