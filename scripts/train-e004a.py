#!/usr/bin/env python3
"""Prepare the frozen E004a feature cache and run its sequential label-budget fits."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pedestrian_behavior.experiments.e004a_run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--smoke", action="store_true",
        help="Run budgets 1 and 5 once for 200 updates with offline W&B in a fresh attempt directory.")
    parser.add_argument("--early-checks", action="store_true",
        help="Check at 0, 1, 5, 10, 25, 50, 100, then every 100 updates through 10,000.")
    args = parser.parse_args()
    print(run(args.device, args.outputs_root, args.smoke, args.early_checks))


if __name__ == "__main__":
    main()
