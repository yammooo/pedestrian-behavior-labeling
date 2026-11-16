#!/usr/bin/env python3
"""Run the fixed E004c adaptation-depth comparison or one bounded target-only smoke."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pedestrian_behavior.experiments.e004c_run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path,
        default=Path("outputs/experiments/E002a/rgb-cache-preparation-03/cache.json"))
    parser.add_argument("--smoke", action="store_true",
        help="Run one temporal and one target-only 1,000-track draw for one update, with offline W&B.")
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
