"""Fit matched ROAD-only and joint heads on Euro-PVI track budgets."""

import argparse
from pathlib import Path

from pedestrian_behavior.experiments.e006c_head import AUDIT, DATASET, E006B_RUN, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--e006b-run", type=Path, default=E006B_RUN)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--fit-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true", help="Use only three calibration sequences")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
