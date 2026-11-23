"""Run pinned E005 checkpoints on the frozen Euro-PVI E006b assessment split."""

import argparse
from pathlib import Path

from pedestrian_behavior.experiments.e006b_run import AUDIT, DATASET, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh local directory for metrics and unpadded predictions")
    parser.add_argument("--smoke", action="store_true",
                        help="Use three fixed calibration sequences; never reads assessment sequences")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
