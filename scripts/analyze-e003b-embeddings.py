"""Fit and plot the prespecified PCA and UMAP E003b validation diagnostic."""

import argparse
from pathlib import Path

from pedestrian_behavior.experiments.e003b_embedding_diagnostic import analyze


def main():
    root = Path("outputs/experiments/E003b/embedding-diagnostic-20261105")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=root / "input")
    parser.add_argument("--output", type=Path, default=root / "analysis")
    args = parser.parse_args()
    analyze(args.input_root, args.output)


if __name__ == "__main__":
    main()
