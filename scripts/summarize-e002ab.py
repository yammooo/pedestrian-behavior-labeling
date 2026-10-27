from argparse import ArgumentParser
from pathlib import Path

from pedestrian_behavior.experiments.e002ab_results import summarize


def main():
    parser = ArgumentParser(description="Verify and summarize all completed E002a/E002b source-validation checkpoints.")
    parser.add_argument("outputs_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.outputs_root, args.output)
    print(f"Wrote {len(result['results'])} verified checkpoints to {args.output}")


if __name__ == "__main__":
    main()
