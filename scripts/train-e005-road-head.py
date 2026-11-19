"""Extract frozen E005 ROAD embeddings and fit the label-budget heads."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from pedestrian_behavior.experiments import e005_head, e005_transfer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Use a separate cache and run one bounded fit")
    args = parser.parse_args()

    args.outputs_root = args.outputs_root.resolve()
    root = args.outputs_root / "experiments/E005"
    args.source_run = root / "runs/loki-g-rgb-utonia-s0-20261118-074440"
    args.reference_predictions = (
        root / "transfer/loki-to-road-20261119-112734/road-waymo-validation/predictions.npz"
    )
    args.probe = args.utonia_code = args.tiny_checkpoint = None
    args.device = "cuda"
    budget_root = root / "head-budgets"
    cache_name = "road-embeddings-smoke" if args.smoke else "road-embeddings"
    attempt = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    if args.smoke:
        attempt = f"smoke-{attempt}"
    output = args.output or budget_root / "runs" / attempt
    if output.exists():
        parser.error(f"Use a fresh output directory: {output}")

    cache = e005_transfer.load_or_extract_embeddings(args, budget_root / cache_name)
    e005_head.run(output, cache, device="cpu", smoke=args.smoke)
    print(f"E005 ROAD head-budget results: {output}", flush=True)


if __name__ == "__main__":
    main()
