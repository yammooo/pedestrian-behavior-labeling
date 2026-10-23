"""Manual sequential six-attempt E002a pilot; source validation only, seed zero."""

import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import sys

from .e002a_run import ARMS, FUSION_PROTOCOL, SETTINGS, DEFAULT_POINT_WORKERS
from .sweep import run_sweep


def attempts():
    return [{"source": d, "arm": arm, "seed": 0, "name": f"e002a-{d}-{arm}-s0"}
            for d in ("road-waymo", "loki") for arm in ARMS]


def attempt_command(row, output, device, options, indexed_root=None):
    return [sys.executable, "scripts/train-e002a.py", "--source", row["source"], "--arm", row["arm"],
            "--device", device, "--output", str(output / row["name"])] + options + (
                ["--indexed-point-cache", str(indexed_root / ("indexed-" + row["source"]))]
                if indexed_root is not None and row["arm"] == "g-rgb-dditr" else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/E002a/runs") / ("pilot-" + timestamp))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rgb-preparation", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--indexed-cache-root", type=Path, default=Path("outputs/experiments/E002a"),
                        help="Parent of indexed-road-waymo and indexed-loki")
    parser.add_argument("--point-workers", type=int, default=DEFAULT_POINT_WORKERS)
    args = parser.parse_args()
    options = ["--profile", str(args.profile), "--rgb-preparation", str(args.rgb_preparation),
               "--rgb-cache-root", str(args.rgb_cache_root), "--point-workers", str(args.point_workers)]
    run_sweep(args.output, attempts(), partial(attempt_command, options=options, indexed_root=args.indexed_cache_root), jobs=1, seeds=(0,), device=args.device,
              settings={"epochs": SETTINGS["epochs"], "patience": SETTINGS["patience"], "held_out_tests": False,
                        "kind": "exploratory-source-validation", "fusion_protocol": FUSION_PROTOCOL,
                        "point_workers":args.point_workers,"profile": str(args.profile)})
    from .e002a_summary import summarize
    summarize(args.output)
