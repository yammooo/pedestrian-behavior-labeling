"""Manual sequential six-attempt E002a pilot; source validation only, seed zero."""

import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import sys

from .e002a_run import ARMS, SETTINGS
from .sweep import run_sweep


def attempts():
    return [{"source": d, "arm": arm, "seed": 0, "name": f"e002a-{d}-{arm}-s0"}
            for d in ("road-waymo", "loki") for arm in ARMS]


def attempt_command(row, output, device, options):
    return [sys.executable, "scripts/train-e002a.py", "--source", row["source"], "--arm", row["arm"],
            "--device", device, "--output", str(output / row["name"])] + options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/E002a/runs") / ("pilot-" + timestamp))
    parser.add_argument("--profile", type=Path, default=Path("outputs/experiments/E002a/model-calibration/profile.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--point-cache-root", type=Path, default=Path("outputs/experiments/E002a/point-cache"))
    parser.add_argument("--point-cache-gb", type=float, default=450)
    args = parser.parse_args()
    options = ["--profile", str(args.profile), "--rgb-cache-root", str(args.rgb_cache_root),
               "--point-cache-root", str(args.point_cache_root), "--point-cache-gb", str(args.point_cache_gb)]
    run_sweep(args.output, attempts(), partial(attempt_command, options=options), jobs=1, seeds=(0,), device=args.device,
              settings={"epochs": SETTINGS["epochs"], "patience": SETTINGS["patience"], "held_out_tests": False,
                        "kind": "exploratory-source-validation", "profile": str(args.profile)})
    from .e002a_summary import summarize
    summarize(args.output)
