"""Run E002a geometry + RGB on both sources with seeds 1–4."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .e002a_run import FUSION_PROTOCOL, SETTINGS
from .sweep import run_sweep


SOURCES = ("road-waymo", "loki")
SEEDS = (1, 2, 3, 4)
ARM = "g-rgb"


def attempts():
    return [{"source": source, "arm": ARM, "seed": seed,
             "name": f"e002a-{source}-{ARM}-s{seed}"}
            for source in SOURCES for seed in SEEDS]


def attempt_command(row, output, device, options):
    return [sys.executable, "scripts/train-e002a.py", "--source", row["source"], "--arm", ARM,
            "--device", device, "--seed", str(row["seed"]), "--output", str(output / row["name"])] + options


def check_native_mount(probe):
    config = json.loads((probe / "probe.json").read_text())["config"]
    for key in ("waymo_root", "loki_root"):
        if config.get(key) and not Path(config[key]).is_dir():
            raise FileNotFoundError(f"Native source root unavailable; mount the dataset partition: {config[key]}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/experiments/E002a/runs") / ("g-rgb-seeds-1-4-" + timestamp))
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--rgb-preparation", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    return parser


def run(args):
    check_native_mount(args.probe)
    options = ["--profile", str(args.profile), "--rgb-preparation", str(args.rgb_preparation),
               "--probe", str(args.probe)]
    run_sweep(args.output, attempts(), lambda row, output, device: attempt_command(row, output, device, options),
              jobs=1, seeds=SEEDS, device=args.device,
              settings={"arm": ARM, "epochs": SETTINGS["epochs"], "patience": SETTINGS["patience"],
                        "held_out_tests": False, "kind": "exploratory-source-validation",
                        "fusion_protocol": FUSION_PROTOCOL, "profile": str(args.profile)})


def main():
    run(build_parser().parse_args())
