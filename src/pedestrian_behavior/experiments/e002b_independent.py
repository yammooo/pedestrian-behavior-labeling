"""Sequential manual launcher for independent E002b geometry/RGB attempts."""

import argparse
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import sys

from . import e002b, e002b_run
from .e002a_run import FUSION_PROTOCOL, SETTINGS
from .sweep import run_sweep


def attempts(source, arms, variants, dropouts=(.1,)):
    return [{"source": source, "arm": arm, "variant": variant, "dropout": dropout, "seed": 0,
             "name": f"e002b-{source}-{arm}-{variant}"
             + (f"-d{int(dropout * 100):02d}" if dropout != .1 else "") + "-s0"}
            for arm in arms for variant in variants for dropout in dropouts]


def attempt_command(row, output, device, options):
    return [sys.executable, "scripts/train-e002b.py", "--source", row["source"],
            "--arm", row["arm"], "--variant", row["variant"], "--dropout", str(row["dropout"]), "--device", device,
            "--output", str(output / row["name"])] + options


def run(args):
    dropouts = getattr(args, "dropouts", (.1,))
    if not args.arms or any(arm not in e002b_run.ARMS for arm in args.arms):
        raise ValueError("Independent launcher supports only g and g-rgb; grouped g-dditr is pending")
    if not args.variants or any(variant not in e002b.VARIANTS for variant in args.variants):
        raise ValueError("Unknown E002b variant")
    if "g-rgb" in args.arms and args.rgb_preparation is None:
        raise ValueError("g-rgb requires completed matching --rgb-preparation metadata")
    if any(dropout not in (.1, .3, .5) for dropout in dropouts):
        raise ValueError("Dropout must be 0.1, 0.3 or 0.5")
    if any(dropout != .1 for dropout in dropouts) and (args.arms != ["g-rgb"] or
            any(variant not in ("norm", "temporal-small") for variant in args.variants)):
        raise ValueError("Higher dropout is scoped to g-rgb norm/temporal-small")
    for arm in args.arms:
        for variant in args.variants:
            e002b_run._profile_for(argparse.Namespace(profile=args.profile, arm=arm,
                model_variant=variant, device=args.device))
    if not args.output.resolve().is_relative_to(args.outputs_root.resolve() / "experiments/E002b"):
        raise ValueError("Launcher output must be under outputs-root/experiments/E002b")
    options = ["--profile", str(args.profile.resolve()), "--outputs-root", str(args.outputs_root.resolve()),
               "--probe", str(args.probe.resolve()), "--rgb-cache-root", str(args.rgb_cache_root.resolve())]
    if args.rgb_preparation is not None:
        options += ["--rgb-preparation", str(args.rgb_preparation.resolve())]
    plan = attempts(args.source, args.arms, args.variants, dropouts)
    return run_sweep(args.output, plan, partial(attempt_command, options=options), jobs=1, seeds=(0,),
        device=args.device, settings={"epochs": SETTINGS["epochs"], "patience": SETTINGS["patience"],
            "held_out_tests": False, "kind": "exploratory-source-validation", "fusion_protocol": FUSION_PROTOCOL,
            "profile": str(args.profile.resolve()), "dropouts": list(dropouts),
            "execution": "independent sequential attempts"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--source", choices=("road-waymo", "loki"), required=True)
    parser.add_argument("--arms", nargs="+", choices=e002b_run.ARMS, default=e002b_run.ARMS)
    parser.add_argument("--variants", nargs="+", choices=tuple(e002b.VARIANTS), default=tuple(e002b.VARIANTS))
    parser.add_argument("--dropouts", nargs="+", type=float, choices=(.1, .3, .5), default=[.1])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path,
        default=Path("outputs/experiments/E002b/runs") / ("independent-" + timestamp))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path,
        help="Completed full-source RGB preparation JSON; required when g-rgb is selected")
    run(parser.parse_args())
