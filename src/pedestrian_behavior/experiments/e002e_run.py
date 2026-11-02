"""E002e: change only AdamW weight decay for the E002b g-RGB reference."""

import argparse
from pathlib import Path

import torch

from . import e002a_run, e002b, e002b_run
from .run import run_attempt


WEIGHT_DECAYS = (0.01, 0.1)


def validate_roots(args, config):
    rgb, points, output = (path.resolve() for path in
                           (args.rgb_cache_root, args.point_cache_root, args.output))
    root = args.outputs_root.resolve()
    if not output.is_relative_to(root / "experiments/E002e"):
        raise ValueError("E002e evidence must be under outputs-root/experiments/E002e")
    locations = (rgb, points, output)
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for index, a in enumerate(locations) for b in locations[index + 1:]):
        raise ValueError("E002a feature caches and E002e evidence must have separate roots")
    protected = [root / "experiments" / name for name in ("E001", "E001b", "E001c")]
    protected.extend(Path(config[key]).resolve() for key in ("loki_root", "waymo_root", "road_index")
                     if config.get(key))
    if any(location.is_relative_to(path) or path.is_relative_to(location)
           for location in locations for path in protected):
        raise ValueError("E002e evidence must not overlap native data or established E001 artifacts")


def attempt_settings(weight_decay):
    if weight_decay not in WEIGHT_DECAYS:
        raise ValueError("E002e weight decay must be 0.01 or 0.1")
    return {**e002a_run.SETTINGS,
            "optimizer": e002a_run.SETTINGS["optimizer"] | {"weight_decay": weight_decay}}


def assembly(args):
    e002b_run._profile_for(args)
    name = f"e002e-{args.source}-g-rgb-temporal-small-wd{str(args.weight_decay).replace('.', 'p')}-s0"
    policy = {
        "experiment": "E002e",
        "resource_code": e002b_run.resource_code,
        "validate_roots": validate_roots,
        "model_spec": lambda arm, variant: e002b.model_spec(arm, variant),
        "model_factory": lambda arm, variant, profile: e002b.build_model(arm, variant, profile),
        "run_name": name,
    }
    prepared = e002a_run.assembly(args, policy=policy)
    prepared["config"]["weight_decay_sensitivity"] = {
        "weight_decay": args.weight_decay,
        "control": 0.0001,
        "all_other_training_settings": "E002b g-rgb temporal-small seed0",
    }
    return prepared


def run(args):
    if args.arm != "g-rgb" or args.model_variant != "temporal-small" or args.dropout != .1 \
            or args.rgb_branch_dropout != 0. or args.lidar_branch_dropout != 0. or args.smoke:
        raise ValueError("E002e is fixed to full g-rgb temporal-small with 0.1 temporal dropout")
    settings = attempt_settings(args.weight_decay)
    torch.set_num_threads(2)
    return run_attempt(args.source, args.arm, args.model_variant, args.output, args.device,
                       settings=settings, assemble=lambda *_: assembly(args),
                       classes=e002a_run.CLASSES,
                       axes={key: value for key, value in e002a_run.STRATA.items()
                             if key != "lidar-observed"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("road-waymo", "loki"), required=True)
    parser.add_argument("--weight-decay", choices=(.01, .1), type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path, required=True)
    parser.add_argument("--indexed-point-cache", type=Path)
    parser.set_defaults(arm="g-rgb", model_variant="temporal-small", dropout=.1,
                        rgb_branch_dropout=0., lidar_branch_dropout=0., smoke=False,
                        smoke_selection=None, point_workers=e002a_run.DEFAULT_POINT_WORKERS,
                        seed=0)
    args = parser.parse_args()
    args.point_cache_root = args.indexed_point_cache or (
        args.outputs_root / "experiments/E002a" / ("indexed-" + args.source))
    run(args)


if __name__ == "__main__":
    main()
