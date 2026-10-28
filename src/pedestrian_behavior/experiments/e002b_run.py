"""Independent E002b source-validation attempts for geometry and geometry + RGB."""

import argparse
import json
from pathlib import Path

import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.models import multimodal
from pedestrian_behavior.data import multimodal as loading, prepared_rgb
from pedestrian_behavior import accumulation, evaluation, training
from . import run as shared_run
from . import e002a_features, e002a_run, e002b
from .run import run_attempt as execute_attempt


ARMS = ("g", "g-rgb")
SETTINGS = e002a_run.SETTINGS


def resource_code():
    from pedestrian_behavior.experiments import e002a_run as assembly
    modules = (multimodal, loading, prepared_rgb, accumulation, evaluation, training, shared_run,
               e002a_features, assembly, e002b)
    return {module.__name__: checksum(module.__file__) for module in modules} | {__name__: checksum(__file__)}


def validate_roots(args, config):
    """E002b output guard; keep E002a's hashed extraction-policy module untouched."""
    rgb, points, output = (path.resolve() for path in (args.rgb_cache_root, args.point_cache_root, args.output))
    root = args.outputs_root.resolve()
    experiment = "E002c" if getattr(args, "rgb_branch_dropout", 0.) else "E002b"
    if not output.is_relative_to(root / f"experiments/{experiment}"):
        raise ValueError(f"{experiment} evidence must be under outputs-root/experiments/{experiment}")
    locations = (rgb, points, output)
    if any(left.is_relative_to(right) or right.is_relative_to(left)
           for index, left in enumerate(locations) for right in locations[index + 1:]):
        raise ValueError(f"E002a feature caches and {experiment} evidence must have separate roots")
    protected = [root / "experiments" / name for name in ("E001", "E001b", "E001c")]
    protected.extend(Path(config[key]).resolve() for key in ("loki_root", "waymo_root", "road_index") if config.get(key))
    for location in locations:
        if any(location.is_relative_to(path) or path.is_relative_to(location) for path in protected):
            raise ValueError(f"{experiment} evidence must not overlap native data or established E001 artifacts")


def _profile_for(args):
    record = json.loads(args.profile.read_text())
    if record.get("status") != "complete" or record.get("experiment") != "E002b":
        raise ValueError("Completed E002b calibration profile required")
    if record.get("device") != args.device:
        raise ValueError("E002b calibration device differs from the requested device")
    if record.get("code") != resource_code():
        raise ValueError("E002b model or execution code changed since calibration")
    if record.get("torch") != str(torch.__version__) or record.get("cuda") != torch.version.cuda:
        raise ValueError("PyTorch/CUDA backend differs from E002b calibration")
    if args.device == "cuda" and (not torch.cuda.is_available() or record.get("gpu") != torch.cuda.get_device_name()):
        raise ValueError("E002b calibration hardware differs from the requested CUDA device")
    key = f"{args.arm}/{args.model_variant}"
    # E002c adds training-time masking without changing the calibrated tensor shape.
    if record.get("model_specs", {}).get(key) != e002b.model_spec(args.arm, args.model_variant):
        raise ValueError("Selected arm/variant is not covered by this exact E002b calibration")
    if key not in record.get("model_coverage", ()):
        raise ValueError("Selected arm/variant is not listed as completed in E002b calibration")
    e002a_run.validate_profile(record.get("profile", {}))
    return record


def assembly(args):
    _profile_for(args)
    dropout = args.dropout
    rgb_branch_dropout = getattr(args, "rgb_branch_dropout", 0.)
    experiment = "E002c" if rgb_branch_dropout else "E002b"
    run_name = (f"e002c-{args.source}-g-rgb-temporal-small-rgbdrop{int(rgb_branch_dropout * 100)}-s0"
                if rgb_branch_dropout else
                f"e002b-{args.source}-{args.arm}-{args.model_variant}"
                + (f"-d{int(dropout * 100):02d}" if dropout != .1 else "") + "-s0"
                + ("-smoke-" + args.output.name if args.smoke else ""))
    policy = {
        "experiment": experiment,
        "resource_code": resource_code,
        "validate_roots": validate_roots,
        "model_spec": lambda arm, variant: e002b.model_spec(arm, variant, dropout, rgb_branch_dropout),
        "model_factory": lambda arm, variant, profile: e002b.build_model(
            arm, variant, profile, dropout, rgb_branch_dropout),
        "run_name": run_name,
    }
    return e002a_run.assembly(args, policy=policy)


def run(args):
    if args.arm not in ARMS:
        raise ValueError("Independent E002b runs support only g and g-rgb; g-dditr requires the grouped runner")
    if args.arm == "g-rgb" and args.rgb_preparation is None:
        raise ValueError("g-rgb requires matching completed --rgb-preparation metadata")
    if args.smoke and args.smoke_selection is None:
        raise ValueError("Smoke requires an explicit bounded source train/validation selection")
    if args.dropout not in (.1, .3, .5):
        raise ValueError("E002b dropout must be 0.1, 0.3 or 0.5")
    rgb_branch_dropout = getattr(args, "rgb_branch_dropout", 0.)
    if rgb_branch_dropout not in (0., .3, .5):
        raise ValueError("RGB branch dropout must be 0, 0.3 or 0.5")
    if rgb_branch_dropout and (args.arm != "g-rgb" or args.model_variant != "temporal-small"
                              or args.dropout != .1 or args.smoke):
        raise ValueError("E002c RGB branch dropout requires full g-rgb temporal-small runs with temporal dropout 0.1")
    if args.dropout != .1 and (args.arm != "g-rgb" or args.model_variant not in ("norm", "temporal-small")):
        raise ValueError("Higher dropout is scoped to the g-rgb norm/temporal-small comparison")
    _profile_for(args)
    torch.set_num_threads(2)
    axes = {key: value for key, value in e002a_run.STRATA.items()
            if key not in ("rgb-observed", "lidar-observed") or key == "rgb-observed" and args.arm == "g-rgb"}
    return execute_attempt(args.source, args.arm, args.model_variant, args.output, args.device, args.smoke,
                           settings=SETTINGS, assemble=lambda *_: assembly(args), classes=e002a_run.CLASSES,
                           axes=axes)


def add_arguments(parser):
    parser.add_argument("--source", choices=("road-waymo", "loki"), required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--variant", dest="model_variant", choices=tuple(e002b.VARIANTS), required=True)
    parser.add_argument("--dropout", type=float, choices=(.1, .3, .5), default=.1)
    parser.add_argument("--rgb-branch-dropout", type=float, choices=(0., .3, .5), default=0.)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path,
                        help="Completed full-source RGB preparation JSON; required for g-rgb")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path,
                        help="Explicit bounded source train/validation selection; required for --smoke")
    # The shared E002a assembly accepts these fields; no point data is opened for either supported arm.
    parser.set_defaults(indexed_point_cache=None, point_workers=e002a_run.DEFAULT_POINT_WORKERS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())
