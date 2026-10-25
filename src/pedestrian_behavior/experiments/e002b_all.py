"""One-source E002b launcher: independent geometry/RGB, then shared geometry/LiDAR."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from . import e002a_run, e002b, e002b_group, e002b_independent, e002b_run
from .run import write_json


def run(args):
    output = Path(args.output)
    if not output.resolve().is_relative_to(args.outputs_root.resolve() / "experiments/E002b"):
        raise ValueError("All-model output must be under outputs-root/experiments/E002b")
    e002b_group.profile_for(argparse.Namespace(profile=args.shared_profile, device=args.device))
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "incomplete", "source": args.source, "arms": ["g", "g-rgb", "g-dditr"],
              "variants": list(e002b.VARIANTS), "held_out_tests": False}
    write_json(output / "status.json", status)
    try:
        independent = argparse.Namespace(source=args.source, arms=e002b_run.ARMS, variants=e002b.VARIANTS,
            device=args.device, profile=args.independent_profile, output=output / "independent",
            outputs_root=args.outputs_root, probe=args.probe, rgb_cache_root=args.rgb_cache_root,
            rgb_preparation=args.rgb_preparation)
        e002b_independent.run(independent)
        grouped = argparse.Namespace(source=args.source, device=args.device, profile=args.shared_profile,
            output=output / "shared-lidar", outputs_root=args.outputs_root, probe=args.probe,
            rgb_cache_root=args.rgb_cache_root, indexed_point_cache=args.indexed_point_cache,
            point_workers=args.point_workers, smoke=False, smoke_selection=None)
        result = e002b_group.run(grouped)
        status["status"] = "complete"
        status["completed_utc"] = datetime.now(timezone.utc).isoformat()
        return result
    except BaseException as error:
        status["status"] = "failed"
        status["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        write_json(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("road-waymo", "loki"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--independent-profile", type=Path, required=True,
                        help="Completed E002b geometry/RGB profile")
    parser.add_argument("--shared-profile", type=Path, required=True,
                        help="Completed shared-LiDAR E002b profile")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/E002b/runs") / f"all-{timestamp}")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path, required=True)
    parser.add_argument("--indexed-point-cache", type=Path)
    parser.add_argument("--point-workers", type=int, default=e002a_run.DEFAULT_POINT_WORKERS)
    run(parser.parse_args())
