"""Sequential E002a geometry+RGB repeats on source validation, seeds 1–4."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

import torch

from .e002a_features import extraction_identity, load_configuration
from .e002a_run import FUSION_PROTOCOL, SETTINGS, resource_code, validate_profile
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


def validate_inputs(args):
    """Reject stale execution assets before the fresh sweep directory is created."""
    run_root = (args.outputs_root / "experiments/E002a/runs").resolve()
    output = args.output.resolve()
    if output == run_root or not output.is_relative_to(run_root):
        raise ValueError("Sweep output must be a fresh child of outputs-root/experiments/E002a/runs")
    if output.exists():
        raise FileExistsError(output)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    if not args.profile.is_file():
        raise FileNotFoundError(f"Calibration profile missing: {args.profile}")
    profile = json.loads(args.profile.read_text())
    if profile.get("status") != "complete" or profile.get("device") != args.device:
        raise ValueError("Completed calibration on the requested device required")
    if profile.get("code") != resource_code():
        raise ValueError("Execution code changed since resource calibration")
    if profile.get("torch") != str(torch.__version__) or profile.get("cuda") != torch.version.cuda:
        raise ValueError("PyTorch/CUDA backend differs from resource calibration")
    if args.device == "cuda" and profile.get("gpu") != torch.cuda.get_device_name():
        raise ValueError("Resource profile was calibrated on different GPU hardware")
    validate_profile(profile.get("profile", {}))

    if not args.rgb_preparation.is_file():
        raise FileNotFoundError(f"RGB preparation record missing: {args.rgb_preparation}")
    preparation = json.loads(args.rgb_preparation.read_text())
    config, _ = load_configuration(args.probe)
    for key in ("waymo_root", "loki_root"):
        if config.get(key) and not Path(config[key]).is_dir():
            raise FileNotFoundError(f"Native source root unavailable; mount the dataset partition: {config[key]}")
    identity = extraction_identity(args.device, config)
    if preparation.get("status") != "complete" or preparation.get("extraction") != identity:
        raise ValueError("Matching completed RGB cache preparation required")
    if Path(preparation.get("config", {}).get("rgb_cache_root", "")).resolve() != args.rgb_cache_root.resolve():
        raise ValueError("RGB preparation points to a different cache root")

    tracks = preparation.get("tracks", [])
    for source in SOURCES:
        setup_path = args.outputs_root / "experiments/E001/data-setup" / source / "setup.json"
        if not setup_path.is_file():
            raise FileNotFoundError(f"Frozen source setup missing: {setup_path}")
        setup = json.loads(setup_path.read_text())
        if setup.get("status") != "complete":
            raise ValueError(f"Completed frozen source setup required: {setup_path}")
        expected = {entry["locator"] for entry in setup["tracks"]
                    if not entry["exclusions"] and entry["split"] in ("training", "validation")}
        rows = [row for row in tracks if row.get("dataset") == source]
        actual = {row.get("locator") for row in rows}
        if not expected or actual != expected or len(rows) != len(expected):
            raise ValueError(f"Full frozen {source} RGB train/validation population required")
        if any("rgb_valid" not in row for row in rows):
            raise ValueError(f"RGB cache preparation is incomplete for {source}")

    index = args.rgb_cache_root.resolve() / "index.sqlite3"
    entries = index.parent / "entries"
    if not index.is_file() or not entries.is_dir():
        raise FileNotFoundError(f"Prepared RGB cache index/entries missing: {args.rgb_cache_root}")
    connection = sqlite3.connect(index.as_uri() + "?mode=ro", uri=True)
    try:
        if connection.execute("SELECT * FROM format").fetchall() != [(1,)]:
            raise ValueError("Unsupported prepared RGB cache index")
        cached = connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    finally:
        connection.close()
    if cached < len(tracks):
        raise ValueError("Prepared RGB cache has fewer entries than its completed preparation record")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/experiments/E002a/runs") / ("g-rgb-seeds-1-4-" + timestamp))
    parser.add_argument("--profile", type=Path, required=True,
                        help="Current fixed-limit CUDA profile for this E002a execution code")
    parser.add_argument("--rgb-preparation", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    return parser


def run(args):
    validate_inputs(args)
    options = ["--profile", str(args.profile), "--rgb-preparation", str(args.rgb_preparation),
               "--outputs-root", str(args.outputs_root), "--probe", str(args.probe),
               "--rgb-cache-root", str(args.rgb_cache_root)]
    run_sweep(args.output, attempts(), lambda row, output, device: attempt_command(row, output, device, options),
              jobs=1, seeds=SEEDS, device=args.device,
              settings={"arm": ARM, "epochs": SETTINGS["epochs"], "patience": SETTINGS["patience"],
                        "held_out_tests": False, "kind": "exploratory-source-validation",
                        "fusion_protocol": FUSION_PROTOCOL, "profile": str(args.profile)})


def main():
    run(build_parser().parse_args())
