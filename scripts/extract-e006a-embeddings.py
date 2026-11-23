"""Export E006a pre-head validation embeddings after exact prediction replay."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import load_predictions
from pedestrian_behavior.experiments import e005_run, e005_transfer
from pedestrian_behavior.training import seed_run


DEFAULT_RUN = Path("outputs/experiments/E005/runs/joint-g-rgb-utonia-s0-20261122-001910")
DEFAULT_PROFILE = Path("outputs/experiments/E005/benchmarks/joint-20261121-174411/benchmark.json")
DEFAULT_LOKI_CACHE = Path("outputs/experiments/E005/loki-cache-full-20261117")
DEFAULT_ROAD_CACHE = Path("/run/media/user20/F47C60057C5FC152/projects/e005-road-cache-full-20261119-224937")
SOURCES = ("road-waymo", "loki")
LOGIT_RTOL, LOGIT_ATOL = 1e-4, 1e-5


def _verify_run(run):
    status = json.loads((run / "status.json").read_text())
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint_config = json.loads(json.dumps(checkpoint.get("config")))
    if (status.get("status") != "complete" or config.get("experiment") != "E005"
            or config.get("source") != "joint" or config.get("configuration") != "g-rgb-utonia"
            or config.get("held_out_tests") is not False or config.get("smoke_limits") is not None
            or checkpoint_config != config or checkpoint.get("epoch") != provenance.get("selected_checkpoint_epoch")
            or checkpoint.get("selection") != provenance.get("selection")
            or checkpoint.get("epoch") != checkpoint.get("selection", {}).get("epoch")):
        raise ValueError("Expected the completed full E006a joint best checkpoint")
    return config, checkpoint, checksum(checkpoint_path)


def _verify_logits(records, predictions):
    expected = {row["locator"]: row for row in predictions}
    if len(expected) != len(predictions) or {row["locator"] for row in records} != expected.keys():
        raise ValueError("Extracted validation tracks differ from saved E006a predictions")
    maximum = 0.
    for row in records:
        prior = expected[row["locator"]]
        logits = row["source_logits"]
        if (row["group"] != prior["group"] or Path(row["archive"]).name != Path(prior["archive"]).name
                or not np.array_equal(row["targets"], prior["targets"])
                or not np.array_equal(row["gt_valid"], prior["gt_valid"])
                or any(not np.array_equal(row["source_flags"][name], prior["source_flags"][name])
                       for name in ("rgb_valid", "lidar_valid"))
                or logits.shape != prior["logits"].shape
                or not np.allclose(logits, prior["logits"], rtol=LOGIT_RTOL, atol=LOGIT_ATOL)):
            raise ValueError(f"Validation replay differs from saved E006a predictions for {row['locator']}")
        maximum = max(maximum, float(np.abs(logits - prior["logits"]).max()))
    return maximum


def run(args):
    run_dir = args.run.resolve()
    config, checkpoint, checkpoint_sha = _verify_run(run_dir)
    if args.output.suffix != ".npz" or args.output.exists():
        raise ValueError("Choose a fresh .npz output path")
    if not torch.cuda.is_available() and args.device == "cuda":
        raise ValueError("CUDA requested but unavailable")
    seed_run(int(config["settings"]["seed"]))
    torch.set_num_threads(2)
    prepared = e005_run.assembly(SimpleNamespace(
        outputs_root=args.outputs_root, source="joint", arm="g-rgb-utonia", cache=None,
        loki_cache=args.loki_cache, road_cache=args.road_cache, joint_profile=args.profile,
        aalto_fast=False, device=args.device))
    try:
        if prepared["config"]["profile"] != config["profile"]:
            raise ValueError("Joint calibration profile differs from the selected E006a run")
        model = prepared["build_model"]().to(args.device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()

        output = {key: [] for key in ("features", "datasets", "groups", "locators", "slots", "labels",
                                      "rgb_valid", "lidar_valid")}
        replay_deltas = {}
        for source in SOURCES:
            dataset = prepared["dataset"](source, "validation")
            batches = prepared["batches"](dataset, seed=int(config["settings"]["seed"]))
            records = e005_transfer._extract_embedding_records(model, batches, args.device, mode=None)
            reference = load_predictions(run_dir / f"{source}-validation" / "predictions.npz")
            replay_deltas[source] = _verify_logits(records, reference)
            for track in records:
                valid = track["gt_valid"].astype(bool)
                if not valid.any() or not np.isfinite(track["features"]).all():
                    raise ValueError(f"Invalid accepted-GT embeddings for {track['locator']}")
                count = int(valid.sum())
                output["features"].append(track["features"][valid].astype(np.float32, copy=False))
                output["datasets"].append(np.full(count, source))
                output["groups"].append(np.full(count, track["group"]))
                output["locators"].append(np.full(count, track["locator"]))
                output["slots"].append(np.flatnonzero(valid).astype(np.int64))
                output["labels"].append(track["targets"][valid].astype(np.int64, copy=False))
                for field, flag in (("rgb_valid", "rgb_valid"), ("lidar_valid", "lidar_valid")):
                    output[field].append(track["source_flags"][flag][valid].astype(bool, copy=False))

        arrays = {key: np.concatenate(values) for key, values in output.items()}
        count = len(arrays["labels"])
        if (arrays["features"].shape != (count, 128) or arrays["features"].dtype != np.float32
                or not np.isfinite(arrays["features"]).all()
                or not np.isin(arrays["labels"], (0, 1, 2, 3)).all()
                or set(arrays["datasets"]) != set(SOURCES)):
            raise ValueError("Invalid joint E006a embedding export")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        print({"output": str(args.output.resolve()), "rows": count, "checkpoint_sha256": checkpoint_sha,
               "max_abs_logit_delta": replay_deltas}, flush=True)
    finally:
        prepared["close"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--loki-cache", type=Path, default=DEFAULT_LOKI_CACHE)
    parser.add_argument("--road-cache", type=Path, default=DEFAULT_ROAD_CACHE)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True, help="Fresh pickle-free compressed .npz path")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
