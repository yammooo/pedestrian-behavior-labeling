"""Paired ROAD validation ablation of E005's LiDAR inside-box cue."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum, load_track
from pedestrian_behavior.evaluation import load_predictions, prediction_metrics, save_predictions, track_cross_entropy
from pedestrian_behavior.experiments import e001
from pedestrian_behavior.experiments import e005_run
from pedestrian_behavior.training import reload_checkpoint, seed_run


DEFAULT_RUN = Path("outputs/experiments/E005/runs/road-waymo-g-rgb-utonia-s0-20261120-030721")
DEFAULT_CACHE = Path("/run/media/user20/F47C60057C5FC152/projects/e005-road-cache-full-20261119-224937")
DEFAULT_OUTPUT = Path("outputs/experiments/E006b/diagnostics/road-e005-inside-box-cue")
LOGIT_ATOL = LOGIT_RTOL = 1e-5


def _canonical(value):
    return json.loads(json.dumps(value))


def _read_source(run):
    status = json.loads((run / "status.json").read_text())
    config = json.loads((run / "config.json").read_text())
    provenance = json.loads((run / "provenance.json").read_text())
    if (status.get("status") != "complete" or status.get("kind") != "exploratory-source-validation"
            or config.get("experiment") != "E005" or config.get("source") != "road-waymo"
            or config.get("configuration") != "g-rgb-utonia" or config.get("variant") != "bilstm-96"
            or config.get("smoke_limits") is not None):
        raise ValueError("Expected the completed, full ROAD E005 g-rgb-utonia run")
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (_canonical(checkpoint.get("config")) != config
            or _canonical(checkpoint.get("normalization")) != _canonical(provenance.get("source_statistics"))
            or checkpoint.get("epoch") != checkpoint.get("selection", {}).get("epoch")
            or checkpoint.get("selection") != provenance.get("selection")
            or checkpoint.get("epoch") != provenance.get("selected_checkpoint_epoch")):
        raise ValueError("Pinned E005 checkpoint differs from its saved selection evidence")
    return config, provenance, checkpoint, checksum(checkpoint_path)


def _prediction_rows(model, batches, device):
    """Run both arms over one read of each CPU batch and count the original cue."""
    model.eval()
    baseline, ablated = [], []
    totals = {"stage3_tokens": 0, "stage3_inside_tokens": 0,
              "stage5_tokens": 0, "stage5_inside_tokens": 0,
              "gt_frames_with_inside_token": 0, "gt_tracks_with_inside_token": 0}
    with torch.inference_mode():
        for batch in batches:
            levels = batch["lidar_levels"]
            if len(levels) != 2:
                raise ValueError("Expected E005 stage3/stage5 token lists")
            changed_levels = []
            frame_has_inside = []
            for level_index, frames in enumerate(levels):
                width = (77, 221)[level_index]
                copied = []
                for tokens in frames:
                    if tokens.ndim != 2 or tokens.shape[1] != width or not torch.isfinite(tokens).all():
                        raise ValueError("Malformed E005 spatial tokens")
                    inside = tokens[:, -1] > 0
                    key = "stage3" if level_index == 0 else "stage5"
                    totals[f"{key}_tokens"] += len(tokens)
                    totals[f"{key}_inside_tokens"] += int(inside.sum())
                    copied.append(tokens.clone())
                    copied[-1][:, -1] = 0
                    if not torch.equal(copied[-1][:, :-1], tokens[:, :-1]) or (copied[-1][:, -1] != 0).any():
                        raise AssertionError("Cue ablation changed more than the final cue column")
                changed_levels.append(copied)

            max_length = int(batch["lengths"].max())
            for i, length in enumerate(batch["lengths"].tolist()):
                flags = []
                for j in range(length):
                    frame = i * max_length + j
                    flags.append(any(bool(level[frame].numel() and (level[frame][:, -1] > 0).any())
                                     for level in levels))
                frame_has_inside.append(flags)
                accepted = batch["gt_valid"][i, :length].tolist()
                totals["gt_frames_with_inside_token"] += sum(ok and flag for ok, flag in zip(accepted, flags))
                totals["gt_tracks_with_inside_token"] += int(any(ok and flag for ok, flag in zip(accepted, flags)))

            original_logits = model(batch)
            masked = dict(batch)
            masked["lidar_levels"] = tuple(changed_levels)
            for key in ("geometry", "flags", "targets", "gt_valid", "lengths", "padding_mask"):
                if not torch.equal(masked[key], batch[key]):
                    raise AssertionError(f"Cue ablation changed {key}")
            zero_logits = model(masked)
            for logits in (original_logits, zero_logits):
                track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device),
                                    batch["padding_mask"].to(device))
            for i, length in enumerate(batch["lengths"].tolist()):
                record = batch["prediction_metadata"][i] | {
                    "logits": original_logits[i, :length].cpu().numpy().astype(np.float32),
                    "targets": batch["targets"][i, :length].numpy().copy(),
                    "gt_valid": batch["gt_valid"][i, :length].numpy().copy(),
                    "locator": batch["locators"][i], "archive": batch["archives"][i]}
                baseline.append(record)
                ablated.append(record | {"logits": zero_logits[i, :length].cpu().numpy().astype(np.float32)})
    if not baseline:
        raise ValueError("Empty ROAD validation split")
    return baseline, ablated, totals


def _attach_evaluation_metadata(tracks, prepared):
    for track in tracks:
        metadata, arrays = load_track(track["archive"])
        conditions, flags = prepared["conditions"](metadata, arrays, track, "road-waymo")
        track["conditions"] = conditions
        track["source_flags"] = track.get("source_flags", {}) | flags
    return tracks


def _verify_baseline(current, saved, saved_metrics):
    current_by_id = {row["locator"]: row for row in current}
    saved_by_id = {row["locator"]: row for row in saved}
    if len(current_by_id) != len(current) or current_by_id.keys() != saved_by_id.keys():
        raise ValueError("Replayed validation tracks differ from saved E005 predictions")
    max_abs = 0.
    for locator, row in current_by_id.items():
        prior = saved_by_id[locator]
        if row.get("group") != prior.get("group"):
            raise ValueError(f"Validation group changed for {locator}")
        for key in ("targets", "gt_valid"):
            if not np.array_equal(row[key], prior[key]):
                raise ValueError(f"Validation {key} changed for {locator}")
        if row["source_flags"].keys() != prior["source_flags"].keys():
            raise ValueError(f"Validation source flags changed for {locator}")
        for key in row["source_flags"]:
            if not np.array_equal(row["source_flags"][key], prior["source_flags"][key]):
                raise ValueError(f"Validation source flag {key} changed for {locator}")
        logits = np.asarray(row["logits"], dtype=np.float32)
        prior_logits = np.asarray(prior["logits"], dtype=np.float32)
        if logits.shape != prior_logits.shape:
            raise ValueError(f"Validation logits shape changed for {locator}")
        max_abs = max(max_abs, float(np.max(np.abs(logits - prior_logits))))
        if not np.allclose(logits, prior_logits, rtol=LOGIT_RTOL, atol=LOGIT_ATOL):
            raise ValueError(f"Baseline logits do not reproduce saved validation predictions ({locator})")
    metric = prediction_metrics(current, len(e001.CLASSES))
    for key in ("track_ce",):
        if not np.isclose(metric[key], saved_metrics["primary"][key], rtol=1e-9, atol=1e-10):
            raise ValueError(f"Replayed baseline {key} differs from saved validation metrics")
    if not np.isclose(metric["track_weighted"]["macro_f1"],
                      saved_metrics["primary"]["track_weighted"]["macro_f1"], rtol=0, atol=1e-12):
        raise ValueError("Replayed baseline macro-F1 differs from saved validation metrics")
    return metric, max_abs


def _score(metrics):
    return {"track_weighted_macro_f1": metrics["track_weighted"]["macro_f1"],
            "track_ce": metrics["track_ce"], "class_f1": metrics["track_weighted"]["f1"],
            "class_support_tracks": metrics["denominators"]["class_tracks"],
            "denominators": metrics["denominators"]}


def run(args):
    run = args.source_run.resolve()
    cache_path = args.cache.resolve()
    output = args.output.resolve()
    if output == run or output.is_relative_to(run) or output.is_relative_to(cache_path):
        raise ValueError("Output must be outside the source run and E005 cache")
    config, provenance, checkpoint, checkpoint_sha256 = _read_source(run)
    seed_run(config["settings"]["seed"])
    torch.set_num_threads(2)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if config["execution_profile"] == "aalto-fast":
            e005_run.enable_aalto_fast(config["configuration"], args.device)

    assembly_args = SimpleNamespace(source=config["source"], arm=config["configuration"], device=args.device,
        outputs_root=args.outputs_root.resolve(), cache=cache_path, loki_cache=None, road_cache=None,
        joint_profile=None, aalto_fast=config["execution_profile"] == "aalto-fast",
        allocator_cap_bytes=config.get("allocator_cap_bytes"))
    prepared = e005_run.assembly(assembly_args)
    try:
        if (prepared["config"]["profile"] != config["profile"]
                or prepared["config"]["batching"] != config["batching"]
                or prepared["config"]["cache"]["manifest_sha256"] != config["cache"]["manifest_sha256"]):
            raise ValueError("Current E005 cache, profile, or batching differs from the pinned run")
        model = e005_run.e005.build_model(config["configuration"], config["profile"]).to(args.device)
        reload_checkpoint(run / "best.pt", model, args.device)
        dataset = prepared["dataset"]("road-waymo", "validation")
        batches = prepared["batches"](dataset, seed=config["settings"]["seed"])
        current, zero, cue_counts = _prediction_rows(model, batches, torch.device(args.device))
        _attach_evaluation_metadata(current, prepared)
        _attach_evaluation_metadata(zero, prepared)

        saved_predictions = load_predictions(run / "road-waymo-validation" / "predictions.npz")
        saved_metrics = json.loads((run / "road-waymo-validation" / "metrics.json").read_text())
        baseline_metrics, max_abs = _verify_baseline(current, saved_predictions, saved_metrics)
        zero_metrics = prediction_metrics(zero, len(e001.CLASSES))
        if output.exists():
            raise FileExistsError(f"Use a fresh output directory: {output}")
        output.mkdir(parents=True)
        save_predictions(output / "zero-inside-box-predictions.npz", zero)
        summary = {
            "experiment": "E006b-preflight: E005 inside-box cue ablation",
            "source_run": str(run), "source_checkpoint_sha256": checkpoint_sha256,
            "selected_epoch": checkpoint["epoch"], "source_cache": str(cache_path),
            "source_cache_manifest_sha256": prepared["config"]["cache"]["manifest_sha256"],
            "device": args.device, "classes": e001.CLASSES,
            "intervention": "Set only the final inside_pedestrian_box cue column to zero for stage3 and stage5 tokens; retain all tracks, other token values, branch flags, labels, and batches.",
            "baseline_replay_check": {"passed": True, "track_count": len(current),
                "max_absolute_logit_difference": max_abs, "logit_atol": LOGIT_ATOL,
                "logit_rtol": LOGIT_RTOL, "groups_labels_masks_source_flags_match": True},
            "cue_support": cue_counts,
            "baseline": _score(baseline_metrics), "zero_inside_box": _score(zero_metrics),
            "delta_zero_minus_baseline": {
                "track_weighted_macro_f1": zero_metrics["track_weighted"]["macro_f1"] - baseline_metrics["track_weighted"]["macro_f1"],
                "track_ce": zero_metrics["track_ce"] - baseline_metrics["track_ce"]},
            "predictions": "zero-inside-box-predictions.npz"}
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2, allow_nan=False))
        return summary
    finally:
        prepared["close"]()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
