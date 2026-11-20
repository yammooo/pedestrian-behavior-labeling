"""Directly evaluate the completed ROAD E005 checkpoint on frozen LOKI validation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import (load_predictions, predict, prediction_metrics,
                                            save_predictions)
from pedestrian_behavior.experiments import e001b, e001b_run, e004a_analysis, e004a_run
from pedestrian_behavior.models import e005
from pedestrian_behavior.experiments.e005_cache import E005Cache, E005FeatureDataset
from pedestrian_behavior.training import seed_run

from . import e005_run
from .run import write_json


SOURCE, TARGET, ARM, VARIANT = "road-waymo", "loki", "g-rgb-utonia", "bilstm-96"
EXPECTED_SPLITS = {
    "development": {"tracks": 930, "groups": 46, "gt_timesteps": 29626},
    "assessment": {"tracks": 931, "groups": 47, "gt_timesteps": 30605},
}


def _load_source(run):
    status = json.loads((run / "status.json").read_text())
    config = json.loads((run / "config.json").read_text())
    path = run / "best.pt"
    provenance = json.loads((run / "provenance.json").read_text())
    if (status.get("status") != "complete" or status.get("kind") != "exploratory-source-validation"
            or config.get("experiment") != "E005" or config.get("source") != SOURCE
            or config.get("configuration", config.get("arm")) != ARM
            or config.get("variant") != VARIANT or config.get("held_out_tests") is not False
            or config.get("smoke_limits") is not None):
        raise ValueError("Expected a completed full ROAD E005 g-rgb-utonia BiLSTM-96 run")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    selected = checkpoint["selection"]
    if (json.loads(json.dumps(checkpoint["config"])) != config or checkpoint["epoch"] != selected["epoch"]
            or selected != provenance["selection"]
            or checkpoint["epoch"] != provenance["selected_checkpoint_epoch"]):
        raise ValueError("E005 best.pt differs from the recorded ROAD selection")
    return path, config, checkpoint, provenance


def _frozen_data(outputs_root, checkpoint, source_provenance, cache_path):
    e001_root, e001b_root = outputs_root / "experiments/E001", outputs_root / "experiments/E001b"
    frozen = e001b_run.assemble(SOURCE, "geometry", "B", e001_root=e001_root, e001b_root=e001b_root)
    stats = checkpoint.get("normalization")
    if stats != frozen["normalization"]:
        raise ValueError("ROAD checkpoint normalization differs from the frozen ROAD source statistics")
    saved = source_provenance["datasets"].get(SOURCE)
    if not saved or any(saved.get(key) != frozen["datasets"][SOURCE][key]
            for key in ("setup_sha256", "collection_sha256")):
        raise ValueError("E005 source run does not match the frozen ROAD E001 population")
    if (source_provenance["datasets"][SOURCE]["bbox"]["manifest_sha256"]
            != frozen["datasets"][SOURCE]["bbox"]["manifest_sha256"]):
        raise ValueError("E005 source run does not match the frozen ROAD E001b extension")

    cache = E005Cache(cache_path)
    population = cache.record["population"]
    loki = frozen["datasets"][TARGET]
    if (cache.record.get("dataset") != TARGET or population.get("setup_sha256") != loki["setup_sha256"]
            or population.get("manifest_sha256") != loki["collection_sha256"]):
        cache.close()
        raise ValueError("LOKI E005 cache does not match the frozen E001 collection and setup")
    source_encoders = checkpoint["config"]["cache"]["provenance"]
    target_encoders = cache.record["provenance"]
    identity_fields = {"DINO": ("revision", "checkpoint_sha256"),
        "Tiny-Utonia": ("code_revision", "checkpoint_revision", "checkpoint_sha256")}
    contract_fields = {"DINO": ("crop", "output"),
        "Tiny-Utonia": ("scale", "grid_size_scaled", "stages", "selection")}
    for name, fields in identity_fields.items():
        source_identity, target_identity = source_encoders.get(name, {}), target_encoders.get(name, {})
        if any(source_identity.get(key) != target_identity.get(key) for key in fields):
            cache.close()
            raise ValueError(f"ROAD checkpoint and LOKI cache use different {name} encoder assets")
        for key in contract_fields[name]:
            if key in source_identity and key in target_identity and source_identity[key] != target_identity[key]:
                cache.close()
                raise ValueError(f"ROAD checkpoint and LOKI cache use different {name} {key} contract")

    baselines = {}
    for split in ("training", "validation"):
        baselines[split] = e001b.dataset_from_extension(loki["collection_path"], loki["setup_path"],
            e001b_root / "bbox-native" / TARGET, split, "geometry", stats["baseline"], stats["bbox"])
    expected = {entry["locator"]: entry for dataset in baselines.values() for entry in dataset.entries}
    if set(expected) != set(cache.tracks) or any(
            any(entry[key] != cache.tracks[locator][key]
                for key in ("archive", "group", "split", "slots"))
            or entry["e001_sha256"] != cache.tracks[locator]["sha256"]
            for locator, entry in expected.items()):
        cache.close()
        raise ValueError("LOKI E005 cache track identities differ from the frozen E001 train/validation population")
    return frozen, cache, baselines


def _partitions(reference, validation):
    plan = json.loads((reference / "subset-plans.json").read_text())
    split = plan.get("validation_split", {})
    development, assessment, identity = e004a_run.split_development_assessment(validation)
    if (split.get("development_groups") != identity["development_groups"]
            or split.get("assessment_groups") != identity["assessment_groups"]):
        raise ValueError("E004a reference plan and current frozen LOKI validation partition differ")
    partitions = {"development": development, "assessment": assessment}
    for name, rows in partitions.items():
        summary = e004a_run._population_summary(rows)
        expected = EXPECTED_SPLITS[name]
        actual = {"tracks": summary["tracks"], "groups": summary["groups"],
                  "gt_timesteps": summary["gt_timesteps"]}
        if actual != expected:
            raise ValueError(f"Frozen LOKI {name} partition changed: {actual}")
    return partitions, identity


def _matched_historical(reference, current_by_locator, partitions):
    directory = reference / "zero-source-head"
    dev = load_predictions(directory / "development-predictions.npz")
    assessment = load_predictions(directory / "assessment-predictions.npz")
    historical = dev + assessment
    if len(historical) != len(current_by_locator) or {row["locator"] for row in historical} != set(current_by_locator):
        raise ValueError("Historical E004a zero-source-head predictions do not cover the frozen LOKI validation tracks")
    for name, rows in (("development", dev), ("assessment", assessment)):
        if {row["locator"] for row in rows} != {row["locator"] for row in partitions[name]}:
            raise ValueError(f"Historical E004a {name} predictions do not match the pinned partition")
    for row in historical:
        current = current_by_locator[row["locator"]]
        if (not np.array_equal(row["targets"], current["targets"])
                or not np.array_equal(row["gt_valid"], current["gt_valid"])):
            raise ValueError(f"Historical prediction targets changed for {row['locator']}")
    return {"full_validation": prediction_metrics(historical, 4),
            "development": prediction_metrics(dev, 4),
            "assessment": prediction_metrics(assessment, 4)}, dev, assessment


def _write_partition(output, name, tracks):
    folder = output / name
    folder.mkdir()
    save_predictions(folder / "predictions.npz", tracks)
    metrics = prediction_metrics(tracks, 4)
    write_json(folder / "metrics.json", {"split": name, "target_source": TARGET, "primary": metrics})
    return metrics


def _progress(batches, total):
    seen = 0
    for batch in batches:
        seen += len(batch["locators"])
        if seen == total or seen // 128 > (seen - len(batch["locators"])) // 128:
            print(f"LOKI validation tracks: {seen}/{total}", flush=True)
        yield batch


def run(args):
    outputs_root, output = args.outputs_root.resolve(), args.output.resolve()
    source_run, cache_path, reference = args.source_run.resolve(), args.cache.resolve(), args.reference.resolve()
    allowed = outputs_root / "experiments/E005/transfer"
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(allowed) or output == allowed or output.is_relative_to(source_run):
        raise ValueError("Use a fresh output directory under outputs/experiments/E005/transfer")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    path, config, checkpoint, source_provenance = _load_source(source_run)
    frozen, cache, baselines = _frozen_data(outputs_root, checkpoint, source_provenance, cache_path)
    output.mkdir(parents=True)
    started = time.monotonic()
    status = {"status": "incomplete", "held_out_tests": False,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        deterministic = seed_run(0)
        torch.set_num_threads(2)
        dataset = E005FeatureDataset(baselines["validation"], cache, ARM)
        max_tracks = int(config.get("batching", {}).get("max_tracks", 2))
        logical = e005_run.SpatialBatches(dataset, {"max_tracks": max_tracks}, training=False,
                                          prefetch=True, prefetch_workers=2)
        model = e005.build_model(ARM, config["profile"]).to(args.device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        inference_started = time.monotonic()
        tracks = predict(model, _progress(logical.predictions(), len(dataset)), args.device,
                         forward_batch=lambda current, batch, _: current(batch, mode="ordinary"))
        inference_seconds = time.monotonic() - inference_started
        expected_locators = {entry["locator"] for entry in baselines["validation"].entries}
        if len(tracks) != 1861 or {row["locator"] for row in tracks} != expected_locators:
            raise ValueError("Predictions do not cover the complete frozen LOKI validation split")
        groups = {entry["locator"]: entry["group"] for entry in baselines["validation"].entries}
        for row in tracks:
            row["group"] = groups[row["locator"]]
            row.setdefault("conditions", {})
        by_locator = {row["locator"]: row for row in tracks}
        partitions, split_identity = _partitions(reference, tracks)
        partition_tracks = {name: [by_locator[row["locator"]] for row in rows]
                            for name, rows in partitions.items()}

        full_metrics = _write_partition(output, "loki-validation", tracks)
        new_metrics = {name: _write_partition(output, name, rows)
                       for name, rows in partition_tracks.items()}
        historical, old_dev, old_assessment = _matched_historical(reference, by_locator, partition_tracks)
        coverage = e004a_analysis.analyze_checkpoint(partition_tracks["development"],
            partition_tracks["assessment"], output / "coverage/e005", error_targets=e004a_analysis.ERROR_TARGETS)
        historical_coverage = e004a_analysis.analyze_checkpoint(old_dev, old_assessment,
            output / "coverage/e004a-zero-source-head", error_targets=e004a_analysis.ERROR_TARGETS)
        summary = {"training_source": SOURCE, "target_source": TARGET, "architecture": VARIANT,
            "configuration": ARM, "selected_epoch": checkpoint["epoch"],
            "checkpoint_sha256": checksum(path), "loki_cache_manifest_sha256": cache.manifest_sha256,
            "held_out_tests": False, "selection_source": SOURCE,
            "normalization_source": SOURCE, "inference_seconds": inference_seconds,
            "track_weighted_macro_f1": {"e005_full_validation": full_metrics["track_weighted"]["macro_f1"],
                "e005_development": new_metrics["development"]["track_weighted"]["macro_f1"],
                "e005_assessment": new_metrics["assessment"]["track_weighted"]["macro_f1"],
                "e004a_zero_full_validation": historical["full_validation"]["track_weighted"]["macro_f1"],
                "e004a_zero_development": historical["development"]["track_weighted"]["macro_f1"],
                "e004a_zero_assessment": historical["assessment"]["track_weighted"]["macro_f1"]},
            "matched_e004a_zero_source_head": historical,
            "coverage": {"e005": coverage, "e004a_zero_source_head": historical_coverage},
            "validation_split": split_identity}
        write_json(output / "comparison.json", summary)
        provenance = {"command": getattr(sys, "orig_argv", sys.argv),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
            "device": str(args.device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device) if args.device == "cuda" else None,
            "determinism": deterministic, "source_run": str(source_run), "source_checkpoint": str(path),
            "source_checkpoint_sha256": checksum(path), "source_run_revision": source_provenance["revision"],
            "source_normalization": checkpoint["normalization"], "source_datasets": source_provenance["datasets"],
            "loki_cache": str(cache.root), "loki_cache_manifest_sha256": cache.manifest_sha256,
            "reference_attempt": str(reference), "reference_subset_plan_sha256": checksum(reference / "subset-plans.json"),
            "reference_zero_predictions": {"development": checksum(reference / "zero-source-head/development-predictions.npz"),
                "assessment": checksum(reference / "zero-source-head/assessment-predictions.npz")},
            "validation_split": split_identity, "inference_seconds": inference_seconds,
            "elapsed_seconds": time.monotonic() - started}
        write_json(output / "config.json", {"experiment": "E005", "kind": "ROAD-to-LOKI-direct-validation",
            "configuration": ARM, "architecture": VARIANT, "training_source": SOURCE,
            "target_source": TARGET, "split": "validation", "source_config": config,
            "batching": {"max_tracks": max_tracks, "prefetch_microbatches": 2}, "held_out_tests": False})
        write_json(output / "provenance.json", provenance)
        status.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
                      inference_seconds=inference_seconds, tracks=len(tracks))
    except BaseException as error:
        status.update(status="failed", failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)
        cache.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="Completed full LOKI E005 cache")
    parser.add_argument("--reference", type=Path, required=True, help="E004a early-check attempt with matched zero-head predictions")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
