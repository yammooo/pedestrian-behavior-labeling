"""E004c adaptation-depth fits on the frozen E004a ROAD→LOKI draws."""

from datetime import datetime, timezone
from itertools import chain, count
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import wandb

from pedestrian_behavior.data.bbox import load_bbox, normalization_sample
from pedestrian_behavior.data.features import features, fit_normalization
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches
from pedestrian_behavior.data.preparation import checksum, load_track
from pedestrian_behavior.evaluation import load_predictions, predict, prediction_metrics, save_predictions
from pedestrian_behavior.experiments import (e001b, e002a_run, e002f_run, e003b_run, e003c_run,
    e004a_analysis, e004a_run, e004c_analysis, run as shared_run)
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.models import e002f
from pedestrian_behavior.training import fit, reload_checkpoint, seed_run
from pedestrian_behavior import training as shared_training


SOURCE, TARGET, ARCHITECTURE = "road-waymo", "loki", "bilstm-96"
BUDGETS = (25, 100, 1000)
EARLY_REFERENCE = "20261113-082847-737608Z"
PATIENT_UPDATES = 1500
CHECK_STARTS = (1, 5, 10, 25, 50, 100)
SETTINGS = {"seed": 0, "batch_size": 64, "clip_norm": 1., "precision": "float32",
    "optimizer": {"lr": .0003, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8},
    "temporal_lr": .00003}
CLASSES = e003b_run.CLASSES


def _check_updates():
    return chain(CHECK_STARTS, count(200, 100))


def _write_predictions(path, tracks):
    for row in tracks:
        row.setdefault("conditions", {})
        row.setdefault("source_flags", {})
    save_predictions(path, tracks)


def _decorate_groups(tracks, dataset):
    groups = {entry["locator"]: entry["group"] for entry in dataset.entries}
    for row in tracks:
        try:
            row["group"] = groups[row["locator"]]
        except KeyError as error:
            raise ValueError(f"Prediction is outside the frozen LOKI population: {row['locator']}") from error
    return tracks


def _metric_class_f1(metrics):
    return [float(value) for value in metrics["track_weighted"]["f1"]]


def _check_prediction_roundtrip(original, restored):
    left, right = {row["locator"]: row for row in original}, {row["locator"]: row for row in restored}
    if set(left) != set(right):
        raise ValueError("Saved prediction roundtrip changed the track population")
    for locator, row in left.items():
        other = right[locator]
        if (row["archive"] != other["archive"] or row["group"] != other["group"]
                or any(not np.array_equal(row[key], other[key]) for key in ("logits", "targets", "gt_valid"))
                or any(not np.array_equal(row["source_flags"][key], other["source_flags"].get(key))
                    for key in row["source_flags"])):
            raise ValueError(f"Saved prediction roundtrip changed values or provenance: {locator}")


def _read_reference(outputs_root):
    root = Path(outputs_root) / "experiments/E004a/runs" / EARLY_REFERENCE
    plan_path = root / "subset-plans.json"
    status_path = root / "status.json"
    if not plan_path.is_file() or not status_path.is_file():
        raise ValueError(f"Completed E004a early-check reference is missing under {root}")
    status, plan = json.loads(status_path.read_text()), json.loads(plan_path.read_text())
    protocol = plan.get("protocol", {})
    if (status.get("status") != "complete" or status.get("early_checks") is not True
            or status.get("held_out_tests") is not False or plan.get("experiment") != "E004a"
            or protocol.get("source_checkpoint_sha256") != e004a_run.SOURCE_SHA256
            or protocol.get("draw_seeds") != list(range(5)) or protocol.get("fit_seed") != 0
            or protocol.get("early_checks") is not True or protocol.get("patience_updates") != 1500
            or plan.get("population", {}).get("training", {}).get("tracks") != e004a_run.ALL_TRACKS
            or plan.get("population", {}).get("development") != e004a_run.EXPECTED_VALIDATION["development"]
            or plan.get("population", {}).get("assessment") != e004a_run.EXPECTED_VALIDATION["assessment"]):
        raise ValueError("E004a early-check reference plan/status differs from the frozen E004c population")
    by_key = {(row["budget"], row["draw"]): row for row in plan["runs"]}
    expected = {(budget, draw) for budget in BUDGETS for draw in range(5)}
    if not expected.issubset(by_key):
        raise ValueError("E004a reference is missing one or more E004c nested draws")
    return root, plan, by_key, checksum(plan_path)


def _load_caches(outputs_root):
    outputs_root = Path(outputs_root).resolve()
    compact = e004a_run._load_compact_cache(
        outputs_root / "experiments/E004a/features/postnorm-bilstm-96")
    frozen = e002f_run._source_run(outputs_root / e002f_run.SOURCE_RUN, ARCHITECTURE, outputs_root)
    if frozen["sha256"] != e004a_run.SOURCE_SHA256 or compact["manifest"]["normalization"] != frozen["normalization"]:
        raise ValueError("E004a compact cache and pinned E002f source normalization/checkpoint differ")
    pre_root = outputs_root / e002f_run.SOURCE_FEATURES
    pre_training, pre_training_meta = e002f_run._load_cache(pre_root, "training", ARCHITECTURE,
        frozen["sha256"], frozen["normalization"])
    pre_validation, pre_validation_meta = e002f_run._load_cache(pre_root, "validation", ARCHITECTURE,
        frozen["sha256"], frozen["normalization"])
    for split, rows in (("training", pre_training), ("validation", pre_validation)):
        compact_rows = compact["records"][split]
        left = {row["locator"]: row for row in compact_rows}
        right = {row["locator"]: row for row in rows}
        if set(left) != set(right):
            raise ValueError(f"E002f/E004a {split} cache populations differ")
        for locator, row in left.items():
            other = right[locator]
            if (row["archive"] != other["archive"] or not np.array_equal(row["targets"], other["targets"])
                    or not np.array_equal(row["gt_valid"], other["gt_valid"])
                    or row["group"] != other["group"]):
                raise ValueError(f"E002f/E004a {split} cached labels or identity differ: {locator}")
    return compact, frozen, {"training": pre_training, "validation": pre_validation}, {
        "training": pre_training_meta, "validation": pre_validation_meta}


def _load_splits(compact):
    training = sorted(compact["records"]["training"], key=lambda row: row["locator"])
    development, assessment, split_identity = e004a_run.split_development_assessment(compact["records"]["validation"])
    for name, rows in (("development", development), ("assessment", assessment)):
        if e004a_run._population_summary(rows) != e004a_run.EXPECTED_VALIDATION[name]:
            raise ValueError(f"Frozen LOKI {name} population/class support differs from E004a")
    return training, development, assessment, split_identity


def _target_normalization(training_rows, source_baseline):
    """Fit both physical feature statistics from only this draw's selected tracks."""
    entries = {row["locator"]: row for row in source_baseline.entries}
    baseline_samples, bbox_samples = [], []
    for row in training_rows:
        try:
            entry = entries[row["locator"]]
        except KeyError as error:
            raise ValueError(f"Selected E004a track is outside target E001b training: {row['locator']}") from error
        archive = source_baseline.collection / "tracks" / entry["archive"]
        if checksum(archive) != entry["e001_sha256"]:
            raise ValueError("Selected target E001 track changed before normalization")
        metadata, arrays = load_track(archive)
        bbox_path = source_baseline.extension / "tracks" / entry["archive"]
        if checksum(bbox_path) != entry["bbox_sha256"]:
            raise ValueError("Selected target E001b box track changed before normalization")
        _, bbox_arrays = load_bbox(bbox_path, metadata, entry["e001_sha256"])
        baseline_samples.append(features(arrays, "K+T+R"))
        bbox_samples.append(normalization_sample(bbox_arrays))
    if len(baseline_samples) != len(training_rows):
        raise ValueError("Target normalization did not consume exactly the selected tracks")
    return {"baseline": fit_normalization(iter(baseline_samples)),
        "bbox": fit_normalization(iter(bbox_samples))}


def _target_dataset(outputs_root, source_baseline, normalization, supply, split, group_ids=None, locators=None):
    setup_path = Path(outputs_root) / "experiments/E001/data-setup/loki/setup.json"
    baseline = e001b.dataset_from_extension(source_baseline.collection,
        setup_path, source_baseline.extension, split, "geometry",
        normalization["baseline"], normalization["bbox"])
    if group_ids is not None:
        baseline.entries = [entry for entry in baseline.entries if entry["group"] in group_ids]
    if locators is not None:
        baseline.entries = [entry for entry in baseline.entries if entry["locator"] in locators]
    baseline.entries.sort(key=lambda row: row["locator"])
    return e002a_run.FeatureDataset(baseline, supply, TARGET, "g-rgb-dditr")


def _reference_metrics(reference_root, plan, split_identity, output):
    dev_groups = set(split_identity["development_groups"])
    assessment_groups = set(split_identity["assessment_groups"])
    rows = []
    for budget, draw, name in [(0, None, "zero-source-head")] + [
            (budget, draw, f"budget-{budget:04d}-draw-{draw:02d}")
            for draw in range(5) for budget in BUDGETS]:
        fit_dir = reference_root / name
        if not (fit_dir / "selected-metrics.json").is_file():
            raise ValueError(f"Completed E004a selected metrics are missing: {fit_dir}")
        metadata = json.loads((fit_dir / "selected-metrics.json").read_text())
        if (metadata.get("status") != "complete" or metadata.get("held_out_tests") is not False
                or metadata.get("source_checkpoint_sha256") != e004a_run.SOURCE_SHA256
                or metadata.get("budget") != budget or metadata.get("draw") != draw
                or metadata.get("early_checks") is not True):
            raise ValueError(f"E004a head/source reference identity mismatch: {fit_dir}")
        if budget:
            config_path = fit_dir / "config.json"
            if not config_path.is_file():
                raise ValueError(f"E004a head configuration is missing: {fit_dir}")
            config = json.loads(config_path.read_text())
            if (config.get("budget_tracks") != budget or config.get("draw") != draw
                    or config.get("source_checkpoint_sha256") != e004a_run.SOURCE_SHA256
                    or config.get("early_checks") is not True
                    or metadata.get("annotation_access", {}).get("training_subset", {}).get("tracks") != budget):
                raise ValueError(f"E004a head configuration/label access differs from subset plan: {fit_dir}")
        development = load_predictions(fit_dir / "development-predictions.npz")
        assessment = load_predictions(fit_dir / "assessment-predictions.npz")
        if ({row["group"] for row in development} != dev_groups
                or {row["group"] for row in assessment} != assessment_groups
                or len(development) != 930 or len(assessment) != 931):
            raise ValueError(f"E004a predictions differ from its frozen development/assessment split: {fit_dir}")
        dev_metrics = prediction_metrics(development, len(CLASSES))
        assessment_metrics = prediction_metrics(assessment, len(CLASSES))
        saved = metadata["assessment_metrics"]
        if (not math.isclose(assessment_metrics["track_ce"], saved["track_ce"], rel_tol=1e-7, abs_tol=1e-8)
                or not math.isclose(assessment_metrics["track_weighted"]["macro_f1"],
                    saved["track_weighted"]["macro_f1"], rel_tol=1e-7, abs_tol=1e-8)):
            raise ValueError(f"E004a prediction metrics do not reproduce saved results: {fit_dir}")
        coverage = e004a_analysis.analyze_checkpoint(development, assessment,
            output / ("source" if budget == 0 else f"budget-{budget:04d}-draw-{draw:02d}"),
            error_targets=e004a_run.ERROR_TARGETS)
        if budget == 0 and (metadata.get("selected_update") != 0
                or metadata.get("source_checkpoint_sha256") != e004a_run.SOURCE_SHA256):
            raise ValueError("E004a source anchor is not the fixed no-adaptation checkpoint")
        plan_row = None if budget == 0 else plan[(budget, draw)]
        if plan_row is not None and len(plan_row.get("selected_tracks", ())) != budget:
            raise ValueError(f"E004a subset plan has the wrong track count for {budget}/draw-{draw}")
        rows.append({"method": "source" if budget == 0 else "head", "budget": budget, "draw": draw,
            "macro_f1": assessment_metrics["track_weighted"]["macro_f1"],
            "track_ce": assessment_metrics["track_ce"], "class_f1": _metric_class_f1(assessment_metrics),
            "coverage": coverage, "annotation_access": metadata["annotation_access"],
            "development_metrics": dev_metrics, "assessment_metrics": assessment_metrics,
            "selected_update": metadata.get("selected_update", 0),
            "stopping_reason": metadata.get("stopping_reason", "source-anchor"),
            "fit_dir": str(fit_dir), "prediction_references": {
                "development": {"path": str(fit_dir / "development-predictions.npz"),
                    "sha256": checksum(fit_dir / "development-predictions.npz")},
                "assessment": {"path": str(fit_dir / "assessment-predictions.npz"),
                    "sha256": checksum(fit_dir / "assessment-predictions.npz")}},
            "source_checkpoint_sha256": e004a_run.SOURCE_SHA256,
            "subset_tracks": [] if plan_row is None else [row["locator"] for row in plan_row["selected_tracks"]]})
    return rows


class _LimitedLogicalBatches:
    """Take a fixed number of updates from a persistent logical-batch stream."""

    def __init__(self, source, updates):
        self.source, self.updates = source, updates
        if type(updates) is not int or updates <= 0:
            raise ValueError("Logical update count must be positive")

    def __iter__(self):
        for _ in range(self.updates):
            yield self.source.next()

    def microbatches(self, indices):
        return self.source.microbatches(indices)


class _PersistentLogicalBatches:
    def __init__(self, source):
        self.source = source
        self.iterator = iter(source)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.source)
            try:
                return next(self.iterator)
            except StopIteration as error:
                raise ValueError("E004c training loader is empty") from error

    def microbatches(self, indices):
        return self.source.microbatches(indices)


def _multimodal_update(model, batches, optimizer, device, clip_norm, *, on_step=None,
                       step_offset=0, updates):
    from pedestrian_behavior.accumulation import train_accumulated_epoch
    selected = _LimitedLogicalBatches(batches, updates)
    return train_accumulated_epoch(model, selected, optimizer, device, clip_norm,
        on_step=on_step, step_offset=step_offset, peak_lr=SETTINGS["optimizer"]["lr"], warmup_steps=1)


def _temporal_optimizer(model):
    expected = {name for name, _ in model.named_parameters()}
    if (not expected or any(not name.startswith(("temporal.", "final_norm.", "head.")) for name in expected)
            or not any(name.startswith("temporal.") for name in expected)
            or not any(name.startswith("head.") for name in expected)):
        raise ValueError("Temporal adaptation must train only the source temporal block, projection, final norm and head")
    return torch.optim.AdamW([
        {"params": list(model.head.parameters()), "lr": SETTINGS["optimizer"]["lr"], "name": "head"},
        {"params": list(model.temporal.parameters()) + list(model.final_norm.parameters()),
            "lr": SETTINGS["temporal_lr"], "name": "temporal-projection-and-final-norm"},
    ], **SETTINGS["optimizer"])


def _fit_one(attempt, method, budget, draw, training_rows, development_groups, assessment_groups,
             compact, frozen, pre_records, target_data, target_supply, profile, outputs_root,
             device, smoke, source_reference_path):
    fit_dir = attempt / f"{method}-n{budget:04d}-d{draw:02d}"
    fit_dir.mkdir()
    local = {"status": "incomplete", "method": method, "budget": budget, "draw": draw,
        "smoke": smoke, "held_out_tests": False}
    shared_run.write_json(fit_dir / "status.json", local)
    run = None
    try:
        seed_info = seed_run(0)
        locators = {row["locator"] for row in training_rows}
        if len(locators) != budget:
            raise ValueError("E004c fit subset does not match its declared absolute track budget")
        source_normalization = compact["manifest"]["normalization"]
        normalization, model, trainable_names = None, None, None
        normalization_seconds = 0.
        if method == "temporal-head":
            training = e003c_run._canonical_records([row for row in pre_records["training"]
                if row["locator"] in locators])
            development = e003c_run._canonical_records([row for row in pre_records["validation"]
                if row["group"] in development_groups])
            assessment = e003c_run._canonical_records([row for row in pre_records["validation"]
                if row["group"] in assessment_groups])
            if len(training) != budget or len(development) != 930 or len(assessment) != 931:
                raise ValueError("E004c temporal cache subsets/populations changed")
            source_model = e002f_run._source_model(frozen, ARCHITECTURE, device)
            model = e002f_run._adaptation_model("temporal-head", source_model, device)
            del source_model
            train_dataset = e003b_run.CachedTracks(training)
            development_dataset = e003b_run.CachedTracks(development)
            assessment_dataset = e003b_run.CachedTracks(assessment)
            batches = e003b_run._batches(train_dataset, training=True, seed=0)
            validation_batches = e003b_run._batches(development_dataset, seed=0)
            assessment_batches = e003b_run._batches(assessment_dataset, seed=0)
            normalization = source_normalization
            optimizer = _temporal_optimizer(model)
            trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            forward_batch = None
            epoch_operation = e004a_run._fixed_update_epoch(100)
            initialization = "pinned-source-checkpoint"
            dropout = "disabled for all temporal adaptation modules"
        elif method == "target-only":
            normalization_started = time.monotonic()
            normalization = _target_normalization(training_rows, target_data["training"])
            normalization_seconds = time.monotonic() - normalization_started
            training_dataset = _target_dataset(outputs_root, target_data["training"], normalization,
                target_supply, "training", locators=locators)
            development_dataset = _target_dataset(outputs_root, target_data["training"], normalization,
                target_supply, "validation", group_ids=development_groups)
            assessment_dataset = _target_dataset(outputs_root, target_data["training"], normalization,
                target_supply, "validation", group_ids=assessment_groups)
            if (len(training_dataset) != budget or len(development_dataset) != 930
                    or len(assessment_dataset) != 931
                    or {row["locator"] for row in training_dataset.entries} != locators):
                raise ValueError("E004c target-only dataset subsets/populations changed")
            model = e002f.build_model(ARCHITECTURE, profile).to(device)
            if not all(parameter.requires_grad for parameter in model.parameters()):
                raise ValueError("Target-only initialization must train every task-model parameter")
            train_dataset = training_dataset
            batches = _PersistentLogicalBatches(LogicalBatches(train_dataset, profile,
                training=True, seed=0, prefetch=True))
            validation_batches = PredictionBatches(LogicalBatches(development_dataset, profile,
                training=False, seed=0, prefetch=True))
            assessment_batches = PredictionBatches(LogicalBatches(assessment_dataset, profile,
                training=False, seed=0, prefetch=True))
            optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
            trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
            forward_batch = lambda current, batch, _device: current(batch)
            epoch_operation = _multimodal_update
            initialization = "random-seed-0"
            dropout = {"rgb_branch": .5, "spatial_readout": .1, "temporal": .1}
        else:
            raise ValueError(f"Unsupported E004c method: {method}")

        if method == "temporal-head":
            train_dataset = e003b_run.CachedTracks(training)
            development_dataset = e003b_run.CachedTracks(development)
            assessment_dataset = e003b_run.CachedTracks(assessment)
        if normalization is not source_normalization:
            write_json(fit_dir / "normalization.json", {
                "source": "selected target training tracks only", "track_count": budget,
                "locators_sha256": hashlib.sha256("\n".join(sorted(locators)).encode()).hexdigest(),
                "statistics": normalization})
        setting_record = {**SETTINGS, "method": method,
            "max_optimizer_updates": None if not smoke else 1,
            "check_schedule": [1] if smoke else "1,5,10,25,50,100,then every 100 updates",
            "patience_updates": PATIENT_UPDATES, "exact_f1_tie": "lower CE then earlier update",
            "CE_only_improvement_resets_patience": False}
        trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        config = {"experiment": "E004c", "method": method, "source": SOURCE, "target": TARGET,
            "architecture": ARCHITECTURE, "modalities": "geometry+RGB+D-DITR",
            "budget_tracks": budget, "draw": draw, "training_seed": 0, "initialization": initialization,
            "source_checkpoint_sha256": frozen["sha256"], "feature_cache_identity": {
                split: compact["manifest"]["derived_features"][split]["sha256"]
                for split in ("training", "validation")},
            "pre_temporal_cache_identity": {"training": pre_records["training_meta"]["sha256"],
                "validation": pre_records["validation_meta"]["sha256"]} if method == "temporal-head" else None,
            "trainable_parameter_names": trainable_names, "trainable_parameters": trainable_count,
            "normalization_source": "ROAD source checkpoint" if method == "temporal-head" else "selected target training tracks",
            "normalization_sha256": hashlib.sha256(json.dumps(normalization, sort_keys=True).encode()).hexdigest(),
            "normalization_fit_seconds": normalization_seconds,
            "dropout": dropout, "optimizer": setting_record,
            "source_checkpoint_normalization": source_normalization,
            "development_groups": sorted(development_groups), "assessment_groups": sorted(assessment_groups),
            "development_label_access": e004a_run._population_summary([
                row for row in compact["records"]["validation"] if row["group"] in development_groups]),
            "assessment_label_access": e004a_run._population_summary([
                row for row in compact["records"]["validation"] if row["group"] in assessment_groups]),
            "held_out_tests": False}
        config_path = fit_dir / "config.json"
        write_json(config_path, config)
        write_json(fit_dir / "provenance.json", {"config": str(config_path), "experiment": "E004c",
            "method": method, "source_checkpoint_sha256": frozen["sha256"], "training_seed": 0,
            "draw": draw, "budget": budget, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[3], text=True).strip(),
            "determinism": seed_info, "actual_optimizer_updates": 0, "selected_update": None})

        run_name = f"e004c-road-to-loki-{method}-n{budget:04d}-d{draw:02d}-s0"
        run = wandb.init(entity="yammo-unipd", project="pedestrian-behaviour-labeling", name=run_name,
            job_type="smoke" if smoke else "comparison", config=config,
            dir=str(fit_dir.resolve()), save_code=False, resume="never", mode="offline" if smoke else "online")
        run.define_metric("optimizer_step", hidden=True, summary="none")
        run.define_metric("train/ce_step", step_metric="optimizer_step", step_sync=False, summary="none")
        run.define_metric("train/*", step_metric="optimizer_step", step_sync=False)
        run.define_metric("development/*", step_metric="optimizer_step", step_sync=False)
        validation_seconds = 0.
        first_validation = True
        last_validation_predictions = None
        selected_development_predictions = None

        def on_step(step, loss):
            with (fit_dir / "steps.jsonl").open("a") as stream:
                stream.write(json.dumps({"optimizer_step": step, "training_ce": loss}, allow_nan=False) + "\n")
            run.log({"optimizer_step": step, "train/ce_step": loss})

        def validate(current):
            nonlocal first_validation, validation_seconds, last_validation_predictions
            begin = time.monotonic()
            tracks = predict(current, validation_batches, device, forward_batch=forward_batch)
            if method == "target-only":
                tracks = _decorate_groups(tracks, development_dataset)
            last_validation_predictions = tracks
            metrics = prediction_metrics(tracks, len(CLASSES))
            validation_seconds += time.monotonic() - begin
            if first_validation and method == "temporal-head":
                e003c_run._check_initial_predictions(tracks, source_reference_path, require_all=False)
            first_validation = False
            return metrics

        def on_check(row, history):
            nonlocal selected_development_predictions
            write_json(fit_dir / "history.json", history)
            if row["selected"]:
                selected_development_predictions = last_validation_predictions
            run.log({"optimizer_step": row["optimizer_step"],
                "development/track_ce": row["validation_ce"],
                "development/macro_f1": row["validation_f1"],
                **({"train/track_ce": row["training_ce"], "train/macro_f1": row["training_f1"]}
                    if row["training_ce"] is not None else {})})

        started = time.monotonic()
        if smoke:
            fit_result = fit(model, batches, validate, optimizer, device, fit_dir,
                {"config": config, "normalization": normalization, "provenance_file": "provenance.json",
                 "fit_plan": {"budget": budget, "draw": draw}}, on_check,
                epochs=1, patience=15, clip_norm=SETTINGS["clip_norm"], on_step=on_step,
                epoch_operation=epoch_operation, initial_candidate=True, check_updates=(1,),
                patience_updates=PATIENT_UPDATES)
        else:
            fit_result = fit(model, batches, validate, optimizer, device, fit_dir,
                {"config": config, "normalization": normalization, "provenance_file": "provenance.json",
                 "fit_plan": {"budget": budget, "draw": draw}}, on_check,
                epochs=None, patience=15, clip_norm=SETTINGS["clip_norm"], on_step=on_step,
                epoch_operation=epoch_operation, initial_candidate=True, check_updates=_check_updates(),
                patience_updates=PATIENT_UPDATES)
        fit_seconds = time.monotonic() - started
        selected = reload_checkpoint(fit_dir / "best.pt", model, device)
        selected_update = int(selected["selection"]["optimizer_step"])
        if selected_development_predictions is None:
            raise ValueError("E004c checkpoint selection did not retain matching development predictions")
        evaluation_started = time.monotonic()
        development_predictions = selected_development_predictions
        assessment_predictions = predict(model, assessment_batches, device, forward_batch=forward_batch)
        if method == "target-only":
            assessment_predictions = _decorate_groups(assessment_predictions, assessment_dataset)
        evaluation_seconds = time.monotonic() - evaluation_started
        original_development, original_assessment = development_predictions, assessment_predictions
        _write_predictions(fit_dir / "development-predictions.npz", development_predictions)
        _write_predictions(fit_dir / "assessment-predictions.npz", assessment_predictions)
        development_predictions = load_predictions(fit_dir / "development-predictions.npz")
        assessment_predictions = load_predictions(fit_dir / "assessment-predictions.npz")
        _check_prediction_roundtrip(original_development, development_predictions)
        _check_prediction_roundtrip(original_assessment, assessment_predictions)
        analysis = e004a_analysis.analyze_checkpoint(development_predictions, assessment_predictions,
            fit_dir / "risk-analysis", error_targets=e004a_run.ERROR_TARGETS)
        dev_metrics = prediction_metrics(development_predictions, len(CLASSES))
        assessment_metrics = prediction_metrics(assessment_predictions, len(CLASSES))
        exposure = {key: sum(row[key] for row in fit_result["history"])
            for key in ("optimizer_steps", "processed_tracks", "context_slots", "gt_slots",
                        "padding_slots", "batch_slots")}
        if selected.get("optimizer_step") != selected["selection"].get("optimizer_step"):
            raise ValueError("E004c reloaded best checkpoint update differs from saved selection")
        metadata = {"status": "complete", "experiment": "E004c", "method": method,
            "budget": budget, "draw": draw, "training_seed": 0, "source_checkpoint_sha256": frozen["sha256"],
            "selected_update": selected_update, "selected_epoch_check_ordinal": selected["epoch"],
            "stopping_reason": fit_result["stopping_reason"], "actual_optimizer_updates": exposure["optimizer_steps"],
            "training_track_presentations": exposure["processed_tracks"],
            "effective_dataset_passes": exposure["processed_tracks"] / budget,
            "development_validation_checks": len(fit_result["history"]),
            "development_validation_seconds_total": validation_seconds,
            "development_validation_seconds_mean": validation_seconds / len(fit_result["history"]),
            "normalization_fit_seconds": normalization_seconds, "fit_seconds": fit_seconds,
            "assessment_prediction_seconds": evaluation_seconds,
            "first_optimizer_update_seconds": next((row["seconds"] for row in fit_result["history"]
                if row.get("optimizer_step") == 1), None),
            "actual_training_seconds": sum(row.get("seconds", 0.) for row in fit_result["history"]),
            "selection": fit_result["selection"], "development_metrics": dev_metrics,
            "assessment_metrics": assessment_metrics, "coverage": analysis,
            "annotation_access": e004a_run._annotation_access({"population": {
                "development": config["development_label_access"],
                "assessment": config["assessment_label_access"]}}, training_rows),
            "held_out_tests": False}
        write_json(fit_dir / "selected-metrics.json", metadata)
        provenance_path = fit_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance.update(selected_update=selected_update, selected_epoch_check_ordinal=selected["epoch"],
            stopping_reason=fit_result["stopping_reason"], actual_optimizer_updates=exposure["optimizer_steps"],
            training_track_presentations=exposure["processed_tracks"],
            effective_dataset_passes=exposure["processed_tracks"] / budget,
            fit_seconds=fit_seconds,
            development_validation_checks=len(fit_result["history"]),
            development_validation_seconds_total=validation_seconds,
            development_validation_seconds_mean=validation_seconds / len(fit_result["history"]),
            normalization_fit_seconds=normalization_seconds,
            assessment_prediction_seconds=evaluation_seconds,
            determinism=seed_info)
        write_json(provenance_path, provenance)
        fit_plot = e004a_run._plots(fit_dir / "plots", fit_result["history"])
        run.log({"assessment/full_coverage_macro_f1": assessment_metrics["track_weighted"]["macro_f1"],
            "assessment/track_ce": assessment_metrics["track_ce"],
            "assessment/risk_targets": e004a_run._wandb_risk_table(analysis),
            "assessment/class_f1": wandb.Table(columns=["class", "f1", "support"],
                data=[[name, assessment_metrics["track_weighted"]["f1"][index],
                    assessment_metrics["track_weighted"]["support"][index]]
                    for index, name in enumerate(CLASSES)]), "details/fit-history": wandb.Image(str(fit_plot))})
        run.finish()
        run = None
        local["status"] = "complete"
        write_json(fit_dir / "status.json", local)
        return {"method": method, "budget": budget, "draw": draw,
            "macro_f1": assessment_metrics["track_weighted"]["macro_f1"],
            "track_ce": assessment_metrics["track_ce"], "class_f1": _metric_class_f1(assessment_metrics),
            "coverage": analysis, "annotation_access": metadata["annotation_access"],
            "selected_update": selected_update, "stopping_reason": fit_result["stopping_reason"],
            "fit_dir": str(fit_dir), "assessment_metrics": assessment_metrics,
            "development_metrics": dev_metrics, "effective_dataset_passes": metadata["effective_dataset_passes"],
            "normalization_fit_seconds": normalization_seconds,
            "training_track_presentations": exposure["processed_tracks"],
            "prediction_references": {"development": {"path": str(fit_dir / "development-predictions.npz"),
                "sha256": checksum(fit_dir / "development-predictions.npz")},
                "assessment": {"path": str(fit_dir / "assessment-predictions.npz"),
                    "sha256": checksum(fit_dir / "assessment-predictions.npz")}}}
    except BaseException as error:
        local.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
        write_json(fit_dir / "status.json", local)
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        raise


def run(args):
    device = torch.device(args.device)
    outputs_root = Path(args.outputs_root).resolve()
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested E004c device is unavailable")
    torch.set_num_threads(2)
    runs_root = outputs_root / "experiments/E004c/runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    attempt = runs_root / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
        + ("-smoke" if args.smoke else ""))
    attempt.mkdir()
    started = time.monotonic()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=Path(__file__).resolve().parents[3])
    compact = frozen = pre_records = pre_meta = None
    reader = None
    protected = {}
    provenance = {"command": getattr(sys, "orig_argv", sys.argv),
        "started_utc": datetime.now(timezone.utc).isoformat(), "smoke": args.smoke,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3], text=True).strip(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(), "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_checkpoint_sha256": e004a_run.SOURCE_SHA256,
        "reference_e004a_attempt": str(outputs_root / "experiments/E004a/runs" / EARLY_REFERENCE),
        "code_sha256": {"runner": checksum(__file__), "training": checksum(shared_training.__file__),
            "source_cache": checksum(e002f_run.__file__), "multimodal_inputs": checksum(e002a_run.__file__),
            "analysis": checksum(e004a_analysis.__file__), "report": checksum(e004c_analysis.__file__),
            "launcher": checksum(Path(__file__).resolve().parents[3] / "scripts/train-e004c.py")}}
    shared_run.write_json(attempt / "provenance.json", provenance)
    shared_run.write_json(attempt / "status.json", {"status": "incomplete", "smoke": args.smoke,
        "held_out_tests": False, "failures": []})
    try:
        compact, frozen, pre_records, pre_meta = _load_caches(outputs_root)
        training, development, assessment, split_identity = _load_splits(compact)
        reference_root, reference_plan, reference_by_key, plan_sha = _read_reference(outputs_root)
        if reference_plan.get("validation_split") != split_identity:
            raise ValueError("E004a validation groups differ from deterministic E004c split")
        expected_plan_keys = {(budget, draw) for budget in BUDGETS for draw in range(5)}
        for budget, draw in expected_plan_keys:
            expected = [row["locator"] for row in e004a_run.recording_round_robin(training, draw)[:budget]]
            actual = [row["locator"] for row in reference_by_key[(budget, draw)]["selected_tracks"]]
            if actual != expected:
                raise ValueError(f"E004a nested target subset differs for budget {budget}, draw {draw}")

        target_data, target_reference = e002f_run._source_target_data(outputs_root, frozen["normalization"])
        open_args = argparse.Namespace(outputs_root=outputs_root, device=str(device), probe=args.probe,
            rgb_cache_root=args.rgb_cache_root, rgb_preparation=args.rgb_preparation)
        reader, profile, _source_input_datasets, catalog, points, protected = e003c_run._open_target(
            open_args, frozen | {"run_config": frozen["config"]}, target_data, frozen["normalization"])
        supply = reader.supply({"input": "geometry+rgb+lidar"}, frozen["config"])
        plan_rows = [{"budget": budget, "draw": draw,
            "selected_tracks": reference_by_key[(budget, draw)]["selected_tracks"]}
            for draw in range(5) for budget in BUDGETS]
        plan = {"status": "planned", "experiment": "E004c", "smoke": args.smoke,
            "reference_e004a_subset_plan": str(reference_root / "subset-plans.json"),
            "reference_e004a_subset_plan_sha256": plan_sha,
            "source_checkpoint_sha256": frozen["sha256"], "training_seed": 0,
            "budgets": list(BUDGETS), "draw_seeds": list(range(5)),
            "check_updates": "0,1,5,10,25,50,100,then every 100 updates",
            "patience_updates": PATIENT_UPDATES, "max_optimizer_updates": None,
            "validation_split": split_identity,
            "population": {"training": e004a_run._population_summary(training),
                "development": e004a_run._population_summary(development),
                "assessment": e004a_run._population_summary(assessment)},
            "runs": plan_rows}
        shared_run.write_json(attempt / "subset-plans.json", plan)
        provenance.update(source_cache=compact["manifest"]["source_cache"],
            postnorm_cache={split: compact["manifest"]["derived_features"][split]
                for split in ("training", "validation")},
            pre_temporal_cache={split: pre_meta[split] for split in ("training", "validation")},
            e004a_reference_plan={"path": str(reference_root / "subset-plans.json"),
                "sha256": plan_sha}, target_reference=target_reference,
            feature_catalog=catalog.identity, prepared_point_cache=points.identity,
            model_profile=profile)
        shared_run.write_json(attempt / "provenance.json", provenance)

        if args.smoke:
            reference_rows = []
        else:
            reference_rows = _reference_metrics(reference_root, reference_by_key, split_identity,
                attempt / "references")
        rows = list(reference_rows)
        all_post_training = compact["records"]["training"]
        all_pre_training = pre_records["training"]
        if args.smoke:
            fit_keys = [(1000, 0)]
            methods = ("temporal-head", "target-only")
        else:
            fit_keys = [(budget, draw) for draw in range(5) for budget in BUDGETS]
            methods = ("temporal-head", "target-only")
        fit_pre_records = pre_records | {"training_meta": pre_meta["training"],
            "validation_meta": pre_meta["validation"]}
        for method in methods:
            for budget, draw in fit_keys:
                selected_ids = {row["locator"] for row in reference_by_key[(budget, draw)]["selected_tracks"]}
                fit_rows = e003c_run._canonical_records([row for row in all_post_training
                    if row["locator"] in selected_ids])
                if len(fit_rows) != budget:
                    raise ValueError("E004c selected cache records do not match the nested draw")
                if method == "temporal-head":
                    expected_pre = {row["locator"] for row in all_pre_training}
                    if expected_pre != {row["locator"] for row in all_post_training}:
                        raise ValueError("Temporal adaptation feature cache lost a target training track")
                result = _fit_one(attempt, method, budget, draw, fit_rows,
                    set(split_identity["development_groups"]), set(split_identity["assessment_groups"]),
                    compact, frozen, fit_pre_records, target_data, supply, profile, outputs_root,
                    device, args.smoke, outputs_root / e004a_run.SOURCE_FEATURES / "source-validation-predictions.npz")
                rows.append(result)
                write_json(attempt / "results.json", rows)
        if not args.smoke:
            e004c_analysis.make_report(rows, attempt / "report")
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started)
        shared_run.write_json(attempt / "provenance.json", provenance)
        shared_run.write_json(attempt / "status.json", {"status": "complete", "smoke": args.smoke,
            "fit_count": 2 if args.smoke else 30, "reused_head_fits": 0 if args.smoke else 15,
            "fixed_source_anchor": False if args.smoke else True,
            "results": len(rows), "held_out_tests": False,
            "report": None if args.smoke else str(attempt / "report")})
        return {"attempt": str(attempt), "fits": 2 if args.smoke else 30,
            "reused_head_fits": 0 if args.smoke else 15, "smoke": args.smoke,
            "results": len(rows), "report": None if args.smoke else str(attempt / "report")}
    except BaseException as error:
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            failure={"type": type(error).__name__, "message": str(error)})
        shared_run.write_json(attempt / "provenance.json", provenance)
        shared_run.write_json(attempt / "status.json", {"status": "failed", "smoke": args.smoke,
            "held_out_tests": False, "failures": [{"type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc()}]})
        raise
    finally:
        if reader is not None:
            reader.close()
        if protected:
            changed = [path for path, digest in protected.items() if checksum(path) != digest]
            shared_run.write_json(attempt / "protected.json", {"hashes": protected, "changed": changed})
            if changed:
                raise ValueError("Frozen LOKI feature/cache files changed during E004c")
