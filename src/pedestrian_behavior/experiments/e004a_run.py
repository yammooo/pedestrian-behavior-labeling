"""E004a ROAD-to-LOKI source-head adaptation over nested recording-aware budgets."""

from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import wandb

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import (load_predictions, predict, prediction_metrics,
    save_predictions)
from pedestrian_behavior.experiments import (e002f_run, e003b_run, e003c_run,
    e004a_analysis, run as shared_run)
from pedestrian_behavior import training as shared_training
from pedestrian_behavior.training import fit, reload_checkpoint, seed_run, train_epoch


SOURCE, TARGET, ARCHITECTURE = "road-waymo", "loki", "bilstm-96"
SOURCE_SHA256 = "ac9bacf4c8c4bea16a63cf4ff2b9bf916079b7474b3a49e5b36ef7fe46df4574"
SOURCE_RUN = Path("experiments/E002f/runs/temporal-architecture-20261110-091857/source/bilstm-96")
SOURCE_FEATURES = Path("experiments/E002f/features/temporal-architecture-20261110-091857/bilstm-96")
CLASSES = e003b_run.CLASSES
BUDGETS = (1, 5, 10, 25, 50, 100, 250, 500, 1000)
ALL_TRACKS = 8574
DEVELOPMENT_GROUPS, ASSESSMENT_GROUPS = 46, 47
EXPECTED_VALIDATION = {"development": {"tracks": 930, "groups": 46, "gt_timesteps": 29626,
    "class_gt_counts": [17633, 1789, 5691, 4513], "nominal_gt_duration_seconds": 5925.2},
    "assessment": {"tracks": 931, "groups": 47, "gt_timesteps": 30605,
    "class_gt_counts": [19777, 2417, 2629, 5782], "nominal_gt_duration_seconds": 6121.0}}
ERROR_TARGETS = (.01, .05, .10, .20)
SETTINGS = {"seed": 0, "batch_size": 64, "max_updates": 10000, "updates_per_check": 100,
    "patience_checks": 15, "clip_norm": 1., "precision": "float32",
    "optimizer": {"lr": .0003, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8}}


def _head_digest(weight, bias):
    digest = hashlib.sha256()
    digest.update(np.asarray(weight, dtype=np.float32).tobytes())
    digest.update(np.asarray(bias, dtype=np.float32).tobytes())
    return digest.hexdigest()


def _save_predictions(path, tracks):
    for row in tracks:
        row.setdefault("conditions", {})
        row.setdefault("source_flags", {})
    save_predictions(path, tracks)


def _source_paths(outputs_root):
    return outputs_root / SOURCE_RUN, outputs_root / SOURCE_FEATURES


def _cache_identity(frozen, split, source_cache_entry):
    return {"experiment": "E004a", "stage": "post-bilstm-final-layernorm", "source": SOURCE,
        "target": TARGET, "architecture": ARCHITECTURE, "split": split,
        "checkpoint_sha256": frozen["sha256"], "source_normalization": frozen["normalization"],
        "source_cache": {"path": source_cache_entry["path"], "sha256": source_cache_entry["sha256"],
                         "identity": {key: value for key, value in source_cache_entry.items()
                                      if key not in ("path", "sha256", "context_slots")}}}


def _write_compact_cache(root, outputs_root, device):
    source_run, source_features = _source_paths(outputs_root)
    frozen = e002f_run._source_run(source_run, ARCHITECTURE, outputs_root)
    if frozen["sha256"] != SOURCE_SHA256:
        raise ValueError("Pinned E002f BiLSTM-96 checkpoint SHA256 mismatch")
    source_records, source_meta = {}, {}
    for split in ("training", "validation"):
        records, entry = e002f_run._load_cache(source_features, split, ARCHITECTURE,
            frozen["sha256"], frozen["normalization"])
        source_records[split], source_meta[split] = records, entry
        expected = ALL_TRACKS if split == "training" else 1861
        if len(records) != expected or len({row["locator"] for row in records}) != expected:
            raise ValueError(f"E002f {split} cache does not contain the pinned full LOKI population")

    reference_path = source_features / "source-validation-predictions.npz"
    if not reference_path.is_file():
        raise ValueError("Pinned E002f direct LOKI validation predictions are missing")
    source_reference_sha = checksum(reference_path)
    device = torch.device(device)
    source_model = e002f_run._source_model(frozen, ARCHITECTURE, device)
    records = {split: e002f_run._postnorm_records(source_records[split], source_model, device)
               for split in ("training", "validation")}
    state = frozen["checkpoint"]["model_state"]
    weight = state["head.weight"].detach().cpu().numpy().astype(np.float32, copy=True)
    bias = state["head.bias"].detach().cpu().numpy().astype(np.float32, copy=True)
    if weight.shape != (4, 128) or bias.shape != (4,):
        raise ValueError("Pinned source classifier is not Linear(128, 4)")
    head_sha = _head_digest(weight, bias)
    del source_model

    copied_head = _new_head({"weight": weight, "bias": bias}, device)
    source_tracks = predict(copied_head,
        e003b_run._batches(e003b_run.CachedTracks(records["validation"]), seed=0), device)
    e003c_run._check_initial_predictions(source_tracks, reference_path, require_all=True)
    if len(source_tracks) != 1861:
        raise ValueError("Derived source-head validation predictions changed the frozen population")

    cache_entries = {}
    for split in ("training", "validation"):
        identity = _cache_identity(frozen, split, source_meta[split]) | {
            "source_head_sha256": head_sha,
            "source_validation_predictions_sha256": source_reference_sha if split == "validation" else None}
        path = root / f"features-{split}.npz"
        e003b_run._save_cache(path, records[split], identity)
        e003b_run._load_cache(path, identity)
        cache_entries[split] = {"path": path.name, "sha256": checksum(path), "identity": identity,
            "tracks": len(records[split]), "gt_timesteps": int(sum(row["gt_valid"].sum() for row in records[split]))}

    head_path = root / "source-head.npz"
    with head_path.open("wb") as stream:
        np.savez_compressed(stream, weight=weight, bias=bias,
            metadata=np.asarray(json.dumps({"checkpoint_sha256": frozen["sha256"], "head_sha256": head_sha},
                                           sort_keys=True)))
    predictions_path = root / "source-validation-predictions.npz"
    _save_predictions(predictions_path, source_tracks)
    roundtrip = load_predictions(predictions_path)
    if {row["locator"] for row in roundtrip} != {row["locator"] for row in source_tracks}:
        raise ValueError("Compact source-validation prediction roundtrip changed track identities")

    manifest = {"status": "complete", "experiment": "E004a", "source_checkpoint": str(source_run / "best.pt"),
        "source_checkpoint_sha256": frozen["sha256"], "source_checkpoint_identity": frozen["identity"],
        "architecture": ARCHITECTURE, "normalization_source": SOURCE, "normalization": frozen["normalization"],
        "source_cache": {"root": str(source_features), "splits": source_meta},
        "derived_features": cache_entries,
        "source_head": {"path": head_path.name, "sha256": checksum(head_path), "state_sha256": head_sha},
        "source_validation_predictions": {"path": predictions_path.name,
            "sha256": checksum(predictions_path), "reference_sha256": source_reference_sha},
        "population": {split: len(records[split]) for split in records}, "held_out_tests": False}
    shared_run.write_json(root / "cache.json", manifest)
    return _load_compact_cache(root, require_status=False)


def _load_compact_cache(root, *, require_status=True):
    root = Path(root)
    manifest_path = root / "cache.json"
    if not manifest_path.is_file():
        raise ValueError("Completed E004a post-temporal feature cache is missing")
    if require_status and (not (root / "status.json").is_file()
            or json.loads((root / "status.json").read_text()).get("status") != "complete"):
        raise ValueError("E004a compact-cache build is not complete")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("experiment") != "E004a"
            or manifest.get("source_checkpoint_sha256") != SOURCE_SHA256
            or manifest.get("architecture") != ARCHITECTURE
            or manifest.get("population") != {"training": ALL_TRACKS, "validation": 1861}):
        raise ValueError("E004a compact cache identity/population mismatch")
    records = {}
    for split, entry in manifest["derived_features"].items():
        path = root / entry["path"]
        if checksum(path) != entry["sha256"]:
            raise ValueError(f"E004a {split} compact cache checksum mismatch")
        records[split] = e003b_run._load_cache(path, entry["identity"])
        if len(records[split]) != manifest["population"][split]:
            raise ValueError(f"E004a {split} compact cache track count mismatch")
    head_entry = manifest["source_head"]
    head_path = root / head_entry["path"]
    if checksum(head_path) != head_entry["sha256"]:
        raise ValueError("E004a pinned source-head cache checksum mismatch")
    with np.load(head_path, allow_pickle=False) as archive:
        head = {key: archive[key].copy() for key in ("weight", "bias")}
        metadata = json.loads(str(archive["metadata"]))
    if (metadata != {"checkpoint_sha256": SOURCE_SHA256, "head_sha256": head_entry["state_sha256"]}
            or head["weight"].shape != (4, 128) or head["bias"].shape != (4,)
            or _head_digest(head["weight"], head["bias"]) != head_entry["state_sha256"]):
        raise ValueError("E004a source-head cache is not the pinned 516-parameter classifier")
    prediction_entry = manifest["source_validation_predictions"]
    prediction_path = root / prediction_entry["path"]
    if checksum(prediction_path) != prediction_entry["sha256"]:
        raise ValueError("E004a source-validation prediction checksum mismatch")
    return {"root": root, "manifest": manifest, "records": records, "head": head,
            "source_validation": load_predictions(prediction_path)}


def load_or_build_cache(outputs_root, device):
    outputs_root = Path(outputs_root).resolve()
    root = outputs_root / "experiments/E004a/features/postnorm-bilstm-96"
    if root.exists():
        if not (root / "cache.json").is_file():
            raise ValueError(f"Incomplete E004a cache exists at {root}; preserve it and choose a fresh outputs root")
        return _load_compact_cache(root)
    root.mkdir(parents=True)
    shared_run.write_json(root / "status.json", {"status": "incomplete", "held_out_tests": False})
    try:
        result = _write_compact_cache(root, outputs_root, device)
        status = {"status": "complete", "source_checkpoint_sha256": SOURCE_SHA256,
            "tracks": result["manifest"]["population"], "held_out_tests": False}
        shared_run.write_json(root / "status.json", status)
        return result
    except BaseException as error:
        shared_run.write_json(root / "status.json", {"status": "failed", "held_out_tests": False,
            "failures": [{"type": type(error).__name__, "message": str(error)}]})
        raise


def recording_round_robin(records, draw):
    """Shuffle sorted groups and their sorted tracks, then take one track per group per round."""
    by_group = {}
    for row in sorted(records, key=lambda item: item["locator"]):
        by_group.setdefault(row["group"], []).append(row)
    if not by_group:
        raise ValueError("Cannot draw from an empty training population")
    rng = np.random.default_rng(draw)
    groups = sorted(by_group)
    group_order = [str(value) for value in rng.permutation(groups)]
    tracks = {group: list(rng.permutation(by_group[group])) for group in groups}
    positions = dict.fromkeys(groups, 0)
    output = []
    while len(output) < len(records):
        for group in group_order:
            position = positions[group]
            if position < len(tracks[group]):
                output.append(tracks[group][position])
                positions[group] += 1
    if len({row["locator"] for row in output}) != len(records):
        raise ValueError("Recording-aware draw duplicated or lost tracks")
    return output


def split_development_assessment(records):
    groups = sorted({row["group"] for row in records})
    if len(groups) != DEVELOPMENT_GROUPS + ASSESSMENT_GROUPS:
        raise ValueError(f"Expected 93 LOKI validation scenarios, found {len(groups)}")
    order = [str(value) for value in np.random.default_rng(0).permutation(groups)]
    development, assessment = set(order[:DEVELOPMENT_GROUPS]), set(order[DEVELOPMENT_GROUPS:])
    if len(development) != DEVELOPMENT_GROUPS or len(assessment) != ASSESSMENT_GROUPS or development & assessment:
        raise ValueError("LOKI development/assessment group split is not disjoint 46/47")
    return ([row for row in records if row["group"] in development],
            [row for row in records if row["group"] in assessment],
            {"seed": 0, "development_groups": sorted(development), "assessment_groups": sorted(assessment)})


def _track_label_record(row):
    labels = row["targets"][row["gt_valid"]]
    return {"locator": row["locator"], "group": row["group"], "gt_timesteps": int(len(labels)),
        "class_gt_counts": np.bincount(labels, minlength=4).astype(int).tolist()}


def _plan(training, development, assessment, smoke=False):
    plan_rows = []
    draws = (0,) if smoke else tuple(range(5))
    budgets = (1, 5) if smoke else BUDGETS
    for draw in draws:
        order = recording_round_robin(training, draw)
        for budget in budgets:
            selected = order[:budget]
            plan_rows.append({"budget": budget, "draw": draw,
                "selected_tracks": [_track_label_record(row) for row in selected]})
    if not smoke:
        plan_rows.append({"budget": ALL_TRACKS, "draw": None,
            "selected_tracks": [_track_label_record(row) for row in sorted(training, key=lambda item: item["locator"])]})
    return {"status": "planned", "experiment": "E004a", "smoke": smoke,
        "protocol": {"source_checkpoint_sha256": SOURCE_SHA256, "training_seed": 0,
            "draw_seeds": list(draws), "budget_grid": list(budgets) + ([] if smoke else [ALL_TRACKS]),
            "fit_seed": 0, "optimizer": SETTINGS["optimizer"], "max_optimizer_updates": 200 if smoke else SETTINGS["max_updates"],
            "validation_every_updates": SETTINGS["updates_per_check"], "patience_checks": SETTINGS["patience_checks"],
            "development_groups_seed": 0, "development_group_count": DEVELOPMENT_GROUPS,
            "assessment_group_count": ASSESSMENT_GROUPS},
        "population": {"training": _population_summary(training),
            "development": _population_summary(development), "assessment": _population_summary(assessment)},
        "validation_split": {"development_groups": sorted({row["group"] for row in development}),
            "assessment_groups": sorted({row["group"] for row in assessment})}, "runs": plan_rows}


def _new_head(state, device):
    model = e003b_run.AdaptationHead()
    model.load_state_dict({"linear.weight": torch.from_numpy(state["weight"].copy()),
                           "linear.bias": torch.from_numpy(state["bias"].copy())}, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 516:
        raise ValueError("E004a must train exactly the original 516 classifier parameters")
    return model.to(device).eval()


def _fixed_update_epoch(updates):
    iterator = None

    def operation(model, batches, optimizer, device, clip_norm, on_step=None, step_offset=0):
        nonlocal iterator

        def next_batches():
            nonlocal iterator
            for _ in range(updates):
                if iterator is None:
                    iterator = iter(batches)
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(batches)
                    try:
                        yield next(iterator)
                    except StopIteration as error:
                        raise ValueError("E004a training loader is empty") from error

        return train_epoch(model, next_batches(), optimizer, device, clip_norm,
                           on_step=on_step, step_offset=step_offset)
    return operation


def _population_summary(rows):
    class_counts = np.zeros(4, dtype=np.int64)
    for row in rows:
        class_counts += np.bincount(row["targets"][row["gt_valid"]], minlength=4)
    return {"tracks": len(rows), "groups": len({row["group"] for row in rows}),
        "gt_timesteps": int(class_counts.sum()), "class_gt_counts": class_counts.tolist(),
        "nominal_gt_duration_seconds": float(class_counts.sum() / 5)}


def _annotation_access(plan, training_rows):
    training = _population_summary(training_rows)
    development = plan["population"]["development"]
    assessment = plan["population"]["assessment"]
    training_and_development = {"tracks": training["tracks"] + development["tracks"],
        "groups": training["groups"] + development["groups"],
        "gt_timesteps": training["gt_timesteps"] + development["gt_timesteps"],
        "class_gt_counts": (np.asarray(training["class_gt_counts"]) + development["class_gt_counts"]).tolist()}
    training_and_development["nominal_gt_duration_seconds"] = training_and_development["gt_timesteps"] / 5
    total = {"tracks": training_and_development["tracks"] + assessment["tracks"],
        "groups": training_and_development["groups"] + assessment["groups"],
        "gt_timesteps": training_and_development["gt_timesteps"] + assessment["gt_timesteps"],
        "class_gt_counts": (np.asarray(training_and_development["class_gt_counts"])
            + assessment["class_gt_counts"]).tolist()}
    total["nominal_gt_duration_seconds"] = total["gt_timesteps"] / 5
    return {"training_subset": training,
        "development_calibration": plan["population"]["development"],
        "assessment_audit": plan["population"]["assessment"],
        "training_plus_development": training_and_development, "total_labeled_access": total}


def _plots(output, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(exist_ok=True)
    checks = [row for row in history if row["epoch"] > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([row["epoch"] * SETTINGS["updates_per_check"] for row in history],
            [row["validation_f1"] for row in history], marker="o", label="development F1")
    ax.set(xlabel="Optimizer update", ylabel="Track-weighted macro-F1", ylim=(0, 1))
    ax2 = ax.twinx()
    ax2.plot([row["epoch"] * SETTINGS["updates_per_check"] for row in checks],
             [row["training_f1"] for row in checks], color="tab:orange", alpha=.7, label="training F1")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"fit-history.{suffix}", dpi=150)
    plt.close(fig)
    return output / "fit-history.png"


def _wandb_risk_table(analysis):
    return wandb.Table(columns=["calibration_error_target", "threshold", "reject_all",
        "calibration_coverage", "calibration_error", "assessment_coverage", "assessment_error"],
        data=[[row["error_target"], row["threshold"], row["reject_all"],
            row["calibration"].get("coverage"), row["calibration"].get("error"),
            row["assessment"].get("coverage"), row["assessment"].get("error")] for row in analysis["targets"]])


def _run_one(attempt, cache, training_rows, development_rows, assessment_rows, plan_row,
             plan, device, smoke):
    budget, draw = plan_row["budget"], plan_row["draw"]
    suffix = "all" if budget == ALL_TRACKS else f"{budget:04d}"
    fit_dir = attempt / (f"budget-{suffix}" if draw is None else f"budget-{suffix}-draw-{draw:02d}")
    fit_dir.mkdir()
    local = {"status": "incomplete", "budget": budget, "draw": draw, "held_out_tests": False}
    shared_run.write_json(fit_dir / "status.json", local)
    run = None
    try:
        seed_info = seed_run(0)
        model = _new_head(cache["head"], device)
        dataset = e003b_run.CachedTracks(training_rows)
        development = e003b_run.CachedTracks(development_rows)
        batches = e003b_run._batches(dataset, training=True, seed=0)
        development_batches = e003b_run._batches(development, seed=0)
        training_steps = 2 if smoke else SETTINGS["max_updates"] // SETTINGS["updates_per_check"]
        settings = {**SETTINGS, "max_updates": training_steps * SETTINGS["updates_per_check"],
            "check_interval_updates": SETTINGS["updates_per_check"], "smoke": smoke}
        config = {"experiment": "E004a", "source": SOURCE, "target": TARGET,
            "architecture": ARCHITECTURE, "trainable_parameters": 516,
            "source_checkpoint_sha256": SOURCE_SHA256, "source_head_sha256": cache["manifest"]["source_head"]["state_sha256"],
            "feature_cache": {split: cache["manifest"]["derived_features"][split]["sha256"]
                              for split in ("training", "validation")},
            "budget_tracks": budget, "draw": draw, "training_seed": 0,
            "development_groups": DEVELOPMENT_GROUPS, "assessment_groups": ASSESSMENT_GROUPS,
            "settings": settings, "training_subset_groups": len({row["group"] for row in training_rows}),
            "training_subset_gt_timesteps": int(sum(row["gt_valid"].sum() for row in training_rows)),
            "development_label_access": plan["population"]["development"],
            "assessment_label_access": plan["population"]["assessment"],
            "normalization_source": SOURCE, "held_out_tests": False}
        config_path = fit_dir / "config.json"
        shared_run.write_json(config_path, config)
        shared_run.write_json(fit_dir / "provenance.json", {"config": str(config_path),
            "plan": str(attempt / "subset-plans.json"), "determinism": seed_info,
            "source_checkpoint_sha256": SOURCE_SHA256, "cache_manifest": str(cache["root"] / "cache.json"),
            "actual_optimizer_updates": 0, "selected_update": None})
        run_name = f"e004a-road-to-loki-head-n{budget:04d}-d{draw:02d}-s0" if draw is not None else \
            f"e004a-road-to-loki-head-n{budget:04d}-s0"
        run = wandb.init(entity="yammo-unipd", project="pedestrian-behaviour-labeling", name=run_name,
            job_type="smoke" if smoke else "comparison", config=config,
            dir=str(fit_dir.resolve()), save_code=False, resume="never", mode="offline" if smoke else "online")
        run.define_metric("optimizer_step", hidden=True, summary="none")
        run.define_metric("train/ce_step", step_metric="optimizer_step", step_sync=False, summary="none")
        run.define_metric("optimizer_check", hidden=True, summary="none")
        run.define_metric("train/*", step_metric="optimizer_check", step_sync=False)
        run.define_metric("development/*", step_metric="optimizer_check", step_sync=False)

        def on_step(step, loss):
            row = {"optimizer_step": step, "training_ce": loss}
            with (fit_dir / "steps.jsonl").open("a") as step_file:
                step_file.write(json.dumps(row, allow_nan=False) + "\n")
            run.log({"optimizer_step": step, "train/ce_step": loss})

        history_path = fit_dir / "history.json"
        def on_check(row, history):
            shared_run.write_json(history_path, history)
            step = row["epoch"] * SETTINGS["updates_per_check"]
            run.log({"optimizer_check": step, "development/track_ce": row["validation_ce"],
                "development/macro_f1": row["validation_f1"],
                **({"train/track_ce": row["training_ce"], "train/macro_f1": row["training_f1"]}
                   if row["training_ce"] is not None else {})})

        def validate(current):
            return prediction_metrics(predict(current, development_batches, device), len(CLASSES))

        optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
        fit_result = fit(model, batches, validate, optimizer, device, fit_dir,
            {"config": config, "normalization": cache["manifest"]["normalization"],
             "provenance_file": "provenance.json", "fit_plan": {"budget": budget, "draw": draw}},
            on_check, epochs=training_steps, patience=SETTINGS["patience_checks"],
            clip_norm=SETTINGS["clip_norm"], on_step=on_step,
            epoch_operation=_fixed_update_epoch(SETTINGS["updates_per_check"]), initial_candidate=True)
        selected = reload_checkpoint(fit_dir / "best.pt", model, device)
        # Assessment predictions are first computed after checkpoint selection and reload.
        selected_development = predict(model, development_batches, device)
        assessment = e003b_run.CachedTracks(assessment_rows)
        assessment_tracks = predict(model, e003b_run._batches(assessment, seed=0), device)
        _save_predictions(fit_dir / "development-predictions.npz", selected_development)
        _save_predictions(fit_dir / "assessment-predictions.npz", assessment_tracks)
        analysis = e004a_analysis.analyze_checkpoint(selected_development, assessment_tracks,
            fit_dir / "risk-analysis", error_targets=ERROR_TARGETS)
        dev_metrics = prediction_metrics(selected_development, len(CLASSES))
        assessment_metrics = prediction_metrics(assessment_tracks, len(CLASSES))
        exposure = {key: sum(row[key] for row in fit_result["history"])
                    for key in ("optimizer_steps", "processed_tracks", "context_slots", "gt_slots",
                                "padding_slots", "batch_slots")}
        selected_update = int(selected["epoch"] * SETTINGS["updates_per_check"])
        metadata = {"status": "complete", "budget": budget, "draw": draw,
            "training_seed": 0, "source_checkpoint_sha256": SOURCE_SHA256,
            "source_head_sha256": config["source_head_sha256"],
            "selected_update": selected_update, "selected_epoch": selected["epoch"],
            "stopping_reason": fit_result["stopping_reason"],
            "actual_optimizer_updates": exposure["optimizer_steps"], "exposure": exposure,
            "selection": fit_result["selection"], "development_metrics": dev_metrics,
            "assessment_metrics": assessment_metrics, "coverage": analysis,
            "annotation_access": _annotation_access(plan, training_rows), "held_out_tests": False}
        shared_run.write_json(fit_dir / "selected-metrics.json", metadata)
        provenance_path = fit_dir / "provenance.json"
        provenance = json.loads(provenance_path.read_text())
        provenance.update(selected_update=selected_update, selected_epoch=selected["epoch"],
            stopping_reason=fit_result["stopping_reason"], actual_optimizer_updates=exposure["optimizer_steps"],
            exposure=exposure, determinism=seed_info)
        shared_run.write_json(provenance_path, provenance)
        plot_path = _plots(fit_dir / "plots", fit_result["history"])
        risk_table = _wandb_risk_table(analysis)
        run.log({"assessment/full_coverage_macro_f1": assessment_metrics["track_weighted"]["macro_f1"],
            "assessment/track_ce": assessment_metrics["track_ce"],
            "assessment/risk_targets": risk_table, "assessment/class_f1": wandb.Table(
                columns=["class", "precision", "recall", "f1", "support"],
                data=[[name] + [assessment_metrics["track_weighted"][key][index]
                       for key in ("precision", "recall", "f1", "support")]
                      for index, name in enumerate(CLASSES)]),
            "details/fit-history": wandb.Image(str(plot_path))})
        run.finish()
        run = None
        local["status"] = "complete"
        shared_run.write_json(fit_dir / "status.json", local)
        return {"budget": budget, "draw": draw,
            "macro_f1": assessment_metrics["track_weighted"]["macro_f1"],
            "track_ce": assessment_metrics["track_ce"], "coverage": analysis,
            "annotation_access": metadata["annotation_access"], "fit_dir": str(fit_dir),
            "selected_update": selected_update, "stopping_reason": fit_result["stopping_reason"]}
    except BaseException as error:
        local.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
        shared_run.write_json(fit_dir / "status.json", local)
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception:
                pass
        raise


def _evaluate_zero(attempt, cache, development_rows, assessment_rows, plan):
    output = attempt / "zero-source-head"
    output.mkdir()
    dev_groups = {row["group"] for row in development_rows}
    assessment_groups = {row["group"] for row in assessment_rows}
    dev_tracks = [row for row in cache["source_validation"] if row["group"] in dev_groups]
    assessment_tracks = [row for row in cache["source_validation"] if row["group"] in assessment_groups]
    if len(dev_tracks) != len(development_rows) or len(assessment_tracks) != len(assessment_rows):
        raise ValueError("Zero anchor source-validation predictions do not match the frozen group split")
    _save_predictions(output / "development-predictions.npz", dev_tracks)
    _save_predictions(output / "assessment-predictions.npz", assessment_tracks)
    analysis = e004a_analysis.analyze_checkpoint(dev_tracks, assessment_tracks,
        output / "risk-analysis", error_targets=ERROR_TARGETS)
    dev_metrics, assessment_metrics = prediction_metrics(dev_tracks, len(CLASSES)), prediction_metrics(assessment_tracks, len(CLASSES))
    summary = {"status": "complete", "budget": 0, "draw": None,
        "source_checkpoint_sha256": SOURCE_SHA256, "selected_update": 0,
        "selection": {"epoch": 0, "f1": dev_metrics["track_weighted"]["macro_f1"], "ce": dev_metrics["track_ce"]},
        "development_metrics": dev_metrics, "assessment_metrics": assessment_metrics,
        "coverage": analysis, "annotation_access": _annotation_access(plan, []),
        "kind": "original source head; no optimizer", "held_out_tests": False}
    shared_run.write_json(output / "selected-metrics.json", summary)
    return {"budget": 0, "draw": None, "macro_f1": assessment_metrics["track_weighted"]["macro_f1"],
        "track_ce": assessment_metrics["track_ce"], "coverage": analysis,
        "annotation_access": summary["annotation_access"], "fit_dir": str(output), "selected_update": 0,
        "stopping_reason": "source-anchor"}


def run(device="cuda", outputs_root=Path("outputs"), smoke=False):
    device = torch.device(device)
    outputs_root = Path(outputs_root).resolve()
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested E004a device is unavailable")
    runs_root = outputs_root / "experiments/E004a/runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    attempt = runs_root / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    attempt.mkdir()
    started = time.monotonic()
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=Path(__file__).resolve().parents[3])
    provenance = {"command": getattr(sys, "orig_argv", sys.argv),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3], text=True).strip(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(), "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "code_sha256": {"runner": checksum(__file__), "analysis": checksum(e004a_analysis.__file__),
            "training": checksum(shared_training.__file__), "source_cache": checksum(e002f_run.__file__),
            "cached_tracks": checksum(e003b_run.__file__), "source_check": checksum(e003c_run.__file__),
            "launcher": checksum(Path(__file__).resolve().parents[3] / "scripts/train-e004a.py")}}
    shared_run.write_json(attempt / "provenance.json", provenance)
    shared_run.write_json(attempt / "status.json", {"status": "incomplete", "smoke": smoke,
        "held_out_tests": False, "failures": []})
    try:
        seed_run(0)
        cache = load_or_build_cache(outputs_root, device)
        training = sorted(cache["records"]["training"], key=lambda row: row["locator"])
        development, assessment, split_identity = split_development_assessment(cache["records"]["validation"])
        for name, rows in (("development", development), ("assessment", assessment)):
            if _population_summary(rows) != EXPECTED_VALIDATION[name]:
                raise ValueError(f"Frozen LOKI {name} population/class support differs from the agreed 46/47 split")
        plan = _plan(training, development, assessment, smoke=smoke) | {"validation_split": split_identity}
        shared_run.write_json(attempt / "subset-plans.json", plan)
        results = [_evaluate_zero(attempt, cache, development, assessment, plan)]
        plan_by_key = {(row["budget"], row["draw"]): row for row in plan["runs"]}
        for key, plan_row in plan_by_key.items():
            selected = {row["locator"] for row in plan_row["selected_tracks"]}
            training_rows = e003c_run._canonical_records(
                [row for row in training if row["locator"] in selected])
            if len(training_rows) != plan_row["budget"]:
                raise ValueError("Persisted E004a subset plan does not match cached training tracks")
            results.append(_run_one(attempt, cache, training_rows, development, assessment,
                plan_row, plan, device, smoke))
        shared_run.write_json(attempt / "results.json", results)
        report_rows = [{"budget": row["budget"], "draw": row["draw"], "macro_f1": row["macro_f1"],
            "track_ce": row["track_ce"], "coverage": row["coverage"],
            "annotation_access": row["annotation_access"], "selected_update": row["selected_update"],
            "stopping_reason": row["stopping_reason"]} for row in results]
        e004a_analysis.make_report(report_rows, attempt / "report")
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        shared_run.write_json(attempt / "provenance.json", provenance)
        shared_run.write_json(attempt / "status.json", {"status": "complete", "smoke": smoke,
            "fit_count": len(plan["runs"]), "zero_anchor": True, "held_out_tests": False,
            "results": len(results), "report": str(attempt / "report")})
        return {"attempt": str(attempt), "cache": str(cache["root"]), "fits": len(plan["runs"]),
                "results": len(results), "smoke": smoke}
    except BaseException as error:
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
            failure={"type": type(error).__name__, "message": str(error)})
        shared_run.write_json(attempt / "provenance.json", provenance)
        shared_run.write_json(attempt / "status.json", {"status": "failed", "smoke": smoke,
            "held_out_tests": False, "failures": [{"type": type(error).__name__, "message": str(error),
                                                     "traceback": traceback.format_exc()}]})
        raise
