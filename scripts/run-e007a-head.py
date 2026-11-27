"""Adapt both E007a checkpoints to Euro-PVI with the frozen E006c head recipe."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import load_predictions
from pedestrian_behavior.experiments import e003b_run, e004a_analysis, e004a_run, e007a_run
from pedestrian_behavior.experiments.e006c_head import (
    AUDIT, DATASET, DEFAULT_SUBSET_PLAN, E006B_RUN, DRAWS, EXPECTED,
    _check_audit, _extract, _load_models, _load_saved_subset_plan, _metric_tracks,
    _split_records,
)
from pedestrian_behavior.models.e005 import build_model
from pedestrian_behavior.training import seed_run


DEFAULT_E007A_RUN = Path("outputs/experiments/E007a/runs/joint-s0-20261126-170812")
E006A_RUN = Path("experiments/E005/runs/joint-g-rgb-utonia-s0-20261122-001910")
ROAD_RUN = Path("experiments/E005/runs/road-waymo-g-rgb-utonia-s0-20261120-030721")
CHECKPOINTS = ("best-f1", "best-ce")
CONTROL = "plain-ce-epoch6"
FIT_BUDGET = 100
ERROR_TARGET = 0.05
SUBSET_PLAN_SHA256 = "556d549ea3fa45692974078e8a14ab8b922d553f45339d30bd31105aaa0f2de1"


def _canonical(value):
    return json.loads(json.dumps(value))


def _load_model(path, checkpoint, config, device):
    model = build_model(config["arm"], config["profile"]).to(device).eval()
    state = checkpoint["model_state"]
    model.load_state_dict(state, strict=True)
    return (
        {"model": model, "config": config, "checkpoint": checkpoint,
         "normalization": checkpoint["normalization"]},
        {"run": str(path.parent.resolve()), "checkpoint": str(path.resolve()),
         "checkpoint_sha256": checksum(path), "epoch": checkpoint["epoch"],
         "selection": checkpoint["selection"],
         "source_head_sha256": e004a_run._head_digest(
             state["head.weight"].cpu().numpy(), state["head.bias"].cpu().numpy())},
    )


def _load_control(run_root, reference, epoch, device):
    path = run_root / "last.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = json.loads((run_root / "config.json").read_text())
    status = json.loads((run_root / "status.json").read_text())
    original = json.loads((reference / "config.json").read_text())
    source_stats = json.loads((reference / "provenance.json").read_text())["source_statistics"]
    if (status["status"] != "complete" or checkpoint["epoch"] != epoch
            or checkpoint["stopping_reason"] != "budget"
            or _canonical(checkpoint["config"]) != config
            or _canonical(checkpoint["normalization"]) != source_stats
            or config["settings"] != (original["settings"] | {"epochs": epoch})
            or any(config.get(key) != original.get(key) for key in
                   ("source", "arm", "experiment", "variant", "profile", "batching",
                    "joint_sampling", "determinism", "held_out_tests", "smoke_limits"))
            or {s: r["manifest_sha256"] for s, r in _source_caches(config).items()}
            != {s: r["manifest_sha256"] for s, r in _source_caches(original).items()}):
        raise ValueError(f"Expected the completed {epoch}-epoch replay of the original {original['source']} recipe")
    history = json.loads((run_root / "history.json").read_text())
    original_history = json.loads((reference / "history.json").read_text())[:epoch]
    metrics = ["validation_f1", "validation_ce",
               *(key for key in original_history[0] if key.startswith("val/"))]
    for actual, expected in zip(history, original_history, strict=True):
        if actual["epoch"] != expected["epoch"] or not np.allclose(
                [actual[k] for k in metrics], [expected[k] for k in metrics], atol=1e-7, rtol=1e-7):
            raise ValueError(f"{config['source']} validation replay differs at epoch {actual['epoch']}")
    return _load_model(path, checkpoint, config, device)


def _load_e007a_models(run_root, device):
    config_path, status_path = run_root / "config.json", run_root / "status.json"
    if not config_path.is_file() or not status_path.is_file():
        raise ValueError(f"Missing completed E007a run metadata: {run_root}")
    config, status = json.loads(config_path.read_text()), json.loads(status_path.read_text())
    expected_contrastive = e007a_run.SETTINGS["contrastive"]
    if (status.get("status") != "complete" or status.get("kind") == "gpu-smoke"
            or config.get("experiment") != "E007a" or config.get("source") != "joint"
            or config.get("configuration") != "g-rgb-utonia" or config.get("variant") != "bilstm-96"
            or config.get("held_out_tests") is not False or config.get("smoke_limits") is not None
            or config.get("settings", {}).get("seed") != 0
            or config.get("contrastive") != expected_contrastive
            or config.get("settings", {}).get("contrastive") != expected_contrastive):
        raise ValueError("E007a run is incomplete or does not match the frozen joint contrastive setup")

    models, evidence = {}, {}
    normalization = None
    for name in CHECKPOINTS:
        path = run_root / f"{name}.pt"
        if not path.is_file():
            raise ValueError(f"Missing E007a {name} checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint.get("checkpoint") != name
                or _canonical(checkpoint.get("config")) != config
                or checkpoint.get("normalization") is None
                or not isinstance(checkpoint.get("model_state"), dict)):
            raise ValueError(f"E007a {name} checkpoint metadata differs from its completed run")
        if normalization is None:
            normalization = checkpoint["normalization"]
        elif _canonical(checkpoint["normalization"]) != _canonical(normalization):
            raise ValueError("E007a selection checkpoints use different frozen normalization")

        models[name], evidence[name] = _load_model(path, checkpoint, config, device)
    return models, evidence


def _source_caches(config):
    return config["caches"] if config["source"] == "joint" else {config["source"]: config["cache"]}


def _check_encoder_lineage(config, e006b):
    caches = _source_caches(config)
    expected = {"road-waymo", "loki"} if config["source"] == "joint" else {"road-waymo"}
    if set(caches) != expected:
        raise ValueError("Checkpoint lacks the frozen source cache lineage")
    for cache in caches.values():
        provenance = cache.get("provenance", {})
        if (provenance.get("DINO", {}).get("checkpoint_sha256") != e006b["encoder_sha256"]["dino"]
                or provenance.get("Tiny-Utonia", {}).get("checkpoint_sha256")
                != e006b["encoder_sha256"]["tiny_utonia"]):
            raise ValueError("E007a and Euro-PVI extraction use different frozen encoders")


def _save_features(output, checkpoint_name, split, rows, audit_sha, plan_sha, evidence):
    path = output / "features" / checkpoint_name / f"{split}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {"experiment": "E007a", "checkpoint": checkpoint_name, "split": split,
        "checkpoint_sha256": evidence[checkpoint_name]["checkpoint_sha256"],
        "audit_sha256": audit_sha, "subset_plan_sha256": plan_sha,
        "representation": "post-temporal-final-layernorm-128d",
        "normalization": "checkpoint source statistics", "dtype": "float32"}
    e003b_run._save_cache(path, rows, identity)
    loaded = e003b_run._load_cache(path, identity)
    if ([row["locator"] for row in loaded] != [row["locator"] for row in rows]
            or any(not np.array_equal(a["features"], b["features"])
                   for a, b in zip(loaded, rows, strict=True))):
        raise ValueError("E007a compact feature cache roundtrip changed track embeddings")
    return {"path": str(path), "sha256": checksum(path), "tracks": len(rows),
            "gt_timesteps": int(sum(int(row["gt_valid"].sum()) for row in rows)),
            "representation": identity["representation"], "metadata": identity}


def _fixed_coverage(prediction_path):
    population = e004a_analysis._population(load_predictions(prediction_path))
    curve = e004a_analysis._curve(population)
    result = {}
    for target in (0.30, 0.60):
        index = int(np.searchsorted(curve["coverage"], target))
        result[str(target)] = {key: float(curve[key][index]) for key in ("coverage", "error", "threshold")}
    return result


def _add_fixed_coverage(result):
    result["assessment_at_fixed_coverage"] = _fixed_coverage(
        Path(result["fit_dir"]) / "assessment-predictions.npz")
    return result


def _summary(rows):
    output = []
    for checkpoint in dict.fromkeys(row["checkpoint"] for row in rows):
        for budget in sorted({row["budget"] for row in rows if row["checkpoint"] == checkpoint}):
            group = [row for row in rows if row["checkpoint"] == checkpoint and row["budget"] == budget]
            metrics = {"assessment_macro_f1": [row["macro_f1"] for row in group],
                       "assessment_track_ce": [row["track_ce"] for row in group]}
            target_rows = [next(item for item in row["coverage"]["targets"]
                                if item["error_target"] == ERROR_TARGET)["assessment"] for row in group]
            metrics["coverage_at_5pct_calibration_error"] = [item["coverage"] for item in target_rows]
            metrics["error_at_5pct_calibration_cutoff"] = [item["error"] for item in target_rows]
            fixed = {str(level): [row["assessment_at_fixed_coverage"][str(level)]["error"] for row in group]
                     for level in (0.30, 0.60)}
            summary = {"checkpoint": checkpoint, "budget": budget,
                "draws": [row["draw"] for row in group], "n": len(group)}
            for name, values in metrics.items():
                observed = [value for value in values if value is not None]
                summary[name + "_mean"] = float(np.mean(observed)) if observed else None
                summary[name + "_sd"] = float(np.std(observed, ddof=1)) if len(observed) > 1 else None
                if name == "error_at_5pct_calibration_cutoff":
                    summary[name + "_n"] = len(observed)
            summary["assessment_error_at_fixed_coverage"] = {
                level: {"mean": float(np.mean(values)),
                        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
                for level, values in fixed.items()}
            output.append(summary)
    return output


def _full_split(rows, audit):
    for name, expected in EXPECTED.items():
        found = e004a_run._population_summary(rows[name])
        if (found["tracks"] != expected["tracks"] or found["gt_timesteps"] != expected["gt_timesteps"]
                or found["groups"] != expected["groups"]
                or len({row["sequence"] for row in rows[name]}) != expected["sequences"]):
            raise ValueError(f"E007a {name} population differs from frozen E006c split: {found}")
    fitting_counts = e004a_run._population_summary(rows["training"])["class_gt_counts"]
    development_counts = e004a_run._population_summary(rows["development"])["class_gt_counts"]
    calibration_counts = audit["partition"]["calibration"]["nominal_5Hz_class_frames"]
    names = ("moving", "stopped", "waiting to cross", "crossing")
    if ((np.asarray(fitting_counts) + np.asarray(development_counts)).tolist()
            != [calibration_counts[name] for name in names]):
        raise ValueError("E007a fit/development labels differ from the frozen calibration audit")
    counts = e004a_run._population_summary(rows["assessment"])["class_gt_counts"]
    expected_counts = audit["partition"]["assessment"]["nominal_5Hz_class_frames"]
    if counts != [expected_counts[name] for name in names]:
        raise ValueError("E007a assessment labels differ from the frozen Euro-PVI audit")


def _copy_plan(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if checksum(source) != checksum(destination):
        raise ValueError("Copied E006c subset plan SHA-256 changed")


def _run_checkpoint(output, checkpoint_name, rows, plan, plan_path, evidence, args, smoke):
    attempt = output / checkpoint_name
    attempt.mkdir()
    _copy_plan(plan_path, attempt / "subset-plans.json")
    source_rows = rows[checkpoint_name]
    cache_root = output / "features" / checkpoint_name
    checkpoint = torch.load(evidence[checkpoint_name]["checkpoint"], map_location="cpu", weights_only=True)
    state = checkpoint["model_state"]
    head_state = {"weight": state["head.weight"].numpy().copy(),
                  "bias": state["head.bias"].numpy().copy()}
    manifest = {"source_checkpoint_sha256": evidence[checkpoint_name]["checkpoint_sha256"],
        "normalization": _canonical(checkpoint["normalization"]),
        "source_head": {"state_sha256": evidence[checkpoint_name]["source_head_sha256"]},
        "derived_features": {"training": {"sha256": checksum(cache_root / "training.npz")},
                             "validation": {"sha256": checksum(cache_root / "development.npz")},
                             "assessment": {"sha256": checksum(cache_root / "assessment.npz")}}}
    e004a_run.shared_run.write_json(cache_root / "cache.json", manifest)
    cache = {"root": cache_root, "manifest": manifest, "head": head_state,
             "source_validation": _metric_tracks(source_rows["development"] + source_rows["assessment"])}
    development, assessment = source_rows["development"], source_rows["assessment"]
    zero = e004a_run._evaluate_zero(attempt, cache, development, assessment, plan, True,
                                    experiment="E007a-head-follow-up")
    zero["checkpoint"], zero["draw"] = checkpoint_name, None
    _add_fixed_coverage(zero)
    results = [zero]
    plan_rows = {(row["budget"], row["draw"]): row for row in plan["runs"]}
    budget_draws = ((1, 0),) if smoke else tuple((FIT_BUDGET, draw) for draw in DRAWS)
    for budget, draw in budget_draws:
        plan_row = plan_rows[budget, draw]
        selected = {entry["locator"] for entry in plan_row["selected_tracks"]}
        fit_rows = sorted((row for row in source_rows["training"] if row["locator"] in selected),
                          key=lambda row: row["locator"])
        if len(fit_rows) != budget:
            raise ValueError(f"Saved E006c selection {budget}/{draw} does not match extracted tracks")
        result = e004a_run._run_one(attempt, cache, fit_rows, development, assessment, plan_row,
            plan, torch.device(args.fit_device), smoke, True, experiment="E007a-head-follow-up",
            run_name_prefix=f"e007a-{checkpoint_name}-head", source=checkpoint["config"]["source"], target="euro-pvi",
            wandb_enabled=False)
        result["checkpoint"], result["draw"] = checkpoint_name, draw
        _add_fixed_coverage(result)
        results.append(result)
        print(f"E007a {checkpoint_name}, {budget} tracks, draw {draw}: "
              f"assessment F1={result['macro_f1']:.4f}", flush=True)
    return results


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh E007a follow-up output directory: {output}")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E007a Euro-PVI feature extraction requires CUDA")
    if args.fit_device not in ("cpu", "cuda") or (args.fit_device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested head-fitting device is unavailable")
    output.mkdir(parents=True)
    status = {"status": "incomplete", "experiment": "E007a-head-follow-up",
              "smoke": args.smoke, "held_out_tests": False}
    e004a_run.shared_run.write_json(output / "status.json", status)
    started = time.monotonic()
    provenance = {"command": getattr(sys, "orig_argv", sys.argv),
        "started_utc": datetime.now(timezone.utc).isoformat(), "dataset": str(args.dataset.resolve()),
        "audit": str(args.audit.resolve()), "e006b_run": str(args.e006b_run.resolve()),
                  "source_run": str((args.road_epoch11_run or args.e006a_epoch6_run or args.e007a_run).resolve()),
                  "subset_plan": str(args.subset_plan.resolve()),
        "device": args.device, "fit_device": args.fit_device, "smoke": args.smoke,
        "held_out_tests": False}
    try:
        torch.set_num_threads(2)
        determinism = seed_run(0)
        audit, cal_groups, fit_groups, dev_groups, assessment_groups = _check_audit(args.audit, args.dataset)
        plan_sha = checksum(args.subset_plan)
        if plan_sha != SUBSET_PLAN_SHA256:
            raise ValueError("E007a follow-up must use the pinned completed E006c subset plan")
        if args.road_epoch11_run:
            model, record = _load_control(
                args.road_epoch11_run.resolve(), args.outputs_root / ROAD_RUN, 11, args.device)
            models, evidence = {"road-ce-epoch11": model}, {"road-ce-epoch11": record}
        elif args.e006a_epoch6_run:
            model, record = _load_control(
                args.e006a_epoch6_run.resolve(), args.outputs_root / E006A_RUN, 6, args.device)
            models, evidence = {CONTROL: model}, {CONTROL: record}
        else:
            models, evidence = _load_e007a_models(args.e007a_run.resolve(), args.device)
        reference_model = next(iter(models))
        _, _, probe, tiny_path, e006b = _load_models(
            args.outputs_root, args.e006b_run, args.device, sources=())
        for model in models.values():
            _check_encoder_lineage(model["config"], e006b)
        provenance.update(subset_plan_sha256=plan_sha,
            source_checkpoints={name: value["checkpoint_sha256"] for name, value in evidence.items()},
            encoder_sha256=e006b["encoder_sha256"], determinism=determinism)

        encoded, tiny_sweeps = _extract(args, cal_groups, assessment_groups, models, probe, tiny_path,
                                        smoke=args.smoke)
        if args.smoke:
            sequences = ("train/0067", "train/0158", "train/0339")
            fit_groups = [group for group in cal_groups if sequences[0] in group]
            dev_groups = [group for group in cal_groups if sequences[1] in group]
            assessment_groups = [group for group in cal_groups if sequences[2] in group]
            if any(len(group) != 1 for group in fit_groups + dev_groups + assessment_groups):
                raise ValueError("E006c smoke sequences must remain separate timestamp groups")
        split_results = _split_records(encoded,
            [sequence for group in fit_groups for sequence in group],
            [sequence for group in dev_groups for sequence in group],
            [sequence for group in assessment_groups for sequence in group])
        if args.smoke:
            plan = {"status": "smoke", "experiment": "E006c",
                "reference_subset_plan_sha256": plan_sha,
                "fit_groups": [list(group) for group in fit_groups],
                "development_groups": [list(group) for group in dev_groups],
                "assessment_groups": [list(group) for group in assessment_groups],
                "population": {name: e004a_run._population_summary(split_results[reference_model][name])
                               for name in ("training", "development", "assessment")},
                "protocol": {"budget_tracks": [1], "draws": [0], "fit_seed": 0,
                             "error_target": ERROR_TARGET}, "runs": []}
            fit = sorted(split_results[reference_model]["training"], key=lambda row: row["locator"])
            if not fit:
                raise ValueError("E006c smoke fitting sequence has no scored track")
            plan["runs"] = [{"budget": 1, "draw": 0,
                "selected_tracks": [e004a_run._track_label_record(fit[0])]}]
            # Save the actual frozen E006c plan at the root; smoke fits use their own clearly marked plan.
            _copy_plan(args.subset_plan, output / "subset-plans.json")
        else:
            for source_rows in split_results.values():
                _full_split(source_rows, audit)
            plan = _load_saved_subset_plan(args.subset_plan, split_results[reference_model]["training"],
                                           fit_groups, dev_groups, assessment_groups)
            _copy_plan(args.subset_plan, output / "subset-plans.json")
        plan_path = output / ("smoke-plan.json" if args.smoke else "subset-plans.json")
        if args.smoke:
            e004a_run.shared_run.write_json(plan_path, plan)

        for split in ("training", "development", "assessment"):
            baseline = split_results[reference_model][split]
            for name in list(split_results)[1:]:
                other = split_results[name][split]
                if [row["locator"] for row in baseline] != [row["locator"] for row in other]:
                    raise ValueError(f"E007a {split} track identities differ between checkpoints")
                for left, right in zip(baseline, other, strict=True):
                    if (left["group"] != right["group"] or left["sequence"] != right["sequence"]
                            or not np.array_equal(left["targets"], right["targets"])
                            or not np.array_equal(left["gt_valid"], right["gt_valid"])):
                        raise ValueError(f"E007a {split} labels differ between checkpoints")

        if not args.smoke:
            saved_population = plan.get("population", {})
            if any(saved_population.get(split) != e004a_run._population_summary(split_results[reference_model][split])
                   for split in ("training", "development", "assessment")):
                raise ValueError("Saved E006c population provenance differs from extracted Euro-PVI tracks")

        feature_summaries = {}
        for name, rows in split_results.items():
            feature_summaries[name] = {}
            for split in ("training", "development", "assessment"):
                feature_summaries[name][split] = _save_features(
                    output, name, split, rows[split], checksum(args.audit), plan_sha, evidence)

        del models
        if args.device == "cuda":
            torch.cuda.empty_cache()
        all_results = []
        for name in split_results:
            all_results.extend(_run_checkpoint(output, name, split_results, plan, plan_path,
                                               evidence, args, args.smoke))
        report = {"status": "complete", "experiment": "E007a-head-follow-up",
            "kind": "smoke" if args.smoke else "matched-head-adaptation",
            "budgets": [0, 1] if args.smoke else [0, FIT_BUDGET],
            "draws": [0] if args.smoke else list(DRAWS), "error_target": ERROR_TARGET,
            "retrospective_coverage": [0.30, 0.60], "results": all_results,
            "summary": _summary(all_results), "features": feature_summaries,
            "split": {"fit_groups": [list(group) for group in fit_groups],
                      "development_groups": [list(group) for group in dev_groups],
                      "assessment_groups": [list(group) for group in assessment_groups],
                      "population": {name: e004a_run._population_summary(split_results[reference_model][name])
                                     for name in ("training", "development", "assessment")}},
            "subset_plan": {"path": str(args.subset_plan.resolve()), "sha256": plan_sha,
                            "run_plan": str(plan_path.resolve()), "run_plan_sha256": checksum(plan_path)},
            "source_models": evidence, "audit_sha256": checksum(args.audit),
            "encoder_sha256": e006b["encoder_sha256"], "tiny_sweeps": tiny_sweeps,
            "normalization": "each checkpoint's frozen joint E007a source statistics",
            "assessment_used_for_selection_or_cutoffs": False}
        e004a_run.shared_run.write_json(output / "summary.json", report)
        provenance.update(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
            status="complete")
        e004a_run.shared_run.write_json(output / "provenance.json", provenance)
        status.update(status="complete", results=len(all_results), report=str(output / "summary.json"))
        e004a_run.shared_run.write_json(output / "status.json", status)
        print(f"E007a Euro-PVI head follow-up complete: {output}", flush=True)
        return report
    except BaseException as error:
        status.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc()})
        provenance.update(status="failed", failure={"type": type(error).__name__, "message": str(error)},
            elapsed_seconds=time.monotonic() - started)
        e004a_run.shared_run.write_json(output / "status.json", status)
        e004a_run.shared_run.write_json(output / "provenance.json", provenance)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--e006b-run", type=Path, default=E006B_RUN)
    parser.add_argument("--e007a-run", type=Path, default=DEFAULT_E007A_RUN)
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument("--e006a-epoch6-run", type=Path, help="Evaluate only the epoch-6 plain CE control")
    replay.add_argument("--road-epoch11-run", type=Path, help="Evaluate only the ROAD minimum-CE replay")
    parser.add_argument("--subset-plan", type=Path, default=DEFAULT_SUBSET_PLAN)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--fit-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true",
                        help="Use the existing three-sequence smoke split, one track and up to 200 updates")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
