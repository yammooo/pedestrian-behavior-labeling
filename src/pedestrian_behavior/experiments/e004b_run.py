"""E004b class-covered LOKI subsets using the frozen E004a source-head recipe."""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.experiments import e004a_analysis, e004a_run
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.training import seed_run


EXPERIMENT = "E004b"
SOURCE, TARGET = e004a_run.SOURCE, e004a_run.TARGET
BUDGETS = (5, 10, 25, 50, 100, 250, 500, 1000)
ALL_TRACKS = e004a_run.ALL_TRACKS
MIN_ROLE_GT = 5
POLICY = ("For each draw, shuffle the four class roles and uniformly select one track per role "
    "with at least five accepted GT frames of that class, excluding prior tracks and scenarios; "
    "prepend the four anchors once, then append the E004a recording-round-robin order without them.")


def _support(row):
    return np.bincount(row["targets"][row["gt_valid"]], minlength=len(e004a_run.CLASSES)).astype(int)


def _track_record(row):
    counts = _support(row)
    return {"locator": row["locator"], "group": row["group"],
        "gt_timesteps": int(counts.sum()), "class_gt_counts": counts.tolist()}


def _candidate_access(records):
    candidates = sorted(records, key=lambda row: row["locator"])
    support_rows = [_track_record(row) for row in candidates]
    classes = []
    for class_id, name in enumerate(e004a_run.CLASSES):
        eligible = [row for row in support_rows if row["class_gt_counts"][class_id] >= MIN_ROLE_GT]
        classes.append({"class_id": class_id, "class_name": name,
            "tracks_with_any_class_gt": sum(row["class_gt_counts"][class_id] > 0 for row in support_rows),
            "gt_class_timesteps": int(sum(row["class_gt_counts"][class_id] for row in support_rows)),
            "eligible_tracks_at_least_5_gt": len(eligible),
            "eligible_scenarios_at_least_5_gt": len({row["group"] for row in eligible})})
    return {"candidate_tracks": len(candidates), "candidate_scenarios": len({row["group"] for row in candidates}),
        "accessed_metadata": "accepted GT class-support counts for every candidate training track",
        "minimum_role_gt_timesteps": MIN_ROLE_GT, "classes": classes,
        "candidate_track_class_support": support_rows,
        "candidate_support_path": "subset-plans.json#screening_access.candidate_track_class_support",
        "practical_screening_cost": "unknown; screening time was not measured",
        "separate_from_dense_label_accounting": True}


def class_covered_order(records, draw):
    """Select one >=5-GT anchor per class, then append the E004a draw remainder."""
    candidates = e004a_run.e003c_run._canonical_records(records)
    if not candidates:
        raise ValueError("E004b cannot sample an empty LOKI training pool")
    counts = {row["locator"]: _support(row) for row in candidates}
    rng = np.random.default_rng(draw)
    role_order = [int(value) for value in rng.permutation(len(e004a_run.CLASSES))]
    anchors, used_groups = [], set()
    for class_id in role_order:
        eligible = [row for row in candidates
            if counts[row["locator"]][class_id] >= MIN_ROLE_GT and row["group"] not in used_groups]
        if not eligible:
            name = e004a_run.CLASSES[class_id]
            raise ValueError(f"E004b draw {draw} infeasible: no >=5-GT {name} track in a remaining scenario")
        chosen = eligible[int(rng.integers(len(eligible)))]
        role = {"class_id": class_id, "class_name": e004a_run.CLASSES[class_id],
            "locator": chosen["locator"], "group": chosen["group"],
            "role_gt_timesteps": int(counts[chosen["locator"]][class_id]),
            "track_gt_timesteps": int(chosen["gt_valid"].sum()),
            "class_gt_counts": counts[chosen["locator"]].tolist()}
        anchors.append((chosen, role))
        used_groups.add(chosen["group"])
    anchor_ids = {row["locator"] for row, _ in anchors}
    remainder = [row for row in e004a_run.recording_round_robin(candidates, draw)
        if row["locator"] not in anchor_ids]
    ordered = [row for row, _ in anchors] + remainder
    if len(ordered) != len(candidates) or len({row["locator"] for row in ordered}) != len(candidates):
        raise ValueError("E004b sampler must return every training track exactly once")
    if len({row["group"] for row, _ in anchors}) != len(e004a_run.CLASSES):
        raise ValueError("E004b class anchors must come from four distinct scenarios")
    sampling = {"policy": POLICY, "draw": draw, "role_order": [e004a_run.CLASSES[i] for i in role_order],
        "anchors": [role for _, role in anchors],
        "remainder": "E004a recording_round_robin(training, draw), excluding the four selected anchors"}
    return ordered, sampling


def build_plan(training, development, assessment, *, smoke=False):
    training = e004a_run.e003c_run._canonical_records(training)
    if len(training) != ALL_TRACKS:
        raise ValueError(f"E004b requires all {ALL_TRACKS} frozen LOKI training tracks")
    screening = _candidate_access(training)
    candidate_support = screening.pop("candidate_track_class_support")
    draws, budgets = ((0,), (5, 10)) if smoke else (tuple(range(5)), BUDGETS)
    runs, draw_sampling = [], {}
    for draw in draws:
        ordered, sampling = class_covered_order(training, draw)
        draw_sampling[draw] = sampling
        for budget in budgets:
            runs.append({"budget": budget, "draw": draw,
                "selected_tracks": [_track_record(row) for row in ordered[:budget]],
                "sampling": sampling})
    if not smoke:
        runs.append({"budget": ALL_TRACKS, "draw": None,
            "selected_tracks": [_track_record(row) for row in training],
            "sampling": {"policy": "canonical full-data fit", "anchors_used": False}})
    max_updates = 200 if smoke else e004a_run.SETTINGS["max_updates"]
    schedule = e004a_run._check_schedule(True, max_updates)
    return {"status": "planned", "experiment": EXPERIMENT, "smoke": smoke,
        "protocol": {"source_checkpoint_sha256": e004a_run.SOURCE_SHA256,
            "training_seed": 0, "draw_seeds": list(draws),
            "budget_grid": list(budgets) + ([] if smoke else [ALL_TRACKS]),
            "fit_seed": 0, "optimizer": e004a_run.SETTINGS["optimizer"], "max_optimizer_updates": max_updates,
            "check_updates": [0, *schedule], "patience_updates": e004a_run.EARLY_PATIENCE_UPDATES,
            "early_checks": True, "minimum_role_gt_timesteps": MIN_ROLE_GT,
            "sampler_policy": POLICY, "sampler_rng_seed": "draw id; independent from E004a remainder sampler"},
        "population": {"training": e004a_run._population_summary(training),
            "development": e004a_run._population_summary(development),
            "assessment": e004a_run._population_summary(assessment)},
        "screening_access": {"accounting": screening, "candidate_track_class_support": candidate_support},
        "draw_sampling": draw_sampling, "runs": runs}


def run(device="cuda", outputs_root=Path("outputs"), smoke=False):
    device = torch.device(device)
    outputs_root = Path(outputs_root).resolve()
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested E004b device is unavailable")
    cache_root = outputs_root / "experiments/E004a/features/postnorm-bilstm-96"
    runs_root = outputs_root / "experiments/E004b/runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    attempt = runs_root / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%fZ")
    attempt.mkdir()
    started = time.monotonic()
    repository = Path(__file__).resolve().parents[3]
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=repository)
    provenance = {"experiment": EXPERIMENT, "command": getattr(sys, "orig_argv", sys.argv),
        "started_utc": datetime.now(timezone.utc).isoformat(), "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "diff_sha256": hashlib.sha256(diff).hexdigest(), "python": platform.python_version(),
        "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "early_checks": True, "smoke": smoke,
        "cache_manifest": str(cache_root / "cache.json"), "sampler_policy": POLICY}
    write_json(attempt / "provenance.json", provenance)
    write_json(attempt / "status.json", {"status": "incomplete", "experiment": EXPERIMENT,
        "smoke": smoke, "early_checks": True, "held_out_tests": False, "failures": []})
    try:
        seed_run(0)
        cache = e004a_run._load_compact_cache(cache_root)
        training = e004a_run.e003c_run._canonical_records(cache["records"]["training"])
        development, assessment, split = e004a_run.split_development_assessment(cache["records"]["validation"])
        for name, rows in (("development", development), ("assessment", assessment)):
            if e004a_run._population_summary(rows) != e004a_run.EXPECTED_VALIDATION[name]:
                raise ValueError(f"Frozen LOKI {name} population/class support differs from the agreed 46/47 split")
        plan = build_plan(training, development, assessment, smoke=smoke) | {"validation_split": split}
        plan_path = attempt / "subset-plans.json"
        write_json(plan_path, plan)
        plan_sha = checksum(plan_path)
        provenance.update(subset_plan=str(plan_path), subset_plan_sha256=plan_sha,
            screening_access=plan["screening_access"]["accounting"])
        write_json(attempt / "provenance.json", provenance)

        results = [e004a_run._evaluate_zero(attempt, cache, development, assessment, plan,
            early_checks=True, experiment=EXPERIMENT)]
        for plan_row in plan["runs"]:
            selected = {row["locator"] for row in plan_row["selected_tracks"]}
            training_rows = [row for row in training if row["locator"] in selected]
            if len(training_rows) != plan_row["budget"]:
                raise ValueError("Persisted E004b subset plan does not match cached training tracks")
            results.append(e004a_run._run_one(attempt, cache, training_rows, development, assessment,
                plan_row, plan, device, smoke, True, experiment=EXPERIMENT,
                run_name_prefix="e004b-road-to-loki-head", run_name_suffix="-class-covered"))
        write_json(attempt / "results.json", results)
        report_rows = [{"budget": row["budget"], "draw": row["draw"], "macro_f1": row["macro_f1"],
            "track_ce": row["track_ce"], "coverage": row["coverage"],
            "annotation_access": row["annotation_access"], "selected_update": row["selected_update"],
            "stopping_reason": row["stopping_reason"], "sampler_policy": POLICY} for row in results]
        e004a_analysis.make_report(report_rows, attempt / "report")
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(attempt / "provenance.json", provenance)
        write_json(attempt / "status.json", {"status": "complete", "experiment": EXPERIMENT,
            "smoke": smoke, "early_checks": True, "fit_count": len(plan["runs"]),
            "zero_anchor": True, "held_out_tests": False, "results": len(results),
            "report": str(attempt / "report")})
        return {"experiment": EXPERIMENT, "attempt": str(attempt), "cache": str(cache_root),
            "fits": len(plan["runs"]), "results": len(results), "smoke": smoke, "early_checks": True}
    except BaseException as error:
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started,
            failure={"type": type(error).__name__, "message": str(error)})
        write_json(attempt / "provenance.json", provenance)
        write_json(attempt / "status.json", {"status": "failed", "experiment": EXPERIMENT,
            "smoke": smoke, "early_checks": True, "held_out_tests": False,
            "failures": [{"type": type(error).__name__, "message": str(error),
                          "traceback": traceback.format_exc()}]})
        raise
