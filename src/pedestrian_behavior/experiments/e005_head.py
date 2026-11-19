"""Fit the original LOKI E005 head on nested ROAD training subsets."""

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

from pedestrian_behavior import training as shared_training
from pedestrian_behavior.data.preparation import checksum

from . import e003b_run, e003c_run, e004a_analysis, e004a_run
from .run import write_json


BUDGETS = (25, 50, 100, 250, 500, 1000)
DRAWS = tuple(range(5))
SOURCE, TARGET = "loki", "road-waymo"
RUN_NAME = "e005-loki-to-road-head"
EXPECTED = {"training": (4380, 358, 332940, [86095, 39059, 11336, 30790]),
            "validation": (1110, 78, 87824, [24018, 11232, 3317, 6821])}


def _split_validation(records):
    clips = sorted({row["group"] for row in records})
    if len(clips) < 2:
        raise ValueError("E005 head budgets require at least two ROAD validation clips")
    shuffled = np.random.default_rng(0).permutation(clips)
    development_clips = set(shuffled[:len(clips) // 2].tolist())
    development = [row for row in records if row["group"] in development_clips]
    assessment = [row for row in records if row["group"] not in development_clips]
    return development, assessment


def _plan(training, development, assessment, source_sha, *, smoke):
    draws = DRAWS[:1] if smoke else DRAWS
    budgets = BUDGETS[:1] if smoke else BUDGETS
    if len(training) < budgets[-1]:
        raise ValueError(f"E005 head budgets need at least {budgets[-1]} cached training tracks")
    runs = []
    for draw in draws:
        order = e004a_run.recording_round_robin(training, draw)
        for budget in budgets:
            runs.append({"budget": budget, "draw": draw, "selected_tracks": [
                e004a_run._track_label_record(row) for row in order[:budget]]})
    max_updates = 200 if smoke else e004a_run.SETTINGS["max_updates"]
    return {"status": "planned", "experiment": "E005", "kind": RUN_NAME, "smoke": smoke,
        "protocol": {"source": SOURCE, "target": TARGET, "source_checkpoint_sha256": source_sha,
            "draw_seeds": list(draws), "budget_grid": [0, *budgets], "fit_seed": 0,
            "training_settings": {**e004a_run.SETTINGS, "max_updates": max_updates, "patience_checks": None},
            "patience_updates": e004a_run.EARLY_PATIENCE_UPDATES,
            "check_updates": [0, *e004a_run._check_schedule(True, max_updates)],
            "development_groups_seed": 0, "error_targets": e004a_run.ERROR_TARGETS},
        "population": {name: e004a_run._population_summary(rows) for name, rows in
            (("training", training), ("development", development), ("assessment", assessment))},
        "validation_split": {"seed": 0, "development_groups": sorted({row["group"] for row in development}),
            "assessment_groups": sorted({row["group"] for row in assessment})}, "runs": runs}


def run(output, cache, device="cpu", *, smoke=False):
    """Run head fits from a verified, source-normalized ROAD embedding cache."""
    device = torch.device(device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested E005 head-fitting device is unavailable")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "incomplete", "experiment": "E005", "kind": RUN_NAME,
              "smoke": smoke, "held_out_tests": False}
    write_json(output / "status.json", status)
    repo = Path(__file__).resolve().parents[3]
    provenance = {"started_utc": datetime.now(timezone.utc).isoformat(),
        "command": getattr(sys, "orig_argv", sys.argv), "device": str(device), "torch": str(torch.__version__),
        "numpy": np.__version__, "python": platform.python_version(), "wandb_enabled": False}
    try:
        manifest = cache["manifest"]
        source_sha = manifest["source_checkpoint_sha256"]
        provenance.update(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)).hexdigest(),
            code_sha256={module.__name__.rsplit(".", 1)[-1]: checksum(module.__file__)
                         for module in (e003b_run, e004a_run, e004a_analysis, shared_training)} | {
                             "e005_head": checksum(__file__)},
            cache_manifest=str(cache["root"] / "cache.json"), source_checkpoint_sha256=source_sha,
            source_head_sha256=manifest["source_head"]["state_sha256"])
        write_json(output / "provenance.json", provenance)

        training, validation = (e003c_run._canonical_records(cache["records"][split])
                                for split in ("training", "validation"))
        if {row["group"] for row in training} & {row["group"] for row in validation}:
            raise ValueError("E005 ROAD training and validation clips overlap")
        if not smoke:
            for split, records in (("training", training), ("validation", validation)):
                population = e004a_run._population_summary(records)
                found = (population["tracks"], population["groups"], sum(len(row["targets"]) for row in records),
                         population["class_gt_counts"])
                if found != EXPECTED[split]:
                    raise ValueError(f"E005 cached ROAD {split} population differs from the frozen split")
        development, assessment = _split_validation(validation)
        plan = _plan(training, development, assessment, source_sha, smoke=smoke)
        write_json(output / "subset-plans.json", plan)
        print("E005 ROAD head budgets: evaluating the zero-fitting-label anchor", flush=True)
        results = [e004a_run._evaluate_zero(output, cache, development, assessment, plan, True, "E005")]
        training_by_locator = {row["locator"]: row for row in training}
        for index, row in enumerate(plan["runs"], start=1):
            print(f"E005 ROAD head fit {index}/{len(plan['runs'])}, {row['budget']:,} tracks, draw {row['draw']}", flush=True)
            selected = [training_by_locator[item["locator"]] for item in row["selected_tracks"]]
            results.append(e004a_run._run_one(output, cache, e003c_run._canonical_records(selected),
                development, assessment, row, plan, device, smoke, True, experiment="E005",
                run_name_prefix=RUN_NAME, source=SOURCE, target=TARGET, wandb_enabled=False))
        write_json(output / "results.json", results)
        report = output / "report"
        e004a_analysis.make_report(results, report, all_tracks=None)
        status.update(status="complete", fit_count=len(plan["runs"]), results=len(results), report=str(report))
    except BaseException as error:
        status.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                                                 "traceback": traceback.format_exc()})
        raise
    finally:
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        write_json(output / "provenance.json", provenance)
        write_json(output / "status.json", status)
    return {"attempt": str(output), "fits": status["fit_count"], "results": status["results"], "smoke": smoke}
