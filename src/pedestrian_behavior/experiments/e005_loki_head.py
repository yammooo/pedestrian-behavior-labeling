"""Fit the ROAD E005 classifier head on matched LOKI E004a label budgets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import load_predictions, predict
from pedestrian_behavior.experiments import e003b_run, e003c_run, e004a_analysis, e004a_run
from pedestrian_behavior.models import e005
from pedestrian_behavior.training import seed_run

from . import e005_road_to_loki, e005_run
from .run import write_json


SOURCE_RUN = Path("outputs/experiments/E005/runs/road-waymo-g-rgb-utonia-s0-20261120-030721")
LOKI_CACHE = Path("outputs/experiments/E005/loki-cache-full-20261117")
E004A_REFERENCE = Path("outputs/experiments/E004a/runs/20261113-082847-737608Z")
DIRECT_TRANSFER = Path("outputs/experiments/E005/transfer/road-to-loki-direct-20261120")
BUDGETS = (25, 50, 100, 250, 500, 1000)
DRAWS = tuple(range(5))
EXPECTED_EXTRACTED_TRAINING = 3099


def _reference_plan(reference, training_entries, development, assessment):
    path = Path(reference) / "subset-plans.json"
    saved = json.loads(path.read_text())
    protocol = saved.get("protocol", {})
    if (saved.get("experiment") != "E004a" or saved.get("smoke") is not False
            or protocol.get("early_checks") is not True
            or protocol.get("check_updates") != [0, *e004a_run.EARLY_CHECK_UPDATES]
            or protocol.get("patience_updates") != e004a_run.EARLY_PATIENCE_UPDATES
            or protocol.get("fit_seed") != 0
            or protocol.get("training_seed") != 0
            or protocol.get("max_optimizer_updates") != e004a_run.SETTINGS["max_updates"]
            or protocol.get("optimizer") != e004a_run.SETTINGS["optimizer"]):
        raise ValueError("Reference must be the full E004a early-check run with the agreed fit recipe")
    expected_split = {
        "development_groups": sorted({row["group"] for row in development}),
        "assessment_groups": sorted({row["group"] for row in assessment}),
    }
    if any(saved.get("validation_split", {}).get(key) != groups for key, groups in expected_split.items()):
        raise ValueError("E004a and E005 LOKI validation partitions differ")

    by_locator = {row["locator"]: row for row in training_entries}
    wanted = {(budget, draw) for budget in BUDGETS for draw in DRAWS}
    selected_runs = [row for row in saved.get("runs", []) if (row.get("budget"), row.get("draw")) in wanted]
    if len(selected_runs) != len(wanted) or {(row["budget"], row["draw"]) for row in selected_runs} != wanted:
        raise ValueError("E004a reference does not contain exactly the requested six budgets × five draws")
    for row in selected_runs:
        selected = row.get("selected_tracks", [])
        if len(selected) != row["budget"] or len({item["locator"] for item in selected}) != row["budget"]:
            raise ValueError(f"Invalid saved E004a subset for budget {row['budget']}, draw {row['draw']}")
        for item in selected:
            current = by_locator.get(item["locator"])
            if current is None or item.get("group") != current["group"]:
                raise ValueError(f"Saved E004a selected track changed: {item.get('locator')}")
    by_draw = {draw: [row for row in selected_runs if row["draw"] == draw] for draw in DRAWS}
    for draw, rows in by_draw.items():
        rows.sort(key=lambda row: BUDGETS.index(row["budget"]))
        for smaller, larger in zip(rows, rows[1:]):
            if smaller["selected_tracks"] != larger["selected_tracks"][:smaller["budget"]]:
                raise ValueError(f"E004a budget subsets are not nested for draw {draw}")
    union = {item["locator"] for row in selected_runs for item in row["selected_tracks"]}
    if len(union) != EXPECTED_EXTRACTED_TRAINING:
        raise ValueError(f"Expected {EXPECTED_EXTRACTED_TRAINING} selected training tracks, found {len(union)}")

    for name, rows in (("development", development), ("assessment", assessment)):
        actual = e004a_run._population_summary(rows)
        if actual != saved.get("population", {}).get(name):
            raise ValueError(f"Frozen LOKI {name} population differs from E004a")
    return saved, selected_runs, union


def _verify_selected_labels(selected_runs, training_records):
    by_locator = {row["locator"]: row for row in training_records}
    for run in selected_runs:
        for item in run["selected_tracks"]:
            current = by_locator.get(item["locator"])
            if current is None or item != e004a_run._track_label_record(current):
                raise ValueError(f"Saved E004a GT labels changed for {item['locator']}")


def _extract(model, dataset, group_by_locator, max_tracks, device, split):
    batches = e005_run.SpatialBatches(dataset, {"max_tracks": max_tracks}, training=False,
                                      prefetch=True, prefetch_workers=2)
    captured = []

    def forward(current, batch, _):
        logits = current(batch, mode="ordinary")
        if len(captured) != 1:
            raise ValueError("E005 final LayerNorm hook did not run exactly once")
        features = captured.pop()
        if (features.shape != (*logits.shape[:2], 128) or features.dtype != torch.float32
                or not torch.isfinite(features).all()):
            raise ValueError("Expected finite float32 E005 post-temporal features")
        for index, length in enumerate(batch["lengths"].tolist()):
            metadata = batch["prediction_metadata"][index]
            metadata["features"] = features[index, :length].cpu().numpy().copy()
            metadata["group"] = group_by_locator[batch["locators"][index]]
        return logits

    hook = model.final_norm.register_forward_hook(lambda _module, _inputs, value: captured.append(value))
    records = []
    seen = 0

    def progress(batches):
        nonlocal seen
        for batch in batches:
            seen += len(batch["locators"])
            if seen == len(dataset) or seen // 128 > (seen - len(batch["locators"])) // 128:
                print(f"LOKI {split}: {seen}/{len(dataset)} tracks", flush=True)
            yield batch

    try:
        prediction_batches = progress(batches.predictions())
        for row in predict(model, prediction_batches, device,
                           forward_batch=lambda current, batch, _: forward(current, batch, None)):
            row["source_logits"] = row.pop("logits")
            records.append(row)
    finally:
        hook.remove()
    if len(records) != len(dataset) or {row["locator"] for row in records} != set(group_by_locator):
        raise ValueError(f"LOKI {split} frozen-feature extraction changed the track population")
    return records


def e005_transfer_check(actual, expected):
    """Validate extraction against the saved direct logits and return that anchor."""
    saved = {row["locator"]: row for row in expected}
    if len(saved) != len(expected) or {row["locator"] for row in actual} != set(saved):
        raise ValueError("E005 replay and saved direct predictions have different track identities")
    max_delta = 0.0
    for row in actual:
        prior = saved[row["locator"]]
        logits = row["source_logits"]
        if (row["group"] != prior["group"] or Path(row["archive"]).name != Path(prior["archive"]).name
                or not np.array_equal(row["targets"], prior["targets"])
                or not np.array_equal(row["gt_valid"], prior["gt_valid"])
                or set(row["source_flags"]) != set(prior["source_flags"])
                or any(not np.array_equal(value, prior["source_flags"][name])
                       for name, value in row["source_flags"].items())
                or logits.shape != prior["logits"].shape
                or not np.allclose(logits, prior["logits"], rtol=1e-4, atol=1e-5)):
            raise ValueError(f"E005 LOKI direct replay differs for {row['locator']}")
        max_delta = max(max_delta, float(np.abs(logits - prior["logits"]).max()))
    return [saved[row["locator"]] for row in actual], max_delta


def _plan(reference_plan, selected_runs, development, assessment, source_sha, reference_sha):
    return {"status": "planned", "experiment": "E005", "kind": "road-to-loki-head-budget",
        "smoke": False, "held_out_tests": False,
        "protocol": {"source": "road-waymo", "target": "loki", "source_checkpoint_sha256": source_sha,
            "matched_e004a_plan_sha256": reference_sha, "budget_grid": [0, *BUDGETS],
            "draw_seeds": list(DRAWS), "fit_seed": 0, "early_checks": True,
            "optimizer": e004a_run.SETTINGS["optimizer"],
            "max_optimizer_updates": e004a_run.SETTINGS["max_updates"],
            "check_updates": reference_plan["protocol"]["check_updates"],
            "patience_updates": e004a_run.EARLY_PATIENCE_UPDATES,
            "training_settings": e004a_run.SETTINGS},
        "population": {"training": reference_plan["population"]["training"],
            "development": e004a_run._population_summary(development),
            "assessment": e004a_run._population_summary(assessment)},
        "validation_split": reference_plan["validation_split"],
        "runs": selected_runs}


def run(args):
    outputs_root, output = args.outputs_root.resolve(), args.output.resolve()
    source_run, cache_path = args.source_run.resolve(), args.cache.resolve()
    device = torch.device(args.device)
    reference, direct = args.reference.resolve(), args.direct.resolve()
    allowed = (outputs_root / "experiments/E005/head-budgets/runs").resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError("Use a fresh output directory under outputs/experiments/E005/head-budgets/runs")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")

    source_path, config, checkpoint, source_provenance = e005_road_to_loki._load_source(source_run)
    frozen, spatial_cache, baselines = e005_road_to_loki._frozen_data(
        outputs_root, checkpoint, source_provenance, cache_path)
    output.mkdir(parents=True)
    started = time.monotonic()
    status = {"status": "incomplete", "experiment": "E005", "kind": "road-to-loki-head-budget",
              "held_out_tests": False, "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    try:
        # The saved direct predictions pin the two E004a validation partitions before extraction.
        reference_status = json.loads((reference / "status.json").read_text())
        reference_json = json.loads((reference / "subset-plans.json").read_text())
        if (reference_status.get("status") != "complete"
                or reference_json.get("experiment") != "E004a" or reference_json.get("smoke") is not False
                or reference_json.get("protocol", {}).get("early_checks") is not True):
            raise ValueError("Expected the full E004a early-check subset plan")
        direct_status = json.loads((direct / "status.json").read_text())
        direct_comparison = json.loads((direct / "comparison.json").read_text())
        direct_config = json.loads((direct / "config.json").read_text())
        direct_predictions = load_predictions(direct / "loki-validation/predictions.npz")
        if (direct_status.get("status") != "complete" or direct_comparison.get("held_out_tests") is not False
                or direct_config.get("training_source") != "road-waymo"
                or direct_config.get("target_source") != "loki"
                or direct_comparison.get("checkpoint_sha256") != checksum(source_path)):
            raise ValueError("Saved direct-transfer predictions do not match the pinned ROAD E005 checkpoint")

        group_by_locator = {entry["locator"]: entry["group"] for entry in baselines["validation"].entries}
        if len(direct_predictions) != len(group_by_locator):
            raise ValueError("Saved direct predictions do not cover full LOKI validation")
        for row in direct_predictions:
            if row["group"] != group_by_locator.get(row["locator"]):
                raise ValueError(f"Direct prediction group differs from frozen LOKI metadata: {row['locator']}")
        partitions, split_identity = e005_road_to_loki._partitions(reference, direct_predictions)
        development, assessment = partitions["development"], partitions["assessment"]
        reference_plan, selected_runs, selected_union = _reference_plan(
            reference, baselines["training"].entries, development, assessment)
        source_sha = checksum(source_path)
        run_plan = _plan(reference_plan, selected_runs, development, assessment, source_sha,
                         checksum(reference / "subset-plans.json"))
        run_plan["validation_split"] = split_identity
        write_json(output / "subset-plans.json", run_plan)

        baselines["training"].entries[:] = [entry for entry in baselines["training"].entries
                                             if entry["locator"] in selected_union]
        train_dataset = e005_road_to_loki.E005FeatureDataset(baselines["training"], spatial_cache, "g-rgb-utonia")
        validation_dataset = e005_road_to_loki.E005FeatureDataset(baselines["validation"], spatial_cache, "g-rgb-utonia")
        seed_info = seed_run(0)
        torch.set_num_threads(2)
        model = e005.build_model("g-rgb-utonia", config["profile"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.requires_grad_(False).eval()
        groups = {"training": {entry["locator"]: entry["group"] for entry in train_dataset.entries},
            "validation": group_by_locator}
        max_tracks = int(config.get("batching", {}).get("max_tracks", 2))
        extracted = {
            "training": _extract(model, train_dataset, groups["training"], max_tracks, device, "selected training union"),
            "validation": _extract(model, validation_dataset, groups["validation"], max_tracks, device, "validation"),
        }
        validation_anchor, replay_max_delta = e005_transfer_check(extracted["validation"], direct_predictions)
        training_by_locator = {row["locator"]: row for row in extracted["training"]}
        _verify_selected_labels(selected_runs, extracted["training"])
        model_head = checkpoint["model_state"]
        head = {"weight": model_head["head.weight"].detach().cpu().numpy().astype(np.float32, copy=True),
                "bias": model_head["head.bias"].detach().cpu().numpy().astype(np.float32, copy=True)}
        if head["weight"].shape != (4, 128) or head["bias"].shape != (4,):
            raise ValueError("ROAD E005 source classifier is not Linear(128, 4)")
        output_features = output / "embeddings"
        output_features.mkdir()
        for split, records in extracted.items():
            identity = {"experiment": "E005", "stage": "post-bilstm-final-layernorm", "source": "road-waymo",
                "target": "loki", "split": split, "checkpoint_sha256": source_sha,
                "spatial_cache_manifest_sha256": spatial_cache.manifest_sha256,
                "direct_validation_predictions_sha256": checksum(direct / "loki-validation/predictions.npz")
                    if split == "validation" else None}
            path = output_features / f"features-{split}.npz"
            e003b_run._save_cache(path, records, identity)
            extracted[split] = e003b_run._load_cache(path, identity)

        manifest = {"source_checkpoint_sha256": source_sha,
            "source_head": {"state_sha256": hashlib.sha256(head["weight"].tobytes() + head["bias"].tobytes()).hexdigest()},
            "normalization": checkpoint["normalization"],
            "derived_features": {split: {"sha256": checksum(output_features / f"features-{split}.npz")}
                                 for split in extracted}}
        write_json(output_features / "cache.json", {"experiment": "E005", "stage": "post-temporal head-budget features",
            "source_checkpoint_sha256": source_sha, "source_head": manifest["source_head"],
            "normalization": manifest["normalization"], "derived_features": manifest["derived_features"],
            "held_out_tests": False})
        cache_for_fits = {"root": output_features, "manifest": manifest, "head": head,
            "records": extracted, "source_validation": validation_anchor}
        zero = e004a_run._evaluate_zero(output, cache_for_fits, development, assessment,
                                        run_plan, True, experiment="E005")
        validation_by_locator = {row["locator"]: row for row in extracted["validation"]}
        development_features = [validation_by_locator[row["locator"]] for row in development]
        assessment_features = [validation_by_locator[row["locator"]] for row in assessment]
        results = [zero]
        for index, plan_row in enumerate(selected_runs, start=1):
            print(f"E005 ROAD→LOKI head fit {index}/30: {plan_row['budget']} tracks, draw {plan_row['draw']}", flush=True)
            selected = e003c_run._canonical_records(
                [training_by_locator[item["locator"]] for item in plan_row["selected_tracks"]])
            results.append(e004a_run._run_one(output, cache_for_fits, selected, development_features, assessment_features,
                plan_row, run_plan, device, False, True, experiment="E005",
                run_name_prefix="e005-road-to-loki-head", source="road-waymo", target="loki", wandb_enabled=False))
        write_json(output / "results.json", results)
        report = output / "report"
        e004a_analysis.make_report(results, report, all_tracks=None)
        write_json(output / "provenance.json", {"command": getattr(sys, "orig_argv", sys.argv),
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True).strip(),
            "source_run": str(source_run), "source_checkpoint": str(source_path), "source_checkpoint_sha256": source_sha,
            "source_run_revision": source_provenance["revision"], "source_normalization": checkpoint["normalization"],
            "loki_cache": str(spatial_cache.root), "loki_cache_manifest_sha256": spatial_cache.manifest_sha256,
            "e004a_reference": str(reference), "e004a_subset_plan_sha256": checksum(reference / "subset-plans.json"),
            "direct_predictions": str(direct / "loki-validation/predictions.npz"),
            "direct_predictions_sha256": checksum(direct / "loki-validation/predictions.npz"),
            "direct_replay_max_absolute_logit_delta": replay_max_delta,
            "extracted_tracks": {split: len(rows) for split, rows in extracted.items()},
            "selected_training_union": len(selected_union), "validation_split": split_identity,
            "device": str(device), "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "determinism": seed_info, "held_out_tests": False})
        status.update(status="complete", results=len(results), fit_count=len(selected_runs),
                      extracted_training_tracks=len(selected_union), extracted_validation_tracks=len(validation_anchor),
                      report=str(report), finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                                               "traceback": traceback.format_exc()})
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        status.setdefault("finished_utc", datetime.now(timezone.utc).isoformat())
        write_json(output / "status.json", status)
        spatial_cache.close()
    return {"attempt": str(output), "fits": status["fit_count"], "results": status["results"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--cache", type=Path, default=LOKI_CACHE)
    parser.add_argument("--reference", type=Path, default=E004A_REFERENCE)
    parser.add_argument("--direct", type=Path, default=DIRECT_TRANSFER)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
