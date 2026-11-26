#!/usr/bin/env python3
"""Adapt the E005 LOKI LiDAR readout and classifier on Euro-PVI.

The frozen RGB/geometry branches, fusion layer and temporal block stay in the
forward graph. Only ``model.lidar``, ``model.lidar_norm`` and ``model.head``
receive gradients.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import load_predictions, prediction_metrics, save_predictions, track_cross_entropy
from pedestrian_behavior.experiments import e004a_analysis, e004a_run
from pedestrian_behavior.experiments.e005_cache import atomic_json, collate_e005
from pedestrian_behavior.experiments.e006b_euro_pvi import sample_for
from pedestrian_behavior.experiments.e006c_head import (
    AUDIT, DATASET, DEFAULT_SUBSET_PLAN, E006B_RUN, BUDGETS, DRAWS,
    CLASS_NAMES, _check_audit, _load_models, _load_saved_subset_plan,
)
from pedestrian_behavior.experiments.e005_transfer import DEFAULT_UTONIA_CODE
from pedestrian_behavior.models.e005 import build_model
from pedestrian_behavior.training import seed_run
from torch.utils.data import DataLoader


HEAD_ONLY_RUN = Path("outputs/experiments/E006c/runs/euro-pvi-loki-head-adaptation-20261123-143333")
MICROBATCH = 2
EFFECTIVE_BATCH = 64
MAX_UPDATES = 10000
PATIENCE_UPDATES = 1500
CLIP_NORM = 1.0
OPTIMIZER = {"lr": 3e-4, "weight_decay": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8}


def _records_for(args, groups, probe, tiny_path, normalization):
    """Encode each selected sequence once and retain CPU token samples."""
    from pedestrian_behavior.data.feature_cache import isolated_extraction
    from pedestrian_behavior.experiments.e006b_euro_pvi import EuroPVIAdapter
    from pedestrian_behavior.inspection.pretrained import ImageEncoder
    from pedestrian_behavior.experiments.e005_cache import TinyUtoniaEncoder

    adapter = EuroPVIAdapter(args.dataset.resolve())
    capture_groups = {sequence: json.dumps(list(group), separators=(",", ":"))
                      for group in groups for sequence in group}
    output = []
    with isolated_extraction():
        dino = ImageEncoder(probe["dino_code"], probe["dino_weights"], args.device)
        tiny = TinyUtoniaEncoder(args.outputs_root / DEFAULT_UTONIA_CODE, tiny_path, args.device)
        for index, sequence_id in enumerate(capture_groups, 1):
            sequence = adapter.load_sequence(sequence_id)
            records = adapter.encode_sequence(sequence, dino, tiny, args.device)
            for record in records:
                sample = sample_for(record, normalization)
                sample["prediction_metadata"] = sample["prediction_metadata"] | {
                    "sequence": sequence_id, "capture_group": capture_groups[sequence_id]}
                sample["group"] = capture_groups[sequence_id]
                sample["sequence"] = sequence_id
                output.append(sample)
            if index == 1 or index % 10 == 0 or index == len(sequences):
                print(f"Encoded {index}/{len(capture_groups)} sequences: {sequence_id}", flush=True)
            del sequence, records
    if not output:
        raise ValueError("Euro-PVI extraction produced no tracks")
    return output


def _batches(samples):
    for start in range(0, len(samples), MICROBATCH):
        yield collate_e005(samples[start:start + MICROBATCH])


def _prediction_rows(model, samples, device):
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in _batches(samples):
            logits = model(batch, mode="ordinary")
            targets = batch["targets"]
            valid = batch["gt_valid"]
            padding = batch["padding_mask"]
            track_cross_entropy(logits, targets.to(device), valid.to(device), padding.to(device))
            for index, length in enumerate(batch["lengths"].tolist()):
                metadata = batch["prediction_metadata"][index]
                rows.append({
                    "logits": logits[index, :length].cpu().numpy().astype(np.float32),
                    "targets": targets[index, :length].numpy().astype(np.int64),
                    "gt_valid": valid[index, :length].numpy().astype(bool),
                    "locator": batch["locators"][index], "archive": batch["archives"][index],
                    "group": metadata["capture_group"], "sequence": metadata["sequence"],
                    "frame_indices": list(metadata["frame_indices"]),
                    "source_flags": {key: value[:length].copy()
                                     for key, value in metadata["source_flags"].items()},
                    "conditions": {},
                })
    return rows


def _set_trainable(model):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.lidar.parameters():
        parameter.requires_grad_(True)
    for parameter in model.lidar_norm.parameters():
        parameter.requires_grad_(True)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    model.eval()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or sum(parameter.numel() for parameter in trainable) <= 516:
        raise ValueError("LiDAR adaptation did not enable spatial parameters")
    return trainable


def _update(model, samples, optimizer, trainable, device):
    # The single-layer cuDNN LSTM requires this mode for input gradients.
    model.temporal.lstm.train()
    # Enable spatial activation recomputation while leaving dropout disabled.
    model.training = True
    optimizer.zero_grad(set_to_none=True)
    for batch in _batches(samples):
        logits = model(batch)
        losses = track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device),
                                     batch["padding_mask"].to(device))
        (losses.sum() / len(samples)).backward()
    torch.nn.utils.clip_grad_norm_(trainable, CLIP_NORM, error_if_nonfinite=True)
    optimizer.step()


def _training_batches(samples):
    loader = DataLoader(samples, batch_size=EFFECTIVE_BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(0), collate_fn=list)
    while True:
        yield from loader


def _assessment_pair(current, baseline):
    by_id = {row["locator"]: row for row in current}
    if len(by_id) != len(current) or len(current) != len(baseline):
        raise ValueError("Assessment tracks differ from E006c head-only")
    for old in baseline:
        row = by_id[old["locator"]]
        if any(not np.array_equal(row[key], old[key]) for key in
               ("frame_indices", "targets", "gt_valid")):
            raise ValueError("Assessment frames differ from E006c head-only")

    def accepted_errors(rows):
        data = e004a_analysis._population(rows)
        curve = e004a_analysis._curve(data)
        cutoff = curve["threshold"][np.searchsorted(curve["coverage"], .30)]
        accepted = data["scores"] >= cutoff
        wrong_crossing = accepted & (data["targets"] == 0) & (data["predicted"] == 3)
        mass = data["weights"][accepted].sum()
        return {"coverage": float(mass / data["weights"].sum()),
                "error": float(data["weights"][accepted & (data["targets"] != data["predicted"])].sum() / mass),
                "moving_as_crossing_frames": int(wrong_crossing.sum()),
                "moving_as_crossing_error": float(data["weights"][wrong_crossing].sum() / mass)}

    return {"head_only": accepted_errors(baseline), "lidar_plus_head": accepted_errors(current)}


def _fit_one(args, source_checkpoint, split, plan_row, head_only):
    budget, draw = plan_row["budget"], plan_row["draw"]
    selected = {entry["locator"] for entry in plan_row["selected_tracks"]}
    fit_rows = [row for row in split["training"] if row["locator"] in selected]
    if len(fit_rows) != budget:
        raise ValueError(f"Subset has {len(fit_rows)} tracks, expected {budget}")
    fit_dir = args.output / f"budget-{budget:04d}-draw-{draw:02d}"
    fit_dir.mkdir()
    seed_run(0)
    model = build_model(source_checkpoint["config"]["arm"], source_checkpoint["config"]["profile"]).to(args.device)
    model.load_state_dict(source_checkpoint["model_state"], strict=True)
    trainable = _set_trainable(model)
    optimizer = torch.optim.AdamW(trainable, **OPTIMIZER)
    history, best, last_improvement = [], None, 0
    update_seconds = validation_seconds = 0.
    started = time.monotonic()

    def check(step):
        nonlocal best, last_improvement, validation_seconds
        begin = time.monotonic()
        rows = _prediction_rows(model, split["development"], args.device)
        metrics = prediction_metrics(rows, len(CLASS_NAMES))
        validation_seconds += time.monotonic() - begin
        f1, ce = metrics["track_weighted"]["macro_f1"], metrics["track_ce"]
        strict_f1 = best is None or f1 > best["f1"]
        if strict_f1:
            last_improvement = step
        selected = strict_f1 or (f1 == best["f1"] and ce < best["ce"])
        if selected:
            best = {"f1": f1, "ce": ce, "optimizer_step": step}
            pending = fit_dir / "best.pt.tmp"
            torch.save({"config": source_checkpoint["config"],
                        "normalization": source_checkpoint["normalization"],
                        "model_state": model.state_dict(), "selection": best}, pending)
            pending.replace(fit_dir / "best.pt")
        history.append({"optimizer_step": step, "development_f1": f1,
                        "development_ce": ce, "selected": selected,
                        "seconds_per_update": update_seconds / step if step else None})
        atomic_json(fit_dir / "history.json", history)
        print(f"n={budget} draw={draw} update={step} dev_f1={f1:.4f} dev_ce={ce:.4f} "
              f"train_sec={update_seconds:.2f} val_sec={validation_seconds:.2f}", flush=True)

    schedule = set(e004a_run._check_schedule(True, args.max_updates)) | {args.max_updates}
    check(0)
    batches = _training_batches(fit_rows)
    stopping_reason = "update_cap"
    for step in range(1, args.max_updates + 1):
        begin = time.monotonic()
        _update(model, next(batches), optimizer, trainable, args.device)
        torch.cuda.synchronize(args.device)
        update_seconds += time.monotonic() - begin
        if step in schedule:
            check(step)
            if step - last_improvement >= args.patience_updates:
                stopping_reason = "patience"
                break

    checkpoint = torch.load(fit_dir / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    dev_rows = _prediction_rows(model, split["development"], args.device)
    assessment_rows = _prediction_rows(model, split["assessment"], args.device)
    save_predictions(fit_dir / "development-predictions.npz", dev_rows)
    save_predictions(fit_dir / "assessment-predictions.npz", assessment_rows)
    coverage = e004a_analysis.analyze_checkpoint(dev_rows, assessment_rows, fit_dir / "risk-analysis")
    head_dir = head_only / "confidence-inputs" / "loki" / fit_dir.name
    paired = _assessment_pair(assessment_rows, load_predictions(head_dir / "assessment-predictions.npz"))
    result = {"budget": budget, "draw": draw, "selected_update": best["optimizer_step"],
              "actual_optimizer_updates": step, "stopping_reason": stopping_reason,
              "seconds_per_update": update_seconds / step, "validation_seconds": validation_seconds,
              "trainable_parameters": sum(parameter.numel() for parameter in trainable),
              "assessment_metrics": prediction_metrics(assessment_rows, len(CLASS_NAMES)),
              "coverage": coverage, "matched_30pct_coverage": paired,
              "elapsed_seconds": time.monotonic() - started, "fit_dir": str(fit_dir)}
    atomic_json(fit_dir / "selected-metrics.json", result)
    return result


def run(args):
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E006c LiDAR adaptation requires CUDA")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    args.output.mkdir(parents=True)
    started = time.monotonic()
    atomic_json(args.output / "status.json", {"status": "running", "experiment": "E006c-lidar-adaptation"})
    try:
        torch.set_num_threads(2)
        seed_run(0)
        _, _, fit_groups, dev_groups, assessment_groups = _check_audit(args.audit, args.dataset)
        models, evidence, probe, tiny_path, _ = _load_models(args.outputs_root, args.e006b_run, args.device, ("loki",))
        model_record = models["loki"]["checkpoint"]
        head_only = args.outputs_root / HEAD_ONLY_RUN.relative_to("outputs")
        baseline = json.loads((head_only / "summary.json").read_text())
        plan_sha = checksum(args.subset_plan)
        if (baseline["status"] != "complete" or baseline["fit_count"] != 20
                or baseline["source_models"]["loki"]["checkpoint_sha256"]
                != evidence["loki"]["checkpoint_sha256"]
                or plan_sha != checksum(head_only / "subset-plans.json")):
            raise ValueError("E006c head-only checkpoint or subset plan mismatch")
        samples = _records_for(args, fit_groups + dev_groups + assessment_groups, probe, tiny_path,
                               models["loki"]["normalization"])
        extraction_seconds = time.monotonic() - started
        by_split = {"training": [], "development": [], "assessment": []}
        memberships = {sequence: "training" for group in fit_groups for sequence in group}
        memberships.update({sequence: "development" for group in dev_groups for sequence in group})
        memberships.update({sequence: "assessment" for group in assessment_groups for sequence in group})
        for sample in samples:
            by_split[memberships[sample["sequence"]]].append(sample)
        plan = _load_saved_subset_plan(args.subset_plan, by_split["training"], fit_groups, dev_groups, assessment_groups)
        rows = plan["runs"]
        if args.budget is not None:
            rows = [row for row in rows if int(row["budget"]) == args.budget]
        if args.draw is not None:
            rows = [row for row in rows if int(row["draw"]) == args.draw]
        if not rows:
            raise ValueError("No subset rows selected")
        results = []
        for row in rows:
            result = _fit_one(args, model_record, by_split, row, head_only)
            reference = next(old for old in baseline["results"]
                             if (old["budget"], old["draw"]) == (row["budget"], row["draw"]))
            result["head_only"] = {key: reference[key] for key in ("macro_f1", "coverage", "selected_update")}
            results.append(result)
        summary = {"status": "complete", "experiment": "E006c-lidar-adaptation",
                   "budgets": sorted({int(row["budget"]) for row in rows}),
                   "draws": sorted({int(row["draw"]) for row in rows}),
                   "results": results, "source_model": evidence["loki"],
                   "audit_sha256": checksum(args.audit), "subset_plan_sha256": plan_sha,
                   "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                   "extraction_seconds": extraction_seconds, "elapsed_seconds": time.monotonic() - started,
                   "max_updates": args.max_updates, "patience_updates": args.patience_updates,
                   "microbatch": MICROBATCH, "effective_batch": EFFECTIVE_BATCH,
                   "optimizer": OPTIMIZER, "dropout": "disabled", "gradient_clip": CLIP_NORM,
                   "assessment_used_for_selection_or_cutoffs": False}
        atomic_json(args.output / "summary.json", summary)
        atomic_json(args.output / "status.json", {"status": "complete", "fits": len(results),
                                              "summary": str(args.output / "summary.json")})
        print(f"Complete: {len(results)} fits in {time.monotonic() - started:.1f}s", flush=True)
    except BaseException as error:
        atomic_json(args.output / "status.json", {"status": "failed", "error": type(error).__name__ + ": " + str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--e006b-run", type=Path, default=E006B_RUN)
    parser.add_argument("--subset-plan", type=Path, default=DEFAULT_SUBSET_PLAN)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    parser.add_argument("--draw", type=int, choices=DRAWS)
    parser.add_argument("--max-updates", type=int, default=MAX_UPDATES)
    parser.add_argument("--patience-updates", type=int, default=PATIENCE_UPDATES)
    args = parser.parse_args()
    if args.max_updates < 1 or args.patience_updates < 1:
        parser.error("update limits must be positive")
    run(args)


if __name__ == "__main__":
    main()
