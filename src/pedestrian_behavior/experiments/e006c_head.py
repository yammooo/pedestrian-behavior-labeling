"""Matched E006c head adaptation on the frozen Euro-PVI split."""

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import isolated_extraction
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import load_predictions
from pedestrian_behavior.experiments import e003b_run, e004a_run
from pedestrian_behavior.experiments.e005_cache import TinyUtoniaEncoder, collate_e005
from pedestrian_behavior.experiments.e005_transfer import (
    DEFAULT_PROBE, DEFAULT_TINY_CHECKPOINT, DEFAULT_UTONIA_CODE,
)
from pedestrian_behavior.experiments.e006b_euro_pvi import EuroPVIAdapter, sample_for
from pedestrian_behavior.inspection.pretrained import ImageEncoder
from pedestrian_behavior.models.e005 import build_model
from pedestrian_behavior.training import seed_run


AUDIT = Path("outputs/experiments/E006b/dataset-readiness-20261123/audit.json")
AUDIT_SHA256 = "6661e8c148b1db2a4ddbdbeff3929fac2693dcd2f318e8148e99f04dc9b7a85c"
DATASET = Path("/run/media/user20/T9/ped_intention_1000")
E006B_RUN = Path("outputs/experiments/E006b/runs/euro-pvi-assessment-20261123-113400")
E005_RUNS = {
    "road-waymo": "road-waymo-g-rgb-utonia-s0-20261120-030721",
    "joint": "joint-g-rgb-utonia-s0-20261122-001910",
}
CLASS_NAMES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
BUDGETS = (25, 50, 100, 250)
DRAWS = tuple(range(5))
FIT_GROUPS, DEV_GROUPS = 79, 40
EXPECTED = {
    "training": {"groups": 79, "sequences": 89, "tracks": 337, "gt_timesteps": 5306},
    "development": {"groups": 40, "sequences": 43, "tracks": 194, "gt_timesteps": 3110},
    "assessment": {"groups": 120, "sequences": 133, "tracks": 469, "gt_timesteps": 7190},
}
SMOKE_SEQUENCES = ("train/0067", "train/0158", "train/0339")


def _write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _canonical(value):
    return json.loads(json.dumps(value))


def _audit_class(name):
    return name.lower().replace("_", " ")


def _check_audit(path, dataset):
    if checksum(path) != AUDIT_SHA256:
        raise ValueError("E006c must use the frozen E006b audit")
    audit = json.loads(path.read_text())
    for name in ("calib.json", "labels/frames.csv"):
        if checksum(dataset / name) != audit["release_hashes"][name]:
            raise ValueError(f"Euro-PVI release hash mismatch: {name}")
    calibration = set(audit["partition"]["calibration"]["groups"])
    assessment = set(audit["partition"]["assessment"]["groups"])
    if calibration & assessment or len(calibration) != 132 or len(assessment) != 133:
        raise ValueError("Invalid frozen E006b sequence partition")
    capture_groups = audit["capture_groups"]
    cal_groups, assessment_groups = [], []
    seen = set()
    for group in capture_groups:
        group = tuple(group)
        overlap = set(group) & calibration
        if overlap and not set(group) <= calibration:
            raise ValueError("An audited timestamp group crosses the E006b calibration/assessment split")
        overlap = set(group) & assessment
        if overlap and not set(group) <= assessment:
            raise ValueError("An audited timestamp group crosses the E006b assessment/calibration split")
        if set(group) <= calibration:
            cal_groups.append(group)
        elif set(group) <= assessment:
            assessment_groups.append(group)
        if set(group) & seen:
            raise ValueError("Euro-PVI sequence occurs in more than one timestamp group")
        seen.update(group)
    if (seen != calibration | assessment or len(cal_groups) != 119 or len(assessment_groups) != 120
            or sum(map(len, cal_groups)) != 132 or sum(map(len, assessment_groups)) != 133):
        raise ValueError("Audited capture-group membership does not reconstruct E006b splits")

    order = np.random.default_rng(1).permutation(len(cal_groups))
    fit_groups = [cal_groups[int(i)] for i in order[:FIT_GROUPS]]
    dev_groups = [cal_groups[int(i)] for i in order[FIT_GROUPS:]]
    return audit, cal_groups, fit_groups, dev_groups, assessment_groups


def _load_models(outputs_root, e006b_run, device):
    summary_path = e006b_run / "summary.json"
    status_path = e006b_run / "status.json"
    if not summary_path.is_file() or json.loads(status_path.read_text()).get("status") != "complete":
        raise ValueError("A completed E006b assessment is required for the zero-anchor check")
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "complete" or summary.get("audit_sha256") != AUDIT_SHA256:
        raise ValueError("E006b assessment lineage does not match the frozen audit")

    models, evidence = {}, {}
    for source, run_name in E005_RUNS.items():
        run = outputs_root / "experiments/E005/runs" / run_name
        config = json.loads((run / "config.json").read_text())
        checkpoint_path = run / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        expected = summary["models"][source]
        if (checksum(checkpoint_path) != expected["checkpoint_sha256"]
                or config.get("source") != source
                or config.get("experiment") != "E005"
                or config.get("configuration") != "g-rgb-utonia"
                or config.get("variant") != "bilstm-96"
                or config.get("held_out_tests") is not False
                or config.get("smoke_limits") is not None
                or _canonical(checkpoint["config"]) != config
                or _canonical(checkpoint["normalization"]) != _canonical(expected["source_normalization"])):
            raise ValueError(f"Pinned E005 checkpoint changed: {source}")
        model = build_model(config["arm"], config["profile"]).to(device).eval()
        model.load_state_dict(checkpoint["model_state"], strict=True)
        models[source] = {"model": model, "config": config, "checkpoint": checkpoint,
                          "normalization": checkpoint["normalization"]}
        evidence[source] = {"run": run_name, "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": expected["checkpoint_sha256"], "selected_epoch": expected["selected_epoch"],
            "source_head_sha256": e004a_run._head_digest(
                checkpoint["model_state"]["head.weight"].cpu().numpy(),
                checkpoint["model_state"]["head.bias"].cpu().numpy()),
            "selection": expected["selection"]}

    # Match the exact two externally pretrained assets used by E006b.
    probe = json.loads((outputs_root / DEFAULT_PROBE / "probe.json").read_text())["config"]
    dino_sha = checksum(probe["dino_weights"])
    tiny_path = outputs_root / DEFAULT_TINY_CHECKPOINT
    tiny_sha = checksum(tiny_path)
    if ({"dino": dino_sha, "tiny_utonia": tiny_sha} != summary["encoder_sha256"]):
        raise ValueError("E006b frozen encoder assets differ from the completed assessment")
    for source in E005_RUNS:
        caches = models[source]["config"]["caches"].values() if source == "joint" else (
            models[source]["config"]["cache"],)
        for cache in caches:
            provenance = cache["provenance"]
            if (provenance["DINO"]["checkpoint_sha256"] != dino_sha
                    or provenance["Tiny-Utonia"]["checkpoint_sha256"] != tiny_sha):
                raise ValueError(f"E005 source run used a different frozen encoder: {source}")
    return models, evidence, probe, tiny_path, summary


def _capture(model, samples):
    batch = collate_e005(samples)
    captured = []
    hook = model.final_norm.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
    try:
        with torch.inference_mode():
            logits = model(batch)
        if len(captured) != 1 or captured[0].shape != (*logits.shape[:2], 128):
            raise ValueError("E005 pre-head embedding hook did not produce B×T×128")
        features = captured.pop()
    finally:
        hook.remove()
    rows = []
    for index, sample in enumerate(samples):
        length = len(sample["targets"])
        metadata = sample["prediction_metadata"]
        rows.append({
            "features": features[index, :length].detach().cpu().numpy().astype(np.float32, copy=True),
            "source_logits": logits[index, :length].detach().cpu().numpy().astype(np.float32, copy=True),
            "targets": sample["targets"].numpy().astype(np.int64, copy=True),
            "gt_valid": sample["gt_valid"].numpy().astype(bool, copy=True),
            "locator": sample["locator"], "archive": sample["archive"],
            "group": metadata["capture_group"], "sequence": metadata["sequence"],
            "frame_indices": list(metadata["frame_indices"]),
            "source_flags": {key: value.copy() for key, value in metadata["source_flags"].items()},
        })
    return rows


def _extract(args, split_groups, assessment_groups, models, probe, tiny_path, *, smoke=False):
    dataset = args.dataset.resolve()
    adapter = EuroPVIAdapter(dataset)
    sequence_to_group = {}
    for group in split_groups + assessment_groups:
        group_name = json.dumps(list(group), separators=(",", ":"))
        for sequence in group:
            if sequence in sequence_to_group:
                raise ValueError(f"Sequence occurs in multiple capture groups: {sequence}")
            sequence_to_group[sequence] = group_name
    split_by_sequence = {sequence: "calibration" for group in split_groups for sequence in group}
    split_by_sequence.update({sequence: "assessment" for group in assessment_groups for sequence in group})

    with isolated_extraction():
        dino = ImageEncoder(probe["dino_code"], probe["dino_weights"], args.device)
        tiny = TinyUtoniaEncoder(args.outputs_root / DEFAULT_UTONIA_CODE, tiny_path, args.device)
    results = {source: {"calibration": [], "assessment": []} for source in models}
    sequences = [s for group in split_groups for s in group] + [s for group in assessment_groups for s in group]
    if smoke:
        allowed = set(SMOKE_SEQUENCES)
        sequences = [sequence for sequence in sequences if sequence in allowed]
        if len(sequences) != len(allowed):
            raise ValueError("Smoke sequences are not present in audited non-assessment groups")

    for index, sequence_id in enumerate(sequences, 1):
        sequence = adapter.load_sequence(sequence_id)
        with isolated_extraction():
            records = adapter.encode_sequence(sequence, dino, tiny, args.device)
        part = split_by_sequence[sequence_id]
        for source, state in models.items():
            samples = []
            for record in records:
                sample = sample_for(record, state["normalization"])
                sample["prediction_metadata"] = sample["prediction_metadata"] | {
                    "sequence": sequence_id, "capture_group": sequence_to_group[sequence_id]}
                samples.append(sample)
            for start in range(0, len(samples), 2):
                rows = _capture(state["model"], samples[start:start + 2])
                results[source][part].extend(rows)
        if index == 1 or index % 10 == 0 or index == len(sequences):
            print(f"Encoded {index}/{len(sequences)} sequences: {sequence_id}", flush=True)
        del sequence, records

    return results, tiny.calls


def _metric_tracks(rows):
    return [{"locator": row["locator"], "archive": row["archive"], "group": row["group"],
             "sequence": row["sequence"], "frame_indices": row["frame_indices"],
             "logits": row["source_logits"], "targets": row["targets"], "gt_valid": row["gt_valid"],
             "source_flags": row["source_flags"], "conditions": {}} for row in rows]


def _check_e006b_replay(assessment, source, e006b_run):
    baseline_path = e006b_run / f"{source}-predictions.npz"
    baseline = load_predictions(baseline_path)
    current = {row["locator"]: row for row in assessment}
    if len(current) != 469 or set(current) != {row["locator"] for row in baseline}:
        raise ValueError(f"E006c assessment tracks differ from E006b {source} predictions")
    maximum = 0.
    for row in baseline:
        found = current[row["locator"]]
        if (found["sequence"] != row["group"]
                or found["frame_indices"] != row["frame_indices"]
                or not np.array_equal(found["targets"], row["targets"])
                or not np.array_equal(found["gt_valid"], row["gt_valid"])):
            raise ValueError(f"E006c assessment labels/mask differ from E006b for {row['locator']}")
        for flag in ("rgb_valid", "lidar_valid"):
            if not np.array_equal(found["source_flags"][flag], row["source_flags"][flag]):
                raise ValueError(f"E006c assessment {flag} differs from E006b")
        error = np.max(np.abs(found["source_logits"] - row["logits"]))
        maximum = max(maximum, float(error))
        if not np.allclose(found["source_logits"], row["logits"], rtol=1e-5, atol=2e-5):
            raise ValueError(f"E006c source logits do not reproduce E006b for {row['locator']} (max {error})")
    return {"tracks": len(current), "max_abs_logit_difference": maximum, "reference": str(baseline_path)}


def _save_features(output, source, split, records, audit_sha, checkpoint_sha):
    path = output / "features" / source / f"{split}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {"experiment": "E006c", "source": source, "split": split,
                "checkpoint_sha256": checkpoint_sha, "audit_sha256": audit_sha,
                "representation": "post-temporal-final-layernorm-128d",
                "normalization": "checkpoint source statistics", "dtype": "float32"}
    e003b_run._save_cache(path, records, identity)
    loaded = e003b_run._load_cache(path, identity)
    if len(loaded) != len(records) or not np.array_equal(loaded[0]["features"], records[0]["features"]):
        raise ValueError("E006c compact feature cache roundtrip failed")
    return {"path": str(path), "sha256": checksum(path), "tracks": len(records),
            "gt_timesteps": int(sum(int(row["gt_valid"].sum()) for row in records))}


def _split_records(results, fit_sequences, dev_sequences, assessment_sequences):
    parts = {"training": set(fit_sequences), "development": set(dev_sequences),
             "assessment": set(assessment_sequences)}
    output = {}
    for source, splits in results.items():
        all_rows = splits["calibration"] + splits["assessment"]
        rows = {name: [] for name in parts}
        for row in all_rows:
            sequence = row["sequence"]
            destinations = [name for name, members in parts.items() if sequence in members]
            if len(destinations) != 1:
                raise ValueError(f"Track sequence has no unique E006c split: {sequence}")
            rows[destinations[0]].append(row)
        output[source] = rows
    return output


def _summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source"], row["budget"]].append(row)
    output = []
    for (source, budget), group in sorted(grouped.items()):
        target = [next(item for item in row["coverage"]["targets"] if item["error_target"] == .05)
                  for row in group]
        f1 = [row["macro_f1"] for row in group]
        error = [item["assessment"]["error"] for item in target if item["assessment"]["error"] is not None]
        coverage = [item["assessment"]["coverage"] for item in target]
        class_retention = {CLASS_NAMES[i]: [item["assessment"]["classes"][i]["gt_class_retention"]
                                                  for item in target]
                           for i in range(4)}
        output.append({"source": source, "budget": budget, "draws": [row["draw"] for row in group],
            "n": len(group), "assessment_macro_f1_mean": float(np.mean(f1)),
            "assessment_macro_f1_sd": float(np.std(f1, ddof=1)) if len(f1) > 1 else None,
            "assessment_track_ce_mean": float(np.mean([row["track_ce"] for row in group])),
            "coverage_5pct_mean": float(np.mean(coverage)),
            "coverage_5pct_sd": float(np.std(coverage, ddof=1)) if len(coverage) > 1 else None,
            "achieved_error_5pct_mean": float(np.mean(error)) if error else None,
            "achieved_error_5pct_sd": float(np.std(error, ddof=1)) if len(error) > 1 else None,
            "assessment_error_overruns": sum(value > .05 for value in error),
            "gt_class_retention_5pct": {
                name: {"mean": float(np.mean(values)),
                       "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
                for name, values in class_retention.items()},
        })
    return output


def _paired(rows):
    by_key = {(row["source"], row["budget"], row["draw"]): row for row in rows}
    def target(row):
        return next(item for item in row["coverage"]["targets"] if item["error_target"] == .05)
    def difference(first, second, key):
        a, b = target(first)["assessment"].get(key), target(second)["assessment"].get(key)
        return None if a is None or b is None else b - a

    comparisons = []
    for budget in (0, *BUDGETS):
        draws = (None,) if budget == 0 else DRAWS
        for draw in draws:
            road = by_key["road-waymo", budget, draw]
            joint = by_key["joint", budget, draw]
            comparisons.append({"budget": budget, "draw": draw,
                "joint_minus_road_macro_f1": joint["macro_f1"] - road["macro_f1"],
                "joint_minus_road_track_ce": joint["track_ce"] - road["track_ce"],
                "joint_minus_road_coverage_5pct": difference(road, joint, "coverage"),
                "joint_minus_road_error_5pct": difference(road, joint, "error"),
            })
    return comparisons


def _paired_summary(rows):
    comparisons = _paired(rows)
    result = []
    for budget in (0, *BUDGETS):
        values = [row for row in comparisons if row["budget"] == budget]
        metrics = {}
        for name in ("macro_f1", "track_ce", "coverage_5pct", "error_5pct"):
            observed = [row[f"joint_minus_road_{name}"] for row in values
                        if row[f"joint_minus_road_{name}"] is not None]
            metrics[name] = {"mean": float(np.mean(observed)) if observed else None,
                "sd": float(np.std(observed, ddof=1)) if len(observed) > 1 else None,
                "n": len(observed)}
        result.append({"budget": budget, "paired_joint_minus_road": metrics})
    return result


def _plots(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    colors = {"road-waymo": "#4477AA", "joint": "#228833"}
    for name, metric, ylabel in (
        ("budget-f1", "assessment_macro_f1", "Track-weighted macro-F1 (%)"),
        ("budget-coverage-5pct", "coverage_5pct", "Track-weighted coverage at 5% calibration error (%)"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.set_xscale("symlog", linthresh=1)
        for source in E005_RUNS:
            points = [row for row in summaries if row["source"] == source]
            xs = [row["budget"] for row in points]
            means = [row[metric + "_mean"] * 100 for row in points]
            deviations = [0 if row[metric + "_sd"] is None else row[metric + "_sd"] * 100 for row in points]
            ax.errorbar(xs, means, yerr=deviations, fmt="-o", capsize=3,
                        label=source, color=colors[source])
            if metric == "coverage_5pct":
                overrun = [row["assessment_error_overruns"] > 0 for row in points]
                ax.scatter(xs, means, facecolors=["none" if value else colors[source] for value in overrun],
                           edgecolors=colors[source], s=42, linewidths=1.3, zorder=4)
        ax.set_xticks((0, *BUDGETS))
        ax.set_xticklabels(("0", *[str(value) for value in BUDGETS]))
        ax.set(xlabel="Labeled Euro-PVI fitting tracks", ylabel=ylabel, ylim=(0, 100))
        ax.set_title("E006c: " + ("full-coverage assessment F1" if metric == "assessment_macro_f1"
                                  else "automatic coverage, 5% calibration target"))
        ax.legend(frameon=False)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"{name}.{suffix}", dpi=160)
        plt.close(fig)


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Use a fresh E006c output directory: {output}")
    output.mkdir(parents=True)
    status = {"status": "incomplete", "experiment": "E006c", "held_out_tests": False}
    e004a_run.shared_run.write_json(output / "status.json", status)
    started = time.monotonic()
    provenance = {"command": getattr(sys, "orig_argv", sys.argv),
                  "started_utc": datetime.now(timezone.utc).isoformat(),
                  "dataset": str(args.dataset.resolve()), "audit": str(args.audit.resolve()),
                  "e006b_run": str(args.e006b_run.resolve()), "device": args.device,
                  "fit_device": args.fit_device, "held_out_tests": False}
    try:
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("E006c frozen E005 encoder extraction requires CUDA")
        if args.fit_device not in ("cpu", "cuda") or (args.fit_device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("E006c head-fitting device is unavailable")
        torch.set_num_threads(2)
        determinism = seed_run(0)
        torch.cuda.reset_peak_memory_stats(args.device)
        audit, cal_groups, fit_groups, dev_groups, assessment_groups = _check_audit(args.audit, args.dataset)
        models, model_evidence, probe, tiny_path, e006b = _load_models(
            args.outputs_root, args.e006b_run, args.device)
        if args.smoke:
            encoded, tiny_sweeps = _extract(args, cal_groups, assessment_groups, models, probe, tiny_path,
                                            smoke=True)
        else:
            encoded, tiny_sweeps = _extract(args, cal_groups, assessment_groups, models, probe, tiny_path)

        if args.smoke:
            sequences = list(SMOKE_SEQUENCES)
            fit_groups = [group for group in cal_groups if sequences[0] in group]
            dev_groups = [group for group in cal_groups if sequences[1] in group]
            assessment_groups = [group for group in cal_groups if sequences[2] in group]
            if any(len(group) != 1 for group in fit_groups + dev_groups + assessment_groups):
                raise ValueError("Smoke sequences must be separate capture groups")
            budgets, draws = (1,), (0,)
        else:
            budgets, draws = BUDGETS, DRAWS
        split_results = _split_records(encoded,
            [sequence for group in fit_groups for sequence in group],
            [sequence for group in dev_groups for sequence in group],
            [sequence for group in assessment_groups for sequence in group])

        summaries = {}
        for source, rows in split_results.items():
            if not args.smoke:
                for key, expected in EXPECTED.items():
                    found = e004a_run._population_summary(rows[key])
                    if found["tracks"] != expected["tracks"] or found["gt_timesteps"] != expected["gt_timesteps"]:
                        raise ValueError(f"E006c {source} {key} split differs from the frozen plan: {found}")
                    if len({row["sequence"] for row in rows[key]}) != expected["sequences"]:
                        raise ValueError(f"E006c {source} {key} sequence count differs from frozen plan")
                    if found["groups"] != expected["groups"]:
                        raise ValueError(f"E006c {source} {key} capture-group count differs from frozen plan")
                counts = e004a_run._population_summary(rows["assessment"])["class_gt_counts"]
                classes = dict(zip(CLASS_NAMES, counts, strict=True))
                expected_classes = audit["partition"]["assessment"]["nominal_5Hz_class_frames"]
                if classes != {name: expected_classes[_audit_class(name)] for name in CLASS_NAMES}:
                    raise ValueError(f"E006c {source} assessment class support differs from audit")
                fit_dev = e004a_run._population_summary(rows["training"] + rows["development"])["class_gt_counts"]
                calibration_counts = {name: count for name, count in
                    zip(CLASS_NAMES, fit_dev, strict=True)}
                expected_calibration = audit["partition"]["calibration"]["nominal_5Hz_class_frames"]
                if calibration_counts != {name: expected_calibration[_audit_class(name)] for name in CLASS_NAMES}:
                    raise ValueError("E006c fit/development class support differs from calibration audit")

            for name, key in (("training", "training"), ("development", "development"),
                              ("assessment", "assessment")):
                summaries.setdefault(source, {})[name] = _save_features(
                    output, source, key, rows[key], AUDIT_SHA256,
                    model_evidence[source]["checkpoint_sha256"])
            if not args.smoke:
                summaries[source]["e006b_replay"] = _check_e006b_replay(rows["assessment"], source, args.e006b_run)

        provenance["representation_caches"] = summaries
        provenance["frozen_assessment_replay"] = {
            source: value["e006b_replay"] for source, value in summaries.items()
            if "e006b_replay" in value}

        # Drop the large frozen encoders before the small, repeated head fits.
        del models
        if args.device == "cuda":
            torch.cuda.empty_cache()

        all_results, pair_plan = [], {source: {} for source in split_results}
        plan = {"status": "planned", "experiment": "E006c", "draw_seed": 1,
            "fit_groups": [list(group) for group in fit_groups],
            "development_groups": [list(group) for group in dev_groups],
            "assessment_groups": [list(group) for group in assessment_groups],
            "population": {name: e004a_run._population_summary(split_results["road-waymo"][name])
                           for name in ("training", "development", "assessment")},
            "validation_split": {"development_groups": sorted({row["group"]
                                                                  for row in split_results["road-waymo"]["development"]}),
                                 "assessment_groups": sorted({row["group"]
                                                              for row in split_results["road-waymo"]["assessment"]})},
            "protocol": {"budget_tracks": list(budgets), "draws": list(draws),
                         "fit_seed": 0, "error_target": .05,
                         "fit_parameter_count": 516, "fit_optimizer": e004a_run.SETTINGS["optimizer"],
                         "check_updates": [0, *e004a_run._check_schedule(True, e004a_run.SETTINGS["max_updates"])],
                         "patience_updates": e004a_run.EARLY_PATIENCE_UPDATES},
            "runs": []}
        for draw in draws:
            order = e004a_run.recording_round_robin(split_results["road-waymo"]["training"], draw)
            for budget in budgets:
                selected = order[:budget]
                plan["runs"].append({"budget": budget, "draw": draw,
                    "selected_tracks": [e004a_run._track_label_record(row) for row in selected]})
        pair_plan_rows = {(row["budget"], row["draw"]): row for row in plan["runs"]}
        plan_path = output / "subset-plans.json"
        e004a_run.shared_run.write_json(plan_path, plan)

        for source, rows in split_results.items():
            fit_pool = rows["training"]
            dev, assessment_rows = rows["development"], rows["assessment"]
            source_attempt = output / source
            source_attempt.mkdir()
            e004a_run.shared_run.write_json(source_attempt / "subset-plans.json", plan)
            # Each source's original Linear(128,4) initializes every independent adaptation.
            model_record = torch.load(model_evidence[source]["checkpoint"], map_location="cpu", weights_only=True)
            head = {"weight": model_record["model_state"]["head.weight"].numpy().copy(),
                    "bias": model_record["model_state"]["head.bias"].numpy().copy()}
            head_sha = e004a_run._head_digest(head["weight"], head["bias"])
            manifest = {"source_checkpoint_sha256": model_evidence[source]["checkpoint_sha256"],
                "normalization": _canonical(model_record["normalization"]),
                "source_head": {"state_sha256": head_sha},
                "derived_features": {"training": {"sha256": summaries[source]["training"]["sha256"]},
                                     "validation": {"sha256": summaries[source]["development"]["sha256"]}}}
            source_cache_root = output / "features" / source
            e004a_run.shared_run.write_json(source_cache_root / "cache.json", manifest)
            cache = {"root": source_cache_root, "manifest": manifest, "head": head,
                     "source_validation": _metric_tracks(dev + assessment_rows)}
            zero = e004a_run._evaluate_zero(source_attempt, cache, dev, assessment_rows, plan,
                                            True, "E006c")
            zero["source"] = source
            all_results.append(zero)
            for budget, draw in ((row["budget"], row["draw"]) for row in plan["runs"]):
                plan_row = pair_plan_rows[budget, draw]
                selected_ids = {entry["locator"] for entry in plan_row["selected_tracks"]}
                fit_rows = sorted((row for row in fit_pool if row["locator"] in selected_ids),
                                  key=lambda row: row["locator"])
                if len(fit_rows) != budget:
                    raise ValueError("E006c saved nested subset does not match cached fit tracks")
                result = e004a_run._run_one(
                    source_attempt, cache, fit_rows, dev, assessment_rows, plan_row, plan,
                    torch.device(args.fit_device), args.smoke, True, experiment="E006c",
                    run_name_prefix=f"e006c-{source}-head", source=source, target="euro-pvi",
                    wandb_enabled=False)
                result["source"] = source
                all_results.append(result)
                pair_plan[source][budget, draw] = sorted(selected_ids)
                print(f"E006c {source} fit {budget} tracks, draw {draw}: "
                      f"assessment F1={result['macro_f1']:.4f}", flush=True)

        if not args.smoke:
            for budget in BUDGETS:
                for draw in DRAWS:
                    if pair_plan["road-waymo"][budget, draw] != pair_plan["joint"][budget, draw]:
                        raise ValueError("Paired E006c arm subsets differ")
            _plots(output / "report", _summary(all_results))
        report = {"status": "complete", "experiment": "E006c",
            "kind": "smoke" if args.smoke else "matched-head-adaptation",
            "split": {"fit_groups": [list(group) for group in fit_groups],
                      "development_groups": [list(group) for group in dev_groups],
                      "assessment_groups": [list(group) for group in assessment_groups]},
            "class_names": list(CLASS_NAMES), "budgets": list(budgets), "draws": list(draws),
            "fit_count": sum(row["budget"] > 0 for row in all_results), "results": all_results,
            "summary": _summary(all_results) if not args.smoke else [],
            "paired_differences": _paired(all_results) if not args.smoke else [],
            "paired_summary": _paired_summary(all_results) if not args.smoke else [],
            "representation_caches": summaries,
            "encoders": {"dino_sha256": e006b["encoder_sha256"]["dino"],
                         "tiny_utonia_sha256": e006b["encoder_sha256"]["tiny_utonia"],
                         "tiny_sweeps": tiny_sweeps},
            "source_models": model_evidence, "audit_sha256": AUDIT_SHA256,
            "assessment_used_for_selection_or_cutoffs": False,
        }
        _write(output / "summary.json", report)
        provenance.update({"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started,
            "determinism": determinism, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device)})
        _write(output / "provenance.json", provenance)
        status.update(status="complete", fit_count=report["fit_count"], results=len(all_results),
                      report=str(output / "summary.json"))
        e004a_run.shared_run.write_json(output / "status.json", status)
        print(f"E006c complete: {output} ({report['fit_count']} fits)", flush=True)
        return report
    except BaseException as error:
        status.update(status="failed", failure={"type": type(error).__name__, "message": str(error),
                                                  "traceback": traceback.format_exc()})
        e004a_run.shared_run.write_json(output / "status.json", status)
        provenance.update(failure={"type": type(error).__name__, "message": str(error)},
                          elapsed_seconds=time.monotonic() - started)
        _write(output / "provenance.json", provenance)
        raise
