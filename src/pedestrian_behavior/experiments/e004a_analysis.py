"""Selective coverage analysis and compact reporting for E004a."""

import csv
import json
import math
from pathlib import Path

import numpy as np


CLASS_NAMES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
ERROR_TARGETS = (0.01, 0.05, 0.10, 0.20)
ALL_TRACKS = 8574


def _population(tracks):
    if not tracks:
        raise ValueError("E004a analysis requires nonempty tracks")
    if any(not isinstance(row.get("source_flags", {}), dict) for row in tracks):
        raise ValueError("source_flags must be a mapping")
    seen = set()
    arrays = {key: [] for key in ("scores", "predicted", "targets", "weights", "track", "group_ids")}
    flags = {}
    flag_names = {name for row in tracks for name in row.get("source_flags", {})
                  if isinstance(name, str) and name.endswith("_valid")}
    group_ids = {}
    for index, row in enumerate(tracks):
        locator, group = row.get("locator"), row.get("group")
        if not isinstance(locator, str) or not locator or locator in seen:
            raise ValueError("Each track needs a unique nonempty locator")
        if not isinstance(group, str) or not group:
            raise ValueError("Each track needs a nonempty group")
        seen.add(locator)
        logits = np.asarray(row["logits"], dtype=np.float64)
        targets = np.asarray(row["targets"])
        gt = np.asarray(row["gt_valid"], dtype=bool)
        if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES) or targets.ndim != 1 or gt.shape != targets.shape or logits.shape[0] != len(targets):
            raise ValueError("Invalid E004a prediction shapes")
        if not np.isfinite(logits).all() or targets.dtype.kind not in "iu":
            raise ValueError("Predictions must be finite and targets integral")
        if not gt.any():
            raise ValueError("Each track needs at least one scored GT frame")
        labels = targets[gt].astype(np.int64, copy=False)
        if ((labels < 0) | (labels >= len(CLASS_NAMES))).any():
            raise ValueError("Invalid scored target")

        # Stable softmax makes the acceptance score depend only on model output.
        selected = logits[gt]
        shifted = selected - selected.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predictions = selected.argmax(axis=1)
        count = int(gt.sum())
        arrays["scores"].append(probabilities.max(axis=1))
        arrays["predicted"].append(predictions)
        arrays["targets"].append(labels)
        arrays["weights"].append(np.full(count, 1.0 / count, dtype=np.float64))
        arrays["track"].append(np.full(count, index, dtype=np.int64))
        group_id = group_ids.setdefault(group, len(group_ids))
        arrays["group_ids"].append(np.full(count, group_id, dtype=np.int64))

        source_flags = row.get("source_flags", {})
        if not isinstance(source_flags, dict):
            raise ValueError("source_flags must be a mapping")
        for name in flag_names:
            if name not in source_flags:
                value = np.full(count, 2, dtype=np.uint8)  # unknown availability
            else:
                raw = np.asarray(source_flags[name])
                if raw.shape != targets.shape or raw.dtype.kind not in "biu" or not np.isin(raw, (0, 1)).all():
                    raise ValueError(f"Availability flag {name!r} must be a boolean track array")
                value = raw[gt].astype(np.uint8, copy=False)
            flags.setdefault(name, []).append(value)

    result = {key: np.concatenate(values) for key, values in arrays.items()}
    result["flag_values"] = {name: np.concatenate(values) for name, values in flags.items()}
    result["track_count"] = len(tracks)
    result["group_count"] = len(group_ids)
    result["frame_count"] = len(result["scores"])
    return result


def _metrics(population, accepted, subset=None):
    n = population["frame_count"]
    subset = np.ones(n, dtype=bool) if subset is None else np.asarray(subset, dtype=bool)
    accepted = np.asarray(accepted, dtype=bool) & subset
    wrong = population["predicted"] != population["targets"]
    errors = accepted & wrong
    weights = population["weights"]
    total_mass = float(weights[subset].sum())
    accepted_mass = float(weights[accepted].sum())
    error_mass = float(weights[errors].sum())
    accepted_frames = int(accepted.sum())
    error_frames = int(errors.sum())
    accepted_track_ids = np.unique(population["track"][accepted])
    error_track_ids = np.unique(population["track"][errors])
    accepted_group_ids = np.unique(population["group_ids"][accepted])
    error_group_ids = np.unique(population["group_ids"][errors])
    classes = []
    for class_id, name in enumerate(CLASS_NAMES):
        original = subset & (population["targets"] == class_id)
        accepted_gt = accepted & (population["targets"] == class_id)
        accepted_predicted = accepted & (population["predicted"] == class_id)
        correct = accepted_gt & (population["predicted"] == class_id)
        original_mass = float(weights[original].sum())
        accepted_gt_mass = float(weights[accepted_gt].sum())
        predicted_mass = float(weights[accepted_predicted].sum())
        correct_mass = float(weights[correct].sum())
        original_frames = int(original.sum())
        accepted_gt_frames = int(accepted_gt.sum())
        predicted_frames = int(accepted_predicted.sum())
        correct_frames = int(correct.sum())
        classes.append({
            "class_id": class_id, "class_name": name,
            "original_gt_frames": original_frames, "original_gt_mass": original_mass,
            "accepted_gt_frames": accepted_gt_frames, "accepted_gt_mass": accepted_gt_mass,
            "gt_class_retention": accepted_gt_mass / original_mass if original_mass else None,
            "accepted_predicted_frames": predicted_frames, "accepted_predicted_mass": predicted_mass,
            "accepted_precision": correct_mass / predicted_mass if predicted_mass else None,
            "accepted_correct_frames": correct_frames, "accepted_correct_mass": correct_mass,
            "automated_correct_fraction": correct_mass / original_mass if original_mass else None,
        })
    return {
        "coverage": accepted_mass / total_mass if total_mass else None,
        "error": error_mass / accepted_mass if accepted_mass else None,
        "total_mass": total_mass, "accepted_mass": accepted_mass, "error_mass": error_mass,
        "raw_coverage": accepted_frames / int(subset.sum()) if subset.any() else None,
        "raw_error": error_frames / accepted_frames if accepted_frames else None,
        "total_frames": int(subset.sum()), "accepted_frames": accepted_frames,
        "error_frames": error_frames,
        "accepted_tracks": int(len(accepted_track_ids)), "accepted_groups": int(len(accepted_group_ids)),
        "error_tracks": int(len(error_track_ids)), "error_groups": int(len(error_group_ids)),
        "classes": classes,
    }


def _curve(population):
    scores = population["scores"]
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(scores) - 1]
    thresholds = sorted_scores[ends]
    wrong = population["predicted"] != population["targets"]
    accepted_frames = ends + 1
    accepted_mass = np.cumsum(population["weights"][order])[ends]
    error_frames = np.cumsum(wrong[order], dtype=np.int64)[ends]
    error_mass = np.cumsum(population["weights"][order] * wrong[order])[ends]
    track_max = np.full(population["track_count"], -np.inf)
    np.maximum.at(track_max, population["track"][order], sorted_scores)
    group_max = np.full(population["group_count"], -np.inf)
    np.maximum.at(group_max, population["group_ids"][order], sorted_scores)
    error_track_max = np.full(population["track_count"], -np.inf)
    np.maximum.at(error_track_max, population["track"][wrong], scores[wrong])
    error_group_max = np.full(population["group_count"], -np.inf)
    np.maximum.at(error_group_max, population["group_ids"][wrong], scores[wrong])

    def active_count(max_scores):
        return len(max_scores) - np.searchsorted(np.sort(max_scores), thresholds, side="left")

    total_mass = float(population["track_count"])
    return {
        "threshold": thresholds.astype(np.float64),
        "accepted_frames": accepted_frames.astype(np.int64),
        "accepted_mass": accepted_mass.astype(np.float64),
        "error_frames": error_frames.astype(np.int64),
        "error_mass": error_mass.astype(np.float64),
        "coverage": (accepted_mass / total_mass).astype(np.float64),
        "error": np.divide(error_mass, accepted_mass, out=np.full_like(error_mass, np.nan), where=accepted_mass > 0),
        "raw_coverage": (accepted_frames / population["frame_count"]).astype(np.float64),
        "raw_error": np.divide(error_frames, accepted_frames, out=np.full(len(ends), np.nan), where=accepted_frames > 0),
        "accepted_tracks": active_count(track_max).astype(np.int64),
        "accepted_groups": active_count(group_max).astype(np.int64),
        "error_tracks": active_count(error_track_max).astype(np.int64),
        "error_groups": active_count(error_group_max).astype(np.int64),
    }


def _choose_threshold(curve, error_target):
    eligible = np.flatnonzero(curve["error"] <= error_target)
    if not len(eligible):
        return None, True
    # Coverage is monotone in this prefix; error is not, so inspect every prefix.
    selected = eligible[np.argmax(curve["accepted_frames"][eligible])]
    return float(curve["threshold"][selected]), False


def _json_dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def analyze_checkpoint(calibration_tracks, assessment_tracks, output: Path, error_targets=ERROR_TARGETS):
    """Choose calibration cutoffs and evaluate them unchanged on assessment tracks."""
    targets = tuple(float(value) for value in error_targets)
    if not targets or any(not math.isfinite(value) or not 0 <= value <= 1 for value in targets):
        raise ValueError("Error targets must be finite values in [0, 1]")
    calibration = _population(calibration_tracks)
    assessment = _population(assessment_tracks)
    calibration_curve, assessment_curve = _curve(calibration), _curve(assessment)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    curve_files = {"calibration": "calibration-curve.npz", "assessment": "assessment-curve.npz"}
    np.savez_compressed(output / curve_files["calibration"], **calibration_curve)
    np.savez_compressed(output / curve_files["assessment"], **assessment_curve)
    result = {"error_targets": list(targets), "curve_files": curve_files, "targets": []}
    for error_target in targets:
        threshold, reject_all = _choose_threshold(calibration_curve, error_target)
        calibration_accept = calibration["scores"] >= threshold if not reject_all else np.zeros(calibration["frame_count"], bool)
        assessment_accept = assessment["scores"] >= threshold if not reject_all else np.zeros(assessment["frame_count"], bool)
        target_result = {
            "error_target": error_target, "threshold": threshold, "reject_all": reject_all,
            "calibration": _metrics(calibration, calibration_accept),
            "assessment": _metrics(assessment, assessment_accept),
        }
        for name, population, accepted in (("calibration", calibration, calibration_accept),
                                           ("assessment", assessment, assessment_accept)):
            cohorts = {}
            for flag, values in population["flag_values"].items():
                cohorts[flag] = {
                    label: _metrics(population, accepted, values == code)
                    for label, code in (("unavailable", 0), ("available", 1), ("unknown", 2))
                }
            target_result[name]["natural_availability_cohorts"] = cohorts
        result["targets"].append(target_result)
    _json_dump(output / "analysis.json", result)
    return result


def _safe(value):
    if isinstance(value, dict):
        return {str(key): safe for key, item in value.items() if (safe := _safe(item)) is not _UNSAFE}
    if isinstance(value, (list, tuple)):
        return [safe for item in value if (safe := _safe(item)) is not _UNSAFE]
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, np.generic):
        return _safe(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _UNSAFE


_UNSAFE = object()


def _target_result(coverage, error_target):
    for row in coverage["targets"]:
        if math.isclose(float(row["error_target"]), error_target, rel_tol=0, abs_tol=1e-12):
            return row
    raise ValueError(f"Run is missing calibration error target {error_target:g}")


def _mean_sd(values, anchor=False):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if len(values) else None, (float(values.std(ddof=1)) if len(values) > 1 and not anchor else None)


def _annotation_caption(rows):
    def shared_counts(key):
        values = []
        for row in rows:
            access = row.get("annotation_access", {})
            section = access.get(key) if isinstance(access, dict) else None
            if (not isinstance(section, dict) or not isinstance(section.get("tracks"), (int, np.integer))
                    or not isinstance(section.get("gt_timesteps"), (int, np.integer))):
                return None
            values.append((section["tracks"], section["gt_timesteps"]))
        return values[0] if values and len(set(values)) == 1 else None

    calibration = shared_counts("development_calibration")
    audit = shared_counts("assessment_audit")
    if calibration is None:
        return "Training-label budget excludes development/calibration labels; see each run's annotation_access record."
    lines = [f"Additional calibration/development labels: {calibration[0]:,} tracks / {calibration[1]:,} GT timesteps."]
    if audit is not None:
        lines.append(f"Assessment/audit labels: {audit[0]:,} tracks / {audit[1]:,} GT timesteps.")
    return "\n".join(lines)


def make_report(rows: list[dict], output: Path) -> dict:
    """Write budget summaries and the two fixed E004a plots."""
    if not rows:
        raise ValueError("E004a report requires at least one run")
    targets = ERROR_TARGETS
    runs = [_safe(row) for row in rows]
    if any(run is _UNSAFE for run in runs):
        raise ValueError("Each run row must be JSON-safe")
    grouped = {}
    for row in rows:
        budget = row.get("budget")
        f1, ce = row.get("macro_f1"), row.get("track_ce")
        if not isinstance(budget, (int, np.integer)) or budget < 0:
            raise ValueError("Run budget must be a nonnegative integer")
        if not isinstance(f1, (int, float, np.number)) or not math.isfinite(float(f1)) or not 0 <= float(f1) <= 1:
            raise ValueError("Run macro_f1 must be finite and in [0, 1]")
        if not isinstance(ce, (int, float, np.number)) or not math.isfinite(float(ce)):
            raise ValueError("Run track_ce must be finite")
        for target in targets:
            _target_result(row["coverage"], target)
        grouped.setdefault(int(budget), []).append(row)

    summary = []
    for budget in sorted(grouped):
        group = grouped[budget]
        anchor = budget in (0, ALL_TRACKS)
        f1_mean, f1_sd = _mean_sd([float(row["macro_f1"]) for row in group], anchor)
        ce_mean, ce_sd = _mean_sd([float(row["track_ce"]) for row in group], anchor)
        entry = {"budget": budget, "n": len(group), "draws": [row.get("draw") for row in group],
                 "macro_f1_mean": f1_mean, "macro_f1_sd": f1_sd, "track_ce_mean": ce_mean, "track_ce_sd": ce_sd,
                 "targets": {}}
        for target in targets:
            observed = [_target_result(row["coverage"], target) for row in group]
            coverage_values = [float(item["assessment"]["coverage"]) for item in observed]
            errors = [float(item["assessment"]["error"]) for item in observed
                      if item["assessment"].get("accepted_frames", 0) > 0 and item["assessment"].get("error") is not None]
            overruns = sum(error > target for error in errors)
            empties = sum(item["assessment"].get("accepted_frames", 0) == 0 for item in observed)
            reject_all = sum(bool(item.get("reject_all")) for item in observed)
            coverage_mean, coverage_sd = _mean_sd(coverage_values, anchor)
            error_mean, error_sd = _mean_sd(errors, anchor)
            entry["targets"][str(target)] = {
                "coverage_mean": coverage_mean, "coverage_sd": coverage_sd,
                "assessment_error_mean_nonempty": error_mean, "assessment_error_sd_nonempty": error_sd,
                "assessment_error_n_nonempty": len(errors), "overrun_count": int(overruns),
                "empty_assessment_count": int(empties), "reject_all_count": int(reject_all),
            }
        summary.append(entry)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    caption = _annotation_caption(rows)
    report = {"targets": list(targets), "all_tracks": ALL_TRACKS, "summary": summary, "runs": runs, "caption": caption}
    _json_dump(output / "budget-report.json", report)
    _write_csv(output / "budget-summary.csv", summary, targets)
    _write_run_csv(output / "budget-runs.csv", runs, targets)
    _plot_f1(summary, output, caption)
    _plot_coverage(summary, output, targets, caption)
    return report


def _write_csv(path, rows, targets):
    fields = ["budget", "n", "draws", "macro_f1_mean", "macro_f1_sd", "track_ce_mean", "track_ce_sd"]
    for target in targets:
        prefix = f"target_{target:g}"
        fields.extend(f"{prefix}_{name}" for name in ("coverage_mean", "coverage_sd", "assessment_error_mean_nonempty",
            "assessment_error_sd_nonempty", "assessment_error_n_nonempty", "overrun_count", "empty_assessment_count", "reject_all_count"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: value for key, value in row.items() if key != "targets"}
            flat["draws"] = json.dumps(row["draws"])
            for target, metrics in row["targets"].items():
                flat.update({f"target_{float(target):g}_{key}": value for key, value in metrics.items()})
            writer.writerow(flat)


def _write_run_csv(path, rows, targets):
    fields = ["budget", "draw", "macro_f1", "track_ce", "annotation_access"]
    for target in targets:
        fields.extend(f"target_{target:g}_{name}" for name in (
            "threshold", "reject_all", "calibration_coverage", "calibration_error", "assessment_coverage",
            "assessment_error", "accepted_frames", "error_frames", "accepted_mass", "error_mass", "classes"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in rows:
            flat = {key: run.get(key) for key in ("budget", "draw", "macro_f1", "track_ce", "annotation_access")}
            if isinstance(flat.get("annotation_access"), (dict, list)):
                flat["annotation_access"] = json.dumps(flat["annotation_access"], separators=(",", ":"))
            for target in targets:
                item = _target_result(run["coverage"], target)
                prefix = f"target_{target:g}_"
                flat.update({prefix + "threshold": item.get("threshold"), prefix + "reject_all": item.get("reject_all"),
                    prefix + "calibration_coverage": item["calibration"].get("coverage"),
                    prefix + "calibration_error": item["calibration"].get("error"),
                    prefix + "assessment_coverage": item["assessment"].get("coverage"),
                    prefix + "assessment_error": item["assessment"].get("error"),
                    prefix + "accepted_frames": item["assessment"].get("accepted_frames"),
                    prefix + "error_frames": item["assessment"].get("error_frames"),
                    prefix + "accepted_mass": item["assessment"].get("accepted_mass"),
                    prefix + "error_mass": item["assessment"].get("error_mass"),
                    prefix + "classes": json.dumps(item["assessment"].get("classes", []), separators=(",", ":"))})
            writer.writerow(flat)


def _plot_axes(summary, ylabel, ylim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Labeled training tracks")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(sorted({0, ALL_TRACKS, *(row["budget"] for row in summary)}))
    ax.set_xticklabels([f"{value:,}" for value in ax.get_xticks()])
    return plt, fig, ax


def _plot_f1(summary, output, caption):
    plt, fig, ax = _plot_axes(summary, "Track-weighted macro-F1 (%)", (0, 100))
    budgets = [row["budget"] for row in summary]
    means = np.asarray([row["macro_f1_mean"] * 100 for row in summary])
    deviations = np.asarray([row["macro_f1_sd"] * 100 if row["macro_f1_sd"] is not None else 0 for row in summary])
    ax.errorbar(budgets, means, yerr=deviations, fmt="-o", capsize=3, color="#345995")
    ax.set_title("Full-coverage performance by labeled-track budget")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.text(0.01, 0.015, caption, fontsize=7.5, va="bottom")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"budget-f1.{suffix}", dpi=160)
    plt.close(fig)


def _plot_coverage(summary, output, targets, caption):
    plt, fig, ax = _plot_axes(summary, "Track-weighted automatic coverage (%)", (0, 100))
    budgets = [row["budget"] for row in summary]
    colors = ("#d1495b", "#00798c", "#edae49", "#30638e")
    for target, color in zip(targets, colors):
        key = str(target)
        means = np.asarray([row["targets"][key]["coverage_mean"] * 100 for row in summary])
        deviations = np.asarray([row["targets"][key]["coverage_sd"] * 100 if row["targets"][key]["coverage_sd"] is not None else 0
                                 for row in summary])
        central = target == 0.05
        ax.errorbar(budgets, means, yerr=deviations, fmt="-", capsize=3, color=color,
                    linewidth=2.4 if central else 1.2, label=f"{target:.0%} calibration error target" + (" (central)" if central else ""))
        overrun = [row["targets"][key]["overrun_count"] > 0 for row in summary]
        ax.scatter(budgets, means, facecolors=["none" if value else color for value in overrun],
                   edgecolors=color, s=42, linewidths=1.3, zorder=4)
    ax.set_title("Automatic coverage at calibration error targets")
    ax.legend(frameon=False)
    fig.tight_layout(rect=(0, 0.22, 1, 1))
    fig.text(0.01, 0.01, caption + "\nHollow markers indicate a nonempty assessment overrun; empty accepted sets have undefined error.",
             fontsize=7.5, va="bottom")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"budget-coverage.{suffix}", dpi=160)
    plt.close(fig)
