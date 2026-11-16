"""Compact method and label-budget reporting for E004c."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from pedestrian_behavior.experiments.e004a_analysis import (
    CLASS_NAMES,
    ERROR_TARGETS,
    _annotation_caption,
    _json_dump,
    _mean_sd,
    _safe,
    _target_result,
)


METHODS = ("head", "temporal-head", "target-only")
ALL_METHODS = (*METHODS, "source")
PLOTTED_TARGETS = (0.05, 0.10)
COLORS = {"head": "#345995", "temporal-head": "#d1495b", "target-only": "#00798c"}


def make_report(rows: list[dict], output: Path) -> dict:
    """Write method-by-budget summaries, raw rows, and three figures."""
    if not rows:
        raise ValueError("E004c report requires at least one run")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Each run row must be a mapping")
    runs = [_safe(row) for row in rows]
    if any(not isinstance(run, dict) for run in runs):
        raise ValueError("Each run row must be JSON-safe")

    grouped = {}
    source_count = 0
    for row in rows:
        method, budget = row.get("method"), row.get("budget")
        f1, ce, class_f1 = row.get("macro_f1"), row.get("track_ce"), row.get("class_f1")
        if method not in ALL_METHODS:
            raise ValueError(f"Unknown E004c method: {method!r}")
        if not isinstance(budget, (int, np.integer)) or isinstance(budget, (bool, np.bool_)) or budget < 0:
            raise ValueError("Run budget must be a nonnegative integer")
        if (method == "source") != (budget == 0):
            raise ValueError("Only the fixed source row may have budget zero")
        source_count += method == "source"
        if not isinstance(f1, (int, float, np.number)) or not math.isfinite(float(f1)) or not 0 <= float(f1) <= 1:
            raise ValueError("Run macro_f1 must be finite and in [0, 1]")
        if not isinstance(ce, (int, float, np.number)) or not math.isfinite(float(ce)):
            raise ValueError("Run track_ce must be finite")
        if (not isinstance(class_f1, (list, tuple)) or len(class_f1) != len(CLASS_NAMES)
                or any(not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value))
                       or not 0 <= float(value) <= 1 for value in class_f1)):
            raise ValueError("Run class_f1 must contain four finite values in [0, 1]")
        for target in ERROR_TARGETS:
            _target_result(row["coverage"], target)
        grouped.setdefault((method, int(budget)), []).append(row)
    if source_count != 1:
        raise ValueError("E004c report requires exactly one fixed source row")
    if not any(method != "source" for method, _ in grouped):
        raise ValueError("E004c report requires at least one target-fit row")

    summary = []
    for (method, budget), group in sorted(grouped.items(), key=lambda item: (ALL_METHODS.index(item[0][0]), item[0][1])):
        source = method == "source"
        f1_mean, f1_sd = _mean_sd([float(row["macro_f1"]) for row in group], source)
        ce_mean, ce_sd = _mean_sd([float(row["track_ce"]) for row in group], source)
        class_values = np.asarray([row["class_f1"] for row in group], dtype=np.float64)
        entry = {
            "method": method, "budget": budget, "n": len(group),
            "draws": [row.get("draw") for row in group],
            "macro_f1_mean": f1_mean, "macro_f1_sd": f1_sd,
            "track_ce_mean": ce_mean, "track_ce_sd": ce_sd,
            "class_f1_mean": class_values.mean(axis=0).tolist(),
            "class_f1_sd": (class_values.std(axis=0, ddof=1).tolist() if len(group) > 1 and not source else [None] * len(CLASS_NAMES)),
            "targets": {},
        }
        for target in ERROR_TARGETS:
            observed = [_target_result(row["coverage"], target) for row in group]
            coverage_values = [float(item["assessment"]["coverage"]) for item in observed]
            errors = [float(item["assessment"]["error"]) for item in observed
                      if item["assessment"].get("error") is not None
                      and item["assessment"].get("accepted_frames", 0) > 0]
            coverage_mean, coverage_sd = _mean_sd(coverage_values, source)
            error_mean, error_sd = _mean_sd(errors, source)
            entry["targets"][str(target)] = {
                "coverage_mean": coverage_mean, "coverage_sd": coverage_sd,
                "assessment_error_mean_nonempty": error_mean,
                "assessment_error_sd_nonempty": error_sd,
                "assessment_error_n_nonempty": len(errors),
                "assessment_error_n_undefined": len(observed) - len(errors),
                "overrun_count": sum(error > target for error in errors),
                "empty_assessment_count": sum(item["assessment"].get("accepted_frames", 0) == 0 for item in observed),
                "reject_all_count": sum(bool(item.get("reject_all")) for item in observed),
            }
        summary.append(entry)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    caption = (_annotation_caption(rows)
               + "\nFull comparison: five subset draws at fixed training seed 0; per-series n is saved. "
                 "Validation groups have historical project exposure.")
    report = {"targets": list(ERROR_TARGETS), "class_names": list(CLASS_NAMES),
              "summary": summary, "runs": runs, "caption": caption}
    _json_dump(output / "budget-report.json", report)
    _write_summary_csv(output / "budget-summary.csv", summary)
    _write_run_csv(output / "budget-runs.csv", runs)
    _plot_f1(summary, output, caption)
    _plot_class_f1(summary, output, caption)
    _plot_coverage(summary, output, caption)
    return report


def _write_summary_csv(path, summary):
    fields = ["method", "budget", "n", "draws", "macro_f1_mean", "macro_f1_sd", "track_ce_mean", "track_ce_sd"]
    for name in CLASS_NAMES:
        fields.extend((f"class_f1_{name}_mean", f"class_f1_{name}_sd"))
    for target in ERROR_TARGETS:
        prefix = f"target_{target:g}"
        fields.extend(f"{prefix}_{name}" for name in (
            "coverage_mean", "coverage_sd", "assessment_error_mean_nonempty", "assessment_error_sd_nonempty",
            "assessment_error_n_nonempty", "assessment_error_n_undefined", "overrun_count",
            "empty_assessment_count", "reject_all_count"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            flat = {key: row[key] for key in fields if key in row}
            flat["draws"] = json.dumps(row["draws"])
            for index, name in enumerate(CLASS_NAMES):
                flat[f"class_f1_{name}_mean"] = row["class_f1_mean"][index]
                flat[f"class_f1_{name}_sd"] = row["class_f1_sd"][index]
            for target, metrics in row["targets"].items():
                flat.update({f"target_{float(target):g}_{key}": value for key, value in metrics.items()})
            writer.writerow(flat)


def _write_run_csv(path, runs):
    fields = list(dict.fromkeys(key for row in runs for key in row if key != "coverage"))
    fields.extend(f"target_{target:g}_{name}" for target in ERROR_TARGETS for name in (
        "threshold", "reject_all", "calibration_coverage", "calibration_error", "assessment_coverage",
        "assessment_error", "accepted_frames", "error_frames", "classes"))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            flat = {key: run.get(key) for key in fields if not key.startswith("target_")}
            for target in ERROR_TARGETS:
                item = _target_result(run["coverage"], target)
                prefix = f"target_{target:g}_"
                flat.update({prefix + "threshold": item.get("threshold"), prefix + "reject_all": item.get("reject_all"),
                    prefix + "calibration_coverage": item["calibration"].get("coverage"),
                    prefix + "calibration_error": item["calibration"].get("error"),
                    prefix + "assessment_coverage": item["assessment"].get("coverage"),
                    prefix + "assessment_error": item["assessment"].get("error"),
                    prefix + "accepted_frames": item["assessment"].get("accepted_frames"),
                    prefix + "error_frames": item["assessment"].get("error_frames"),
                    prefix + "classes": item["assessment"].get("classes", [])})
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                             for key, value in flat.items()})


def _axes(ax, budgets, ylabel):
    ax.set_xscale("log")
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{budget:,}" for budget in budgets])
    ax.set_xlabel("Target training tracks (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.2)


def _save_figure(plt, fig, output, name, caption, rect=(0, 0.16, 1, 1)):
    fig.tight_layout(rect=rect)
    fig.text(0.01, 0.01, caption, fontsize=7.5, va="bottom")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=160)
    plt.close(fig)


def _plot_f1(summary, output, caption):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    budgets = sorted({row["budget"] for row in summary if row["method"] != "source"})
    _axes(ax, budgets, "Track-weighted assessment macro-F1 (%)")
    source = next(row for row in summary if row["method"] == "source")
    ax.axhline(source["macro_f1_mean"] * 100, color="#555555", linestyle="--",
               label=f"Fixed source reference ({source['macro_f1_mean']:.1%})")
    for method in METHODS:
        rows = [row for row in summary if row["method"] == method]
        if rows:
            ax.errorbar([row["budget"] for row in rows], [row["macro_f1_mean"] * 100 for row in rows],
                        yerr=[(row["macro_f1_sd"] or 0) * 100 for row in rows], fmt="-o", capsize=3,
                        color=COLORS[method], label=method)
    ax.set_title("Assessment macro-F1 by target-label budget")
    ax.legend(frameon=False)
    _save_figure(plt, fig, output, "budget-f1", caption)


def _plot_class_f1(summary, output, caption):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    budgets = sorted({row["budget"] for row in summary if row["method"] != "source"})
    source = next(row for row in summary if row["method"] == "source")
    for index, (name, ax) in enumerate(zip(CLASS_NAMES, axes.flat)):
        _axes(ax, budgets, "Assessment F1 (%)")
        ax.axhline(source["class_f1_mean"][index] * 100, color="#555555", linestyle="--",
                   label="Fixed source reference")
        for method in METHODS:
            rows = [row for row in summary if row["method"] == method]
            if rows:
                ax.errorbar([row["budget"] for row in rows], [row["class_f1_mean"][index] * 100 for row in rows],
                            yerr=[(row["class_f1_sd"][index] or 0) * 100 for row in rows], fmt="-o", capsize=3,
                            color=COLORS[method], label=method)
        ax.set_title(name)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False)
    fig.suptitle("Per-class assessment F1 by target-label budget", y=0.99)
    _save_figure(plt, fig, output, "budget-class-f1", caption, rect=(0, 0.16, 1, 0.86))


def _plot_coverage(summary, output, caption):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey="col")
    budgets = sorted({row["budget"] for row in summary if row["method"] != "source"})
    source = next(row for row in summary if row["method"] == "source")
    for row_index, target in enumerate(PLOTTED_TARGETS):
        key = str(target)
        coverage_ax, error_ax = axes[row_index]
        for ax, metric, sd_metric, ylabel in (
            (coverage_ax, "coverage_mean", "coverage_sd", "Coverage (%)"),
            (error_ax, "assessment_error_mean_nonempty", "assessment_error_sd_nonempty", "Accepted error (%)"),
        ):
            _axes(ax, budgets, ylabel)
            ax.set_title(f"{target:.0%} target: {'coverage' if ax is coverage_ax else 'accepted error'}")
            if metric.startswith("assessment_error"):
                ax.axhline(target * 100, color="#555555", linestyle="--", label="Calibration target")
            source_metrics = source["targets"][key]
            source_value = source_metrics[metric]
            if source_value is not None:
                ax.axhline(source_value * 100, color="#888888", linestyle=":", label="Fixed source reference")
        for method in METHODS:
            rows = [row for row in summary if row["method"] == method]
            if not rows:
                continue
            metrics = [row["targets"][key] for row in rows]
            x = [row["budget"] for row in rows]
            for ax, metric, sd_metric in (
                (coverage_ax, "coverage_mean", "coverage_sd"),
                (error_ax, "assessment_error_mean_nonempty", "assessment_error_sd_nonempty"),
            ):
                values = [np.nan if item[metric] is None else item[metric] * 100 for item in metrics]
                deviations = [0 if item[sd_metric] is None else item[sd_metric] * 100 for item in metrics]
                ax.errorbar(x, values, yerr=deviations, fmt="-", capsize=3, color=COLORS[method], label=method)
                valid = [(budget, value, item) for budget, value, item in zip(x, values, metrics)
                         if math.isfinite(value)]
                if valid:
                    ax.scatter([point[0] for point in valid], [point[1] for point in valid], marker="o",
                               facecolors=["none" if point[2]["overrun_count"] else COLORS[method] for point in valid],
                               edgecolors=COLORS[method], s=42, linewidths=1.3, zorder=4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for handle, label in zip(*axes[0, 1].get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=5, frameon=False)
    fig.suptitle("Assessment automatic coverage and achieved error", y=0.99)
    note = ("Hollow markers indicate at least one assessment overrun; error means exclude undefined empty-set cases. "
            "1% and 20% targets are retained in the tables. Empirical calibration targets are not risk guarantees.")
    _save_figure(plt, fig, output, "budget-coverage", caption + "\n" + note, rect=(0.08, 0.20, 1, 0.86))
