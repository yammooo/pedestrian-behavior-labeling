"""Summarize per-track error ranges and saved geometry; no inference."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("outputs/experiments")
TRACK_ERRORS = ROOT / "E004/track-difficulty-20261116/track-errors.csv"
PREDICTIONS = ROOT / "E002/transfer-validation-20261102/cells/road-waymo/geometry+rgb+lidar/predictions.npz"
OUTPUT = ROOT / "E004/track-error-ranges-20261116"
MODELS = ("road-direct", "road-head", "loki-trained")
CLASSES = ("moving", "stopped", "waiting_to_cross", "crossing")
TYPES = ("MOVING", "STOPPED", "WAITING", "CROSSING", "MIXED")
ERROR_LABELS = ("0–5%", "(5–20%]", "(20–50%]", "(50–80%]", "(80–95%]", "(95–100%]")
ERROR_UPPER = np.array((0.05, 0.20, 0.50, 0.80, 0.95, 1.0))
ERROR_COLORS = ("#1a9850", "#91cf60", "#d9ef8b", "#fee08b", "#fc8d59", "#d73027")
MODEL_LABELS = {"road-direct": "ROAD direct", "road-head": "ROAD + head", "loki-trained": "LOKI trained"}


def error_bin(errors, gt_frames):
    if gt_frames <= 0 or errors < 0 or errors > gt_frames:
        raise ValueError(f"Invalid per-track counts: errors={errors}, GT={gt_frames}")
    return int(np.searchsorted(ERROR_UPPER, errors / gt_frames, side="left"))


def read_error_rows(path):
    by_model = {model: {} for model in MODELS}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            model, locator = row["model"], row["locator"]
            if model not in by_model or locator in by_model[model]:
                raise ValueError(f"Unexpected or duplicate model/track row: {model}/{locator}")
            by_model[model][locator] = row
    locators = set(by_model[MODELS[0]])
    if len(locators) != 931 or any(set(rows) != locators for rows in by_model.values()):
        raise ValueError("Expected identical saved predictions for the 931-track E004 subset")
    for locator in locators:
        first = by_model[MODELS[0]][locator]
        for model in MODELS[1:]:
            row = by_model[model][locator]
            for field in ("gt_frames", *(f"gt_{name}_frames" for name in CLASSES)):
                if row[field] != first[field]:
                    raise ValueError(f"GT counts differ across models: {locator}/{field}")
    return by_model


def track_type(row):
    present = [name for name in CLASSES if int(row[f"gt_{name}_frames"]) > 0]
    if not present:
        raise ValueError(f"Track has no accepted GT: {row['locator']}")
    if len(present) > 1:
        return "MIXED"
    return {"moving": "MOVING", "stopped": "STOPPED", "waiting_to_cross": "WAITING",
            "crossing": "CROSSING"}[present[0]]


def extract_track_metadata(by_model, destination):
    reference = by_model["road-direct"]
    with np.load(PREDICTIONS, allow_pickle=False) as saved:
        prediction_metadata = json.loads(str(saved["metadata"]))
    archives = {item["locator"]: item["archive"] for item in prediction_metadata}
    missing = set(reference) - set(archives)
    if missing:
        raise ValueError(f"Missing saved geometry archive references for {len(missing)} tracks")

    records = []
    for locator in sorted(reference):
        row = reference[locator]
        archive_path = Path(archives[locator])
        if not archive_path.is_file():
            raise FileNotFoundError(f"Saved prepared geometry archive unavailable: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["track_locator"] != locator:
                raise ValueError(f"Prepared archive locator mismatch: {locator}")
            ped, ego = archive["ped_position"], archive["ego_position"]
            valid = archive["ped_position_valid"].astype(bool) & archive["ego_pose_valid"].astype(bool)
            if ped.shape != ego.shape or ped.shape[0] != len(valid) or not len(valid):
                raise ValueError(f"Invalid unpadded position arrays: {locator}")
            distance = np.linalg.norm(ped[valid] - ego[valid], axis=1)
            if not np.isfinite(distance).all():
                raise ValueError(f"Nonfinite valid pedestrian–ego range: {locator}")
        records.append({
            "locator": locator,
            "scenario": row["scenario"],
            "track_type": track_type(row),
            "duration_seconds_nominal": float(row["duration_seconds_nominal"]),
            "gt_frames": int(row["gt_frames"]),
            "median_ped_ego_range_m": float(np.median(distance)) if len(distance) else None,
            "distance_valid_slots": len(distance),
            "distance_unknown": int(not len(distance)),
            **{f"gt_{name}_frames": int(row[f"gt_{name}_frames"]) for name in CLASSES},
        })
    write_csv(destination, records)
    return {row["locator"]: row for row in records}


def read_track_metadata(path):
    records = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            locator = row["locator"]
            if locator in records:
                raise ValueError(f"Duplicate cached track metadata: {locator}")
            for field in ("duration_seconds_nominal", "gt_frames", "distance_valid_slots", "distance_unknown",
                          *(f"gt_{name}_frames" for name in CLASSES)):
                row[field] = int(row[field]) if field not in ("duration_seconds_nominal",) else float(row[field])
            row["median_ped_ego_range_m"] = (float(row["median_ped_ego_range_m"])
                                               if row["median_ped_ego_range_m"] else None)
            records[locator] = row
    return records


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values):
    if not values:
        return None, None, None
    q1, median, q3 = np.quantile(np.asarray(values, dtype=float), (0.25, 0.50, 0.75))
    return float(median), float(q1), float(q3)


def build_summary(by_model, metadata):
    summary = {"bins": {}, "distance_semantics": (
        "Per-track median Euclidean pedestrian–ego distance in meters over all unpadded context slots "
        "where ped_position_valid and ego_pose_valid are true; no GT mask is applied. Position formula "
        "matches E001 condition_range. Empty valid-position tracks are retained as unknown."),
        "error_weighting": "Per-track error is raw misclassified accepted GT frames divided by that track's total accepted GT frames.",
        "models": {}}
    rows = []
    for model in MODELS:
        tracks = []
        for locator, row in by_model[model].items():
            gt = int(row["gt_frames"])
            wrong = int(row["raw_errors"])
            record = metadata[locator]
            tracks.append({"locator": locator, "error_bin": error_bin(wrong, gt), "track_type": record["track_type"],
                           "duration": record["duration_seconds_nominal"], "gt_frames": gt,
                           "distance": record["median_ped_ego_range_m"], "unknown_distance": record["distance_unknown"]})
        model_bins = {}
        for bin_id, label in enumerate(ERROR_LABELS):
            cohort = [track for track in tracks if track["error_bin"] == bin_id]
            durations = [track["duration"] for track in cohort]
            gt_counts = [track["gt_frames"] for track in cohort]
            distances = [track["distance"] for track in cohort if track["distance"] is not None]
            d_med, d_q1, d_q3 = quantiles(durations)
            gt_med, gt_q1, gt_q3 = quantiles(gt_counts)
            range_med, range_q1, range_q3 = quantiles(distances)
            composition = Counter(track["track_type"] for track in cohort)
            item = {"track_count": len(cohort), "track_fraction": len(cohort) / len(tracks),
                    "duration_median_seconds": d_med, "duration_q1_seconds": d_q1, "duration_q3_seconds": d_q3,
                    "gt_frames_median": gt_med, "gt_frames_q1": gt_q1, "gt_frames_q3": gt_q3,
                    "distance_median_m": range_med, "distance_q1_m": range_q1, "distance_q3_m": range_q3,
                    "unknown_distance_tracks": sum(track["unknown_distance"] for track in cohort),
                    "class_composition_tracks": {kind: composition[kind] for kind in TYPES}}
            model_bins[label] = item
            rows.append({"model": model, "error_bin": label, **{key: value for key, value in item.items()
                         if key != "class_composition_tracks"},
                         **{f"tracks_{kind.lower()}": composition[kind] for kind in TYPES}})
        summary["models"][model] = model_bins
    summary["total_tracks_per_model"] = {model: len(by_model[model]) for model in MODELS}
    return summary, rows


def plot_composition(summary, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharex=True)
    for ax, model in zip(axes, MODELS):
        counts = np.array([[summary["models"][model][label]["class_composition_tracks"][kind]
                            for label in ERROR_LABELS] for kind in TYPES])
        totals = counts.sum(axis=1)
        percentages = np.divide(counts * 100, totals[:, None], out=np.zeros_like(counts, dtype=float),
                                where=totals[:, None] > 0)
        left = np.zeros(len(TYPES))
        for bin_id, label in enumerate(ERROR_LABELS):
            ax.barh(np.arange(len(TYPES)), percentages[:, bin_id], left=left,
                    color=ERROR_COLORS[bin_id], edgecolor="white", label=label)
            left += percentages[:, bin_id]
        ax.set_title(MODEL_LABELS[model])
        ax.set_yticks(np.arange(len(TYPES)), [f"{kind} (n={n})" for kind, n in zip(TYPES, totals)])
        ax.set_xlim(0, 100)
        ax.set_xticks((0, 25, 50, 75, 100))
        ax.grid(axis="x", alpha=0.2)
    axes[0].set_xlabel("Tracks within class-composition type (%)")
    axes[1].set_xlabel("Tracks within class-composition type (%)")
    axes[2].set_xlabel("Tracks within class-composition type (%)")
    fig.legend(ERROR_LABELS, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, frameon=False)
    fig.suptitle("Per-track GT error by class composition", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_ranges(summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col", sharey="row")
    x = np.arange(len(ERROR_LABELS))
    for column, model in enumerate(MODELS):
        bins = summary["models"][model]
        for row, (median_key, low_key, high_key, ylabel) in enumerate((
                ("duration_median_seconds", "duration_q1_seconds", "duration_q3_seconds", "Nominal duration (s)"),
                ("distance_median_m", "distance_q1_m", "distance_q3_m", "Median ped–ego range (m)"))):
            ax = axes[row, column]
            med = np.array([bins[label][median_key] if bins[label][median_key] is not None else np.nan
                            for label in ERROR_LABELS], dtype=float)
            low = np.array([bins[label][low_key] if bins[label][low_key] is not None else np.nan
                            for label in ERROR_LABELS], dtype=float)
            high = np.array([bins[label][high_key] if bins[label][high_key] is not None else np.nan
                             for label in ERROR_LABELS], dtype=float)
            ax.errorbar(x, med, yerr=np.vstack((med - low, high - med)), fmt="o", capsize=3,
                        color=ERROR_COLORS[2], linewidth=1.5)
            for i, label in enumerate(ERROR_LABELS):
                n = bins[label]["track_count"]
                if np.isfinite(med[i]):
                    ax.annotate(f"n={n}", (x[i], med[i]), xytext=(0, 7), textcoords="offset points",
                                ha="center", fontsize=8)
                elif n:
                    ax.annotate(f"n={n}", (x[i], 0), xytext=(0, 7), textcoords="offset points",
                                ha="center", fontsize=8)
            ax.set_title(MODEL_LABELS[model] if row == 0 else "")
            if column == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(x, ERROR_LABELS, rotation=25, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Track duration and median pedestrian–ego range by per-track error bin\n"
                 "Points are medians, bars are Q1–Q3; labels give track counts", y=1.02)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run(metadata_csv=None):
    by_model = read_error_rows(TRACK_ERRORS)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT / "per-track-metadata.csv"
    metadata = read_track_metadata(metadata_csv) if metadata_csv else extract_track_metadata(by_model, cache_path)
    if set(metadata) != set(by_model["road-direct"]):
        raise ValueError("Cached track metadata does not match the 931-track error table")
    summary, rows = build_summary(by_model, metadata)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    write_csv(OUTPUT / "summary.csv", rows)
    plot_composition(summary, OUTPUT / "error-composition.png")
    plot_ranges(summary, OUTPUT / "error-ranges.png")
    print(f"Wrote E004 error-range summary to {OUTPUT}")


def self_check():
    expected = (0, 0, 1, 1, 2, 3, 4, 5)
    counts = Counter(error_bin(errors, 20) for errors in (0, 1, 2, 4, 10, 16, 19, 20))
    assert tuple(error_bin(errors, 20) for errors in (0, 1, 2, 4, 10, 16, 19, 20)) == expected
    assert sum(counts.values()) == 8 and len(counts) == 6
    assert error_bin(1, 20) == error_bin(0, 20)  # 0.05 is right-inclusive in the first bin.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path,
                        help="reuse the saved per-track metadata CSV; skips Aalto archive reads")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("Error-bin boundary and partition checks passed")
    else:
        run(args.metadata_csv)
