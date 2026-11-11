"""Class-conditioned motion and prediction summaries for frozen E002d transfer."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path("outputs/experiments/E002/transfer-validation-20261102")
OUTPUT = Path("outputs/experiments/E002/transfer-statistics")
MANIFEST = Path("experiments/E002-multimodal-reference/transfer-comparison.json")
AGGREGATE = ROOT / "source-validation-metrics.json"
CLASSES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
SOURCES = ("road-waymo", "loki")
SPEED_EDGES = (0.0, 0.2, 0.5, 1.0, 2.0, float("inf"))
SPEED_BINS = ("invalid", "[0, 0.2)", "[0.2, 0.5)", "[0.5, 1)", "[1, 2)", "[2, inf)")
QUANTILE_GRID = np.arange(0.01, 1.0, 0.01)
SLOT_SECONDS = 0.2
EXPECTED = {"road-waymo": (78, 1110, 45388), "loki": (93, 1861, 60231)}


def read_predictions(path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        offsets = saved["offsets"]
        targets, valid, logits = saved["targets"], saved["gt_valid"], saved["logits"]
        records = {}
        for i, item in enumerate(metadata):
            start, stop = offsets[i:i + 2]
            locator = item["locator"]
            if locator in records:
                raise ValueError(f"Duplicate prediction locator: {locator}")
            if stop - start != len(valid[start:stop]) or not valid[start:stop].any():
                raise ValueError(f"Invalid GT population: {locator}")
            records[locator] = {
                "locator": locator,
                "archive": item["archive"],
                "targets": targets[start:stop].copy(),
                "gt_valid": valid[start:stop].copy(),
                "predictions": logits[start:stop].argmax(axis=1),
                "source_times_us": item.get("source_times_us"),
            }
    return records


def check_aligned(reference, candidate, label):
    if reference.keys() != candidate.keys():
        raise ValueError(f"Population differs for {label}")
    for locator, first in reference.items():
        second = candidate[locator]
        if (not np.array_equal(first["targets"], second["targets"])
                or not np.array_equal(first["gt_valid"], second["gt_valid"])
                or first.get("source_times_us") != second.get("source_times_us")):
            raise ValueError(f"GT, mask or timestamp alignment differs for {label}: {locator}")


def attach_track_arrays(records, cache):
    for locator, record in records.items():
        if locator not in cache:
            with np.load(record["archive"], allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
                if metadata["track_locator"] != locator:
                    raise ValueError(f"Track archive locator mismatch: {locator}")
                cache[locator] = {key: saved[key].copy() for key in
                                  ("ped_position", "ped_position_valid", "ped_velocity", "ped_velocity_valid")}
        arrays = cache[locator]
        if len(arrays["ped_velocity"]) != len(record["targets"]):
            raise ValueError(f"Track grid length differs from predictions: {locator}")
        record["ped_speed"] = np.linalg.norm(arrays["ped_velocity"], axis=1)
        record["ped_speed_valid"] = arrays["ped_velocity_valid"].astype(bool) & np.isfinite(record["ped_speed"])
        record["ped_position"] = arrays["ped_position"]
        record["ped_position_valid"] = arrays["ped_position_valid"].astype(bool)


def weighted_quantile(values, weights, probability):
    if not len(values) or weights.sum() <= 0:
        return None
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    index = min(np.searchsorted(np.cumsum(weights), probability * weights.sum(), side="left"), len(values) - 1)
    return float(values[index])


def weighted_cdf(values, weights):
    return {"probability": QUANTILE_GRID.tolist(),
            "value": [weighted_quantile(values, weights, float(q)) for q in QUANTILE_GRID]}


def class_masks(records, class_id):
    return [(record, record["gt_valid"] & (record["targets"] == class_id))
            for record in records.values()]


def frame_weight(record):
    return 1.0 / int(record["gt_valid"].sum())


def get_class_runs(population, class_id):
    """Attach nominal slot-count duration to every accepted GT frame in a run."""
    values, weights = [], []
    raw_frame_count = 0
    for record in population.values():
        gt, valid = record["targets"], record["gt_valid"]
        base = frame_weight(record)
        i = 0
        while i < len(gt):
            if not valid[i] or gt[i] != class_id:
                i += 1
                continue
            end = i + 1
            while end < len(gt) and valid[end] and gt[end] == class_id:
                end += 1
            length = (end - i) * SLOT_SECONDS
            values.extend([length] * (end - i))
            weights.extend([base] * (end - i))
            raw_frame_count += end - i
            i = end
    return np.asarray(values), np.asarray(weights), raw_frame_count


def sustained_speeds(population, class_id):
    values, weights = [], []
    raw_centers, weighted_centers, eligible_tracks = 0, 0.0, set()
    for locator, record in population.items():
        gt = record["gt_valid"] & (record["targets"] == class_id)
        positions_valid = record["ped_position_valid"]
        times = record["source_times_us"]
        if times is None or len(times) != len(gt):
            raise ValueError(f"Missing source timestamps: {locator}")
        time_us = [int(t) if t is not None else None for t in times]
        base = frame_weight(record)
        for center in range(3, len(gt) - 3):
            start, stop = center - 3, center + 4
            if not (gt[start:stop].all() and positions_valid[start:stop].all()):
                continue
            # GT/position validity defines consecutive grid slots; use actual endpoint times.
            if time_us[start] is None or time_us[stop - 1] is None:
                continue
            elapsed_us = int(time_us[stop - 1]) - int(time_us[start])
            if elapsed_us <= 0:
                raise ValueError(f"Nonpositive observed center-window duration: {locator}")
            elapsed = elapsed_us / 1e6
            speed = float(np.linalg.norm(record["ped_position"][stop - 1]
                                         - record["ped_position"][start]) / elapsed)
            if not np.isfinite(speed):
                raise ValueError(f"Nonfinite sustained speed: {locator}")
            values.append(speed)
            weights.append(base)
            raw_centers += 1
            weighted_centers += base
            eligible_tracks.add(locator)
    return np.asarray(values), np.asarray(weights), raw_centers, weighted_centers, len(eligible_tracks)


def speed_bin_index(speed, valid):
    if not valid:
        return 0
    return min(int(np.searchsorted(SPEED_EDGES[1:], speed, side="right")) + 1, len(SPEED_BINS) - 1)


def compute_dataset(dataset, evaluations):
    population = evaluations["own-trained"]
    total_weight = float(sum(record["gt_valid"].sum() * frame_weight(record)
                             for record in population.values()))
    total_frames = int(sum(record["gt_valid"].sum() for record in population.values()))
    total_tracks = len(population)
    groups = set()
    for locator in population:
        value = json.loads(locator)
        group_key = "clip" if dataset == "road-waymo" else "scenario"
        if group_key not in value:
            raise ValueError(f"Missing {group_key} in locator: {locator}")
        groups.add(value[group_key])
    expected_groups, expected_tracks, expected_frames = EXPECTED[dataset]
    if (len(groups), total_tracks, total_frames) != (expected_groups, expected_tracks, expected_frames):
        raise ValueError(f"Unexpected {dataset} validation population: {(len(groups), total_tracks, total_frames)}")
    if not np.isclose(total_weight, total_tracks, rtol=0, atol=1e-7):
        raise ValueError(f"Track weights do not sum to track count for {dataset}")

    class_stats, cdfs, by_speed_bin = {}, {}, {}
    for class_id, class_name in enumerate(CLASSES):
        selections = class_masks(population, class_id)
        class_raw = sum(int(mask.sum()) for _, mask in selections)
        class_tracks = sum(bool(mask.any()) for _, mask in selections)
        class_mass = sum(float(mask.sum()) * frame_weight(record) for record, mask in selections)
        inst_values, inst_weights, valid_raw, valid_mass = [], [], 0, 0.0
        bin_records = [{"raw_frames": 0, "weight_mass": 0.0, "tracks": set(),
                        "own-trained": np.zeros(len(CLASSES)),
                        "opposite-trained": np.zeros(len(CLASSES))} for _ in SPEED_BINS]
        for model_label, records in evaluations.items():
            for locator, record in records.items():
                mask = record["gt_valid"] & (record["targets"] == class_id)
                base = frame_weight(record)
                speed_valid = mask & record["ped_speed_valid"]
                for index in np.flatnonzero(mask):
                    bin_id = speed_bin_index(record["ped_speed"][index], bool(record["ped_speed_valid"][index]))
                    bin_item = bin_records[bin_id]
                    bin_item[model_label][int(record["predictions"][index])] += base
                    if model_label == "own-trained":
                        bin_item["raw_frames"] += 1
                        bin_item["weight_mass"] += base
                        bin_item["tracks"].add(locator)
                if model_label == "own-trained" and speed_valid.any():
                    inst_values.append(record["ped_speed"][speed_valid])
                    inst_weights.append(np.full(int(speed_valid.sum()), base))
                    valid_raw += int(speed_valid.sum())
                    valid_mass += float(speed_valid.sum()) * base
        if class_raw == 0 or class_mass <= 0:
            raise ValueError(f"Empty GT class {class_name} in {dataset}")
        inst_values = np.concatenate(inst_values) if inst_values else np.asarray([])
        inst_weights = np.concatenate(inst_weights) if inst_weights else np.asarray([])
        sustained_values, sustained_weights, sustained_raw, sustained_mass, sustained_tracks = sustained_speeds(
            population, class_id)
        run_values, run_weights, run_raw = get_class_runs(population, class_id)
        if run_raw != class_raw:
            raise ValueError(f"Run-length assignment lost GT frames for {dataset}/{class_name}")
        quantiles = lambda v, w: {"p10": weighted_quantile(v, w, .10),
                                  "p50": weighted_quantile(v, w, .50),
                                  "p90": weighted_quantile(v, w, .90)}
        class_stats[class_name] = {
            "raw_gt_frames": class_raw,
            "tracks_with_class": class_tracks,
            "track_weighted_gt_mass": class_mass,
            "track_weighted_prevalence": class_mass / total_weight,
            "ped_speed": {"valid_raw_frames": valid_raw,
                          "invalid_raw_frames": class_raw - valid_raw,
                          "valid_track_weighted_mass": valid_mass,
                          "invalid_track_weighted_mass": class_mass - valid_mass,
                          "raw_coverage": valid_raw / class_raw,
                          "invalid_raw_fraction": (class_raw - valid_raw) / class_raw,
                          "track_weighted_coverage": valid_mass / class_mass,
                          "invalid_track_weighted_fraction": (class_mass - valid_mass) / class_mass,
                          **quantiles(inst_values, inst_weights),
                          "below_0_2_fraction": (float(inst_weights[inst_values < .2].sum() / inst_weights.sum())
                                                 if len(inst_values) else None)},
            "sustained_net_speed": {"window_seconds": 1.2,
                                    "eligible_raw_centers": sustained_raw,
                                    "eligible_track_weighted_mass": sustained_mass,
                                    "eligible_tracks": sustained_tracks,
                                    "raw_coverage": sustained_raw / class_raw,
                                    "track_weighted_coverage": sustained_mass / class_mass,
                                    **quantiles(sustained_values, sustained_weights),
                                    "below_0_2_fraction": (float(sustained_weights[sustained_values < .2].sum()
                                                                  / sustained_weights.sum())
                                                           if len(sustained_values) else None)},
            "observed_run_length": {"weighting": "GT-timestep-weighted",
                                    "raw_frames": run_raw,
                                    "p50_seconds": weighted_quantile(run_values, run_weights, .50),
                                    "p90_seconds": weighted_quantile(run_values, run_weights, .90)},
        }
        cdfs[class_name] = {
            "ped_speed": weighted_cdf(inst_values, inst_weights),
            "sustained_net_speed": weighted_cdf(sustained_values, sustained_weights),
        }
        by_speed_bin[class_name] = {}
        for bin_name, item in zip(SPEED_BINS, bin_records):
            predictions = {}
            for label in evaluations:
                mass = float(item[label].sum())
                predictions[label] = ([float(value / mass) for value in item[label]] if mass else None)
            by_speed_bin[class_name][bin_name] = {
                "raw_gt_frames": item["raw_frames"],
                "tracks_with_gt": len(item["tracks"]),
                "track_weighted_gt_mass": item["weight_mass"],
                "predicted_class_fraction": predictions,
            }
    return {"groups": len(groups), "tracks": total_tracks, "gt_frames": total_frames,
            "class_statistics": class_stats, "speed_cdf": cdfs,
            "predictions_by_gt_class_and_speed_bin": by_speed_bin}


def compute():
    aggregate = json.loads(AGGREGATE.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("seed") != 0:
        raise ValueError("Expected the frozen seed0 transfer manifest")
    rows = {(row["training_source"], row["input"]): row for row in aggregate["models"]}
    frozen = {(row["training_source"], row["input"]): row for row in manifest["models"]}
    evaluations, source_paths, lineage, cache = {}, {}, {}, {}
    for target in SOURCES:
        source_predictions = {}
        own_row = rows[(target, "geometry+rgb+lidar")]
        own_manifest = frozen[(target, "geometry+rgb+lidar")]
        if (own_manifest["experiment"] != "E002d" or own_manifest["input"] != "geometry+rgb+lidar"
                or own_manifest["rgb_branch_dropout"] != 0.5):
            raise ValueError(f"Unexpected frozen full-fusion model for {target}")
        if own_row["checkpoint_sha256"] != own_manifest["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash differs from frozen manifest: {target}")
        own_path = Path(own_row["metrics_path"]).parent / "predictions.npz"
        source_predictions["own-trained"] = read_predictions(own_path)
        other = "loki" if target == "road-waymo" else "road-waymo"
        transfer_path = ROOT / "cells" / other / "geometry+rgb+lidar" / "predictions.npz"
        source_predictions["opposite-trained"] = read_predictions(transfer_path)
        check_aligned(source_predictions["own-trained"], source_predictions["opposite-trained"],
                      f"{target} own/opposite full-fusion validation")
        attach_track_arrays(source_predictions["own-trained"], cache)
        attach_track_arrays(source_predictions["opposite-trained"], cache)
        evaluations[target] = source_predictions
        source_paths[target] = {"own_source_validation_predictions": str(own_path),
                                "opposite_source_validation_predictions": str(transfer_path),
                                "prepared_archive_roots": sorted({str(Path(r["archive"]).parent)
                                    for records in source_predictions.values() for r in records.values()})}
        lineage[target] = {key: own_manifest[key] for key in
                           ("experiment", "arm", "variant", "rgb_branch_dropout", "selected_epoch",
                            "checkpoint", "checkpoint_sha256", "source_validation")}

    data = {dataset: compute_dataset(dataset, evaluations[dataset]) for dataset in SOURCES}
    return {
        "schema_version": 1,
        "method": {
            "scope": "original E002d full-fusion geometry+RGB+LiDAR, RGB branch dropout 0.5, seed 0; own-source validation and direct opposite-source validation only",
            "training": "saved predictions only; no training, inference, native-data access or cache rebuild",
            "weights": "each accepted GT timestep has weight 1 / total accepted GT timesteps in its track, applied once and retained in every class, speed-bin and run-length slice",
            "ped_speed": "norm of saved unnormalized ped_velocity; ped_velocity_valid plus finite values; invalid values excluded from quantiles/fractions and separately counted for coverage",
            "sustained_speed": "centered seven-slot window; all seven consecutive grid slots have GT labels equal to the class and valid positions; valid actual endpoint timestamps and positive elapsed time; endpoint displacement / actual elapsed time",
            "run_length": "GT-timestep-weighted observed run length: each accepted GT timestep inherits its contiguous same-class run length and original track weight; invalid/unlabeled frames and label changes split runs; nominal run slots × 0.2 s; not true uncensored event duration",
            "speed_bins": list(SPEED_BINS),
            "motion_limit": "velocity and displacement are motion proxies, not independently verified physical truth",
            "prediction_fractions": "within each GT class and physical-speed bin, predicted-class proportions use the original track weights and normalize within that bin; empty bins are unscored",
            "validation_limit": "these are the two frozen validation populations; they do not establish the physical speed or duration range in either training population",
        },
        "population_denominators": {d: {"groups": values["groups"], "tracks": values["tracks"],
                                         "gt_frames": values["gt_frames"]} for d, values in data.items()},
        "source_paths": source_paths,
        "checkpoint_lineage": lineage,
        "datasets": data,
    }


def plot_figures(summary):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    dataset_labels = {"road-waymo": "ROAD-Waymo", "loki": "LOKI"}
    class_labels = {"MOVING": "Moving", "STOPPED": "Stopped",
                    "WAITING_TO_CROSS": "Waiting to cross", "CROSSING": "Crossing"}
    class_colors = {name: plt.get_cmap("tab10")(i) for i, name in enumerate(CLASSES)}

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
    motion_rows = (("ped_speed", "Instantaneous speed"),
                   ("sustained_net_speed", "1.2 s net speed"))
    dataset_colors = {"road-waymo": "#1f77b4", "loki": "#d95f02"}
    for row, (speed_key, row_label) in enumerate(motion_rows):
        for col, name in enumerate(CLASSES):
            ax = axes[row, col]
            for dataset in SOURCES:
                cdf = summary["datasets"][dataset]["speed_cdf"][name][speed_key]
                ax.plot(cdf["value"], cdf["probability"], color=dataset_colors[dataset], linewidth=1.7,
                        label=dataset_labels[dataset])
            ax.set_title(class_labels[name])
            ax.set_xlim(left=0)
            ax.set_ylim(0, 1)
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel(f"{row_label}\nTrack-weighted CDF")
            if row == 1:
                ax.set_xlabel("Speed (m/s)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("Validation motion distributions by GT class", y=1.02)
    fig.tight_layout(rect=(0, .04, 1, .98))
    fig.savefig(figure_dir / "class-motion-cdf.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "class-motion-cdf.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    bin_labels = ("Invalid", "0–0.2", "0.2–0.5", "0.5–1", "1–2", "≥2")
    x = np.arange(len(SPEED_BINS))
    width = 0.36
    for row, dataset in enumerate(SOURCES):
        for col, gt_class in enumerate(CLASSES):
            ax = axes[row, col]
            class_bins = summary["datasets"][dataset]["predictions_by_gt_class_and_speed_bin"][gt_class]
            for model_offset, model_label, hatch in ((-width / 2, "own-trained", ""),
                                                     (width / 2, "opposite-trained", "///")):
                bottoms = np.zeros(len(SPEED_BINS))
                for pred_class in CLASSES:
                    heights = []
                    for bin_name in SPEED_BINS:
                        proportions = class_bins[bin_name]["predicted_class_fraction"][model_label]
                        heights.append(0.0 if proportions is None else proportions[CLASSES.index(pred_class)])
                    bars = ax.bar(x + model_offset, heights, width, bottom=bottoms,
                                  color=class_colors[pred_class], edgecolor="#333333", linewidth=.35,
                                  hatch=hatch)
                    bottoms += np.asarray(heights)
            for i, bin_name in enumerate(SPEED_BINS):
                ax.text(i, 1.025, f"n={class_bins[bin_name]['raw_gt_frames']:,}",
                        ha="center", va="bottom", fontsize=5.5)
                if class_bins[bin_name]["raw_gt_frames"] == 0:
                    ax.text(i, 0.04, "—", ha="center", va="bottom", fontsize=7, color="#555555")
            ax.set_title(f"{dataset_labels[dataset]} · GT {class_labels[gt_class]}")
            ax.set_xticks(x, bin_labels, rotation=35, ha="right")
            ax.set_ylim(0, 1.12)
            ax.grid(axis="y", alpha=.2)
            if col == 0:
                ax.set_ylabel("Predicted-class fraction")
    pred_handles = [Patch(facecolor=class_colors[name], edgecolor="#333333", label=class_labels[name])
                    for name in CLASSES]
    model_handles = [Patch(facecolor="white", edgecolor="#333333", label="Own-source trained"),
                     Patch(facecolor="white", edgecolor="#333333", hatch="///", label="Opposite-source trained")]
    fig.legend(handles=pred_handles + model_handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle("Predictions within GT class and physical-speed bin", y=1.02)
    fig.tight_layout(rect=(0, .08, 1, .98))
    fig.savefig(figure_dir / "predictions-by-speed-bin.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / "predictions-by-speed-bin.svg", bbox_inches="tight")
    plt.close(fig)


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def speed_q(stats):
    if stats["p50"] is None:
        return "—"
    return f"{stats['p10']:.2f}/{stats['p50']:.2f}/{stats['p90']:.2f}"


def report_snippet(summary):
    labels = {"road-waymo": "ROAD-Waymo", "loki": "LOKI"}
    classes = {"MOVING": "Moving", "STOPPED": "Stopped",
               "WAITING_TO_CROSS": "Waiting to cross", "CROSSING": "Crossing"}
    out = ["## Class-conditioned statistics and transfer failures", "",
           "This diagnostic uses the original E002d full-fusion geometry + RGB + LiDAR checkpoints (RGB branch dropout 0.5, seed 0), their own-source validation predictions, and direct opposite-source validation predictions. It uses the frozen ROAD-Waymo validation population (78 groups, 1,110 tracks, 45,388 accepted GT frames) and LOKI validation population (93 groups, 1,861 tracks, 60,231 accepted GT frames). Both validation sets already informed development; these statistics do not establish the physical speed or duration range in either training population.", "",
           "Every accepted GT frame keeps weight `1 / total accepted GT frames in its track` in every subset. Speed fractions are normalized within the valid speed subset; valid-velocity coverage is shown in raw and track-weighted terms, with the complement counted as invalid rather than imputed. The sustained proxy uses centered seven-slot windows with all seven consecutive grid slots labeled as the same GT class and valid positions, plus valid actual endpoint timestamps and positive elapsed time. The instantaneous and sustained quantiles use different eligible populations; their differences should not be attributed to smoothing alone. Sustained-window coverage is restricted to about 49–70%, and the motion tail remains. Run lengths are GT-timestep-weighted observed runs: each accepted GT timestep inherits its contiguous same-class run length; unlabeled frames and label changes split runs, and duration is nominal slots × 0.2 seconds. These observed runs are not true uncensored event durations. Velocity and displacement are proxies, not independently verified physical truth.", "",
           "| Dataset | GT class | Frames / tracks | GT share (weighted) | Valid ped speed coverage (raw / weighted) | Ped speed p10 / p50 / p90 (m/s) | Ped speed <0.2 | 1.2 s coverage (raw / weighted) | 1.2 s net speed p10 / p50 / p90 (m/s) | Net speed <0.2 | GT-timestep-weighted observed run p50 / p90 (s) |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset in SOURCES:
        for name in CLASSES:
            row = summary["datasets"][dataset]["class_statistics"][name]
            ped, sustained, run = row["ped_speed"], row["sustained_net_speed"], row["observed_run_length"]
            out.append(f"| {labels[dataset]} | {classes[name]} | {row['raw_gt_frames']:,} / {row['tracks_with_class']:,} | {pct(row['track_weighted_prevalence'])} | {ped['valid_raw_frames']:,}/{row['raw_gt_frames']:,} ({pct(ped['raw_coverage'])} / {pct(ped['track_weighted_coverage'])}) | {speed_q(ped)} | {pct(ped['below_0_2_fraction'])} | {sustained['eligible_raw_centers']:,}/{row['raw_gt_frames']:,} ({pct(sustained['raw_coverage'])} / {pct(sustained['track_weighted_coverage'])}) | {speed_q(sustained)} | {pct(sustained['below_0_2_fraction'])} | {run['p50_seconds']:.1f} / {run['p90_seconds']:.1f} |")
    crossing_bin = summary["datasets"]["road-waymo"]["predictions_by_gt_class_and_speed_bin"]["CROSSING"]["[1, 2)"]
    own_cross = crossing_bin["predicted_class_fraction"]["own-trained"][3]
    opposite_cross = crossing_bin["predicted_class_fraction"]["opposite-trained"][3]
    opposite_moving = crossing_bin["predicted_class_fraction"]["opposite-trained"][0]
    out.extend(["", f"The clearest matched-speed difference is ROAD-Waymo GT Crossing at 1–2 m/s: across {crossing_bin['raw_gt_frames']:,} frames in {crossing_bin['tracks_with_gt']:,} tracks (track-weighted mass {crossing_bin['track_weighted_gt_mass']:.1f}), the ROAD-trained model predicts Crossing on {100 * own_cross:.1f}% of the weighted GT mass, while the LOKI-trained model predicts Crossing on {100 * opposite_cross:.1f}% and Moving on {100 * opposite_moving:.1f}%. Both summaries use the same target GT class and speed bin, so the prediction difference persists at comparable observed speed; it does not isolate why."])
    out.extend(["", "Figure 1 shows track-weighted empirical CDFs for instantaneous velocity and the sustained net-displacement proxy. Figure 2 compares the own-source-trained and opposite-source-trained full-fusion model predictions within each GT-class and physical-speed bin. The shared bins are invalid, [0, 0.2), [0.2, 0.5), [0.5, 1), [1, 2), and [2, inf) m/s; empty bins are unscored. Percentages are predicted-class shares within each GT class/bin, with the original frame weights retained.", "", "![Class-conditioned instantaneous and sustained motion distributions](../../outputs/experiments/E002/transfer-statistics/figures/class-motion-cdf.png)", "", "[SVG](../../outputs/experiments/E002/transfer-statistics/figures/class-motion-cdf.svg).", "", "![Predicted classes by GT class and speed bin](../../outputs/experiments/E002/transfer-statistics/figures/predictions-by-speed-bin.png)", "", "[SVG](../../outputs/experiments/E002/transfer-statistics/figures/predictions-by-speed-bin.svg).", "", "The paired speed-bin predictions show how errors change between own-source and opposite-source training at comparable observed speed, but speed is only one measured correlate and does not resolve annotation semantics or scene context. Overlapping sustained windows are not independent examples. Invalid velocity and position reduce eligible coverage, so those summaries do not describe the full GT population.", "", "Reproduce on Aalto from the repository root of the `pedestrian-behavior-labeling-e002b-evaluation` checkout with the saved validation predictions and prepared track archives:", "", "```bash", "MPLCONFIGDIR=/tmp/mpl-e002-transfer-statistics ~/.local/bin/uv run --locked --extra cpu python scripts/analyze-e002-transfer-statistics.py", "```", "", "The script checks source checkpoint lineage against [transfer-comparison.json](transfer-comparison.json), prediction population/GT/timestamp alignment and frozen denominators. It reads only saved E002d predictions and referenced prepared track arrays; no inference or training is run. Numerical data, paths and quantile CDF points are saved in `outputs/experiments/E002/transfer-statistics/summary.json`."])
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", type=Path, help="render report artifacts from a saved numerical summary")
    parser.add_argument("--compute-only", action="store_true", help="write the summary without rendering figures")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(args.plot_only.read_text()) if args.plot_only else compute()
    if not args.plot_only:
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2, allow_nan=False))
    if not args.compute_only:
        plot_figures(summary)
    (OUTPUT / "summary-section.md").write_text(report_snippet(summary))


if __name__ == "__main__":
    main()
