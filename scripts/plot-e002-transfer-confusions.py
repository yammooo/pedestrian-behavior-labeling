#!/usr/bin/env python3
"""Rebuild E002 source/target validation confusion-matrix grids from saved metrics."""

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("Moving", "Stopped", "Waiting to cross", "Crossing")
INPUTS = ("geometry", "geometry+rgb", "geometry+lidar", "geometry+rgb+lidar")
LABELS = {"geometry": "Geometry", "geometry+rgb": "Geometry + RGB",
    "geometry+lidar": "Geometry + LiDAR", "geometry+rgb+lidar": "Geometry + RGB + LiDAR"}
DIRECTIONS = (("road-waymo", "loki"), ("loki", "road-waymo"))
TARGET_TRACKS = {"road-waymo": 1110, "loki": 1861}
DIRECTION_LABELS = {"road-waymo": "ROAD-Waymo", "loki": "LOKI"}
CLASS_LABELS = ("Moving", "Stopped", "Waiting to cross", "Crossing")


def _close(actual, expected, label, *, atol=1e-10):
    if not np.isclose(actual, expected, rtol=1e-10, atol=atol):
        raise ValueError(f"{label}: {actual} != {expected}")


def _validate_metrics(metrics, *, training_source, target_source, input_name, expected_tracks, checkpoint_sha=None):
    if metrics.get("split") != "validation":
        raise ValueError("Confusion figures only accept validation metrics")
    if (metrics.get("training_source") != training_source
            or metrics.get("target_source") not in (None, target_source)
            or metrics.get("input") not in (None, input_name)):
        raise ValueError("Metric training/target/input identity mismatch")
    if checkpoint_sha is not None and metrics.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Metric checkpoint identity differs from frozen comparison manifest")
    if metrics.get("dataset_tracks") != expected_tracks or metrics.get("prediction_tracks") != expected_tracks:
        raise ValueError("Validation metrics do not cover the expected full track population")
    primary = metrics.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("Missing global primary validation metrics")
    denominators = primary.get("denominators", {})
    if denominators.get("tracks") != expected_tracks:
        raise ValueError("Confusion matrix track denominator mismatch")
    class_frames = np.asarray(denominators.get("class_frames"), dtype=float)
    if class_frames.shape != (4,) or not np.isfinite(class_frames).all() or np.any(class_frames < 0):
        raise ValueError("Expected four finite nonnegative class-frame denominators")
    result = {}
    for weighting in ("raw", "track_weighted"):
        record = primary.get(weighting)
        if not isinstance(record, dict):
            raise ValueError(f"Missing {weighting} confusion metric")
        matrix = np.asarray(record.get("confusion"), dtype=float)
        support = np.asarray(record.get("support"), dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError(f"{weighting} confusion matrix must be finite, nonnegative and 4×4")
        if support.shape != (4,) or not np.isfinite(support).all() or np.any(support < 0):
            raise ValueError(f"{weighting} support must contain four finite nonnegative values")
        _close(matrix.sum(), expected_tracks if weighting == "track_weighted" else denominators["gt_frames"],
            f"{weighting} confusion total", atol=1e-6)
        if not np.allclose(matrix.sum(axis=1), support, rtol=1e-9, atol=1e-7):
            raise ValueError(f"{weighting} confusion row sums differ from reported support")
        if weighting == "raw" and not np.allclose(support, class_frames, rtol=0, atol=1e-6):
            raise ValueError("Raw confusion rows differ from class-frame denominators")
        predicted_support = matrix.sum(axis=0)
        precision = np.divide(np.diag(matrix), predicted_support, out=np.zeros(4), where=predicted_support > 0)
        recall = np.divide(np.diag(matrix), support, out=np.zeros(4), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
            out=np.zeros(4), where=precision + recall > 0)
        for metric, actual in (("precision", precision), ("recall", recall), ("f1", f1)):
            reported = np.asarray(record.get(metric), dtype=float)
            if reported.shape != (4,) or not np.isfinite(reported).all() or not np.allclose(
                    reported, actual, rtol=1e-10, atol=1e-10):
                raise ValueError(f"{weighting} {metric} differs from confusion-matrix calculation")
        _close(record["macro_f1"], float(f1.mean()),
            f"{weighting} macro-F1 from per-class F1")
        normalized = np.divide(matrix, support[:, None], out=np.zeros_like(matrix), where=support[:, None] > 0)
        result[weighting] = {"matrix": matrix, "support": support,
            "normalized": normalized * 100., "class_frames": class_frames,
            "precision": precision, "recall": recall, "f1": f1,
            "macro_f1": float(record["macro_f1"]), "gt_frames": int(denominators["gt_frames"]),
            "tracks": expected_tracks}
    return result


def _load_cells(comparison_path, source_metrics_path, target_root):
    manifest = json.loads(Path(comparison_path).read_text())
    source = json.loads(Path(source_metrics_path).read_text())
    if (manifest.get("status") != "frozen-before-cross-dataset-validation-inference"
            or manifest.get("evaluation", {}).get("held_out_tests_accessed") is not False
            or source.get("comparison") != manifest.get("comparison")):
        raise ValueError("Frozen E002 comparison and source-metric aggregate are required")
    expected = {(source_name, input_name) for source_name, _ in DIRECTIONS for input_name in INPUTS}
    manifest_rows = {(row["training_source"], row["input"]): row for row in manifest.get("models", [])}
    source_rows = {(row["training_source"], row["input"]): row for row in source.get("models", [])}
    if set(manifest_rows) != expected or set(source_rows) != expected or len(manifest.get("models", [])) != 8:
        raise ValueError("Expected exactly eight frozen source/input model cells")
    cells = {}
    for training_source, target_source in DIRECTIONS:
        for input_name in INPUTS:
            key = training_source, input_name
            model = manifest_rows[key]
            source_row = source_rows[key]
            checkpoint_sha = model["checkpoint_sha256"]
            if (source_row.get("checkpoint") != model.get("checkpoint")
                    or source_row.get("checkpoint_sha256") != checkpoint_sha):
                raise ValueError(f"Source metric checkpoint lineage mismatch: {key}")
            expected_source = model["source_validation"]
            source_f1 = source_row["primary"]["track_weighted"]["macro_f1"]
            source_ce = source_row["primary"]["track_ce"]
            _close(source_f1, expected_source["track_weighted_macro_f1"], f"Source F1 mismatch {key}")
            _close(source_ce, expected_source["track_ce"], f"Source CE mismatch {key}")
            source_metrics = {"split": "validation", "training_source": training_source,
                "target_source": training_source, "input": input_name,
                "checkpoint_sha256": checkpoint_sha,
                "dataset_tracks": TARGET_TRACKS[training_source], "prediction_tracks": TARGET_TRACKS[training_source],
                "primary": source_row["primary"]}
            source_validated = _validate_metrics(source_metrics, training_source=training_source,
                target_source=training_source, input_name=input_name,
                expected_tracks=TARGET_TRACKS[training_source], checkpoint_sha=checkpoint_sha)
            target_path = Path(target_root) / "cells" / training_source / input_name / "metrics.json"
            target_metrics = json.loads(target_path.read_text())
            target_validated = _validate_metrics(target_metrics, training_source=training_source,
                target_source=target_source, input_name=input_name,
                expected_tracks=TARGET_TRACKS[target_source], checkpoint_sha=checkpoint_sha)
            if target_metrics.get("normalization_source") != training_source:
                raise ValueError(f"Target metric normalization is not from its source: {key}")
            if target_metrics.get("selected_epoch") != model.get("selected_epoch"):
                raise ValueError(f"Target metric selected epoch differs from frozen manifest: {key}")
            cells[key] = {"manifest": model, "source": source_validated, "target": target_validated}
    return cells


def per_class_table(cells, source, target, weighting):
    rows = []
    for input_name in INPUTS:
        metrics = cells[source, input_name]
        source_metrics, target_metrics = metrics["source"][weighting], metrics["target"][weighting]
        for index, label in enumerate(CLASS_LABELS):
            values = (source_metrics["precision"][index], source_metrics["recall"][index],
                      source_metrics["f1"][index], target_metrics["precision"][index],
                      target_metrics["recall"][index], target_metrics["f1"][index])
            rows.append(f"| {LABELS[input_name]} | {label} | "
                        + " | ".join(f"{value * 100:.2f}" for value in values) + " |")
    heading = "| Inputs | Class | Source P (%) | Source R (%) | Source F1 (%) | Target P (%) | Target R (%) | Target F1 (%) |\n"
    separator = "|---|---|---:|---:|---:|---:|---:|---:|\n"
    return heading + separator + "\n".join(rows) + "\n"


def per_class_section(cells):
    parts = [
        "## Per-class precision, recall and F1",
        "",
        "These are global per-class metrics computed from each saved confusion matrix: precision uses the predicted-class column sum; recall uses the GT-class row sum; F1 is their harmonic mean. Source columns are the checkpoint's own-source validation results; target columns are direct opposite-source validation before adaptation. The main tables use the same track-weighted confusion matrices as the transfer summary. Metrics are percentages; class order matches the confusion figures.",
        "",
    ]
    for source, target in DIRECTIONS:
        parts.extend([f"### {DIRECTION_LABELS[source]} → {DIRECTION_LABELS[target]}", "", per_class_table(cells, source, target, "track_weighted")])
    parts.extend(["<details>", "<summary>Pooled-frame per-class metrics</summary>", "",
        "These use raw accepted GT frames pooled across tracks rather than equal total mass per track.", ""])
    for source, target in DIRECTIONS:
        parts.extend([f"### {DIRECTION_LABELS[source]} → {DIRECTION_LABELS[target]}", "",
                      per_class_table(cells, source, target, "raw")])
    parts.extend(["</details>", ""])
    return "\n".join(parts)


def _figure(cells, source, target, weighting, view, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    rows = [[cells[source, input_name][dataset][weighting] for dataset in ("source", "target")]
        for input_name in INPUTS]
    matrices = [row_item[view if view != "count" else "matrix"] for row in rows for row_item in row]
    vmax = 100. if view == "normalized" else max(float(matrix.max()) for matrix in matrices)
    norm = Normalize(vmin=0., vmax=vmax if vmax > 0 else 1.)
    cmap = "Blues"
    figure, axes = plt.subplots(4, 2, figsize=(11, 17))
    figure.subplots_adjust(left=.17, right=.88, top=.91, bottom=.07, wspace=.55, hspace=.30)
    figure.suptitle(f"{DIRECTION_LABELS[source]} → {DIRECTION_LABELS[target]} · "
        f"{weighting.replace('_', '-')} {view} confusion\n"
        "Rows are ground truth; columns are predictions · Target predictions are before adaptation",
        fontsize=12, y=.985)
    image = None
    for row_index, input_name in enumerate(INPUTS):
        for column, dataset in enumerate(("source", "target")):
            item = cells[source, input_name][dataset][weighting]
            matrix = item["normalized"] if view == "normalized" else item["matrix"]
            ax = axes[row_index, column]
            image = ax.imshow(matrix, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
            ax.set_xticks(range(4), labels=CLASSES if row_index == 3 else ("", "", "", ""),
                rotation=25, ha="right")
            ax.tick_params(axis="x", labelsize=9)
            ylabels = []
            for class_index, name in enumerate(CLASSES):
                frame_count = int(item["class_frames"][class_index])
                if weighting == "track_weighted":
                    ylabels.append(f"{name}\nGT={frame_count:,}; w={item['support'][class_index]:.1f}")
                else:
                    ylabels.append(f"{name}\nGT={frame_count:,}")
            ax.set_yticks(range(4), labels=ylabels)
            score = 100 * item["macro_f1"]
            panel = (f"Source val ({DIRECTION_LABELS[source]})" if dataset == "source" else
                f"Target val ({DIRECTION_LABELS[target]})")
            ax.set_title(f"{LABELS[input_name]}\n{panel} · F1 {score:.2f}%", fontsize=9, pad=5)
            ax.tick_params(axis="y", labelsize=9)
            if row_index == 3:
                ax.set_xlabel("Predicted class")
            for (i, j), value in np.ndenumerate(matrix):
                label = f"{value:.1f}%" if view == "normalized" else (f"{value:.0f}" if weighting == "raw" else f"{value:.1f}")
                ax.text(j, i, label, ha="center", va="center", fontsize=9,
                    color="white" if value > vmax * .57 else "black")
    color_label = "Row-normalized GT (%)" if view == "normalized" else (
        "GT frames" if weighting == "raw" else "Track-weighted GT mass")
    colorbar_axes = figure.add_axes([.92, .15, .025, .70])
    figure.colorbar(image, cax=colorbar_axes, label=color_label,
        ticks=[0, 25, 50, 75, 100] if view == "normalized" else None)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        figure.savefig(output.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight", pad_inches=.14)
    plt.close(figure)
    return output.with_suffix(".png")


def generate(cells, output_dir):
    paths = []
    for source, target in DIRECTIONS:
        for weighting in ("raw", "track_weighted"):
            for view in ("count", "normalized"):
                name = f"{source}-to-{target}-{weighting.replace('_', '-')}-{view}"
                paths.append(_figure(cells, source, target, weighting, view, Path(output_dir) / name))
    if len(paths) != 8 or any(not path.is_file() or not path.with_suffix(".svg").is_file() for path in paths):
        raise ValueError("Expected eight PNG/SVG source-target confusion grids")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path,
        default=ROOT / "experiments/E002-multimodal-reference/transfer-comparison.json")
    parser.add_argument("--outputs-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--source-metrics", type=Path,
        default=ROOT / "outputs/experiments/E002/transfer-validation-20261102/source-validation-metrics.json")
    parser.add_argument("--output", type=Path,
        default=ROOT / "outputs/experiments/E002/transfer-validation-20261102/figures/confusions")
    args = parser.parse_args()
    target_root = args.outputs_root.resolve() / "experiments/E002/transfer-validation-20261102"
    cells = _load_cells(args.comparison, args.source_metrics, target_root)
    paths = generate(cells, args.output)
    table_path = args.output.parent / "per-class-metrics.md"
    table_path.write_text(per_class_section(cells))
    print(f"Wrote {len(paths)} source/target confusion grids to {args.output.resolve()} and per-class tables to {table_path.resolve()}")


if __name__ == "__main__":
    main()
