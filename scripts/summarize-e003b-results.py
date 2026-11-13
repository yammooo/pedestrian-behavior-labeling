"""Summarize copied E003b JSON evidence and write compact review figures."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs/experiments/E003b/results-review-20261104"
EVIDENCE = REVIEW / "evidence/E003b/runs/20261104-092202"
FIGURES = REVIEW / "figures"
CASES = (
    ("road-waymo", "loki", "e002d"),
    ("road-waymo", "loki", "e003a"),
    ("loki", "road-waymo", "e002d"),
    ("loki", "road-waymo", "e003a"),
)
POPULATIONS = {"loki": {"training": 8574, "validation": 1861},
               "road-waymo": {"training": 4380, "validation": 1110}}
SOURCE_EPOCHS = {("road-waymo", "e002d"): 17, ("road-waymo", "e003a"): 24,
                 ("loki", "e002d"): 12, ("loki", "e003a"): 12}
DIRECT_F1 = {
    ("road-waymo", "loki", "e002d"): 0.46496012823104105,
    ("road-waymo", "loki", "e003a"): 0.40259446938008453,
    ("loki", "road-waymo", "e002d"): 0.4502332861111214,
    ("loki", "road-waymo", "e003a"): 0.45277970507727267,
}
CLASSES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
CLASS_LABELS = ("Moving", "Stopped", "Waiting to cross", "Crossing")
E002D_EXPECTED_F1 = {
    ("road-waymo", "loki"): (0.6607025742847996, 0.46496012823104105,
                              0.6415806253538071, 0.7467957180106706),
    ("loki", "road-waymo"): (0.7467957180106706, 0.4502332861111214,
                              0.5948823481676118, 0.6607025742847996),
}


def dataset_name(name):
    return "ROAD" if name == "road-waymo" else "LOKI"


def read(path):
    return json.loads(path.read_text())


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12), (actual, expected)


def _confusion_metrics(record, expected_tracks):
    matrix = record["confusion"]
    support = record["support"]
    assert len(matrix) == len(support) == len(CLASSES)
    assert all(len(row) == len(CLASSES) for row in matrix)
    assert all(math.isfinite(value) and value >= 0 for row in matrix for value in row)
    for index, value in enumerate(support):
        close(sum(matrix[index]), value)
    close(sum(support), expected_tracks)
    column_support = [sum(matrix[row][column] for row in range(4)) for column in range(4)]
    precision = [matrix[i][i] / column_support[i] if column_support[i] else 0.0 for i in range(4)]
    recall = [matrix[i][i] / support[i] if support[i] else 0.0 for i in range(4)]
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
    for key, values in (("precision", precision), ("recall", recall), ("f1", f1)):
        assert len(record[key]) == 4
        for actual, expected in zip(record[key], values):
            close(actual, expected)
    macro_f1 = sum(f1) / 4
    close(record["macro_f1"], macro_f1)
    return {
        "matrix": matrix,
        "support": support,
        "normalized": [[100 * value / support[i] if support[i] else 0.0
                        for value in row] for i, row in enumerate(matrix)],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }


def _load_e002d_panels(source, target):
    def run_data(source_name, target_name):
        run = EVIDENCE / f"e002d-{source_name}-to-{target_name}"
        return read(run / "config.json"), read(run / "validation-comparison.json")

    source_config, source_comparison = run_data(source, target)
    target_config, _ = run_data(target, source)
    source_metrics = source_config["source_checkpoint"]["validation_metrics"]
    target_metrics = target_config["source_checkpoint"]["validation_metrics"]
    assert source_metrics["denominators"]["tracks"] == POPULATIONS[source]["validation"]
    assert target_metrics["denominators"]["tracks"] == POPULATIONS[target]["validation"]
    assert source_metrics["track_weighted"] == source_comparison["source_checkpoint"]["validation_metrics"]["track_weighted"]

    expected_source_tracks = POPULATIONS[source]["validation"]
    expected_target_tracks = POPULATIONS[target]["validation"]
    target_denominators = (source_comparison["source_head"]["denominators"],
                           source_comparison["adapted_head"]["denominators"],
                           target_metrics["denominators"])
    for key in ("groups", "tracks", "gt_frames", "class_frames", "class_tracks"):
        assert target_denominators[0][key] == target_denominators[1][key] == target_denominators[2][key]
    assert target_denominators[0]["tracks"] == expected_target_tracks
    panels = [
        (f"Source model · source validation\n{dataset_name(source)}", source_metrics["track_weighted"],
         expected_source_tracks),
        (f"Direct transfer · target validation\n{dataset_name(target)}", source_comparison["source_head"]["track_weighted"],
         expected_target_tracks),
        (f"E003b frozen body + fresh target head\n{dataset_name(target)} target validation",
         source_comparison["adapted_head"]["track_weighted"], expected_target_tracks),
        (f"Same-family target-trained model\n{dataset_name(target)} validation", target_metrics["track_weighted"],
         expected_target_tracks),
    ]
    validated = [{"label": label, **_confusion_metrics(record, expected_tracks)}
                 for label, record, expected_tracks in panels]

    for panel in validated[1:]:
        for actual, expected in zip(panel["support"], validated[1]["support"]):
            close(actual, expected)
    expected_f1 = E002D_EXPECTED_F1[source, target]
    for panel, expected in zip(validated, expected_f1):
        close(panel["macro_f1"], expected)
    return validated


def _plot_e002d_confusions(panels, source, target):
    from matplotlib.colors import Normalize

    direction = f"e002d-{source}-to-{target}"
    for view in ("normalized", "matrix"):
        normalized = view == "normalized"
        vmax = 100 if normalized else max(max(max(row) for row in panel[view]) for panel in panels)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9.2), constrained_layout=True)
        fig.suptitle(f"E002d {dataset_name(source)} → {dataset_name(target)} track-weighted "
                     f"{'row-normalized confusions' if normalized else 'confusion mass'}", fontsize=14)
        for axis, panel in zip(axes.flat, panels):
            matrix = panel[view]
            axis.imshow(matrix, cmap="Blues", norm=Normalize(0, vmax if vmax else 1), interpolation="nearest")
            axis.set_xticks(range(4), CLASS_LABELS, rotation=25, ha="right")
            axis.set_yticks(range(4), [f"{name}\nGT {mass:.1f}" for name, mass in zip(CLASS_LABELS, panel["support"])])
            axis.set_title(f"{panel['label']} · F1 {100 * panel['macro_f1']:.2f}%", fontsize=9)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("Ground truth · weighted support")
            for i, row in enumerate(matrix):
                for j, value in enumerate(row):
                    label = f"{value:.1f}%" if normalized else f"{value:.1f}"
                    axis.text(j, i, label, ha="center", va="center", fontsize=8,
                              color="white" if value > vmax * (.57 if not normalized else .57) else "black")
        colorbar = fig.colorbar(axes.flat[0].images[0], ax=list(axes.flat),
                                ticks=(0, 25, 50, 75, 100) if normalized else None,
                                shrink=0.86, pad=0.02)
        colorbar.set_label("Ground-truth row share (%)" if normalized else "Track-weighted ground-truth mass")
        suffix = "normalized" if normalized else "counts"
        if normalized:
            fig.savefig(FIGURES / f"{direction}-confusion-{suffix}.png", dpi=180, bbox_inches="tight")
        fig.savefig(FIGURES / f"{direction}-confusion-{suffix}.svg", bbox_inches="tight")
        plt.close(fig)


def _write_e002d_class_metrics(directions):
    lines = [
        "# E002d per-class track-weighted metrics",
        "",
        "Precision, recall and F1 are recomputed from the saved track-weighted confusion matrices. Values are percentages.",
        "",
    ]
    for source, target, panels in directions:
        lines.extend([f"## {dataset_name(source)} → {dataset_name(target)}", "",
                      "| Panel | Class | Precision | Recall | F1 |", "|---|---|---:|---:|---:|"])
        for panel in panels:
            for index, label in enumerate(CLASS_LABELS):
                lines.append(f"| {panel['label'].replace(chr(10), ' ')} | {label} | "
                             f"{100 * panel['precision'][index]:.2f} | "
                             f"{100 * panel['recall'][index]:.2f} | {100 * panel['f1'][index]:.2f} |")
        lines.append("")
    (FIGURES / "e002d-per-class-metrics.md").write_text("\n".join(lines))


def load_case(source, target, reference):
    name = f"{reference}-{source}-to-{target}"
    run = EVIDENCE / name
    status = read(run / "status.json")
    config = read(run / "config.json")
    provenance = read(run / "provenance.json")
    history = read(run / "history.json")
    comparison = read(run / "validation-comparison.json")
    target_metrics = read(run / f"{target}-validation/metrics.json")

    assert status == {"status": "complete", "failures": [], "kind": "comparison"}
    assert config["experiment"] == "E003b" and config["kind"] == "comparison"
    assert (config["original_source"], config["target"], config["reference"]) == (source, target, reference)
    assert config["target_normalization_source"] == source and config["held_out_tests"] is False
    assert config["settings"]["seed"] == provenance["seed"] == 0
    assert config["settings"]["epochs"] == 100 and config["smoke_limits"] is None
    assert config["head"] == {"shape": [128, 4], "parameters": 516}
    assert config["target_population"] == POPULATIONS[target]
    assert set(config["feature_cache"]) == {"training", "validation"}

    checkpoint = config["source_checkpoint"]
    expected_path = (f"fusion-20261030-140937-{source}-rgbdrop50/best.pt" if reference == "e002d"
                     else f"E003a/runs/20261103-105614/{source}/best.pt")
    assert expected_path in checkpoint["path"]
    assert checkpoint["selected_epoch"] == SOURCE_EPOCHS[source, reference]
    assert comparison["source_checkpoint"]["sha256"] == checkpoint["sha256"]
    assert comparison["source_checkpoint"]["selected_epoch"] == checkpoint["selected_epoch"]
    source_metrics = checkpoint["validation_metrics"]
    assert source_metrics == comparison["source_checkpoint"]["validation_metrics"]
    assert source_metrics["denominators"]["tracks"] == POPULATIONS[source]["validation"]
    close(source_metrics["track_weighted"]["macro_f1"],
          sum(source_metrics["track_weighted"]["f1"]) / len(CLASSES))
    for split, cache in config["feature_cache"].items():
        assert cache["split"] == split and cache["source"] == source and cache["target"] == target
        assert cache["tracks"] == config["target_population"][split]
        assert cache["checkpoint_sha256"] == checkpoint["sha256"]
        assert cache["source_normalization"] == config["normalization"]

    assert comparison["scope"] == "target validation only; no test access"
    assert (comparison["training_source"], comparison["target"], comparison["reference"]) == (source, target, reference)
    assert comparison["tracks"] == POPULATIONS[target]["validation"]
    direct, adapted = comparison["source_head"], comparison["adapted_head"]
    assert target_metrics["primary"] == adapted
    assert direct["denominators"] == adapted["denominators"]
    assert direct["denominators"]["tracks"] == POPULATIONS[target]["validation"]
    assert sum(direct["denominators"]["class_frames"]) == direct["denominators"]["gt_frames"]
    close(direct["track_weighted"]["macro_f1"], DIRECT_F1[source, target, reference])

    selection = max(history, key=lambda row: (row["validation_f1"], -row["validation_ce"], -row["epoch"]))
    assert len(history) == provenance["completed_epochs"] == provenance["last_completed_epoch"]
    assert history[-1]["epoch"] == provenance["last_completed_epoch"]
    assert history[-1]["stopping_reason"] == provenance["stopping_reason"]
    assert selection["epoch"] == provenance["selected_checkpoint_epoch"] == provenance["selection"]["epoch"]
    close(selection["validation_f1"], provenance["selection"]["f1"])
    close(selection["validation_ce"], provenance["selection"]["ce"])
    if provenance["stopping_reason"] == "patience":
        assert history[-1]["non_improving_epochs"] == config["settings"]["patience"]
    else:
        assert provenance["stopping_reason"] == "budget" and provenance["last_completed_epoch"] == 100

    close(adapted["track_weighted"]["macro_f1"],
          provenance["selection"]["f1"])
    close(adapted["track_ce"], provenance["selection"]["ce"])
    close(comparison["delta_track_weighted_macro_f1"],
          adapted["track_weighted"]["macro_f1"] - direct["track_weighted"]["macro_f1"])
    close(comparison["delta_track_weighted_macro_f1_pp"],
          100 * comparison["delta_track_weighted_macro_f1"])
    return {
        "source": source,
        "target": target,
        "reference": reference,
        "validation_tracks": comparison["tracks"],
        "source_validation_tracks": source_metrics["denominators"]["tracks"],
        "source_macro_f1": source_metrics["track_weighted"]["macro_f1"],
        "source_ce": source_metrics["track_ce"],
        "direct_macro_f1": direct["track_weighted"]["macro_f1"],
        "adapted_macro_f1": adapted["track_weighted"]["macro_f1"],
        "delta_macro_f1_pp": comparison["delta_track_weighted_macro_f1_pp"],
        "direct_ce": direct["track_ce"],
        "adapted_ce": adapted["track_ce"],
        "direct_class_f1": dict(zip(CLASSES, direct["track_weighted"]["f1"])),
        "adapted_class_f1": dict(zip(CLASSES, adapted["track_weighted"]["f1"])),
        "selected_epoch": selection["epoch"],
        "last_epoch": provenance["last_completed_epoch"],
        "stopping_reason": provenance["stopping_reason"],
        "elapsed_seconds": provenance["elapsed_seconds"],
        "train_f1": [row["training_f1"] for row in history],
        "validation_f1": [row["validation_f1"] for row in history],
        "train_ce": [row["training_ce"] for row in history],
        "validation_ce": [row["validation_ce"] for row in history],
        "checkpoint_sha256": checkpoint["sha256"],
        "source_normalization": config["normalization"],
    }


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = [load_case(*case) for case in CASES]
    e002d_directions = [(source, target, _load_e002d_panels(source, target))
                        for source, target in (("road-waymo", "loki"), ("loki", "road-waymo"))]
    for row in rows:
        reference = next(other for other in rows
                         if other["source"] == row["target"] and other["reference"] == row["reference"])
        assert reference["source_validation_tracks"] == row["validation_tracks"]
        row["target_trained_macro_f1"] = reference["source_macro_f1"]
        row["target_trained_ce"] = reference["source_ce"]
        row["remaining_f1_gap_pp"] = 100 * (row["target_trained_macro_f1"] - row["adapted_macro_f1"])
    summary = {
        "evidence_revision": read(EVIDENCE / f"e002d-road-waymo-to-loki/provenance.json")["revision"],
        "scope": "seed0, full-target-label head adaptation; target validation only; no test access",
        "cases": [{key: value for key, value in row.items()
                   if key not in {"train_f1", "validation_f1", "train_ce", "validation_ce", "source_normalization"}}
                  for row in rows],
    }
    assert all(read(EVIDENCE / f"{ref}-{source}-to-{target}/provenance.json")["revision"]
               == summary["evidence_revision"] for source, target, ref in CASES)
    (REVIEW / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    labels = [f"{dataset_name(row['source'])}→{dataset_name(row['target'])}\n{row['reference'].upper()}"
              for row in rows]
    x = list(range(len(rows)))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for axis, metric, direct_key, adapted_key, ylabel, title in (
        (axes[0], "f1", "direct_macro_f1", "adapted_macro_f1", "track-weighted macro-F1 (%)",
         "Target-validation F1"),
        (axes[1], "ce", "direct_ce", "adapted_ce", "mean track cross-entropy",
         "Target-validation CE"),
    ):
        for offset, (label, key, color) in enumerate((
            ("Same-family target-trained model", "target_trained_macro_f1" if metric == "f1" else "target_trained_ce", "#8c8c8c"),
            ("Direct target validation", direct_key, "#4c78a8"),
            ("Adapted target validation", adapted_key, "#f58518"),
        )):
            scale = 100 if metric == "f1" else 1
            values = [scale * row[key] for row in rows]
            bars = axis.bar([i + (offset - 1) * width for i in x], values, width,
                            label=label, color=color)
            for bar, value in zip(bars, values):
                label_text = f"{value:.1f}" if metric == "f1" else f"{value:.3f}"
                axis.text(bar.get_x() + bar.get_width() / 2, value + (1.0 if metric == "f1" else 0.025),
                          label_text, ha="center", va="bottom", fontsize=8)
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(0, 1.55)
    axes[0].legend(fontsize=8)
    fig.suptitle("Direct transfer and head adaptation versus target-trained references", fontsize=13)
    fig.text(0.5, -0.04,
             "All bars evaluate the same target-validation population. Gray trains the task network on the target; orange fits only its head.\n"
             "Target-trained references use target normalization; adapted models retain source normalization. Direct transfer uses no target labels.",
             ha="center", fontsize=8)
    fig.savefig(FIGURES / "head-adaptation-comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "head-adaptation-comparison.svg", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col", constrained_layout=True)
    for column, target in enumerate(("loki", "road-waymo")):
        selected_rows = [row for row in rows if row["target"] == target]
        for row in selected_rows:
            color = "#4c78a8" if row["reference"] == "e002d" else "#f58518"
            label = row["reference"].upper()
            epochs = range(1, row["last_epoch"] + 1)
            for axis, metric, train_key, val_key, scale in (
                (axes[0, column], "ce", "train_ce", "validation_ce", 1),
                (axes[1, column], "f1", "train_f1", "validation_f1", 100),
            ):
                axis.plot(epochs, [scale * value for value in row[train_key]], "--",
                          color=color, alpha=0.9, label=f"{label} online train")
                axis.plot(epochs, [scale * value for value in row[val_key]], "-",
                          color=color, label=f"{label} validation")
                selected_value = (row["validation_ce"][row["selected_epoch"] - 1] if metric == "ce"
                                  else row["validation_f1"][row["selected_epoch"] - 1])
                axis.scatter([row["selected_epoch"]], [scale * selected_value], color=color,
                             edgecolor="black", zorder=4)
        source = "road-waymo" if target == "loki" else "loki"
        axes[0, column].set_title(f"{dataset_name(source)}→{dataset_name(target)}")
        axes[0, column].set_ylabel("mean track cross-entropy")
        axes[1, column].set_ylabel("track-weighted macro-F1 (%)")
        axes[1, column].set_ylim(0, 100)
        axes[1, column].set_xlabel("epoch")
        for axis in axes[:, column]:
            axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Seed0 target-head training and validation curves", fontsize=13)
    fig.text(0.5, -0.005,
             "Online training uses changing head weights; source body stays in evaluation mode. Dots mark selected epochs.",
             ha="center", fontsize=8)
    fig.savefig(FIGURES / "head-training-validation-curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "head-training-validation-curves.svg", bbox_inches="tight")
    plt.close(fig)

    for source, target, panels in e002d_directions:
        _plot_e002d_confusions(panels, source, target)
    _write_e002d_class_metrics(e002d_directions)

    print(f"Wrote {REVIEW / 'summary.json'}")
    print(f"Wrote E003b summary/curves and E002d confusion figures/tables to {FIGURES}")
    for row in summary["cases"]:
        print(f"{row['source']}→{row['target']}: {row['reference']} "
              f"F1 {100 * row['direct_macro_f1']:.2f}→{100 * row['adapted_macro_f1']:.2f}% "
              f"({row['delta_macro_f1_pp']:+.2f} pp), selected e{row['selected_epoch']}/"
              f"{row['last_epoch']} {row['stopping_reason']}, {row['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
