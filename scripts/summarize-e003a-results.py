"""Build the compact E003a review figures from copied run evidence."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = ROOT / "outputs/experiments/E003a/results-review-20261104/evidence"
EVIDENCE = EVIDENCE_ROOT / "E003a/runs/20261103-105614"
FIGURES = ROOT / "outputs/experiments/E003a/results-review-20261104/figures"
SOURCES = {
    "road-waymo": ("ROAD", "loki", 1110, 1861),
    "loki": ("LOKI", "road-waymo", 1861, 1110),
}
DIAGNOSTICS = ("geometry-only", "rgb-only", "lidar-only", "fixedmixed")


def read(path):
    return json.loads(path.read_text())


def pair(source):
    run = EVIDENCE / source
    history = read(run / "history.json")
    provenance = read(run / "provenance.json")
    comparison = read(run / "natural-comparison.json")
    status = read(run / "status.json")
    control_name = f"fusion-20261030-140937-{source}-rgbdrop50"
    control_run = EVIDENCE_ROOT / "E002d/runs" / control_name
    control_history = read(control_run / "history.json")
    control_status = read(control_run / "status.json")
    assert status == {"status": "complete", "failures": [], "kind": "exploratory-source-validation"}
    assert control_status == status
    assert provenance["revision"] == "0be4a7016fa5ab92b7f14e01ca898acb835d9718"
    assert provenance["diff_sha256"].startswith("e3b0c44298fc1c149")
    assert len(history) == provenance["completed_epochs"] == provenance["last_completed_epoch"]
    assert history[-1]["stopping_reason"] == provenance["stopping_reason"] == "patience"
    assert history[-1]["non_improving_epochs"] == 15
    best = max(history, key=lambda row: row["validation_f1"])
    assert best["epoch"] == provenance["selected_checkpoint_epoch"]
    control_best = max(control_history, key=lambda row: row["validation_f1"])
    assert control_history[-1]["stopping_reason"] == "patience"
    assert control_history[-1]["non_improving_epochs"] == 15
    return history, best, provenance, comparison, control_history, control_best


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = {source: pair(source) for source in SOURCES}
    comparisons = {}
    diagnostics = {}
    for source, (name, opposite, source_tracks, target_tracks) in SOURCES.items():
        history, best, provenance, comparison, _, control_best = runs[source]
        assert comparison["scope"] == "source and opposite validation; no test access"
        rows = {row["evaluation_source"]: row for row in comparison["rows"]}
        assert set(rows) == {source, opposite}
        assert rows[source]["tracks"] == source_tracks
        assert rows[opposite]["tracks"] == target_tracks
        assert abs(rows[source]["e003a_macro_f1"] - best["validation_f1"]) < 1e-12
        assert abs(rows[source]["e002d_control_macro_f1"] - control_best["validation_f1"]) < 1e-12
        assert abs(rows[source]["e002d_control_ce"] - control_best["validation_ce"]) < 1e-12
        transfer = comparison["control_transfer"]["target"]
        assert transfer["target_source"] == opposite
        assert transfer["target_tracks"] == target_tracks
        assert abs(transfer["target_f1"] - rows[opposite]["e002d_control_macro_f1"]) < 1e-12
        assert abs(transfer["target_ce"] - rows[opposite]["e002d_control_ce"]) < 1e-12
        comparisons[source] = rows

        summary = read(EVIDENCE / source / "diagnostics" / source / "summary.json")
        diagnostics[source] = summary
        for view in DIAGNOSTICS:
            masked, control = summary[view]["e003a"], summary[view]["e002d-control"]
            assert masked["tracks"] == control["tracks"] == source_tracks
            assert masked["no_input_tracks"] == control["no_input_tracks"]
            assert masked["primary"]["track_weighted"]["macro_f1"] is not None
            assert control["primary"]["track_weighted"]["macro_f1"] is not None

    # Online training uses changing weights and active dropout; E003a also masks observations.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex="col", constrained_layout=True)
    for column, (source, (name, _, _, _)) in enumerate(SOURCES.items()):
        history, best, _, _, control_history, control_best = runs[source]
        epochs = [row["epoch"] for row in history]
        control_epochs = [row["epoch"] for row in control_history]
        for axis, metric, scale in ((axes[0, column], "ce", 1), (axes[1, column], "f1", 100)):
            for rows, run_label, color, selected in (
                (control_history, "E002d control", "#4c78a8", control_best),
                (history, "E003a masking", "#f58518", best),
            ):
                run_epochs = control_epochs if rows is control_history else epochs
                axis.plot(run_epochs, [scale * row[f"training_{metric}"] for row in rows], "--", color=color, label=f"{run_label} online train")
                axis.plot(run_epochs, [scale * row[f"validation_{metric}"] for row in rows], "-", color=color, label=f"{run_label} natural validation")
                axis.axvline(selected["epoch"], color=color, linestyle=":", alpha=0.8)
                axis.scatter([selected["epoch"]], [scale * selected[f"validation_{metric}"]], color=color, edgecolor="black", zorder=4)
            axis.grid(axis="y", alpha=0.25)
        axes[0, column].set_title(f"{name} · selected e{control_best['epoch']} / e{best['epoch']}")
        axes[0, column].set_ylabel("mean track cross-entropy")
        axes[1, column].set_ylabel("track-weighted macro-F1 (%)")
        axes[1, column].set_ylim(0, 100)
        axes[1, column].set_xlabel("epoch")
    axes[0, 0].set_ylim(0, 1.4)
    axes[0, 1].set_ylim(0, 1.4)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Seed0 source-validation curves: E002d control and E003a masking", fontsize=13)
    fig.savefig(FIGURES / "training-validation-curves.png", dpi=180)
    fig.savefig(FIGURES / "training-validation-curves.svg")
    plt.close(fig)

    categories = []
    f1 = {"E002d RGB dropout 0.5": [], "E003a masking": []}
    ce = {"E002d RGB dropout 0.5": [], "E003a masking": []}
    for source, (name, opposite, source_tracks, target_tracks) in SOURCES.items():
        for evaluation, count, label in ((source, source_tracks, "source"), (opposite, target_tracks, "opposite")):
            row = comparisons[source][evaluation]
            categories.append(f"{name}→{('ROAD' if evaluation == 'road-waymo' else 'LOKI')}\n{label} ({count:,})")
            f1["E002d RGB dropout 0.5"].append(100 * row["e002d_control_macro_f1"])
            f1["E003a masking"].append(100 * row["e003a_macro_f1"])
            ce["E002d RGB dropout 0.5"].append(row["e002d_control_ce"])
            ce["E003a masking"].append(row["e003a_ce"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    x = list(range(len(categories)))
    width = 0.36
    colors = ("#4c78a8", "#f58518")
    for offset, (label, color) in enumerate(zip(f1, colors)):
        bars = axes[0].bar([i + (offset - 0.5) * width for i in x], f1[label], width, label=label, color=color)
        axes[1].bar([i + (offset - 0.5) * width for i in x], ce[label], width, color=color)
        for bar, value in zip(bars, f1[label]):
            axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        for bar, value in zip(axes[1].containers[-1], ce[label]):
            axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    for axis in axes:
        axis.set_xticks(x, categories)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("track-weighted macro-F1 (%)")
    axes[0].set_title("Natural validation F1")
    axes[1].set_ylim(0, 1.55)
    axes[1].set_ylabel("mean track CE")
    axes[1].set_title("Natural validation CE")
    axes[0].legend(fontsize=8)
    fig.suptitle("Selected seed0 checkpoints: source and opposite validation", fontsize=13)
    fig.savefig(FIGURES / "natural-validation-comparison.png", dpi=180)
    fig.savefig(FIGURES / "natural-validation-comparison.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True, constrained_layout=True)
    labels = ("Geometry", "RGB", "LiDAR", "Mixed")
    for axis, (source, (name, _, _, _)) in zip(axes, SOURCES.items()):
        x = list(range(len(DIAGNOSTICS)))
        control = [100 * diagnostics[source][view]["e002d-control"]["primary"]["track_weighted"]["macro_f1"] for view in DIAGNOSTICS]
        masked = [100 * diagnostics[source][view]["e003a"]["primary"]["track_weighted"]["macro_f1"] for view in DIAGNOSTICS]
        for offset, (values, label, color) in enumerate(((control, "E002d RGB dropout 0.5", "#4c78a8"), (masked, "E003a masking", "#f58518"))):
            bars = axis.bar([i + (offset - 0.5) * width for i in x], values, width, label=label, color=color)
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 1.1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
        axis.set_title(f"{name} source validation")
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("track-weighted macro-F1 (%)")
    fig.suptitle("Fixed branch-view diagnostics; full validation tracks retained", fontsize=13)
    fig.text(0.5, -0.015, "LOKI RGB-only includes 717 no-input tracks; these remain in the reported metric.", ha="center", fontsize=8)
    fig.savefig(FIGURES / "fixed-mask-diagnostics.png", dpi=180, bbox_inches="tight")
    fig.savefig(FIGURES / "fixed-mask-diagnostics.svg", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote six PNG/SVG figures to {FIGURES}")
    for source, (name, _, _, _) in SOURCES.items():
        _, best, provenance, _, control_history, control_best = runs[source]
        print(f"{name}: E002d selected e{control_best['epoch']}; E003a selected e{best['epoch']}, stopped e{provenance['last_completed_epoch']} ({provenance['stopping_reason']})")


if __name__ == "__main__":
    main()
