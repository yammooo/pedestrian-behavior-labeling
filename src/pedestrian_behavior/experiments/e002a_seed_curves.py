"""Plot E002a g-rgb online histories for repeat seeds 1–4."""

import argparse
import json
import math
from pathlib import Path


SOURCES = ("road-waymo", "loki")
SEEDS = (1, 2, 3, 4)
METRICS = ("epoch", "training_ce", "validation_ce", "training_f1", "validation_f1")


def _read_run(path):
    status = json.loads((path / "status.json").read_text())
    history = json.loads((path / "history.json").read_text())
    provenance = json.loads((path / "provenance.json").read_text())
    if status.get("status") != "complete" or not isinstance(history, list) or not history:
        raise ValueError(f"Completed non-empty history required: {path}")
    epochs = []
    for row in history:
        if any(key not in row for key in METRICS):
            raise ValueError(f"Incomplete epoch history: {path}")
        values = [float(row[key]) for key in METRICS]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite epoch history: {path}")
        epochs.append(int(row["epoch"]))
    if epochs != sorted(set(epochs)):
        raise ValueError(f"Epochs must be unique and increasing: {path}")
    selected_epoch = provenance.get("selection", {}).get("epoch")
    if selected_epoch not in epochs:
        raise ValueError(f"Selected epoch is absent from history: {path}")
    return history, selected_epoch


def load_histories(runs_root):
    runs_root = Path(runs_root)
    return {
        source: {
            seed: _read_run(runs_root / f"e002a-{source}-g-rgb-s{seed}")
            for seed in SEEDS
        }
        for source in SOURCES
    }


def plot_histories(runs_root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    histories = load_histories(runs_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = {"training": "#4063d8", "validation": "#df8544"}
    for source, runs in histories.items():
        fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex="col", sharey="row")
        for column, seed in enumerate(SEEDS):
            history, selected_epoch = runs[seed]
            epochs = [row["epoch"] for row in history]
            ce, f1 = axes[:, column]
            for ax in (ce, f1):
                ax.axvline(selected_epoch, color="#30343b", linestyle=":", linewidth=1, alpha=.7)
                ax.grid(alpha=.2)
            ce.plot(epochs, [row["training_ce"] for row in history], color=colors["training"],
                    linestyle="--", alpha=.7, linewidth=1.5)
            ce.plot(epochs, [row["validation_ce"] for row in history], color=colors["validation"],
                    linewidth=1.7)
            f1.plot(epochs, [row["training_f1"] for row in history], color=colors["training"],
                    linestyle="--", alpha=.7, linewidth=1.5)
            f1.plot(epochs, [row["validation_f1"] for row in history], color=colors["validation"],
                    linewidth=1.7)
            ce.set_title(f"Seed {seed}")
            f1.set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Track-weighted CE")
        axes[1, 0].set_ylabel("Track-weighted macro-F1")
        axes[1, 0].set_ylim(0, 1)
        fig.suptitle(f"E002a g-rgb · {source} · seeds 1–4", y=.995)
        fig.legend(
            [Line2D([], [], color=colors["training"], linestyle="--"),
             Line2D([], [], color=colors["validation"]),
             Line2D([], [], color="#30343b", linestyle=":")],
            ["Training (online)", "Source validation", "Selected epoch"],
            loc="upper center", bbox_to_anchor=(.5, .955), ncol=3, frameon=False,
        )
        fig.text(.5, .015, "Training CE/F1 are online epoch traces with changing weights and active dropout.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, .89))
        for extension in ("png", "svg"):
            fig.savefig(output / f"{source}-g-rgb-seed-curves.{extension}", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Completed repeat root containing the eight named run directories")
    parser.add_argument("--output", type=Path, required=True, help="Directory for PNG and SVG figures")
    args = parser.parse_args()
    plot_histories(args.runs_root, args.output)


if __name__ == "__main__":
    main()
