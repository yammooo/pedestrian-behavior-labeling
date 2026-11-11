#!/usr/bin/env python3
"""Summarize the completed E002f architecture comparison from saved validation metrics."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs/experiments"
RUN = OUTPUTS / "E002f/runs/temporal-architecture-20261110-091857"
FEATURES = OUTPUTS / "E002f/features/temporal-architecture-20261110-091857"
E002D = OUTPUTS / "E002d/runs/fusion-20261030-140937-road-waymo-rgbdrop50"
E002_TRANSFER = OUTPUTS / "E002/transfer-validation-20261102/cells/road-waymo/geometry+rgb+lidar"
E003C = OUTPUTS / "E003c/runs/source-adaptation-20261106-121040"
OUT = OUTPUTS / "E002f/results-review"
CLASSES = ("Moving", "Stopped", "Waiting to cross", "Crossing")
STAGES = ("source", "direct", "head", "temporal-head")
STAGE_LABELS = {"source": "ROAD source validation", "direct": "LOKI direct transfer",
    "head": "LOKI head-only", "temporal-head": "LOKI temporal + head"}
W_B = "https://wandb.ai/yammo-unipd/pedestrian-behaviour-labeling/runs/{}"


def _json(path):
    return json.loads(Path(path).read_text())


def _case(metrics, status, history=None, *, dataset, run_id, checkpoint_epoch=None):
    metrics = _json(metrics)
    status = _json(status)
    if status.get("status") != "complete" or status.get("failures"):
        raise ValueError(f"Incomplete result: {status}")
    primary = metrics.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("Missing saved primary metrics")
    denominators = primary.get("denominators", {})
    tracks = denominators.get("tracks")
    if metrics.get("split", "validation") != "validation" or tracks != (1110 if dataset == "ROAD-Waymo" else 1861):
        raise ValueError(f"Unexpected validation population for {dataset}: {tracks}")

    history_rows = _json(history) if history else []
    checkpoint_epoch = metrics.get("selected_epoch", checkpoint_epoch)
    selected = [row for row in history_rows if row.get("selected")]
    if selected:
        choice = selected[-1]
        checkpoint_epoch = choice["epoch"]
        if not np.isclose(choice["validation_f1"], primary["track_weighted"]["macro_f1"], atol=1e-9):
            raise ValueError("Saved history selection differs from saved validation metric")
    stop_epoch = history_rows[-1]["epoch"] if history_rows else None

    for weighting in ("raw", "track_weighted"):
        record = primary[weighting]
        matrix = np.asarray(record["confusion"], dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError(f"Invalid {weighting} confusion matrix")
        support = matrix.sum(axis=1)
        precision = np.divide(np.diag(matrix), matrix.sum(axis=0), out=np.zeros(4), where=matrix.sum(axis=0) > 0)
        recall = np.divide(np.diag(matrix), support, out=np.zeros(4), where=support > 0)
        f1 = np.divide(2 * precision * recall, precision + recall,
            out=np.zeros(4), where=precision + recall > 0)
        # Keep displayed per-class metrics tied to the saved confusion matrix.
        for key, values in (("precision", precision), ("recall", recall), ("f1", f1)):
            if not np.allclose(record[key], values, rtol=1e-9, atol=1e-9):
                raise ValueError(f"Saved {weighting} {key} differs from confusion reconstruction")
        if not np.isclose(record["macro_f1"], f1.mean(), rtol=1e-9, atol=1e-9):
            raise ValueError(f"Saved {weighting} macro-F1 differs from per-class F1")

    return {"dataset": dataset, "status": status["status"], "tracks": tracks,
        "selected_epoch": checkpoint_epoch, "stopped_epoch": stop_epoch,
        "wandb": W_B.format(run_id) if run_id else None,
        "history": str(history.relative_to(ROOT)) if history else None,
        "metrics": primary}


MODELS = [
    {"key": "transformer-2layer", "name": "Transformer ×2", "parameters": 397316,
     "cases": {
        "source": (E002D / "road-waymo-validation/metrics.json", E002D / "status.json", E002D / "history.json", "ROAD-Waymo", "9k11kn3b"),
        "direct": (E002_TRANSFER / "metrics.json", E002D / "status.json", None, "LOKI", "9k11kn3b"),
        "head": (E003C / "head/loki-validation/metrics.json", E003C / "head/status.json", E003C / "head/history.json", "LOKI", "27vi7vgw"),
        "temporal-head": (E003C / "temporal-head/loki-validation/metrics.json", E003C / "temporal-head/status.json", E003C / "temporal-head/history.json", "LOKI", "h1yg4q96")}},
    {"key": "bilstm-96", "name": "BiLSTM", "parameters": 199044,
     "cases": {
        "source": (RUN / "source/bilstm-96/road-waymo-validation/metrics.json", RUN / "source/bilstm-96/status.json", RUN / "source/bilstm-96/history.json", "ROAD-Waymo", "qbd56oyy"),
        "direct": (FEATURES / "bilstm-96/loki-validation/metrics.json", FEATURES / "bilstm-96/status.json", None, "LOKI", "qbd56oyy"),
        "head": (RUN / "adaptation/bilstm-96/head/loki-validation/metrics.json", RUN / "adaptation/bilstm-96/head/status.json", RUN / "adaptation/bilstm-96/head/history.json", "LOKI", "4k4gnen5"),
        "temporal-head": (RUN / "adaptation/bilstm-96/temporal-head/loki-validation/metrics.json", RUN / "adaptation/bilstm-96/temporal-head/status.json", RUN / "adaptation/bilstm-96/temporal-head/history.json", "LOKI", "hfvk38it")}},
    {"key": "transformer-1layer", "name": "Transformer ×1", "parameters": 199044,
     "cases": {
        "source": (RUN / "source/transformer-1layer/road-waymo-validation/metrics.json", RUN / "source/transformer-1layer/status.json", RUN / "source/transformer-1layer/history.json", "ROAD-Waymo", "iy94qap8"),
        "direct": (FEATURES / "transformer-1layer/loki-validation/metrics.json", FEATURES / "transformer-1layer/status.json", None, "LOKI", "iy94qap8"),
        "head": (RUN / "adaptation/transformer-1layer/head/loki-validation/metrics.json", RUN / "adaptation/transformer-1layer/head/status.json", RUN / "adaptation/transformer-1layer/head/history.json", "LOKI", "rip98fjj"),
        "temporal-head": (RUN / "adaptation/transformer-1layer/temporal-head/loki-validation/metrics.json", RUN / "adaptation/transformer-1layer/temporal-head/status.json", RUN / "adaptation/transformer-1layer/temporal-head/history.json", "LOKI", "mkiu997e")}}
]


def _collect():
    result = {"scope": "seed0; ROAD source validation and LOKI direct/adapted validation; no held-out test access",
        "class_order": list(CLASSES), "models": {}}
    for model in MODELS:
        stages = {}
        for stage, (metrics, status, history, dataset, run_id) in model["cases"].items():
            stages[stage] = _case(metrics, status, history, dataset=dataset, run_id=run_id)
        result["models"][model["key"]] = {"name": model["name"],
            "temporal_norm_head_parameters": model["parameters"], "stages": stages}
    return result


def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=180, bbox_inches="tight", pad_inches=.12)


def _plots(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    figures = OUT / "figures"
    models = list(data["models"].values())
    x = np.arange(len(models))
    width = .18
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"source": "#536d8a", "direct": "#8d99ae", "head": "#dc8f35", "temporal-head": "#31917b"}
    for index, stage in enumerate(STAGES):
        offset = (index - 1.5) * width
        f1 = [100 * model["stages"][stage]["metrics"]["track_weighted"]["macro_f1"] for model in models]
        ce = [model["stages"][stage]["metrics"]["track_ce"] for model in models]
        axes[0].bar(x + offset, f1, width, label=STAGE_LABELS[stage], color=colors[stage])
        axes[1].bar(x + offset, ce, width, label=STAGE_LABELS[stage], color=colors[stage])
    for ax, title, ylabel in ((axes[0], "Selected validation macro-F1", "Track-weighted macro-F1 (%)"),
            (axes[1], "Selected validation cross-entropy", "Mean track CE")):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels=[model["name"] for model in models])
        ax.grid(axis="y", alpha=.22)
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.04))
    fig.suptitle("E002f temporal architectures · seed0", y=1.13, fontsize=14)
    fig.text(.5, -.015, "Source scores use ROAD-Waymo validation; direct and adapted scores use LOKI validation.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .93))
    _save_figure(fig, figures / "summary-f1-ce.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex="col")
    styles = {"source": ("#536d8a", "ROAD source"), "head": ("#dc8f35", "LOKI head"),
        "temporal-head": ("#31917b", "LOKI temporal + head")}
    legend = []
    for column, model in enumerate(MODELS):
        data_model = data["models"][model["key"]]
        for stage in ("source", "head", "temporal-head"):
            case = data_model["stages"][stage]
            rows = _json(ROOT / case["history"])
            color, label = styles[stage]
            epochs = [row["epoch"] for row in rows]
            train_f1 = [row["training_f1"] for row in rows]
            val_f1 = [row["validation_f1"] for row in rows]
            train_ce = [row["training_ce"] for row in rows]
            val_ce = [row["validation_ce"] for row in rows]
            axes[0, column].plot(epochs, train_f1, color=color, lw=1.35, label=f"{label} train")
            axes[0, column].plot(epochs, val_f1, color=color, lw=1.35, ls="--", label=f"{label} val")
            axes[1, column].plot(epochs, train_ce, color=color, lw=1.35)
            axes[1, column].plot(epochs, val_ce, color=color, lw=1.35, ls="--")
            selected = next((row for row in reversed(rows) if row.get("selected")), None)
            if selected:
                axes[0, column].scatter(selected["epoch"], selected["validation_f1"], color=color, s=18, zorder=3)
                axes[1, column].scatter(selected["epoch"], selected["validation_ce"], color=color, s=18, zorder=3)
        axes[0, column].set_title(model["name"])
        axes[0, column].set_ylim(0, 1)
        axes[1, column].set_ylim(bottom=0)
        axes[1, column].set_xlabel("Epoch")
        for row in range(2):
            axes[row, column].grid(alpha=.22)
    axes[0, 0].set_ylabel("Track-weighted macro-F1")
    axes[1, 0].set_ylabel("Mean track CE")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.05))
    fig.suptitle("Online training and validation histories", y=1.12, fontsize=14)
    fig.text(.5, -.015, "Training metrics use changing weights; source dropout is active, adaptation dropout is disabled.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .91))
    _save_figure(fig, figures / "training-curves.png")
    plt.close(fig)

    confusion_stages = STAGES
    for weighting, view in (("track_weighted", "row-normalized"), ("track_weighted", "count"),
            ("raw", "row-normalized"), ("raw", "count")):
        arrays = []
        for model in models:
            for stage in confusion_stages:
                matrix = np.asarray(model["stages"][stage]["metrics"][weighting]["confusion"], dtype=float)
                if view == "row-normalized":
                    matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix),
                        where=matrix.sum(axis=1, keepdims=True) > 0) * 100
                arrays.append(matrix)
        vmax = 100. if view == "row-normalized" else max(float(matrix.max()) for matrix in arrays)
        norm = Normalize(0, vmax if vmax > 0 else 1)
        fig, axes = plt.subplots(3, 4, figsize=(17, 11))
        fig.subplots_adjust(left=.16, right=.89, top=.91, bottom=.10, wspace=.30, hspace=.42)
        image = None
        for row, model in enumerate(models):
            for column, stage in enumerate(confusion_stages):
                ax = axes[row, column]
                record = model["stages"][stage]["metrics"][weighting]
                matrix = np.asarray(record["confusion"], dtype=float)
                if view == "row-normalized":
                    matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix),
                        where=matrix.sum(axis=1, keepdims=True) > 0) * 100
                image = ax.imshow(matrix, cmap="Blues", norm=norm, interpolation="nearest")
                macro = 100 * record["macro_f1"]
                ax.set_title(f"{STAGE_LABELS[stage]}\nF1 {macro:.2f}%", fontsize=9)
                ax.set_xticks(range(4), labels=CLASSES if row == 2 else ("", "", "", ""), rotation=28, ha="right")
                ax.set_yticks(range(4), labels=CLASSES if column == 0 else ("", "", "", ""))
                ax.tick_params(labelsize=8)
                for (i, j), value in np.ndenumerate(matrix):
                    label = f"{value:.1f}%" if view == "row-normalized" else f"{value:.1f}" if weighting == "track_weighted" else f"{value:.0f}"
                    ax.text(j, i, label, ha="center", va="center", fontsize=7,
                        color="white" if value > vmax * .57 else "black")
                if row == 2:
                    ax.set_xlabel("Predicted class", fontsize=8)
            axes[row, 0].set_ylabel(f"{model['name']}\nGround-truth class", fontsize=9)
        figure_title = f"Validation confusion · {weighting.replace('_', '-')} {view}"
        fig.suptitle(figure_title + "\nSource panels use ROAD-Waymo; remaining panels use LOKI · rows are ground truth", fontsize=12, y=.985)
        colorbar_axes = fig.add_axes([.92, .17, .022, .68])
        fig.colorbar(image, cax=colorbar_axes,
            label="Row-normalized GT (%)" if view == "row-normalized" else "GT frames" if weighting == "raw" else "Track-weighted GT mass",
            ticks=[0, 25, 50, 75, 100] if view == "row-normalized" else None)
        _save_figure(fig, figures / f"validation-confusions-{weighting.replace('_', '-')}-{view}.png")
        plt.close(fig)


def _per_class(data):
    lines = ["# E002f cross-architecture per-class validation metrics", "",
        "Track-weighted precision, recall and F1 in percent, computed from the saved confusion matrices. Class order is shared across models.", ""]
    for stage in STAGES:
        dataset = "ROAD-Waymo" if stage == "source" else "LOKI"
        lines.extend([f"## {STAGE_LABELS[stage]} · {dataset}", "",
            "| Architecture | Class | Precision (%) | Recall (%) | F1 (%) |", "|---|---|---:|---:|---:|"])
        for model in data["models"].values():
            metrics = model["stages"][stage]["metrics"]["track_weighted"]
            for index, label in enumerate(CLASSES):
                values = [100 * metrics[key][index] for key in ("precision", "recall", "f1")]
                lines.append(f"| {model['name']} | {label} | " + " | ".join(f"{value:.2f}" for value in values) + " |")
        lines.append("")
    (OUT / "per-class-metrics.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = _collect()
    (OUT / "summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    _per_class(data)
    _plots(data)
    print(f"Wrote E002f result summary and figures to {OUT}")


if __name__ == "__main__":
    main()
