"""Validate copied E003c evidence and write a compact training-curve figure."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "outputs/experiments/E003c/results-review"
EVIDENCE = REVIEW / "evidence"
E003B = ROOT / "outputs/experiments/E003b/results-review-20261104/summary.json"
FIGURES = REVIEW / "figures"
ARMS = ("head", "temporal-head")
POPULATION = {"training": 8574, "validation": 1861}
SOURCE_SHA = "d1f83183dcff61f8d9925a74c2c76811d1eca225119e588df8aa3770720e42a8"
COLORS = {"head": "#4c78a8", "temporal-head": "#f58518"}


def read(path):
    return json.loads(path.read_text())


def close(a, b, tol=1e-10):
    assert math.isclose(a, b, rel_tol=tol, abs_tol=tol), (a, b)


def load_arm(arm, normalization=None):
    run = EVIDENCE / arm
    status, config, provenance = (read(run / f) for f in ("status.json", "config.json", "provenance.json"))
    history, validation = read(run / "history.json"), read(run / "loki-validation/metrics.json")["primary"]
    assert status == {"status": "complete", "failures": [], "kind": "comparison"}
    assert (config["experiment"], config["arm"], config["original_source"], config["target"]) == ("E003c", arm, "road-waymo", "loki")
    assert not config["dropout_enabled"] and config["optimizer_reset"] and not config["held_out_tests"] and config["smoke_limits"] is None
    assert config["source_checkpoint_sha256"] == config["source_checkpoint"]["sha256"] == SOURCE_SHA and config["source_checkpoint"]["selected_epoch"] == 17
    assert config.get("initialization") == "source-checkpoint"
    assert "adapted_head_checkpoint" not in config
    assert config["normalization_source"] == "road-waymo" and config["population"] == POPULATION
    assert normalization is None or config["normalization"] == normalization
    settings = config["settings"]
    assert (settings["seed"], settings["epochs"], settings["patience"], settings["batch_size"]) == (0, 100, 15, 64)
    assert settings["precision"] == "float32" and settings["warmup_epochs"] == 3 and provenance["seed"] == settings["seed"] == 0
    trainable = config["trainable_parameters"]
    assert (trainable == ["linear.weight", "linear.bias"] and provenance["trainable_parameters"] == 516
            if arm == "head" else trainable and all(n.startswith(("temporal.", "final_norm.", "head.")) for n in trainable)
            and provenance["trainable_parameters"] == 397316)
    last = provenance["last_completed_epoch"]
    assert provenance["completed_epochs"] == last and len(history) == last + 1
    assert [r["epoch"] for r in history] == list(range(last + 1))
    initial = history[0]
    assert initial["training_ce"] is None and initial["training_f1"] is None
    assert all(initial[k] == 0 for k in ("optimizer_steps", "processed_tracks", "context_slots", "gt_slots", "padding_slots", "batch_slots"))
    assert history[-1]["stopping_reason"] == provenance["stopping_reason"] == "patience" and history[-1]["non_improving_epochs"] == settings["patience"]
    selected = max(history, key=lambda r: (r["validation_f1"], -r["validation_ce"], -r["epoch"]))
    assert selected["selected"] and selected["epoch"] == provenance["selected_checkpoint_epoch"] == provenance["selection"]["epoch"]
    close(selected["validation_f1"], provenance["selection"]["f1"]); close(selected["validation_ce"], provenance["selection"]["ce"])
    assert validation["denominators"]["tracks"] == POPULATION["validation"]
    close(validation["track_weighted"]["macro_f1"], selected["validation_f1"]); close(validation["track_ce"], selected["validation_ce"])
    original = config["initial_validation"]
    close(initial["validation_f1"], original["track_weighted"]["macro_f1"]); close(initial["validation_ce"], original["track_ce"], 1e-6)
    close(initial["validation_f1"], 0.46496012823104105, 1e-6)
    minimum_ce = min(history, key=lambda r: (r["validation_ce"], r["epoch"]))
    return {"arm": arm, "config": config, "provenance": provenance, "history": history,
        "validation": validation, "initial": initial, "selected": selected, "minimum_ce": minimum_ce}


def main():
    first = load_arm(ARMS[0])
    rows = [first, load_arm(ARMS[1], first["config"]["normalization"])]
    assert rows[0]["config"]["initialization_checkpoint"] == rows[1]["config"]["initialization_checkpoint"]
    FIGURES.mkdir(parents=True, exist_ok=True)
    close(rows[0]["initial"]["validation_f1"], rows[1]["initial"]["validation_f1"]); close(rows[0]["initial"]["validation_ce"], rows[1]["initial"]["validation_ce"], 1e-8)
    reference = next(r for r in read(E003B)["cases"] if (r["source"], r["target"], r["reference"]) == ("road-waymo", "loki", "e002d"))
    initial_f1 = rows[0]["initial"]["validation_f1"]
    output_rows = []
    for row in rows:
        selected, last = row["selected"], row["history"][-1]
        classes = dict(zip(row["config"]["classes"], row["validation"]["track_weighted"]["f1"]))
        output_rows.append({"arm": row["arm"], "selected_epoch": selected["epoch"],
            "completed_epochs": row["provenance"]["completed_epochs"], "stopping_reason": row["provenance"]["stopping_reason"],
            "trainable_parameters": row["provenance"]["trainable_parameters"],
            "selected_validation": {"track_weighted_macro_f1": selected["validation_f1"], "track_ce": selected["validation_ce"],
                "delta_from_epoch0_f1_pp": 100 * (selected["validation_f1"] - initial_f1), "class_f1": classes},
            "minimum_validation_ce": {"epoch": row["minimum_ce"]["epoch"], "value": row["minimum_ce"]["validation_ce"]},
            "terminal": {"epoch": last["epoch"], "online_training_f1": last["training_f1"],
                "validation_f1": last["validation_f1"], "online_training_ce": last["training_ce"], "validation_ce": last["validation_ce"]},
            "validation_tracks": row["validation"]["denominators"]["tracks"], "wandb": row["provenance"].get("wandb"),
            "revision": row["provenance"]["revision"],
            "history": [{k: e.get(k) for k in ("epoch", "training_f1", "validation_f1", "training_ce", "validation_ce", "selected")}
                for e in row["history"]]})
    class_delta = {name: 100 * (output_rows[1]["selected_validation"]["class_f1"][name]
        - output_rows[0]["selected_validation"]["class_f1"][name]) for name in output_rows[0]["selected_validation"]["class_f1"]}
    source = rows[0]["config"]["source_checkpoint"]
    summary = {"scope": "E003c seed0 ROAD-Waymo→LOKI; full validation selection; no held-out test access",
        "population": POPULATION, "source_checkpoint": {k: source[k] for k in ("path", "sha256", "selected_epoch")},
        "initialization": "source-checkpoint",
        "source_normalization": "ROAD-Waymo (retained for both target splits)", "dropout_enabled": False,
        "shared_epoch0": {"track_weighted_macro_f1": initial_f1, "track_ce": rows[0]["initial"]["validation_ce"]},
        "target_trained_reference": {"track_weighted_macro_f1": reference["target_trained_macro_f1"],
            "track_ce": reference["target_trained_ce"],
            "qualification": "Same-family model trained on LOKI task data and using LOKI training normalization."},
        "arms": output_rows, "temporal_head_minus_head_class_f1_pp": class_delta,
        "figure": ["figures/training-validation-curves.png", "figures/training-validation-curves.svg"]}
    (REVIEW / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True, sharey="row")
    for col, row in enumerate(rows):
        color, hist = COLORS[row["arm"]], row["history"]
        train = [r for r in hist if r["epoch"] > 0]
        for ax, metric, scale in ((axes[0, col], "ce", 1), (axes[1, col], "f1", 100)):
            ax.plot([r["epoch"] for r in train], [scale * r[f"training_{metric}"] for r in train], "--", color=color, lw=1.35)
            ax.plot([r["epoch"] for r in hist], [scale * r[f"validation_{metric}"] for r in hist], "-", color=color, lw=1.7)
            ax.scatter([0], [scale * row["initial"][f"validation_{metric}"]], marker="D", s=34,
                facecolor="white", edgecolor=color, linewidth=1.2, zorder=5)
            selected = row["selected"]
            ax.scatter([selected["epoch"]], [scale * selected[f"validation_{metric}"]], marker="o", s=56,
                color=color, edgecolor="black", linewidth=.8, zorder=6)
            ax.grid(axis="y", alpha=.23); ax.set_axisbelow(True)
        axes[0, col].set_title(f"{row['arm']} · selected e{row['selected']['epoch']} / stopped e{hist[-1]['epoch']}")
        axes[1, col].set_xlabel("epoch")
    for ax in axes[0]: ax.set_ylabel("mean track cross-entropy")
    for ax in axes[1]:
        ax.set_ylabel("track-weighted macro-F1 (%)")
        ax.axhline(100 * initial_f1, color="#777777", linestyle=":", lw=1, alpha=.75)
    legend = [Line2D([0], [0], color="#444", ls="--", label="online train (changing weights)"),
        Line2D([0], [0], color="#444", ls="-", label="target validation"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor="#444", label="epoch 0 validation"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777", markeredgecolor="black", label="selected checkpoint"),
        Line2D([0], [0], color="#777", ls=":", label="shared epoch 0 F1")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .945), ncol=3, frameon=False, fontsize=8)
    fig.suptitle("E003c · ROAD-Waymo → LOKI source-checkpoint fine-tuning", y=.99, fontsize=13)
    fig.text(.5, .025, f"Shared starting F1: {100*initial_f1:.2f}%. Dropout is disabled; training is online under changing weights. "
        f"Target-trained LOKI reference: {100*reference['target_trained_macro_f1']:.2f}% F1 / "
        f"{reference['target_trained_ce']:.3f} CE (trained and normalized on LOKI).", ha="center", fontsize=8)
    fig.tight_layout(rect=(.03, .10, .98, .87))
    fig.savefig(FIGURES / "training-validation-curves.png", dpi=180)
    fig.savefig(FIGURES / "training-validation-curves.svg")
    plt.close(fig)
    print(f"Wrote {REVIEW / 'summary.json'} and PNG/SVG figures under {FIGURES}")
    for r in output_rows:
        s = r["selected_validation"]
        print(f"{r['arm']}: e{r['selected_epoch']}/{r['completed_epochs']} ({r['stopping_reason']}), "
            f"F1 {100*s['track_weighted_macro_f1']:.2f}% CE {s['track_ce']:.4f}; minimum validation CE e{r['minimum_validation_ce']['epoch']}")


if __name__ == "__main__":
    main()
