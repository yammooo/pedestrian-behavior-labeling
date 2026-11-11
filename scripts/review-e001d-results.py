"""Summarize the E001d five-seed runs against the frozen E001 controls."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SWEEP_ID = "seeds-20261009-084417-412679"
OLD = ROOT / "outputs/experiments/E001b/analysis/20261015-seed-review/metadata/E001" / SWEEP_ID
SNAPSHOT = ROOT / "outputs/experiments/E001d/results-review/metadata-snapshot.json"
OUT = SNAPSHOT.parent
SOURCES = ("road-waymo", "loki")
TARGETS = ("road-waymo", "loki")
SEEDS = range(5)
CLASSES = ("Moving", "Stopped", "Waiting", "Crossing")
DIRECTIONS = (("road-waymo", "road-waymo"), ("road-waymo", "loki"),
              ("loki", "loki"), ("loki", "road-waymo"))
DIR_NAMES = {("road-waymo", "road-waymo"): "R→R", ("road-waymo", "loki"): "R→L",
             ("loki", "loki"): "L→L", ("loki", "road-waymo"): "L→R"}
MODELS = ("small MLP", "matched MLP", "BiLSTM")
PARAMETERS = {"small MLP": 5252, "matched MLP": 30342, "BiLSTM": 30340}
COLORS = {"small MLP": "#58759a", "matched MLP": "#d88921", "BiLSTM": "#20836f"}


def read(path):
    return json.loads(Path(path).read_text())


def moments(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == (5,) and np.isfinite(values).all()
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
            "seeds": values.tolist()}


def check_metric(metric):
    primary = metric["primary"]
    weighted = primary["track_weighted"]
    confusion = np.asarray(weighted["confusion"], dtype=np.float64)
    assert confusion.shape == (4, 4) and np.isfinite(confusion).all()
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * np.diag(confusion), denominator,
                   out=np.zeros(4), where=denominator > 0)
    assert np.allclose(f1, weighted["f1"], rtol=0, atol=1e-12)
    assert np.isclose(f1.mean(), weighted["macro_f1"], rtol=0, atol=1e-12)
    assert np.all(np.asarray(weighted["supported_classes"]) == np.arange(4))
    assert np.isfinite(primary["track_ce"])
    return f1, primary["track_ce"], primary["denominators"]


def metric_files_new(attempt):
    return {Path(name).parts[0].removesuffix("-test"): data["json"]
            for name, data in attempt["files"].items() if name.endswith("-test/metrics.json")}


def compare_protocol(new_config, new_prov, old_config, old_prov):
    assert new_config["source"] == old_config["source"]
    assert new_config["configuration"] == old_config["configuration"] == "K+T+R"
    assert new_config["classes"] == old_config["classes"]
    assert new_config["settings"] == old_config["settings"]
    assert new_config["determinism"] == old_config["determinism"]
    assert new_prov["source_statistics"] == old_prov["source_statistics"]
    for dataset in TARGETS:
        n, o = new_prov["datasets"][dataset], old_prov["datasets"][dataset]
        for key in ("setup_sha256", "collection_sha256", "policy_sha256", "population", "support"):
            assert n[key] == o[key], (dataset, key)


def history_summary(config, provenance, history):
    assert history and history[-1]["epoch"] == provenance["last_completed_epoch"]
    selected = [row for row in history if row.get("selected")]
    assert selected
    chosen = next(row for row in selected if row["epoch"] == provenance["selection"]["epoch"])
    assert np.isclose(chosen["validation_f1"], provenance["selection"]["f1"])
    assert np.isclose(chosen["validation_ce"], provenance["selection"]["ce"])
    assert chosen["validation_f1"] == max(row["validation_f1"] for row in history)
    last = history[-1]
    return {"selected_epoch": chosen["epoch"], "completed_epochs": last["epoch"],
            "stopping_reason": provenance["stopping_reason"],
            "selected_training_ce": chosen["training_ce"],
            "selected_validation_ce": chosen["validation_ce"],
            "final_training_ce": last["training_ce"],
            "final_validation_ce": last["validation_ce"]}


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(stats, scale=1):
    return f"{stats['mean'] * scale:.2f} ± {stats['std'] * scale:.2f}"


def plot_primary(records, primary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    x = np.arange(len(MODELS))
    jitter = np.linspace(-0.1, 0.1, 5)
    all_values = [r["macro_f1_percent"] for r in records]
    low, high = min(all_values), max(all_values)
    for ax, direction in zip(axes.flat, DIRECTIONS):
        for i, model in enumerate(MODELS):
            stats = primary[(direction[0], direction[1], model)]
            for j, value in enumerate(stats["seeds"]):
                ax.scatter(i + jitter[j], value, color=COLORS[model], s=21, alpha=.68, zorder=2)
            ax.errorbar(i, stats["mean"], yerr=stats["std"], fmt="D", color=COLORS[model],
                        markersize=6, capsize=4, linewidth=1.6, zorder=3)
        ax.set_title(DIR_NAMES[direction])
        ax.set_xticks(x, ["MLP\n5,252", "MLP\n30,342", "BiLSTM\n30,340"])
        ax.grid(axis="y", alpha=.25)
    axes[0, 0].set_ylabel("Track-weighted macro-F1 (%)")
    axes[1, 0].set_ylabel("Track-weighted macro-F1 (%)")
    pad = max(1.5, (high-low)*.12)
    axes[0, 0].set_ylim(low-pad, high+pad)
    fig.suptitle("E001d capacity control: held-out test macro-F1", y=.99)
    fig.tight_layout(rect=(0, .035, 1, .97))
    fig.text(.01, .005, "R = ROAD-Waymo; L = LOKI. Dots = individual seeds; diamonds/error bars = mean ± sample SD.",
             ha="left", fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / "figures" / f"primary-f1.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_classes(per_class):
    values = np.array([[per_class[(direction[0], direction[1], cls)]["mean"]
                        for cls in CLASSES] for direction in DIRECTIONS])
    positive = np.array([[per_class[(direction[0], direction[1], cls)]["positive_seeds"]
                          for cls in CLASSES] for direction in DIRECTIONS])
    limit = max(1, float(np.abs(values).max()))
    fig, ax = plt.subplots(figsize=(8, 3.7))
    image = ax.imshow(values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(4), CLASSES)
    ax.set_yticks(range(4), [DIR_NAMES[d] for d in DIRECTIONS])
    ax.set_title("Per-class F1: BiLSTM − matched MLP (percentage points)")
    for y in range(4):
        for x in range(4):
            color = "white" if abs(values[y, x]) > limit * .55 else "black"
            value = "0.0" if abs(values[y, x]) < .05 else f"{values[y, x]:+.1f}"
            ax.text(x, y, f"{value}\n{positive[y, x]}/5 ↑", ha="center", va="center",
                    fontsize=9, color=color)
    fig.colorbar(image, ax=ax, label="F1 difference (pp)", shrink=.85)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.text(.01, .005, "R = ROAD-Waymo; L = LOKI. n/5 = seeds with BiLSTM F1 above matched MLP.",
             ha="left", fontsize=8)
    for suffix in ("png", "svg"):
        fig.savefig(OUT / "figures" / f"per-class-difference.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    snapshot = read(SNAPSHOT)
    assert snapshot["head"] == "873ddd0198bc4d19c9691068c63bf019d5b2481b"
    sweep = snapshot["sweep"]
    assert sweep["status"] == "complete" and sweep["failures"] == []
    assert sweep["seeds"] == list(SEEDS) and sweep["epochs"] == 75 and sweep["patience"] == 8
    assert len(sweep["attempts"]) == 10
    assert all(a["status"] == "complete" and a["returncode"] == 0 for a in sweep["attempts"])
    new = {(a["files"]["config.json"]["json"]["source"], a["files"]["config.json"]["json"]["settings"]["seed"]): a
           for a in snapshot["attempts"]}
    assert len(new) == 10 and set(new) == {(source, seed) for source in SOURCES for seed in SEEDS}

    records, optim = [], {}
    data = {}
    metadata_hash = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
    metric_checks = 0
    for source in SOURCES:
        for seed in SEEDS:
            e001d = new[(source, seed)]
            new_files = e001d["files"]
            configs = {"matched MLP": (new_files["config.json"]["json"],
                                       new_files["provenance.json"]["json"],
                                       new_files["history.json"]["json"], e001d)}
            for model, variant in (("small MLP", "MLP"), ("BiLSTM", "BiLSTM")):
                old_dir = OLD / f"{source}-K+T+R-{variant}-seed{seed}"
                cfg, prov, hist = (read(old_dir / name) for name in
                                   ("config.json", "provenance.json", "history.json"))
                assert read(old_dir / "status.json")["status"] == "complete"
                compare_protocol(new_files["config.json"]["json"], new_files["provenance.json"]["json"], cfg, prov)
                assert prov["parameters"] == PARAMETERS[model]
                configs[model] = (cfg, prov, hist, old_dir)
            for model, (cfg, prov, hist, loc) in configs.items():
                if model == "matched MLP":
                    assert cfg["model"]["parameter_count"] == PARAMETERS[model]
                    assert cfg["model"]["input_dim"] == 12
                    assert cfg["model"]["framewise_block"] == [64, 194, 64]
                    assert cfg["model"]["recurrent_hidden"] is None
                    assert cfg["model"]["readout"] == [64, 4]
                    assert cfg["source"] == source and cfg["settings"]["seed"] == seed
                    assert cfg["variant"] == "mlp-matched" and cfg["configuration"] == "K+T+R"
                    assert prov["revision"] == snapshot["head"] and prov["diff_sha256"] == hashlib.sha256(b"").hexdigest()
                    metrics = metric_files_new(loc)
                else:
                    assert cfg["source"] == source and cfg["settings"]["seed"] == seed
                    assert cfg["configuration"] == "K+T+R"
                    assert cfg["variant"] == {"small MLP": "A", "BiLSTM": "B"}[model]
                    metrics = {target: read(loc / f"{target}-test/metrics.json") for target in TARGETS}
                assert set(metrics) == set(TARGETS)
                assert prov["parameters"] == PARAMETERS[model]
                run_history = history_summary(cfg, prov, hist)
                optim[(source, seed, model)] = run_history
                for target in TARGETS:
                    f1, ce, denom = check_metric(metrics[target])
                    target_support = prov["datasets"][target]["support"]["test"]
                    assert denom["groups"] == target_support["groups"]
                    assert denom["tracks"] == target_support["tracks"]
                    assert denom["gt_frames"] == target_support["gt_frames"]
                    assert denom["class_frames"] == target_support["class_frames"]
                    row = {"source": source, "seed": seed, "model": model, "target": target,
                           "direction": DIR_NAMES[(source, target)], "macro_f1_percent": float(f1.mean()*100),
                           "track_ce": ce, **{f"f1_{cls.lower()}_percent": float(v*100)
                                              for cls, v in zip(CLASSES, f1)},
                           **{f"test_{key}": value for key, value in denom.items()},
                           **run_history}
                    records.append(row)
                    data[(source, seed, model, target)] = row
                    metric_checks += 1

    assert metric_checks == 60
    primary, paired, per_class = {}, {}, {}
    for source, target in DIRECTIONS:
        for model in MODELS:
            values = [data[(source, seed, model, target)]["macro_f1_percent"] for seed in SEEDS]
            primary[(source, target, model)] = moments(values)
        for before, after in (("small MLP", "matched MLP"), ("matched MLP", "BiLSTM")):
            values = [data[(source, seed, after, target)]["macro_f1_percent"] -
                      data[(source, seed, before, target)]["macro_f1_percent"] for seed in SEEDS]
            paired[(source, target, f"{after} − {before}")] = moments(values) | {
                "positive_seeds": int(np.sum(np.asarray(values) > 0))}
        for cls in CLASSES:
            key = f"f1_{cls.lower()}_percent"
            values = [data[(source, seed, "BiLSTM", target)][key] -
                      data[(source, seed, "matched MLP", target)][key] for seed in SEEDS]
            per_class[(source, target, cls)] = moments(values) | {
                "positive_seeds": int(np.sum(np.asarray(values) > 0))}

    optimization = {}
    for source in SOURCES:
        for model in MODELS:
            for component in ("selected_training_ce", "selected_validation_ce",
                              "final_training_ce", "final_validation_ce"):
                optimization[(source, model, component)] = moments([
                    optim[(source, seed, model)][component] for seed in SEEDS])

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    csv_write(OUT / "per-seed.csv", records)
    plot_primary(records, primary)
    plot_classes(per_class)
    summary = {
        "metadata": {"remote_checkout": snapshot["remote_checkout"], "head": snapshot["head"],
                     "sweep_path": snapshot["sweep_path"], "sweep_id": Path(snapshot["sweep_path"]).parent.name,
                     "snapshot": str(SNAPSHOT.relative_to(ROOT)), "snapshot_sha256": metadata_hash,
                     "old_controls": str(OLD.relative_to(ROOT)), "old_review": "2026-10-15 E001 five-seed review"},
        "checks": {"sweep_complete": True, "new_attempts": len(new), "matched_old_controls": 20,
                   "metric_cells_and_confusions_checked": metric_checks,
                   "all_settings_split_setup_collection_policy_source_normalization_match": True,
                   "test_primary_support_counts_match_frozen_support": True,
                   "parameters": PARAMETERS,
                   "completed_epoch_range": [min(x["completed_epochs"] for x in optim.values()),
                                             max(x["completed_epochs"] for x in optim.values())],
                   "selected_epoch_range": [min(x["selected_epoch"] for x in optim.values()),
                                           max(x["selected_epoch"] for x in optim.values())],
                   "stopping_reasons": sorted(set(x["stopping_reason"] for x in optim.values()))},
        "primary_f1_percent": {f"{DIR_NAMES[(s,t)]} / {m}": v for (s,t,m),v in primary.items()},
        "paired_f1_difference_pp": {f"{DIR_NAMES[(s,t)]} / {label}": v for (s,t,label),v in paired.items()},
        "per_class_bilstm_minus_matched_mlp_pp": {f"{DIR_NAMES[(s,t)]} / {cls}": v
                                                   for (s,t,cls),v in per_class.items()},
        "optimization_ce": {f"{s} / {m} / {component}": v for (s,m,component),v in optimization.items()}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for direction in DIRECTIONS:
        label = DIR_NAMES[direction]
        print(label, "|", " | ".join(f"{model} {fmt(primary[(direction[0], direction[1], model)])}"
                                      for model in MODELS))
        print("  deltas:", " | ".join(f"{name} {fmt(paired[(direction[0], direction[1], name)])} pp "
                                      f"({paired[(direction[0], direction[1], name)]['positive_seeds']}/5)"
                                      for name in ("matched MLP − small MLP", "BiLSTM − matched MLP")))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
