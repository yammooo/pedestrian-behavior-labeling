"""Visualize dataset structure in the joint ROAD–LOKI pre-head representations."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.experiments.e003b_embedding_diagnostic import (
    CLASSES as LABEL_CLASSES,
    COLORS,
    DATASETS as SOURCES,
    DATASET_COLORS,
    LABELS,
)


SOURCE_NAMES = tuple(LABELS[source] for source in SOURCES)
CLASSES = tuple(name.replace("_", " ").capitalize() for name in LABEL_CLASSES)
COHORTS = {"balanced": "Class-balanced validation", "both": "RGB and LiDAR both present"}
TRACK_CAP, GROUP_CAP, CELL_CAP = 5, 40, 600


def sample(domains, labels, tracks, groups, eligible):
    cells = {(d, y): [] for d in range(len(SOURCES)) for y in range(len(CLASSES))}
    by_track, by_group = Counter(), Counter()
    for i in np.random.default_rng(0).permutation(np.flatnonzero(eligible)):
        if by_track[tracks[i]] >= TRACK_CAP or by_group[groups[i]] >= GROUP_CAP:
            continue
        by_track[tracks[i]] += 1
        by_group[groups[i]] += 1
        cells[domains[i], labels[i]].append(i)
    per_cell = min(CELL_CAP, *(len(cell) for cell in cells.values()))
    if not per_cell:
        raise ValueError("Every dataset × behavior cell must be represented")
    selected = np.array([i for cell in cells.values() for i in cell[:per_cell]])
    return np.random.default_rng(0).permutation(selected)


def probe(features, flags, domains, labels, groups):
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(
        features, domains * len(CLASSES) + labels, groups))

    def score(values, population):
        predictions = np.full(len(domains), -1, dtype=np.int64)
        for train, test in folds:
            train, test = train[population[train]], test[population[test]]
            if len(test):
                classifier = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
                classifier.fit(values[train], domains[train])
                predictions[test] = classifier.predict(values[test])
        assert (predictions[population] >= 0).all()
        return float(balanced_accuracy_score(domains[population], predictions[population]))

    all_rows = np.ones(len(domains), dtype=bool)
    return {
        "embeddings": score(features, all_rows),
        "availability_only": score(flags, all_rows),
        "within_class_embeddings": {name: score(features, labels == i) for i, name in enumerate(CLASSES)},
    }


def plot(method, panels, domains, labels, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for row, (cohort, (indices, xy, names)) in enumerate(panels.items()):
        for column, (values, colors, legend) in enumerate((
            (domains, tuple(DATASET_COLORS[s] for s in SOURCES), SOURCE_NAMES),
            (labels, COLORS, CLASSES),
        )):
            ax = axes[row, column]
            classes = values[indices]
            ax.scatter(xy[:, 0], xy[:, 1], c=np.asarray(colors)[classes], s=7, alpha=.52,
                       linewidths=0, rasterized=True)
            ax.set_title(f"{COHORTS[cohort]} · {'Dataset' if column == 0 else 'Behavior'}", fontsize=11)
            ax.set(xlabel=names[0], ylabel=names[1])
            ax.grid(alpha=.13)
            if row == 0:
                handles = [plt.Line2D([], [], marker="o", linestyle="none", color=c, label=l, markersize=5)
                           for c, l in zip(colors, legend, strict=True)]
                ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(f"E006a joint checkpoint · {method.upper()} of 128-D pre-head features")
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{method}.{suffix}", dpi=180)
    plt.close(fig)


def run(inputs, output):
    with np.load(inputs, allow_pickle=False) as archive:
        features, datasets, groups, tracks, labels, rgb, lidar = (
            archive[key] for key in ("features", "datasets", "groups", "locators", "labels", "rgb_valid", "lidar_valid"))
    if features.shape != (len(labels), 128) or not np.isfinite(features).all() or set(datasets) != set(SOURCES):
        raise ValueError("Expected finite 128-D embeddings from ROAD and LOKI")
    domains = (datasets == SOURCES[1]).astype(np.int64)
    groups = np.char.add(np.char.add(datasets, ":"), groups.astype(str))
    tracks = np.char.add(np.char.add(datasets, ":"), tracks.astype(str))
    rgb, lidar = rgb.astype(bool), lidar.astype(bool)
    flags = np.column_stack((rgb, lidar))
    output.mkdir(parents=True, exist_ok=False)
    report = {"input": str(inputs), "input_sha256": checksum(inputs), "seed": 0,
              "per_track_cap": TRACK_CAP, "per_recording_cap": GROUP_CAP, "per_cell_cap": CELL_CAP,
              "probe_folds": 5, "umap": {"n_neighbors": 15, "min_dist": .1}, "cohorts": {}}
    arrays = {}
    panels = {"pca": {}, "umap": {}}
    for cohort, mask in (("balanced", np.ones(len(labels), dtype=bool)), ("both", rgb & lidar)):
        i = sample(domains, labels, tracks, groups, mask)
        x = features[i]
        pca = PCA(n_components=2, svd_solver="full")
        pc = pca.fit_transform(x)
        explained = pca.explained_variance_ratio_.tolist()
        embedding = UMAP(n_neighbors=15, min_dist=.1, random_state=0, n_jobs=1).fit_transform(x)
        names = [f"PC{n} ({v:.1%})" for n, v in enumerate(explained, start=1)]
        panels["pca"][cohort] = i, pc, names
        panels["umap"][cohort] = i, embedding, ("UMAP 1", "UMAP 2")
        arrays.update({f"{cohort}_rows": i, f"{cohort}_pca": pc, f"{cohort}_umap": embedding})
        report["cohorts"][cohort] = {
            "samples": len(i), "samples_per_dataset_class": len(i) // 8,
            "recordings": {s: len(set(groups[i][datasets[i] == s])) for s in SOURCES},
            "tracks": {s: len(set(tracks[i][datasets[i] == s])) for s in SOURCES},
            "pca_explained_variance_ratio": explained,
            "domain_balanced_accuracy": probe(x, flags[i], domains[i], labels[i], groups[i]),
        }
    np.savez_compressed(output / "projection-coordinates.npz", **arrays)
    for method, values in panels.items():
        plot(method, values, domains, labels, output)
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["cohorts"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.input, args.output)
