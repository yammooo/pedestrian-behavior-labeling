"""PCA and UMAP views of the frozen E003b validation representations."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


DATASETS = ("road-waymo", "loki")
MODELS = DATASETS
CLASSES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
COLORS = ("#277da1", "#f8961e", "#43aa8b", "#d1495b")
DATASET_COLORS = {"road-waymo": "#355070", "loki": "#e76f51"}
LABELS = {"road-waymo": "ROAD-Waymo", "loki": "LOKI"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_cache(path, model, dataset):
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "targets", "gt_valid", "offsets", "locators", "archives", "metadata"}
        if not required.issubset(archive.files):
            raise ValueError(f"Incomplete frozen feature cache: {path}")
        arrays = {key: archive[key] for key in required}
    metadata = json.loads(str(arrays.pop("metadata")))
    target = metadata.get("dataset", metadata.get("target"))
    if (metadata.get("source") != model or target != dataset or metadata.get("split") != "validation"):
        raise ValueError(f"Feature-cache lineage mismatch: {path}")
    features, targets, gt_valid, offsets = (arrays[key] for key in ("features", "targets", "gt_valid", "offsets"))
    locators, archives = arrays["locators"], arrays["archives"]
    if (features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != 128
            or not np.isfinite(features).all() or targets.dtype != np.int64 or gt_valid.dtype != np.bool_
            or targets.shape != (len(features),) or gt_valid.shape != targets.shape
            or offsets.dtype != np.int64 or offsets[0] != 0 or offsets[-1] != len(features)
            or len(offsets) != len(locators) + 1 or len(archives) != len(locators)
            or np.any(np.diff(offsets) <= 0)
            or np.any((targets[gt_valid] < 0) | (targets[gt_valid] >= len(CLASSES)))):
        raise ValueError(f"Invalid frozen feature cache arrays: {path}")
    tracks = {}
    for i, locator in enumerate(locators):
        start, end = offsets[i:i + 2]
        key = str(locator)
        if key in tracks or not gt_valid[start:end].any():
            raise ValueError(f"Duplicate locator or GT-empty validation track in {path}")
        tracks[key] = {"features": features[start:end], "targets": targets[start:end],
            "gt_valid": gt_valid[start:end], "archive": str(archives[i])}
    return {"tracks": tracks, "metadata": metadata, "path": str(Path(path).resolve()), "sha256": _sha256(path)}


def _sample_validation(track_maps, max_per_track=10):
    """Choose accepted frames evenly per track, then equalize total dataset counts."""
    candidates = {}
    for dataset in DATASETS:
        rows = []
        for locator in sorted(track_maps[dataset]):
            track = track_maps[dataset][locator]
            valid = np.flatnonzero(track["gt_valid"])
            count = min(max_per_track, len(valid))
            for frame in valid[np.linspace(0, len(valid) - 1, count, dtype=np.int64)]:
                rows.append((locator, int(frame), int(track["targets"][frame]), track["archive"]))
        candidates[dataset] = rows
    total = min(map(len, candidates.values()))
    if total == 0:
        raise ValueError("Both validation datasets need accepted GT frames")
    selected = {}
    for dataset, rows in candidates.items():
        indices = np.linspace(0, len(rows) - 1, total, dtype=np.int64)
        selected[dataset] = [rows[i] for i in indices]
        counts = Counter(locator for locator, *_ in selected[dataset])
        if max(counts.values(), default=0) > max_per_track:
            raise ValueError("Balanced sampling exceeded the per-track frame cap")
    return selected


def _features_for_sample(model, samples, caches):
    rows = []
    for dataset in DATASETS:
        tracks = caches[(model, dataset)]["tracks"]
        for locator, frame, _label, _archive in samples[dataset]:
            track = tracks[locator]
            if frame >= len(track["features"]) or not track["gt_valid"][frame]:
                raise ValueError("Sampled frame does not have accepted GT in every model cache")
            rows.append(track["features"][frame])
    features = np.asarray(rows, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 128 or not np.isfinite(features).all():
        raise ValueError(f"Invalid sampled representation for {model}")
    return features


def _flatten_samples(samples):
    return [(dataset, *row) for dataset in DATASETS for row in samples[dataset]]


def _plot_projection(coordinates, samples, output, *, title, model_labels,
                     panel_descriptions=None, sensitivity=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    configs = list(coordinates)
    rows = len(configs)
    fig, axes = plt.subplots(rows, 2, figsize=(13, max(5, rows * (3.3 if sensitivity else 4.3))), squeeze=False)
    sample_rows = _flatten_samples(samples)
    dataset_values = np.asarray([row[0] for row in sample_rows])
    labels = np.asarray([row[3] for row in sample_rows], dtype=np.int64)
    draw_order = np.random.default_rng(0).permutation(len(sample_rows))
    for row_index, config in enumerate(configs):
        coords = coordinates[config]
        if coords.shape != (len(sample_rows), 2) or not np.isfinite(coords).all():
            raise ValueError(f"Invalid saved projection coordinates: {config}")
        prefix = model_labels[config[0]]
        description = (panel_descriptions or {}).get(config[0], "")
        if sensitivity:
            axes[row_index, 0].set_title(f"{prefix}, {config[1]}: behavior")
            axes[row_index, 1].set_title(f"{prefix}, {config[1]}: dataset")
        else:
            axes[row_index, 0].set_title(f"{prefix}: behavior {description}".strip())
            axes[row_index, 1].set_title(f"{prefix}: dataset {description}".strip())
        class_colors = np.asarray(COLORS)[labels[draw_order]]
        dataset_colors = np.asarray([DATASET_COLORS[value] for value in dataset_values[draw_order]])
        axes[row_index, 0].scatter(coords[draw_order, 0], coords[draw_order, 1], s=4, alpha=.45,
            c=class_colors, linewidths=0, rasterized=True)
        axes[row_index, 1].scatter(coords[draw_order, 0], coords[draw_order, 1], s=4, alpha=.45,
            c=dataset_colors, linewidths=0, rasterized=True)
        if row_index == 0:
            axes[row_index, 0].legend(handles=[Line2D([], [], marker="o", linestyle="", color=COLORS[i],
                label=label, markersize=5) for i, label in enumerate(CLASSES)],
                title="Behavior", fontsize=8, title_fontsize=8)
            axes[row_index, 1].legend(handles=[Line2D([], [], marker="o", linestyle="", color=DATASET_COLORS[key],
                label=LABELS[key], markersize=5) for key in DATASETS],
                title="Dataset", fontsize=8, title_fontsize=8)
        for ax in axes[row_index]:
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(alpha=.12)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, .98))
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def analyze(input_root, output):
    input_root, output = Path(input_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    names = {("road-waymo", "road-waymo"): "road-waymo-on-road-waymo.npz",
        ("road-waymo", "loki"): "road-waymo-on-loki.npz",
        ("loki", "road-waymo"): "loki-on-road-waymo.npz",
        ("loki", "loki"): "loki-on-loki.npz"}
    caches = {(model, dataset): _read_cache(input_root / name, model, dataset)
        for (model, dataset), name in names.items()}
    track_maps = {dataset: caches[(DATASETS[0], dataset)]["tracks"] for dataset in DATASETS}
    for dataset in DATASETS:
        reference = track_maps[dataset]
        other = caches[(DATASETS[1], dataset)]["tracks"]
        if set(reference) != set(other):
            raise ValueError(f"Validation track identities differ between frozen models on {dataset}")
        for locator in reference:
            first, second = reference[locator], other[locator]
            if (first["archive"] != second["archive"] or not np.array_equal(first["targets"], second["targets"])
                    or not np.array_equal(first["gt_valid"], second["gt_valid"])):
                raise ValueError(f"Validation GT/context differs between frozen models: {locator}")
    # Require the source checkpoint lineage to match between each model's two datasets.
    checkpoint_hashes, normalizations = {}, {}
    for model in MODELS:
        hashes, norms = set(), set()
        for dataset in DATASETS:
            metadata = caches[(model, dataset)]["metadata"]
            checkpoint = metadata.get("checkpoint_sha256") or metadata.get("checkpoint", {}).get("sha256")
            normalization = metadata.get("source_normalization")
            if not checkpoint or normalization is None:
                raise ValueError(f"Frozen checkpoint/normalization lineage is missing for {model} on {dataset}")
            hashes.add(checkpoint)
            norms.add(json.dumps(normalization, sort_keys=True, allow_nan=False))
        if len(hashes) != 1 or len(norms) != 1:
            raise ValueError(f"Frozen checkpoint lineage is missing or inconsistent for {model}")
        checkpoint_hashes[model] = hashes.pop()
        normalizations[model] = json.loads(norms.pop())

    samples = _sample_validation(track_maps)
    # The shared IDs make every row compare the same frames without class balancing.
    sample_rows = _flatten_samples(samples)
    ids = {"dataset": np.asarray([row[0] for row in sample_rows], dtype=np.str_),
        "locators": np.asarray([row[1] for row in sample_rows], dtype=np.str_),
        "frame_indices": np.asarray([row[2] for row in sample_rows], dtype=np.int64),
        "targets": np.asarray([row[3] for row in sample_rows], dtype=np.int64),
        "archives": np.asarray([row[4] for row in sample_rows], dtype=np.str_)}
    features = {model: _features_for_sample(model, samples, caches) for model in MODELS}
    from sklearn.decomposition import PCA
    import umap

    coordinates, pca_variance, umap_parameters = {}, {}, {}
    for model in MODELS:
        print(f"{model}: PCA on {len(features[model])} sampled frames", flush=True)
        pca = PCA(n_components=2, svd_solver="full")
        coordinates[(model, "pca")] = pca.fit_transform(features[model]).astype(np.float32)
        pca_variance[model] = pca.explained_variance_ratio_.tolist()
        for neighbors in (15, 50):
            for seed in (0, 1):
                key = f"umap-n{neighbors}-s{seed}"
                print(f"{model}: fitting {key}", flush=True)
                started = time.perf_counter()
                projector = umap.UMAP(n_components=2, n_neighbors=neighbors, min_dist=.1,
                    metric="euclidean", random_state=seed, transform_seed=seed, n_jobs=1)
                coordinates[(model, key)] = projector.fit_transform(features[model]).astype(np.float32)
                umap_parameters[key] = projector.get_params(deep=False)
                print(f"  completed in {time.perf_counter() - started:.1f}s", flush=True)

    output.mkdir(parents=True)
    arrays = dict(ids)
    for (model, name), coords in coordinates.items():
        arrays[f"{name.replace('-', '_')}__{model.replace('-', '_')}"] = coords
    coords_path = output / "projection-coordinates.npz"
    with coords_path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(coords_path.with_suffix(".tmp"), coords_path)
    model_labels = {model: f"{LABELS[model]}-trained E002d" for model in MODELS}
    pca_descriptions = {model: (f"(PC1 {100*pca_variance[model][0]:.1f}%, "
        f"PC2 {100*pca_variance[model][1]:.1f}%)") for model in MODELS}
    _plot_projection({(model, "pca"): coordinates[(model, "pca")] for model in MODELS},
        samples, output / "pca", title="PCA of frozen post-LayerNorm E002d features",
        model_labels=model_labels, panel_descriptions=pca_descriptions)
    _plot_projection({(model, "UMAP neighbors=15, seed=0"): coordinates[(model, "umap-n15-s0")] for model in MODELS},
        samples, output / "umap", title="UMAP of frozen post-LayerNorm E002d features (neighbors=15, seed=0)",
        model_labels=model_labels)
    sensitivity = {(model, label): coordinates[(model, key)] for model in MODELS
        for key, label in (("umap-n15-s1", "neighbors=15, seed=1"),
            ("umap-n50-s0", "neighbors=50, seed=0"), ("umap-n50-s1", "neighbors=50, seed=1"))}
    _plot_projection(sensitivity, samples, output / "umap-sensitivity",
        title="UMAP sensitivity to neighbors and seed", model_labels=model_labels, sensitivity=True)

    cache_provenance = {f"{model}-on-{dataset}": {"path": Path(cache["path"]).name,
        "sha256": cache["sha256"], "checkpoint_sha256": checkpoint_hashes[model]}
        for (model, dataset), cache in caches.items()}
    sample_counts = {}
    for dataset in DATASETS:
        sample_counts[dataset] = {"tracks": len({row[0] for row in samples[dataset]}),
            "frames": len(samples[dataset]), "classes": {CLASSES[i]: int(np.sum(ids["targets"][ids["dataset"] == dataset] == i))
                for i in range(len(CLASSES))},
            "accepted_gt_frames": int(sum(track["gt_valid"].sum() for track in track_maps[dataset].values())),
            "validation_tracks": len(track_maps[dataset])}
    provenance = {"scope": "frozen E002d source models on ROAD-Waymo and LOKI validation only",
        "analysis_code_sha256": _sha256(Path(__file__)),
        "held_out_tests_accessed": False, "training_or_adaptation": False,
        "features": "post-final-LayerNorm 128-D; no additional scaling",
        "sampling": {"accepted_gt_only": True, "maximum_frames_per_track": 10,
            "within_track": "evenly spaced in accepted-GT frame order",
            "dataset_balance": "same total sampled frame count per dataset by deterministic even subsampling",
            "class_balancing": False, "seed": 0},
        "sample_counts": sample_counts, "frozen_checkpoints": checkpoint_hashes,
        "source_normalization": normalizations,
        "input_caches": cache_provenance,
        "pca": {model: {"solver": "full", "components": 2, "whiten": False,
            "explained_variance_ratio": pca_variance[model]} for model in MODELS},
        "umap": {"metric": "euclidean", "min_dist": .1, "neighbors": [15, 50],
            "seeds": [0, 1], "n_components": 2, "n_jobs": 1, "parameters": umap_parameters},
        "software": {"numpy": np.__version__, "scikit_learn": __import__("sklearn").__version__,
            "umap_learn": umap.__version__},
        "coordinates_sha256": _sha256(coords_path)}
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")

