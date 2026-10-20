"""Fixed training-only PCA colors for aligned frozen point features."""

import numpy as np
import torch


def fit_pca(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or min(features.shape) < 3 or not np.isfinite(features).all():
        raise ValueError("PCA requires at least three finite rows and channels")
    mean = features.mean(0)
    centered = features-mean
    values, vectors = np.linalg.eigh(centered.T @ centered/(len(features)-1))
    components = vectors[:, -3:][:, ::-1].copy()
    # Resolve arbitrary eigenvector signs once; colors never refit per frame.
    components *= np.where(components[np.abs(components).argmax(0), np.arange(3)] < 0, -1, 1)
    scores = centered @ components
    low, high = np.percentile(scores, [1, 99], axis=0)
    return {"mean": mean.astype(np.float32), "components": components.astype(np.float32),
            "low": low.astype(np.float32), "high": high.astype(np.float32),
            "explained_variance_ratio": (np.maximum(values[-3:][::-1], 0)/max(values.clip(0).sum(), 1e-30)).astype(np.float32)}


def pca_colors(features, pca):
    """Apply saved PCA and fixed 1st/99th-percentile color bounds on CPU/GPU."""
    if features.ndim != 2 or features.shape[1] != len(pca["mean"]) or not torch.isfinite(features).all():
        raise ValueError("Invalid features for frozen PCA")
    mean, axes, low, high = [torch.as_tensor(pca[k], device=features.device, dtype=features.dtype)
                            for k in ("mean", "components", "low", "high")]
    scores = (features-mean) @ axes
    return ((scores-low)/(high-low).clamp_min(1e-6)).clamp(0, 1)


def calibration_frames(source, dataset):
    """Two existing probe frames in each of four training scenes, no GT ranking."""
    groups = {}
    for row in source["frames"]:
        if row["dataset"] == dataset and row["split"] == "training":
            groups.setdefault(row["group"], set()).add(row["key"])
    eligible = sorted(g for g, keys in groups.items() if len(keys) >= 2)[:4]
    if len(eligible) != 4:
        raise ValueError("Need four training calibration scenes per dataset")
    return [(g, k) for g in eligible for k in sorted(groups[g], key=int)[:2]]
