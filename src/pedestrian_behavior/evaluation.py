"""Equal-track loss and reproducible confusion metrics; no experiment policy."""

import numpy as np
import torch
from torch.nn import functional as F


def track_cross_entropy(logits, targets, gt_valid, padding_mask):
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or gt_valid.shape != targets.shape or padding_mask.shape != targets.shape:
        raise ValueError("Incompatible loss shapes")
    context = ~padding_mask
    if not torch.isfinite(logits[context]).all():
        raise ValueError("Nonfinite context logits")
    accepted = gt_valid & context
    counts = accepted.sum(1)
    if (counts == 0).any():
        raise ValueError("Empty-GT track")
    y = targets[accepted]
    if ((y < 0) | (y >= logits.shape[-1])).any():
        raise ValueError("Invalid accepted target")
    losses = torch.zeros_like(targets, dtype=logits.dtype)
    losses[accepted] = F.cross_entropy(logits[accepted], y, reduction="none")
    result = losses.sum(1) / counts
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite track loss")
    return result


def confusion_metrics(confusion, supported_only=False):
    c = np.asarray(confusion, dtype=np.float64)
    support, predicted = c.sum(1), c.sum(0)
    diagonal = c.diagonal()
    divide = lambda a, b: np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    precision, recall = divide(diagonal, predicted), divide(diagonal, support)
    f1 = divide(2*diagonal, support+predicted)
    supported = support > 0
    scored = supported if supported_only else np.ones(len(c), dtype=bool)
    total = float(support.sum())
    return {"confusion": c.tolist(), "support": support.tolist(),
            "supported_classes": np.flatnonzero(supported).tolist(),
            "precision": [float(v) if not supported_only or s else None for v, s in zip(precision, supported)],
            "recall": [float(v) if not supported_only or s else None for v, s in zip(recall, supported)],
            "f1": [float(v) if not supported_only or s else None for v, s in zip(f1, supported)],
            "macro_f1": float(f1[scored].mean()) if scored.any() and total else None,
            "support_weighted_f1": float((f1*support).sum()/total) if total else None,
            "accuracy": float(diagonal.sum()/total) if total else None}


def prediction_metrics(tracks, classes, memberships=None, supported_only=False):
    raw = np.zeros((classes, classes), dtype=np.float64)
    weighted = raw.copy()
    accuracies, losses, groups = [], [], set()
    context_frames, class_tracks = 0, np.zeros(classes, dtype=int)
    for i, track in enumerate(tracks):
        logits = np.asarray(track["logits"])
        y, gt = np.asarray(track["targets"]), np.asarray(track["gt_valid"], dtype=bool)
        if logits.shape != (len(y), classes) or gt.shape != y.shape or not np.isfinite(logits).all():
            raise ValueError("Invalid/nonfinite predictions")
        if not gt.any():
            raise ValueError("Empty-GT track")
        if ((y[gt] < 0) | (y[gt] >= classes)).any():
            raise ValueError("Invalid accepted target")
        selected = gt if memberships is None else gt & memberships[i]
        if not selected.any():
            continue
        label, scores = y[selected], logits[selected].astype(np.float64)
        predicted = scores.argmax(1)
        counts = np.zeros_like(raw)
        np.add.at(counts, (label, predicted), 1)
        raw += counts
        weighted += counts / len(label)
        class_tracks += counts.sum(1) > 0
        accuracies.append(float((label == predicted).mean()))
        shifted = scores - scores.max(1, keepdims=True)
        ce = np.log(np.exp(shifted).sum(1))-shifted[np.arange(len(label)), label]
        if not np.isfinite(ce).all():
            raise ValueError("Nonfinite evaluation loss")
        losses.append(float(ce.mean()))
        groups.add(track["group"])
        context_frames += len(y)
    return {"raw": confusion_metrics(raw, supported_only), "track_weighted": confusion_metrics(weighted, supported_only),
            "track_ce": float(np.mean(losses)) if losses else None,
            "mean_track_accuracy": float(np.mean(accuracies)) if accuracies else None,
            "median_track_accuracy": float(np.median(accuracies)) if accuracies else None,
            "denominators": {"groups": len(groups), "tracks": len(losses), "context_frames": context_frames,
                             "gt_frames": int(raw.sum()), "class_frames": raw.sum(1).astype(int).tolist(),
                             "class_tracks": class_tracks.tolist()}}


@torch.no_grad()
def predict(model, batches, device):
    model.eval()
    tracks = []
    for batch in batches:
        logits = model(batch["inputs"].to(device), batch["lengths"])
        track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device), batch["padding_mask"].to(device))
        for i, length in enumerate(batch["lengths"].tolist()):
            tracks.append({"logits": logits[i, :length].cpu().numpy().astype(np.float32),
                           "targets": batch["targets"][i, :length].numpy(), "gt_valid": batch["gt_valid"][i, :length].numpy(),
                           "locator": batch["locators"][i], "archive": batch["archives"][i]})
    if not tracks:
        raise ValueError("Empty evaluation split")
    return tracks


def save_predictions(path, tracks):
    """Unpadded pickle-free arrays, with enough metadata to recompute all scores."""
    import json
    offsets = np.cumsum([0]+[len(t["targets"]) for t in tracks], dtype=np.int64)
    fields = ("logits", "targets", "gt_valid")
    saved = {key: np.concatenate([t[key] for t in tracks]) for key in fields}
    saved["offsets"] = offsets
    metadata = [{k: v for k, v in t.items() if k not in fields and k != "conditions" and k != "source_flags"} for t in tracks]
    saved["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    for axis in tracks[0]["conditions"]:
        saved["condition_"+axis] = np.concatenate([t["conditions"][axis] for t in tracks])
    for flag in tracks[0]["source_flags"]:
        saved["source_"+flag] = np.concatenate([t["source_flags"][flag] for t in tracks])
    np.savez_compressed(path, **saved)


def load_predictions(path):
    import json
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        tracks = []
        for i, entry in enumerate(metadata):
            start, end = saved["offsets"][i:i+2]
            track = entry | {k: saved[k][start:end].copy() for k in ("logits", "targets", "gt_valid")}
            track["conditions"] = {k.removeprefix("condition_"): saved[k][start:end].copy() for k in saved.files if k.startswith("condition_")}
            track["source_flags"] = {k.removeprefix("source_"): saved[k][start:end].copy() for k in saved.files if k.startswith("source_")}
            tracks.append(track)
    return tracks


def stratified_metrics(tracks, classes, axes):
    return {axis: {label: prediction_metrics(tracks, classes,
                    [t["conditions"][axis] == i for t in tracks], supported_only=True)
                  for i, label in enumerate(labels)} for axis, labels in axes.items()}
