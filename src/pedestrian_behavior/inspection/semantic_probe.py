"""Source-only standardization and a small frozen-feature linear probe."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from pedestrian_behavior.training import reload_checkpoint, save_checkpoint, seed_run
from .segmentation import semantic_confusion, semantic_metrics


def standardization(features):
    x = np.asarray(features)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty training features")
    return {"mean": x.mean(0, dtype=np.float64).astype(np.float32),
            "std": np.maximum(x.std(0, dtype=np.float64), 1e-6).astype(np.float32)}


def evaluate(head, features, targets, batch_size=65536):
    head.eval()
    predictions, ce = [], 0.
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            logits = head(features[start:start+batch_size])
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite linear probe logits")
            ce += float(nn.functional.cross_entropy(logits, targets[start:start+batch_size], reduction="sum"))
            predictions.append(logits.argmax(1).cpu().numpy())
    predictions = np.concatenate(predictions)
    metrics = semantic_metrics(semantic_confusion(targets.cpu().numpy(), predictions, head.out_features))
    return metrics | {"ce": ce/len(features)}, predictions


def fit_linear(train_x, train_y, val_x, val_y, classes, device, output, settings, on_epoch):
    """Only an affine head is optimized; input features are detached constants."""
    for x, y in ((train_x, train_y), (val_x, val_y)):
        if (x.ndim != 2 or x.shape[1] != train_x.shape[1] or not len(x)
                or y.shape != (len(x),) or not np.isfinite(x).all()
                or not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= classes).any()):
            raise ValueError("Invalid linear-probe features/targets")
    deterministic = seed_run(0)
    stats = standardization(train_x)
    x, vx = (torch.as_tensor((a-stats["mean"])/stats["std"], device=device) for a in (train_x, val_x))
    y, vy = (torch.as_tensor(a, dtype=torch.long, device=device) for a in (train_y, val_y))
    head = nn.Linear(x.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(0)
    history, best, stale = [], None, 0
    for epoch in range(1, settings["epochs"]+1):
        head.train()
        order = torch.randperm(len(x), generator=generator)
        ce, steps = 0., 0
        for start in range(0, len(x), settings["batch_size"]):
            indices = order[start:start+settings["batch_size"]].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x[indices]), y[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite linear-probe loss")
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            ce += float(loss.detach())*len(indices)
            steps += 1
        validation, _ = evaluate(head, vx, vy)
        score = validation["supported_macro_iou"]
        strict = best is None or score > best["iou"]
        selected = strict or (score == best["iou"] and validation["ce"] < best["ce"])
        stale = 0 if strict else stale+1
        if selected:
            best = {"epoch": epoch, "iou": score, "ce": validation["ce"]}
        reason = "patience" if stale >= settings["patience"] else "budget" if epoch == settings["epochs"] else None
        row = {"epoch": epoch, "training_ce": ce/len(x), "validation_ce": validation["ce"],
               "validation_macro_iou": score, "selected": selected, "stale": stale, "steps": steps, "stopping_reason": reason}
        history.append(row)
        evidence = {"selection": best, "epoch": epoch, "normalization": {k:torch.from_numpy(v) for k,v in stats.items()},
                    "settings": settings, "classes": classes, "input_channels": train_x.shape[1], "stopping_reason": reason}
        if selected:
            save_checkpoint(Path(output)/"best.pt", head, evidence)
        save_checkpoint(Path(output)/"last.pt", head, evidence)
        on_epoch(row, history)
        if reason:
            break
    reload_checkpoint(Path(output)/"best.pt", head, device)
    metrics, predictions = evaluate(head, vx, vy)
    return head, stats, {"selection": best, "history": history, "validation": metrics,
                        "stopping_reason": reason, "deterministic": deterministic}, predictions
