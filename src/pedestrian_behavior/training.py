"""Shared deterministic training, source-validation selection and atomic checkpoints."""

import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .evaluation import track_cross_entropy


def seed_run(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"seed": seed, "deterministic_algorithms": True, "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False, "loader_workers": 0,
            "cross_platform_identity_promised": False}


def select_epoch(best, f1, ce, epoch, stale):
    if not np.isfinite([f1, ce]).all():
        raise ValueError("Nonfinite validation evidence")
    improvement = best is None or f1 > best["f1"]
    selected = improvement or (f1 == best["f1"] and ce < best["ce"])
    if selected:
        best = {"epoch": epoch, "f1": f1, "ce": ce}
    return best, 0 if improvement else stale+1, selected


def save_checkpoint(path, model, evidence):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(evidence | {"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, temporary)
    os.replace(temporary, path)


def reload_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def train_epoch(model, batches, optimizer, device, clip_norm, on_step=None, step_offset=0):
    model.train()
    started = time.monotonic()
    total_loss, tracks, steps, slots, gt_slots, padding, capacity = 0., 0, 0, 0, 0, 0, 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["inputs"].to(device), batch["lengths"])
        losses = track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device), batch["padding_mask"].to(device))
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite model parameters")
        loss_sum = float(losses.detach().double().sum())
        total_loss += loss_sum
        tracks += len(losses)
        steps += 1
        slots += int(batch["lengths"].sum())
        gt_slots += int(batch["gt_valid"].sum())
        padding += int(batch["padding_mask"].sum())
        capacity += batch["padding_mask"].numel()
        if on_step is not None:
            on_step(step_offset+steps, loss_sum/len(losses))
    if not tracks:
        raise ValueError("Empty training split")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"training_ce": total_loss/tracks, "optimizer_steps": steps, "processed_tracks": tracks,
            "context_slots": slots, "gt_slots": gt_slots, "padding_slots": padding,
            "batch_slots": capacity, "padding_fraction": padding/capacity, "seconds": time.monotonic()-started}


def fit(model, batches, validate, optimizer, device, output, checkpoint_references, on_epoch,
        epochs, patience, clip_norm, on_step=None):
    history, best, stale, step_offset = [], None, 0, 0
    for epoch in range(1, epochs+1):
        row = {"epoch": epoch} | train_epoch(model, batches, optimizer, device, clip_norm, on_step=on_step, step_offset=step_offset)
        step_offset += row["optimizer_steps"]
        validation = validate(model)
        row.update(validation_ce=validation["track_ce"], validation_f1=validation["track_weighted"]["macro_f1"])
        best, stale, selected = select_epoch(best, row["validation_f1"], row["validation_ce"], epoch, stale)
        reason = "patience" if stale >= patience else "budget" if epoch == epochs else None
        row.update(selected=selected, non_improving_epochs=stale, stopping_reason=reason)
        history.append(row)
        evidence = checkpoint_references | {"epoch": epoch, "selection": best, "validation": validation,
                                            "stopping_reason": reason}
        if selected:
            save_checkpoint(Path(output)/"best.pt", model, evidence)
        save_checkpoint(Path(output)/"last.pt", model, evidence)
        on_epoch(row, history)
        if reason:
            break
    return {"history": history, "selection": best, "stopping_reason": reason}
