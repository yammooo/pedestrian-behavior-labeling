"""Equal-track accumulation and update-based warmup; no experiment settings."""

import time

import torch

from .evaluation import track_cross_entropy


def warmup_lr(peak, step, warmup_steps):
    if step < 1 or warmup_steps < 1:
        raise ValueError("Positive update and warmup lengths required")
    return peak * min(step / warmup_steps, 1.)


def train_accumulated_epoch(model, batches, optimizer, device, clip_norm, on_step=None, step_offset=0, *, peak_lr, warmup_steps):
    model.train()
    begin = time.monotonic()
    total_loss, tracks, steps, slots, gt, padding, capacity, physical, tokens = 0., 0, 0, 0, 0, 0, 0, 0, 0
    for indices in batches:
        count, update_loss, seen = len(indices), 0., 0
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = warmup_lr(peak_lr, step_offset + steps + 1, warmup_steps)
        for batch in batches.microbatches(indices):
            logits = model(batch)
            losses = track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device), batch["padding_mask"].to(device))
            (losses.sum() / count).backward()
            value = float(losses.detach().double().sum())
            update_loss += value
            seen += len(losses)
            slots += int(batch["lengths"].sum())
            gt += int(batch["gt_valid"].sum())
            padding += int(batch["padding_mask"].sum())
            capacity += batch["padding_mask"].numel()
            tokens += sum(len(x) for x in batch.get("points", ()))
            physical += 1
            del logits, losses, batch
        if seen != count or count < 1:
            raise ValueError("Physical packing changed logical update membership")
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite model parameters")
        total_loss += update_loss
        tracks += count
        steps += 1
        if on_step:
            on_step(step_offset + steps, update_loss / count)
    if not tracks:
        raise ValueError("Empty training split")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"training_ce": total_loss / tracks, "optimizer_steps": steps, "processed_tracks": tracks,
            "context_slots": slots, "gt_slots": gt, "padding_slots": padding, "batch_slots": capacity,
            "padding_fraction": padding / capacity, "physical_microbatches": physical,
            "point_tokens": tokens, "seconds": time.monotonic() - begin,
            "learning_rate": optimizer.param_groups[0]["lr"]}
