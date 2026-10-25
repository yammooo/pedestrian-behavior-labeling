"""Sequential optimizer accumulation over shared immutable physical batches."""

from contextlib import closing
import time

import torch

from .accumulation import warmup_lr
from .evaluation import track_cross_entropy


def rng_state(device):
    device = torch.device(device)
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def set_rng_state(state, device):
    device = torch.device(device)
    cpu, cuda = state
    torch.set_rng_state(cpu)
    if device.type == "cuda":
        torch.cuda.set_rng_state(cuda, device)


def train_shared_accumulated_epoch(models, batches, optimizers, device, clip_norm, *, rng_states,
                                   peak_lr, warmup_steps, step_offsets=None, on_step=None, order=None):
    """Train independent models on each prepared physical batch before releasing it."""
    names = list(models)
    order = list(order or names)
    if not names or set(order) != set(names) or len(order) != len(names):
        raise ValueError("Model execution order must name every active model exactly once")
    if set(optimizers) != set(names) or set(rng_states) != set(names):
        raise ValueError("Each model needs its own optimizer and RNG state")
    device = torch.device(device)
    step_offsets = step_offsets or {name: 0 for name in names}
    on_step = on_step or {}
    models = {name: models[name] for name in order}
    total_loss = {name: 0. for name in names}
    processed = {name: 0 for name in names}
    steps = {name: 0 for name in names}
    common = {"context_slots": 0, "gt_slots": 0, "padding_slots": 0, "batch_slots": 0,
              "physical_microbatches": 0, "point_tokens": 0}
    begin = time.monotonic()

    for model in models.values():
        model.train()
    for indices in batches:
        count = len(indices)
        if count < 1:
            raise ValueError("Empty logical update")
        update_loss = {name: 0. for name in names}
        seen = {name: 0 for name in names}
        for name in names:
            optimizers[name].zero_grad(set_to_none=True)
            lr = warmup_lr(peak_lr, step_offsets[name] + steps[name] + 1, warmup_steps)
            for group in optimizers[name].param_groups:
                group["lr"] = lr

        with closing(batches.microbatches(indices)) as microbatches:
            for batch in microbatches:
                for name, model in models.items():
                    set_rng_state(rng_states[name], device)
                    logits = model(batch)
                    losses = track_cross_entropy(logits, batch["targets"].to(device),
                        batch["gt_valid"].to(device), batch["padding_mask"].to(device))
                    (losses.sum() / count).backward()
                    rng_states[name] = rng_state(device)
                    update_loss[name] += float(losses.detach().double().sum())
                    seen[name] += len(losses)
                    del logits, losses

                common["context_slots"] += int(batch["lengths"].sum())
                common["gt_slots"] += int(batch["gt_valid"].sum())
                common["padding_slots"] += int(batch["padding_mask"].sum())
                common["batch_slots"] += batch["padding_mask"].numel()
                common["point_tokens"] += sum(len(x) for x in batch.get("points", ()))
                common["physical_microbatches"] += 1
                del batch

        if any(seen[name] != count for name in names):
            raise ValueError("Physical packing changed logical update membership")
        for name, model in models.items():
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
            optimizers[name].step()
            if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                raise ValueError("Nonfinite model parameters")
            total_loss[name] += update_loss[name]
            processed[name] += count
            steps[name] += 1
            if name in on_step:
                on_step[name](step_offsets[name] + steps[name], update_loss[name] / count)

    if not all(processed.values()):
        raise ValueError("Empty training split")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - begin
    padding = common["padding_slots"]
    capacity = common["batch_slots"]
    rows = {}
    for name in names:
        rows[name] = {"training_ce": total_loss[name] / processed[name],
            "optimizer_steps": steps[name], "processed_tracks": processed[name],
            **common, "padding_fraction": padding / capacity, "shared_group_seconds": elapsed,
            "learning_rate": optimizers[name].param_groups[0]["lr"]}
    return rows, rng_states
