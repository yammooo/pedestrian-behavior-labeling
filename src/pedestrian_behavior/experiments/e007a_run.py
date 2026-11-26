"""E007a joint ROAD--LOKI training with cross-source contrastive pairing."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from functools import partial
from itertools import islice
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from pedestrian_behavior.accumulation import warmup_lr
from pedestrian_behavior.evaluation import (predict, prediction_metrics, track_cross_entropy,
                                            track_weighted_confusion_batch,
                                            training_track_weighted_macro_f1)
from pedestrian_behavior.training import save_checkpoint

from . import e005_run
from .e001 import CLASSES
from .e002a_run import SETTINGS as BASE_SETTINGS
from .run import run_attempt


SOURCE = "joint"
ARM = "g-rgb-utonia"
TEMPERATURE = 0.1
LAMBDA = 0.1
EPOCHS = 40
PATIENCE = 10
SETTINGS = {**BASE_SETTINGS, "epochs": EPOCHS, "patience": PATIENCE,
            "contrastive": {"lambda": LAMBDA, "temperature": TEMPERATURE,
                            "anchors_per_track": 2, "context_radius": 2,
                            "negative_per_other_class": 1,
                            "bank": "previous-effective-update-only"}}


def _anchors(batch, source_by_locator, rng):
    """Sample up to two distinct steady GT classes per complete track."""
    targets, gt_valid, lengths = batch["targets"], batch["gt_valid"], batch["lengths"]
    result = []
    for b, locator in enumerate(batch["locators"]):
        source = source_by_locator[locator]
        length = int(lengths[b])
        eligible = defaultdict(list)
        for t in range(2, length - 2):
            window = gt_valid[b, t - 2:t + 3]
            if not bool(window.all()):
                continue
            labels = targets[b, t - 2:t + 3].tolist()
            if len(set(labels)) == 1:
                eligible[int(labels[2])].append(t)
        labels = list(eligible)
        for label in rng.sample(labels, min(2, len(labels))):
            result.append((b, rng.choice(eligible[label]), source, label))
    return result


def _contrastive(features, anchors, bank, rng, temperature):
    """Build track-summed InfoNCE against the detached previous-update bank."""
    next_bank, by_track = [], defaultdict(list)
    opportunities, pairs = Counter(), Counter()
    for b, t, source, label in anchors:
        value = F.normalize(features[b, t], dim=0)
        next_bank.append((source, label, value.detach()))
        opportunities[(source, label)] += 1
        candidates = [(other_label, other_value) for other_source, other_label, other_value in bank or ()
                      if other_source != source]
        positives = [value for other_label, value in candidates if other_label == label]
        negative_labels = sorted({other_label for other_label, _ in candidates if other_label != label})
        if not positives or not negative_labels:
            continue
        positive = rng.choice(positives)
        negatives = [rng.choice([value for other_label, value in candidates if other_label == negative_label])
                     for negative_label in negative_labels]
        scores = torch.stack([positive, *negatives]) @ value / temperature
        by_track[b].append(F.cross_entropy(scores[None], torch.zeros(1, dtype=torch.long, device=scores.device)))
        pairs[(source, label)] += 1
    total = features.sum() * 0.
    for losses in by_track.values():
        total = total + torch.stack(losses).mean()
    return total, next_bank, opportunities, pairs, len(by_track)


def train_contrastive_epoch(model, batches, optimizer, device, clip_norm, on_step=None, step_offset=0,
                            *, peak_lr, warmup_steps, source_by_locator, rng,
                            bank_state, temperature=TEMPERATURE, contrastive_lambda=LAMBDA):
    """E005 accumulation plus detached previous-update cross-source InfoNCE."""
    model.train()
    started = time.monotonic()
    total_loss, tracks, steps, slots, gt, padding, capacity, physical, tokens = 0., 0, 0, 0, 0, 0, 0, 0, 0
    training_confusion = None
    bank = bank_state["bank"]
    total_contrastive, opportunities, pairs, pair_tracks = 0., Counter(), Counter(), 0
    for indices in batches:
        count, update_loss, update_contrastive, seen = len(indices), 0., 0., 0
        update_opportunities, update_pairs, update_pair_tracks = Counter(), Counter(), 0
        next_bank = []
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = warmup_lr(peak_lr, step_offset + steps + 1, warmup_steps)
        with closing(batches.microbatches(indices)) as microbatches:
            for batch in microbatches:
                captured = []
                hook = model.final_norm.register_forward_hook(lambda _m, _i, output: captured.append(output))
                try:
                    logits = model(batch)
                finally:
                    hook.remove()
                if len(captured) != 1:
                    raise RuntimeError("E007a expected one final temporal representation")
                features = captured[0]
                if not torch.isfinite(features).all():
                    raise ValueError("Nonfinite E007a temporal representation")
                targets = batch["targets"].to(device)
                gt_valid = batch["gt_valid"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                losses = track_cross_entropy(logits, targets, gt_valid, padding_mask)
                batch_confusion = track_weighted_confusion_batch(logits, targets, gt_valid, padding_mask)
                training_confusion = batch_confusion if training_confusion is None else training_confusion + batch_confusion
                anchors = _anchors(batch, source_by_locator, rng)
                contrastive, generated_bank, anchor_opportunities, anchor_pairs, used_tracks = _contrastive(
                    features, anchors, bank, rng, temperature)
                next_bank.extend(generated_bank)
                update_opportunities.update(anchor_opportunities)
                update_pairs.update(anchor_pairs)
                update_pair_tracks += used_tracks
                (losses.sum() / count + contrastive_lambda * contrastive / count).backward()
                update_loss += float(losses.detach().double().sum())
                update_contrastive += float(contrastive.detach().double())
                seen += len(losses)
                slots += int(batch["lengths"].sum())
                gt += int(batch["gt_valid"].sum())
                padding += int(batch["padding_mask"].sum())
                capacity += batch["padding_mask"].numel()
                tokens += sum(len(x) for x in batch.get("points", ()))
                physical += 1
                del logits, features, losses, batch
        if seen != count or count < 1:
            raise ValueError("Physical packing changed logical update membership")
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite model parameters")
        bank = next_bank
        bank_state["bank"] = bank
        total_loss += update_loss
        total_contrastive += update_contrastive
        opportunities.update(update_opportunities)
        pairs.update(update_pairs)
        pair_tracks += update_pair_tracks
        tracks += count
        steps += 1
        if on_step:
            on_step(step_offset + steps, update_loss / count)
    if not tracks:
        raise ValueError("Empty training split")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    as_source = lambda values: {source: {str(label): int(values[(source, label)]) for label in range(len(CLASSES))}
                                for source in ("road-waymo", "loki")}
    return {"training_ce": total_loss / tracks,
            "training_f1": training_track_weighted_macro_f1(training_confusion),
            "optimizer_steps": steps, "processed_tracks": tracks,
            "context_slots": slots, "gt_slots": gt, "padding_slots": padding, "batch_slots": capacity,
            "padding_fraction": padding / capacity, "physical_microbatches": physical,
            "point_tokens": tokens, "seconds": time.monotonic() - started,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "contrastive": {"loss": total_contrastive / tracks, "opportunities": sum(opportunities.values()),
                            "pairs": sum(pairs.values()), "pair_tracks": pair_tracks,
                            "opportunities_by_source": as_source(opportunities),
                            "pairs_by_source": as_source(pairs)}}


def _checkpoint_callback():
    best_ce = [None]

    def save_extra(output, model, evidence):
        validation = evidence["validation"]
        epoch = evidence["epoch"]
        if evidence["selection"]["epoch"] == epoch:
            save_checkpoint(Path(output) / "best-f1.pt", model, evidence | {"checkpoint": "best-f1"})
        ce = validation["track_ce"]
        if best_ce[0] is None or ce < best_ce[0]["ce"]:
            best_ce[0] = {"epoch": epoch, "ce": ce}
            save_checkpoint(Path(output) / "best-ce.pt", model,
                            evidence | {"checkpoint": "best-ce", "selection": best_ce[0]})

    return save_extra


def assembly(args, *, smoke=False):
    if args.source != SOURCE or args.arm != ARM:
        raise ValueError("E007a is the joint g-rgb-utonia experiment")
    prepared = e005_run.assembly(args)
    training = prepared["dataset"]("joint", "training")
    source_by_locator = {entry["locator"]: entry["source"] for entry in training.entries}
    rng = random.Random(SETTINGS["seed"])
    bank_state = {"bank": None}
    prepared["config"] = prepared["config"] | {
        "experiment": "E007a", "objective": "joint CE + cross-source InfoNCE",
        "contrastive": SETTINGS["contrastive"], "epochs": EPOCHS, "patience": PATIENCE,
        "checkpoints": {"best-f1": "highest mean source validation macro-F1",
                         "best-ce": "lowest mean source validation track CE"}}
    prepared["epoch_operation"] = partial(
        train_contrastive_epoch, peak_lr=BASE_SETTINGS["optimizer"]["lr"],
        warmup_steps=BASE_SETTINGS["warmup_epochs"] * math.ceil(len(training) / (2 * 32)),
        source_by_locator=source_by_locator, rng=rng, bank_state=bank_state)
    prepared["checkpoint_callback"] = _checkpoint_callback()
    if smoke:
        prepared["supports_smoke"] = True
        prepared["kind"] = "gpu-smoke"
        prepared["smoke_limits"] = {"epochs": 1, "training_batches": 2,
                                    "validation_microbatches_per_source": 1, "held_out_tests": False}
        prepared["wandb_mode"] = "offline"

        def smoke_validate(model):
            values = []
            for source in ("road-waymo", "loki"):
                validation = prepared["batches"](prepared["dataset"](source, "validation"), seed=0)
                values.extend(predict(model, islice(validation, 1), torch.device(args.device), prepared["forward_batch"]))
            return prediction_metrics(values, len(CLASSES))

        prepared["validation_metrics"] = smoke_validate
        prepared["evaluation_splits"] = []
    return prepared


def run(args):
    if args.source != SOURCE or args.arm != ARM:
        raise ValueError("E007a requires --source joint --arm g-rgb-utonia")
    root = args.outputs_root.resolve()
    runs = (root / "experiments/E007a/runs").resolve()
    output = args.output.resolve()
    if output == runs or not output.is_relative_to(runs):
        raise ValueError("Use a fresh output directory under outputs/experiments/E007a/runs")
    if args.aalto_fast is None:
        args.aalto_fast = (args.device == "cuda" and torch.cuda.is_available()
                           and torch.cuda.get_device_name() == e005_run.AALTO_GPU)
    if args.aalto_fast:
        args.allocator_cap_bytes = e005_run.enable_aalto_fast(args.arm, args.device)
    torch.set_num_threads(2)
    return run_attempt(SOURCE, args.arm, "bilstm-96", output, args.device, smoke=args.smoke,
                       settings=SETTINGS, assemble=lambda *_: assembly(args, smoke=args.smoke),
                       classes=CLASSES, axes=e005_run.axes(args.arm))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=(SOURCE,), default=SOURCE)
    parser.add_argument("--arm", choices=(ARM,), default=ARM)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--loki-cache", type=Path, required=True)
    parser.add_argument("--road-cache", type=Path, required=True)
    parser.add_argument("--joint-profile", type=Path, required=True)
    parser.add_argument("--aalto-fast", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--smoke", action="store_true", help="Two-update offline W&B smoke check")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
