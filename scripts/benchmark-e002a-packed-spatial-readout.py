"""Compare padded and segmented LiDAR spatial attention on E002a training inputs."""

import argparse
import gc
import json
import resource
import statistics
import subprocess
import time
import types

import torch

from pedestrian_behavior.accumulation import warmup_lr
from pedestrian_behavior.evaluation import track_cross_entropy
from pedestrian_behavior.experiments.e002a_run import SETTINGS, add_arguments, assembly
from pedestrian_behavior.experiments.e002a_throughput import TimedBatches, cases
from pedestrian_behavior.models.multimodal import SpatialReadout, frame_chunks
from pedestrian_behavior.training import seed_run
from pedestrian_behavior.experiments.run import write_json


def packed_forward(self, cpu_tokens, lengths, seed):
    """SpatialReadout with per-frame softmax over concatenated, unpadded tokens."""
    x = cpu_tokens.to(seed.device)
    h = self.project(torch.cat((self.feature_norm(x[:, :64]), x[:, 64:]), -1))
    h = self.point_norm(h)
    frame = torch.repeat_interleave(torch.arange(len(lengths), device=h.device),
                                    torch.as_tensor(lengths, device=h.device))
    heads, head_width = 4, seed.shape[-1] // 4
    q = self.q(self.seed_norm(seed)).reshape(1, heads, head_width)
    k, v = (linear(h).reshape(-1, heads, head_width) for linear in (self.k, self.v))
    scores = (q * k).sum(-1) * head_width ** -0.5
    frame_heads = frame[:, None].expand_as(scores)
    maxima = torch.full((len(lengths), heads), -torch.inf, device=h.device, dtype=h.dtype)
    maxima = maxima.scatter_reduce(0, frame_heads, scores, reduce="amax", include_self=True)
    weights = torch.exp(scores - maxima.index_select(0, frame))
    totals = torch.zeros((len(lengths), heads), device=h.device, dtype=h.dtype)
    totals = totals.index_add(0, frame, weights)
    attended = torch.zeros((len(lengths), heads, head_width), device=h.device, dtype=h.dtype)
    attended = attended.index_add(0, frame, (weights / totals.index_select(0, frame))[..., None] * v)
    y = attended.reshape(len(lengths), seed.shape[-1])
    u = seed + self.dropout(self.out(y))
    return u + self.dropout(self.ffn(self.norm_ffn(u)))


def _set_variant(model, variant):
    if variant == "packed":
        model.spatial.forward = types.MethodType(packed_forward, model.spatial)
    elif variant != "padded":
        raise ValueError(f"Unknown spatial variant: {variant}")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_spatial(device, token_values, lengths, state, repeats):
    rows = []
    for variant in ("padded", "packed"):
        module = SpatialReadout().to(device)
        module.load_state_dict(state)
        _set_variant(types.SimpleNamespace(spatial=module), variant)
        module.eval()

        def step():
            module.zero_grad(set_to_none=True)
            result = module(token_values, lengths, module.seed)
            result.square().mean().backward()
            _sync(device)

        step()  # warm kernels before timing
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
        elapsed = []
        for _ in range(repeats):
            begin = time.perf_counter()
            step()
            elapsed.append(time.perf_counter() - begin)
        rows.append({"variant": variant, "seconds": elapsed,
                     "median_seconds": statistics.median(elapsed),
                     "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                     "peak_cuda_delta_bytes": torch.cuda.max_memory_allocated(device) - before if device.type == "cuda" else None,
                     "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        del module
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def _new_model(prepared, state, device, variant):
    model = prepared["build_model"]().to(device)
    model.load_state_dict(state)
    _set_variant(model, variant)
    model.eval()  # dropout disabled for matched gradients and update timing
    return model


def _update(model, batches, device, *, warmup_updates):
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    optimizer.param_groups[0]["lr"] = warmup_lr(.0003, 1, warmup_updates)
    optimizer.zero_grad(set_to_none=True)
    count = sum(len(batch["locators"]) for batch in batches)
    total_loss = 0.
    for batch in batches:
        logits = model(batch)
        losses = track_cross_entropy(logits, batch["targets"].to(device),
                                     batch["gt_valid"].to(device), batch["padding_mask"].to(device))
        (losses.sum() / count).backward()
        total_loss += float(losses.detach().double().sum())
        del logits, losses
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Nonfinite parameters after matched optimizer update")
    _sync(device)
    return optimizer, total_loss / count


def _model_update(prepared, state, batches, device, variant):
    model = _new_model(prepared, state, device, variant)
    warm_optimizer, _ = _update(model, batches, device, warmup_updates=prepared["config"]["warmup_updates"])
    del warm_optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = _new_model(prepared, state, device, variant)
    optimizer = torch.optim.AdamW(model.parameters(), **SETTINGS["optimizer"])
    optimizer.param_groups[0]["lr"] = warmup_lr(.0003, 1, prepared["config"]["warmup_updates"])
    optimizer.zero_grad(set_to_none=True)
    track_count = sum(len(batch["locators"]) for batch in batches)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
    begin = time.perf_counter()
    total_loss = 0.
    for batch in batches:
        logits = model(batch)
        losses = track_cross_entropy(logits, batch["targets"].to(device),
                                     batch["gt_valid"].to(device), batch["padding_mask"].to(device))
        (losses.sum() / track_count).backward()
        total_loss += float(losses.detach().double().sum())
        del logits, losses
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    _sync(device)
    row = {"variant": variant, "seconds": time.perf_counter() - begin,
           "tracks": track_count, "physical_microbatches": len(batches),
           "training_ce": total_loss / track_count,
           "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
           "peak_cuda_delta_bytes": torch.cuda.max_memory_allocated(device) - before if device.type == "cuda" else None,
           "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
           "lr": optimizer.param_groups[0]["lr"]}
    row["updated_state"] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def _load_update(prepared, dataset, indices, output):
    logical = prepared["batches"](dataset, training=True, seed=0)
    timed = TimedBatches(logical, indices)
    batches = []
    begin = time.perf_counter()
    for group in timed:
        batches.extend(timed.microbatches(group))
    elapsed = time.perf_counter() - begin
    expected = len(indices)
    if sum(len(batch["locators"]) for batch in batches) != expected:
        raise ValueError("Prepared update changed the selected 64-track membership")
    if len(indices) != 64:
        raise ValueError("A complete 64-track source training update is required")
    stats = {"tracks": expected, "physical_microbatches": len(batches),
             "point_tokens": sum(len(frame) for batch in batches for frame in batch["points"]),
             "input_pipeline_wall_seconds": elapsed,
             "input_wait_seconds": timed.loading_seconds,
             "loader_prepared_seconds": logical.stats["prepared_seconds"],
             "loader_wait_seconds": logical.stats["wait_seconds"],
             "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    return batches, stats


def _select_spatial_example(batches, profile):
    examples = []
    for batch_index, batch in enumerate(batches):
        for chunk in frame_chunks(batch["points"], profile["frame_chunk"], profile["frame_tokens"]):
            if len(chunk) == 4:
                lengths = tuple(len(frame) for _, frame in chunk)
                tokens = torch.cat([frame for _, frame in chunk])
                padded = max(lengths) * len(lengths)
                examples.append((padded / sum(lengths), batch_index, lengths, tokens, padded))
    if not examples:
        raise ValueError("Selected training update had no complete four-frame spatial chunk")
    _, batch_index, lengths, tokens, padded = max(examples, key=lambda row: (row[0], row[4]))
    return {"tokens": tokens, "lengths": lengths,
            "shape": {"frames": 4, "tokens": sum(lengths), "lengths": list(lengths),
                      "padded_tokens": padded, "padding_ratio": padded / sum(lengths),
                      "physical_microbatch": batch_index}}


def run(args):
    if args.device != "cuda" or args.arm != "g-dditr" or args.smoke:
        raise ValueError("Benchmark requires the full-source g-dditr CUDA path; no smoke or held-out data")
    if not 1 <= args.repeats <= 10:
        raise ValueError("Use one to ten bounded spatial-only repeats")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    record = {"status": "incomplete", "source": args.source, "arm": args.arm,
              "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "device": torch.cuda.get_device_name(), "seed": 0, "repeats": args.repeats,
              "scope": "one shuffled64 source-training optimizer update per variant; same prepared CPU microbatches/weights; no W&B or evaluation",
              "dropout": "disabled with eval mode for parity and timing",
              "input_timing_note": "The input pipeline prepares one batch ahead; consumer wait is part of the wall time, while loader preparation and wait counters can overlap.",
              "attention_only_note": "The current readout projects real point tokens before padding; packed attention changes padding after projection, not point projection work.",
              "variants": [], "failures": []}
    prepared = None
    try:
        seed_run(0)
        prepared = assembly(args)
        dataset = prepared["dataset"](args.source, "training")
        indices = cases(dataset)["shuffled"]
        if len(indices) != 64:
            raise ValueError("Expected a complete shuffled64 training update")
        record["selected_training_locators"] = [dataset.entries[i]["locator"] for i in indices]
        batches, load_stats = _load_update(prepared, dataset, indices, args.output)
        record["input_preparation"] = load_stats
        example = _select_spatial_example(batches, prepared["config"]["profile"])
        model = prepared["build_model"]()
        model_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        spatial_state = {name.removeprefix("spatial."): value for name, value in model_state.items()
                         if name.startswith("spatial.")}
        del model

        record["spatial_example"] = example["shape"]
        record["spatial_only_forward_backward"] = _measure_spatial(
            torch.device("cuda"), example["tokens"], example["lengths"], spatial_state, args.repeats)
        del example

        updated = []
        for variant in ("padded", "packed"):
            row = _model_update(prepared, model_state, batches, torch.device("cuda"), variant)
            updated.append(row)
            write_json(args.output / "packed-spatial.json", record | {"optimizer_updates": [
                {k: v for k, v in item.items() if k != "updated_state"} for item in updated]})
        left, right = updated
        max_delta = max((left["updated_state"][name] - right["updated_state"][name]).abs().max().item()
                        for name in left["updated_state"])
        record["optimizer_updates"] = [{k: v for k, v in item.items() if k != "updated_state"} for item in updated]
        record["max_parameter_delta_after_update"] = max_delta
        record["status"] = "complete"
        write_json(args.output / "packed-spatial.json", record)
        return record
    except BaseException as error:
        record["status"] = "failed"
        record["failures"].append(f"{type(error).__name__}: {error}")
        write_json(args.output / "packed-spatial.json", record)
        raise
    finally:
        if prepared is not None:
            try:
                prepared["close"]()
            except BaseException as error:
                record["status"] = "failed"
                record["failures"].append(f"cleanup {type(error).__name__}: {error}")
                write_json(args.output / "packed-spatial.json", record)
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
