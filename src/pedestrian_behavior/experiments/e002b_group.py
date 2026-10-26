"""Shared-input execution for E002b's three independent geometry + LiDAR models."""

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from importlib.metadata import version

import numpy as np
import torch
import wandb

from pedestrian_behavior.data.preparation import checksum, load_track, source_times
from pedestrian_behavior.evaluation import prediction_metrics, save_predictions, stratified_metrics, track_cross_entropy
from pedestrian_behavior.experiments import e002a_run, e002b, e002b_run
from pedestrian_behavior.experiments.run import (configure_logging, log_final, plots, write_json)
from pedestrian_behavior.shared_accumulation import rng_state, train_shared_accumulated_epoch
from pedestrian_behavior.training import reload_checkpoint, save_checkpoint, seed_run, select_epoch


VARIANTS = tuple(e002b.VARIANTS)
ARM = "g-dditr"
SETTINGS = e002a_run.SETTINGS
CLASSES = e002a_run.CLASSES
AXES = {key: value for key, value in e002a_run.STRATA.items() if key != "rgb-observed"}


def resource_code():
    from pedestrian_behavior import shared_accumulation
    return e002b_run.resource_code() | {
        shared_accumulation.__name__: checksum(shared_accumulation.__file__),
        __name__: checksum(__file__),
    }


def profile_for(args):
    record = json.loads(args.profile.read_text())
    if record.get("status") != "complete" or record.get("experiment") != "E002b" \
            or record.get("execution") != "shared-g-dditr":
        raise ValueError("Completed shared-LiDAR E002b calibration required")
    if record.get("device") != args.device or record.get("code") != resource_code():
        raise ValueError("Shared-LiDAR calibration identity differs from this run")
    if record.get("torch") != str(torch.__version__) or record.get("cuda") != torch.version.cuda:
        raise ValueError("PyTorch/CUDA backend differs from shared-LiDAR calibration")
    if record.get("point_storage_protocol") != e002a_run.POINT_PROTOCOL:
        raise ValueError("Indexed point-storage calibration required")
    if args.device == "cuda" and (not torch.cuda.is_available() or record.get("gpu") != torch.cuda.get_device_name()):
        raise ValueError("Shared-LiDAR calibration hardware differs from the requested CUDA device")
    expected = {f"{ARM}/{variant}": e002b.model_spec(ARM, variant) for variant in VARIANTS}
    if record.get("model_specs") != expected or set(record.get("model_coverage", ())) != set(expected):
        raise ValueError("Shared-LiDAR profile must cover all three exact model variants")
    e002a_run.validate_profile(record.get("profile", {}))
    return record


def assembly(args):
    profile_for(args)
    args.arm, args.model_variant = ARM, VARIANTS[0]
    policy = {"experiment": "E002b", "resource_code": resource_code,
              "validate_roots": e002b_run.validate_roots,
              "model_spec": e002b.model_spec,
              "model_factory": lambda arm, variant, profile: e002b.build_model(arm, variant, profile),
              "run_name": f"e002b-{args.source}-{ARM}-shared-s0"}
    return e002a_run.assembly(args, policy=policy)


def predict_shared(models, batches, device, *, forward_batch=None):
    """Predict all active models from each physical batch before advancing its producer."""
    device = torch.device(device)
    tracks = {name: [] for name in models}
    for model in models.values():
        model.eval()
    iterator = iter(batches)
    with closing(iterator), torch.no_grad():
        for batch in iterator:
            for name, model in models.items():
                logits = model(batch) if forward_batch is None else forward_batch(model, batch, device)
                track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device),
                                    batch["padding_mask"].to(device))
                metadata = batch.get("prediction_metadata", [{} for _ in batch["locators"]])
                for index, length in enumerate(batch["lengths"].tolist()):
                    locator = batch["locators"][index]
                    identity = json.loads(locator)
                    tracks[name].append(metadata[index] | {
                        "logits": logits[index, :length].detach().cpu().numpy().astype(np.float32),
                        "targets": batch["targets"][index, :length].numpy(),
                        "gt_valid": batch["gt_valid"][index, :length].numpy(),
                        "locator": locator, "archive": batch["archives"][index],
                        "group": identity.get("clip", identity.get("scenario")),
                    })
                del logits
            del batch
    if any(not rows for rows in tracks.values()):
        raise ValueError("Empty shared evaluation split")
    return tracks


def _write_step(output, run, optimizer, step, loss):
    row = {"optimizer_step": step, "training_ce": loss, "learning_rate": optimizer.param_groups[0]["lr"]}
    with (output / "steps.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
    run.log({"optimizer_step": step, "train/loss_step": loss, "train/learning_rate": row["learning_rate"]})


def _decorate(tracks, prepared, cache, source):
    for track in tracks:
        locator = track["locator"]
        if locator not in cache:
            metadata, arrays = load_track(track["archive"])
            identity = json.loads(locator)
            initial_keys = set(track)
            conditions, flags = prepared["conditions"](metadata, arrays, track, source)
            additions = {key: value for key, value in track.items() if key not in initial_keys}
            cache[locator] = additions | {"conditions": conditions, "source_flags": track.get("source_flags", {}) | flags,
                "source_times_us": source_times(metadata), "native_extent_us": metadata["native_extent_us"],
                "grid_times_us": (metadata["native_extent_us"][0] + np.arange(len(track["targets"])) * 200000).tolist()}
        track.update(cache[locator])


def run(args):
    if not 0 <= SETTINGS["seed"] < 2**32 or SETTINGS["epochs"] < 1 or SETTINGS["patience"] < 1:
        raise ValueError("Invalid frozen E002b training settings")
    if args.smoke and args.smoke_selection is None:
        raise ValueError("Shared group smoke requires an explicit source train/validation selection")
    output, device = Path(args.output), torch.device(args.device)
    if not output.resolve().is_relative_to(args.outputs_root.resolve() / "experiments/E002b"):
        raise ValueError("Shared group output must be under outputs-root/experiments/E002b")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    model_outputs = {variant: output / variant for variant in VARIANTS}
    for path in model_outputs.values():
        path.mkdir()
    status = {"status": "incomplete", "failures": [], "execution": "shared-g-dditr",
              "variants": list(VARIANTS), "held_out_tests": False}
    write_json(output / "status.json", status)
    for path in model_outputs.values():
        write_json(path / "status.json", {"status": "incomplete", "failures": [], "execution": "shared-g-dditr"})

    started = time.monotonic()
    provenance = {"command": sys.orig_argv, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "execution": "shared-g-dditr", "variants": list(VARIANTS)}
    runs, prepared, prepared_closed, completed = {}, None, False, set()
    statuses = {variant: {"status": "incomplete", "failures": [], "kind": "comparison"} for variant in VARIANTS}
    try:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        profile_record = profile_for(args)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prepared = assembly(args)
        status["kind"] = "integration-smoke" if args.smoke else "exploratory-source-validation"
        for attempt_status in statuses.values():
            attempt_status["kind"] = status["kind"]
        provenance.update(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD", "--", "src", "tests", "scripts"])).hexdigest(),
            code_sha256={str(path): checksum(path) for path in sorted(Path("src").rglob("*.py"))},
            versions={name: version(name) for name in ("numpy", "torch", "wandb", "matplotlib", "pillow", "pyarrow")},
            python=platform.python_version(), dependency_lock_sha256=checksum("uv.lock"),
            pyproject_sha256=checksum("pyproject.toml"), cuda_build=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), host=platform.node(),
            hardware={"platform": platform.platform(), "cpu": platform.processor() or "unknown",
                      "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            cpu_threads={"torch": torch.get_num_threads(), "environment":
                {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
            datasets=prepared["datasets"], source_statistics=prepared["normalization"],
            profile_sha256=checksum(args.profile), execution_profile=prepared["config"]["profile"])
        deterministic = None
        models, optimizers, rng_states = {}, {}, {}
        for variant in VARIANTS:
            deterministic = seed_run(SETTINGS["seed"])
            model = e002b.build_model(ARM, variant, prepared["config"]["profile"]).to(device)
            models[variant] = model
            rng_states[variant] = rng_state(device)
            optimizers[variant] = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                                     **SETTINGS["optimizer"])
            provenance.setdefault("parameters", {})[variant] = sum(parameter.numel() for parameter in model.parameters())
            provenance.setdefault("trainable_parameters", {})[variant] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        provenance["determinism"] = deterministic
        write_json(output / "provenance.json", provenance)

        configs = {}
        group_id = output.name
        for variant in VARIANTS:
            model_output = model_outputs[variant]
            config = prepared["config"] | {"experiment": "E002b", "source": args.source, "arm": ARM,
                "model_variant": variant, "model": e002b.model_spec(ARM, variant), "classes": CLASSES,
                "settings": SETTINGS, "determinism": deterministic, "device": str(device),
                "execution": "shared-g-dditr", "shared_execution_id": group_id, "held_out_tests": False,
                "smoke": args.smoke, "smoke_limits": prepared.get("smoke_limits")}
            configs[variant] = config
            write_json(model_output / "config.json", config)
            (model_output / "source-diff.patch").write_bytes(
                subprocess.check_output(["git", "diff", "HEAD", "--", "src", "tests", "scripts"]))
            write_json(model_output / "provenance.json", provenance | {"model_variant": variant,
                "parameters": provenance["parameters"][variant], "trainable_parameters": provenance["trainable_parameters"][variant]})

        for variant in VARIANTS:
            name = f"e002b-{args.source}-{ARM}-{variant}-s0" + ("-smoke" if args.smoke else "")
            wandb_run = wandb.init(entity="yammo-unipd", project="pedestrian-behaviour-labeling", name=name,
                job_type=status["kind"], config=configs[variant] | {"provenance": provenance},
                dir=str(model_outputs[variant].resolve()), save_code=False, resume="never", group=group_id,
                reinit="create_new", mode="online")
            runs[variant] = wandb_run
            provenance.setdefault("wandb", {})[variant] = {"id": wandb_run.id, "url": wandb_run.url}
            configure_logging(wandb_run)
        write_json(output / "provenance.json", provenance)

        dataset = prepared["dataset"](args.source, "training")
        batching = prepared["batches"]
        training_batches = batching(dataset, training=True, seed=SETTINGS["seed"])
        validation_dataset = prepared["dataset"](args.source, "validation")
        validation_batches = batching(validation_dataset, training=False, seed=SETTINGS["seed"])
        warmup_steps = prepared["config"]["warmup_updates"]
        step_offsets = {variant: 0 for variant in VARIANTS}
        histories = {variant: [] for variant in VARIANTS}
        bests, stales = {variant: None for variant in VARIANTS}, {variant: 0 for variant in VARIANTS}
        active = list(VARIANTS)
        axes = AXES
        epoch_budget = 1 if args.smoke else SETTINGS["epochs"]
        for epoch in range(1, epoch_budget + 1):
            if not active:
                break
            on_step = {variant: lambda step, loss, variant=variant: _write_step(
                model_outputs[variant], runs[variant], optimizers[variant], step, loss) for variant in active}
            rows, rng_states = train_shared_accumulated_epoch(
                {variant: models[variant] for variant in active}, training_batches,
                {variant: optimizers[variant] for variant in active}, device, SETTINGS["clip_norm"],
                rng_states={variant: rng_states[variant] for variant in active}, peak_lr=SETTINGS["optimizer"]["lr"],
                warmup_steps=warmup_steps, step_offsets={variant: step_offsets[variant] for variant in active},
                on_step=on_step)
            validation_tracks = predict_shared({variant: models[variant] for variant in active}, validation_batches,
                                               device, forward_batch=prepared.get("forward_batch"))
            next_active = []
            for variant in active:
                step_offsets[variant] += rows[variant]["optimizer_steps"]
                metrics = prediction_metrics(validation_tracks[variant], len(CLASSES))
                row = {"epoch": epoch} | rows[variant] | {"validation_ce": metrics["track_ce"],
                    "validation_f1": metrics["track_weighted"]["macro_f1"]}
                bests[variant], stales[variant], selected = select_epoch(
                    bests[variant], row["validation_f1"], row["validation_ce"], epoch, stales[variant])
                reason = "patience" if stales[variant] >= SETTINGS["patience"] else "budget" if epoch == epoch_budget else None
                row.update(selected=selected, non_improving_epochs=stales[variant], stopping_reason=reason)
                histories[variant].append(row)
                evidence = {"config": configs[variant], "normalization": prepared["normalization"],
                    "provenance_file": "provenance.json", "epoch": epoch, "selection": bests[variant],
                    "validation": metrics, "stopping_reason": reason}
                if selected:
                    save_checkpoint(model_outputs[variant] / "best.pt", models[variant], evidence)
                save_checkpoint(model_outputs[variant] / "last.pt", models[variant], evidence)
                write_json(model_outputs[variant] / "history.json", histories[variant])
                if "runtime_snapshot" in prepared:
                    provenance.setdefault("runtime_by_epoch", {}).setdefault(variant, []).append(prepared["runtime_snapshot"]())
                write_json(model_outputs[variant] / "provenance.json", provenance | {"model_variant": variant,
                    "selection": bests[variant], "last_completed_epoch": epoch,
                    "shared_group_seconds": row["shared_group_seconds"]})
                runs[variant].log({"epoch": epoch, "train/loss_epoch": row["training_ce"],
                    "train/macro_f1": row["training_f1"],
                    "val/loss": row["validation_ce"], "val/macro_f1": row["validation_f1"]})
                statuses[variant]["last_completed_epoch"] = epoch
                if reason:
                    optimizers[variant].zero_grad(set_to_none=True)
                    statuses[variant]["status"] = "training-stopped"
                    statuses[variant]["stopping_reason"] = reason
                    write_json(model_outputs[variant] / "status.json", statuses[variant])
                else:
                    next_active.append(variant)
            active = next_active

        final_batches = batching(validation_dataset, training=False, seed=SETTINGS["seed"])
        evaluation_split = "validation-smoke" if args.smoke else "validation"
        for variant in VARIANTS:
            if bests[variant] is None:
                raise ValueError(f"No selected source-validation checkpoint for {variant}")
            selected = reload_checkpoint(model_outputs[variant] / "best.pt", models[variant], device)
            if selected["epoch"] != bests[variant]["epoch"]:
                raise ValueError("Selected checkpoint epoch disagrees with in-memory selection")
            statuses[variant]["selected_checkpoint_epoch"] = selected["epoch"]

        final_tracks = predict_shared(models, final_batches, device, forward_batch=prepared.get("forward_batch"))
        metadata_cache = {}
        evaluations, plot_paths = {}, {}
        for variant in VARIANTS:
            tracks = final_tracks[variant]
            _decorate(tracks, prepared, metadata_cache, args.source)
            destination = model_outputs[variant] / f"{args.source}-{evaluation_split}"
            destination.mkdir()
            save_predictions(destination / "predictions.npz", tracks)
            result = {"split": evaluation_split, "primary": prediction_metrics(tracks, len(CLASSES)),
                      "strata": stratified_metrics(tracks, len(CLASSES), axes)}
            write_json(destination / "metrics.json", result)
            evaluations[variant] = {args.source: result}
            plot_paths[variant] = plots(model_outputs[variant] / "plots", histories[variant], bests[variant],
                                        evaluations[variant], CLASSES)

        if "runtime_snapshot" in prepared:
            provenance["runtime"] = prepared["runtime_snapshot"]()
        prepared["close"]()
        prepared_closed = True
        for variant in VARIANTS:
            model_provenance = provenance | {"model_variant": variant, "selection": bests[variant],
                "parameters": provenance["parameters"][variant],
                "trainable_parameters": provenance["trainable_parameters"][variant],
                "stopping_reason": statuses[variant].get("stopping_reason", "budget"),
                "completed_epochs": len(histories[variant]), "selected_checkpoint_epoch": bests[variant]["epoch"],
                "exposure": {key: sum(row[key] for row in histories[variant]) for key in
                    ("optimizer_steps", "processed_tracks", "context_slots", "gt_slots", "padding_slots", "batch_slots")},
                "elapsed_seconds": time.monotonic() - started,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "shared_group_seconds_per_epoch": [row["shared_group_seconds"] for row in histories[variant]]}
            model_provenance["exposure"]["padding_fraction"] = (
                model_provenance["exposure"]["padding_slots"] / model_provenance["exposure"]["batch_slots"])
            model_provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(model_outputs[variant] / "provenance.json", model_provenance)
            log_final(runs[variant], evaluations[variant], plot_paths[variant], model_provenance, CLASSES, axes)
            runs[variant].finish()
        for variant in VARIANTS:
            completed.add(variant)
            statuses[variant]["status"] = "complete"
            statuses[variant]["stopping_reason"] = statuses[variant].get("stopping_reason", "budget")
            write_json(model_outputs[variant] / "status.json", statuses[variant])
        status["status"] = "complete"
        status["completed_utc"] = datetime.now(timezone.utc).isoformat()
        return {variant: {"history": histories[variant], "selection": bests[variant],
                          "stopping_reason": statuses[variant].get("stopping_reason")} for variant in VARIANTS}
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        status["status"] = "failed"
        status["failures"].append(failure)
        for variant in VARIANTS:
            if variant in completed:
                continue
            statuses[variant]["status"] = "failed"
            statuses[variant]["failures"].append(failure)
            if variant in runs:
                try:
                    runs[variant].finish(exit_code=1)
                except Exception as finish_error:
                    statuses[variant]["failures"].append({"type": type(finish_error).__name__,
                        "message": str(finish_error)})
            write_json(model_outputs[variant] / "status.json", statuses[variant])
        raise
    finally:
        cleanup_error = None
        try:
            if prepared is not None and "runtime_snapshot" in prepared and not prepared_closed:
                provenance["runtime"] = prepared["runtime_snapshot"]()
            if prepared is not None and "close" in prepared and not prepared_closed:
                prepared["close"]()
                prepared_closed = True
        except BaseException as error:
            cleanup_error = error
            status["status"] = "failed"
            status["failures"].append({"type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc()})
            for variant in VARIANTS:
                if statuses[variant]["status"] != "failed":
                    statuses[variant]["status"] = "failed"
                    statuses[variant]["failures"].append({"type": type(error).__name__, "message": str(error)})
                    write_json(model_outputs[variant] / "status.json", statuses[variant])
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=time.monotonic() - started)
        write_json(output / "provenance.json", provenance)
        write_json(output / "status.json", status)
        if cleanup_error is not None:
            raise cleanup_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("road-waymo", "loki"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--profile", type=Path, required=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments/E002b/runs") / f"shared-lidar-{timestamp}")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--indexed-point-cache", type=Path,
                        help="Completed source cache; default outputs-root/experiments/E002a/indexed-<source>")
    parser.add_argument("--point-workers", type=int, default=e002a_run.DEFAULT_POINT_WORKERS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path,
                        help="Explicit bounded source train/validation selection; required for --smoke")
    run(parser.parse_args())
