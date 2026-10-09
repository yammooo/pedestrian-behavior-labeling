"""Shared attempt execution, local evidence, plots and W&B; policies are supplied."""

from datetime import datetime, timezone
import hashlib
from itertools import islice
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

from pedestrian_behavior.data.loading import track_batches
from pedestrian_behavior.data.preparation import checksum, load_track, source_times
from pedestrian_behavior.evaluation import (predict, prediction_metrics, save_predictions, stratified_metrics)
from pedestrian_behavior.training import fit, reload_checkpoint, seed_run


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def dataset_references(collections, setups, records):
    return {d: {"setup_path": str(setups[d]), "setup_sha256": checksum(setups[d]),
                "collection_path": str(collections[d]), "collection_sha256": checksum(collections[d]/"manifest.json"),
                "policy_sha256": hashlib.sha256(json.dumps(records[d]["policy"], sort_keys=True).encode()).hexdigest(),
                "native_policy": records[d]["native_audit"]["policy"], "population": records[d]["population"],
                "support": records[d]["support"]} for d in setups}


def plots(output, history, selection, evaluations, classes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(exist_ok=True)
    paths = []
    def save(name, figure):
        figure.tight_layout()
        for suffix in ("png", "svg"):
            figure.savefig(output/(name+"."+suffix), dpi=150)
        paths.append(output/(name+".png"))
        plt.close(figure)
    epochs = [r["epoch"] for r in history]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [r["training_ce"] for r in history], label="training (changing weights)")
    ax.plot(epochs, [r["validation_ce"] for r in history], label="source validation")
    ax.set(xlabel="Epoch", ylabel="Track CE")
    ax.legend()
    save("loss", fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [r["validation_f1"] for r in history], marker="o")
    ax.axvline(selection["epoch"], color="black", linestyle="--", label="selected")
    ax.set(xlabel="Epoch", ylabel="Four-class track-weighted macro-F1", ylim=(0, 1))
    ax.legend()
    save("validation-f1", fig)
    for dataset, result in evaluations.items():
        for weighting in ("raw", "track_weighted"):
            counts = np.array(result["primary"][weighting]["confusion"])
            for view in ("count", "row-normalized"):
                support = counts.sum(1, keepdims=True)
                values = counts if view == "count" else np.divide(counts, support, out=np.zeros_like(counts), where=support != 0)
                fig, ax = plt.subplots(figsize=(8, 6))
                image = ax.imshow(values, cmap="Blues")
                fig.colorbar(image, ax=ax)
                for (i, j), v in np.ndenumerate(values):
                    ax.text(j, i, f"{v:.3g}", ha="center", va="center")
                ax.set(xticks=range(4), yticks=range(4), xticklabels=classes, yticklabels=classes,
                       xlabel="Prediction", ylabel="GT", title=f"{dataset}: {weighting}, {view}")
                plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
                save(dataset+"-"+weighting+"-"+view, fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, measure in enumerate(("precision", "recall", "f1")):
            ax.bar(np.arange(4)+(i-1)*.25, result["primary"]["track_weighted"][measure], width=.25, label=measure)
        ax.set(xticks=range(4), xticklabels=[f"{c}\nGT={n}" for c, n in zip(classes, result["primary"]["denominators"]["class_frames"])], ylim=(0, 1), title=dataset)
        ax.legend()
        save(dataset+"-per-class", fig)
        for axis, slices in result.get("strata", {}).items():
            fig, ax = plt.subplots(figsize=(max(8, len(slices)*2), 5))
            for i, (label, score) in enumerate(slices.items()):
                value = score["track_weighted"]["macro_f1"]
                d = score["denominators"]
                supported = score["track_weighted"]["supported_classes"]
                if value is not None:
                    ax.bar(i, value)
                ax.text(i, (value or 0)+.025, f'G/T/F={d["groups"]}/{d["tracks"]}/{d["gt_frames"]}\nclasses={supported}\nGT={d["class_frames"]}', ha="center", fontsize=8)
            ax.set(xticks=range(len(slices)), xticklabels=list(slices), ylim=(0, 1.25),
                   ylabel="Supported-class track-weighted macro-F1", title=dataset+": "+axis)
            save(dataset+"-"+axis, fig)
    return paths


def configure_logging(run):
    run.define_metric("optimizer_step", hidden=True, summary="none")
    run.define_metric("epoch", hidden=True, summary="none")
    run.define_metric("train/loss_step", step_metric="optimizer_step", step_sync=False, summary="none")
    run.define_metric("train/loss_epoch", step_metric="epoch", step_sync=False)
    run.define_metric("val/*", step_metric="epoch", step_sync=False)
    run.define_metric("details/*", hidden=True)


def log_final(run, evaluations, plot_paths, provenance, classes, axes):
    run.config.update({"provenance": provenance,
                       "evaluation_support": {d: r["primary"]["denominators"] for d, r in evaluations.items()}},
                      allow_val_change=True)
    payload = {}
    for dataset, result in evaluations.items():
        for weighting in ("raw", "track_weighted"):
            scores = result["primary"][weighting]
            for metric in ("macro_f1", "support_weighted_f1", "accuracy"):
                payload[f"eval/{dataset}/{weighting}/{metric}"] = scores[metric]
            payload[f"details/{dataset}/{weighting}/per-class"] = wandb.Table(
                columns=["class", "precision", "recall", "f1", "support"],
                data=[[c]+[scores[k][i] for k in ("precision", "recall", "f1", "support")] for i, c in enumerate(classes)])
            payload[f"details/{dataset}/{weighting}/confusion"] = wandb.Table(columns=["gt"]+list(classes),
                data=[[c]+scores["confusion"][i] for i, c in enumerate(classes)])
        for metric in ("track_ce", "mean_track_accuracy", "median_track_accuracy"):
            payload[f"eval/{dataset}/{metric}"] = result["primary"][metric]
        for axis, slices in result.get("strata", {}).items():
            payload[f"details/{dataset}/strata/{axis}"] = wandb.Table(
                columns=["bin", "macro_f1", "groups", "tracks", "frames", "classes", "class_frames"],
                data=[[label, s["track_weighted"]["macro_f1"], s["denominators"]["groups"], s["denominators"]["tracks"],
                       s["denominators"]["gt_frames"], s["track_weighted"]["supported_classes"], s["denominators"]["class_frames"]]
                      for label, s in slices.items()])
            for weighting in ("raw", "track_weighted"):
                payload[f"details/{dataset}/strata/{axis}/{weighting}/per-class"] = wandb.Table(
                    columns=["bin", "class", "precision", "recall", "f1", "support"],
                    data=[[label, c]+[s[weighting][k][i] for k in ("precision", "recall", "f1", "support")]
                          for label, s in slices.items() for i, c in enumerate(classes)])
    for p in plot_paths:
        key = "details/plots/"+p.stem
        for dataset in evaluations:
            axis = p.stem.removeprefix(dataset+"-")
            if p.stem.startswith(dataset+"-") and axis in axes:
                key = f"strata/{dataset}/{axis}"
                break
        payload[key] = wandb.Image(str(p))
    run.log(payload)


def run_attempt(source, configuration, variant, output, device, smoke=False, *, settings, assemble, classes, axes):
    if not 0 <= settings["seed"] < 2**32 or settings["epochs"] < 1 or settings["patience"] < 1:
        raise ValueError("Seed must be in [0, 2**32); epochs and patience must be positive")
    output, device = Path(output), torch.device(device)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {"status": "incomplete", "failures": [], "kind": "gpu-smoke" if smoke else "comparison"}
    provenance = {"command": sys.orig_argv, "started_utc": datetime.now(timezone.utc).isoformat()}
    run = None
    write_json(output/"status.json", status)
    try:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        if smoke and (source != "road-waymo" or variant != "B" or device.type != "cuda"):
            raise ValueError("Smoke requires ROAD BiLSTM on CUDA")
        deterministic = seed_run(settings["seed"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prepared = assemble(source, configuration, variant)
        statistics = prepared["normalization"]
        config = prepared["config"] | {"source": source, "configuration": configuration, "variant": variant,
                  "classes": classes, "settings": settings, "determinism": deterministic,
                  "device": str(device), "kind": status["kind"], "smoke_limits": {"epochs": 1, "training_batches": 2, "validation_batches": 1, "held_out_tests": False} if smoke else None}
        write_json(output/"config.json", config)
        diff = subprocess.check_output(["git", "diff", "HEAD"])
        (output/"source-diff.patch").write_bytes(diff)
        provenance.update(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            diff_sha256=hashlib.sha256(diff).hexdigest(), code_sha256={str(p): checksum(p) for p in sorted(Path("src").rglob("*.py"))},
            versions={p: version(p) for p in ("numpy", "torch", "wandb", "matplotlib", "pillow", "pyarrow")}, python=platform.python_version(),
            dependency_lock_sha256=checksum("uv.lock"), pyproject_sha256=checksum("pyproject.toml"),
            cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version(), host=platform.node(),
            hardware={"platform": platform.platform(), "cpu": platform.processor() or "unknown", "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            seed=settings["seed"], determinism=deterministic, source_statistics=statistics,
            cpu_threads={"torch": torch.get_num_threads(), "environment": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
            datasets=prepared["datasets"])
        model = prepared["build_model"]().to(device)
        provenance["parameters"] = sum(p.numel() for p in model.parameters())
        write_json(output/"provenance.json", provenance)
        dataset = prepared["dataset"]
        training = track_batches(dataset(source, "training"), training=True, seed=settings["seed"])
        validation = track_batches(dataset(source, "validation"), seed=settings["seed"])
        # The same training loader survives every epoch; smoke only restricts iteration.
        class SmokeBatches:
            def __iter__(self):
                return islice(training, 2)
        def validation_predictions(m):
            tracks = predict(m, islice(validation, 1) if smoke else validation, device)
            for t in tracks:
                identity = json.loads(t["locator"])
                t["group"] = identity.get("clip", identity.get("scenario"))
            return tracks
        model_name = {"A": "MLP", "B": "BiLSTM"}[variant]
        attempt_time = datetime.fromisoformat(provenance["started_utc"]).strftime("%Y%m%d-%H%M%S")
        run_name = f"{config['experiment']}-{source}-{configuration}-{model_name}-seed{settings['seed']}-{attempt_time}"
        if smoke:
            run_name += "-smoke"
        run = wandb.init(entity="yammo-unipd", project="pedestrian-behaviour-labeling", name=run_name,
                         job_type=status["kind"], config=config | {"provenance": provenance},
                         dir=str(output.resolve()), save_code=False, resume="never", mode="online")
        provenance["wandb"] = {"id": run.id, "url": run.url}
        configure_logging(run)
        def on_step(step, loss):
            row = {"optimizer_step": step, "training_ce": loss}
            with (output/"steps.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
            run.log({"optimizer_step": step, "train/loss_step": loss})
        def on_epoch(row, history):
            write_json(output/"history.json", history)
            provenance.update(completed_epochs=len(history), last_completed_epoch=row["epoch"],
                exposure={k: sum(r[k] for r in history) for k in ("optimizer_steps", "processed_tracks", "context_slots", "gt_slots", "padding_slots", "batch_slots")})
            write_json(output/"provenance.json", provenance)
            run.log({"epoch": row["epoch"], "train/loss_epoch": row["training_ce"],
                     "val/loss": row["validation_ce"], "val/macro_f1": row["validation_f1"]})
        fitted = fit(model, SmokeBatches() if smoke else training,
                     lambda m: prediction_metrics(validation_predictions(m), len(classes)),
                     torch.optim.AdamW(model.parameters(), **settings["optimizer"]), device, output,
                     {"config": config, "normalization": statistics, "provenance_file": "provenance.json"}, on_epoch,
                     epochs=1 if smoke else settings["epochs"], patience=settings["patience"], clip_norm=settings["clip_norm"], on_step=on_step)
        selected = reload_checkpoint(output/"best.pt", model, device)
        evaluations = {}
        for d in ([source] if smoke else list(prepared["datasets"])):
            destination = output/(d+"-validation-smoke" if smoke else d+"-test")
            destination.mkdir()
            tracks = validation_predictions(model) if smoke else predict(model, track_batches(dataset(d, "test"), seed=settings["seed"]), device)
            for t in tracks:
                metadata, arrays = load_track(t["archive"])
                identity = json.loads(t["locator"])
                t["group"] = identity.get("clip", identity.get("scenario"))
                t["conditions"], t["source_flags"] = prepared["conditions"](metadata, arrays, t, d)
                t["source_times_us"] = source_times(metadata)
                t["native_extent_us"] = metadata["native_extent_us"]
                t["grid_times_us"] = (metadata["native_extent_us"][0]+np.arange(len(t["targets"]))*200000).tolist()
            save_predictions(destination/"predictions.npz", tracks)
            result = {"split": "validation-smoke" if smoke else "test", "primary": prediction_metrics(tracks, len(classes)),
                      "strata": stratified_metrics(tracks, len(classes), axes)}
            write_json(destination/"metrics.json", result)
            evaluations[d] = result
        history = fitted["history"]
        provenance.update(selection=fitted["selection"], stopping_reason=fitted["stopping_reason"], completed_epochs=len(history),
            exposure={k: sum(r[k] for r in history) for k in ("optimizer_steps", "processed_tracks", "context_slots", "gt_slots", "padding_slots", "batch_slots")},
            selected_checkpoint_epoch=selected["epoch"])
        provenance["exposure"]["padding_fraction"] = provenance["exposure"]["padding_slots"]/provenance["exposure"]["batch_slots"]
        plot_paths = plots(output/"plots", history, fitted["selection"], evaluations, classes)
        provenance["elapsed_seconds"] = time.monotonic()-started
        provenance["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        write_json(output/"provenance.json", provenance)
        log_final(run, evaluations, plot_paths, provenance, classes, axes)
        run.finish()
        run = None
        status["status"] = "complete"
        return fitted
    except BaseException as error:
        status["status"] = "failed"
        status["failures"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        if run is not None:
            try:
                run.finish(exit_code=1)
            except Exception as finish_error:
                status["failures"].append({"type": type(finish_error).__name__, "message": str(finish_error)})
        raise
    finally:
        provenance.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" and torch.cuda.is_available() else None)
        write_json(output/"provenance.json", provenance)
        write_json(output/"status.json", status)

