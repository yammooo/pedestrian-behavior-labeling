"""Bounded same-source shared-vs-independent LiDAR update and validation check."""

import argparse
from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import torch

from pedestrian_behavior.accumulation import train_accumulated_epoch
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches
from pedestrian_behavior.data.prepared_points import PreparedPoints
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import predict
from pedestrian_behavior.experiments import e001b_run, e002a_run, e002a_features, e002b, e002b_group, e002b_run
from pedestrian_behavior.inspection.pretrained import DITR_CONFIG_SHA256, DITR_REVISION, DITR_SHA256
from pedestrian_behavior.shared_accumulation import rng_state, set_rng_state, train_shared_accumulated_epoch
from pedestrian_behavior.training import seed_run
from .run import write_json


ARM = "g-dditr"
VARIANTS = tuple(e002b.VARIANTS)
SETTINGS = e002a_run.SETTINGS


class CountingDataset:
    def __init__(self, dataset):
        self.dataset = dataset
        self.entries = dataset.entries
        self.reads = 0

    def __len__(self):
        return len(self.dataset)

    def cost(self, index):
        return self.dataset.cost(index)

    def __getitem__(self, index):
        self.reads += 1
        return self.dataset[index]


class PackingDigest:
    def __init__(self):
        self.digest = hashlib.sha256()

    def __call__(self, row):
        value = {key: row[key] for key in ("logical_indices", "physical_indices", "locators", "lengths", "point_counts")}
        self.digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())

    def hexdigest(self):
        return self.digest.hexdigest()


def _models(profile, device):
    models, optimizers, states = {}, {}, {}
    for variant in VARIANTS:
        seed_run(SETTINGS["seed"])
        model = e002b.build_model(ARM, variant, profile).to(device)
        models[variant] = model
        states[variant] = rng_state(device)
        optimizers[variant] = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                                 **SETTINGS["optimizer"])
    return models, optimizers, states


def _logical(dataset, profile, training, updates, on_pack=None):
    return LogicalBatches(dataset, profile, training=training, seed=SETTINGS["seed"],
                          max_updates=updates, on_pack=on_pack, prefetch=True)


def _measure_independent(dataset, profile, device, warmup_steps, *, updates, warmup):
    counter = CountingDataset(dataset)
    row_results, model_states, model_rng_states = {}, {}, {}
    model_init_seconds = 0.
    peak_allocated, peak_reserved = 0, 0
    for variant in VARIANTS:
        init_start = time.perf_counter()
        seed_run(SETTINGS["seed"])
        model = e002b.build_model(ARM, variant, profile).to(device)
        optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                      **SETTINGS["optimizer"])
        initial_rng = rng_state(device)
        model_init_seconds += time.perf_counter() - init_start
        set_rng_state(initial_rng, device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before_reads = counter.reads
        packing = PackingDigest()
        loader = _logical(counter, profile, True, updates, packing)
        result = train_accumulated_epoch(model, loader, optimizer, device, SETTINGS["clip_norm"],
            peak_lr=SETTINGS["optimizer"]["lr"], warmup_steps=warmup_steps)
        row_results[variant] = result | {"input_track_reads": counter.reads - before_reads,
                                       "passes": loader.iteration, "preparation": loader.stats,
                                       "packing_sha256": packing.hexdigest()}
        model_states[variant] = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        model_rng_states[variant] = rng_state(device)
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(device))
        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(device))
        del model, optimizer, loader
        gc.collect()
    torch.cuda.synchronize(device)
    return {"seconds": sum(row["seconds"] for row in row_results.values()),
            "model_init_seconds": model_init_seconds, "input_track_reads": counter.reads,
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "models": row_results, "model_states": model_states, "rng_states": model_rng_states,
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "warmup": warmup}


def _measure_shared(dataset, profile, device, warmup_steps, *, updates, warmup):
    counter = CountingDataset(dataset)
    init_start = time.perf_counter()
    models, optimizers, states = _models(profile, device)
    model_init_seconds = time.perf_counter() - init_start
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    packing = PackingDigest()
    loader = _logical(counter, profile, True, updates, packing)
    rows, states = train_shared_accumulated_epoch(models, loader, optimizers, device, SETTINGS["clip_norm"],
        rng_states=states, peak_lr=SETTINGS["optimizer"]["lr"], warmup_steps=warmup_steps)
    torch.cuda.synchronize(device)
    model_states = {variant: {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    for variant, model in models.items()}
    return {"seconds": max(row["shared_group_seconds"] for row in rows.values()),
            "input_track_reads": counter.reads,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "models": rows,
            "model_states": model_states, "rng_states": states,
            "model_init_seconds": model_init_seconds,
            "preparation": loader.stats, "packing_sha256": packing.hexdigest(),
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "warmup": warmup, "loader_passes": loader.iteration}


def _measure_validation_independent(dataset, profile, device, model_states):
    counter = CountingDataset(dataset)
    seconds, counts, peaks, reserves, digests = {}, {}, [], [], {}
    for variant in VARIANTS:
        seed_run(SETTINGS["seed"])
        model = e002b.build_model(ARM, variant, profile).to(device)
        model.load_state_dict(model_states[variant])
        loader = PredictionBatches(_logical(counter, profile, False, 1))
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before, start = counter.reads, time.perf_counter()
        tracks = predict(model, loader, device, lambda current, batch, _device: current(batch))
        seconds[variant] = time.perf_counter() - start
        counts[variant] = {"track_reads": counter.reads - before, "tracks": len(tracks)}
        digest = hashlib.sha256()
        for track in tracks:
            digest.update(track["locator"].encode())
            digest.update(track["logits"].tobytes())
        digests[variant] = digest.hexdigest()
        if device.type == "cuda":
            peaks.append(torch.cuda.max_memory_allocated(device))
            reserves.append(torch.cuda.max_memory_reserved(device))
        counts[variant]["preparation"] = loader.logical.stats
        del model, loader, tracks
        gc.collect()
    return {"seconds": sum(seconds.values()), "seconds_by_model": seconds,
            "input_track_reads": counter.reads, "track_reads_by_model": counts,
            "logits_sha256_by_model": digests,
            "peak_allocated_bytes": max(peaks, default=0),
            "peak_reserved_bytes": max(reserves, default=0),
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def _open_source(args, source, profile, stack):
    config, transform = e002a_run.load_point_configuration(args.probe)
    point_root = args.indexed_point_cache or (args.outputs_root / "experiments/E002a" / ("indexed-" + source))
    guard_args = argparse.Namespace(**(vars(args) | {"source": source, "arm": ARM, "point_cache_root": point_root}))
    e002b_run.validate_roots(guard_args, config)
    baseline = e001b_run.assemble(source, "geometry", "B", e001_root=args.outputs_root / "experiments/E001",
                                  e001b_root=args.outputs_root / "experiments/E001b")
    catalog = e002a_features.FeatureCatalog(args.outputs_root, source, config, transform)
    points = PreparedPoints(point_root, source, catalog.identity, catalog.tracks,
        {"revision": DITR_REVISION, "checkpoint": DITR_SHA256, "configuration": DITR_CONFIG_SHA256},
        frame_workers=args.point_workers)
    stack.callback(points.close)
    supply = e002a_run.PreparedSupply(catalog, None, points, None)
    result = {}
    for split in ("training", "validation"):
        dataset = baseline["dataset"](source, split)
        for entry in dataset.entries:
            if entry["slots"] ** 2 > profile["temporal_cells"]:
                raise ValueError("Complete track exceeds reference temporal ceiling")
            points.require_track(entry["locator"])
            if points.counts[entry["locator"]].max(initial=0) > profile["frame_tokens"]:
                raise ValueError("Prepared frame exceeds point-token ceiling")
        result[split] = e002a_run.FeatureDataset(dataset, supply, source, ARM)
    return {"datasets": result, "baseline": baseline, "catalog": catalog, "points": points,
            "config": config, "transform": transform, "protected": catalog.protected | points.protected}


def _gpu_processes():
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader"], check=True, capture_output=True, text=True)
        return {"status": "available", "rows": [line.strip() for line in result.stdout.splitlines() if line.strip()]}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"status": "unavailable", "error": str(error)}


def run(args):
    torch.set_num_threads(2)
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("Shared-LiDAR calibration requires the RTX CUDA device")
    output = args.output.resolve()
    if not output.is_relative_to(args.outputs_root.resolve() / "experiments/E002b"):
        raise ValueError("Calibration output must be under outputs-root/experiments/E002b")
    reference_path = args.reference_profile.resolve()
    reference = json.loads(reference_path.read_text())
    if reference.get("status") != "complete" or reference.get("point_storage_protocol") != e002a_run.POINT_PROTOCOL:
        raise ValueError("Completed indexed E002a reference profile required")
    profile = e002a_run.validate_profile(reference["profile"])
    if profile["max_tracks"] != 64:
        raise ValueError("Shared E002b retains the complete logical64-track schedule")
    reference_workloads = [row for row in reference.get("workloads", [])
        if row.get("status") == "complete" and row.get("arm") in ("g-dditr", "g-rgb-dditr")
        and isinstance(row.get("peak_allocated"), int) and row["peak_allocated"] > 0]
    if not reference_workloads:
        raise ValueError("Reference profile has no measured LiDAR spatial allocation for memory preflight")
    reference_peak = max(reference_workloads, key=lambda row: row["peak_allocated"])
    free_vram, total_vram = torch.cuda.mem_get_info(torch.device(args.device))
    estimated_group_peak = int(reference_peak["peak_allocated"] * 1.10 + 512 * 2**20)
    if free_vram < estimated_group_peak:
        raise ValueError(f"Insufficient free VRAM for bounded group benchmark: {free_vram} available, "
                         f"{estimated_group_peak} estimated from matched LiDAR profile")
    output.mkdir(parents=True, exist_ok=False)
    record = {"status": "incomplete", "experiment": "E002b", "execution": "shared-g-dditr",
        "device": args.device, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": sys.orig_argv, "code": e002b_group.resource_code(), "profile": profile,
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "total_vram_bytes": total_vram,
        "free_vram_preflight_bytes": free_vram,
        "memory_preflight": {"reference_workload": {key: reference_peak.get(key)
            for key in ("name", "tracks", "slots", "padded_tokens", "mode", "peak_allocated")},
            "margin": {"multiplier": 1.10, "fixed_bytes": 512 * 2**20},
            "estimated_group_peak_bytes": estimated_group_peak},
        "allocated_ceiling_bytes": int(total_vram * .78),
        "point_storage_protocol": e002a_run.POINT_PROTOCOL,
        "reference_profile": {"path": str(reference_path), "sha256": checksum(reference_path),
                              "status": reference["status"], "code": reference.get("code")},
        "settings": SETTINGS, "model_specs": {f"{ARM}/{variant}": e002b.model_spec(ARM, variant)
                                                    for variant in VARIANTS},
        "model_coverage": [f"{ARM}/{variant}" for variant in VARIANTS],
        "workloads": [], "sources": {}, "held_out_tests": False,
        "gpu_processes_before": _gpu_processes(),
        "timing_scope": "CPU point preparation plus model forward/backward/clip/AdamW; model construction and source assembly reported separately",
        "interpretation": "One warm and one measured complete logical64 update, plus one bounded source-validation batch per source; discarded weights, no W&B or full training"}
    write_json(output / "profile.json", record)
    telemetry = telemetry_log = None
    try:
        try:
            telemetry_log = (output / "system.csv").open("w")
            telemetry = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu",
                "--format=csv", "-l", "1"], stdout=telemetry_log, stderr=subprocess.STDOUT)
            record["gpu_telemetry"] = {"status": "sampling", "path": str(output / "system.csv"),
                                       "scope": "whole-GPU, including other active processes"}
        except OSError as error:
            record["gpu_telemetry"] = {"status": "unavailable", "error": str(error)}
            if telemetry_log is not None:
                telemetry_log.close()
                telemetry_log = None
        stack = ExitStack()
        try:
            for source in ("road-waymo", "loki"):
                assembly_started = time.perf_counter()
                opened = _open_source(args, source, profile, stack)
                assembly_seconds = time.perf_counter() - assembly_started
                training = opened["datasets"]["training"]
                validation = opened["datasets"]["validation"]
                warmup_steps = 3 * ((len(training) + 63) // 64)
                if len(training) < 64 or len(validation) < 1:
                    raise ValueError("Calibration source is too small for a complete logical64 update")
                # A matched warm update touches the same indexed batches and preparation cache.
                _measure_shared(training, profile, torch.device(args.device), warmup_steps, updates=1, warmup=True)
                _measure_independent(training, profile, torch.device(args.device), warmup_steps, updates=1, warmup=True)
                torch.cuda.synchronize()
                independent = _measure_independent(training, profile, torch.device(args.device), warmup_steps,
                                                   updates=1, warmup=False)
                shared = _measure_shared(training, profile, torch.device(args.device), warmup_steps,
                                         updates=1, warmup=False)
                diffs = {}
                for variant in VARIANTS:
                    maxima = [float((shared["model_states"][variant][key] - independent["model_states"][variant][key]).abs().max())
                              for key in shared["model_states"][variant]]
                    diffs[variant] = max(maxima, default=0.)
                    if diffs[variant] > 2e-6:
                        raise ValueError(f"Shared/independent update differs beyond FP32 tolerance: {source}/{variant}")
                    loss_diff = abs(shared["models"][variant]["training_ce"] - independent["models"][variant]["training_ce"])
                    if loss_diff > 1e-7:
                        raise ValueError(f"Shared/independent training loss differs: {source}/{variant}")
                    left, right = shared["rng_states"][variant], independent["rng_states"][variant]
                    if not torch.equal(left[0], right[0]) or (left[1] is not None and not torch.equal(left[1], right[1])):
                        raise ValueError(f"Shared/independent training RNG differs: {source}/{variant}")
                    if shared["packing_sha256"] != independent["models"][variant]["packing_sha256"]:
                        raise ValueError(f"Shared/independent packing differs: {source}/{variant}")
                shared_prediction = None
                shared_validation = None
                # Shared validation evaluation reuses each prepared physical batch for all three models.
                models, _, _ = _models(profile, torch.device(args.device))
                for variant in VARIANTS:
                    models[variant].load_state_dict(shared["model_states"][variant])
                validation_counter = CountingDataset(validation)
                validation_loader = PredictionBatches(_logical(validation_counter, profile, False, 1))
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
                begin = time.perf_counter()
                shared_prediction = e002b_group.predict_shared(models, validation_loader, torch.device(args.device))
                shared_validation = {"seconds": time.perf_counter() - begin,
                    "input_track_reads": validation_counter.reads,
                    "tracks_by_model": {variant: len(rows) for variant, rows in shared_prediction.items()},
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(args.device)),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.device(args.device)),
                    "preparation": validation_loader.logical.stats,
                    "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
                prediction_hashes = {}
                for variant, tracks in shared_prediction.items():
                    digest = hashlib.sha256()
                    for track in tracks:
                        digest.update(track["locator"].encode())
                        digest.update(track["logits"].tobytes())
                    prediction_hashes[variant] = digest.hexdigest()
                del models, shared_prediction
                gc.collect()
                independent_validation = _measure_validation_independent(validation, profile,
                    torch.device(args.device), independent["model_states"])
                for variant in VARIANTS:
                    if prediction_hashes[variant] != independent_validation["logits_sha256_by_model"][variant]:
                        raise ValueError(f"Shared/independent validation prediction differs: {source}/{variant}")
                shared_validation["independent_control"] = independent_validation
                record["sources"][source] = {"warmup_updates_per_path": 1,
                    "source_assembly_seconds": assembly_seconds,
                    "measured_logical_updates_per_model": 1,
                    "shared_training": {key: value for key, value in shared.items() if key not in ("model_states", "rng_states")},
                    "independent_training": {key: value for key, value in independent.items() if key not in ("model_states", "rng_states")},
                    "validation": shared_validation,
                    "validation_tracks": next(iter(shared_validation["tracks_by_model"].values())),
                    "max_parameter_abs_diff": diffs,
                    "max_training_ce_abs_diff": {variant: abs(shared["models"][variant]["training_ce"] -
                        independent["models"][variant]["training_ce"]) for variant in VARIANTS},
                    "rng_states_match": True,
                    "packing_sha256": shared["packing_sha256"],
                    "packing_matches_independent": True,
                    "validation_logits_match": prediction_hashes == independent_validation["logits_sha256_by_model"],
                    "speedup": independent["seconds"] / shared["seconds"],
                    "preparation_read_ratio": independent["input_track_reads"] / shared["input_track_reads"],
                    "validation_preparation_read_ratio": independent_validation["input_track_reads"] /
                        shared_validation["input_track_reads"],
                    "point_reader": opened["points"].snapshot(),
                    "status": "complete"}
                record["workloads"].append({"source": source, "phase": "shared-vs-independent",
                    "tracks": 64, "variants": list(VARIANTS), "status": "complete"})
                # Current selected checkpoint/loss is not part of this resource test.
                gc.collect()
                record["max_peak_allocated_bytes"] = max(record.get("max_peak_allocated_bytes", 0),
                    record["sources"][source]["shared_training"]["peak_allocated_bytes"],
                    record["sources"][source]["independent_training"]["peak_allocated_bytes"],
                    shared_validation["peak_allocated_bytes"], independent_validation["peak_allocated_bytes"])
                record["max_peak_reserved_bytes"] = max(record.get("max_peak_reserved_bytes", 0),
                    record["sources"][source]["shared_training"]["peak_reserved_bytes"],
                    record["sources"][source]["independent_training"]["peak_reserved_bytes"],
                    shared_validation["peak_reserved_bytes"], independent_validation["peak_reserved_bytes"])
                record["host_peak_rss_bytes"] = max(record.get("host_peak_rss_bytes", 0),
                    record["sources"][source]["shared_training"]["host_peak_rss_bytes"],
                    record["sources"][source]["independent_training"]["host_peak_rss_bytes"],
                    shared_validation["host_peak_rss_bytes"], independent_validation["host_peak_rss_bytes"])
                if record["max_peak_allocated_bytes"] > record["allocated_ceiling_bytes"]:
                    raise ValueError("Shared-LiDAR calibration exceeded 78% allocated VRAM ceiling")
                write_json(output / "profile.json", record)
                print(source, "shared", round(shared["seconds"], 2), "s; independent",
                      round(independent["seconds"], 2), "s; reads",
                      independent["input_track_reads"], "->", shared["input_track_reads"],
                      "; peak", record["max_peak_allocated_bytes"], flush=True)
                for path, expected_sha in opened["protected"].items():
                    if checksum(path) != expected_sha:
                        raise ValueError("Protected source/cache file changed during calibration")
                stack.close()
                stack = ExitStack()
        finally:
            stack.close()
        record["status"] = "complete"
        return record
    except BaseException:
        record["status"] = "failed"
        record["failures"] = [traceback.format_exc()]
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=10)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
        if telemetry_log is not None:
            telemetry_log.close()
        record["gpu_processes_after"] = _gpu_processes()
        record["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json(output / "profile.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--indexed-point-cache", type=Path)
    parser.add_argument("--point-workers", type=int, default=e002a_run.DEFAULT_POINT_WORKERS)
    parser.add_argument("--output", type=Path,
        default=Path("outputs/experiments/E002b/calibration-lidar-group"))
    run(parser.parse_args())
