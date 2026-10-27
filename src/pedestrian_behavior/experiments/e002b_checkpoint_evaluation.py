"""Evaluation-only export of selected checkpoints from interrupted shared E002b runs."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
import torch
import wandb

from pedestrian_behavior.data.indexed_points import PROTOCOL as POINT_PROTOCOL
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches, pack_tracks
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.data.prepared_points import PreparedPoints
from pedestrian_behavior.data.bbox import load_bbox
from pedestrian_behavior.evaluation import (load_predictions, prediction_metrics, save_predictions,
                                            stratified_metrics)
from pedestrian_behavior.experiments import e001, e001b, e001b_run, e002a_run, e002b
from pedestrian_behavior.experiments.e002a_features import FeatureCatalog
from pedestrian_behavior.experiments.e002b_group import AXES, ARM, CLASSES, _decorate, predict_shared
from pedestrian_behavior.experiments.run import configure_logging, dataset_references, log_final, plots, write_json
from pedestrian_behavior.inspection.pretrained import DITR_CONFIG_SHA256, DITR_REVISION, DITR_SHA256
from pedestrian_behavior.training import seed_run


SOURCE = "loki"
EXPECTED_ORIGINAL = "shared-lidar-20261026-060327-526425"
BEST_EPOCHS = {"norm": 11, "temporal-small": 12, "branches-small": 17}
EXPECTED_VALIDATION_TRACKS = 1861
METRIC_RTOL = 1e-6
METRIC_ATOL = 1e-8


def _same_value(expected, actual, path="metric"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"Checkpoint metric structure differs at {path}")
        for key in expected:
            _same_value(expected[key], actual[key], f"{path}.{key}")
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise ValueError(f"Checkpoint metric structure differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _same_value(left, right, f"{path}[{index}]")
        return
    if expected is None or isinstance(expected, (str, bool)):
        if expected != actual:
            raise ValueError(f"Checkpoint metric differs at {path}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if isinstance(expected, int) and isinstance(actual, int):
            equal = expected == actual
        else:
            equal = math.isclose(float(expected), float(actual), rel_tol=METRIC_RTOL, abs_tol=METRIC_ATOL)
        if not equal:
            raise ValueError(f"Checkpoint metric differs at {path}: {expected!r} != {actual!r}")
        return
    raise ValueError(f"Unsupported checkpoint metric type at {path}: {type(expected).__name__}")


def verify_checkpoint_metrics(saved, recomputed):
    """Require full source-validation metric and confusion parity with the checkpoint."""
    _same_value(saved, recomputed)
    return {"matches_checkpoint": True, "rtol": METRIC_RTOL, "atol": METRIC_ATOL,
            "track_ce": recomputed["track_ce"],
            "track_weighted_macro_f1": recomputed["track_weighted"]["macro_f1"],
            "track_weighted_confusion": recomputed["track_weighted"]["confusion"]}


def _load_original(original_root):
    original_root = Path(original_root).resolve()
    if original_root.name != EXPECTED_ORIGINAL or original_root.parent.name != "runs":
        raise ValueError("Only the identified interrupted LOKI shared-LiDAR run may be evaluated")
    group_status = json.loads((original_root / "status.json").read_text())
    if (group_status.get("status") != "incomplete" or group_status.get("execution") != "shared-g-dditr"
            or group_status.get("failures")):
        raise ValueError("Original group is not the expected failure-free interrupted training record")
    group_provenance = json.loads((original_root / "provenance.json").read_text())
    checkpoints, configs, histories, origins = {}, {}, {}, {}
    for variant, epoch in BEST_EPOCHS.items():
        directory = original_root / variant
        status = json.loads((directory / "status.json").read_text())
        config = json.loads((directory / "config.json").read_text())
        history = json.loads((directory / "history.json").read_text())
        if status.get("status") != "incomplete" or status.get("failures"):
            raise ValueError(f"{variant}: expected an interrupted, failure-free 23-epoch child")
        if (config.get("experiment") != "E002b" or config.get("source") != SOURCE
                or config.get("arm") != ARM or config.get("model_variant") != variant
                or config.get("execution") != "shared-g-dditr" or config.get("smoke")
                or config.get("held_out_tests") is not False):
            raise ValueError(f"{variant}: original checkpoint configuration is not the expected full LOKI run")
        if len(history) != 23 or [row.get("epoch") for row in history] != list(range(1, 24)):
            raise ValueError(f"{variant}: original epoch history is incomplete or discontinuous")
        best_path, last_path = directory / "best.pt", directory / "last.pt"
        best = torch.load(best_path, map_location="cpu", weights_only=True)
        last = torch.load(last_path, map_location="cpu", weights_only=True)
        spec = e002b.model_spec(ARM, variant)
        if config.get("model") != spec or best.get("config", {}).get("model") != spec:
            raise ValueError(f"{variant}: saved model specification mismatch")
        if best.get("epoch") != epoch or best.get("selection", {}).get("epoch") != epoch:
            raise ValueError(f"{variant}: selected checkpoint epoch mismatch")
        if last.get("epoch") != 23 or last.get("selection", {}).get("epoch") != epoch:
            raise ValueError(f"{variant}: last checkpoint does not retain the selected epoch")
        if (best.get("normalization") != last.get("normalization")
                or best.get("normalization") != group_provenance.get("source_statistics")):
            raise ValueError(f"{variant}: checkpoint normalization differs from the interrupted run")
        selected_row = history[epoch - 1]
        if (not selected_row.get("selected") or
                not math.isclose(selected_row["validation_f1"], best["selection"]["f1"], rel_tol=METRIC_RTOL, abs_tol=METRIC_ATOL)
                or not math.isclose(selected_row["validation_ce"], best["selection"]["ce"], rel_tol=METRIC_RTOL, abs_tol=METRIC_ATOL)):
            raise ValueError(f"{variant}: history selection disagrees with selected checkpoint")
        model = e002b.build_model(ARM, variant, config["profile"])
        model.load_state_dict(best["model_state"], strict=True)
        for name, tensor in best["model_state"].items():
            if not torch.isfinite(tensor).all().item():
                raise ValueError(f"{variant}: non-finite checkpoint tensor {name}")
        origin = group_provenance.get("wandb", {}).get(variant, {})
        if not origin.get("url") or not origin.get("id"):
            raise ValueError(f"{variant}: original W&B run lineage is missing")
        checkpoints[variant] = best
        configs[variant] = config
        histories[variant] = history
        origins[variant] = {"run_id": origin["id"], "run_url": origin["url"],
                            "best_checkpoint_sha256": checksum(best_path),
                            "last_checkpoint_sha256": checksum(last_path),
                            "selected_epoch": epoch, "completed_training_epochs": 23}
    return original_root, group_status, group_provenance, checkpoints, configs, histories, origins


def _snapshot_original(original):
    paths = [original / "status.json", original / "provenance.json"]
    for variant in BEST_EPOCHS:
        directory = original / variant
        paths.extend(directory / name for name in ("config.json", "status.json", "history.json", "best.pt", "last.pt"))
    return {str(path): checksum(path) for path in paths}


def _source_reference(record):
    bbox = record["bbox"]
    return {key: record[key] for key in ("setup_sha256", "collection_sha256", "policy_sha256",
            "native_policy", "population", "support")} | {"bbox": {key: bbox[key] for key in
            ("manifest_sha256", "policy_sha256", "coverage")}}


def _safe_source_identity(record):
    return {"setup_sha256": record["setup_sha256"], "collection_manifest_sha256": record["collection_sha256"],
            "bbox_manifest_sha256": record["bbox"]["manifest_sha256"],
            "bbox_policy_sha256": record["bbox"]["policy_sha256"]}


def _validate_calibration(profile_path, configs, device):
    profile_path = Path(profile_path).resolve()
    record = json.loads(profile_path.read_text())
    expected_specs = {f"{ARM}/{variant}": e002b.model_spec(ARM, variant) for variant in BEST_EPOCHS}
    if (record.get("status") != "complete" or record.get("experiment") != "E002b"
            or record.get("execution") != "shared-g-dditr" or record.get("device") != str(device)
            or record.get("torch") != str(torch.__version__) or record.get("cuda") != torch.version.cuda
            or record.get("model_specs") != expected_specs
            or set(record.get("model_coverage", ())) != set(expected_specs)
            or record.get("point_storage_protocol") != POINT_PROTOCOL):
        raise ValueError("Saved shared-LiDAR calibration does not cover these checkpoints/backend")
    profile = e002a_run.validate_profile(record.get("profile", {}))
    for variant, config in configs.items():
        if (config.get("profile") != profile or config.get("profile_sha256") != checksum(profile_path)
                or config.get("point_storage_protocol") != POINT_PROTOCOL):
            raise ValueError(f"{variant}: saved profile or indexed-storage identity mismatch")
    return profile, record, profile_path, checksum(profile_path)


def _inference_code_identity():
    from pedestrian_behavior.data import bbox, features, indexed_points, loading, multimodal, preparation, prepared_points
    from pedestrian_behavior.datasets import loki
    from pedestrian_behavior.experiments import e002a_features, e002a_run, e002b_group
    from pedestrian_behavior.inspection import pretrained
    from pedestrian_behavior.models import multimodal as model_code
    from pedestrian_behavior import evaluation
    modules = (model_code, e002b, multimodal, indexed_points, prepared_points, preparation, bbox, features, loading,
               loki, e002a_features, e002a_run, e001, e001b, e001b_run, e002b_group, evaluation, pretrained)
    return {module.__name__: checksum(module.__file__) for module in modules} | {__name__: checksum(__file__)}


def _verify_frozen_code(old_provenance):
    saved = {}
    for path, digest in old_provenance.get("code_sha256", {}).items():
        candidate = Path(path)
        try:
            candidate = candidate.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            pass
        saved[candidate.as_posix()] = digest
    current = _inference_code_identity()
    critical = ("src/pedestrian_behavior/models/multimodal.py", "src/pedestrian_behavior/experiments/e002b.py",
                "src/pedestrian_behavior/data/multimodal.py", "src/pedestrian_behavior/data/indexed_points.py",
                "src/pedestrian_behavior/data/prepared_points.py", "src/pedestrian_behavior/data/preparation.py",
                "src/pedestrian_behavior/data/bbox.py", "src/pedestrian_behavior/data/features.py",
                "src/pedestrian_behavior/data/loading.py", "src/pedestrian_behavior/datasets/loki.py",
                "src/pedestrian_behavior/experiments/e001.py",
                "src/pedestrian_behavior/experiments/e001b.py", "src/pedestrian_behavior/experiments/e001b_run.py",
                "src/pedestrian_behavior/experiments/e002a_features.py",
                "src/pedestrian_behavior/inspection/pretrained.py")
    changed = [path for path in critical
               if saved.get(path) != current.get(path.removesuffix(".py").replace("/", ".").removeprefix("src."))]
    if changed:
        raise ValueError(f"Inference-critical model/cache code differs from the interrupted run: {changed}")
    return {path: saved[path] for path in critical}


def _prepare_validation(outputs_root, configs, profile):
    outputs_root = Path(outputs_root).resolve()
    probe = outputs_root / "experiments/E002a/native-probe-07"
    config, transform = e002a_run.load_point_configuration(probe)
    e001_root, e001b_root = outputs_root / "experiments/E001", outputs_root / "experiments/E001b"
    setup_path = e001_root / "data-setup/loki/setup.json"
    collection = e001_root / "reader-preparation/loki"
    extension = e001b_root / "bbox-native/loki"
    setup = json.loads(setup_path.read_text())
    extension_record = e001b.load_extension(collection, setup_path, extension)
    if extension_record.get("baseline_normalization") != setup["normalization"]["K+T+R"]:
        raise ValueError("LOKI bbox cache baseline normalization differs from frozen E001")
    statistics = {"baseline": setup["normalization"]["K+T+R"],
                  "bbox": extension_record["normalization"]}
    datasets = dataset_references({SOURCE: collection}, {SOURCE: setup_path}, {SOURCE: setup})
    datasets[SOURCE]["bbox"] = {
        "manifest_path": str((extension / "manifest.json").resolve()),
        "manifest_sha256": checksum(extension / "manifest.json"),
        "policy_sha256": hashlib.sha256(json.dumps(extension_record["policy"], sort_keys=True).encode()).hexdigest(),
        "coverage": extension_record["coverage"],
    }
    bbox_entries = {row["locator"]: row for row in extension_record["tracks"]}
    def baseline_conditions(metadata, arrays, track, source):
        native, flags = e001.observation_conditions(metadata, arrays, track["targets"], track["gt_valid"], source)
        entry = bbox_entries[track["locator"]]
        archive = extension / "tracks" / entry["archive"]
        if checksum(archive) != entry["bbox_sha256"]:
            raise ValueError("LOKI bbox archive changed from the frozen E001b extension")
        _, bbox = load_bbox(archive, metadata, entry["e001_sha256"])
        track.update(bbox_archive=str(archive), bbox_sha256=entry["bbox_sha256"])
        flags.update({key: bbox[key] for key in ("bbox_valid", "bbox_velocity_valid", "bbox_source")})
        return native | e001b_run.bbox_conditions(bbox), flags
    catalog = FeatureCatalog(outputs_root, SOURCE, config, transform)
    point_policy = configs["norm"]["point_loading"]
    points = PreparedPoints(outputs_root / "experiments/E002a/indexed-loki", SOURCE, catalog.identity,
        catalog.tracks, {"revision": DITR_REVISION, "checkpoint": DITR_SHA256,
                         "configuration": DITR_CONFIG_SHA256},
        max_open_scenes=point_policy["open_scenes"], max_metadata_bytes=point_policy["metadata_limit_bytes"],
        frame_workers=point_policy["frame_workers"])
    supply = e002a_run.PreparedSupply(catalog, None, points, None)
    baseline_validation = e001b.dataset_from_extension(collection, setup_path, extension, "validation", "geometry",
        statistics["baseline"], statistics["bbox"])
    dataset = e002a_run.FeatureDataset(baseline_validation, supply, SOURCE, ARM)
    if len(dataset) != EXPECTED_VALIDATION_TRACKS or any(entry["split"] != "validation" for entry in dataset.entries):
        points.close()
        raise ValueError("LOKI source-validation population must be the complete frozen 1,861 tracks")
    source_hashes = {path: digest for path, digest in catalog.protected.items()
                     if Path(path).name in ("setup.json", "manifest.json")}
    source_hashes.update(points.protected)
    source_hashes[str((extension / "manifest.json").resolve())] = checksum(extension / "manifest.json")

    def conditions(metadata, arrays, track, source):
        values, flags = baseline_conditions(metadata, arrays, track, source)
        observed = {"lidar-observed": track["source_flags"]["lidar_valid"].astype(np.int64)}
        return values | observed, flags
    prepared = {"config": {"profile": profile}, "normalization": statistics,
                "datasets": {SOURCE: datasets[SOURCE]}, "dataset": dataset,
                "conditions": conditions, "points": points, "catalog": catalog,
                "source_files_sha256": source_hashes}
    packing = _packing_summary(dataset, profile)
    prepared["packing"] = packing
    return prepared


def _packing_summary(dataset, profile):
    lengths = {i: dataset.entries[i]["slots"] for i in range(len(dataset))}
    costs = {i: dataset.cost(i) for i in range(len(dataset))}
    max_cells = max_tokens = max_tracks = max_frame_tokens = 0
    physical_batches = oversized_singletons = max_oversized_tokens = 0
    for start in range(0, len(dataset), 64):
        logical = list(range(start, min(start + 64, len(dataset))))
        groups = pack_tracks(logical, lengths, costs, profile)
        for group in groups:
            n, slots = len(group), max(lengths[i] for i in group)
            padded_cells = n * slots * slots
            tokens = sum(costs[i] for i in group)
            frame_tokens = max(int(dataset.supply.points.counts[dataset.entries[i]["locator"]].max(initial=0))
                               for i in group)
            oversized = _validate_packed_shape(n, padded_cells, tokens, frame_tokens, profile)
            max_tracks, max_cells, max_tokens = max(max_tracks, n), max(max_cells, padded_cells), max(max_tokens, tokens)
            max_frame_tokens = max(max_frame_tokens, frame_tokens)
            oversized_singletons += oversized
            if oversized:
                max_oversized_tokens = max(max_oversized_tokens, tokens)
            physical_batches += 1
    return {"logical_batch_size": 64, "physical_batches": physical_batches,
            "max_physical_tracks": max_tracks, "max_padded_temporal_cells": max_cells,
            "max_physical_point_tokens": max_tokens, "max_frame_tokens": max_frame_tokens,
            "oversized_singleton_batches": oversized_singletons,
            "max_oversized_singleton_point_tokens": max_oversized_tokens,
            "profile": profile}


def _validate_packed_shape(tracks, temporal_cells, point_tokens, max_frame_tokens, profile):
    oversized_singleton = tracks == 1 and point_tokens > profile["microbatch_tokens"]
    if (tracks > profile["max_tracks"] or temporal_cells > profile["temporal_cells"]
            or max_frame_tokens > profile["frame_tokens"]
            or point_tokens > profile["microbatch_tokens"] and not oversized_singleton):
        raise ValueError("Validation packing exceeds the saved track/token/frame resource bounds")
    return oversized_singleton


def _conditions_from_checkpoint(checkpoint):
    metrics = checkpoint.get("validation")
    if not isinstance(metrics, dict):
        raise ValueError("Selected checkpoint lacks source-validation metrics")
    return metrics


def _write_local_results(output, configs, checkpoints, histories, origins, tracks, evaluations, provenance):
    metric_records, plot_records = {}, {}
    for variant in BEST_EPOCHS:
        directory = output / variant
        directory.mkdir()
        original_config = configs[variant]
        selected = checkpoints[variant]["selection"]
        evaluation_config = original_config | {
            "kind": "checkpoint-evaluation", "evaluation_only": True,
            "execution": "checkpoint-evaluation", "origin_execution": "shared-g-dditr",
            "evaluation_group_id": output.name,
            "training_state": "interrupted", "training_completed_epochs": 23,
            "selected_checkpoint_epoch": BEST_EPOCHS[variant],
            "origin_training_run": origins[variant], "held_out_tests": False,
            "source_normalization": provenance["input_identity"]["normalization"],
        }
        evaluation_config.pop("shared_execution_id", None)
        write_json(directory / "config.json", evaluation_config)
        write_json(directory / "status.json", {"status": "incomplete", "kind": "checkpoint-evaluation",
                    "execution": "checkpoint-evaluation",
                    "evaluation_only": True, "training_state": "interrupted", "failures": []})
        write_json(directory / "history.json", histories[variant])
        variant_tracks = tracks[variant]
        result = evaluations[variant][SOURCE]
        metrics = result["primary"]
        split_dir = directory / f"{SOURCE}-validation"
        split_dir.mkdir()
        prediction_path = split_dir / "predictions.npz"
        save_predictions(prediction_path, variant_tracks)
        write_json(split_dir / "metrics.json", result)
        roundtrip = prediction_metrics(load_predictions(prediction_path), len(CLASSES))
        verify_checkpoint_metrics(metrics, roundtrip)
        plot_paths = plots(directory / "plots", histories[variant], selected,
                           {SOURCE: result}, CLASSES)
        metric_records[variant] = {"primary": metrics, "prediction_roundtrip_verified": True,
            "strata": result["strata"], "predictions_sha256": checksum(prediction_path),
            "prediction_tracks": len(variant_tracks), "plots": [str(path.relative_to(directory)) for path in plot_paths]}
        plot_records[variant] = plot_paths
    provenance = provenance | {"local_results_written": True, "evaluation_only": True,
        "metric_verification": metric_records,
        "output_root": str(output.resolve()), "selected_epochs": dict(BEST_EPOCHS),
        "training_state": "interrupted", "training_completed_epochs": 23}
    for variant in BEST_EPOCHS:
        variant_provenance = provenance | {"model_variant": variant,
            "origin_training_run": origins[variant],
            "selection": checkpoints[variant]["selection"],
            "validation_metrics": metric_records[variant]["primary"],
            "prediction_sha256": metric_records[variant]["predictions_sha256"]}
        write_json(output / variant / "provenance.json", variant_provenance)
    write_json(output / "provenance.json", provenance)
    write_json(output / "evaluation-evidence.json", {"status": "local-results-written",
        "kind": "checkpoint-evaluation", "training_state": "interrupted", "source": SOURCE,
        "training_completed_epochs": 23, "selected_epochs": dict(BEST_EPOCHS),
        "verification": metric_records, "checkpoint_metric_equivalence": "pending", "wandb_uploads": "pending"})
    return metric_records, plot_records, provenance


def run_evaluation(*, outputs_root, original_root, profile_path, output_root, device="cuda"):
    """Export selected LOKI validation predictions and log three evaluation-only W&B runs."""
    outputs_root = Path(outputs_root).resolve()
    original_root = Path(original_root).resolve()
    profile_path = Path(profile_path).resolve()
    output_root = Path(output_root).resolve()
    allowed_parent = outputs_root / "experiments/E002b/runs"
    if not output_root.is_relative_to(allowed_parent) or output_root == allowed_parent:
        raise ValueError("Evaluation exports must be fresh children of E002b/runs")
    if output_root.is_relative_to(original_root) or original_root.is_relative_to(output_root):
        raise ValueError("Evaluation exports and original training run must remain separate")
    if original_root != allowed_parent / EXPECTED_ORIGINAL:
        raise ValueError("Original run path is not the identified canonical interrupted group")
    if output_root.exists():
        raise FileExistsError(output_root)
    if device != "cuda":
        raise ValueError("This recovery export requires the calibrated CUDA backend")
    torch.set_num_threads(2)
    determinism = seed_run(0)
    if not torch.cuda.is_available():
        raise ValueError("This recovery export requires the calibrated CUDA backend")
    started = time.monotonic()
    original, old_status, old_provenance, checkpoints, configs, histories, origins = _load_original(original_root)
    original_files = _snapshot_original(original)
    frozen_code = _verify_frozen_code(old_provenance)
    profile, calibration, profile_path, calibration_sha256 = _validate_calibration(profile_path, configs, device)
    if (old_provenance.get("profile_sha256") != calibration_sha256
            or old_provenance.get("execution_profile") != profile):
        raise ValueError("Saved calibration differs from the original training profile identity")
    if torch.cuda.get_device_name() != calibration.get("gpu"):
        raise ValueError("Current GPU differs from the saved checkpoint calibration")
    reference_config = configs["norm"]
    for variant, config in configs.items():
        if config.get("profile") != profile or checkpoints[variant].get("normalization") != checkpoints["norm"].get("normalization"):
            raise ValueError(f"{variant}: source normalization or resource profile differs from checkpoint")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    group_status = {"status": "incomplete", "kind": "checkpoint-evaluation", "execution": "checkpoint-evaluation",
                    "origin_execution": "shared-g-dditr",
                    "evaluation_only": True, "training_state": "interrupted", "training_completed_epochs": 23,
                    "held_out_tests": False, "variants": list(BEST_EPOCHS), "failures": []}
    write_json(output_root / "status.json", group_status)
    run_handles, finished_runs = {}, set()
    prepared = None
    prepared_closed = False
    try:
        prepared = _prepare_validation(outputs_root, configs, profile)
        if prepared["normalization"] != checkpoints["norm"]["normalization"]:
            raise ValueError("Current LOKI validation normalization differs from the selected checkpoints")
        if _source_reference(old_provenance["datasets"][SOURCE]) != _source_reference(prepared["datasets"][SOURCE]):
            raise ValueError("Frozen LOKI setup/collection/bbox source identities differ from the original run")
        input_identity = {
            "frozen_loki_source": _safe_source_identity(prepared["datasets"][SOURCE]),
            "feature_population": prepared["catalog"].identity,
            "indexed_point_cache": prepared["points"].identity,
            "point_storage_protocol": POINT_PROTOCOL,
            "normalization": prepared["normalization"],
            "source_files_sha256": prepared["source_files_sha256"],
        }
        if old_provenance.get("source_statistics") != prepared["normalization"]:
            raise ValueError("LOKI source statistics differ from the original training record")
        for variant, config in configs.items():
            if config.get("indexed_point_cache") != prepared["points"].identity:
                raise ValueError(f"{variant}: checkpoint/indexed-point cache identity differs")
        packing = prepared["packing"]
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        estimated = calibration.get("memory_preflight", {}).get("estimated_group_peak_bytes", 0)
        if free_bytes < max(6 * 2**30, estimated):
            raise ValueError("Free VRAM is below the calibrated shared-LiDAR headroom gate")
        provenance = {
            "kind": "checkpoint-evaluation", "execution": "shared-validation-prediction",
            "training_state": "interrupted", "training_completed_epochs": 23,
            "selected_epochs": dict(BEST_EPOCHS), "original_training_group": str(original),
            "original_group_status": old_status, "original_group_provenance_sha256": checksum(original / "provenance.json"),
            "calibration_profile_sha256": calibration_sha256,
            "calibration_profile_code_identity": calibration.get("code"),
            "calibration_identity_note": "Saved profile used only for checkpoint/backend and bounded-input validation; changed training code hashes are not asserted as current training calibration.",
            "inference_compatibility_code_sha256": frozen_code,
            "current_inference_code_sha256": _inference_code_identity(),
            "backend": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(), "free_vram_before_bytes": free_bytes,
                        "total_vram_bytes": total_bytes},
            "source": SOURCE, "input_identity": input_identity, "packing": packing,
            "cpu_threads": {"torch": torch.get_num_threads(),
                "environment": {key: os.environ.get(key)
                    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
            "python": platform.python_version(), "dependency_lock_sha256": checksum("uv.lock"),
            "host": platform.node(), "source_settings": reference_config["settings"],
            "profile": profile, "determinism": determinism,
            "original_training_files_sha256": original_files,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "original_training_code_sha256": old_provenance.get("code_sha256", {}),
        }
        write_json(output_root / "provenance.json", provenance)
        models = {}
        for variant in BEST_EPOCHS:
            model = e002b.build_model(ARM, variant, profile).to(device)
            model.load_state_dict(checkpoints[variant]["model_state"], strict=True)
            model.eval()
            models[variant] = model
        torch.cuda.reset_peak_memory_stats()
        logical = LogicalBatches(prepared["dataset"], profile, training=False, seed=0,
                                 prefetch=True)
        tracks = predict_shared(models, PredictionBatches(logical), device)
        if any(len(tracks[variant]) != EXPECTED_VALIDATION_TRACKS for variant in BEST_EPOCHS):
            raise ValueError("Evaluation did not produce exactly 1,861 complete validation tracks per checkpoint")
        evaluations = {}
        metadata_cache = {}
        for variant in BEST_EPOCHS:
            _decorate(tracks[variant], prepared, metadata_cache, SOURCE)
            metrics = prediction_metrics(tracks[variant], len(CLASSES))
            evaluations[variant] = {SOURCE: {"split": "validation", "primary": metrics,
                "strata": stratified_metrics(tracks[variant], len(CLASSES), AXES)}}
        torch.cuda.synchronize()
        provenance["runtime"] = {"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "prepared_validation_seconds": logical.stats["prepared_seconds"],
            "point_cache": prepared["points"].snapshot(), "encoder_loads": {}, "encoder_calls": {}}
        prepared["points"].close()
        prepared_closed = True
        metric_records, plot_records, provenance = _write_local_results(
            output_root, configs, checkpoints, histories, origins, tracks, evaluations, provenance)
        metric_failures = []
        for variant in BEST_EPOCHS:
            result = evaluations[variant][SOURCE]
            metrics = result["primary"]
            try:
                verification = verify_checkpoint_metrics(_conditions_from_checkpoint(checkpoints[variant]), metrics)
                selected = checkpoints[variant]["selection"]
                if (not math.isclose(metrics["track_ce"], selected["ce"], rel_tol=METRIC_RTOL, abs_tol=METRIC_ATOL)
                        or not math.isclose(metrics["track_weighted"]["macro_f1"], selected["f1"],
                                            rel_tol=METRIC_RTOL, abs_tol=METRIC_ATOL)):
                    raise ValueError("Recomputed CE/F1 differs from the selected checkpoint record")
                verification["matches_selected_checkpoint"] = True
            except ValueError as error:
                verification = {"matches_checkpoint": False, "rtol": METRIC_RTOL, "atol": METRIC_ATOL,
                                "error": str(error)}
                metric_failures.append({"variant": variant, "error": str(error)})
            result["checkpoint_verification"] = verification
            metric_records[variant]["checkpoint_verification"] = verification
            write_json(output_root / variant / f"{SOURCE}-validation" / "metrics.json", result)
        provenance["metric_verification"] = metric_records
        provenance["local_evaluation_verified"] = not metric_failures
        write_json(output_root / "provenance.json", provenance)
        for variant in BEST_EPOCHS:
            child_provenance = json.loads((output_root / variant / "provenance.json").read_text())
            child_provenance["metric_verification"] = metric_records[variant]
            child_provenance["local_evaluation_verified"] = not metric_failures
            write_json(output_root / variant / "provenance.json", child_provenance)
        if metric_failures:
            write_json(output_root / "evaluation-evidence.json", {"status": "checkpoint-metric-mismatch",
                "kind": "checkpoint-evaluation", "training_state": "interrupted", "source": SOURCE,
                "training_completed_epochs": 23, "selected_epochs": dict(BEST_EPOCHS),
                "verification": metric_records, "wandb_uploads": "not-started"})
            raise ValueError(f"Checkpoint metric equivalence failed: {metric_failures}")
        write_json(output_root / "evaluation-evidence.json", {"status": "local-evaluation-verified",
            "kind": "checkpoint-evaluation", "training_state": "interrupted", "source": SOURCE,
            "training_completed_epochs": 23, "selected_epochs": dict(BEST_EPOCHS),
            "verification": metric_records, "wandb_uploads": "pending"})
        group_status["phase"] = "local-evaluation-verified"
        write_json(output_root / "status.json", group_status)
        ceiling = calibration.get("allocated_ceiling_bytes")
        peak = provenance["runtime"]["cuda_peak_allocated_bytes"]
        if ceiling is not None and peak > ceiling:
            raise ValueError(f"Evaluation peak allocation {peak} exceeds the calibrated ceiling {ceiling}")
        changed_inputs = [path for path, digest in prepared["source_files_sha256"].items()
                          if checksum(path) != digest]
        changed_original = [path for path, digest in original_files.items() if checksum(path) != digest]
        if changed_inputs or changed_original:
            raise ValueError(f"Evaluation inputs changed; source={changed_inputs}, original={changed_original}")

        group_id = output_root.name
        for variant in BEST_EPOCHS:
            directory = output_root / variant
            original_run = origins[variant]
            config = json.loads((directory / "config.json").read_text()) | {
                "evaluation_only": True, "training_state": "interrupted",
                "training_completed_epochs": 23, "origin_training_run": original_run,
                "checkpoint_sha256": original_run["best_checkpoint_sha256"],
            }
            evaluation_run = wandb.init(entity="yammo-unipd", project="pedestrian-behaviour-labeling",
                name=f"e002b-loki-g-dditr-{variant}-checkpoint-eval-e{BEST_EPOCHS[variant]}-s0",
                job_type="checkpoint-evaluation", tags=["evaluation-only", "interrupted-training"],
                group=group_id, config=config | {"provenance": provenance},
                dir=str(directory.resolve()), save_code=False, resume="never",
                reinit="create_new", mode="online")
            run_handles[variant] = evaluation_run
            configure_logging(evaluation_run)
            provenance.setdefault("evaluation_wandb", {})[variant] = {"id": evaluation_run.id,
                                                                         "url": evaluation_run.url}
            write_json(directory / "provenance.json", provenance | {"model_variant": variant,
                "origin_training_run": original_run, "evaluation_wandb": provenance["evaluation_wandb"]})
            write_json(output_root / "provenance.json", provenance)
        write_json(output_root / "provenance.json", provenance)
        for variant in BEST_EPOCHS:
            log_final(run_handles[variant], evaluations[variant], plot_records[variant],
                      provenance | {"model_variant": variant,
                          "origin_training_run": origins[variant],
                          "evaluation_wandb": provenance["evaluation_wandb"]}, CLASSES, AXES)
            run_handles[variant].finish()
            finished_runs.add(variant)
        provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
        provenance["elapsed_seconds"] = time.monotonic() - started
        changed_inputs = [path for path, digest in prepared["source_files_sha256"].items()
                          if checksum(path) != digest]
        changed_original = [path for path, digest in original_files.items() if checksum(path) != digest]
        if changed_inputs or changed_original:
            raise ValueError(f"Evaluation inputs changed before completion; source={changed_inputs}, original={changed_original}")
        write_json(output_root / "provenance.json", provenance)
        for variant in BEST_EPOCHS:
            write_json(output_root / variant / "provenance.json", provenance | {
                "model_variant": variant, "origin_training_run": origins[variant],
                "selection": checkpoints[variant]["selection"],
                "selected_checkpoint_epoch": BEST_EPOCHS[variant],
                "checkpoint_sha256": origins[variant]["best_checkpoint_sha256"],
                "evaluation_wandb": provenance["evaluation_wandb"],
                "validation_metrics": metric_records[variant]["primary"]})
            write_json(output_root / variant / "status.json", {"status": "complete",
                "kind": "checkpoint-evaluation", "execution": "checkpoint-evaluation", "evaluation_only": True,
                "training_state": "interrupted", "training_completed_epochs": 23,
                "selected_checkpoint_epoch": BEST_EPOCHS[variant], "failures": []})
        group_status.update(status="complete", phase="complete", completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(output_root / "status.json", group_status)
        write_json(output_root / "evaluation-evidence.json", {"status": "complete",
            "kind": "checkpoint-evaluation", "training_state": "interrupted", "source": SOURCE,
            "training_completed_epochs": 23, "selected_epochs": dict(BEST_EPOCHS),
            "verification": metric_records, "wandb_runs": provenance.get("evaluation_wandb", {})})
        return {"output": str(output_root), "metrics": metric_records,
                "wandb": provenance.get("evaluation_wandb", {})}
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        group_status.update(status="failed", phase="failed", failures=[failure])
        evidence_path = output_root / "evaluation-evidence.json"
        if evidence_path.is_file():
            evidence = json.loads(evidence_path.read_text())
            evidence.update(status="wandb-failed" if run_handles else "evaluation-failed",
                failure={"type": type(error).__name__, "message": str(error)},
                wandb_runs={variant: {"id": run.id, "url": run.url,
                    "status": "finished" if variant in finished_runs else "failed-or-open"}
                    for variant, run in run_handles.items()},
                wandb_uploads="partial-failed" if run_handles else "not-started")
            write_json(evidence_path, evidence)
        for variant in BEST_EPOCHS:
            child = output_root / variant
            if child.is_dir():
                for key, run in run_handles.items():
                    if key == variant and key not in finished_runs:
                        try:
                            run.finish(exit_code=1)
                        except Exception:
                            pass
                write_json(child / "status.json", {"status": "failed", "kind": "checkpoint-evaluation",
                    "execution": "checkpoint-evaluation", "evaluation_only": True,
                    "training_state": "interrupted", "failures": [failure]})
        write_json(output_root / "status.json", group_status)
        raise
    finally:
        if prepared is not None and not prepared_closed:
            prepared["points"].close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--original-run", type=Path, default=Path("outputs/experiments/E002b/runs") / EXPECTED_ORIGINAL)
    parser.add_argument("--profile", type=Path,
                        default=Path("outputs/experiments/E002b/calibration-lidar-group-extractor/profile.json"))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/experiments/E002b/runs") / f"checkpoint-eval-loki-g-dditr-{timestamp}")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    result = run_evaluation(outputs_root=args.outputs_root, original_root=args.original_run,
                            profile_path=args.profile, output_root=args.output, device=args.device)
    print(json.dumps(result, allow_nan=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
