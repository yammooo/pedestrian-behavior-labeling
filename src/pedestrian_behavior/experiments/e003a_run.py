"""E003a sequential source-validation training and bounded diagnostics."""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import torch

from pedestrian_behavior.data.indexed_points import PROTOCOL as POINT_PROTOCOL
from pedestrian_behavior.data.preparation import checksum, load_track, source_times
from pedestrian_behavior.data.prepared_points import PreparedPoints
from pedestrian_behavior.evaluation import (load_predictions, predict, prediction_metrics,
                                            save_predictions, stratified_metrics)
from pedestrian_behavior.experiments import e001b, e001b_run, e002a_run, e002b, e002b_run, run as shared_run
from pedestrian_behavior.experiments.e002a_features import FeatureCatalog, extraction_identity
from pedestrian_behavior.experiments.e003a import BRANCHES, FeatureDataset, MaskSampler
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.inspection.pretrained import DITR_CONFIG_SHA256, DITR_REVISION, DITR_SHA256


ARM = "g-rgb-dditr"
VARIANT = "temporal-small"
RGB_DROPOUT = .5
SETTINGS = e002a_run.SETTINGS | {"seed": 0}
SOURCES = ("road-waymo", "loki")
CONTROL = {
    "road-waymo": {"name": "fusion-20261030-140937-road-waymo-rgbdrop50",
                    "sha256": "d1f83183dcff61f8d9925a74c2c76811d1eca225119e588df8aa3770720e42a8",
                    "epoch": 17},
    "loki": {"name": "fusion-20261030-140937-loki-rgbdrop50",
             "sha256": "ee097048df9e98367ca1f355632a32fa8ad389cf32796d24bdf0c7d871be4e17",
             "epoch": 12},
}


def resource_code():
    from pedestrian_behavior import evaluation, training
    from pedestrian_behavior.data import indexed_points, multimodal, prepared_points, prepared_rgb
    from pedestrian_behavior.experiments import (e002a_features, e002a_run, e002b_run, e003a,
                                                 run as shared_run_module)
    from pedestrian_behavior.models import multimodal as multimodal_model
    modules = (multimodal_model, multimodal, prepared_points, prepared_rgb, indexed_points,
               evaluation, training, shared_run_module, e002a_features, e002a_run, e002b, e002b_run,
               e003a, __import__(__name__, fromlist=["assembly"]))
    return {module.__name__: checksum(module.__file__) for module in modules}


def model_spec():
    return e002b.model_spec(ARM, VARIANT, dropout=.1, rgb_branch_dropout=RGB_DROPOUT)


def calibration_code():
    from pedestrian_behavior.experiments import e002a_calibrate, e002b_calibrate
    paths = (Path(__file__).with_name("e003a_calibrate.py"), Path(e002a_calibrate.__file__),
             Path(e002b_calibrate.__file__))
    return {str(path): checksum(path) for path in paths}


def validate_profile(args):
    record = json.loads(args.profile.read_text())
    key = f"{ARM}/{VARIANT}"
    if (record.get("status") != "complete" or record.get("experiment") != "E003a"
            or record.get("device") != args.device or record.get("code") != resource_code()
            or record.get("calibration_code_sha256") != calibration_code()
            or record.get("model_specs", {}).get(key) != model_spec()
            or key not in record.get("model_coverage", ())
            or record.get("point_storage_protocol") != POINT_PROTOCOL):
        raise ValueError("Completed E003a calibration for this exact model/code is required")
    if record.get("torch") != str(torch.__version__) or record.get("cuda") != torch.version.cuda:
        raise ValueError("PyTorch/CUDA backend differs from E003a calibration")
    if args.device == "cuda" and (not torch.cuda.is_available() or record.get("gpu") != torch.cuda.get_device_name()):
        raise ValueError("E003a calibration hardware differs from the requested CUDA device")
    profile = e002a_run.validate_profile(record.get("profile", {}))
    if profile["max_tracks"] != 64:
        raise ValueError("E003a retains an effective 64-track logical batch")
    return record


def _validate_roots(args, config):
    output = args.output.resolve()
    root = args.outputs_root.resolve()
    if not output.is_relative_to(root / "experiments/E003a"):
        raise ValueError("E003a evidence must be under outputs-root/experiments/E003a")
    opposite = "loki" if args.source == "road-waymo" else "road-waymo"
    target_points = args.opposite_indexed_point_cache or (root / "experiments/E002a" / f"indexed-{opposite}")
    caches = [args.rgb_cache_root.resolve(), e002a_run.indexed_path(args).resolve(),
              Path(target_points).resolve(), output]
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(caches) for b in caches[i + 1:]):
        raise ValueError("E003a evidence and frozen feature caches must have separate roots")
    protected = [root / "experiments" / name for name in ("E001", "E001b", "E001c")]
    protected.append(args.control_run.resolve())
    protected.extend(Path(config[key]).resolve() for key in ("loki_root", "waymo_root", "road_index") if config.get(key))
    for location in caches:
        if any(location.is_relative_to(path) or path.is_relative_to(location) for path in protected):
            raise ValueError("E003a evidence must not overlap frozen/native data")


def _profile_for_assembly(args):
    profile = validate_profile(args)
    source_key = f"{ARM}/{VARIANT}"
    # Reuse the frozen E002d model dimensions, but require the separately measured E003a code profile.
    if profile.get("model_specs", {}).get(source_key) != model_spec():
        raise ValueError("E003a profile model specification mismatch")
    return profile


def _validate_rgb_preparation(path, source, catalog, identity, rgb_root):
    if path is None:
        raise ValueError(f"RGB-enabled E003a validation requires the {source} RGB preparation record")
    record = json.loads(path.read_text())
    expected = set(catalog.tracks)
    actual = {row["locator"] for row in record.get("tracks", []) if row.get("dataset") == source}
    if (record.get("status") != "complete" or record.get("extraction") != identity or expected != actual
            or Path(record["config"]["rgb_cache_root"]).resolve() != rgb_root.resolve()):
        raise ValueError(f"Matching full {source} train/validation RGB cache preparation required")


def _smoke_locators(path, source):
    if path is None:
        return None
    entries = json.loads(path.read_text()).get("tracks", [])
    return {row["locator"] for row in entries if row.get("dataset") == source and row.get("split") == "validation"}


def _validate_control(control_root, source, normalization, expected_locators):
    control_root = Path(control_root).resolve()
    expected = CONTROL[source]
    if control_root.name != expected["name"] or control_root.parent.name != "runs":
        raise ValueError("Only the named E002d RGB-dropout0.5 source control is accepted")
    config = json.loads((control_root / "config.json").read_text())
    status = json.loads((control_root / "status.json").read_text())
    model_path = control_root / "best.pt"
    if (checksum(model_path) != expected["sha256"] or status.get("status") != "complete"
            or status.get("failures") or config.get("experiment") != "E002d"
            or config.get("source") != source or config.get("configuration", config.get("arm")) != ARM
            or config.get("variant", config.get("model_variant")) != VARIANT
            or config.get("settings", {}).get("seed") != 0 or config.get("held_out_tests") is not False
            or config.get("model") != model_spec()):
        raise ValueError("E002d control run/checkpoint identity mismatch")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    if (checkpoint.get("epoch") != expected["epoch"] or checkpoint.get("selection", {}).get("epoch") != expected["epoch"]
            or checkpoint.get("config", {}).get("model") != model_spec()
            or checkpoint.get("normalization") != normalization):
        raise ValueError("E002d selected checkpoint or source-normalization identity mismatch")
    metrics_path = control_root / f"{source}-validation" / "metrics.json"
    prediction_path = control_root / f"{source}-validation" / "predictions.npz"
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("split") != "validation":
        raise ValueError("E002d control validation artifact is not source validation")
    tracks = load_predictions(prediction_path)
    if {track["locator"] for track in tracks} != expected_locators:
        raise ValueError("E002d control predictions do not cover the frozen source validation population")
    recomputed = prediction_metrics(tracks, len(e002a_run.CLASSES))
    _same_metrics(metrics["primary"], recomputed)
    selection = checkpoint["selection"]
    if (not math.isclose(selection["f1"], recomputed["track_weighted"]["macro_f1"], rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(selection["ce"], recomputed["track_ce"], rel_tol=1e-6, abs_tol=1e-8)):
        raise ValueError("E002d control source-validation metrics disagree with the selected checkpoint")
    return checkpoint, {"path": str(model_path), "sha256": expected["sha256"],
                        "selected_epoch": expected["epoch"], "validation_metrics": metrics["primary"],
                        "validation_predictions": str(prediction_path),
                        "validation_predictions_sha256": checksum(prediction_path)}


def _control_transfer_result(path, source, opposite, control, target_locators):
    path = Path(path).resolve()
    manifest_path, results_path = path / "transfer-comparison.json", path / "results.json"
    status = json.loads((path / "status.json").read_text())
    manifest, results = json.loads(manifest_path.read_text()), json.loads(results_path.read_text())
    if (status.get("status") != "complete" or manifest.get("status") != "frozen-before-cross-dataset-validation-inference"
            or manifest.get("evaluation", {}).get("held_out_tests_accessed") is not False):
        raise ValueError("Completed frozen E002 validation-transfer evidence required")
    rows = [row for row in manifest.get("models", []) if row.get("training_source") == source
            and row.get("input") == "geometry+rgb+lidar"]
    target_rows = [row for row in results if row.get("training_source") == source
                   and row.get("target_source") == opposite and row.get("input") == "geometry+rgb+lidar"]
    if (len(rows) != 1 or len(target_rows) != 1 or rows[0].get("checkpoint_sha256") != control["sha256"]
            or target_rows[0].get("checkpoint_sha256") != control["sha256"]):
        raise ValueError("Frozen E002 transfer result does not match the canonical E002d control checkpoint")
    predictions_path = path / "cells" / source / "geometry+rgb+lidar" / "predictions.npz"
    predictions = load_predictions(predictions_path)
    if {track["locator"] for track in predictions} != set(target_locators):
        raise ValueError("Frozen E002d opposite-validation predictions do not cover the target population")
    metrics = prediction_metrics(predictions, len(e002a_run.CLASSES))
    if (len(predictions) != target_rows[0]["target_tracks"]
            or not math.isclose(metrics["track_weighted"]["macro_f1"], target_rows[0]["target_f1"], rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(metrics["track_ce"], target_rows[0]["target_ce"], rel_tol=1e-6, abs_tol=1e-8)):
        raise ValueError("Frozen E002d opposite-validation metrics disagree with local predictions")
    source_metrics = control["validation_metrics"]
    if (not math.isclose(rows[0]["source_validation"]["track_weighted_macro_f1"],
                         source_metrics["track_weighted"]["macro_f1"], rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(rows[0]["source_validation"]["track_ce"],
                                source_metrics["track_ce"], rel_tol=1e-6, abs_tol=1e-8)):
        raise ValueError("Frozen E002 transfer manifest source metrics differ from the canonical control run")
    return {"path": str(path), "manifest_sha256": checksum(manifest_path),
            "results_sha256": checksum(results_path), "target_predictions": str(predictions_path),
            "target_predictions_sha256": checksum(predictions_path), "target_support": metrics["denominators"],
            "target": target_rows[0]}


def _same_metrics(expected, actual, path="metrics"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise ValueError(f"Control metric structure differs at {path}")
        for key in expected:
            _same_metrics(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise ValueError(f"Control metric structure differs at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            _same_metrics(left, right, f"{path}[{index}]")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"Control metric differs at {path}")
    elif expected != actual:
        raise ValueError(f"Control metric differs at {path}")


def _decorate(tracks, conditions, dataset):
    for track in tracks:
        metadata, arrays = load_track(track["archive"])
        identity = json.loads(track["locator"])
        track["group"] = identity.get("clip", identity.get("scenario"))
        track["conditions"], flags = conditions(metadata, arrays, track, dataset)
        track["source_flags"] = track.get("source_flags", {}) | flags
        track["source_times_us"] = source_times(metadata)
        track["native_extent_us"] = metadata["native_extent_us"]
        track["grid_times_us"] = (metadata["native_extent_us"][0] + np.arange(len(track["targets"])) * 200000).tolist()
    return tracks


def _diagnostic_result(tracks, classes):
    noinput = [not bool(np.asarray(track["source_flags"]["effective_geometry_valid"]
                   | track["source_flags"].get("effective_rgb_valid", False)
                   | track["source_flags"].get("effective_lidar_valid", False)).any()) for track in tracks]
    return {"tracks": len(tracks), "no_input_tracks": int(sum(noinput)),
            "primary": prediction_metrics(tracks, classes),
            "no_input": prediction_metrics(tracks, classes,
                [np.full(len(track["targets"]), empty, dtype=bool) for track, empty in zip(tracks, noinput)])}


def _write_diagnostic(directory, tracks, result):
    directory.mkdir(parents=True, exist_ok=False)
    save_predictions(directory / "predictions.npz", tracks)
    write_json(directory / "metrics.json", result)


def assembly(args):
    args.arm, args.model_variant = ARM, VARIANT
    args.seed, args.dropout, args.rgb_branch_dropout, args.lidar_branch_dropout = 0, .1, RGB_DROPOUT, 0.
    profile_record = _profile_for_assembly(args)
    _validate_roots(args, {"loki_root": None, "waymo_root": None, "road_index": None})
    spec = model_spec()
    policy = {"experiment": "E003a", "resource_code": resource_code, "validate_roots": _validate_roots,
              "model_spec": lambda *_: spec,
              "model_factory": lambda arm, variant, profile: e002b.build_model(
                  arm, variant, profile, dropout=.1, rgb_branch_dropout=RGB_DROPOUT),
              "run_name": f"e003a-{args.source}-g-rgb-dditr-temporal-small-s0" +
                  ("-smoke-" + args.output.name if args.smoke else "")}
    prepared = e002a_run.assembly(args, policy=policy)
    prepared["config"].update({"experiment": "E003a", "reference": "E002d-g-rgb-dditr-temporal-small-rgbdrop0.5",
        "masking": {"clean_probability": .5, "full_track_probability": .25,
            "block_lengths": list((1, 5, 10, 20)), "block_length_rule": "strictly-less-than-track-length; T=1 full",
            "branches": list(BRANCHES), "retained_branch_count": {"values": [0, 1, 2, 3], "probabilities": [.1, .3, .3, .3]},
            "maximum_candidates": 3, "minimum_retained_observable_slots": .25,
            "maximum_newly_hidden_observable_slots": "min(10,floor(T/4))", "mask_seed": 0},
        "control": None, "opposite_validation": None})
    prepared["config"]["control"] = {"path": str(args.control_run.resolve()), "sha256": CONTROL[args.source]["sha256"]}
    profile = profile_record["profile"]
    source = args.source
    opposite = "loki" if source == "road-waymo" else "road-waymo"
    base_source = prepared["dataset"](source, "training")
    source_validation_base = prepared["dataset"](source, "validation")
    source_features = base_source
    datasets = {}
    training_data = FeatureDataset(base_source, source, ARM, mode="training", sampler=MaskSampler(0), seed=0)
    source_validation = FeatureDataset(source_validation_base, source, ARM, mode="natural")
    datasets[(source, "training")] = training_data
    datasets[(source, "validation")] = source_validation

    source_catalog = source_features.supply.catalogs[source]
    source_val_ids = {locator for locator, entry in source_catalog.tracks.items()
                      if entry["split"] == "validation"}
    control_checkpoint, control_identity = _validate_control(args.control_run, source, prepared["normalization"], source_val_ids)
    prepared["config"]["control"] = control_identity
    control_transfer_path = getattr(args, "control_transfer_results", None)
    target_points, target_protected = None, {}
    try:
        target_config, target_transform = e002a_run.load_configuration(args.probe)
        target_catalog = FeatureCatalog(args.outputs_root, opposite, target_config, target_transform)
        rgb_identity = extraction_identity(args.device, target_config)
        rgb_cache = source_features.supply.rgb_supply.rgb_cache
        _validate_rgb_preparation(args.opposite_rgb_preparation, opposite, target_catalog, rgb_identity, args.rgb_cache_root)
        target_point_root = args.opposite_indexed_point_cache or (args.outputs_root / "experiments/E002a" / f"indexed-{opposite}")
        target_points = PreparedPoints(target_point_root, opposite, target_catalog.identity, target_catalog.tracks,
            {"revision": DITR_REVISION, "checkpoint": DITR_SHA256, "configuration": DITR_CONFIG_SHA256},
            allow_partial=args.smoke, frame_workers=args.point_workers)
        target_protected.update(target_points.protected)
        target_supply = e002a_run.PreparedSupply(target_catalog, rgb_cache, target_points, rgb_identity)
        target_setup = args.outputs_root / "experiments/E001/data-setup" / opposite / "setup.json"
        target_collection = args.outputs_root / "experiments/E001/reader-preparation" / opposite
        target_extension = args.outputs_root / "experiments/E001b/bbox-native" / opposite
        target_protected.update(target_catalog.protected)
        target_protected[str((target_extension / "manifest.json").resolve())] = checksum(target_extension / "manifest.json")
        source_statistics = prepared["normalization"]
        target_baseline = e001b.dataset_from_extension(target_collection, target_setup, target_extension,
            "validation", "geometry", source_statistics["baseline"], source_statistics["bbox"])
        for entry in target_baseline.entries:
            if entry["slots"] ** 2 > profile["temporal_cells"]:
                raise ValueError("Opposite-source validation track exceeds calibrated temporal ceiling")
            target_points.require_track(entry["locator"])
            if target_points.counts[entry["locator"]].max(initial=0) > profile["frame_tokens"]:
                raise ValueError("Opposite-source point frame exceeds calibrated token ceiling")
            path = target_extension / "tracks" / entry["archive"]
            target_protected[str(path.resolve())] = entry["bbox_sha256"]
        if args.smoke:
            selected = _smoke_locators(args.smoke_selection, opposite)
            if not selected or len(selected) > 10:
                raise ValueError("Smoke selection must include at most 10 opposite-source validation tracks")
            if not selected.issubset({entry["locator"] for entry in target_baseline.entries}):
                raise ValueError("Smoke selection contains a non-validation opposite-source track")
            target_baseline.entries = [entry for entry in target_baseline.entries if entry["locator"] in selected]
        target_base = e002a_run.FeatureDataset(target_baseline, target_supply, opposite, ARM)
        target_validation = FeatureDataset(target_base, opposite, ARM, mode="natural")
        datasets[(opposite, "validation")] = target_validation
        target_val_ids = {entry["locator"] for entry in target_baseline.entries}
        if args.smoke and len(target_val_ids) == 0:
            raise ValueError("Smoke selection leaves no opposite-source validation track")
        control_transfer = None if args.smoke else _control_transfer_result(
            control_transfer_path or (args.outputs_root / "experiments/E002/transfer-validation-20261102"),
            source, opposite, control_identity, target_val_ids)
        prepared["config"]["control_transfer"] = control_transfer
        prepared["config"]["opposite_validation"] = {"source": opposite, "normalization_source": source,
            "tracks": len(target_val_ids), "split": "validation-smoke" if args.smoke else "validation"}
        prepared["config"]["opposite_feature_identity"] = target_supply.identity
        prepared["config"]["opposite_rgb_preparation"] = {
            "path": str(args.opposite_rgb_preparation.resolve()), "sha256": checksum(args.opposite_rgb_preparation)}
        prepared["config"]["opposite_point_cache"] = target_points.identity
        prepared["datasets"][opposite] = {"setup": target_catalog.identity,
            "collection_manifest_sha256": target_catalog.identity.get("manifest_sha256"),
            "validation_tracks": len(target_val_ids), "normalization_source": source,
            "bbox_extension_manifest_sha256": target_protected[str((target_extension / "manifest.json").resolve())],
            "indexed_point_cache": target_points.identity}
        target_owner_conditions = e001b_run.assemble(opposite, "geometry", "B",
            e001_root=args.outputs_root / "experiments/E001",
            e001b_root=args.outputs_root / "experiments/E001b")["conditions"]
        source_owner_conditions = prepared["conditions"]
        def target_conditions(metadata, arrays, track, name):
            conditions, flags = target_owner_conditions(metadata, arrays, track, name)
            source_flags = track.get("source_flags", {})
            observed = {axis: np.asarray(source_flags[flag], dtype=np.int64)
                        for axis, flag in (("rgb-observed", "rgb_valid"), ("lidar-observed", "lidar_valid"))
                        if flag in source_flags}
            return conditions | observed, flags
        prepared["conditions"] = lambda metadata, arrays, track, name: (
            source_owner_conditions if name == source else target_conditions)(metadata, arrays, track, name)
    except BaseException:
        try:
            if target_points is not None:
                target_points.close()
        finally:
            prepared["close"]()
        raise

    def dataset(name, split):
        try:
            return datasets[(name, split.removesuffix("-smoke"))]
        except KeyError as error:
            raise ValueError(f"E003a has no {name}/{split} dataset") from error

    def batches(data, training=False, seed=0):
        logical = base_batches(data, training=training, seed=seed)
        if training:
            data.epoch = lambda: logical.iteration
        return logical

    underlying_epoch = prepared["epoch_operation"]
    def epoch_operation(model, logical, optimizer, device, clip_norm, on_step=None, step_offset=0):
        result = underlying_epoch(model, logical, optimizer, device, clip_norm, on_step=on_step, step_offset=step_offset)
        summary, records = logical.dataset.finish_epoch()
        directory = args.output / "masks"
        directory.mkdir(exist_ok=True)
        epoch = logical.iteration
        write_json(directory / f"epoch-{epoch:03d}-summary.json", summary)
        with gzip.open(directory / f"epoch-{epoch:03d}-tracks.jsonl.gz", "wt", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        return result | {"masking": summary}

    def evaluation_predictions(model, name, split, device):
        return predict(model, batches(dataset(name, split), seed=SETTINGS["seed"]), device, prepared["forward_batch"])

    def post_evaluation(model, evaluations, fitted, output, device, wandb_run):
        diagnostic_root = output / "diagnostics" / source
        diagnostic_root.mkdir(parents=True, exist_ok=False)
        diagnostics = {}
        for branch in BRANCHES:
            view = branch + "-only"
            data = FeatureDataset(source_validation_base, source, ARM, mode="branch", branch=branch)
            tracks = predict(model, batches(data, seed=0), device, prepared["forward_batch"])
            tracks = _decorate(tracks, prepared["conditions"], source)
            result = _diagnostic_result(tracks, len(e002a_run.CLASSES))
            result["split"] = "validation-smoke" if args.smoke else "validation"
            destination = diagnostic_root / view / "e003a"
            _write_diagnostic(destination, tracks, result)
            diagnostics[view] = {"e003a": result}

        fixed_masks = {}
        fixed_data = FeatureDataset(source_validation_base, source, ARM, mode="fixedmixed",
            sampler=MaskSampler(10000), seed=10000, fixed_masks=fixed_masks)
        new_tracks = predict(model, batches(fixed_data, seed=10000), device, prepared["forward_batch"])
        new_tracks = _decorate(new_tracks, prepared["conditions"], source)
        result = _diagnostic_result(new_tracks, len(e002a_run.CLASSES))
        result["split"] = "validation-smoke" if args.smoke else "validation"
        _write_diagnostic(diagnostic_root / "fixedmixed" / "e003a", new_tracks, result)
        diagnostics["fixedmixed"] = {"e003a": result}

        control = e002b.build_model(ARM, VARIANT, profile,
            dropout=.1, rgb_branch_dropout=RGB_DROPOUT).to(device)
        control.load_state_dict(control_checkpoint["model_state"], strict=True)
        for branch in BRANCHES:
            view = branch + "-only"
            data = FeatureDataset(source_validation_base, source, ARM, mode="branch", branch=branch)
            tracks = predict(control, batches(data, seed=0), device, prepared["forward_batch"])
            tracks = _decorate(tracks, prepared["conditions"], source)
            result = _diagnostic_result(tracks, len(e002a_run.CLASSES))
            result["split"] = "validation-smoke" if args.smoke else "validation"
            _write_diagnostic(diagnostic_root / view / "e002d-control", tracks, result)
            diagnostics[view]["e002d-control"] = result
        control_tracks = predict(control, batches(fixed_data, seed=10000), device, prepared["forward_batch"])
        control_tracks = _decorate(control_tracks, prepared["conditions"], source)
        if len(control_tracks) != len(new_tracks):
            raise ValueError("Fixedmixed new/control diagnostic populations differ")
        for left, right in zip(new_tracks, control_tracks):
            for name in ("effective_geometry_valid", "effective_rgb_valid", "effective_lidar_valid"):
                if not np.array_equal(left["source_flags"][name], right["source_flags"][name]):
                    raise ValueError("Fixedmixed control/new models received different masks")
        result = _diagnostic_result(control_tracks, len(e002a_run.CLASSES))
        result["split"] = "validation-smoke" if args.smoke else "validation"
        _write_diagnostic(diagnostic_root / "fixedmixed" / "e002d-control", control_tracks, result)
        diagnostics["fixedmixed"]["e002d-control"] = result
        write_json(diagnostic_root / "summary.json", diagnostics)
        comparison_table = None
        if control_transfer is not None:
            transfer_root = Path(control_transfer["path"])
            if (checksum(transfer_root / "transfer-comparison.json") != control_transfer["manifest_sha256"]
                    or checksum(transfer_root / "results.json") != control_transfer["results_sha256"]):
                raise ValueError("Frozen E002 transfer comparison changed during E003a")
            target_reference = control_transfer["target"]
            if evaluations[opposite]["primary"]["denominators"]["tracks"] != target_reference["target_tracks"]:
                raise ValueError("E003a opposite-validation population differs from frozen E002 transfer results")
            references = {source: (control_identity["validation_predictions"],
                                   control_identity["validation_predictions_sha256"],
                                   control_identity["validation_metrics"]["denominators"]),
                          opposite: (control_transfer["target_predictions"],
                                     control_transfer["target_predictions_sha256"],
                                     control_transfer["target_support"])}
            for evaluation_source, (reference_path, reference_sha, reference_support) in references.items():
                if checksum(reference_path) != reference_sha:
                    raise ValueError("Frozen E002d validation predictions changed during E003a")
                current_tracks = load_predictions(output / f"{evaluation_source}-validation" / "predictions.npz")
                frozen_tracks = load_predictions(reference_path)
                current_by_id = {track["locator"]: track for track in current_tracks}
                frozen_by_id = {track["locator"]: track for track in frozen_tracks}
                if current_by_id.keys() != frozen_by_id.keys():
                    raise ValueError("Natural E003a validation population differs from frozen E002d control")
                for locator in current_by_id:
                    for key in ("targets", "gt_valid"):
                        if not np.array_equal(current_by_id[locator][key], frozen_by_id[locator][key]):
                            raise ValueError("Natural E003a validation GT differs from frozen E002d control")
                _same_metrics(evaluations[evaluation_source]["primary"]["denominators"], reference_support,
                              f"{evaluation_source}.validation_support")
            rows = []
            for evaluation_source in (source, opposite):
                actual = evaluations[evaluation_source]["primary"]
                if evaluation_source == source:
                    control_metrics = control_identity["validation_metrics"]
                    control_f1, control_ce = (control_metrics["track_weighted"]["macro_f1"], control_metrics["track_ce"])
                else:
                    control_f1, control_ce = target_reference["target_f1"], target_reference["target_ce"]
                rows.append({"training_source": source, "evaluation_source": evaluation_source,
                    "tracks": actual["denominators"]["tracks"],
                    "e003a_macro_f1": actual["track_weighted"]["macro_f1"], "e003a_ce": actual["track_ce"],
                    "e002d_control_macro_f1": control_f1, "e002d_control_ce": control_ce,
                    "control_checkpoint_sha256": control_identity["sha256"]})
            write_json(output / "natural-comparison.json", {"scope": "source and opposite validation; no test access",
                "control_transfer": control_transfer, "rows": rows})
            with (output / "natural-comparison.csv").open("w", newline="") as stream:
                import csv
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            if wandb_run is not None:
                comparison_table = shared_run.wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows])
        if wandb_run is not None:
            rows = []
            for view, models in diagnostics.items():
                for name, result in models.items():
                    rows.append([view, name, result["tracks"], result["no_input_tracks"],
                        result["primary"]["track_weighted"]["macro_f1"], result["primary"]["track_ce"],
                        result["no_input"]["track_weighted"]["macro_f1"], result["no_input"]["track_ce"]])
            table = shared_run.wandb.Table(columns=["view", "model", "tracks", "no_input_tracks",
                "macro_f1", "ce", "no_input_macro_f1", "no_input_ce"], data=rows)
            wandb_run.log({"diagnostics/summary": table} |
                          ({"comparison/natural-validation": comparison_table} if comparison_table is not None else {}))

    base_close = prepared["close"]
    base_batches = prepared["batches"]
    def close():
        try:
            target_points.close()
        finally:
            try:
                changed = [path for path, digest in target_protected.items() if checksum(path) != digest]
                write_json(args.output / "opposite-protected.json", {"hashes": target_protected, "changed": changed})
                if changed:
                    raise ValueError("Frozen opposite-source files changed")
            finally:
                base_close()

    prepared["dataset"] = dataset
    prepared["batches"] = batches
    prepared["epoch_operation"] = epoch_operation
    prepared["evaluation_predictions"] = evaluation_predictions
    prepared["evaluation_splits"] = [(source, "validation-smoke" if args.smoke else "validation"),
                                     (opposite, "validation-smoke" if args.smoke else "validation")]
    prepared["post_evaluation"] = post_evaluation
    prepared["close"] = close
    prepared["run_name"] = policy["run_name"]
    return prepared


def run(args):
    if args.source not in SOURCES or args.device not in ("cuda", "cpu"):
        raise ValueError("Unknown E003a source or device")
    if args.smoke and args.smoke_selection is None:
        raise ValueError("Smoke requires an explicit bounded source train/validation and opposite validation selection")
    if not args.smoke and args.smoke_selection is not None:
        raise ValueError("--smoke-selection is only valid with --smoke")
    if not args.opposite_rgb_preparation:
        raise ValueError("E003a requires the opposite-source RGB preparation record")
    validate_profile(args)
    torch.set_num_threads(2)
    axes = e002a_run.STRATA
    return shared_run.run_attempt(args.source, ARM, VARIANT, args.output, args.device, args.smoke,
        settings=SETTINGS, assemble=lambda *_: assembly(args), classes=e002a_run.CLASSES, axes=axes)


def add_arguments(parser):
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--control-run", type=Path, required=True,
                        help="Canonical E002d source RGB-dropout0.5 run directory")
    parser.add_argument("--control-transfer-results", type=Path,
                        help="Completed E002d opposite-validation result root; defaults to the documented sibling output")
    parser.add_argument("--opposite-rgb-preparation", type=Path,
                        help="Completed opposite-source RGB cache preparation record")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path, required=True)
    parser.add_argument("--indexed-point-cache", type=Path)
    parser.add_argument("--opposite-indexed-point-cache", type=Path)
    parser.add_argument("--point-workers", type=int, default=e002a_run.DEFAULT_POINT_WORKERS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path)
    parser.set_defaults(indexed_point_cache=None, opposite_indexed_point_cache=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
