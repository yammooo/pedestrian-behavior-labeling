"""E003b frozen-representation target-head adaptation."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from pedestrian_behavior.data.loading import collate_tracks
from pedestrian_behavior.data.indexed_points import PROTOCOL as POINT_PROTOCOL
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches
from pedestrian_behavior.evaluation import (load_predictions, predict, prediction_metrics,
                                            save_predictions)
from pedestrian_behavior.experiments import (e001b, e001b_run, e002a_run, e002b,
                                             e002_transfer, e003a_run, run as shared_run)
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.training import seed_run, train_epoch


ARM, VARIANT = e003a_run.ARM, e003a_run.VARIANT
SOURCES = ("road-waymo", "loki")
CLASSES = ("MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING")
SETTINGS = {"seed": 0, "batch_size": 64, "epochs": 100, "patience": 15, "clip_norm": 1.,
            "precision": "float32", "warmup_epochs": 3,
            "optimizer": {"lr": .0003, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8}}
MASKED = {
    "road-waymo": {"sha256": "069fd312763fb4ee7452559bde7d22b3629bdb9be124376046ab94fd96f0fc01", "epoch": 24},
    "loki": {"sha256": "8d0ec20b63b93558bb0edb0962ce39e266cb63c571981f0874d598c488a0a6e6", "epoch": 12},
}


def _run_root(root, source, reference):
    if reference == "e002d":
        return root / "experiments/E002d/runs" / e003a_run.CONTROL[source]["name"]
    return root / "experiments/E003a/runs/20261103-105614" / source


def _load_source(root, source, reference, normalization, source_val_ids):
    run_root = _run_root(root, source, reference).resolve()
    config_path = run_root / "config.json"
    run_config = json.loads(config_path.read_text())
    if reference == "e002d":
        checkpoint, identity = e003a_run._validate_control(run_root, source, normalization, source_val_ids)
        expected_sha, epoch = e003a_run.CONTROL[source]["sha256"], e003a_run.CONTROL[source]["epoch"]
    else:
        expected_sha, epoch = MASKED[source]["sha256"], MASKED[source]["epoch"]
        if checksum(run_root / "best.pt") != expected_sha:
            raise ValueError("Pinned E003a checkpoint hash changed")
        status = json.loads((run_root / "status.json").read_text())
        checkpoint = torch.load(run_root / "best.pt", map_location="cpu", weights_only=True)
        source_path = run_root / f"{source}-validation"
        metrics = json.loads((source_path / "metrics.json").read_text())["primary"]
        tracks = load_predictions(source_path / "predictions.npz")
        if (status.get("status") != "complete"
                or run_config.get("experiment") != "E003a" or run_config.get("source") != source
                or run_config.get("configuration") != ARM or run_config.get("variant") != VARIANT
                or run_config.get("held_out_tests") is not False or run_config.get("settings", {}).get("seed") != 0
                or {track["locator"] for track in tracks} != source_val_ids):
            raise ValueError("Pinned E003a source-run identity mismatch")
        actual = prediction_metrics(tracks, len(CLASSES))
        e003a_run._same_metrics(metrics, actual)
        if (not math.isclose(checkpoint["selection"]["f1"], actual["track_weighted"]["macro_f1"], rel_tol=1e-6, abs_tol=1e-8)
                or not math.isclose(checkpoint["selection"]["ce"], actual["track_ce"], rel_tol=1e-6, abs_tol=1e-8)):
            raise ValueError("E003a source-selection metrics differ from its saved validation predictions")
        identity = {"path": str(run_root / "best.pt"), "sha256": expected_sha,
            "selected_epoch": epoch, "validation_metrics": metrics,
            "validation_predictions": str(source_path / "predictions.npz"),
            "validation_predictions_sha256": checksum(source_path / "predictions.npz")}
    if (checksum(run_root / "best.pt") != expected_sha or checkpoint.get("epoch") != epoch
            or checkpoint.get("selection", {}).get("epoch") != epoch
            or checkpoint.get("normalization") != normalization
            or run_config.get("source") != source
            or run_config.get("configuration", run_config.get("arm")) != ARM
            or run_config.get("variant", run_config.get("model_variant")) != VARIANT
            or run_config.get("model") != e003a_run.model_spec()
            or run_config.get("held_out_tests") is not False
            or checkpoint.get("config", {}).get("model") != e003a_run.model_spec()):
        raise ValueError("Frozen source checkpoint/configuration/normalization mismatch")
    return {"root": run_root, "checkpoint": checkpoint, "run_config": run_config,
        "identity": identity, "sha256": expected_sha, "epoch": epoch,
        "config_sha256": checksum(config_path)}


def _baselines(source_assembly, target, normalization):
    reference = source_assembly["datasets"][target]
    setup, collection = Path(reference["setup_path"]), Path(reference["collection_path"])
    extension = Path(reference["bbox"]["manifest_path"]).parent
    datasets = {split: e001b.dataset_from_extension(collection, setup, extension, split, "geometry",
        normalization["baseline"], normalization["bbox"]) for split in ("training", "validation")}
    return datasets, reference


def _smoke_subset(path, target, datasets):
    rows = json.loads(Path(path).read_text()).get("tracks", [])
    selected = {split: {row["locator"] for row in rows if row.get("dataset") == target and row.get("split") == split}
                for split in ("training", "validation")}
    if any(not ids or len(ids) > 10 for ids in selected.values()):
        raise ValueError("Smoke selection needs 1–10 target training and validation tracks")
    for split, ids in selected.items():
        if not ids.issubset({entry["locator"] for entry in datasets[split].entries}):
            raise ValueError("Smoke selection is outside the frozen target split")
        datasets[split].entries = [entry for entry in datasets[split].entries if entry["locator"] in ids]


def _population(datasets):
    return {entry["locator"] for split in datasets.values() for entry in split.entries}


def _group(locator):
    value = json.loads(locator)
    group = value.get("clip", value.get("scenario"))
    if group is None:
        raise ValueError("Frozen locator has no clip/scenario group")
    return group


def _extract(dataset, model, profile, device, split, output):
    records, slots, batches = [], 0, 0
    logical = LogicalBatches(dataset, profile, training=False, seed=0, prefetch=True)
    iterator = PredictionBatches(logical)
    progress = output / "extraction-progress.json"
    captured = []
    hook = model.final_norm.register_forward_hook(lambda _module, _inputs, value: captured.append(value))
    print(f"Extracting {split}: {len(dataset)} target tracks", flush=True)
    try:
        for batch in iterator:
            batches += 1
            with torch.inference_mode():
                model.eval()
                logits = model(batch)
                if len(captured) != 1:
                    raise ValueError("Frozen representation hook did not run exactly once")
                features = captured.pop()
                if features.ndim != 3 or features.shape[-1] != 128 or not torch.isfinite(features).all():
                    raise ValueError("Frozen model did not produce finite post-LayerNorm 128-D features")
            for i, length in enumerate(batch["lengths"].tolist()):
                targets = batch["targets"][i, :length].numpy().astype(np.int64, copy=True)
                gt_valid = batch["gt_valid"][i, :length].numpy().astype(bool, copy=True)
                if not gt_valid.any():
                    raise ValueError("Target population contains an empty-GT track")
                locator = batch["locators"][i]
                records.append({"features": features[i, :length].cpu().numpy().astype(np.float32, copy=True),
                    "source_logits": logits[i, :length].cpu().numpy().astype(np.float32, copy=True),
                    "targets": targets, "gt_valid": gt_valid, "locator": locator,
                    "archive": batch["archives"][i], "group": _group(locator),
                    "source_flags": batch["prediction_metadata"][i]["source_flags"]})
                slots += length
            if batches % 10 == 0 or batches == len(logical):
                write_json(progress, {"split": split, "completed_batches": batches,
                    "completed_tracks": len(records), "context_slots": slots})
                print(f"  {split}: {len(records)}/{len(dataset)} tracks, {slots} slots", flush=True)
            del batch, features, logits
    finally:
        hook.remove()
    if len(records) != len(dataset):
        raise ValueError("Feature extraction changed the frozen target population")
    return records


def _save_cache(path, records, identity):
    if not records:
        raise ValueError("Cannot save an empty feature split")
    offsets = np.cumsum([0] + [len(row["targets"]) for row in records], dtype=np.int64)
    flag_names = set(records[0]["source_flags"])
    if any(set(row["source_flags"]) != flag_names for row in records):
        raise ValueError("Target source-flag schema differs between tracks")
    arrays = {"features": np.concatenate([row["features"] for row in records]),
        "source_logits": np.concatenate([row["source_logits"] for row in records]),
        "targets": np.concatenate([row["targets"] for row in records]),
        "gt_valid": np.concatenate([row["gt_valid"] for row in records]), "offsets": offsets,
        "locators": np.asarray([row["locator"] for row in records], dtype=np.str_),
        "archives": np.asarray([row["archive"] for row in records], dtype=np.str_),
        "groups": np.asarray([row["group"] for row in records], dtype=np.str_),
        "metadata": np.asarray(json.dumps(identity, sort_keys=True, allow_nan=False))}
    arrays.update({"flag__" + key: np.concatenate([row["source_flags"][key] for row in records])
                   for key in sorted(flag_names)})
    temporary = Path(path).with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _load_cache(path, identity):
    with np.load(path, allow_pickle=False) as archive:
        if json.loads(str(archive["metadata"])) != identity:
            raise ValueError("Feature-cache lineage mismatch")
        arrays = {key: archive[key] for key in archive.files if key != "metadata"}
    offsets = arrays["offsets"]
    if (arrays["features"].dtype != np.float32 or arrays["features"].ndim != 2
            or arrays["features"].shape[1] != 128 or offsets[0] != 0
            or offsets[-1] != len(arrays["features"]) or len(offsets) != len(arrays["locators"]) + 1
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Invalid pickle-free target feature cache")
    flags = {key.removeprefix("flag__"): value for key, value in arrays.items() if key.startswith("flag__")}
    records = []
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        row = {"features": arrays["features"][start:end], "source_logits": arrays["source_logits"][start:end],
            "targets": arrays["targets"][start:end], "gt_valid": arrays["gt_valid"][start:end],
            "locator": str(arrays["locators"][i]), "archive": str(arrays["archives"][i]),
            "group": str(arrays["groups"][i]), "source_flags": {key: value[start:end] for key, value in flags.items()}}
        if not row["gt_valid"].any():
            raise ValueError("Cached target track has no accepted GT")
        records.append(row)
    return records


class CachedTracks:
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {"inputs": torch.from_numpy(row["features"]), "targets": torch.from_numpy(row["targets"]),
            "gt_valid": torch.from_numpy(row["gt_valid"]), "locator": row["locator"], "archive": row["archive"],
            "prediction_metadata": {"group": row["group"], "source_flags": row["source_flags"]}}


class AdaptationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 4)

    def forward(self, inputs, lengths):
        logits = self.linear(inputs)
        valid = torch.arange(inputs.shape[1], device=inputs.device)[None, :] < lengths.to(inputs.device)[:, None]
        return logits.masked_fill(~valid[..., None], 0.)


def _batches(dataset, training=False, seed=0):
    def collate(samples):
        result = collate_tracks(samples)
        result["prediction_metadata"] = [sample["prediction_metadata"] for sample in samples]
        return result
    return DataLoader(dataset, batch_size=64, shuffle=training, drop_last=False,
        generator=torch.Generator().manual_seed(seed), collate_fn=collate)


def _freeze_source(model):
    if not isinstance(model.head, nn.Linear) or model.head.in_features != 128 or model.head.out_features != 4:
        raise ValueError("E003b requires the pinned 128→4 source classifier")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _warmup_epoch(model, batches, optimizer, device, clip_norm, on_step=None, step_offset=0, *, warmup_steps):
    from pedestrian_behavior.accumulation import warmup_lr
    peak = SETTINGS["optimizer"]["lr"]
    for group in optimizer.param_groups:
        group["lr"] = warmup_lr(peak, step_offset + 1, warmup_steps)
    def report(step, loss):
        if on_step:
            on_step(step, loss)
        rate = warmup_lr(peak, step + 1, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = rate
    return train_epoch(model, batches, optimizer, device, clip_norm, on_step=report, step_offset=step_offset)


def _source_tracks(records):
    return [{"logits": row["source_logits"], "targets": row["targets"], "gt_valid": row["gt_valid"],
        "locator": row["locator"], "archive": row["archive"], "group": row["group"],
        "conditions": {}, "source_flags": row["source_flags"]} for row in records]


def _check_source_predictions(actual, path):
    expected = {row["locator"]: row for row in load_predictions(path)}
    if not {row["locator"] for row in actual}.issubset(expected):
        raise ValueError("Target validation track is absent from frozen source-head predictions")
    for row in actual:
        prior = expected[row["locator"]]
        if (not np.array_equal(row["targets"], prior["targets"])
                or not np.array_equal(row["gt_valid"], prior["gt_valid"])
                or not np.allclose(row["logits"], prior["logits"], rtol=1e-4, atol=1e-5)):
            raise ValueError("Re-extracted source-head logits differ from frozen target predictions")


def _source_prediction_path(root, source, target, reference, outputs_root):
    if reference == "e003a":
        return root / f"{target}-validation/predictions.npz"
    return outputs_root / "experiments/E002/transfer-validation-20261102/cells" / source / "geometry+rgb+lidar/predictions.npz"


def assembly(args):
    outputs_root = args.outputs_root.resolve()
    source, target = args.source, "loki" if args.source == "road-waymo" else "road-waymo"
    source_assembly = e001b_run.assemble(source, "geometry", "B",
        e001_root=outputs_root / "experiments/E001", e001b_root=outputs_root / "experiments/E001b")
    normalization = source_assembly["normalization"]
    source_val_ids = {entry["locator"] for entry in source_assembly["dataset"](source, "validation").entries}
    frozen = _load_source(outputs_root, source, args.reference, normalization, source_val_ids)
    target_data, target_reference = _baselines(source_assembly, target, normalization)
    expected_population = _population(target_data)
    if args.smoke:
        _smoke_subset(args.smoke_selection, target, target_data)

    setups, bbox_records = {}, {}
    for dataset in SOURCES:
        setup_path = outputs_root / "experiments/E001/data-setup" / dataset / "setup.json"
        collection = outputs_root / "experiments/E001/reader-preparation" / dataset
        extension = outputs_root / "experiments/E001b/bbox-native" / dataset
        setups[dataset] = (setup_path, json.loads(setup_path.read_text()), collection, extension)
        bbox_records[dataset] = e001b.load_extension(collection, setup_path, extension)
    if frozen["run_config"].get("point_storage_protocol") != POINT_PROTOCOL:
        raise ValueError("Pinned source checkpoint uses a different indexed point-cache protocol")
    target_reader = e002_transfer._Target(target, outputs_root, setups, bbox_records, args.probe, args.rgb_cache_root)
    row = {"input": "geometry+rgb+lidar"}
    supply = target_reader.supply(row, frozen["run_config"])
    catalog, points = target_reader.catalog, target_reader.points
    _, transform = e002a_run.load_configuration(args.probe)
    identity = target_reader.extraction
    e003a_run._validate_rgb_preparation(args.rgb_preparation, target, catalog, identity, args.rgb_cache_root)
    if expected_population != set(catalog.tracks):
        raise ValueError("Target feature catalog differs from frozen train/validation population")
    profile = e002a_run.validate_profile(frozen["run_config"]["profile"])
    datasets = {}
    protected = catalog.protected | points.protected
    protected[str(args.rgb_preparation.resolve())] = checksum(args.rgb_preparation)
    protected[str((args.rgb_cache_root / "index.sqlite3").resolve())] = checksum(args.rgb_cache_root / "index.sqlite3")
    for split, baseline in target_data.items():
        for entry in baseline.entries:
            if entry["slots"] ** 2 > profile["temporal_cells"]:
                raise ValueError("Target track exceeds pinned temporal execution limit")
            points.require_track(entry["locator"])
            if points.counts[entry["locator"]].max(initial=0) > profile["frame_tokens"]:
                raise ValueError("Target point frame exceeds pinned execution limit")
            bbox_path = baseline.extension / "tracks" / entry["archive"]
            protected[str(bbox_path.resolve())] = entry["bbox_sha256"]
        datasets[split] = e002a_run.FeatureDataset(baseline, supply, target, ARM)

    # Reset backend and RNG immediately before deterministic frozen extraction.
    seed_run(0)
    model = e002b.build_model(ARM, VARIANT, profile, dropout=.1, rgb_branch_dropout=.5).to(args.device)
    model.load_state_dict(frozen["checkpoint"]["model_state"], strict=True)
    _freeze_source(model)
    output = args.output
    feature_records, feature_meta = {}, {}
    code = {"checkpoint_sha256": frozen["sha256"], "source_normalization": normalization,
        "target_catalog": catalog.identity, "target_points": points.identity,
        "code_sha256": checksum(Path(__file__))}
    write_json(output / "extraction-progress.json", {"status": "started", "scope": "target train/validation only"})
    print(f"E003b {source}→{target} ({args.reference}); extracting frozen target representations", flush=True)
    for split, dataset in datasets.items():
        records = _extract(dataset, model, profile, torch.device(args.device), split, output)
        cache_identity = code | {"source": source, "target": target, "split": split, "tracks": len(records)}
        path = output / f"features-{split}.npz"
        _save_cache(path, records, cache_identity)
        feature_records[split] = _load_cache(path, cache_identity)
        feature_meta[split] = cache_identity | {"path": path.name, "sha256": checksum(path),
            "context_slots": int(sum(len(row["targets"]) for row in records))}
        write_json(output / "feature-cache.json", feature_meta)
    del model

    source_predictions = _source_tracks(feature_records["validation"])
    reference_path = _source_prediction_path(frozen["root"], source, target, args.reference, outputs_root)
    _check_source_predictions(source_predictions, reference_path)
    source_metrics = {"split": "validation", "primary": prediction_metrics(source_predictions, 4), "strata": {}}
    source_output = output / "source-head-validation"
    source_output.mkdir()
    save_predictions(source_output / "predictions.npz", source_predictions)
    write_json(source_output / "metrics.json", source_metrics)
    protected[str(reference_path.resolve())] = checksum(reference_path)
    protected[str((args.probe / "probe.json").resolve())] = checksum(args.probe / "probe.json")
    protected[str(Path(transform).resolve())] = checksum(transform)
    for path in (frozen["root"] / "best.pt", frozen["root"] / "config.json", frozen["root"] / "status.json"):
        protected[str(path.resolve())] = checksum(path)
    for path in (Path(target_reference["setup_path"]), Path(target_reference["collection_path"]) / "manifest.json",
                 Path(target_reference["bbox"]["manifest_path"])):
        protected[str(path.resolve())] = checksum(path)

    def close():
        target_reader.close()
        changed = [path for path, digest in protected.items() if checksum(path) != digest]
        write_json(output / "protected.json", {"hashes": protected, "changed": changed})
        if changed:
            raise ValueError("Frozen target or source files changed during E003b")

    def data_for(name, split):
        if split == "validation-smoke":
            split = "validation"
        if name != target or split not in datasets:
            raise ValueError("E003b exposes only target training/validation features")
        return CachedTracks(feature_records[split])

    def build_head():
        seed_run(0)
        return AdaptationHead()

    warmup_steps = SETTINGS["warmup_epochs"] * math.ceil(len(feature_records["training"]) / SETTINGS["batch_size"])
    def post_evaluation(model, evaluations, fitted, output, device, wandb_run):
        adapted = evaluations.pop(target)
        evaluations["source-head"] = source_metrics
        evaluations["adapted-head"] = adapted
        original_f1 = source_metrics["primary"]["track_weighted"]["macro_f1"]
        adapted_f1 = adapted["primary"]["track_weighted"]["macro_f1"]
        write_json(output / "validation-comparison.json", {"scope": "target validation only; no test access",
            "training_source": source, "target": target, "reference": args.reference,
            "source_checkpoint": frozen["identity"], "tracks": len(feature_records["validation"]),
            "source_head": source_metrics["primary"], "adapted_head": adapted["primary"],
            "delta_track_weighted_macro_f1": adapted_f1 - original_f1,
            "delta_track_weighted_macro_f1_pp": 100 * (adapted_f1 - original_f1)})

    run_label = "road" if source == "road-waymo" else "loki"
    target_label = "road" if target == "road-waymo" else "loki"
    return {"config": {"experiment": "E003b", "original_source": source, "target": target,
            "reference": args.reference, "source_checkpoint": frozen["identity"],
            "target_normalization_source": source, "normalization": normalization,
            "target_population": {split: len(data) for split, data in feature_records.items()},
            "feature_cache": feature_meta, "profile": profile, "head": {"shape": [128, 4], "parameters": 516},
            "held_out_tests": False},
        "normalization": normalization, "datasets": {target: target_reference},
        "build_model": build_head,
        "dataset": data_for, "batches": _batches, "conditions": lambda metadata, arrays, track, dataset: ({}, {}),
        "post_evaluation": post_evaluation, "close": close, "custom_smoke": True,
        "evaluation_splits": [(target, "validation-smoke" if args.smoke else "validation")],
        "epoch_operation": lambda model, batches, optimizer, device, clip_norm, on_step=None, step_offset=0:
            _warmup_epoch(model, batches, optimizer, device, clip_norm, on_step, step_offset, warmup_steps=warmup_steps),
        "log_learning_rate": True,
        "run_name": f"e003b-{run_label}-to-{target_label}-{args.reference}-head-s0" + ("-smoke" if args.smoke else "")}


def run(args):
    if args.source not in SOURCES or args.reference not in ("e002d", "e003a"):
        raise ValueError("Unknown E003b source/reference")
    if args.device not in ("cpu", "cuda") or (args.device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device is unavailable")
    if args.smoke != (args.smoke_selection is not None):
        raise ValueError("Smoke requires one explicit selection file")
    target = "loki" if args.source == "road-waymo" else "road-waymo"
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.outputs_root.resolve() / "experiments/E003b"):
        raise ValueError("Fresh E003b attempt path required under outputs/experiments/E003b")
    return shared_run.run_attempt(target, "target-head", args.reference, output, args.device, args.smoke,
        settings=SETTINGS, assemble=lambda *_: assembly(args), classes=CLASSES, axes={})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--reference", choices=("e002d", "e003a"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
