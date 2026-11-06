"""E003c continuations from the pinned ROAD→LOKI E003b adaptation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pedestrian_behavior.data.indexed_points import PROTOCOL as POINT_PROTOCOL
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches
from pedestrian_behavior.evaluation import load_predictions, predict, prediction_metrics
from pedestrian_behavior.experiments import (e001b, e001b_run, e002a_run, e002b,
    e002_transfer, e003a_run, e003b_run, run as shared_run)
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.training import seed_run, train_epoch


SOURCE, TARGET = "road-waymo", "loki"
ARM, VARIANT = "g-rgb-dditr", "temporal-small"
ADAPTED = {"sha256": "fbf040490036aeca09014b5a595beb824b3fad1f118c6ec3c9ed1093a161a8b1",
    "epoch": 87, "f1": 0.6415806253538071, "ce": 0.516183951283285,
    "run": "20261104-092202/e002d-road-waymo-to-loki"}
SETTINGS = {"seed": 0, "batch_size": 64, "epochs": 100, "patience": 15, "clip_norm": 1.,
    "precision": "float32", "warmup_epochs": 3,
    "optimizer": {"lr": .0003, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8},
    "temporal_lr": .00003}
CLASSES = e003b_run.CLASSES


class TemporalHead(nn.Module):
    """The source temporal blocks/norm with the fixed E003b adapted head."""

    def __init__(self, temporal, final_norm, head):
        super().__init__()
        self.temporal, self.final_norm, self.head = temporal, final_norm, head
        for block in self.temporal:
            block.dropout.p = 0.
            for layer in block.ffn.modules():
                if isinstance(layer, nn.Dropout):
                    layer.p = 0.

    def forward(self, inputs, lengths=None):
        if isinstance(inputs, dict):
            lengths, inputs = inputs["lengths"], inputs["inputs"]
        valid = torch.arange(inputs.shape[1], device=inputs.device)[None, :] < lengths.to(inputs.device)[:, None]
        if not torch.isfinite(inputs[valid]).all():
            raise ValueError("Nonfinite cached pre-temporal features")
        x = inputs.masked_fill(~valid[..., None], 0.)
        for block in self.temporal:
            x = block(x, valid)
        return self.head(self.final_norm(x)).masked_fill(~valid[..., None], 0.)


def _source_target_assembly(outputs_root):
    base = e001b_run.assemble(SOURCE, "geometry", "B",
        e001_root=outputs_root / "experiments/E001", e001b_root=outputs_root / "experiments/E001b")
    normalization = base["normalization"]
    source_val_ids = {entry["locator"] for entry in base["dataset"](SOURCE, "validation").entries}
    frozen = e003b_run._load_source(outputs_root, SOURCE, "e002d", normalization, source_val_ids)
    target_data, target_reference = e003b_run._baselines(base, TARGET, normalization)
    return normalization, source_val_ids, frozen, target_data, target_reference


def _open_target(args, frozen, target_data, normalization):
    outputs_root = args.outputs_root.resolve()
    setups, bbox_records = {}, {}
    for dataset in e003b_run.SOURCES:
        setup_path = outputs_root / "experiments/E001/data-setup" / dataset / "setup.json"
        collection = outputs_root / "experiments/E001/reader-preparation" / dataset
        extension = outputs_root / "experiments/E001b/bbox-native" / dataset
        setups[dataset] = (setup_path, json.loads(setup_path.read_text()), collection, extension)
        bbox_records[dataset] = e001b.load_extension(collection, setup_path, extension)
    if frozen["run_config"].get("point_storage_protocol") != POINT_PROTOCOL:
        raise ValueError("Pinned E002d checkpoint uses a different indexed point-cache protocol")
    reader = e002_transfer._Target(TARGET, outputs_root, setups, bbox_records, args.probe, args.rgb_cache_root)
    supply = reader.supply({"input": "geometry+rgb+lidar"}, frozen["run_config"])
    catalog, points = reader.catalog, reader.points
    e003a_run._validate_rgb_preparation(args.rgb_preparation, TARGET, catalog,
        reader.extraction, args.rgb_cache_root)
    expected = {entry["locator"] for split in target_data.values() for entry in split.entries}
    if expected != set(catalog.tracks):
        raise ValueError("Prepared feature catalog differs from the frozen LOKI population")
    profile = e002a_run.validate_profile(frozen["run_config"]["profile"])
    datasets = {}
    for split, baseline in target_data.items():
        for entry in baseline.entries:
            if entry["slots"] ** 2 > profile["temporal_cells"]:
                raise ValueError("Complete target track exceeds the frozen temporal execution limit")
            points.require_track(entry["locator"])
            if points.counts[entry["locator"]].max(initial=0) > profile["frame_tokens"]:
                raise ValueError("Target point frame exceeds the frozen execution limit")
        datasets[split] = e002a_run.FeatureDataset(baseline, supply, TARGET, ARM)
    protected = catalog.protected | points.protected
    protected[str(args.rgb_preparation.resolve())] = checksum(args.rgb_preparation)
    protected[str((args.rgb_cache_root / "index.sqlite3").resolve())] = checksum(args.rgb_cache_root / "index.sqlite3")
    for split, baseline in target_data.items():
        for entry in baseline.entries:
            path = baseline.extension / "tracks" / entry["archive"]
            protected[str(path.resolve())] = entry["bbox_sha256"]
    return reader, profile, datasets, catalog, points, protected


def _smoke_subset(path, datasets):
    rows = json.loads(Path(path).read_text()).get("tracks", [])
    selected = {split: {row["locator"] for row in rows
        if row.get("dataset") == TARGET and row.get("split") == split}
        for split in ("training", "validation")}
    if any(not ids or len(ids) > 3 for ids in selected.values()):
        raise ValueError("E003c smoke needs 1–3 explicitly selected LOKI tracks per split")
    for split, ids in selected.items():
        available = {entry["locator"] for entry in datasets[split].entries}
        if not ids.issubset(available):
            raise ValueError("Smoke selection is outside the frozen LOKI split")
        datasets[split].entries = [entry for entry in datasets[split].entries if entry["locator"] in ids]
    return selected


def _extract_fusion(dataset, model, profile, device, split, output):
    records, logical, slots, batches = [], LogicalBatches(dataset, profile, training=False, seed=0, prefetch=True), 0, 0
    iterator = PredictionBatches(logical)
    captured = []
    hook = model.fusion.register_forward_hook(lambda _module, _inputs, value: captured.append(value))
    print(f"Extracting pre-temporal {split}: {len(dataset)} LOKI tracks", flush=True)
    try:
        with torch.inference_mode():
            for batch in iterator:
                batches += 1
                model.eval()
                logits = model(batch)
                if len(captured) != 1:
                    raise ValueError("Fusion output hook did not run exactly once")
                features = captured.pop()
                if features.shape != (len(batch["lengths"]), int(batch["lengths"].max()), 128):
                    raise ValueError("Unexpected pre-temporal fusion output shape")
                if not torch.isfinite(features).all():
                    raise ValueError("Nonfinite pre-temporal fusion features")
                for i, length in enumerate(batch["lengths"].tolist()):
                    locator = batch["locators"][i]
                    group = json.loads(locator).get("scenario")
                    if group is None:
                        raise ValueError("Frozen LOKI locator has no scenario")
                    records.append({"features": features[i, :length].cpu().numpy().astype(np.float32, copy=True),
                        "source_logits": logits[i, :length].cpu().numpy().astype(np.float32, copy=True),
                        "targets": batch["targets"][i, :length].numpy().astype(np.int64, copy=True),
                        "gt_valid": batch["gt_valid"][i, :length].numpy().astype(bool, copy=True),
                        "locator": locator, "archive": batch["archives"][i], "group": group,
                        "source_flags": batch["prediction_metadata"][i]["source_flags"]})
                    slots += length
                if batches % 10 == 0 or batches == len(logical):
                    write_json(output / "extraction-progress.json", {"split": split,
                        "completed_batches": batches, "completed_tracks": len(records), "context_slots": slots})
                    print(f"  {split}: {len(records)}/{len(dataset)} tracks, {slots} slots", flush=True)
                del batch, features, logits
    finally:
        hook.remove()
    if len(records) != len(dataset):
        raise ValueError("Fusion extraction changed the frozen target population")
    return records


def extract(args):
    outputs_root, output = args.outputs_root.resolve(), args.output.resolve()
    if output.exists() or not output.is_relative_to(outputs_root / "experiments/E003c"):
        raise ValueError("E003c feature cache must be fresh and under outputs/experiments/E003c")
    if args.smoke != (args.smoke_selection is not None):
        raise ValueError("Smoke extraction requires exactly one explicit selection file")
    if args.device not in ("cpu", "cuda") or (args.device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested extraction device is unavailable")
    output.mkdir(parents=True)
    write_json(output / "status.json", {"status": "incomplete", "failures": [], "held_out_tests": False})
    reader = None
    protected = {}
    try:
        normalization, _source_ids, frozen, target_data, target_reference = _source_target_assembly(outputs_root)
        reader, profile, datasets, catalog, points, protected = _open_target(args, frozen, target_data, normalization)
        selected = _smoke_subset(args.smoke_selection, target_data) if args.smoke else None
        if selected:
            for split, dataset in datasets.items():
                dataset.entries = [entry for entry in dataset.entries if entry["locator"] in selected[split]]
        seed_run(0)
        model = e002b.build_model(ARM, VARIANT, profile, dropout=.1, rgb_branch_dropout=.5).to(args.device)
        model.load_state_dict(frozen["checkpoint"]["model_state"], strict=True)
        e003b_run._freeze_source(model)
        source_predictions = e003b_run._source_prediction_path(frozen["root"], SOURCE, TARGET, "e002d", outputs_root)
        feature_cache, counts = {}, {}
        for split, dataset in datasets.items():
            records = _extract_fusion(dataset, model, profile, torch.device(args.device), split, output)
            if split == "validation":
                e003b_run._check_source_predictions(e003b_run._source_tracks(records), source_predictions)
            identity = {"experiment": "E003c", "stage": "pre-temporal-fusion", "source": SOURCE,
                "target": TARGET, "split": split, "tracks": len(records), "checkpoint_sha256": frozen["sha256"],
                "source_checkpoint": frozen["identity"], "source_normalization": normalization,
                "profile": profile, "target_catalog": catalog.identity, "target_points": points.identity,
                "smoke_selection": {k: sorted(v) for k, v in selected.items()} if selected else None,
                "code_sha256": checksum(Path(__file__))}
            path = output / f"features-{split}.npz"
            e003b_run._save_cache(path, records, identity)
            e003b_run._load_cache(path, identity)
            counts[split] = len(records)
            feature_cache[split] = identity | {"path": path.name, "sha256": checksum(path),
                "context_slots": int(sum(len(row["targets"]) for row in records))}
        protected[str(source_predictions.resolve())] = checksum(source_predictions)
        for path in (frozen["root"] / "best.pt", frozen["root"] / "config.json", frozen["root"] / "status.json",
                     Path(target_reference["setup_path"]), Path(target_reference["collection_path"]) / "manifest.json",
                     Path(target_reference["bbox"]["manifest_path"]), args.probe / "probe.json"):
            protected[str(path.resolve())] = checksum(path)
        changed = [path for path, digest in protected.items() if checksum(path) != digest]
        write_json(output / "feature-cache.json", feature_cache)
        write_json(output / "protected.json", {"hashes": protected, "changed": changed})
        write_json(output / "status.json", {"status": "complete", "failures": [], "tracks": counts,
            "scope": "frozen LOKI train/validation only; pre-temporal 128-D features", "held_out_tests": False})
        if changed:
            raise ValueError("A frozen source/cache input changed during E003c extraction")
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "failures": [{
            "type": type(error).__name__, "message": str(error)}], "held_out_tests": False})
        raise
    finally:
        if reader is not None:
            reader.close()


def _read_identity(path):
    with np.load(path, allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"]))


def _load_attempt_cache(root, split, *, stage, source_normalization, source_sha):
    if stage == "pre-temporal-fusion" and json.loads((root / "status.json").read_text()).get("status") != "complete":
        raise ValueError("Pre-temporal feature preparation must be complete")
    manifest = json.loads((root / "feature-cache.json").read_text())
    entry = manifest[split]
    path = root / entry["path"]
    if checksum(path) != entry["sha256"]:
        raise ValueError(f"Frozen E003c/E003b feature cache changed: {path}")
    identity = {key: value for key, value in entry.items()
        if key not in ("path", "sha256", "context_slots")}
    actual = _read_identity(path)
    if actual != identity or actual.get("split") != split or actual.get("target") != TARGET:
        raise ValueError(f"Feature-cache lineage mismatch: {path}")
    if stage == "pre-temporal-fusion" and (actual.get("experiment") != "E003c"
            or actual.get("stage") != "pre-temporal-fusion"
            or actual.get("checkpoint_sha256") != source_sha or actual.get("source_normalization") != source_normalization):
        raise ValueError("Pre-temporal cache differs from pinned E002d source inputs")
    if stage == "post-final-LayerNorm" and (actual.get("source") != SOURCE or actual.get("checkpoint_sha256") != source_sha
            or actual.get("source_normalization") != source_normalization):
        raise ValueError("E003b post-LayerNorm cache differs from pinned source inputs")
    return e003b_run._load_cache(path, identity), entry


def _load_adapted_head(outputs_root, normalization, source_sha):
    root = outputs_root / "experiments/E003b/runs" / ADAPTED["run"]
    checkpoint_path = root / "best.pt"
    if checksum(checkpoint_path) != ADAPTED["sha256"]:
        raise ValueError("Pinned E003b target-adapted head checkpoint hash changed")
    status, config = (json.loads((root / "status.json").read_text()),
                      json.loads((root / "config.json").read_text()))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (status.get("status") != "complete" or status.get("failures")
            or config.get("experiment") != "E003b" or config.get("original_source") != SOURCE
            or config.get("source") != TARGET or config.get("configuration") != "target-head"
            or config.get("variant") != "e002d" or config.get("target") != TARGET
            or config.get("reference") != "e002d" or config.get("held_out_tests") is not False
            or config.get("normalization") != normalization
            or config.get("source_checkpoint", {}).get("sha256") != source_sha
            or config.get("target_population") != {"training": 8574, "validation": 1861}
            or checkpoint.get("epoch") != ADAPTED["epoch"]
            or checkpoint.get("selection", {}).get("epoch") != ADAPTED["epoch"]
            or not math.isclose(checkpoint["selection"]["f1"], ADAPTED["f1"], rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(checkpoint["selection"]["ce"], ADAPTED["ce"], rel_tol=1e-6, abs_tol=1e-8)
            or set(checkpoint.get("model_state", {})) != {"linear.weight", "linear.bias"}):
        raise ValueError("Pinned E003b adapted-head run/configuration/selection mismatch")
    cache_manifest = json.loads((root / "feature-cache.json").read_text())
    if config.get("feature_cache") != cache_manifest:
        raise ValueError("Pinned E003b cache manifest differs from its saved run configuration")
    predictions = load_predictions(root / f"{TARGET}-validation/predictions.npz")
    metrics = prediction_metrics(predictions, len(CLASSES))
    e003a_run._same_metrics(checkpoint["validation"], metrics)
    saved_metrics = json.loads((root / f"{TARGET}-validation/metrics.json").read_text())["primary"]
    e003a_run._same_metrics(saved_metrics, metrics)
    return root, checkpoint, config


def _check_records(first, second, split):
    left, right = {row["locator"]: row for row in first}, {row["locator"]: row for row in second}
    if set(left) != set(right):
        raise ValueError(f"E003b/E003c {split} track populations differ")
    for locator, row in left.items():
        other = right[locator]
        if (row["archive"] != other["archive"] or not np.array_equal(row["targets"], other["targets"])
                or not np.array_equal(row["gt_valid"], other["gt_valid"])
                or any(not np.array_equal(row["source_flags"][key], other["source_flags"][key])
                       for key in row["source_flags"])):
            raise ValueError(f"E003b/E003c {split} track labels/availability differ: {locator}")


def _canonical_records(records):
    ordered = sorted(records, key=lambda row: row["locator"])
    if len({row["locator"] for row in ordered}) != len(ordered):
        raise ValueError("Duplicate cached track locator")
    return ordered


def _check_adapted_predictions(actual, path, *, expected_metrics=None, require_all=True):
    expected = {row["locator"]: row for row in load_predictions(path)}
    actual_ids = {row["locator"] for row in actual}
    if not actual_ids.issubset(expected) or (require_all and actual_ids != set(expected)):
        raise ValueError("E003c epoch0 predictions do not cover the exact E003b validation population")
    for row in actual:
        prior = expected[row["locator"]]
        if (not np.array_equal(row["targets"], prior["targets"])
                or not np.array_equal(row["gt_valid"], prior["gt_valid"])
                or not np.allclose(row["logits"], prior["logits"], rtol=1e-4, atol=1e-5)):
            raise ValueError("E003c epoch0 logits differ from the pinned E003b adapted-head predictions")
    metrics = prediction_metrics(actual, len(CLASSES))
    if expected_metrics is not None:
        e003a_run._same_metrics(expected_metrics, metrics)
    return metrics


def _model(arm, source_model, adapted_checkpoint, device):
    if arm == "head":
        model = e003b_run.AdaptationHead()
        model.load_state_dict(adapted_checkpoint["model_state"], strict=True)
    elif arm == "temporal-head":
        head = source_model.head
        head.load_state_dict({"weight": adapted_checkpoint["model_state"]["linear.weight"],
                              "bias": adapted_checkpoint["model_state"]["linear.bias"]}, strict=True)
        model = TemporalHead(source_model.temporal, source_model.final_norm, head)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError("Unknown E003c continuation arm")
    return model.to(device).eval()


def _warmup_epoch(arm, warmup_steps):
    from pedestrian_behavior.accumulation import warmup_lr
    def run(model, batches, optimizer, device, clip_norm, on_step=None, step_offset=0):
        peak_lrs = [group["peak_lr"] for group in optimizer.param_groups]
        for group, peak in zip(optimizer.param_groups, peak_lrs):
            group["lr"] = warmup_lr(peak, step_offset + 1, warmup_steps)
        def report(step, loss):
            if on_step:
                on_step(step, loss)
            for group, peak in zip(optimizer.param_groups, peak_lrs):
                group["lr"] = warmup_lr(peak, step + 1, warmup_steps)
        return train_epoch(model, batches, optimizer, device, clip_norm, on_step=report, step_offset=step_offset)
    return run


def optimizer_factory(arm):
    def build(model):
        head = model.head if hasattr(model, "head") else model.linear
        groups = [{"params": list(head.parameters()), "lr": SETTINGS["optimizer"]["lr"],
                   "peak_lr": SETTINGS["optimizer"]["lr"], "name": "head"}]
        if arm == "temporal-head":
            groups.append({"params": list(model.temporal.parameters()) + list(model.final_norm.parameters()),
                "lr": SETTINGS["temporal_lr"], "peak_lr": SETTINGS["temporal_lr"], "name": "temporal-and-final-norm"})
        return torch.optim.AdamW(groups, **SETTINGS["optimizer"])
    return build


def assembly(args):
    outputs_root = args.outputs_root.resolve()
    if args.smoke != (args.smoke_selection is not None):
        raise ValueError("Smoke training requires exactly one explicit selection file")
    normalization, _source_ids, frozen, target_data, target_reference = _source_target_assembly(outputs_root)
    head_root, adapted, adapted_config = _load_adapted_head(outputs_root, normalization, frozen["sha256"])
    fusion_records, fusion_meta = _load_attempt_cache(args.feature_cache, "training",
        stage="pre-temporal-fusion", source_normalization=normalization, source_sha=frozen["sha256"])
    fusion_val, fusion_val_meta = _load_attempt_cache(args.feature_cache, "validation",
        stage="pre-temporal-fusion", source_normalization=normalization, source_sha=frozen["sha256"])
    head_records, head_meta = _load_attempt_cache(head_root, "training",
        stage="post-final-LayerNorm", source_normalization=normalization, source_sha=frozen["sha256"])
    head_val, head_val_meta = _load_attempt_cache(head_root, "validation",
        stage="post-final-LayerNorm", source_normalization=normalization, source_sha=frozen["sha256"])
    selected = _smoke_subset(args.smoke_selection, target_data) if args.smoke else None
    if selected:
        head_records = [row for row in head_records if row["locator"] in selected["training"]]
        head_val = [row for row in head_val if row["locator"] in selected["validation"]]
    for first, second, split in ((fusion_records, head_records, "training"), (fusion_val, head_val, "validation")):
        _check_records(first, second, split)
        expected_ids = {entry["locator"] for entry in target_data[split].entries}
        if {row["locator"] for row in first} != expected_ids:
            raise ValueError(f"E003b/E003c {split} cache differs from frozen LOKI population")
    fusion_records = _canonical_records(fusion_records)
    head_records = _canonical_records(head_records)
    fusion_val = _canonical_records(fusion_val)
    head_val = _canonical_records(head_val)
    if selected:
        for split, entries in target_data.items():
            entries.entries = [entry for entry in entries.entries if entry["locator"] in selected[split]]
            fusion = fusion_records if split == "training" else fusion_val
            if {row["locator"] for row in fusion} != {entry["locator"] for entry in entries.entries}:
                raise ValueError(f"E003c smoke cache differs from selected {split} tracks")

    device = torch.device(args.device)
    seed_run(0)
    source_model = e002b.build_model(ARM, VARIANT, frozen["run_config"]["profile"],
        dropout=.1, rgb_branch_dropout=.5)
    source_model.load_state_dict(frozen["checkpoint"]["model_state"], strict=True)
    e003b_run._freeze_source(source_model)
    model = _model(args.arm, source_model, adapted, device)
    del source_model
    datasets = {"training": e003b_run.CachedTracks(head_records if args.arm == "head" else fusion_records),
        "validation": e003b_run.CachedTracks(head_val if args.arm == "head" else fusion_val)}
    val_batches = e003b_run._batches(datasets["validation"], seed=0)
    initial_tracks = predict(model, val_batches, device)
    metrics = _check_adapted_predictions(initial_tracks, head_root / f"{TARGET}-validation/predictions.npz",
        expected_metrics=None if args.smoke else adapted["validation"], require_all=not args.smoke)
    if not args.smoke and (not math.isclose(metrics["track_weighted"]["macro_f1"], ADAPTED["f1"], rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(metrics["track_ce"], ADAPTED["ce"], rel_tol=1e-6, abs_tol=1e-8)):
        raise ValueError("E003c epoch0 metrics differ from the pinned E003b selected head")
    parameter_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if args.arm == "head" and parameter_names != ["linear.weight", "linear.bias"]:
        raise ValueError("Head control must train only the 516 adapted-head parameters")
    if args.arm == "temporal-head" and (not any(name.startswith("temporal.") for name in parameter_names)
            or not all(name.startswith(("temporal.", "final_norm.", "head.")) for name in parameter_names)):
        raise ValueError("Temporal-head must train only temporal blocks, final LayerNorm and head")

    def data_for(name, split):
        if name != TARGET or split not in ("training", "validation", "validation-smoke"):
            raise ValueError("E003c exposes only LOKI training/validation cached features")
        return datasets["validation" if split == "validation-smoke" else split]

    train_steps = math.ceil(len(datasets["training"]) / SETTINGS["batch_size"])
    return {"config": {"experiment": "E003c", "original_source": SOURCE, "target": TARGET,
            "reference": "E002d ROAD→LOKI + E003b head epoch87", "arm": args.arm,
            "dropout_enabled": False, "optimizer_reset": True,
            "training_order": "locator-sorted before persistent seeded shuffle",
            "source_checkpoint": frozen["identity"], "source_checkpoint_sha256": frozen["sha256"],
            "adapted_head_checkpoint": {"path": str(head_root / "best.pt"), "sha256": ADAPTED["sha256"],
                "selected_epoch": ADAPTED["epoch"], "selection": adapted["selection"]},
            "adapted_head_validation": metrics, "normalization_source": SOURCE, "normalization": normalization,
            "population": {"training": len(datasets["training"]), "validation": len(datasets["validation"])},
            "feature_cache": {"pre_temporal": {"training": fusion_meta, "validation": fusion_val_meta},
                "post_final_norm": {"training": head_meta, "validation": head_val_meta}},
            "trainable_parameters": parameter_names, "held_out_tests": False,
            "smoke_selection": {k: sorted(v) for k, v in selected.items()} if selected else None},
        "normalization": normalization, "datasets": {TARGET: target_reference},
        "build_model": lambda: model, "dataset": data_for, "batches": e003b_run._batches,
        "conditions": lambda metadata, arrays, track, dataset: ({}, {}),
        "evaluation_splits": [(TARGET, "validation-smoke" if args.smoke else "validation")],
        "optimizer_factory": optimizer_factory(args.arm), "initial_candidate": True,
        "epoch_operation": _warmup_epoch(args.arm, SETTINGS["warmup_epochs"] * train_steps),
        "log_learning_rate": True,
        "custom_smoke": True,
        "run_name": f"e003c-road-waymo-to-loki-{args.arm}-s0" + ("-smoke" if args.smoke else "")}


def train(args):
    if args.arm not in ("head", "temporal-head"):
        raise ValueError("Unknown E003c continuation arm")
    if args.device not in ("cpu", "cuda") or (args.device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested training device is unavailable")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.outputs_root.resolve() / "experiments/E003c"):
        raise ValueError("Fresh E003c attempt required under outputs/experiments/E003c")
    return shared_run.run_attempt(TARGET, "temporal-adaptation", args.arm, output, args.device, args.smoke,
        settings=SETTINGS, assemble=lambda *_: assembly(args), classes=CLASSES, axes={})


def extract_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path,
        default=Path("outputs/experiments/E002a/rgb-cache-preparation-03/cache.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path)
    extract(parser.parse_args())


def train_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("head", "temporal-head"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path,
        default=Path("outputs/experiments/E002a/rgb-cache-preparation-03/cache.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path)
    train(parser.parse_args())
