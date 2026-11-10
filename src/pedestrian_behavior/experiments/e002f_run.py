"""E002f ROAD source runs and paired LOKI temporal adaptation."""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import (load_predictions, predict, prediction_metrics,
    save_predictions, track_cross_entropy)
from pedestrian_behavior.experiments import (e001b_run, e002a_run, e002b_run,
    e003b_run, e003c_run, run as shared_run)
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.models import e002f
from pedestrian_behavior.training import seed_run


SOURCE, TARGET, ARM = "road-waymo", "loki", "g-rgb-dditr"
CLASSES = e003b_run.CLASSES
SETTINGS = e003c_run.SETTINGS
SOURCE_PROFILE = Path("outputs/experiments/E002d/calibration-20261030-sequential-ready/profile.json")


def resource_code():
    from pedestrian_behavior.experiments import e002b_run as base
    modules = (e002f, e003b_run, e003c_run)
    return base.resource_code() | {module.__name__: checksum(module.__file__) for module in modules} | {
        __name__: checksum(__file__)}


def _validate_roots(args, config):
    root = args.outputs_root.resolve()
    locations = [args.rgb_cache_root.resolve(), args.point_cache_root.resolve(), args.output.resolve()]
    if not locations[-1].is_relative_to(root / "experiments/E002f/runs"):
        raise ValueError("E002f source attempts must be fresh children of outputs/experiments/E002f/runs")
    if any(a.is_relative_to(b) or b.is_relative_to(a)
            for i, a in enumerate(locations) for b in locations[i + 1:]):
        raise ValueError("E002f output and frozen feature caches must have separate roots")
    protected = [root / "experiments" / name for name in ("E001", "E001b")]
    protected.extend(Path(config[k]).resolve() for k in ("loki_root", "waymo_root", "road_index") if config.get(k))
    if any(locations[-1].is_relative_to(path) or path.is_relative_to(locations[-1])
            for path in (root / "experiments/E002a", root / "experiments/E002d")):
        raise ValueError("E002f output cannot overlap established experiment evidence")
    if any(location.is_relative_to(path) or path.is_relative_to(location)
            for location in locations for path in protected):
        raise ValueError("E002f outputs cannot overlap native data or established experiment evidence")


def _profile_record(path, device):
    record = json.loads(Path(path).read_text())
    if (record.get("status") != "complete" or record.get("experiment") != "E002f"
            or record.get("device") != device or record.get("code") != resource_code()
            or record.get("torch") != str(torch.__version__) or record.get("cuda") != torch.version.cuda
            or not set(e002f.ARCHITECTURES).issubset(record.get("model_coverage", ()))
            or record.get("model_specs") != {f"{ARM}/{name}": e002f.model_spec(name)
                for name in e002f.ARCHITECTURES}):
        raise ValueError("Completed E002f full-batch calibration required")
    if device == "cuda" and (not torch.cuda.is_available()
            or record.get("gpu") != torch.cuda.get_device_name()):
        raise ValueError("E002d resource profile hardware differs from requested device")
    return record


def source_assembly(args):
    _profile_record(args.profile, args.device)
    args.arm = ARM
    args.model_variant = args.architecture
    policy = {"experiment": "E002f", "resource_code": resource_code,
        "validate_roots": _validate_roots,
        "model_spec": lambda _arm, variant: e002f.model_spec(variant),
        "model_factory": lambda _arm, variant, profile: e002f.build_model(variant, profile),
        "run_name": f"e002f-road-waymo-to-loki-{args.architecture}-source-s0" + ("-smoke" if args.smoke else "")}
    return e002a_run.assembly(args, policy=policy)


def train_source(args):
    if args.source != SOURCE or args.architecture not in e002f.ARCHITECTURES:
        raise ValueError("E002f trains the two declared architectures on ROAD only")
    if args.smoke != (args.smoke_selection is not None):
        raise ValueError("Source smoke requires one explicit bounded ROAD selection")
    if args.device not in ("cuda", "cpu") or (args.device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested source-training device is unavailable")
    _profile_record(args.profile, args.device)
    torch.set_num_threads(2)
    SETTINGS_SOURCE = e002a_run.SETTINGS | {"seed": 0}
    axes = e002a_run.STRATA
    return shared_run.run_attempt(SOURCE, ARM, args.architecture, args.output, args.device, args.smoke,
        settings=SETTINGS_SOURCE, assemble=lambda *_: source_assembly(args), classes=CLASSES, axes=axes)


def _source_run(path, architecture, outputs_root):
    root = Path(path).resolve()
    status = json.loads((root / "status.json").read_text())
    config_path, checkpoint_path = root / "config.json", root / "best.pt"
    config = json.loads(config_path.read_text())
    digest = checksum(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (status.get("status") != "complete" or config.get("experiment") != "E002f"
            or config.get("source") != SOURCE or config.get("configuration") != ARM
            or config.get("variant") != architecture or config.get("model") != e002f.model_spec(architecture)
            or config.get("settings", {}).get("seed") != 0 or config.get("held_out_tests") is not False
            or checkpoint.get("epoch") != checkpoint.get("selection", {}).get("epoch")
            or checkpoint.get("normalization") is None
            or checkpoint.get("config", {}).get("model") != e002f.model_spec(architecture)):
        raise ValueError("E002f source checkpoint, configuration, or selected epoch mismatch")
    source_data = e001b_run.assemble(SOURCE, "geometry", "B", e001_root=outputs_root / "experiments/E001",
        e001b_root=outputs_root / "experiments/E001b")
    if checkpoint.get("normalization") != source_data["normalization"]:
        raise ValueError("E002f checkpoint normalization differs from ROAD training-source statistics")
    full_validation = {entry["locator"] for entry in source_data["dataset"](SOURCE, "validation").entries}
    smoke_limit = config.get("smoke_limits")
    if smoke_limit:
        expected = full_validation & set(smoke_limit.get("selected_tracks", ()))
        val_name = "validation-smoke"
    else:
        expected, val_name = full_validation, "validation"
    val_path = root / f"{SOURCE}-{val_name}" / "predictions.npz"
    predictions = load_predictions(val_path)
    if {row["locator"] for row in predictions} != expected:
        raise ValueError("E002f source validation predictions do not cover the frozen ROAD validation split")
    metrics = prediction_metrics(predictions, len(CLASSES))
    if (not math.isclose(checkpoint["selection"]["f1"], metrics["track_weighted"]["macro_f1"], rel_tol=1e-6, abs_tol=1e-8)
            or not math.isclose(checkpoint["selection"]["ce"], metrics["track_ce"], rel_tol=1e-6, abs_tol=1e-8)):
        raise ValueError("E002f selected checkpoint does not match saved ROAD validation predictions")
    identity = {"path": str(checkpoint_path), "sha256": digest, "selected_epoch": checkpoint["epoch"],
        "selection": checkpoint["selection"], "config_sha256": checksum(config_path),
        "validation_predictions": str(val_path), "validation_predictions_sha256": checksum(val_path)}
    return {"root": root, "config": config, "checkpoint": checkpoint, "sha256": digest,
        "identity": identity, "normalization": checkpoint["normalization"], "metrics": metrics}


def _source_model(frozen, architecture, device):
    model = e002f.build_model(architecture, frozen["config"]["profile"])
    model.load_state_dict(frozen["checkpoint"]["model_state"], strict=True)
    return model.to(device).eval()


def _source_target_data(outputs_root, normalization):
    base = e001b_run.assemble(SOURCE, "geometry", "B", e001_root=outputs_root / "experiments/E001",
        e001b_root=outputs_root / "experiments/E001b")
    return e003b_run._baselines(base, TARGET, normalization)


def _selected_smoke(path, target, target_data, *, apply=True):
    rows = json.loads(Path(path).read_text()).get("tracks", [])
    selected = {split: {row["locator"] for row in rows
        if row.get("dataset") == target and row.get("split") == split}
        for split in ("training", "validation")}
    if any(not ids or len(ids) > 3 for ids in selected.values()):
        raise ValueError("E002f smoke requires 1–3 explicit target tracks in each split")
    for split, ids in selected.items():
        available = {row["locator"] for row in target_data[split].entries}
        if not ids.issubset(available):
            raise ValueError("E002f smoke selection lies outside the frozen LOKI split")
        if apply:
            target_data[split].entries = [row for row in target_data[split].entries if row["locator"] in ids]
    return selected


def extract_target(args):
    outputs_root, output = args.outputs_root.resolve(), args.output.resolve()
    if args.architecture not in e002f.ARCHITECTURES or output.exists() or not output.is_relative_to(outputs_root / "experiments/E002f/features"):
        raise ValueError("Fresh E002f feature cache required under outputs/experiments/E002f/features")
    if args.smoke != (args.smoke_selection is not None):
        raise ValueError("Target feature smoke requires one explicit LOKI selection")
    frozen = _source_run(args.source_run, args.architecture, outputs_root)
    if frozen["config"].get("smoke_limits") and not args.smoke:
        raise ValueError("A bounded source smoke checkpoint can only feed a bounded target smoke")
    target_data, target_reference = _source_target_data(outputs_root, frozen["normalization"])
    selected = _selected_smoke(args.smoke_selection, TARGET, target_data, apply=False) if args.smoke else None
    output.mkdir(parents=True)
    write_json(output / "status.json", {"status": "incomplete", "failures": [], "held_out_tests": False})
    reader = None
    try:
        reader, profile, datasets, catalog, points, protected = e003c_run._open_target(
            args, frozen | {"run_config": frozen["config"]}, target_data, frozen["normalization"])
        if selected:
            for split, dataset in datasets.items():
                target_data[split].entries = [entry for entry in target_data[split].entries
                    if entry["locator"] in selected[split]]
                dataset.entries = [entry for entry in dataset.entries if entry["locator"] in selected[split]]
        seed_run(0)
        model = _source_model(frozen, args.architecture, args.device)
        cache, counts = {}, {}
        for split, dataset in datasets.items():
            records = e003c_run._extract_fusion(dataset, model, profile, torch.device(args.device), split, output)
            identity = {"experiment": "E002f", "stage": "pre-temporal-fusion", "architecture": args.architecture,
                "source": SOURCE, "target": TARGET, "split": split, "tracks": len(records),
                "checkpoint_sha256": frozen["sha256"], "source_checkpoint": frozen["identity"],
                "source_normalization": frozen["normalization"], "profile": profile,
                "target_catalog": catalog.identity, "target_points": points.identity,
                "smoke_selection": {key: sorted(value) for key, value in selected.items()} if selected else None}
            path = output / f"features-{split}.npz"
            e003b_run._save_cache(path, records, identity)
            e003b_run._load_cache(path, identity)
            counts[split] = len(records)
            cache[split] = identity | {"path": path.name, "sha256": checksum(path),
                "context_slots": int(sum(len(row["targets"]) for row in records))}
            if split == "validation":
                tracks = e003b_run._source_tracks(records)
                save_predictions(output / "source-validation-predictions.npz", tracks)
                metrics = prediction_metrics(tracks, len(CLASSES))
                (output / "loki-validation").mkdir()
                write_json(output / "loki-validation" / "metrics.json", {"split": "validation",
                    "training_source": SOURCE, "target_source": TARGET, "architecture": args.architecture,
                    "selected_epoch": frozen["checkpoint"]["epoch"], "checkpoint_sha256": frozen["sha256"],
                    "normalization_source": SOURCE, "primary": metrics, "dataset_tracks": len(tracks),
                    "prediction_tracks": len(tracks)})
        write_json(output / "feature-cache.json", cache)
        protected.update({str((frozen["root"] / name).resolve()): checksum(frozen["root"] / name)
            for name in ("best.pt", "config.json", "status.json")})
        write_json(output / "protected.json", {"hashes": protected,
            "changed": [path for path, digest in protected.items() if checksum(path) != digest]})
        write_json(output / "status.json", {"status": "complete", "failures": [], "tracks": counts,
            "architecture": args.architecture, "checkpoint_sha256": frozen["sha256"],
            "scope": "full LOKI training/validation; source-normalized pre-temporal features",
            "held_out_tests": False})
        if any(checksum(path) != digest for path, digest in protected.items()):
            raise ValueError("Frozen E002f source or target input changed during extraction")
    except BaseException as error:
        write_json(output / "status.json", {"status": "failed", "failures": [
            {"type": type(error).__name__, "message": str(error)}], "held_out_tests": False})
        raise
    finally:
        if reader is not None:
            reader.close()


class TemporalAdaptationHead(nn.Module):
    """A source temporal backbone, final norm and original source classifier."""

    def __init__(self, temporal, final_norm, head):
        super().__init__()
        self.temporal, self.final_norm, self.head = temporal, final_norm, head
        for module in self.temporal.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.

    def forward(self, inputs, lengths):
        valid = torch.arange(inputs.shape[1], device=inputs.device)[None, :] < lengths.to(inputs.device)[:, None]
        if not torch.isfinite(inputs[valid]).all():
            raise ValueError("Nonfinite cached E002f pre-temporal features")
        x = inputs.masked_fill(~valid[..., None], 0.)
        for block in self.temporal:
            x = block(x, valid)
        return self.head(self.final_norm(x)).masked_fill(~valid[..., None], 0.)


def _load_cache(root, split, architecture, source_sha, normalization):
    if json.loads((root / "status.json").read_text()).get("status") != "complete":
        raise ValueError("Complete E002f feature extraction required")
    manifest = json.loads((root / "feature-cache.json").read_text())
    entry = manifest[split]
    path = root / entry["path"]
    if checksum(path) != entry["sha256"]:
        raise ValueError("E002f feature cache checksum mismatch")
    identity = {key: value for key, value in entry.items() if key not in ("path", "sha256", "context_slots")}
    with np.load(path, allow_pickle=False) as archive:
        actual = json.loads(str(archive["metadata"]))
    if (actual != identity or actual.get("experiment") != "E002f"
            or actual.get("architecture") != architecture or actual.get("checkpoint_sha256") != source_sha
            or actual.get("source_normalization") != normalization or actual.get("split") != split
            or actual.get("target") != TARGET):
        raise ValueError("E002f feature-cache lineage mismatch")
    return e003b_run._load_cache(path, identity), entry


def _adaptation_model(arm, source_model, device):
    if arm == "head":
        model = e003b_run.AdaptationHead()
        model.load_state_dict({"linear.weight": source_model.head.weight.detach().clone(),
            "linear.bias": source_model.head.bias.detach().clone()}, strict=True)
        if not torch.equal(model.linear.weight, source_model.head.weight.cpu()) or not torch.equal(
                model.linear.bias, source_model.head.bias.cpu()):
            raise ValueError("Head control did not copy the original ROAD classifier exactly")
    elif arm == "temporal-head":
        model = TemporalAdaptationHead(source_model.temporal, source_model.final_norm, source_model.head)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError("E002f adaptation supports only head and temporal-head")
    return model.to(device).eval()


def _postnorm_records(records, source_model, device):
    """Derive the frozen-source post-LayerNorm representation for the linear-head control."""
    source_model.eval()
    result = []
    for start in range(0, len(records), 64):
        rows = records[start:start + 64]
        lengths = torch.tensor([len(row["features"]) for row in rows], dtype=torch.long)
        slots = int(lengths.max())
        features = torch.zeros(len(rows), slots, 128, dtype=torch.float32, device=device)
        for index, row in enumerate(rows):
            features[index, :len(row["features"])] = torch.from_numpy(row["features"]).to(device)
        valid = torch.arange(slots, device=device)[None, :] < lengths.to(device)[:, None]
        features.masked_fill_(~valid[..., None], 0.)
        with torch.inference_mode():
            for block in source_model.temporal:
                features = block(features, valid)
            features = source_model.final_norm(features).masked_fill(~valid[..., None], 0.)
        for index, row in enumerate(rows):
            result.append(row | {"features": features[index, :lengths[index]].cpu().numpy().astype(np.float32, copy=True)})
    return result


def adaptation_assembly(args):
    if args.smoke != (args.smoke_selection is not None):
        raise ValueError("Adaptation smoke requires one explicit LOKI selection")
    frozen = _source_run(args.source_run, args.architecture, args.outputs_root.resolve())
    target_data, target_reference = _source_target_data(args.outputs_root.resolve(), frozen["normalization"])
    selected = _selected_smoke(args.smoke_selection, TARGET, target_data) if args.smoke else None
    records, train_meta = _load_cache(args.feature_cache, "training", args.architecture,
        frozen["sha256"], frozen["normalization"])
    val_records, val_meta = _load_cache(args.feature_cache, "validation", args.architecture,
        frozen["sha256"], frozen["normalization"])
    if selected:
        records = [row for row in records if row["locator"] in selected["training"]]
        val_records = [row for row in val_records if row["locator"] in selected["validation"]]
    expected = {split: {entry["locator"] for entry in target_data[split].entries}
        for split in ("training", "validation")}
    for split, rows in (("training", records), ("validation", val_records)):
        if {row["locator"] for row in rows} != expected[split]:
            raise ValueError(f"E002f cached {split} population differs from the frozen LOKI split")
    records, val_records = e003c_run._canonical_records(records), e003c_run._canonical_records(val_records)

    device = torch.device(args.device)
    seed_run(0)
    source_model = _source_model(frozen, args.architecture, device)
    if args.arm == "head":
        records = _postnorm_records(records, source_model, device)
        val_records = _postnorm_records(val_records, source_model, device)
    model = _adaptation_model(args.arm, source_model, device)
    del source_model
    datasets = {"training": e003b_run.CachedTracks(records), "validation": e003b_run.CachedTracks(val_records)}
    initial_tracks = predict(model, e003b_run._batches(datasets["validation"], seed=0), device)
    reference_path = args.feature_cache.resolve() / "source-validation-predictions.npz"
    if not reference_path.is_file():
        raise ValueError("E002f direct LOKI validation predictions are missing")
    metrics = e003c_run._check_initial_predictions(initial_tracks, reference_path, require_all=not args.smoke)
    parameter_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if args.arm == "head" and parameter_names != ["linear.weight", "linear.bias"]:
        raise ValueError("Head control must train only the 516 original source-head parameters")
    if args.arm == "temporal-head" and (not any(name.startswith("temporal.") for name in parameter_names)
            or not all(name.startswith(("temporal.", "final_norm.", "head.")) for name in parameter_names)):
        raise ValueError("Temporal-head may train only the temporal backbone, final norm and source head")

    def data_for(dataset, split):
        if dataset != TARGET or split not in ("training", "validation", "validation-smoke"):
            raise ValueError("E002f adaptation exposes only LOKI train/validation cached features")
        return datasets["validation" if split == "validation-smoke" else split]

    steps = math.ceil(len(datasets["training"]) / SETTINGS["batch_size"])
    return {"config": {"experiment": "E002f", "original_source": SOURCE, "target": TARGET,
            "architecture": args.architecture, "arm": args.arm,
            "initialization": "source-checkpoint", "initialization_checkpoint": frozen["identity"],
            "source_checkpoint_sha256": frozen["sha256"], "normalization_source": SOURCE,
            "normalization": frozen["normalization"], "dropout_enabled": False, "optimizer_reset": True,
            "training_order": "locator-sorted before seeded shuffle", "initial_validation": metrics,
            "population": {"training": len(datasets["training"]), "validation": len(datasets["validation"])},
            "feature_cache": {"training": train_meta, "validation": val_meta},
            "trainable_parameters": parameter_names, "held_out_tests": False,
            "smoke_selection": {key: sorted(value) for key, value in selected.items()} if selected else None},
        "normalization": frozen["normalization"], "datasets": {TARGET: target_reference},
        "build_model": lambda: model, "dataset": data_for, "batches": e003b_run._batches,
        "conditions": lambda metadata, arrays, track, dataset: ({}, {}),
        "evaluation_splits": [(TARGET, "validation-smoke" if args.smoke else "validation")],
        "optimizer_factory": e003c_run.optimizer_factory(args.arm), "initial_candidate": True,
        "epoch_operation": e003c_run._warmup_epoch(args.arm, SETTINGS["warmup_epochs"] * steps),
        "log_learning_rate": True, "custom_smoke": True,
        "run_name": f"e002f-road-waymo-to-loki-{args.architecture}-{args.arm}-s0" + ("-smoke" if args.smoke else "")}


def adapt(args):
    if args.architecture not in e002f.ARCHITECTURES or args.arm not in ("head", "temporal-head"):
        raise ValueError("Unsupported E002f architecture/adaptation arm")
    if args.device not in ("cuda", "cpu") or (args.device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested adaptation device is unavailable")
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.outputs_root.resolve() / "experiments/E002f/runs"):
        raise ValueError("Fresh E002f adaptation attempt required under outputs/experiments/E002f/runs")
    torch.set_num_threads(2)
    if json.loads((Path(args.source_run).resolve() / "config.json").read_text()).get("smoke_limits") and not args.smoke:
        raise ValueError("A bounded source smoke checkpoint can only feed bounded adaptation smokes")
    return shared_run.run_attempt(TARGET, "temporal-adaptation", f"{args.architecture}-{args.arm}", output,
        args.device, args.smoke, settings=SETTINGS, assemble=lambda *_: adaptation_assembly(args),
        classes=CLASSES, axes={})


def calibrate(args):
    reference_path = args.reference_profile.resolve()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(args.outputs_root.resolve() / "experiments/E002f/calibration"):
        raise ValueError("E002f calibration must be a fresh file under outputs/experiments/E002f/calibration")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ValueError("E002f resource calibration requires CUDA")
    reference = json.loads(reference_path.read_text())
    if reference.get("status") != "complete" or reference.get("experiment") != "E002d":
        raise ValueError("Completed E002d input-resource profile required")
    profile = e002a_run.validate_profile(reference.get("profile", {}))
    if profile["max_tracks"] != 64 or profile["temporal_cells"] != 665856:
        raise ValueError("E002f inherits the accepted 64-track/102-frame E002d execution bound")
    torch.set_num_threads(2)
    determinism = seed_run(0)
    rows = []
    for architecture in e002f.ARCHITECTURES:
        model = e002f.build_model(architecture, profile).cuda()
        params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        if params != 552196:
            raise ValueError(f"Unexpected E002f trainable full-model parameter count: {params}")
        batch = {"geometry": torch.zeros(64, 102, 20, device="cuda"),
            "flags": torch.tensor([1., 1., 0.], device="cuda").view(1, 1, 3).expand(64, 102, 3).clone(),
            "rgb": torch.zeros(64, 102, 768, device="cuda"),
            "points": [torch.zeros(0, 68) for _ in range(64 * 102)],
            "lengths": torch.full((64,), 102, dtype=torch.long),
            "targets": torch.arange(102, dtype=torch.long).remainder(4).view(1, 102).expand(64, 102),
            "gt_valid": torch.ones(64, 102, dtype=torch.bool),
            "padding_mask": torch.zeros(64, 102, dtype=torch.bool)}
        optimizer = torch.optim.AdamW(model.parameters(), **e002a_run.SETTINGS["optimizer"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = track_cross_entropy(logits, batch["targets"].cuda(), batch["gt_valid"].cuda(),
            batch["padding_mask"].cuda()).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated()
        ceiling = int(torch.cuda.get_device_properties("cuda").total_memory * .78)
        if not torch.isfinite(loss) or not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise ValueError(f"Nonfinite full-bound E002f calibration step: {architecture}")
        if peak > ceiling:
            raise ValueError(f"E002f {architecture} exceeded 78% of CUDA memory during a 64×102 step")
        rows.append({"architecture": architecture, "tracks": 64, "slots_per_track": 102,
            "temporal_cells": 665856, "trainable_parameters": params,
            "step_seconds": elapsed, "peak_allocated_bytes": peak, "memory_ceiling_bytes": ceiling})
        del model, batch, optimizer, logits, loss
        torch.cuda.empty_cache()
    record = {"status": "complete", "experiment": "E002f", "device": "cuda",
        "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "profile": profile, "code": resource_code(), "model_coverage": list(e002f.ARCHITECTURES),
        "model_specs": {f"{ARM}/{name}": e002f.model_spec(name) for name in e002f.ARCHITECTURES},
        "point_storage_protocol": e002a_run.POINT_PROTOCOL,
        "reference_profile": str(reference_path), "reference_profile_sha256": checksum(reference_path),
        "determinism": determinism, "workloads": rows,
        "note": "E002d numeric input limits inherited; both E002f models passed a full 64×102 temporal forward/backward/optimizer step."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, record)
    return record


def _common_target(parser):
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--rgb-preparation", type=Path,
        default=Path("outputs/experiments/E002a/rgb-cache-preparation-03/cache.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-selection", type=Path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    source = commands.add_parser("source", help="train a ROAD source architecture")
    source.add_argument("--architecture", choices=e002f.ARCHITECTURES, required=True)
    source.add_argument("--profile", type=Path, default=Path("outputs/experiments/E002f/calibration/full-batch/profile.json"))
    source.add_argument("--output", type=Path, required=True)
    source.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    source.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    source.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    source.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    source.add_argument("--rgb-preparation", type=Path,
        default=Path("outputs/experiments/E002a/rgb-cache-preparation-03/cache.json"))
    source.add_argument("--indexed-point-cache", type=Path)
    source.add_argument("--point-workers", type=int, default=e002a_run.DEFAULT_POINT_WORKERS)
    source.add_argument("--smoke", action="store_true")
    source.add_argument("--smoke-selection", type=Path)
    source.set_defaults(func=train_source, source=SOURCE)
    extract = commands.add_parser("extract", help="extract architecture-specific LOKI fusion features")
    extract.add_argument("--architecture", choices=e002f.ARCHITECTURES, required=True)
    extract.add_argument("--source-run", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    _common_target(extract)
    extract.set_defaults(func=extract_target)
    adaptation = commands.add_parser("adapt", help="fit one LOKI cached-feature adaptation arm")
    adaptation.add_argument("--architecture", choices=e002f.ARCHITECTURES, required=True)
    adaptation.add_argument("--arm", choices=("head", "temporal-head"), required=True)
    adaptation.add_argument("--source-run", type=Path, required=True)
    adaptation.add_argument("--feature-cache", type=Path, required=True)
    adaptation.add_argument("--output", type=Path, required=True)
    _common_target(adaptation)
    adaptation.set_defaults(func=adapt)
    args = parser.parse_args()
    args.func(args)


def calibrate_main():
    parser = argparse.ArgumentParser(description="Run E002f full-temporal-bound GPU resource checks.")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--reference-profile", type=Path, default=SOURCE_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    calibrate(parser.parse_args())
