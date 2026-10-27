"""Collect and verify saved E002a/E002b source-validation results, without inference."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pedestrian_behavior.evaluation import load_predictions, prediction_metrics, stratified_metrics


EXPECTED = {"E002a": 8, "E002b": 18}
SOURCES = ("road-waymo", "loki")
ARMS = ("g", "g-rgb", "g-dditr", "g-rgb-dditr")
ARM_LABELS = {"g": "Geometry", "g-rgb": "Geometry + RGB",
              "g-dditr": "Geometry + LiDAR", "g-rgb-dditr": "Geometry + RGB + LiDAR"}
CLASS_LABELS = ("Moving", "Stopped", "Waiting to cross", "Crossing the road")
ALIGNMENT_FIELDS = ("locator", "archive", "group", "bbox_archive", "bbox_sha256",
                    "native_extent_us", "source_times_us", "grid_times_us")


def _read_json(path):
    return json.loads(Path(path).read_text())


def _canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_row(history, provenance, status):
    selection = provenance.get("selection", {})
    epoch = selection.get("epoch", status.get("selected_checkpoint_epoch"))
    selected = [row for row in history if row.get("selected")]
    if epoch is None and selected:
        epoch = selected[-1]["epoch"]
    if type(epoch) is not int or epoch < 1:
        raise ValueError("Completed checkpoint lacks selected epoch")
    rows = [row for row in history if row.get("epoch") == epoch]
    if len(rows) != 1:
        raise ValueError(f"Selected epoch {epoch} is absent or duplicated in history")
    if selected and selected[-1]["epoch"] != epoch:
        raise ValueError("History's last selected epoch disagrees with checkpoint provenance")
    return rows[0]


def _metric_equal(expected, actual, path="metrics", *, rtol=1e-8, atol=1e-10):
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            raise ValueError(f"{path}: metric keys differ")
        for key in expected:
            _metric_equal(expected[key], actual[key], f"{path}.{key}", rtol=rtol, atol=atol)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            raise ValueError(f"{path}: metric lengths differ")
        for index, (a, b) in enumerate(zip(expected, actual)):
            _metric_equal(a, b, f"{path}[{index}]", rtol=rtol, atol=atol)
        return
    if expected is None or actual is None:
        if expected is not actual:
            raise ValueError(f"{path}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=rtol, abs_tol=atol):
            raise ValueError(f"{path}: {expected!r} != {actual!r}")
        return
    if expected != actual:
        raise ValueError(f"{path}: {expected!r} != {actual!r}")


def _model_variant(experiment, config):
    if experiment == "E002a":
        return "base"
    value = config.get("model_variant") or config.get("variant")
    if value not in ("norm", "temporal-small", "branches-small"):
        raise ValueError(f"Unknown E002b variant: {value!r}")
    return value


def _checkpoint_identity(run, experiment, config, provenance):
    if provenance.get("kind") == "checkpoint-evaluation":
        original = provenance.get("origin_training_run", {})
        digest = provenance.get("checkpoint_sha256") or original.get("best_checkpoint_sha256")
        if not digest:
            raise ValueError(f"Missing original selected checkpoint hash for {run}")
        return digest, original
    checkpoint = run / "best.pt"
    if not checkpoint.is_file():
        raise ValueError(f"Missing selected checkpoint file: {checkpoint}")
    return _file_hash(checkpoint), {}


def _dataset_identity(source, provenance):
    datasets = provenance.get("datasets", {})
    dataset = datasets.get(source)
    if dataset:
        return {key: dataset.get(key) for key in ("setup_sha256", "collection_sha256", "policy_sha256", "native_policy")}
    identity = provenance.get("input_identity", {})
    frozen = identity.get(f"frozen_{source.replace('-', '_')}_source", {})
    population = identity.get("feature_population", {})
    return {"setup_sha256": frozen.get("setup_sha256"),
            "collection_sha256": frozen.get("collection_manifest_sha256", population.get("manifest_sha256")),
            "policy_sha256": frozen.get("bbox_policy_sha256")}


def _normalization(source, config, provenance):
    statistics = provenance.get("source_statistics")
    if statistics:
        return statistics
    if config.get("source_normalization"):
        return config["source_normalization"]
    return provenance.get("input_identity", {}).get("normalization")


def _run_record(experiment, config_path):
    config_path = Path(config_path)
    run = config_path.parent
    config = _read_json(config_path)
    status = _read_json(run / "status.json")
    if status.get("status") != "complete":
        return None
    if config.get("held_out_tests") is True or status.get("held_out_tests") is True:
        raise ValueError(f"Held-out results are outside the requested comparison: {run}")
    source, arm = config.get("source"), config.get("arm")
    if source not in SOURCES or arm not in (ARMS if experiment == "E002a" else ARMS[:3]):
        raise ValueError(f"Unexpected source/arm in completed result: {run}")
    split = f"{source}-validation"
    metrics_path = run / split / "metrics.json"
    predictions_path = run / split / "predictions.npz"
    history_path, provenance_path = run / "history.json", run / "provenance.json"
    if not all(path.is_file() for path in (metrics_path, predictions_path, history_path, provenance_path)):
        raise ValueError(f"Incomplete completed result files: {run}")
    metrics, history, provenance = (_read_json(metrics_path), _read_json(history_path), _read_json(provenance_path))
    if metrics.get("split") != split:
        raise ValueError(f"Wrong evaluation split in {metrics_path}: {metrics.get('split')!r}")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Missing epoch history: {history_path}")
    selection = _selected_row(history, provenance, status)
    selected_epoch = int(selection["epoch"])
    if provenance.get("selected_checkpoint_epoch", selected_epoch) != selected_epoch:
        raise ValueError(f"Selected checkpoint epoch mismatch for {run}")
    if status.get("selected_checkpoint_epoch", selected_epoch) != selected_epoch:
        raise ValueError(f"Status selected checkpoint epoch mismatch for {run}")
    primary = metrics.get("primary")
    if not primary or "raw" not in primary or "track_weighted" not in primary:
        raise ValueError(f"Missing primary raw/track-weighted metrics in {metrics_path}")
    _metric_equal(selection.get("validation_f1"), provenance.get("selection", {}).get("f1"), "selected_f1")
    _metric_equal(selection.get("validation_ce"), provenance.get("selection", {}).get("ce"), "selected_ce")
    _metric_equal(primary["track_weighted"]["macro_f1"], provenance.get("selection", {}).get("f1"), "checkpoint_f1")
    _metric_equal(primary["track_ce"], provenance.get("selection", {}).get("ce"), "checkpoint_ce")
    if provenance.get("kind") == "checkpoint-evaluation":
        if status.get("kind") != "checkpoint-evaluation" or provenance.get("training_state") != "interrupted":
            raise ValueError(f"Recovery provenance is incomplete or mislabeled: {run}")
        if status.get("training_completed_epochs") != len(history):
            raise ValueError(f"Recovery epoch count does not match copied original history: {run}")
        parent_status = _read_json(run.parent / "status.json")
        if parent_status.get("status") != "complete" or parent_status.get("held_out_tests") is not False:
            raise ValueError(f"Recovery parent is incomplete or held-out: {run.parent}")
    elif status.get("kind") != "exploratory-source-validation":
        raise ValueError(f"Unexpected result kind for {run}: {status.get('kind')!r}")
    checkpoint_hash, lineage = _checkpoint_identity(run, experiment, config, provenance)
    variant = _model_variant(experiment, config)
    selected_by_f1 = max(history, key=lambda row: (row.get("validation_f1", -math.inf), -row.get("validation_ce", math.inf)))
    first23 = [row for row in history if int(row["epoch"]) <= 23]
    best23 = max(first23, key=lambda row: (row.get("validation_f1", -math.inf), -row.get("validation_ce", math.inf)))
    min_ce = min(history, key=lambda row: (row.get("validation_ce", math.inf), row["epoch"]))
    settings = config.get("settings", {})
    model = config.get("model", {})
    normalization = _normalization(source, config, provenance)
    dataset_identity = _dataset_identity(source, provenance)
    row = {
        "experiment": experiment, "source": source, "arm": arm, "variant": variant,
        "label": ARM_LABELS[arm] + (f" / {variant}" if experiment == "E002b" else ""),
        "run_path": str(run), "split": split, "status": status.get("status"),
        "kind": status.get("kind"), "seed": provenance.get("seed", settings.get("seed")),
        "fusion_protocol": config.get("fusion_protocol"),
        "availability": model.get("availability", []),
        "model_spec": model,
        "profile_sha256": config.get("profile_sha256", provenance.get("profile_sha256")),
        "point_storage_protocol": config.get("point_storage_protocol"),
        "feature_supply": config.get("feature_supply"),
        "frozen_feature_identity": config.get("frozen_features"),
        "settings": settings,
        "parameters": provenance.get("parameters"),
        "trainable_parameters": provenance.get("trainable_parameters"),
        "normalization_sha256": _canonical_hash(normalization) if normalization is not None else None,
        "normalization": normalization,
        "dataset_identity": dataset_identity,
        "dataset_identity_sha256": _canonical_hash(dataset_identity),
        "selection_epoch": selected_epoch,
        "selection_f1": primary["track_weighted"]["macro_f1"],
        "selection_ce": primary["track_ce"],
        "selected_train_ce": selection.get("training_ce"),
        "selected_train_f1": selection.get("training_f1"),
        "last_epoch": int(history[-1]["epoch"]),
        "last_train_ce": history[-1].get("training_ce"),
        "last_train_f1": history[-1].get("training_f1"),
        "last_validation_ce": history[-1].get("validation_ce"),
        "last_validation_f1": history[-1].get("validation_f1"),
        "last_stopping_reason": history[-1].get("stopping_reason", provenance.get("stopping_reason")),
        "min_validation_ce": min_ce.get("validation_ce"),
        "min_validation_ce_epoch": min_ce["epoch"],
        "best_validation_f1_epoch": selected_by_f1["epoch"],
        "best_validation_f1": selected_by_f1.get("validation_f1"),
        "best_f1_through_epoch_23": best23.get("validation_f1"),
        "best_f1_through_epoch_23_epoch": best23["epoch"],
        "training_state": provenance.get("training_state", "complete"),
        "training_completed_epochs": provenance.get("training_completed_epochs", len(history)),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_lineage": lineage,
        "wandb": provenance.get("wandb", provenance.get("evaluation_wandb", {})),
        "metrics_path": str(metrics_path), "predictions_path": str(predictions_path),
        "primary": primary, "strata": metrics.get("strata", {}),
        "history": history,
        "prediction_count": int(primary["denominators"]["tracks"]),
        "denominators": primary["denominators"],
        "classes": config.get("classes", list(CLASS_LABELS)),
    }
    return row


def _array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.dtype).encode() + b"\0" + array.tobytes()).hexdigest()


def _prediction_signature(tracks):
    signatures = []
    for track in tracks:
        metadata = {key: track.get(key) for key in ALIGNMENT_FIELDS if key in track}
        signatures.append({
            "metadata": metadata,
            "targets": _array_hash(track["targets"]),
            "gt_valid": _array_hash(track["gt_valid"]),
            "conditions": {key: _array_hash(value) for key, value in track.get("conditions", {}).items()},
            "source_flags": {key: _array_hash(value) for key, value in track.get("source_flags", {}).items()},
        })
    return signatures


def _verify_alignment(records, signatures_by_record):
    by_source = {}
    for row in records:
        by_source.setdefault(row["source"], []).append(row)
    integrity = {}
    for source, rows in by_source.items():
        baseline = signatures_by_record[rows[0]["run_path"]]
        baseline_locators = [item["metadata"].get("locator") for item in baseline]
        shared_conditions = set.intersection(*(set(signatures_by_record[r["run_path"]][0]["conditions"]) for r in rows))
        all_conditions = set.union(*(set(signatures_by_record[r["run_path"]][0]["conditions"]) for r in rows))
        all_flags = set.union(*(set(signatures_by_record[r["run_path"]][0]["source_flags"]) for r in rows))
        for row in rows[1:]:
            candidate = signatures_by_record[row["run_path"]]
            if len(candidate) != len(baseline):
                raise ValueError(f"{source}: prediction track count differs across runs")
            if [item["metadata"].get("locator") for item in candidate] != baseline_locators:
                raise ValueError(f"{source}: prediction order/locators differ in {row['run_path']}")
            for index, (reference, actual) in enumerate(zip(baseline, candidate)):
                for field in ALIGNMENT_FIELDS:
                    if reference["metadata"].get(field) != actual["metadata"].get(field):
                        raise ValueError(f"{source}: {field} misalignment at track {index} in {row['run_path']}")
                for field in ("targets", "gt_valid"):
                    if reference[field] != actual[field]:
                        raise ValueError(f"{source}: accepted-GT {field} misalignment at track {index} in {row['run_path']}")
        for index in range(len(baseline)):
            for family, label in (("conditions", "condition"), ("source_flags", "modality flag")):
                keys = set.union(*(set(signatures_by_record[r["run_path"]][index][family]) for r in rows))
                for key in keys:
                    values = {signatures_by_record[r["run_path"]][index][family][key]
                              for r in rows if key in signatures_by_record[r["run_path"]][index][family]}
                    if len(values) > 1:
                        raise ValueError(f"{source}: {label} {key} differs at track {index}")
        integrity[source] = {
            "models": len(rows), "tracks": len(baseline),
            "common_strata_axes": sorted(shared_conditions),
            "available_strata_axes": sorted(all_conditions),
            "available_modality_flags": sorted(all_flags),
            "ordered_locator_sha256": _canonical_hash(baseline_locators),
            "target_gt_times_alignment": "exact across all completed models",
            "condition_alignment": "exact on shared axes; absent axes are not compared",
            "modality_flags": "exact wherever the flag is present in both models",
        }
    return integrity


def _verify_source_contracts(records):
    checks = {}
    for source in SOURCES:
        rows = [row for row in records if row["source"] == source]
        if not rows:
            raise ValueError(f"Missing source: {source}")
        if len({row["dataset_identity_sha256"] for row in rows if row["kind"] != "checkpoint-evaluation"}) != 1:
            raise ValueError(f"{source}: frozen dataset identity differs across the original runs")
        for row in rows:
            if row["seed"] != 0:
                raise ValueError(f"Unexpected seed in {row['run_path']}: {row['seed']!r}")
            if row["fusion_protocol"] != "enabled-branches-v2":
                raise ValueError(f"Unexpected fusion protocol in {row['run_path']}")
        counts = {(r["denominators"]["tracks"], r["denominators"]["gt_frames"],
                   tuple(r["denominators"]["class_frames"])) for r in rows}
        if len(counts) != 1:
            raise ValueError(f"{source}: class/support denominators differ across models")
        normalizations = {row["normalization_sha256"] for row in rows if row["normalization_sha256"]}
        if len(normalizations) != 1:
            raise ValueError(f"{source}: source normalization differs across selected checkpoints")
        settings = {_canonical_hash(row["settings"]) for row in rows}
        if len(settings) != 1:
            raise ValueError(f"{source}: shared training recipe differs across selected checkpoints")
        checks[source] = {
            "run_count": len(rows), "tracks": rows[0]["denominators"]["tracks"],
            "groups": rows[0]["denominators"]["groups"],
            "context_frames": rows[0]["denominators"]["context_frames"],
            "gt_frames": rows[0]["denominators"]["gt_frames"],
            "class_frames": rows[0]["denominators"]["class_frames"],
            "class_tracks": rows[0]["denominators"]["class_tracks"],
            "settings_sha256": sorted(settings),
            "normalization_sha256": sorted(normalizations),
            "source_population_sha256": sorted({r["dataset_identity_sha256"] for r in rows if r["kind"] != "checkpoint-evaluation"}),
        }
    return checks


def collect(outputs_root):
    outputs_root = Path(outputs_root)
    records = []
    for experiment, expected in EXPECTED.items():
        run_root = outputs_root / "experiments" / experiment / "runs"
        for config_path in sorted(run_root.rglob("config.json")):
            if any("smoke" in part.lower() for part in config_path.parts):
                continue
            row = _run_record(experiment, config_path)
            if row is not None:
                records.append(row)
        actual = sum(row["experiment"] == experiment for row in records)
        if actual != expected:
            raise ValueError(f"Expected {expected} complete unique {experiment} checkpoints, found {actual}")
    ids = [(r["experiment"], r["source"], r["arm"], r["variant"]) for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source/arm/variant results would be treated as replicates")
    if len({r["checkpoint_sha256"] for r in records}) != len(records):
        raise ValueError("Selected checkpoint SHA-256 values are not unique")
    for experiment, arms in (("E002a", set(ARMS)), ("E002b", set(ARMS[:3]))):
        for source in SOURCES:
            matching = [r for r in records if r["experiment"] == experiment and r["source"] == source]
            if {r["arm"] for r in matching} != arms:
                raise ValueError(f"Missing {experiment}/{source} arm coverage")
            if experiment == "E002b" and any(sum(r["arm"] == arm for r in matching) != 3 for arm in arms):
                raise ValueError(f"{experiment}/{source} must have all three frozen variants per arm")
    return records


def _verify_and_recompute(records):
    signatures = {}
    for row in records:
        saved = _read_json(row["metrics_path"])
        tracks = load_predictions(row["predictions_path"])
        if len(tracks) != row["prediction_count"]:
            raise ValueError(f"Prediction count differs from metrics in {row['run_path']}")
        primary = prediction_metrics(tracks, 4)
        _metric_equal(saved["primary"], primary, "primary")
        axes = {axis: tuple(bins) for axis, bins in saved.get("strata", {}).items()}
        strata = stratified_metrics(tracks, 4, axes)
        _metric_equal(saved.get("strata", {}), strata, "strata")
        signatures[row["run_path"]] = _prediction_signature(tracks)
        row["metric_reconstruction"] = "exact within rtol=1e-8, atol=1e-10 from pickle-free NPZ"
        row["prediction_identity"] = "pickle-free; logits float32; GT/time/condition alignment checked"
        del tracks
    return signatures


def _flat_primary(row):
    primary = row["primary"]
    raw, weighted = primary["raw"], primary["track_weighted"]
    return {
        "experiment": row["experiment"], "source": row["source"], "arm": row["arm"],
        "variant": row["variant"], "label": row["label"], "seed": row["seed"],
        "selected_epoch": row["selection_epoch"], "selected_train_ce": row["selected_train_ce"],
        "selected_train_f1": row["selected_train_f1"], "last_epoch": row["last_epoch"],
        "last_train_ce": row["last_train_ce"], "last_train_f1": row["last_train_f1"],
        "last_validation_ce": row["last_validation_ce"], "last_validation_f1": row["last_validation_f1"],
        "min_validation_ce": row["min_validation_ce"], "min_validation_ce_epoch": row["min_validation_ce_epoch"],
        "best_f1_through_epoch_23": row["best_f1_through_epoch_23"],
        "best_f1_through_epoch_23_epoch": row["best_f1_through_epoch_23_epoch"],
        "validation_track_ce": primary["track_ce"],
        "validation_track_weighted_macro_f1": weighted["macro_f1"],
        "validation_track_weighted_accuracy": weighted["accuracy"],
        "validation_raw_macro_f1": raw["macro_f1"], "validation_raw_accuracy": raw["accuracy"],
        "tracks": primary["denominators"]["tracks"], "groups": primary["denominators"]["groups"],
        "context_frames": primary["denominators"]["context_frames"],
        "gt_frames": primary["denominators"]["gt_frames"],
        "class_frames": json.dumps(primary["denominators"]["class_frames"]),
        "checkpoint_sha256": row["checkpoint_sha256"], "run_path": row["run_path"],
    }


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_tables(output, records):
    primary_rows = [_flat_primary(row) for row in records]
    _write_csv(output / "primary-metrics.csv", primary_rows)
    class_rows = []
    strata_rows = []
    for row in records:
        for mode in ("raw", "track_weighted"):
            metrics = row["primary"][mode]
            for index, label in enumerate(CLASS_LABELS):
                class_rows.append({"experiment": row["experiment"], "source": row["source"],
                    "arm": row["arm"], "variant": row["variant"], "label": row["label"],
                    "mode": mode, "class_index": index, "class_name": label,
                    "support": metrics["support"][index], "precision": metrics["precision"][index],
                    "recall": metrics["recall"][index], "f1": metrics["f1"][index],
                    "checkpoint_sha256": row["checkpoint_sha256"]})
        for axis, bins in row["strata"].items():
            for bin_label, item in bins.items():
                for mode in ("raw", "track_weighted"):
                    metric = item[mode]
                    denoms = item["denominators"]
                    for index, label in enumerate(CLASS_LABELS):
                        strata_rows.append({"experiment": row["experiment"], "source": row["source"],
                            "arm": row["arm"], "variant": row["variant"], "label": row["label"],
                            "axis": axis, "bin": bin_label, "mode": mode,
                            "class_index": index, "class_name": label,
                            "support": metric["support"][index], "precision": metric["precision"][index],
                            "recall": metric["recall"][index], "f1": metric["f1"][index],
                            "macro_f1": metric["macro_f1"], "accuracy": metric["accuracy"],
                            "tracks": denoms["tracks"], "groups": denoms["groups"],
                            "gt_frames": denoms["gt_frames"], "checkpoint_sha256": row["checkpoint_sha256"]})
    _write_csv(output / "per-class-metrics.csv", class_rows)
    _write_csv(output / "strata-metrics.csv", strata_rows)


def _plot_primary(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=True)
    colors = {"E002a": "#586ba4", "E002b": "#e07a5f"}
    for ax, source in zip(axes, SOURCES):
        rows = [r for r in records if r["source"] == source]
        rows.sort(key=lambda r: (r["experiment"], ARMS.index(r["arm"]),
                                ("base", "norm", "temporal-small", "branches-small").index(r["variant"])))
        labels = [f"{r['experiment']} {r['arm']}" + (f"/{r['variant']}" if r["experiment"] == "E002b" else "") for r in rows]
        values = [r["primary"]["track_weighted"]["macro_f1"] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, values, color=[colors[r["experiment"]] for r in rows], alpha=.88)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title(source)
        ax.grid(axis="x", alpha=.25)
        for yi, value in zip(y, values):
            ax.text(value + .01, yi, f"{value:.3f}", va="center", fontsize=8)
    fig.supxlabel("Validation track-weighted macro-F1")
    fig.suptitle("Selected source-validation checkpoint; one seed per model")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"primary-macro-f1.{ext}", dpi=160)
    plt.close(fig)


def _plot_per_class(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharex=True, sharey=True)
    for ax, source in zip(axes, SOURCES):
        rows = [r for r in records if r["source"] == source]
        rows.sort(key=lambda r: (r["experiment"], ARMS.index(r["arm"]),
                                ("base", "norm", "temporal-small", "branches-small").index(r["variant"])))
        values = np.asarray([r["primary"]["track_weighted"]["f1"] for r in rows], dtype=float)
        labels = [f"{r['experiment']} {r['arm']}" + (f"/{r['variant']}" if r["experiment"] == "E002b" else "") for r in rows]
        image = ax.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.set_xticks(np.arange(4), ("Moving", "Stopped", "Waiting", "Crossing"), rotation=25, ha="right")
        ax.set_title(source)
        for i in range(len(rows)):
            for j in range(4):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center",
                        color="white" if values[i, j] < .55 else "black", fontsize=7)
    fig.colorbar(image, ax=axes, label="Validation track-weighted class F1", shrink=.82)
    fig.suptitle("Selected checkpoints · all four supported classes")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"per-class-f1.{ext}", dpi=160)
    plt.close(fig)


def _plot_loss_curves(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex="col", sharey="row")
    palette = {"g": "#4063d8", "g-rgb": "#df8544", "g-dditr": "#379b6b", "g-rgb-dditr": "#9c5ba5"}
    styles = {"base": "-", "norm": "-", "temporal-small": "--", "branches-small": ":"}
    for row_index, source in enumerate(SOURCES):
        for column_index, experiment in enumerate(("E002a", "E002b")):
            ax = axes[row_index, column_index]
            subset = [r for r in records if r["source"] == source and r["experiment"] == experiment]
            for record in subset:
                history = record["history"]
                epochs = [item["epoch"] for item in history]
                color, style = palette[record["arm"]], styles[record["variant"]]
                model_label = record["arm"] + (f"/{record['variant']}" if experiment == "E002b" else "")
                ax.plot(epochs, [item.get("training_ce") for item in history], color=color,
                        linestyle=style, alpha=.32, linewidth=1.1)
                ax.plot(epochs, [item.get("validation_ce") for item in history], color=color,
                        linestyle=style, alpha=.9, linewidth=1.7, label=model_label)
                ax.scatter([record["selection_epoch"]], [record["selection_ce"]], color=color,
                           marker="o", edgecolor="black", linewidth=.35, s=20, zorder=3)
            ax.set_title(f"{source} · {experiment}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Equal-track cross-entropy")
            ax.grid(alpha=.2)
            if subset:
                handles, labels = ax.get_legend_handles_labels()
                unique = dict(zip(labels, handles))
                ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=2, frameon=False)
    fig.suptitle("Training CE (faint) and validation CE (solid/colored), selected validation checkpoint marked")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"loss-curves.{ext}", dpi=160)
    plt.close(fig)


def summarize(outputs_root, output):
    outputs_root, output = Path(outputs_root), Path(output)
    records = collect(outputs_root)
    signatures = _verify_and_recompute(records)
    alignment = _verify_alignment(records, signatures)
    contracts = _verify_source_contracts(records)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing review artifacts: {output}")
    output.mkdir(parents=True)
    _write_tables(output, records)
    _plot_primary(output, records)
    _plot_per_class(output, records)
    _plot_loss_curves(output, records)
    summary = {
        "schema": "e002ab-results-review-v1",
        "scope": "26 unique, selected source-validation checkpoints; no held-out evaluation or inference",
        "created_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "expected_unique_checkpoints": EXPECTED,
        "recovery_note": "The three LOKI E002b g-dditr rows reuse selected checkpoints from one interrupted 23-epoch training group; they are not new training replicates.",
        "metric_selection": "last history row marked selected; tie-breaking and saved checkpoint selection verified against provenance and validation metrics",
        "metric_reconstruction": "All 26 primary and saved-strata metrics recomputed from pickle-free NPZ predictions without model inference.",
        "source_split_integrity": contracts,
        "prediction_alignment_integrity": alignment,
        "normalization_sha256_by_source": {source: sorted({r["normalization_sha256"] for r in records if r["source"] == source and r["normalization_sha256"]}) for source in SOURCES},
        "results": records,
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary
