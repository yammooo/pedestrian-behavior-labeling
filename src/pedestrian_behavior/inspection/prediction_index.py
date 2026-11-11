"""Prediction-only browsing and per-track scores; no native decoding or inference."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import confusion_metrics, load_predictions

INDEX_VERSION = 2
METRICS = ("accuracy", "ce", "macro_f1", "confusion_rate")
SPLITS = ("test", "validation")
VALIDATION_EXPERIMENTS = ("E002c", "E002d", "E002e", "E002f", "E003a", "E003b", "E003c")
VALIDATION_CLASSES = ["MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING"]


def _run_label(config, experiment):
    if experiment.startswith("E001"):
        return ("BiLSTM" if config.get("variant") == "B" else "MLP",
                config.get("configuration", "unknown"), None)
    if experiment == "E002a":
        return "Transformer", config.get("arm", config.get("configuration", "unknown")), None
    variant = config.get("model_variant", config.get("variant", "unknown"))
    return "Transformer", config.get("arm", config.get("configuration", "unknown")), str(variant)


def _validation_record(directory, experiment, config, provenance, prediction_paths, *, role, label):
    """Describe one completed validation prediction set from a known E002/E003 layout."""
    settings = config.get("settings") or {}
    model = config.get("model") or settings.get("model") or {}
    model = model if isinstance(model, dict) else {}
    architecture = config.get("architecture") or model.get("architecture") or config.get("variant")
    architecture = architecture if isinstance(architecture, str) else ""
    variant = config.get("model_variant", config.get("variant"))
    capacity = None
    temporal = model.get("temporal") or {}
    if isinstance(temporal, dict) and temporal:
        capacity = " · ".join(str(value) for value in (
            f"{temporal.get('width')} width" if temporal.get("width") is not None else None,
            f"{temporal.get('heads')} heads" if temporal.get("heads") is not None else None,
            f"{temporal.get('layers')} layers" if temporal.get("layers") is not None else None,
            f"FFN {temporal.get('ffn')}" if temporal.get("ffn") is not None else None) if value)
    if architecture in ("bilstm-96", "transformer-1layer"):
        capacity = ("hidden 96/direction · 192→128 projection" if architecture == "bilstm-96"
                    else "128 width · 4 heads · 1 layer · FFN 512")
        architecture = "BiLSTM" if architecture == "bilstm-96" else "Transformer"
        variant = None
    elif "bilstm" in architecture.lower():
        architecture = "BiLSTM" + (" · " + architecture.split("-", 1)[1] if "-" in architecture else "")
    elif "transformer" in architecture.lower():
        architecture = "Transformer" + (" · " + architecture.split("-", 1)[1] if "-" in architecture else "")
    elif experiment == "E003b" or experiment == "E003c":
        architecture, capacity = "Transformer", "source temporal-small model"
        if isinstance(config.get("head"), dict) and config["head"].get("shape"):
            capacity = "source temporal-small model · head 128→4"
    elif temporal:
        architecture = "Transformer"
    else:
        architecture = "Transformer"
    if capacity is None and variant == "temporal-small":
        capacity = "128 width · 4 heads · 2 layers · FFN 512"
    configuration = config.get("arm", config.get("configuration", "unknown"))
    selected = (config.get("selected_checkpoint_epoch")
                or provenance.get("selected_checkpoint_epoch")
                or (provenance.get("selection") or {}).get("epoch"))
    original_source = config.get("original_source")
    source = original_source or config.get("source")
    target = config.get("target")
    reference = config.get("reference")
    if isinstance(reference, dict):
        reference = reference.get("experiment") or reference.get("checkpoint")
    relative = str(directory)
    key = hashlib.sha256((relative + "\0" + role).encode()).hexdigest()[:16]
    dropout = model.get("rgb_branch_dropout", config.get("rgb_branch_dropout"))
    dropout_label = "RGB branch dropout"
    if dropout is None:
        dropout = model.get("dropout")
        dropout_label = "Model dropout"
    if config.get("dropout_enabled") is False or experiment in ("E003b", "E003c"):
        dropout, dropout_label = 0.0, "Dropout off"
    return {"id": key, "path": str(directory), "relative": relative,
        "experiment": experiment, "source": source, "target": target,
        "original_source": original_source, "configuration": configuration,
        "architecture": architecture, "model_variant": variant,
        "variant_label": {"norm": "Norm", "temporal-small": "Temporal small",
                          "branches-small": "Branches small"}.get(variant, variant),
        "capacity_label": capacity,
        "seed": settings.get("seed", config.get("seed")), "epochs": settings.get("epochs"),
        "patience": settings.get("patience"), "classes": config.get("classes", VALIDATION_CLASSES),
        "datasets": sorted(prediction_paths), "split": "validation",
        "selected_epoch": selected, "evaluation_labels": label,
        "adaptation_label": config.get("adaptation_label"), "reference_label": reference,
        "dropout": dropout, "dropout_label": dropout_label,
        "prediction_paths": {dataset: str(path) for dataset, path in prediction_paths.items()},
        "attempt": directory.name, "_role": role}


def _new_validation_records(directory, experiment, config, status, provenance):
    """Handle only the recorded validation layouts used by completed E002c–f and E003 runs."""
    if experiment not in VALIDATION_EXPERIMENTS or status.get("status") != "complete":
        return []
    kind = config.get("kind")
    if experiment in ("E002c", "E002d", "E002e", "E003a"):
        if kind != "exploratory-source-validation" or status.get("kind") != kind:
            return []
        source = config.get("source")
        paths = {dataset: directory/(dataset + "-validation")/"predictions.npz"
                 for dataset in ("loki", "road-waymo")}
        paths = {dataset: path for dataset, path in paths.items() if path.is_file()}
        if experiment != "E003a":
            paths = {dataset: path for dataset, path in paths.items() if dataset == source}
        if not paths:
            return []
        labels = {dataset: ("Source validation" if dataset == source else "Direct target validation")
                  for dataset in paths}
        return [_validation_record(directory, experiment, config, provenance, paths,
                                   role="source-validation", label=labels)]

    if experiment == "E002f":
        if kind == "exploratory-source-validation" and status.get("kind") == kind:
            source = config.get("source")
            path = directory/(str(source) + "-validation")/"predictions.npz"
            if not path.is_file():
                return []
            return [_validation_record(directory, experiment, config, provenance, {source: path},
                                       role="source-validation", label={source: "Source validation"})]
        if kind == "comparison" and status.get("kind") == kind and config.get("initialization") == "source-checkpoint":
            target = config.get("target", config.get("source"))
            path = directory/(str(target) + "-validation")/"predictions.npz"
            if not path.is_file():
                return []
            arm = config.get("arm", config.get("configuration"))
            label = {target: "Temporal-head adaptation · target validation" if arm == "temporal-head"
                     else "Head-only adaptation · target validation"}
            record_config = dict(config, arm="g-rgb-dditr", configuration="g-rgb-dditr",
                                 adaptation_label=label[target])
            return [_validation_record(directory, experiment, record_config, provenance, {target: path},
                                       role="adaptation-" + str(arm), label=label)]
        return []

    if experiment == "E003b":
        if kind != "comparison" or status.get("kind") != kind:
            return []
        target = config.get("target")
        if target not in ("loki", "road-waymo"):
            return []
        records = []
        roles = (("source-head-validation", "Original source head · target validation", "source-head"),
                 (target + "-validation", "Fresh target head · target validation", "target-head"))
        for folder, text, role in roles:
            path = directory/folder/"predictions.npz"
            if path.is_file():
                record_config = dict(config, original_source=config.get("original_source", config.get("source")),
                                     target=target, arm="g-rgb-dditr", configuration="g-rgb-dditr",
                                     adaptation_label=text)
                records.append(_validation_record(directory, experiment, record_config, provenance,
                    {target: path}, role=role, label={target: text}))
        return records

    if experiment == "E003c":
        if (kind != "comparison" or status.get("kind") != kind
                or config.get("initialization") != "source-checkpoint"):
            return []
        target = config.get("target", config.get("source"))
        if target not in ("loki", "road-waymo"):
            return []
        path = directory/(target + "-validation")/"predictions.npz"
        if not path.is_file():
            return []
        arm = config.get("arm", config.get("configuration"))
        text = "Source head fine-tuned · target validation" if arm == "head" else \
               "Temporal-head fine-tuned · target validation"
        record_config = dict(config, original_source=config.get("original_source", config.get("source")),
                             target=target, arm="g-rgb-dditr", configuration="g-rgb-dditr", adaptation_label=text)
        return [_validation_record(directory, experiment, record_config, provenance, {target: path},
                                   role="adaptation-" + str(arm), label={target: text})]
    return []


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2)+"\n")
    temporary.replace(path)


def track_scores(track, selected=None):
    mask = track["gt_valid"].copy()
    if selected is not None:
        mask &= selected
    n = int(mask.sum())
    classes = track["logits"].shape[1]
    counts = np.bincount(classes*track["targets"][mask]+track["predicted"][mask],
                         minlength=classes*classes).reshape(classes, classes)
    result = confusion_metrics(counts, supported_only=True)
    return {"accuracy": result["accuracy"], "ce": float(track["ce_values"][mask].mean()) if n else None,
            "macro_f1": result["macro_f1"], "gt_count": n, "support": result["support"],
            "confusion": counts.tolist(), "precision": result["precision"],
            "recall": result["recall"], "f1": result["f1"]}


def prepare_track(track, classes, cached_score=None):
    y, gt, logits = track["targets"], track["gt_valid"], track["logits"]
    if (y.ndim != 1 or y.dtype.kind not in "iu" or gt.shape != y.shape or gt.dtype != bool or not gt.any()
            or logits.shape != (len(y), classes) or not np.isfinite(logits).all()
            or ((y[gt] < 0) | (y[gt] >= classes)).any()):
        raise ValueError("Invalid/empty-GT prediction track")
    for key in ("grid_times_us", "source_times_us"):
        if len(track[key]) != len(y):
            raise ValueError("Prediction timestamp length mismatch")
    if any(np.asarray(v).shape != y.shape for v in (*track["conditions"].values(), *track["source_flags"].values())):
        raise ValueError("Condition/source-flag length mismatch")
    track["predicted"] = logits.argmax(1)
    scores = logits.astype(np.float64)
    scores -= scores.max(1, keepdims=True)
    track["ce_values"] = np.zeros(len(y), dtype=np.float64)
    track["ce_values"][gt] = np.log(np.exp(scores[gt]).sum(1))-scores[gt, y[gt]]
    track["scores"] = cached_score if cached_score is not None else track_scores(track)
    return track


class PredictionIndex:
    def __init__(self, outputs, cache, axes):
        self.outputs, self.cache, self.axes = Path(outputs).resolve(), Path(cache).resolve(), axes
        self.runs, self.loaded = {}, OrderedDict()
        self.lock = threading.RLock()

    def catalog(self, split="test"):
        """Read only small run metadata; prediction loading is lazy."""
        if split not in SPLITS:
            raise ValueError("Split must be test or validation")
        found, warnings = {}, []
        config_paths = set(self.outputs.glob("experiments/E001*/runs/**/config.json"))
        for experiment in ("E002a", "E002b"):
            config_paths.update(self.outputs.glob(f"experiments/{experiment}/runs/**/config.json"))
        for experiment in VALIDATION_EXPERIMENTS:
            config_paths.update(self.outputs.glob(f"experiments/{experiment}/runs/**/config.json"))
        for config_path in sorted(config_paths):
            directory = config_path.parent
            try:
                status = json.loads((directory/"status.json").read_text())
                if status.get("status") != "complete":
                    continue
                config = json.loads(config_path.read_text())
                relative = str(directory.relative_to(self.outputs))
                experiment = config.get("experiment", directory.relative_to(self.outputs).parts[1])
                if (any("smoke" in part.lower() or "provisional" in part.lower()
                        for part in Path(relative).parts) or config.get("smoke") or status.get("smoke")
                        or config.get("provisional") or status.get("provisional")):
                    continue
                if experiment in VALIDATION_EXPERIMENTS:
                    try:
                        provenance = json.loads((directory/"provenance.json").read_text())
                    except (OSError, ValueError):
                        provenance = {}
                    for record in _new_validation_records(directory, experiment, config, status, provenance):
                        record["relative"] = relative
                        record["id"] = hashlib.sha256((relative + "\0" + record.pop("_role")).encode()).hexdigest()[:16]
                        if record["split"] == split:
                            found[record["id"]] = record
                    continue
                if experiment.startswith("E001"):
                    if config.get("kind", "comparison") != "comparison" or status.get("kind", "comparison") != "comparison":
                        continue
                    result_split = "test"
                elif experiment in ("E002a", "E002b"):
                    group_child = config.get("execution") == "shared-g-dditr" or bool(config.get("shared_execution_id"))
                    result_split = "validation"
                    variant = config.get("model_variant")
                    origin = config.get("origin_training_run", {})
                    completed = config.get("training_completed_epochs")
                    selected = config.get("selected_checkpoint_epoch")
                    checkpoint_evaluation = (config.get("kind") == "checkpoint-evaluation"
                        and config.get("evaluation_only") is True and config.get("training_state") == "interrupted"
                        and config.get("execution") == "checkpoint-evaluation"
                        and config.get("origin_execution") == "shared-g-dditr"
                        and config.get("held_out_tests") is False
                        and config.get("evaluation_group_id") == directory.parent.name
                        and type(completed) is int and completed > 0
                        and type(selected) is int and 1 <= selected <= completed
                        and experiment == "E002b" and config.get("arm") in ("g-dditr", "g-rgb-dditr")
                        and isinstance(variant, str) and variant
                        and origin.get("run_id") and origin.get("run_url")
                        and origin.get("selected_epoch") == selected
                        and origin.get("completed_training_epochs") == completed
                        and status.get("kind") == "checkpoint-evaluation"
                        and status.get("evaluation_only") is True and status.get("training_state") == "interrupted"
                        and status.get("training_completed_epochs") == completed
                        and status.get("selected_checkpoint_epoch") == selected)
                    if checkpoint_evaluation:
                        parent_status = json.loads((directory.parent/"status.json").read_text())
                        if (parent_status.get("status") != "complete"
                                or parent_status.get("kind") != "checkpoint-evaluation"
                                or parent_status.get("evaluation_only") is not True
                                or parent_status.get("training_state") != "interrupted"
                                or parent_status.get("training_completed_epochs") != completed
                                or parent_status.get("execution") != "checkpoint-evaluation"
                                or parent_status.get("origin_execution") != "shared-g-dditr"
                                or parent_status.get("held_out_tests") is not False
                                or variant not in parent_status.get("variants", ())):
                            continue
                    else:
                        valid_configuration = (config.get("kind") == "exploratory-source-validation"
                            if not group_child else status.get("kind") == "exploratory-source-validation")
                        if (not valid_configuration or status.get("kind") not in (None, "exploratory-source-validation")
                                or config.get("held_out_tests") is True):
                            continue
                    if group_child:
                        parent_status = json.loads((directory.parent/"status.json").read_text())
                        if (parent_status.get("status") != "complete" or parent_status.get("smoke")
                                or parent_status.get("held_out_tests") is True
                                or parent_status.get("execution") != "shared-g-dditr"):
                            continue
                else:
                    continue
                datasets = [d for d in ("loki", "road-waymo")
                            if (directory/(d+"-"+result_split)/"predictions.npz").is_file()]
                if not datasets:
                    continue
                key = hashlib.sha256(relative.encode()).hexdigest()[:16]
                settings = config["settings"]
                architecture, configuration, model_variant = _run_label(config, experiment)
                try:
                    provenance = json.loads((directory/"provenance.json").read_text())
                except (OSError, ValueError):
                    provenance = {}
                selection = provenance.get("selection") or {}
                found[key] = {"id": key, "path": str(directory), "relative": relative,
                    "experiment": experiment, "source": config["source"],
                    "configuration": configuration, "architecture": architecture,
                    "model_variant": model_variant,
                    "variant_label": {"norm": "Norm", "temporal-small": "Temporal small",
                                      "branches-small": "Branches small"}.get(model_variant, model_variant),
                    "seed": settings["seed"], "epochs": settings["epochs"], "patience": settings["patience"],
                    "classes": config["classes"], "datasets": datasets, "split": result_split,
                    "selected_epoch": config.get("selected_checkpoint_epoch",
                        provenance.get("selected_checkpoint_epoch", selection.get("epoch"))),
                    "evaluation_only": config.get("evaluation_only", False),
                    "evaluation_label": (f"Evaluation only · interrupted after {config['training_completed_epochs']} epochs"
                                         if config.get("kind") == "checkpoint-evaluation" else None),
                    "training_state": config.get("training_state"),
                    "training_completed_epochs": config.get("training_completed_epochs"),
                    "origin_training_run": config.get("origin_training_run"),
                    "attempt": directory.name}
            except (OSError, ValueError, KeyError, TypeError) as error:
                warnings.append(f"{directory.name}: {error}")
        if split == "validation":
            for record in self._transfer_validation_records():
                found[record["id"]] = record
            for record in self._e002f_direct_validation_records():
                found[record["id"]] = record
        with self.lock:
            self.runs = found
        return {"runs": [run for run in found.values() if run["split"] == split], "warnings": warnings}

    def load(self, run, dataset, split="test"):
        if split not in SPLITS:
            raise ValueError("Split must be test or validation")
        with self.lock:
            if (run not in self.runs or self.runs[run].get("split") != split
                    or dataset not in self.runs[run]["datasets"]):
                raise ValueError("Unknown run, split, or evaluation dataset; refresh the run list")
            path = Path(self.runs[run].get("prediction_paths", {}).get(
                dataset, Path(self.runs[run]["path"])/(dataset+"-"+split)/"predictions.npz"))
            stat = path.stat()
            stamp = (stat.st_size, stat.st_mtime_ns)
            key = (run, dataset, split)
            if key in self.loaded and self.loaded[key]["stamp"] == stamp:
                self.loaded.move_to_end(key)
                return self.loaded[key]
            digest = checksum(path)
            cache_key = hashlib.sha256(f"{split}:{digest}".encode()).hexdigest()
            cached = self.cache/"indices"/(cache_key+f"-v{INDEX_VERSION}.json")
            stored = json.loads(cached.read_text())["tracks"] if cached.exists() else {}
            tracks = {}
            for track in load_predictions(path):
                prepare_track(track, len(self.runs[run]["classes"]), stored.get(track["locator"]))
                if track["locator"] in tracks:
                    raise ValueError("Duplicate prediction track identity")
                tracks[track["locator"]] = track
            if not tracks:
                raise ValueError("Empty predictions")
            if not cached.exists():
                write_json(cached, {"version": INDEX_VERSION, "split": split, "prediction_sha256": digest,
                                   "tracks": {k: t["scores"] for k, t in tracks.items()}})
            result = {"tracks": tracks, "sha256": digest, "stamp": stamp, "split": split}
            self.loaded[key] = result
            while len(self.loaded) > 4:
                self.loaded.popitem(last=False)
            return result

    def _transfer_validation_records(self):
        """Expose the frozen E002 transfer cells stored outside individual run directories."""
        root = self.outputs/"experiments/E002/transfer-validation-20261102"
        try:
            status = json.loads((root/"status.json").read_text())
            manifest = json.loads((root/"transfer-comparison.json").read_text())
            if (status.get("status") != "complete"
                    or manifest.get("comparison") != "E002-cross-dataset-validation-transfer"
                    or manifest.get("status") != "frozen-before-cross-dataset-validation-inference"):
                return []
        except (OSError, ValueError):
            return []
        records = []
        for model in manifest.get("models", []):
            source, input_name = model.get("training_source"), model.get("input")
            target = "loki" if source == "road-waymo" else "road-waymo" if source == "loki" else None
            if target is None or not input_name:
                continue
            cell = root/"cells"/source/input_name
            prediction_path, metrics_path = cell/"predictions.npz", cell/"metrics.json"
            try:
                metrics = json.loads(metrics_path.read_text())
                if (metrics.get("split") != "validation"
                        or metrics.get("training_source") != source
                        or metrics.get("target_source") != target
                        or metrics.get("input") != input_name
                        or metrics.get("checkpoint_sha256") != model.get("checkpoint_sha256")
                        or metrics.get("normalization_source") != source
                        or type(metrics.get("selected_epoch")) is not int
                        or not prediction_path.is_file()):
                    continue
            except (OSError, ValueError):
                continue
            config = {"source": source, "target": target, "original_source": source,
                      "arm": model.get("arm"), "variant": model.get("variant"),
                      "rgb_branch_dropout": model.get("rgb_branch_dropout"),
                      "classes": VALIDATION_CLASSES, "settings": {"seed": manifest.get("seed")},
                      "reference": model.get("experiment")}
            record = _validation_record(root, "E002 transfer", config,
                {"selected_checkpoint_epoch": metrics["selected_epoch"]}, {target: prediction_path},
                role=source+"-"+input_name, label={target: "Direct target validation"})
            record["relative"] = str(root.relative_to(self.outputs)) + "/cells/" + source + "/" + input_name
            record["id"] = hashlib.sha256((record["relative"] + "\0" + record.pop("_role")).encode()).hexdigest()[:16]
            record["experiment"] = model.get("experiment", "E002") + " direct transfer"
            record["checkpoint_sha256"] = model["checkpoint_sha256"]
            records.append(record)
        return records

    def _e002f_direct_validation_records(self):
        """Expose the architecture-specific direct LOKI predictions saved beside E002f features."""
        records = []
        for prediction_path in self.outputs.glob(
                "experiments/E002f/features/**/source-validation-predictions.npz"):
            directory = prediction_path.parent
            if any("smoke" in part.lower() or "acceptance" in part.lower() for part in directory.parts):
                continue
            try:
                status = json.loads((directory/"status.json").read_text())
                manifest = json.loads((directory/"feature-cache.json").read_text())
                identity = manifest["validation"]
                metrics = json.loads((directory/"loki-validation/metrics.json").read_text())
            except (OSError, ValueError, KeyError, TypeError):
                continue
            source, target, architecture = identity.get("source"), identity.get("target"), identity.get("architecture")
            source_checkpoint = identity.get("source_checkpoint") or {}
            digest = identity.get("checkpoint_sha256")
            if (status.get("status") != "complete" or status.get("held_out_tests") is not False
                    or identity.get("experiment") != "E002f" or identity.get("split") != "validation"
                    or identity.get("smoke_selection") is not None
                    or source != "road-waymo" or target != "loki"
                    or architecture not in ("bilstm-96", "transformer-1layer")
                    or source_checkpoint.get("sha256") != digest
                    or metrics.get("split") != "validation" or metrics.get("training_source") != source
                    or metrics.get("target_source") != target or metrics.get("architecture") != architecture
                    or metrics.get("checkpoint_sha256") != digest
                    or metrics.get("normalization_source") != source
                    or type(metrics.get("selected_epoch")) is not int or not prediction_path.is_file()):
                continue
            config = {"experiment": "E002f", "source": source, "target": target,
                "original_source": source, "architecture": architecture, "variant": architecture,
                "arm": "g-rgb-dditr", "configuration": "g-rgb-dditr",
                "rgb_branch_dropout": 0.5, "reference": "E002f source checkpoint",
                "selected_checkpoint_epoch": metrics["selected_epoch"],
                "classes": VALIDATION_CLASSES,
                "settings": {"seed": 0, "epochs": 100, "patience": 15}}
            relative = str(directory.relative_to(self.outputs))
            record = _validation_record(directory, "E002f", config, {}, {target: prediction_path},
                role="direct-target-validation", label={target: "Direct target validation · source model"})
            record["relative"] = relative
            record["id"] = hashlib.sha256((relative + "\0" + record.pop("_role")).encode()).hexdigest()[:16]
            record["experiment"] = "E002f direct transfer"
            record["checkpoint_sha256"] = digest
            records.append(record)
        return records

    def compare(self, runs, dataset, split="test"):
        if not 1 <= len(runs) <= 4 or len(set(runs)) != len(runs):
            raise ValueError("Select one to four distinct runs")
        if split not in SPLITS:
            raise ValueError("Split must be test or validation")
        loaded = [self.load(run, dataset, split) for run in runs]
        classes = self.runs[runs[0]]["classes"]
        if any(self.runs[run]["classes"] != classes for run in runs):
            raise ValueError("Incompatible class order")
        common = set.intersection(*(set(item["tracks"]) for item in loaded))
        if not common:
            raise ValueError("No common tracks")
        for locator in common:
            tracks = [item["tracks"][locator] for item in loaded]
            for other in tracks[1:]:
                for key in ("grid_times_us", "source_times_us", "targets", "gt_valid"):
                    if not np.array_equal(tracks[0][key], other[key]):
                        raise ValueError(f"Incompatible {key}: {locator}")
        axes = set.intersection(*(set(t["conditions"]) for item in loaded for t in item["tracks"].values()))
        axes &= self.axes.keys()
        for axis in list(axes):
            if any(not np.array_equal(loaded[0]["tracks"][loc]["conditions"][axis], item["tracks"][loc]["conditions"][axis])
                   for loc in common for item in loaded[1:]):
                axes.remove(axis)
        return loaded, sorted(common), {axis: self.axes[axis] for axis in sorted(axes)}

    def query(self, runs, dataset, metric="accuracy", axis="", value=0, scope="matching", confusion=None,
              sort="0", descending=None, search="", minimum=0, page=0, page_size=25, split="test"):
        if metric not in METRICS or scope not in ("matching", "whole"):
            raise ValueError("Unknown metric or score scope")
        loaded, common, axes = self.compare(runs, dataset, split)
        if axis and (axis not in axes or not 0 <= int(value) < len(axes[axis])):
            raise ValueError("Condition unavailable consistently across selected runs")
        if confusion is not None:
            confusion = tuple(map(int, confusion))
            if len(confusion) != 2 or any(c < 0 or c >= len(self.runs[runs[0]]["classes"]) for c in confusion) or confusion[0] == confusion[1]:
                raise ValueError("Choose two distinct valid classes")
        if metric == "confusion_rate" and confusion is None:
            raise ValueError("Choose a specific confusion")
        indices = list(map(int, sort.split("-")))
        if len(indices) not in (1, 2) or any(i not in range(len(runs)) for i in indices):
            raise ValueError("Invalid sort model")
        rows = []
        for locator in common:
            if search.lower() not in locator.lower():
                continue
            tracks = [item["tracks"][locator] for item in loaded]
            gt = tracks[0]["gt_valid"]
            match = tracks[0]["conditions"][axis] == int(value) if axis else np.ones(len(gt), dtype=bool)
            if not match.any():
                continue
            selected = match if scope == "matching" else None
            scores = [track_scores(t, selected) if axis and selected is not None else t["scores"].copy() for t in tracks]
            if scores[0]["gt_count"] < int(minimum):
                continue
            if confusion is not None:
                a, b = confusion
                for score in scores:
                    denominator = score["support"][a]
                    score["confusion_errors"] = score["confusion"][a][b]
                    score["confusion_support"] = int(denominator)
                    score["confusion_rate"] = score["confusion_errors"]/denominator if denominator else None
                if not any(s["confusion_errors"] for s in scores):
                    continue
            values = [s[metric] for s in scores]
            rank = values[indices[0]]
            if len(indices) == 2:
                other = values[indices[1]]
                rank = rank-other if rank is not None and other is not None else None
            rows.append({"locator": locator, "scores": scores, "values": values, "rank": rank,
                         "total_gt": int(gt.sum()), "matching_gt": int((gt & match).sum()),
                         "matching_context": int(match.sum())})
        descending = metric in ("ce", "confusion_rate") if descending is None else descending
        rows.sort(key=lambda r: (r["rank"] is None, (-r["rank"] if descending else r["rank"]) if r["rank"] is not None else 0, r["locator"]))
        start = max(0, int(page))*page_size
        return {"rows": rows[start:start+page_size], "total": len(rows), "common": len(common),
                "missing": [len(item["tracks"])-len(common) for item in loaded], "axes": axes,
                "classes": self.runs[runs[0]]["classes"], "hashes": [item["sha256"] for item in loaded],
                "page": max(0, int(page)), "page_size": page_size, "scope": scope, "split": split}

    def detail(self, runs, dataset, locator, split="test"):
        loaded, common, axes = self.compare(runs, dataset, split)
        if locator not in common:
            raise ValueError("Track is not shared by selected runs")
        tracks = [item["tracks"][locator] for item in loaded]
        first = tracks[0]
        return {"locator": locator, "split": split, "archive": first["archive"], "grid_times_us": first["grid_times_us"],
                "source_times_us": first["source_times_us"], "targets": first["targets"].tolist(),
                "gt_valid": first["gt_valid"].tolist(), "classes": self.runs[runs[0]]["classes"],
                "source_flags": {k: v.tolist() for k, v in first["source_flags"].items()},
                "conditions": {a: first["conditions"][a].tolist() for a in axes},
                "models": [{"run": run, "sha256": item["sha256"], "scores": track["scores"],
                            "predicted": track["predicted"].tolist(),
                            "probabilities": self.probabilities(track["logits"])}
                           for run, item, track in zip(runs, loaded, tracks)]}

    @staticmethod
    def probabilities(logits):
        values = np.exp(logits.astype(np.float64)-logits.max(1, keepdims=True))
        return (values/values.sum(1, keepdims=True)).tolist()
