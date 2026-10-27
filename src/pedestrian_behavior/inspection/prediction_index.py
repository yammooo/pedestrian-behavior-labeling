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


def _run_label(config, experiment):
    if experiment.startswith("E001"):
        return ("BiLSTM" if config.get("variant") == "B" else "MLP",
                config.get("configuration", "unknown"), None)
    if experiment == "E002a":
        return "Transformer", config.get("arm", config.get("configuration", "unknown")), None
    variant = config.get("model_variant", config.get("variant", "unknown"))
    return "Transformer", config.get("arm", config.get("configuration", "unknown")), str(variant)


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
            path = Path(self.runs[run]["path"])/(dataset+"-"+split)/"predictions.npz"
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
