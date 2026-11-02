"""One frozen eight-checkpoint E002 opposite-validation comparison."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.data.prepared_points import PreparedPoints
from pedestrian_behavior.data.prepared_rgb import PreparedRGB
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches
from pedestrian_behavior.evaluation import prediction_metrics, save_predictions
from pedestrian_behavior.experiments import e001b, e001b_run, e002a_run, e002b, e002b_group
from pedestrian_behavior.experiments.e002a_features import FeatureCatalog, load_configuration
from pedestrian_behavior.experiments.run import write_json
from pedestrian_behavior.inspection.pretrained import DITR_CONFIG_SHA256, DITR_REVISION, DITR_SHA256


ROOT = Path(__file__).resolve().parents[3]
ARMS = {"geometry": "g", "geometry+rgb": "g-rgb", "geometry+lidar": "g-dditr",
        "geometry+rgb+lidar": "g-rgb-dditr"}
CLASSES = ["MOVING", "STOPPED", "WAITING_TO_CROSS", "CROSSING"]


def _source_normalization(checkpoint, baseline, bbox):
    expected = {"baseline": baseline, "bbox": bbox}
    actual = checkpoint.get("normalization")
    if actual != expected:
        raise ValueError("Checkpoint normalization differs from its training-source statistics")
    return actual


def _load_frozen(rows, outputs_root):
    frozen = []
    setups, bbox_records = {}, {}
    for source in ("road-waymo", "loki"):
        setup_path = outputs_root / "experiments/E001/data-setup" / source / "setup.json"
        collection = outputs_root / "experiments/E001/reader-preparation" / source
        extension = outputs_root / "experiments/E001b/bbox-native" / source
        setup = json.loads(setup_path.read_text())
        bbox = e001b.load_extension(collection, setup_path, extension)
        setups[source], bbox_records[source] = (setup_path, setup, collection, extension), bbox

    for row in rows:
        checkpoint_path = (ROOT / row["checkpoint"]).resolve()
        if checksum(checkpoint_path) != row["checkpoint_sha256"]:
            raise ValueError(f"Frozen checkpoint hash changed: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        run_config = json.loads((checkpoint_path.parent / "config.json").read_text())
        source = row["training_source"]
        expected_norm = _source_normalization(
            checkpoint, setups[source][1]["normalization"]["K+T+R"], bbox_records[source]["normalization"])
        arm = ARMS[row["input"]]
        rgb_dropout = row["rgb_branch_dropout"]
        model_spec = e002b.model_spec(arm, row["variant"], dropout=0.1,
                                      rgb_branch_dropout=rgb_dropout)
        if (run_config.get("source") != source or run_config.get("configuration") != arm
                or run_config.get("variant") != row["variant"] or run_config.get("model") != model_spec
                or run_config.get("held_out_tests") is not False
                or checkpoint.get("epoch") != row["selected_epoch"]
                or checkpoint.get("selection", {}).get("epoch") != row["selected_epoch"]
                or checkpoint.get("selection", {}).get("f1") != row["source_validation"]["track_weighted_macro_f1"]
                or checkpoint.get("selection", {}).get("ce") != row["source_validation"]["track_ce"]):
            raise ValueError(f"Frozen checkpoint/configuration mismatch: {checkpoint_path}")
        if checkpoint.get("config", {}).get("model") != model_spec:
            raise ValueError(f"Checkpoint model spec mismatch: {checkpoint_path}")
        frozen.append(row | {"checkpoint_path": checkpoint_path, "checkpoint": checkpoint,
                             "run_config": run_config, "source_normalization": expected_norm})
    return frozen, setups, bbox_records


class _Target:
    def __init__(self, source, outputs_root, setups, bbox_records, probe, rgb_root):
        self.source, self.outputs_root = source, outputs_root
        self.setup_path, self.setup, self.collection, self.extension = setups[source]
        self.bbox_record = bbox_records[source]
        self.conditions = e001b_run.assemble(source, "geometry", "B",
            e001_root=outputs_root / "experiments/E001", e001b_root=outputs_root / "experiments/E001b")["conditions"]
        self.probe, self.rgb_root = probe, rgb_root
        self.catalog = self.rgb = self.points = None

    def supply(self, row, run_config):
        arm = ARMS[row["input"]]
        if arm == "g":
            return None
        if self.catalog is None:
            config, transform = load_configuration(self.probe)
            self.catalog = FeatureCatalog(self.outputs_root, self.source, config, transform)
        if arm in ("g-rgb", "g-rgb-dditr") and self.rgb is None:
            self.rgb = PreparedRGB(self.rgb_root)
        if arm in ("g-dditr", "g-rgb-dditr") and self.points is None:
            point_config = run_config["point_loading"]
            self.points = PreparedPoints(
                self.outputs_root / f"experiments/E002a/indexed-{self.source}", self.source,
                self.catalog.identity, self.catalog.tracks,
                {"revision": DITR_REVISION, "checkpoint": DITR_SHA256,
                 "configuration": DITR_CONFIG_SHA256},
                max_open_scenes=point_config["open_scenes"],
                max_metadata_bytes=point_config["metadata_limit_bytes"],
                frame_workers=point_config["frame_workers"])
        identity = run_config.get("frozen_features") if self.rgb is not None else None
        return e002a_run.PreparedSupply(self.catalog, self.rgb if arm in ("g-rgb", "g-rgb-dditr") else None,
            self.points if arm in ("g-dditr", "g-rgb-dditr") else None, identity)

    def close(self):
        if self.points is not None:
            self.points.close()
        if self.rgb is not None:
            self.rgb.close()


def _evaluate(row, target, output, device):
    source, arm = row["training_source"], ARMS[row["input"]]
    baseline = e001b.dataset_from_extension(target.collection, target.setup_path, target.extension,
        "validation", "geometry", row["source_normalization"]["baseline"], row["source_normalization"]["bbox"])
    expected = {entry["locator"] for entry in baseline.entries}
    if len(baseline) != len(expected) or any(entry["split"] != "validation" for entry in baseline.entries):
        raise ValueError(f"Target population is not the frozen validation split: {target.source}")
    supply = target.supply(row, row["run_config"])
    dataset = e002a_run.FeatureDataset(baseline, supply, target.source, arm)
    model_spec = row["run_config"]["model"]
    model = e002b.build_model(arm, row["variant"], row["run_config"]["profile"],
        dropout=model_spec["dropout"], rgb_branch_dropout=model_spec.get("rgb_branch_dropout", 0.)).to(device)
    model.load_state_dict(row["checkpoint"]["model_state"], strict=True)
    model.eval()
    logical = LogicalBatches(dataset, row["run_config"]["profile"], training=False, seed=0,
        prefetch=arm in ("g-dditr", "g-rgb-dditr"))
    begin = time.monotonic()
    tracks = e002b_group.predict_shared({"model": model}, PredictionBatches(logical), device)["model"]
    if len(tracks) != len(expected) or {track["locator"] for track in tracks} != expected:
        raise ValueError("Predictions do not cover the frozen target validation population")
    e002b_group._decorate(tracks, {"conditions": target.conditions}, {}, target.source)
    metrics = prediction_metrics(tracks, len(CLASSES))
    cell = output / source / row["input"]
    cell.mkdir(parents=True)
    save_predictions(cell / "predictions.npz", tracks)
    write_json(cell / "metrics.json", {"split": "validation", "training_source": source,
        "target_source": target.source, "input": row["input"], "selected_epoch": row["selected_epoch"],
        "checkpoint_sha256": row["checkpoint_sha256"], "normalization_source": source,
        "primary": metrics, "inference_seconds": time.monotonic() - begin,
        "dataset_tracks": len(dataset), "prediction_tracks": len(tracks)})
    return {"training_source": source, "target_source": target.source, "input": row["input"],
        "selected_epoch": row["selected_epoch"], "source_f1": row["source_validation"]["track_weighted_macro_f1"],
        "source_ce": row["source_validation"]["track_ce"],
        "target_f1": metrics["track_weighted"]["macro_f1"], "target_ce": metrics["track_ce"],
        "delta_f1_pp": 100 * (metrics["track_weighted"]["macro_f1"] - row["source_validation"]["track_weighted_macro_f1"]),
        "target_tracks": len(tracks), "checkpoint_sha256": row["checkpoint_sha256"]}


def _write_results(output, records):
    fields = list(records[0])
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    write_json(output / "results.json", records)
    lines = ["# E002 exploratory cross-dataset validation transfer", "",
        "This is target-informed validation evidence, not held-out or strict zero-shot evaluation. Source-validation results are the exact selected-checkpoint records; no source inference was repeated. No test split was accessed.", "",
        "| Training source | Inputs | Source val F1 | Source val CE | Opposite val F1 | Opposite val CE | Δ F1 (pp) |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append(f"| {r['training_source']} | {r['input']} | {r['source_f1']:.4f} | {r['source_ce']:.4f} | {r['target_f1']:.4f} | {r['target_ce']:.4f} | {r['delta_f1_pp']:+.2f} |")
    lines.extend(["", "The source checkpoint’s baseline and bbox normalization was applied to opposite-source validation. The frozen checkpoint list, hashes and source metrics are recorded in the repository's `experiments/E002-multimodal-reference/transfer-comparison.json`.", "",
        "![Source and opposite-source validation F1](f1.png)"])
    (output / "transfer-results.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    inputs = list(ARMS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, source in zip(axes, ("road-waymo", "loki")):
        rows = [r for r in records if r["training_source"] == source]
        x = np.arange(len(inputs))
        width = .36
        source_scores = [next(r["source_f1"] for r in rows if r["input"] == name) for name in inputs]
        target_scores = [next(r["target_f1"] for r in rows if r["input"] == name) for name in inputs]
        ax.bar(x - width/2, source_scores, width, label="Source validation")
        ax.bar(x + width/2, target_scores, width, label="Opposite validation")
        ax.set(title=f"Trained on {source}", xticks=x, xticklabels=["G", "G+RGB", "G+LiDAR", "G+RGB+LiDAR"],
               ylim=(0, 1), ylabel="Track-weighted macro-F1")
        ax.tick_params(axis="x", rotation=18)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    fig.suptitle("Exploratory validation transfer")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"f1.{suffix}", dpi=160)
    plt.close(fig)


def run(comparison, outputs_root, output, device="cuda"):
    comparison, outputs_root, output = Path(comparison).resolve(), Path(outputs_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not output.is_relative_to(outputs_root / "experiments/E002") or output == outputs_root / "experiments/E002":
        raise ValueError("Transfer artifacts must be a fresh child of outputs/experiments/E002")
    frozen_config = json.loads(comparison.read_text())
    if (frozen_config.get("status") != "frozen-before-cross-dataset-validation-inference"
            or frozen_config["evaluation"].get("split") != "opposite-source-validation"
            or frozen_config["evaluation"].get("cells") != 8
            or frozen_config["evaluation"].get("held_out_tests_accessed") is not False
            or len(frozen_config.get("models", [])) != 8):
        raise ValueError("Frozen eight-cell validation comparison required")
    rows, setups, bbox_records = _load_frozen(frozen_config["models"], outputs_root)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(2)
    output.mkdir(parents=True)
    started = time.monotonic()
    write_json(output / "transfer-comparison.json", frozen_config)
    write_json(output / "status.json", {"status": "incomplete", "completed_cells": [],
        "scope": "opposite-source validation only; no held-out test access", "started_utc": datetime.now(timezone.utc).isoformat()})
    probe = outputs_root / "experiments/E002a/native-probe-07"
    rgb_root = outputs_root / "experiments/E002a/rgb-cache"
    targets = {}
    records = []
    # Geometry/RGB cells first, so those results are available before the slower LiDAR cells.
    execution_rows = sorted(rows, key=lambda r: "lidar" in r["input"])
    try:
        for row in execution_rows:
            target_source = "loki" if row["training_source"] == "road-waymo" else "road-waymo"
            if target_source not in targets:
                targets[target_source] = _Target(target_source, outputs_root, setups, bbox_records, probe, rgb_root)
            target = targets[target_source]
            record = _evaluate(row, target, output / "cells", device)
            records.append(record)
            write_json(output / "status.json", {"status": "incomplete", "completed_cells": records,
                "scope": "opposite-source validation only; no held-out test access", "started_utc": datetime.now(timezone.utc).isoformat()})
        _write_results(output, records)
        write_json(output / "status.json", {"status": "complete", "completed_cells": records,
            "scope": "exploratory opposite-source validation; no held-out test access",
            "started_utc": json.loads((output / "status.json").read_text())["started_utc"],
            "finished_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - started})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "completed_cells": records,
            "failure": f"{type(error).__name__}: {error}", "finished_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        for target in targets.values():
            target.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, default=ROOT / "experiments/E002-multimodal-reference/transfer-comparison.json")
    parser.add_argument("--outputs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args()
    run(args.comparison, args.outputs_root, args.output, args.device)


if __name__ == "__main__":
    main()
