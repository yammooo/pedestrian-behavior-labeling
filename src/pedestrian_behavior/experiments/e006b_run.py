"""Direct E006b Euro-PVI assessment with pinned E005 source checkpoints."""

from collections import Counter
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import isolated_extraction
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import predict, prediction_metrics, save_predictions
from pedestrian_behavior.experiments.e005_cache import TinyUtoniaEncoder, atomic_json, collate_e005
from pedestrian_behavior.experiments.e005_transfer import (
    DEFAULT_PROBE, DEFAULT_TINY_CHECKPOINT, DEFAULT_UTONIA_CODE,
)
from pedestrian_behavior.experiments.e006b_euro_pvi import EuroPVIAdapter, sample_for
from pedestrian_behavior.inspection.pretrained import ImageEncoder
from pedestrian_behavior.models.e005 import build_model
from pedestrian_behavior.training import seed_run


DATASET = Path("/run/media/user20/T9/ped_intention_1000")
AUDIT = Path("outputs/experiments/E006b/dataset-readiness-20261123/audit.json")
AUDIT_SHA256 = "6661e8c148b1db2a4ddbdbeff3929fac2693dcd2f318e8148e99f04dc9b7a85c"
RUNS = {
    "road-waymo": "road-waymo-g-rgb-utonia-s0-20261120-030721",
    "loki": "loki-g-rgb-utonia-s0-20261118-074440",
    "joint": "joint-g-rgb-utonia-s0-20261122-001910",
}
SMOKE_SEQUENCES = ("train/0067", "train/0158", "train/0339")
CLASS_NAMES = ("moving", "stopped", "waiting to cross", "crossing")


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _canonical(value):
    return json.loads(json.dumps(value))


def _load_audit(audit_path, dataset):
    if checksum(audit_path) != AUDIT_SHA256:
        raise ValueError("E006b assessment audit hash differs from the frozen partition")
    audit = json.loads(audit_path.read_text())
    hashes = audit["release_hashes"]
    for name in ("calib.json", "labels/frames.csv"):
        actual = checksum(dataset / name)
        if actual != hashes[name]:
            raise ValueError(f"Euro-PVI release hash mismatch for {name}: {actual}")
    assessment = audit["partition"]["assessment"]
    groups = assessment["groups"]
    if (len(groups) != 133 or len(set(groups)) != 133
            or assessment["recordings"] != 133 or assessment["tracks"] != 469
            or assessment["nominal_5Hz_frames"] != 7190):
        raise ValueError("Frozen E006b assessment membership/counts differ from the audited 469-track split")
    if set(assessment["nominal_5Hz_class_frames"]) != set(CLASS_NAMES):
        raise ValueError("Unexpected class labels in the frozen E006b assessment")
    return audit, assessment, set(groups)


def _load_models(outputs_root, device, dino_sha, tiny_sha):
    models = {}
    for source, run_name in RUNS.items():
        run = outputs_root / "experiments/E005/runs" / run_name
        status = json.loads((run / "status.json").read_text())
        config = json.loads((run / "config.json").read_text())
        path = run / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        expected_source = source if source != "joint" else "joint"
        if (status.get("status") != "complete"
                or config.get("experiment") != "E005"
                or config.get("source") != expected_source
                or config.get("configuration") != "g-rgb-utonia"
                or config.get("variant") != "bilstm-96"
                or config.get("held_out_tests") is not False
                or config.get("smoke_limits") is not None
                or _canonical(checkpoint["config"]) != config
                or checkpoint["epoch"] != checkpoint["selection"]["epoch"]):
            raise ValueError(f"Pinned full E005 checkpoint/config mismatch: {source}")

        provenance = json.loads((run / "provenance.json").read_text())
        if (checkpoint["selection"] != provenance.get("selection")
                or checkpoint["epoch"] != provenance.get("selected_checkpoint_epoch")):
            raise ValueError(f"Pinned E005 selection evidence mismatch: {source}")
        caches = config["caches"].values() if source == "joint" else (config["cache"],)
        for cache in caches:
            frozen = cache["provenance"]
            if (frozen["DINO"]["checkpoint_sha256"] != dino_sha
                    or frozen["Tiny-Utonia"]["checkpoint_sha256"] != tiny_sha):
                raise ValueError(f"Frozen encoder checkpoint mismatch in E005 {source} run")

        model = build_model(config["arm"], config["profile"]).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        models[source] = (model.eval(), checkpoint["normalization"], {
            "run": run_name, "checkpoint": str(path.resolve()), "checkpoint_sha256": checksum(path),
            "selected_epoch": checkpoint["epoch"], "selection": checkpoint["selection"],
            "source_normalization": checkpoint["normalization"],
        })
    return models


def _plot_results(output, metrics, smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(RUNS)
    colors = ("#4477AA", "#EE6677", "#228833")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(CLASS_NAMES))
    width = .24
    for i, name in enumerate(names):
        values = metrics[name]["track_weighted"]["f1"]
        ax.bar(x + (i - 1) * width, values, width, label=name, color=colors[i])
    label = "calibration smoke" if smoke else "assessment"
    ax.set(xticks=x, xticklabels=CLASS_NAMES, ylim=(0, 1), ylabel="Track-weighted F1",
           title=f"Euro-PVI {label}: per-class F1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "per-class-f1.png", dpi=160)
    fig.savefig(output / "per-class-f1.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), constrained_layout=True)
    for ax, name in zip(axes, names):
        matrix = np.asarray(metrics[name]["track_weighted"]["confusion"], dtype=float)
        support = matrix.sum(1, keepdims=True)
        matrix = np.divide(matrix, support, out=np.zeros_like(matrix), where=support != 0)
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        for (row, col), value in np.ndenumerate(matrix):
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
        ax.set(xticks=range(4), yticks=range(4), xticklabels=("mov", "stop", "wait", "cross"),
               yticklabels=("mov", "stop", "wait", "cross"), title=name,
               xlabel="Prediction", ylabel="GT")
    fig.suptitle(f"Euro-PVI {label}: track-weighted confusion (row-normalized)")
    fig.savefig(output / "track-weighted-confusion.png", dpi=160)
    fig.savefig(output / "track-weighted-confusion.svg")
    plt.close(fig)


def run(args):
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("E006b frozen Tiny-Utonia assessment requires CUDA on Aalto")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / "status.json"
    atomic_json(status_path, {"status": "running", "failures": []})
    try:
        torch.set_num_threads(2)
        determinism = seed_run(0)
        torch.cuda.reset_peak_memory_stats(args.device)
        dataset = args.dataset.resolve()
        audit_path = args.audit.resolve()
        audit, assessment, assessment_groups = _load_audit(audit_path, dataset)
        calibration_groups = set(audit["partition"]["calibration"]["groups"])
        if assessment_groups & calibration_groups:
            raise ValueError("Frozen Euro-PVI calibration and assessment sequences overlap")
        if args.smoke:
            sequence_ids = list(SMOKE_SEQUENCES)
            if not set(sequence_ids) <= calibration_groups:
                raise ValueError("Fixed smoke sequences are not part of the audited calibration split")
        else:
            sequence_ids = assessment["groups"]

        native = json.loads((args.outputs_root / DEFAULT_PROBE / "probe.json").read_text())["config"]
        tiny_path = args.outputs_root / DEFAULT_TINY_CHECKPOINT
        dino_sha, tiny_sha = checksum(native["dino_weights"]), checksum(tiny_path)
        models = _load_models(args.outputs_root, args.device, dino_sha, tiny_sha)
        with isolated_extraction():
            dino = ImageEncoder(native["dino_code"], native["dino_weights"], args.device)
            tiny = TinyUtoniaEncoder(args.outputs_root / DEFAULT_UTONIA_CODE, tiny_path, args.device)

        adapter = EuroPVIAdapter(dataset)
        available = set(adapter.sequences())
        if not set(sequence_ids) <= available:
            raise ValueError("Frozen assessment refers to Euro-PVI sequences absent from labels/frames.csv")

        predictions = {source: [] for source in models}
        totals, classes_seen = Counter(), Counter()
        for index, sequence_id in enumerate(sequence_ids, 1):
            sequence = adapter.load_sequence(sequence_id)
            with isolated_extraction():
                records = adapter.encode_sequence(sequence, dino, tiny, args.device)
            if not records:
                raise ValueError(f"No labeled tracks in frozen sequence {sequence_id}")
            for record in records:
                if record["group"] != sequence_id:
                    raise ValueError(f"Track group mismatch in {sequence_id}")
                totals["tracks"] += 1
                totals["context_slots"] += len(record["targets"])
                totals["gt_slots"] += int(record["gt_valid"].sum())
                for label in record["targets"][record["gt_valid"]]:
                    classes_seen[CLASS_NAMES[int(label)]] += 1
            for source, (model, normalization, _) in models.items():
                samples = [sample_for(record, normalization) for record in records]
                batches = (collate_e005(samples[i:i + 2]) for i in range(0, len(samples), 2))
                rows = predict(model, batches, args.device, forward_batch=lambda m, b, _: m(b))
                for row in rows:
                    row["conditions"] = {}
                predictions[source].extend(rows)
            if index == 1 or index % 10 == 0 or index == len(sequence_ids):
                print(f"Sequences {index}/{len(sequence_ids)} | tracks={totals['tracks']} "
                      f"GT={totals['gt_slots']} | {sequence_id}", flush=True)
            del sequence, records

        expected = (5, 100, 100) if args.smoke else (469, 7190, 7190)
        if (totals["tracks"], totals["context_slots"], totals["gt_slots"]) != expected:
            raise ValueError(f"Unexpected split tracks/context/GT: {totals}; expected {expected}")
        if not args.smoke and dict(classes_seen) != assessment["nominal_5Hz_class_frames"]:
            raise ValueError(f"Assessment class support differs from frozen audit: {dict(classes_seen)}")

        metric_results = {}
        for source, rows in predictions.items():
            metric = prediction_metrics(rows, 4)
            denominators = metric["denominators"]
            if denominators["tracks"] != expected[0] or denominators["gt_frames"] != expected[2]:
                raise ValueError(f"Predictions lost track/GT support for {source}: {denominators}")
            metric_results[source] = metric
            save_predictions(args.output / f"{source}-predictions.npz", rows)
            _write_json(args.output / f"{source}-metrics.json", metric)

        _plot_results(args.output, metric_results, smoke=args.smoke)
        ended_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "status": "complete", "experiment": "E006b", "kind": "calibration-smoke" if args.smoke else "frozen-assessment",
            "dataset": str(dataset), "audit": str(audit_path), "audit_sha256": checksum(audit_path),
            "release_sha256": {name: checksum(dataset / name) for name in ("calib.json", "labels/frames.csv")},
            "split": "calibration-smoke" if args.smoke else "assessment",
            "sequences": sequence_ids, "class_names": list(CLASS_NAMES),
            "support": dict(totals), "class_frames": dict(classes_seen),
            "models": {source: {**evidence, "metrics": metric_results[source]}
                       for source, (_, _, evidence) in models.items()},
            "encoder_sha256": {"dino": dino_sha, "tiny_utonia": tiny_sha},
            "tiny_utonia_sweeps": tiny.calls, "determinism": determinism,
            "command": sys.orig_argv,
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "started_utc": started_at, "ended_utc": ended_at,
            "elapsed_seconds": time.monotonic() - started,
            "hardware": {"host": platform.node(), "gpu": torch.cuda.get_device_name(args.device),
                         "torch": torch.__version__, "cuda": torch.version.cuda},
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        }
        _write_json(args.output / "summary.json", summary)
        atomic_json(status_path, {"status": "complete", "failures": []})
        print(f"E006b {'smoke' if args.smoke else 'assessment'} complete: {args.output}", flush=True)
        return summary
    except BaseException as error:
        atomic_json(status_path, {"status": "failed", "failures": [{
            "type": type(error).__name__, "message": str(error),
        }]})
        raise
