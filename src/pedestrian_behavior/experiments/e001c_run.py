"""E001c box inputs for the unchanged shared BiLSTM training/evaluation loop."""

import argparse
import hashlib
import json
from pathlib import Path

from pedestrian_behavior.data.boxes import COLUMNS, FLAGS, INPUT_COLUMNS, load_boxes
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.models.kinematic import KinematicClassifier
from .e001 import CLASSES, MODEL_SETTINGS, observation_conditions
from .e001_run import E001_ROOT, SETTINGS as E001_SETTINGS
from .e001b_run import STRATA, bbox_conditions
from .e001c import dataset_from_cache, load_cache
from .run import dataset_references, run_attempt as execute_attempt

E001C_ROOT = Path("outputs/experiments/E001c")
SETTINGS = E001_SETTINGS | {"epochs": 75, "patience": 8}
INPUT_DIMENSIONS = {name: count+len(FLAGS) for name, count in INPUT_COLUMNS.items()}


def build_model(configuration):
    return KinematicClassifier(INPUT_DIMENSIONS[configuration], MODEL_SETTINGS["embedding_dim"], len(CLASSES),
                               MODEL_SETTINGS["dropout"], recurrent_hidden=MODEL_SETTINGS["recurrent_hidden"])


def assemble(source, configuration, variant):
    setups = {d: E001_ROOT/"data-setup"/d/"setup.json" for d in ("loki", "road-waymo")}
    collections = {d: E001_ROOT/"reader-preparation"/d for d in setups}
    records = {d: json.loads(path.read_text()) for d, path in setups.items()}
    caches = {d: E001C_ROOT/"box-cache"/d for d in setups}
    manifests = {d: load_cache(collections[d], setups[d], path) for d, path in caches.items()}
    if any(record["dataset"] != d for d, record in manifests.items()):
        raise ValueError("E001c dataset/root mismatch")
    statistics = manifests[source]["normalization"]
    references = dataset_references(collections, setups, records)
    entries = {d: {e["locator"]: e for e in record["tracks"]} for d, record in manifests.items()}
    for d in setups:
        references[d]["boxes"] = {"manifest_path": str(caches[d]/"manifest.json"),
            "manifest_sha256": checksum(caches[d]/"manifest.json"),
            "policy_sha256": hashlib.sha256(json.dumps(manifests[d]["policy"], sort_keys=True).encode()).hexdigest(),
            "coverage": manifests[d]["coverage"]}
    def conditions(metadata, arrays, track, d):
        conditions, flags = observation_conditions(metadata, arrays, track["targets"], track["gt_valid"], d)
        entry = entries[d][track["locator"]]
        archive = caches[d]/"tracks"/entry["archive"]
        if checksum(archive) != entry["boxes_sha256"]:
            raise ValueError("E001c archive changed after preparation")
        _, boxes = load_boxes(archive, metadata, entry["e001_sha256"])
        track.update(boxes_archive=str(archive), boxes_sha256=entry["boxes_sha256"])
        flags.update({key: boxes[key] for key in (*FLAGS, "native_3d", "center3d_valid", "size3d_valid")})
        bbox = {"bbox_valid": boxes["box2d_valid"], "bbox_velocity_valid": boxes["box2d_velocity_valid"]}
        return conditions | bbox_conditions(bbox), flags
    return {"config": {"experiment": "E001c", "cache_policy": manifests[source]["policy"],
                       "inputs": {"numeric_columns": COLUMNS[:INPUT_COLUMNS[configuration]], "flags": FLAGS},
                       "model": MODEL_SETTINGS | {"input_dim": INPUT_DIMENSIONS[configuration]}},
            "normalization": statistics, "datasets": references, "build_model": lambda: build_model(configuration),
            "dataset": lambda d, split: dataset_from_cache(collections[d], setups[d], caches[d], split, configuration, statistics),
            "conditions": conditions}


def run_attempt(source, configuration, output, device, smoke=False, seed=None, epochs=None, patience=None):
    if source not in ("loki", "road-waymo") or configuration not in INPUT_DIMENSIONS:
        raise ValueError("Unknown E001c source or configuration")
    settings = SETTINGS | {k: v for k, v in {"seed": seed, "epochs": epochs, "patience": patience}.items() if v is not None}
    return execute_attempt(source, configuration, "B", output, device, smoke,
                           settings=settings, assemble=assemble, classes=CLASSES, axes=STRATA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("loki", "road-waymo"), required=True)
    parser.add_argument("--configuration", choices=INPUT_DIMENSIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    parser.add_argument("--patience", type=int, default=SETTINGS["patience"])
    parser.add_argument("--smoke", action="store_true", help="ROAD CUDA: 1 epoch, 2 training/1 validation batches, no tests")
    run_attempt(**vars(parser.parse_args()))
