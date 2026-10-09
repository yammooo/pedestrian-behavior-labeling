"""E001b bbox inputs and source normalization for the shared BiLSTM protocol."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pedestrian_behavior.data.bbox import load_bbox
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.models.kinematic import KinematicClassifier
from .e001 import CLASSES, MODEL_SETTINGS, STRATA as E001_STRATA, observation_conditions
from .e001_run import E001_ROOT, SETTINGS as E001_SETTINGS
from .e001b import POLICY, dataset_from_extension, load_extension
from .run import dataset_references, run_attempt as execute_attempt

E001B_ROOT = Path("outputs/experiments/E001b")
SETTINGS = E001_SETTINGS | {"epochs": 75, "patience": 8}
INPUT_DIMENSIONS = {"availability": 14, "geometry": 20}
STRATA = E001_STRATA | {"frame-bbox": ("missing", "valid"), "bbox-velocity": ("missing", "valid"),
                       "track-bbox": ("never", "partial", "complete")}


def build_model(configuration):
    return KinematicClassifier(INPUT_DIMENSIONS[configuration], MODEL_SETTINGS["embedding_dim"], len(CLASSES),
                               MODEL_SETTINGS["dropout"], recurrent_hidden=MODEL_SETTINGS["recurrent_hidden"])


def bbox_conditions(arrays):
    valid = arrays["bbox_valid"]
    cohort = 0 if not valid.any() else 2 if valid.all() else 1
    return {"frame-bbox": valid.astype(np.int64),
            "bbox-velocity": arrays["bbox_velocity_valid"].astype(np.int64),
            "track-bbox": np.full(len(valid), cohort, dtype=np.int64)}


def assemble(source, configuration, variant):
    setups = {d: E001_ROOT/"data-setup"/d/"setup.json" for d in ("loki", "road-waymo")}
    collections = {d: E001_ROOT/"reader-preparation"/d for d in setups}
    records = {d: json.loads(p.read_text()) for d, p in setups.items()}
    extensions = {d: E001B_ROOT/"bbox-native"/d for d in setups}
    caches = {d: load_extension(collections[d], setups[d], p) for d, p in extensions.items()}
    for d, cache in caches.items():
        if cache["baseline_normalization"] != records[d]["normalization"]["K+T+R"]:
            raise ValueError("BBox baseline normalization differs from frozen E001")
    statistics = {"baseline": records[source]["normalization"]["K+T+R"], "bbox": caches[source]["normalization"]}
    references = dataset_references(collections, setups, records)
    entries = {d: {e["locator"]: e for e in r["tracks"]} for d, r in caches.items()}
    for d in setups:
        references[d]["bbox"] = {"manifest_path": str(extensions[d]/"manifest.json"),
            "manifest_sha256": checksum(extensions[d]/"manifest.json"),
            "policy_sha256": hashlib.sha256(json.dumps(caches[d]["policy"], sort_keys=True).encode()).hexdigest(),
            "coverage": caches[d]["coverage"]}
    def conditions(metadata, arrays, track, d):
        conditions, flags = observation_conditions(metadata, arrays, track["targets"], track["gt_valid"], d)
        entry = entries[d][track["locator"]]
        archive = extensions[d]/"tracks"/entry["archive"]
        if checksum(archive) != entry["bbox_sha256"]:
            raise ValueError("BBox archive changed after preparation")
        _, bbox = load_bbox(archive, metadata, entry["e001_sha256"])
        track.update(bbox_archive=str(archive), bbox_sha256=entry["bbox_sha256"])
        flags.update({k: bbox[k] for k in ("bbox_valid", "bbox_velocity_valid", "bbox_source")})
        return conditions | bbox_conditions(bbox), flags
    return {"config": {"experiment": "E001b", "baseline": "K+T+R", "bbox_policy": POLICY,
                       "model": MODEL_SETTINGS | {"input_dim": INPUT_DIMENSIONS[configuration]}},
            "normalization": statistics, "datasets": references,
            "build_model": lambda: build_model(configuration),
            "dataset": lambda d, split: dataset_from_extension(collections[d], setups[d], extensions[d], split,
                configuration, statistics["baseline"], statistics["bbox"]), "conditions": conditions}


def run_attempt(source, configuration, output, device, smoke=False, seed=None, epochs=None, patience=None):
    if source not in ("loki", "road-waymo") or configuration not in INPUT_DIMENSIONS:
        raise ValueError("Unknown E001b source or configuration")
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
