"""E001 datasets and scientific settings for the shared attempt runner."""

import argparse
from functools import partial
import json
from pathlib import Path

import wandb

from pedestrian_behavior.data.features import FEATURE_SETS
from .e001 import CLASSES, MODEL_SETTINGS, STRATA, build_model, dataset_from_setup, observation_conditions
from .run import configure_logging, dataset_references, write_json
from .run import log_final as shared_log_final, plots as shared_plots, run_attempt as execute_attempt

E001_ROOT = Path("outputs/experiments/E001")
SETTINGS = {"seed": 0, "batch_size": 64, "epochs": 30, "patience": 5, "clip_norm": 1.,
            "precision": "float32", "optimizer": {"lr": .001, "weight_decay": .0001, "betas": [.9, .999], "eps": 1e-8}}

# Preserve the existing E001 helper interfaces.
plots = partial(shared_plots, classes=CLASSES)
log_final = partial(shared_log_final, classes=CLASSES, axes=STRATA)


def assemble(source, configuration, variant):
    setups = {d: E001_ROOT/"data-setup"/d/"setup.json" for d in ("loki", "road-waymo")}
    collections = {d: E001_ROOT/"reader-preparation"/d for d in setups}
    records = {d: json.loads(p.read_text()) for d, p in setups.items()}
    statistics = records[source]["normalization"][configuration]
    return {"config": {"experiment": "E001", "model": MODEL_SETTINGS}, "normalization": statistics,
            "datasets": dataset_references(collections, setups, records),
            "build_model": lambda: build_model(configuration, variant),
            "dataset": lambda d, split: dataset_from_setup(collections[d], setups[d], split, configuration, statistics),
            "conditions": lambda metadata, arrays, track, d: observation_conditions(metadata, arrays, track["targets"], track["gt_valid"], d)}


def run_attempt(source, configuration, variant, output, device, smoke=False, seed=None, epochs=None, patience=None):
    settings = SETTINGS | {k: v for k, v in {"seed": seed, "epochs": epochs, "patience": patience}.items() if v is not None}
    return execute_attempt(source, configuration, variant, output, device, smoke,
                           settings=settings, assemble=assemble, classes=CLASSES, axes=STRATA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("loki", "road-waymo"), required=True)
    parser.add_argument("--configuration", choices=FEATURE_SETS, required=True)
    parser.add_argument("--variant", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--seed", type=int, default=SETTINGS["seed"])
    parser.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    parser.add_argument("--patience", type=int, default=SETTINGS["patience"])
    parser.add_argument("--smoke", action="store_true", help="Separate ROAD B CUDA smoke: 1 epoch, 2 training/1 validation batches, no tests")
    run_attempt(**vars(parser.parse_args()))
