"""E001d matched-capacity framewise MLP using the frozen E001 pipeline."""

import argparse
from pathlib import Path

from torch import nn

from . import e001, e001_run
from .run import run_attempt as execute_attempt

SETTINGS = e001_run.SETTINGS | {"epochs": 75, "patience": 8}
MODEL = {"input_dim": 12, "embedding_dim": 64, "dropout": 0.1, "recurrent_hidden": None,
         "framewise_block": [64, 194, 64], "parameter_count": 30342,
         "readout": [64, 4], "readout_dropout": 0.1}
COMPARISON = {"baseline_experiment": "E001", "configuration": "K+T+R", "variants": ["A", "B"],
              "seeds": [0, 1, 2, 3, 4], "matched_by": ["source", "seed"],
              "held_out_tests_previously_inspected": True}


def build_model():
    model = e001.build_model("K+T+R", "A")
    model.encoder = nn.Sequential(*model.encoder.children(), nn.Linear(64, 194), nn.ReLU(), nn.Linear(194, 64))
    return model


def assemble(source, seed=0):
    prepared = e001_run.assemble(source, "K+T+R", "A")
    prepared["config"] = prepared["config"] | {
        "experiment": "E001d", "model": MODEL, "comparison": COMPARISON,
        "development_context": "Original E001 held-out tests informed prior development; this follow-up is not a pristine-test claim."}
    prepared["build_model"] = build_model
    prepared["run_name"] = f"E001d-{source}-ktr-mlp-matched-s{seed}"
    return prepared


def run_attempt(source, output, device, seed=0):
    settings = SETTINGS | {"seed": seed}
    return execute_attempt(source, "K+T+R", "mlp-matched", output, device,
                           settings=settings, assemble=lambda source, *_: assemble(source, seed),
                           classes=e001.CLASSES, axes=e001.STRATA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("loki", "road-waymo"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    run_attempt(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
