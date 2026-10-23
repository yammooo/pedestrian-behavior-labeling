"""Four bounded integration attempts: first/repeated prepared reads, source validation only."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import load_predictions, prediction_metrics
from .run import write_json
from .e002a_run import DEFAULT_POINT_WORKERS


def selection(root, probe, fixture):
    native = json.loads((probe / "probe.json").read_text())
    sample = json.loads((fixture / "execution.json").read_text())
    if native["status"] != "complete" or sample["status"] != "complete":
        raise ValueError("Completed existing development selections required")
    selected = [{"dataset": t["dataset"], "locator": t["locator"]} for t in sample["tracks"]]
    for dataset in ("loki", "road-waymo"):
        setup = json.loads((root / f"experiments/E001/data-setup/{dataset}/setup.json").read_text())
        val_groups = {k for k, v in native["samples"][dataset].items() if v["split"] == "validation"}
        choices = sorted((t for t in setup["tracks"] if not t["exclusions"] and t["split"] == "validation" and t["group"] in val_groups),
                         key=lambda t: (t["slots"], t["locator"]))
        if not choices:
            raise ValueError("No previously inspected source validation group")
        selected.append({"dataset": dataset, "locator": choices[len(choices) // 2]["locator"]})
    return {"scope": "three existing development training tracks plus median-length track in prior validation groups per source",
            "tracks": selected, "probe_sha256": checksum(probe / "probe.json"), "fixture_sha256": checksum(fixture / "execution.json")}


def audit(output, source):
    if json.loads((output / "status.json").read_text())["status"] != "complete":
        raise ValueError("Smoke attempt did not complete")
    destination = output / (source + "-validation-smoke")
    saved = load_predictions(destination / "predictions.npz")
    expected = json.loads((destination / "metrics.json").read_text())["primary"]
    if prediction_metrics(saved, 4) != expected:
        raise ValueError("Saved predictions do not reconstruct metrics")
    history = json.loads((output / "history.json").read_text())
    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    row = history[checkpoint["epoch"] - 1]
    if expected["track_ce"] != row["validation_ce"] or expected["track_weighted"]["macro_f1"] != row["validation_f1"]:
        raise ValueError("Selected-checkpoint reload changed validation predictions")
    pngs, svgs = list((output / "plots").glob("*.png")), list((output / "plots").glob("*.svg"))
    if len(pngs) != len(svgs) or len(pngs) < 20 or any(p.stat().st_size == 0 for p in pngs + svgs):
        raise ValueError("Missing plot exports")
    if list(output.glob("*-test")):
        raise ValueError("Held-out output in source-validation smoke")
    if json.loads((output / "protected.json").read_text())["changed"]:
        raise ValueError("Frozen sources changed")
    for name in ("best.pt", "last.pt", "packing.jsonl", "steps.jsonl", "system.csv"):
        if not (output / name).is_file() or not (output / name).stat().st_size:
            raise ValueError("Missing smoke evidence: " + name)
    provenance = json.loads((output / "provenance.json").read_text())
    return {"source": source, "output": str(output), "tracks": len(saved), "png_svg_pairs": len(pngs),
            "wandb": provenance["wandb"], "seconds": provenance["elapsed_seconds"],
            "peak_cuda_allocated_bytes": provenance["peak_cuda_allocated_bytes"], "runtime": provenance["runtime"]}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    chosen = args.output / "selection.json"
    write_json(chosen, selection(args.outputs_root, args.probe, args.fixtures))
    record = {"status": "incomplete", "scope": "bounded source-validation-only operations, not behavior comparison results",
              "profile_sha256": checksum(args.profile), "attempts": [], "failures": []}
    try:
        for source in ("loki", "road-waymo"):
            for state in ("first", "repeat"):
                output = args.output / (source + "-" + state)
                command = [sys.executable, "scripts/train-e002a.py", "--source", source, "--arm", "g-rgb-dditr",
                    "--outputs-root", str(args.outputs_root), "--probe", str(args.probe),
                    "--device", "cuda", "--smoke", "--smoke-selection", str(chosen), "--profile", str(args.profile),
                    "--rgb-cache-root", str(args.rgb_cache_root), "--rgb-preparation", str(args.rgb_preparation),
                    "--indexed-point-cache", str(args.indexed_cache_root / ("indexed-" + source)), "--output", str(output)]
                command += ["--point-workers", str(args.point_workers)]
                with (args.output / (source + "-" + state + ".log")).open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                record["attempts"].append(audit(output, source) | {"state": state, "command": command})
                write_json(args.output / "integration.json", record)
                print(source, state, "passed", flush=True)
        record["status"] = "complete"
        return record
    except BaseException:
        record["status"] = "failed"; record["failures"].append(traceback.format_exc()); raise
    finally:
        write_json(args.output / "integration.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-preparation", type=Path, required=True)
    parser.add_argument("--rgb-cache-root", type=Path, default=Path("outputs/experiments/E002a/rgb-cache"))
    parser.add_argument("--indexed-cache-root", type=Path, default=Path("outputs/experiments/E002a"))
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--fixtures", type=Path, default=Path("outputs/experiments/E002a/execution-probe-04"))
    parser.add_argument("--point-workers", type=int, default=DEFAULT_POINT_WORKERS)
    run(parser.parse_args())
