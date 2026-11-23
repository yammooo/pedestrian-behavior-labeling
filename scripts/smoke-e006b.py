"""Run the three E005 checkpoints on a small Euro-PVI calibration smoke set."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from pedestrian_behavior.data.feature_cache import isolated_extraction
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import predict, save_predictions
from pedestrian_behavior.experiments.e005_cache import TinyUtoniaEncoder, atomic_json, collate_e005
from pedestrian_behavior.experiments.e005_transfer import DEFAULT_PROBE, DEFAULT_TINY_CHECKPOINT, DEFAULT_UTONIA_CODE
from pedestrian_behavior.experiments.e006b_euro_pvi import EuroPVIAdapter, sample_for
from pedestrian_behavior.inspection.pretrained import ImageEncoder
from pedestrian_behavior.models.e005 import build_model
from pedestrian_behavior.training import seed_run


DATASET = Path("/run/media/user20/T9/ped_intention_1000")
SEQUENCES = ("train/0067", "train/0158", "train/0339")
RUNS = {
    "road-waymo": "road-waymo-g-rgb-utonia-s0-20261120-030721",
    "loki": "loki-g-rgb-utonia-s0-20261118-074440",
    "joint": "joint-g-rgb-utonia-s0-20261122-001910",
}


def load_sources(root, device, dino_sha, tiny_sha):
    result = {}
    for source, name in RUNS.items():
        run = root / "experiments/E005/runs" / name
        path = run / "best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        config = saved["config"]
        assert json.loads((run / "status.json").read_text())["status"] == "complete"
        assert config["source"] == source and config["arm"] == "g-rgb-utonia"
        caches = config["caches"].values() if source == "joint" else (config["cache"],)
        for cache in caches:
            frozen = cache["provenance"]
            assert frozen["DINO"]["checkpoint_sha256"] == dino_sha
            assert frozen["Tiny-Utonia"]["checkpoint_sha256"] == tiny_sha
        model = build_model(config["arm"], config["profile"]).to(device)
        model.load_state_dict(saved["model_state"], strict=True)
        result[source] = (model.eval(), saved["normalization"], {
            "path": str(path.resolve()), "sha256": checksum(path), "selected_epoch": saved["epoch"]})
    return result


def run(args):
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("The E006b smoke test uses the Aalto CUDA encoder environment")
    if args.output.exists():
        raise FileExistsError(f"Use a fresh smoke output directory: {args.output}")
    torch.set_num_threads(2)
    seed_run(0)
    device = torch.device("cuda")
    root = args.outputs_root.resolve()
    dataset = args.dataset.resolve()
    native = json.loads((root / DEFAULT_PROBE / "probe.json").read_text())["config"]
    tiny_path = root / DEFAULT_TINY_CHECKPOINT
    dino_sha, tiny_sha = checksum(native["dino_weights"]), checksum(tiny_path)
    sources = load_sources(root, device, dino_sha, tiny_sha)
    with isolated_extraction():
        dino = ImageEncoder(native["dino_code"], native["dino_weights"], device)
        tiny = TinyUtoniaEncoder(root / DEFAULT_UTONIA_CODE, tiny_path, device)

    adapter = EuroPVIAdapter(dataset)
    predictions = {source: [] for source in sources}
    support = {}
    for sequence_id in SEQUENCES:
        sequence = adapter.load_sequence(sequence_id)
        with isolated_extraction():
            records = adapter.encode_sequence(sequence, dino, tiny, device)
        for source, (model, normalization, _) in sources.items():
            samples = [sample_for(record, normalization) for record in records]
            if source == "road-waymo":
                counts = Counter()
                for sample in samples:
                    mask = sample["flags"].bool().numpy()
                    counts["tracks"] += 1
                    counts["slots"] += len(mask)
                    counts["gt_slots"] += int(sample["gt_valid"].sum())
                    counts["geometry_valid"] += int(mask[:, 0].sum())
                    counts["rgb_valid"] += int(mask[:, 1].sum())
                    counts["lidar_valid"] += int(mask[:, 2].sum())
                    counts["lidar_only"] += int((~mask[:, 1] & mask[:, 2]).sum())
                    counts["neither_spatial_modality"] += int((~mask[:, 1] & ~mask[:, 2]).sum())
                    for level in sample["lidar_levels"]:
                        assert all(not torch.any(tokens[:, -1]) for tokens in level)
                support[sequence_id] = dict(counts)
            batches = (collate_e005(samples[i:i + 2]) for i in range(0, len(samples), 2))
            rows = predict(model, batches, device, forward_batch=lambda m, b, _: m(b))
            for track in rows:
                track["conditions"] = {}
            predictions[source].extend(rows)
        print(sequence_id, support[sequence_id], flush=True)

    totals = Counter()
    for counts in support.values():
        totals.update(counts)
    reference = predictions["road-waymo"]
    assert totals["tracks"] == 5 and totals["slots"] == totals["gt_slots"] == 100
    assert tiny.calls == 84
    assert totals["lidar_only"] and totals["neither_spatial_modality"]
    assert totals["rgb_valid"] and totals["lidar_valid"] and totals["geometry_valid"]
    assert {int(label) for row in reference for label in row["targets"][row["gt_valid"]]} == {0, 1, 2, 3}
    for source, rows in predictions.items():
        for row, expected in zip(rows, reference, strict=True):
            assert row["locator"] == expected["locator"] and row["group"] == expected["group"]
            assert len(row["frame_indices"]) == len(row["targets"])
            assert np.all(np.diff(row["frame_indices"]) == 2)
            assert np.array_equal(row["targets"], expected["targets"])
            assert np.array_equal(row["gt_valid"], expected["gt_valid"])
            assert row["logits"].shape == (len(row["targets"]), 4) and np.isfinite(row["logits"]).all()
            for flag in ("rgb_valid", "lidar_valid"):
                assert np.array_equal(row["source_flags"][flag], expected["source_flags"][flag])

    args.output.mkdir(parents=True, exist_ok=False)
    for source, rows in predictions.items():
        save_predictions(args.output / f"{source}-predictions.npz", rows)
    atomic_json(args.output / "summary.json", {
        "status": "complete", "kind": "calibration-sequence-smoke", "dataset": str(dataset),
        "release_sha256": {name: checksum(dataset / name) for name in ("calib.json", "labels/frames.csv")},
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sequences": support, "totals": dict(totals), "models": {source: item[2] for source, item in sources.items()},
        "encoder_sha256": {"dino": dino_sha, "tiny_utonia": tiny_sha}, "tiny_utonia_sweeps": tiny.calls,
        "inside_box_cue": 0, "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
    })
    print(f"Smoke complete: {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
