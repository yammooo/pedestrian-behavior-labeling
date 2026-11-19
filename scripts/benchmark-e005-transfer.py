"""Time online E005 evaluation on two fixed-size strata of frozen ROAD validation clips."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import tempfile
import threading
import time
import types
from unittest.mock import patch

import numpy as np
import torch

from pedestrian_behavior.experiments import e005_cache, e005_transfer


def benchmark(args):
    module = e005_transfer
    if args.revision:
        module = types.ModuleType(e005_transfer.__name__)
        source = subprocess.check_output([
            "git", "show", f"{args.revision}:src/pedestrian_behavior/experiments/e005_transfer.py",
        ])
        module.__package__ = e005_transfer.__package__
        exec(compile(source, "e005_transfer.py", "exec"), module.__dict__)
    seconds, calls = Counter(), Counter()
    lock = threading.Lock()
    groups, shapes = [], {}
    digest = hashlib.blake2b()

    def record(stage, started):
        duration = time.perf_counter() - started
        with lock:
            seconds[stage] += duration
            calls[stage] += 1

    def timed(stage, function):
        def measure(*values, **kwargs):
            started = time.perf_counter()
            try:
                return function(*values, **kwargs)
            finally:
                record(stage, started)
        return measure

    def timed_frames(stage, function):
        def measure(*values, **kwargs):
            frames = iter(function(*values, **kwargs))
            while True:
                started = time.perf_counter()
                frame = next(frames, None)
                if frame is None:
                    break
                record(stage, started)
                yield frame
        return measure

    index_validation = module._index_validation

    def subset(baseline):
        slots = Counter()
        for entry in baseline.entries:
            slots[entry["group"]] += entry["slots"]
        ordered = sorted(slots, key=lambda group: (slots[group], group))
        groups.extend(ordered[index] for index in (len(ordered) // 2, 3 * len(ordered) // 4))
        baseline.entries = [entry for entry in baseline.entries if entry["group"] in groups]
        for group in groups:
            entries = [entry for entry in baseline.entries if entry["group"] == group]
            shapes[group] = {"tracks": len(entries), "slots": slots[group]}
        print(f"Benchmark clips {shapes}", flush=True)
        return index_validation(baseline)

    batches = module._batches

    def measured_batches(*values, **kwargs):
        for batch in batches(*values, **kwargs):
            started = time.perf_counter()
            for tensor in (batch["geometry"], batch["flags"], *batch["rgb_frames"],
                           *batch["lidar_levels"][0], *batch["lidar_levels"][1]):
                digest.update(tensor.contiguous().view(-1).numpy().data.cast("B"))
            record("input_check", started)
            yield batch

    predict = module.predict

    def measured_predict(model, batches, device, forward_batch):
        def forward(*values):
            started = time.perf_counter()
            result = forward_batch(*values)
            torch.cuda.synchronize()
            record("behavior_forward", started)
            return result

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return timed("online_inference", predict)(model, batches, device, forward_batch=forward)

    root = args.outputs_root.resolve()
    temp_root = root / "experiments/E005/transfer"
    temp_root.mkdir(parents=True, exist_ok=True)
    bench_output = args.output.resolve()
    bench_output.mkdir(parents=True, exist_ok=False)
    args.device, args.probe, args.utonia_code, args.tiny_checkpoint = "cuda", None, None, None
    with tempfile.TemporaryDirectory(prefix="timing-", dir=temp_root) as temp:
        args.output = Path(temp) / "subset"
        with (patch.object(module, "_index_validation", subset),
              patch.object(module, "_batches", measured_batches),
              patch.object(module, "predict", measured_predict),
              patch.object(module, "native_scene", timed_frames("native_lidar_decode", module.native_scene)),
              patch.object(module, "native_camera", timed_frames("native_camera_decode", module.native_camera)),
              patch.object(module, "normalized_crop", timed("rgb_crop_resize", module.normalized_crop)),
              patch.object(module, "collate_e005", timed("collate_only", module.collate_e005)),
              patch.object(module.RoadSpatialFeatures, "lidar_frame", timed("pedestrian_lidar_crop", module.RoadSpatialFeatures.lidar_frame)),
              patch.object(e005_cache, "voxel_sample", timed("cpu_voxel", e005_cache.voxel_sample)),
              patch.object(module.TinyUtoniaEncoder, "stages", timed("tiny_with_voxel", module.TinyUtoniaEncoder.stages)),
              patch.object(module.ImageEncoder, "descriptors", timed("dino", module.ImageEncoder.descriptors))):
            timed("total", module.run)(args)
        cell = args.output / f"{module.TARGET}-{module.SPLIT}"
        metrics = json.loads((cell / "metrics.json").read_text())
        with np.load(cell / "predictions.npz") as prediction:
            np.savez(bench_output / "predictions.npz", **{name: prediction[name] for name in
                     ("logits", "offsets", "targets", "gt_valid", "metadata", "source_rgb_valid", "source_lidar_valid")})
    result = {"revision": args.revision or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "checkpoint": str(args.source_run), "clips": shapes, "gpu": torch.cuda.get_device_name(),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
              "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "seconds": {**seconds, "tiny_excluding_voxel": seconds["tiny_with_voxel"] - seconds["cpu_voxel"],
                          "online_inference_excluding_input_check": seconds["online_inference"] - seconds["input_check"]},
              "calls": calls, "input_digest": digest.hexdigest(), "macro_f1": metrics["primary"]["track_weighted"]["macro_f1"],
              "rgb_crops": metrics["rgb_crops"], "lidar_sweeps": metrics["lidar_sweeps"]}
    if args.reference:
        with (np.load(args.reference / "predictions.npz") as old,
              np.load(bench_output / "predictions.npz") as new):
            delta = abs(old["logits"] - new["logits"])
            result["parity"] = {"max_absolute_logit_difference": float(delta.max()),
                                "mean_absolute_logit_difference": float(delta.mean()),
                                "argmax_differences": int((old["logits"].argmax(1) != new["logits"].argmax(1)).sum()),
                                "identical_non_logits": all(np.array_equal(old[name], new[name]) for name in old.files if name != "logits"),
                                "identical_model_inputs": result["input_digest"] == json.loads(
                                    (args.reference / "benchmark.json").read_text())["input_digest"]}
    module.write_json(bench_output / "benchmark.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if args.reference:
        parity = result["parity"]
        assert parity["identical_non_logits"] and parity["identical_model_inputs"]
        assert parity["max_absolute_logit_difference"] < 1e-4 and parity["argmax_differences"] == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--waymo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--revision", help="Use the transfer implementation from this Git revision")
    parser.add_argument("--reference", type=Path)
    benchmark(parser.parse_args())
