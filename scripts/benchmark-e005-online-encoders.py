"""Bounded Tiny Utonia and DINOv3 timing on cached native probe inputs."""

import argparse
from datetime import datetime, timezone
import gc
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.inspection.feature_probe import RGB_MEAN, RGB_STD, voxel_sample
from pedestrian_behavior.inspection.pretrained import (
    DINO_REVISION,
    ImageEncoder,
    UTONIA_REVISION,
    verify_code,
)

TINY_SHA256 = "e49a1944403d4ec3df459537fb7d5b132326736d0ebdb7c5a74868f3fa50d137"
TINY_REVISION = "9457f025e36c7001ee962a542f0d27a5b600e1e7"
REPEATS = 3
CAP_BYTES = int(1.5 * 1024**3)
SAMPLES = {
    "loki": "loki-scenario_025-0052-spatial.npz",
    "road-waymo": "road-waymo-train_00014-1511379235561640-spatial.npz",
}


def sync(device):
    torch.cuda.synchronize(device)


def timed(fn, device=None):
    if device is not None:
        sync(device)
    fn()  # One warmup, then three measurements.
    if device is not None:
        sync(device)
    values, result = [], None
    for _ in range(REPEATS):
        if device is not None:
            sync(device)
        result = None
        start = time.perf_counter()
        result = fn()
        if device is not None:
            sync(device)
        values.append(time.perf_counter() - start)
    return values, result


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        return out
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def cpu_voxels(native_xyz):
    points = np.c_[native_xyz, np.zeros(len(native_xyz), dtype=np.float32)]
    assert points.ndim == 2 and points.shape[1] == 4 and np.isfinite(points).all()
    points[:, :3] *= .2
    selected, _, grid = voxel_sample(points, grid_size=.01)
    return points, selected, grid


def read_native_xyz(path):
    with np.load(path, allow_pickle=False) as saved:
        points = np.asarray(saved["native_xyz"], dtype=np.float32).copy()
    assert points.ndim == 2 and points.shape[1] == 3 and np.isfinite(points).all()
    return points


def tiny_timing(args, device, result):
    code = verify_code(args.utonia_code, UTONIA_REVISION)
    if checksum(args.tiny_checkpoint) != TINY_SHA256:
        raise ValueError("Tiny checkpoint SHA-256 does not match pinned Hugging Face metadata")
    package = types.ModuleType("utonia")
    package.__path__ = [str(code / "utonia")]
    sys.modules["utonia"] = package
    author = importlib.import_module("utonia.model")
    ckpt = torch.load(args.tiny_checkpoint, map_location="cpu", weights_only=True)
    config = dict(ckpt["config"])
    channels, depths, strides = tuple(config["enc_channels"]), tuple(config["enc_depths"]), tuple(config["stride"])
    assert channels == (18, 36, 72, 144, 216)
    assert depths == (2, 2, 2, 6, 2) and strides == (2, 2, 2, 2)
    config["shuffle_orders"] = False
    model = author.PointTransformerV3(**config)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    del ckpt
    model.requires_grad_(False).eval().to(device)
    for module in model.modules():
        if hasattr(module, "shuffle_orders"):
            module.shuffle_orders = False

    frame_results = []
    for dataset, filename in SAMPLES.items():
        native_xyz = read_native_xyz(args.fixtures / filename)
        prep_times, prepared = timed(lambda: cpu_voxels(native_xyz))
        points, selected, grid = prepared

        def h2d():
            coord = torch.from_numpy(points[selected, :3]).to(device)
            data = {
                "coord": coord,
                "grid_coord": torch.from_numpy(grid).int().to(device),
                "feat": torch.cat((coord, torch.zeros_like(coord), torch.zeros_like(coord)), dim=1),
                "offset": torch.tensor([len(selected)], device=device),
            }
            return data

        h2d_times, data = timed(h2d, device)
        torch.cuda.reset_peak_memory_stats(device)

        def forward():
            with torch.inference_mode():
                return model(data)

        forward_times, point = timed(forward, device)

        hierarchy = [point]
        while "pooling_parent" in hierarchy[-1]:
            hierarchy.append(hierarchy[-1]["pooling_parent"])
        hierarchy.reverse()
        assert len(hierarchy) == 5
        assert hierarchy[2]["feat"].shape[1] == channels[2]
        assert hierarchy[4]["feat"].shape[1] == channels[4]

        def stage_copy():
            stage3 = hierarchy[2]["feat"].to(torch.float16).cpu().numpy()
            stage5 = hierarchy[4]["feat"].to(torch.float16).cpu().numpy()
            assert stage3.dtype == stage5.dtype == np.float16
            assert np.isfinite(stage3).all() and np.isfinite(stage5).all()
            return stage3, stage5

        copy_times, arrays = timed(stage_copy, device)
        frame_results.append({
            "dataset": dataset,
            "fixture": filename,
            "input": "preloaded full-sweep XYZ fixture; NPZ read excluded, zero strength added for the existing voxel helper",
            "native_points": len(points),
            "input_voxels": len(selected),
            "stage3_rows_channels": list(hierarchy[2]["feat"].shape),
            "stage5_rows_channels": list(hierarchy[4]["feat"].shape),
            "stage3_stage5_fp16_bytes": int(arrays[0].nbytes + arrays[1].nbytes),
            "cpu_voxel_prep_seconds": prep_times,
            "h2d_and_input_assembly_seconds": h2d_times,
            "encoder_forward_seconds": forward_times,
            "stage3_stage5_fp16_copy_seconds": copy_times,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        })
        del arrays, point, hierarchy, data, points, prepared, native_xyz

    result["utonia_tiny"] = {
        "checkpoint_sha256": TINY_SHA256,
        "checkpoint_revision": TINY_REVISION,
        "code_revision": UTONIA_REVISION,
        "channels": channels,
        "depths": depths,
        "strides": strides,
        "stage3_stage5_channels": [channels[2], channels[4]],
        "frames": frame_results,
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()


def dino_timing(args, device, result):
    encoder = ImageEncoder(args.dino_code, args.dino_checkpoint, device)
    rng = torch.Generator().manual_seed(20261117)
    pixels = torch.rand((8, 3, 256, 256), generator=rng)
    mean = torch.tensor(RGB_MEAN)[None, :, None, None]
    std = torch.tensor(RGB_STD)[None, :, None, None]
    crops = (pixels - mean) / std
    timings = {}
    for batch_size in (1, 8):
        batch = crops[:batch_size]

        h2d_times, batch_gpu = timed(lambda: batch.to(device), device)

        def forward_only():
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                output = encoder.model.forward_features(batch_gpu)
                cls, patches = output["x_norm_clstoken"], output["x_norm_patchtokens"]
            assert cls.shape == (batch_size, 768) and patches.shape == (batch_size, 256, 768)
            assert torch.isfinite(cls).all() and torch.isfinite(patches).all()
            return cls, patches

        torch.cuda.reset_peak_memory_stats(device)
        forward_times, output_gpu = timed(forward_only, device)
        forward_peak = torch.cuda.max_memory_allocated(device)
        del output_gpu, batch_gpu
        sync(device)

        def forward_copy():
            cls, patches = encoder.descriptors(batch, patches=True)
            cls, patches = cls.astype(np.float16), patches.astype(np.float16)
            assert patches.shape == (batch_size, 16, 16, 768)
            assert np.isfinite(cls).all() and np.isfinite(patches).all()
            return cls, patches

        torch.cuda.reset_peak_memory_stats(device)
        seconds, arrays = timed(forward_copy, device)
        timings[str(batch_size)] = {
            "input": "preloaded synthetic normalized 256x256 crops; no native image I/O or crop preprocessing",
            "cls_shape": list(arrays[0].shape),
            "patch_shape": list(arrays[1].shape),
            "fp16_output_bytes": int(arrays[0].nbytes + arrays[1].nbytes),
            "h2d_seconds": h2d_times,
            "pure_forward_seconds": forward_times,
            "adapter_forward_fp32_d2h_then_cpu_fp16_seconds": seconds,
            "pure_forward_peak_allocated_bytes": forward_peak,
            "adapter_copy_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        }
        del arrays
    result["dinov3"] = {
        "code_revision": DINO_REVISION,
        "checkpoint_sha256": checksum(args.dino_checkpoint),
        "timings": timings,
    }
    del encoder, crops
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--utonia-code", type=Path, required=True)
    parser.add_argument("--tiny-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-code", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    cap = min(CAP_BYTES, total)
    torch.cuda.set_per_process_memory_fraction(cap / total, device)
    free, total = torch.cuda.mem_get_info(device)
    if free < cap + 512 * 1024**2:
        raise RuntimeError(f"Insufficient free GPU memory for capped probe: {free} bytes free")
    result = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "allocator_cap_bytes": cap,
        "gpu_free_before_bytes": free,
        "gpu_total_bytes": total,
        "gpu_snapshot_before": gpu_snapshot(),
        "warmup": 1,
        "repeats": REPEATS,
        "contention": "Concurrent E004c training remained active; timings are contended.",
    }
    try:
        tiny_timing(args, device, result)
        dino_timing(args, device, result)
        result["status"] = "complete"
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom-no-retry"
        result["error"] = repr(exc)
        raise
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["gpu_snapshot_after"] = gpu_snapshot()
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
