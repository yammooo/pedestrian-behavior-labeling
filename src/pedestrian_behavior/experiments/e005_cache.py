"""E005 frozen spatial features, scene-shared storage, and model batch adapter."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import types

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from pedestrian_behavior.data.preparation import checksum, valid_pose
from pedestrian_behavior.data.feature_cache import isolated_extraction
from pedestrian_behavior.experiments.e002a_features import FeatureCatalog, load_configuration
from pedestrian_behavior.inspection.feature_probe import normalized_crop, relative_xyz, square_crop, voxel_sample
from pedestrian_behavior.inspection.native_features import native_camera, native_scene
from pedestrian_behavior.inspection.pretrained import (
    DINO_REVISION,
    ImageEncoder,
    UTONIA_REVISION,
    verify_code,
)


FORMAT = "e005-loki-spatial-v1"
TINY_SHA256 = "e49a1944403d4ec3df459537fb7d5b132326736d0ebdb7c5a74868f3fa50d137"
TINY_REVISION = "9457f025e36c7001ee962a542f0d27a5b600e1e7"
DINO_BATCH = 8
ARRAYS = {
    "rgb_features": ("<f2", (256, 768)),
    "rgb_boxes": ("<f4", (4,)),
    "stage3_features": ("<f2", (72,)),
    "stage3_coords": ("<f4", (3,)),
    "stage5_features": ("<f2", (216,)),
    "stage5_coords": ("<f4", (3,)),
    "stage3_indices": ("<u4", ()),
    "stage5_indices": ("<u4", ()),
}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, separators=(",", ":"), allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class TinyUtoniaEncoder:
    """Pinned Tiny checkpoint; returns its native stage-3/stage-5 tokens and centroids."""

    def __init__(self, code, checkpoint, device):
        self.code = verify_code(code, UTONIA_REVISION)
        if checksum(checkpoint) != TINY_SHA256:
            raise ValueError("Tiny-Utonia checkpoint SHA-256 mismatch")
        package = types.ModuleType("utonia")
        package.__path__ = [str(self.code / "utonia")]
        sys.modules["utonia"] = package
        author = importlib.import_module("utonia.model")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = dict(saved["config"])
        if (tuple(config["enc_channels"]) != (18, 36, 72, 144, 216)
                or tuple(config["enc_depths"]) != (2, 2, 2, 6, 2)
                or tuple(config["stride"]) != (2, 2, 2, 2)):
            raise ValueError("Unexpected Tiny-Utonia architecture")
        config["shuffle_orders"] = False
        self.model = author.PointTransformerV3(**config)
        self.model.load_state_dict(saved["state_dict"], strict=True)
        del saved
        self.model.requires_grad_(False).eval().to(device)
        for module in self.model.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = False
        self.device = torch.device(device)
        self.calls = 0
        self.seconds = 0.

    def stages(self, native_points):
        points = np.asarray(native_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
            raise ValueError("Expected finite full-sweep XYZ+strength points")
        if not len(points):
            return ((np.zeros((0, 72), np.float16), np.zeros((0, 3), np.float32)),
                    (np.zeros((0, 216), np.float16), np.zeros((0, 3), np.float32)))
        started = time.perf_counter()
        scaled = points.copy()
        scaled[:, :3] *= .2
        selected, _, grid = voxel_sample(scaled, grid_size=.01)
        coord = torch.from_numpy(scaled[selected, :3]).to(self.device)
        zeros = torch.zeros_like(coord)
        data = {"coord": coord, "grid_coord": torch.from_numpy(grid).int().to(self.device),
                "feat": torch.cat((coord, zeros, zeros), dim=1),
                "offset": torch.tensor([len(selected)], device=self.device)}
        with torch.inference_mode():
            point = self.model(data)
        hierarchy = [point]
        while "pooling_parent" in hierarchy[-1]:
            hierarchy.append(hierarchy[-1]["pooling_parent"])
        hierarchy.reverse()
        if len(hierarchy) != 5:
            raise ValueError("Expected five Tiny-Utonia levels")
        result = []
        for level, width in ((2, 72), (4, 216)):
            feature = hierarchy[level]["feat"].float().cpu().numpy()
            # Author coordinates use the .2-scaled frame; restore native metres for selection/readout.
            coords = hierarchy[level]["coord"].float().cpu().numpy() * 5.
            if feature.shape != (len(coords), width) or not np.isfinite(feature).all() or not np.isfinite(coords).all():
                raise ValueError("Invalid Tiny-Utonia spatial output")
            result.append((feature.astype(np.float16), coords.astype(np.float32)))
        self.calls += 1
        self.seconds += time.perf_counter() - started
        return tuple(result)


def rgb_crop_box(box, image_size, crop_bounds):
    """Native normalized xyxy mapped through the padded full-square crop."""
    box = np.asarray(box, dtype=np.float64)
    size = np.asarray(image_size, dtype=np.float64)
    bounds = np.asarray(crop_bounds, dtype=np.float64)
    if box.shape != (4,) or size.shape != (2,) or bounds.shape != (4,) or not np.isfinite([*box, *size, *bounds]).all():
        raise ValueError("Invalid RGB crop geometry")
    pixels = box * np.tile(size, 2)
    side = bounds[2] - bounds[0]
    if side <= 0 or pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        raise ValueError("Invalid target box/crop transform")
    return ((pixels - np.tile(bounds[:2], 2)) / side).astype(np.float32)


def rgb_patch_cues(box_crop):
    box = np.asarray(box_crop, dtype=np.float32)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
        raise ValueError("Invalid crop-relative target box")
    x = (np.arange(16, dtype=np.float32) + .5) / 16
    y = (np.arange(16, dtype=np.float32) + .5) / 16
    xx, yy = np.meshgrid(x, y)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    dx, dy = xx - cx, yy - cy
    left, top = np.arange(16, dtype=np.float32) / 16, np.arange(16, dtype=np.float32) / 16
    right, bottom = left + 1 / 16, top + 1 / 16
    overlap_x = np.maximum(0, np.minimum(right[None, :], box[2]) - np.maximum(left[None, :], box[0]))
    overlap_y = np.maximum(0, np.minimum(bottom[:, None], box[3]) - np.maximum(top[:, None], box[1]))
    overlap = np.clip(overlap_x * overlap_y * 256, 0, 1)
    return np.stack((dx, dy, dx * dx + dy * dy, overlap), axis=-1).reshape(256, 4)


def lidar_cues(coords, box, world_from_native, anchor):
    center, size, yaw = np.asarray(box[:3]), np.asarray(box[3:6]), float(box[6])
    relative = relative_xyz(coords, center, world_from_native, anchor)
    cues = np.column_stack((relative / 20,
                            (relative[:, 0] ** 2 + relative[:, 1] ** 2) / 400,
                            __import__("pedestrian_behavior.inspection.feature_probe", fromlist=["box_membership"]).box_membership(
                                coords, center, size, yaw)))
    if not np.isfinite(cues).all():
        raise ValueError("Invalid LiDAR spatial cues")
    return cues.astype(np.float32)


def rgb_tokens(patches, box_crop):
    """Shared cache/online adapter: DINO patch rows plus E005a geometry cues."""
    patches = np.asarray(patches)
    if patches.shape != (256, 768):
        raise ValueError("Expected one 16x16 DINO patch grid")
    return np.concatenate((patches, rgb_patch_cues(box_crop)), axis=1, dtype=np.float16)


def lidar_tokens(features, coords, box, world_from_native, anchor):
    """Shared cache/online adapter for one native stage and target selection."""
    features, coords = np.asarray(features), np.asarray(coords)
    if features.ndim != 2 or coords.shape != (len(features), 3):
        raise ValueError("LiDAR stage features and representative coordinates differ")
    return np.concatenate((features, lidar_cues(coords, box, world_from_native, anchor)), axis=1, dtype=np.float16)


class E005Writer:
    """Append unique scene/frame tokens to global files; publish manifests last."""

    def __init__(self, root, min_free_bytes):
        self.root = Path(root)
        self.min_free_bytes = min_free_bytes
        self.arrays_root = self.root / "arrays"
        self.arrays_root.mkdir()
        self.streams = {name: (self.arrays_root / f"{name}.bin").open("xb") for name in ARRAYS}
        self.hashes = {name: hashlib.sha256() for name in ARRAYS}
        self.rows = {name: 0 for name in ARRAYS}
        self.byte_counts = {name: 0 for name in ARRAYS}
        self.scenes = {}
        self.summaries = {"rgb_crops": 0, "stage3_tokens": 0, "stage5_tokens": 0,
                          "stage3_selection_rows": 0, "stage5_selection_rows": 0,
                          "rgb_candidates": 0, "point_sweeps": 0, "point_targets": 0}
        self.scene = None
        self.frames = None

    def begin_scene(self, group):
        if self.scene is not None or group in self.scenes:
            raise ValueError("Duplicate/nested E005 scene")
        self.scene = group
        self.frames = {}
        self.scene_start = dict(self.rows)
        self.summary_start = dict(self.summaries)

    def _append(self, name, value, *, check_space=True):
        dtype, tail = ARRAYS[name]
        array = np.asarray(value, dtype=dtype)
        expected = (array.shape[0], *tail)
        if array.shape != expected or not np.isfinite(array).all():
            raise ValueError(f"Invalid/nonfinite E005 {name}")
        if array.shape[0] == 0:
            return self.rows[name]
        raw = memoryview(np.ascontiguousarray(array)).cast("B")
        if check_space:
            free = shutil.disk_usage(self.root).free
            if free - len(raw) < self.min_free_bytes:
                raise OSError(f"E005 cache would cross {self.min_free_bytes} byte free-space floor")
        self.streams[name].write(raw)
        self.hashes[name].update(raw)
        self.rows[name] += array.shape[0]
        self.byte_counts[name] += len(raw)
        return self.rows[name] - array.shape[0]

    def add_rgb(self, key, locator, patches, box_crop):
        self.add_rgb_batch(key, [(locator, box_crop)], np.asarray(patches, dtype=np.float16)[None])

    def add_rgb_batch(self, key, items, patches):
        record = self.frames.setdefault(str(key), {"rgb": {}, "targets": {}})
        patches = np.asarray(patches, dtype=np.float16)
        if patches.shape != (len(items), 256, 768) or not np.isfinite(patches).all():
            raise ValueError("Invalid DINO patch features")
        if len({locator for locator, _ in items}) != len(items) or any(locator in record["rgb"] for locator, _ in items):
            raise ValueError("Duplicate cached RGB crop")
        boxes = np.asarray([box for _, box in items], dtype=np.float32)
        row = self._append("rgb_features", patches)
        box_row = self._append("rgb_boxes", boxes)
        if row != box_row:
            raise AssertionError("RGB feature/geometry rows diverged")
        for index, (locator, _) in enumerate(items):
            if locator in record["rgb"]:
                raise ValueError("Duplicate cached RGB crop")
            record["rgb"][locator] = row + index
        self.summaries["rgb_crops"] += len(items)

    def add_points(self, key, stage3, stage5, targets, world_from_native):
        record = self.frames.setdefault(str(key), {"rgb": {}, "targets": {}})
        if "lidar" in record:
            raise ValueError("Duplicate cached LiDAR sweep")
        start3 = self._append("stage3_features", stage3[0])
        coord_start3 = self._append("stage3_coords", stage3[1])
        start5 = self._append("stage5_features", stage5[0])
        coord_start5 = self._append("stage5_coords", stage5[1])
        if (start3, coord_start3, start5, coord_start5) != (start3, start3, start5, start5):
            raise AssertionError("LiDAR feature/coordinate rows diverged")
        record["lidar"] = {"world_from_native": np.asarray(world_from_native).tolist(),
            "stage3_start": start3, "stage3_count": len(stage3[0]),
            "stage5_start": start5, "stage5_count": len(stage5[0])}
        selected_targets = []
        for locator, target in sorted(targets.items()):
            raw3, raw5 = np.asarray(target["indices3"]), np.asarray(target["indices5"])
            if (len(stage3[0]) >= 2**32 or len(stage5[0]) >= 2**32
                    or raw3.ndim != 1 or raw5.ndim != 1
                    or (raw3 < 0).any() or (raw5 < 0).any()
                    or (raw3 >= 2**32).any() or (raw5 >= 2**32).any()):
                raise ValueError("E005 target selection outside uint32 row range")
            indices3 = raw3.astype(np.uint32)
            indices5 = raw5.astype(np.uint32)
            if ((indices3 >= len(stage3[0])).any() or (indices5 >= len(stage5[0])).any()
                    or (np.diff(indices3.astype(np.int64)) <= 0).any()
                    or (np.diff(indices5.astype(np.int64)) <= 0).any()):
                raise ValueError("Invalid per-target E005 selection")
            selected_targets.append((locator, target, indices3, indices5))
        free = shutil.disk_usage(self.root).free
        needed = 4 * sum(len(indices3) + len(indices5) for _, _, indices3, indices5 in selected_targets)
        if free - needed < self.min_free_bytes:
            raise OSError(f"E005 cache would cross {self.min_free_bytes} byte free-space floor")
        next_i3, next_i5 = self.rows["stage3_indices"], self.rows["stage5_indices"]
        for locator, target, indices3, indices5 in selected_targets:
            start_i3, start_i5 = next_i3, next_i5
            self._append("stage3_indices", indices3, check_space=False)
            self._append("stage5_indices", indices5, check_space=False)
            next_i3 += len(indices3)
            next_i5 += len(indices5)
            record["targets"][locator] = {"box": target["box"], "anchor": np.asarray(target["anchor"]).tolist(),
                "stage3_indices": [start_i3, len(indices3)], "stage5_indices": [start_i5, len(indices5)]}
            self.summaries["stage3_selection_rows"] += len(indices3)
            self.summaries["stage5_selection_rows"] += len(indices5)
        self.summaries["stage3_tokens"] += len(stage3[0])
        self.summaries["stage5_tokens"] += len(stage5[0])
        self.summaries["point_sweeps"] += 1
        self.summaries["point_targets"] += len(targets)

    def finish_scene(self, source_sha256):
        if self.scene is None:
            raise RuntimeError("No E005 scene is open")
        safe = Path(self.scene)
        if safe.name != self.scene or safe.name in (".", ".."):
            raise ValueError("Unsafe E005 scene ID")
        folder = self.root / "scenes" / safe
        folder.mkdir(parents=True)
        record = {"format": FORMAT, "group": self.scene, "source_stat_sha256": source_sha256,
                  "array_starts": self.scene_start, "array_rows": {k: self.rows[k] - self.scene_start[k] for k in ARRAYS},
                  "summary": {k: self.summaries[k] - self.summary_start[k] for k in self.summaries},
                  "frames": self.frames}
        atomic_json(folder / "manifest.json", record)
        self.scenes[self.scene] = {"path": f"scenes/{safe}/manifest.json", "sha256": checksum(folder / "manifest.json"),
                                   "frames": len(self.frames), "summary": record["summary"]}
        completed = self.scenes[self.scene]
        self.scene = self.frames = None
        return completed

    def finish(self):
        if self.scene is not None:
            raise RuntimeError("Open E005 scene at finish")
        for stream in self.streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        arrays = {}
        for name, (dtype, tail) in ARRAYS.items():
            digest = {"bytes": self.byte_counts[name], "sha256": self.hashes[name].hexdigest()}
            arrays[name] = {"dtype": dtype, "shape": [self.rows[name], *tail], **digest}
        payload_bytes = sum(spec["bytes"] for spec in arrays.values())
        return arrays, payload_bytes

    def close(self):
        for stream in self.streams.values():
            if not stream.closed:
                stream.close()


class E005Cache:
    """Read-only mmap supply for locally built, finite frozen features."""

    def __init__(self, root, *, allow_partial=False, verify_checksums=False):
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "manifest.json"
        self.manifest_sha256 = checksum(self.manifest_path)
        self.record = json.loads(self.manifest_path.read_text())
        if self.record.get("format") != FORMAT or self.record.get("status") not in (("complete", "bounded-native-check") if allow_partial else ("complete",)):
            raise ValueError("Incomplete or unsupported E005 cache")
        if set(self.record.get("arrays", {})) != set(ARRAYS):
            raise ValueError("E005 array inventory mismatch")
        tracks_path = self.root / "tracks.json"
        if checksum(tracks_path) != self.record["tracks_sha256"]:
            raise ValueError("E005 track manifest checksum mismatch")
        rows = json.loads(tracks_path.read_text())["tracks"]
        self.tracks = {t["locator"]: t for t in rows}
        if len(self.tracks) != len(rows) or any(type(t.get("slots")) is not int or t["slots"] < 1
                or len(t.get("source_keys", ())) != t["slots"] for t in rows):
            raise ValueError("Invalid/duplicate E005 track timeline")
        self.scenes, self.maps = {}, {}
        for name, (dtype, tail) in ARRAYS.items():
            spec = self.record["arrays"][name]
            path = self.root / "arrays" / f"{name}.bin"
            expected = int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize
            if ((spec["dtype"], spec["shape"][1:]) != (dtype, list(tail))
                    or spec["bytes"] != expected or path.stat().st_size != expected):
                raise ValueError(f"E005 array shape/size mismatch: {name}")
        if verify_checksums:
            self.verify()

    def _map(self, name):
        if name not in self.maps:
            spec = self.record["arrays"][name]
            path = self.root / "arrays" / f"{name}.bin"
            if path.stat().st_size != spec["bytes"]:
                raise ValueError(f"E005 array truncated: {name}")
            self.maps[name] = (np.memmap(path, mode="r", dtype=spec["dtype"], shape=tuple(spec["shape"]))
                               if spec["shape"][0] else np.empty(spec["shape"], dtype=spec["dtype"]))
        return self.maps[name]

    def verify(self):
        """Explicit full integrity check; excluded from the per-frame training path."""
        for name, spec in self.record["arrays"].items():
            if checksum(self.root / "arrays" / f"{name}.bin") != spec["sha256"]:
                raise ValueError(f"E005 array checksum mismatch: {name}")
        for group, spec in self.record["scenes"].items():
            relative = Path(spec["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe E005 scene manifest reference")
            if checksum(self.root / relative) != spec["sha256"]:
                raise ValueError(f"E005 scene manifest checksum mismatch: {group}")
        return True

    def _scene(self, group):
        if group not in self.scenes:
            spec = self.record["scenes"].get(group)
            if spec is None:
                raise KeyError(group)
            relative = Path(spec["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe E005 scene manifest reference")
            path = self.root / relative
            if checksum(path) != spec["sha256"]:
                raise ValueError(f"E005 scene manifest checksum mismatch: {group}")
            scene = json.loads(path.read_text())
            if scene["format"] != FORMAT or scene["group"] != group:
                raise ValueError("E005 scene identity mismatch")
            self.scenes[group] = scene
        return self.scenes[group]

    def _frame(self, locator, slot):
        track = self.tracks.get(locator)
        if track is None:
            raise KeyError(locator)
        if type(slot) is not int or not 0 <= slot < track["slots"]:
            raise ValueError("E005 slot outside complete track")
        key = track["source_keys"][slot]
        if key is None:
            return None, None
        frame = self._scene(track["group"])["frames"].get(str(key))
        return track, frame

    def rgb_frame(self, locator, slot):
        track, frame = self._frame(locator, slot)
        empty = np.zeros((0, 772), np.float16)
        row = frame["rgb"].get(locator) if frame is not None else None
        if row is None:
            return empty
        row = int(row)
        if not 0 <= row < self.record["arrays"]["rgb_features"]["shape"][0]:
            raise ValueError("E005 RGB row outside cache")
        patch = np.asarray(self._map("rgb_features")[row])
        box = np.asarray(self._map("rgb_boxes")[row])
        return rgb_tokens(patch, box)

    def lidar_frame(self, locator, slot):
        _, frame = self._frame(locator, slot)
        empty3, empty5 = np.zeros((0, 77), np.float16), np.zeros((0, 221), np.float16)
        target = frame["targets"].get(locator) if frame is not None else None
        if target is None:
            return empty3, empty5
        lidar = frame.get("lidar")
        if lidar is None:
            return empty3, empty5
        world = np.asarray(lidar["world_from_native"], dtype=np.float64)
        box, anchor = np.asarray(target["box"], dtype=np.float64), np.asarray(target["anchor"], dtype=np.float64)
        if box.shape != (7,) or anchor.shape != (4, 4) or world.shape != (4, 4) or not valid_pose(anchor) or not valid_pose(world):
            raise ValueError("Invalid E005 LiDAR target reference")
        result = []
        for level, feature_name, coord_name, index_name, width in (
                ("stage3", "stage3_features", "stage3_coords", "stage3_indices", 72),
                ("stage5", "stage5_features", "stage5_coords", "stage5_indices", 216)):
            offset, count = target[f"{level}_indices"]
            index_total = self.record["arrays"][index_name]["shape"][0]
            if type(offset) is not int or type(count) is not int or offset < 0 or count < 0 or offset + count > index_total:
                raise ValueError("E005 target index range outside cache")
            selected = np.asarray(self._map(index_name)[offset:offset + count], dtype=np.int64)
            start, rows = int(lidar[f"{level}_start"]), int(lidar[f"{level}_count"])
            if (start < 0 or rows < 0 or start + rows > self.record["arrays"][feature_name]["shape"][0]
                    or (selected < 0).any() or (selected >= rows).any()
                    or (np.diff(selected) <= 0).any()):
                raise ValueError("E005 target index outside sweep rows")
            feature_rows = start + selected
            features = np.asarray(self._map(feature_name)[feature_rows])
            coords = np.asarray(self._map(coord_name)[feature_rows])
            if features.shape != (count, width) or coords.shape != (count, 3):
                raise ValueError("E005 LiDAR feature/coordinate mismatch")
            result.append(lidar_tokens(features, coords, box, world, anchor))
        return tuple(result)

    def close(self):
        for value in self.maps.values():
            if isinstance(value, np.memmap):
                value._mmap.close()
        self.maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class E005FeatureDataset:
    """Join frozen E001b geometry and labels with the selected spatial cache arms."""

    def __init__(self, baseline, cache, arm):
        if arm not in ("g-rgb", "g-rgb-utonia"):
            raise ValueError("Unknown E005 feature arm")
        self.baseline, self.cache, self.arm = baseline, cache, arm
        self.entries = baseline.entries
        if set(e["locator"] for e in self.entries) - set(cache.tracks):
            raise ValueError("E005 cache does not cover the frozen baseline population")

    def __len__(self):
        return len(self.baseline)

    def __getitem__(self, index):
        sample = self.baseline[index]
        geometry = sample.pop("inputs")
        if geometry.ndim != 2 or geometry.shape[1] != 20:
            raise ValueError("E005 expects the frozen 20-column E001b geometry")
        n, locator = len(sample["targets"]), sample["locator"]
        if len(geometry) != n:
            raise ValueError("E005 geometry/target timeline mismatch")
        rgb = [torch.from_numpy(self.cache.rgb_frame(locator, slot)) for slot in range(n)]
        rgb_valid = torch.tensor([len(x) > 0 for x in rgb], dtype=torch.bool)
        source_flags = {"rgb_valid": rgb_valid.numpy()}
        flags = [geometry[:, [8, 9, 10, 11, 18, 19]].bool().any(-1), rgb_valid]
        sample["geometry"] = geometry
        sample["rgb_frames"] = rgb
        if self.arm == "g-rgb-utonia":
            levels = list(zip(*(self.cache.lidar_frame(locator, slot) for slot in range(n)))) if n else ([], [])
            stage3, stage5 = [list(level) for level in levels]
            if len(stage3) != n or len(stage5) != n:
                raise ValueError("E005 LiDAR/geometry complete-track mismatch")
            stage3 = [torch.from_numpy(frame) for frame in stage3]
            stage5 = [torch.from_numpy(frame) for frame in stage5]
            lidar_valid = torch.tensor([len(a) + len(b) > 0 for a, b in zip(stage3, stage5)], dtype=torch.bool)
            flags.append(lidar_valid)
            source_flags["lidar_valid"] = lidar_valid.numpy()
            sample["lidar_levels"] = (stage3, stage5)
        return sample | {"flags": torch.stack(flags, -1).float(), "prediction_metadata": {
            "group": self.entries[index]["group"], "source_flags": source_flags}}


def collate_e005(samples):
    if not samples:
        raise ValueError("Empty E005 batch")
    use_lidar = "lidar_levels" in samples[0]
    if any(("lidar_levels" in s) != use_lidar for s in samples):
        raise ValueError("Mixed E005 feature arms in a batch")
    lengths = torch.tensor([len(s["targets"]) for s in samples], dtype=torch.int64)
    t = int(lengths.max())
    empty_rgb = torch.zeros((0, 772), dtype=torch.float16)
    result = {"geometry": pad_sequence([s["geometry"] for s in samples], batch_first=True, padding_value=0.),
              "flags": pad_sequence([s["flags"] for s in samples], batch_first=True, padding_value=0.),
              "targets": pad_sequence([s["targets"] for s in samples], batch_first=True, padding_value=-100),
              "gt_valid": pad_sequence([s["gt_valid"] for s in samples], batch_first=True, padding_value=False),
              "lengths": lengths, "padding_mask": torch.arange(t)[None] >= lengths[:, None],
              "locators": [s["locator"] for s in samples], "archives": [s["archive"] for s in samples],
              "prediction_metadata": [s["prediction_metadata"] for s in samples],
              "rgb_frames": [f for s in samples for f in s["rgb_frames"] + [empty_rgb] * (t - len(s["targets"]))]}
    if use_lidar:
        empty3 = torch.zeros((0, 77), dtype=torch.float16)
        empty5 = torch.zeros((0, 221), dtype=torch.float16)
        result["lidar_levels"] = tuple([
            f for s in samples for f in s["lidar_levels"][level] + [empty3 if level == 0 else empty5] * (t - len(s["targets"]))]
            for level in range(2))
    return result


def _representative_groups(catalog):
    by_split = defaultdict(list)
    splits = {}
    for group in sorted({track["group"] for track in catalog.tracks.values()}):
        values = {track["split"] for track in catalog.tracks.values() if track["group"] == group}
        if len(values) != 1:
            raise ValueError("Frozen LOKI scenario crosses E001 splits")
        splits[group] = values.pop()
    frames = defaultdict(int)
    for group, _ in catalog.frames:
        frames[group] += 1
    for group, split in splits.items():
        by_split[split].append((frames[group], group))
    selected = []
    for rows in by_split.values():
        rows.sort()
        if rows:
            for index in sorted(set((len(rows) // 3, (2 * len(rows)) // 3))):
                selected.append(rows[index][1])
    return sorted(set(selected))


def _population(catalog, groups):
    tracks = [t for t in catalog.tracks.values() if t["group"] in groups]
    rgb_candidates = sum(k is not None for t in tracks for k in t["source_keys"])
    rgb_frames = defaultdict(list)
    for track in tracks:
        for slot, key in enumerate(track["source_keys"]):
            if key is not None:
                rgb_frames[track["group"], str(key)].append((track["locator"], slot, track["track_id"]))
    point_frames = {(group, key) for group, key in catalog.frames if group in groups}
    point_targets = sum(len(catalog.frames[group, key]) for group, key in catalog.frames if group in groups)
    return tracks, rgb_frames, point_frames, {"tracks": len(tracks), "slots": sum(t["slots"] for t in tracks),
        "rgb_candidates": rgb_candidates, "point_sweeps": len(point_frames), "point_targets": point_targets,
        "split_tracks": {split: sum(t["split"] == split for t in tracks) for split in ("training", "validation")}}


def _native_anchors(catalog, group, point_keys):
    participants = sorted({locator for key in point_keys for locator in catalog.frames[group, key]})
    first = {catalog.tracks[locator]["source_keys"][0] for locator in participants}
    if None in first:
        raise ValueError("E005 usable point track has no first native frame")
    reader = catalog.native_reader(group)
    records = reader.read_frames(group, first) if first else {}
    anchors = {locator: records[catalog.tracks[locator]["source_keys"][0]]["ego_pose"] for locator in participants}
    if any(not valid_pose(value) for value in anchors.values()):
        raise ValueError("E005 invalid LOKI anchor pose")
    return reader, anchors


def run(args):
    output = args.output.resolve()
    base = args.outputs_root.resolve() / "experiments" / "E005"
    if not output.is_relative_to(base) or output == base:
        raise ValueError("E005 output must be a fresh child of outputs/experiments/E005")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite E005 output: {output}")
    if args.min_free_gb < 20 or not np.isfinite(args.min_free_gb):
        raise ValueError("E005 cache requires a free-space floor of at least 20 decimal GB")
    floor = int(args.min_free_gb * 10**9)
    if args.scope not in ("bounded", "full"):
        raise ValueError("Invalid E005 scope")

    config, transform = load_configuration(args.probe)
    config["loki_root"] = str(args.loki_root or config["loki_root"])
    if not args.utonia_code or not args.tiny_checkpoint:
        raise ValueError("Pinned Utonia code and Tiny checkpoint paths are required")
    if checksum(args.tiny_checkpoint) != TINY_SHA256:
        raise ValueError("Tiny-Utonia checkpoint SHA-256 mismatch")
    if args.scope == "full":
        if args.estimate is None or not args.estimate.is_file():
            raise ValueError("Full E005 cache requires a completed bounded-cache estimate")
        estimate_record = json.loads(args.estimate.read_text())
        projected = estimate_record.get("projection", {}).get("projected_total_bytes")
        if estimate_record.get("status") != "bounded-native-check" or type(projected) is not int:
            raise ValueError("Invalid bounded-cache sizing evidence")
        free = shutil.disk_usage(args.outputs_root).free
        if free < projected + floor:
            raise OSError(f"Full E005 cache needs {projected} estimated bytes plus {floor} free; {free} available")
    else:
        estimate_record, projected = None, None

    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    progress = {"format": FORMAT, "status": "incomplete", "scope": args.scope,
        "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.orig_argv,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diff": subprocess.check_output(["git", "diff", "HEAD"], text=True), "host": platform.node(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "completed_scenes": 0, "failures": []}
    atomic_json(output / "cache.json", progress)
    catalog = None
    encoders = {}
    protected = {}
    writer = None
    try:
        catalog = FeatureCatalog(args.outputs_root, "loki", config, transform)
        all_groups = sorted({t["group"] for t in catalog.tracks.values()})
        if args.scope == "full":
            groups = all_groups
        elif args.groups:
            requested = json.loads(args.groups.read_text())
            groups = sorted(set(requested))
            if not groups or not set(groups).issubset(all_groups):
                raise ValueError("Bounded E005 groups must be nonempty frozen train/validation LOKI scenarios")
        else:
            groups = _representative_groups(catalog)
        tracks, rgb_targets, point_key_set, sample_population = _population(catalog, set(groups))
        _, _, _, full_population = _population(catalog, set(all_groups))
        if full_population["split_tracks"] != {"training": 8574, "validation": 1861}:
            raise ValueError("Frozen LOKI E005 train/validation population changed")
        scope = "complete" if set(groups) == set(all_groups) and args.scope == "full" else "bounded-native-check"
        protected = dict(catalog.protected)
        protected.update({str(Path(config[k]).resolve()): checksum(config[k]) for k in ("dino_weights",)})
        protected[str(Path(args.tiny_checkpoint).resolve())] = checksum(args.tiny_checkpoint)
        protected[str(Path(args.probe / "probe.json").resolve())] = checksum(args.probe / "probe.json")
        protected[str(Path(transform).resolve())] = checksum(transform)
        from pedestrian_behavior.inspection import feature_probe, native_features, pretrained
        implementations = {m.__name__: checksum(m.__file__) for m in (feature_probe, native_features, pretrained)}
        if args.scope == "full" and estimate_record.get("population") != catalog.identity:
            raise ValueError("Bounded estimate belongs to a different frozen LOKI population")
        progress.update(dataset="loki", population=catalog.identity, selected_groups=groups,
            expected={"tracks": sample_population["tracks"], "rgb_candidates": sample_population["rgb_candidates"],
                      "point_sweeps": len(point_key_set), "point_targets": sample_population["point_targets"]},
            full_population=full_population,
            provenance={"DINO": {"revision": DINO_REVISION, "checkpoint_sha256": checksum(config["dino_weights"]),
                                  "crop": "3x max box side; centered; bicubic; mean padding; 256 square",
                                  "output": "16x16x768 patch tokens; fp16 stored; batches of 8"},
                "Tiny-Utonia": {"code_revision": UTONIA_REVISION, "checkpoint_revision": TINY_REVISION,
                    "checkpoint_sha256": TINY_SHA256, "scale": .2, "grid_size_scaled": .01,
                    "stages": {"stage3": {"level": 2, "channels": 72}, "stage5": {"level": 4, "channels": 216}},
                    "coordinate_storage": "native physical coordinates, fp32",
                    "selection": "union of target-centered 20 m world-horizontal cylinders; representative centroids"},
                "implementation_sha256": implementations},
            minimum_free_bytes=floor, bounded_estimate=str(args.estimate) if args.estimate else None)
        atomic_json(output / "cache.json", progress)
        if shutil.disk_usage(output).free < floor:
            raise OSError("Insufficient disk space before E005 extraction")
        with isolated_extraction():
            encoders["dino"] = ImageEncoder(config["dino_code"], config["dino_weights"], args.device)
            encoders["tiny"] = TinyUtoniaEncoder(args.utonia_code, args.tiny_checkpoint, args.device)
        writer = E005Writer(output, floor)
        dino_batches = 0
        online_probe = {"rgb": None, "lidar": None}

        for scene_index, group in enumerate(groups, 1):
            catalog.source_snapshots.pop(group, None)
            before = catalog.source_identity(group)
            writer.begin_scene(group)
            scene = catalog.manifest["scenes"][group]
            frame_reader = catalog.native_reader(group)
            by_key = defaultdict(list)
            for (g, key), rows in rgb_targets.items():
                if g == group:
                    by_key[key] = sorted(rows)
            rgb_keys = sorted(by_key, key=int)
            crops, crop_refs = [], []

            def flush_rgb():
                nonlocal dino_batches
                if not crops:
                    return
                with isolated_extraction():
                    _, patches = encoders["dino"].descriptors(torch.stack(crops), patches=True)
                dino_batches += 1
                if online_probe["rgb"] is None:
                    online_probe["rgb"] = [{"group": group, "key": key, "locator": locator}
                                            for key, locator, _ in crop_refs]
                references = defaultdict(list)
                values = patches.reshape(-1, 256, 768)
                for (key, locator, box_crop), patch in zip(crop_refs, values):
                    references[key].append((locator, box_crop, patch))
                for key, records in references.items():
                    writer.add_rgb_batch(key, [(locator, box) for locator, box, _ in records],
                                         np.stack([patch for _, _, patch in records]))
                crops.clear(); crop_refs.clear()

            for key, image, boxes in native_camera(frame_reader, group, rgb_keys,
                    [track["track_id"] for track in tracks if track["group"] == group], scene):
                writer.summaries["rgb_candidates"] += len(by_key[str(key)])
                if image is None:
                    continue
                for locator, _, track_id in by_key[str(key)]:
                    box = boxes.get(track_id)
                    pixels, detail = square_crop(image, box, expansion=3.)
                    if pixels is None:
                        continue
                    crop_box = rgb_crop_box(box, image.size, detail["bounds_pixels"])
                    crops.append(normalized_crop(pixels))
                    crop_refs.append((str(key), locator, crop_box))
                    if len(crops) == DINO_BATCH:
                        flush_rgb()
            flush_rgb()

            point_keys = sorted((key for g, key in catalog.frames if g == group), key=int)
            point_reader, anchors = _native_anchors(catalog, group, point_keys)
            participants = sorted({locator for key in point_keys for locator in catalog.frames[group, key]})
            for frame in native_scene(point_reader, group, point_keys,
                    [catalog.tracks[locator]["track_id"] for locator in participants], scene, include_rgb=False):
                key = str(frame["key"])
                locators = sorted(catalog.frames[group, key])
                if online_probe["lidar"] is None and locators:
                    online_probe["lidar"] = {"group": group, "key": key, "locator": locators[0]}
                boxes = {}
                for locator in locators:
                    box = frame["boxes"].get(catalog.tracks[locator]["track_id"])
                    if box is None:
                        raise ValueError("Frozen usable LOKI observation lost its native cuboid")
                    boxes[locator] = box
                if frame["world_from_native"] is None or not valid_pose(frame["world_from_native"]):
                    raise ValueError("Frozen usable LOKI frame lost its native pose")
                points = frame["points"] if frame["points"] is not None else np.zeros((0, 4), np.float32)
                with isolated_extraction():
                    stage3, stage5 = encoders["tiny"].stages(points)
                masks = {"stage3": [], "stage5": []}
                for locator in locators:
                    for level, values in (("stage3", stage3), ("stage5", stage5)):
                        relative = relative_xyz(values[1], boxes[locator][0], frame["world_from_native"], anchors[locator])
                        masks[level].append(np.linalg.norm(relative[:, :2], axis=1) <= 20.)
                target_data = {locator: {"box": [*boxes[locator][0], *boxes[locator][1], float(boxes[locator][2])],
                    "anchor": anchors[locator]} for locator in locators}
                union_values = {}
                for level, values in (("stage3", stage3), ("stage5", stage5)):
                    union = np.logical_or.reduce(masks[level]) if masks[level] else np.zeros(len(values[0]), bool)
                    rows = np.flatnonzero(union)
                    remap = np.full(len(union), -1, np.int64)
                    remap[rows] = np.arange(len(rows), dtype=np.int64)
                    union_values[level] = (values[0][rows], values[1][rows])
                    for locator, selected in zip(locators, masks[level]):
                        target_data[locator]["indices" + level[-1]] = remap[np.flatnonzero(selected)]
                writer.add_points(key, union_values["stage3"], union_values["stage5"], target_data, frame["world_from_native"])
            catalog.source_snapshots.pop(group, None)
            after = catalog.source_identity(group)
            if before != after:
                raise ValueError("Native LOKI scene changed during E005 preparation")
            scene_record = writer.finish_scene(before)
            progress["completed_scenes"] = scene_index
            progress["last_scene"] = group
            progress["summary"] = dict(writer.summaries)
            progress["elapsed_seconds"] = time.perf_counter() - started
            progress["free_bytes"] = shutil.disk_usage(output).free
            progress["scene_manifest_sha256"] = scene_record["sha256"]
            atomic_json(output / "cache.json", progress)
            if scene_index == 1 or scene_index % 10 == 0 or scene_index == len(groups):
                print(f"LOKI E005: {scene_index}/{len(groups)} scenes, "
                      f"{writer.rows['rgb_features']} valid RGB crops, "
                      f"{writer.rows['stage3_features']} stage3 / {writer.rows['stage5_features']} stage5 tokens, "
                      f"{shutil.disk_usage(output).free/1e9:.1f} GB free", flush=True)

        if any(checksum(path) != digest for path, digest in protected.items()):
            raise ValueError("Protected E005 source or encoder asset changed")
        arrays, payload_bytes = writer.finish()
        tracks_record = {"format": FORMAT, "population": catalog.identity, "tracks": [
            {k: t[k] for k in ("locator", "group", "archive", "split", "slots", "source_keys", "times_us", "sha256")}
            for t in sorted(tracks, key=lambda item: item["locator"])]}
        atomic_json(output / "tracks.json", tracks_record)
        projection = None
        if scope == "bounded-native-check":
            sample_rgb_bytes = arrays["rgb_features"]["bytes"] + arrays["rgb_boxes"]["bytes"]
            sample_lidar_bytes = sum(arrays[name]["bytes"] for name in
                ("stage3_features", "stage3_coords", "stage5_features", "stage5_coords", "stage3_indices", "stage5_indices"))
            full_rgb_candidates, full_point_sweeps = full_population["rgb_candidates"], full_population["point_sweeps"]
            projection_rgb = (sample_rgb_bytes * full_rgb_candidates // max(1, sample_population["rgb_candidates"]))
            projection_lidar = (sample_lidar_bytes * full_point_sweeps // max(1, len(point_key_set)))
            projection = {"sample_groups": groups, "sample_rgb_candidates": sample_population["rgb_candidates"],
                "full_rgb_candidates": full_rgb_candidates, "sample_point_sweeps": len(point_key_set),
                "full_point_sweeps": full_point_sweeps, "sample_rgb_bytes": sample_rgb_bytes,
                "sample_lidar_bytes_including_indices": sample_lidar_bytes,
                "projected_rgb_bytes": projection_rgb, "projected_lidar_bytes_including_indices": projection_lidar,
                "projected_total_bytes": projection_rgb + projection_lidar,
                "current_free_bytes": shutil.disk_usage(output).free,
                "minimum_free_bytes": floor,
                "headroom_after_projected_cache": shutil.disk_usage(output).free - projection_rgb - projection_lidar,
                "qualification": "Linear group-average projection from bounded native encodings; scene variation remains unmeasured."}
        status = "complete" if scope == "complete" else "bounded-native-check"
        manifest = {"format": FORMAT, "status": status, "dataset": "loki", "scope": scope,
            "created_utc": datetime.now(timezone.utc).isoformat(), "revision": progress["revision"],
            "population": catalog.identity, "selected_groups": groups, "full_population": full_population,
            "tracks_sha256": checksum(output / "tracks.json"), "arrays": arrays, "scenes": writer.scenes,
            "summary": dict(writer.summaries), "payload_bytes": payload_bytes,
            "minimum_free_bytes": floor, "projection": projection,
            "provenance": progress["provenance"], "protected_files": len(protected),
            "online_probe": online_probe,
            "extractors": {"dino": {"calls": dino_batches, "batch_size": DINO_BATCH},
                "tiny": {"calls": encoders["tiny"].calls, "seconds": encoders["tiny"].seconds}},
            "finite_checked_at_write": True
        }
        atomic_json(output / "manifest.json", manifest)
        progress.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - started, payload_bytes=payload_bytes,
            summary=dict(writer.summaries), projection=projection, manifest_sha256=checksum(output / "manifest.json"))
    except BaseException:
        progress["status"] = "failed"
        progress["failures"].append(traceback.format_exc())
        raise
    finally:
        if writer is not None:
            writer.close()
        for encoder in encoders.values():
            del encoder
        encoders.clear()
        gc.collect()
        if str(args.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        changed = [path for path, digest in protected.items() if not Path(path).is_file() or checksum(path) != digest]
        progress.update(protected_files=len(protected), changed_files=changed,
            finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.perf_counter() - started)
        if changed:
            progress["status"] = "failed"
        atomic_json(output / "cache.json", progress)
        if changed:
            raise ValueError("Protected E005 source or encoder asset changed")
    print(f"E005 {progress['status']}: {output}", flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--probe", type=Path, default=Path("outputs/experiments/E002a/native-probe-07"))
    parser.add_argument("--loki-root", type=Path)
    parser.add_argument("--utonia-code", type=Path, required=True)
    parser.add_argument("--tiny-checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--scope", choices=("bounded", "full"), default="bounded")
    parser.add_argument("--groups", type=Path, help="JSON list of frozen LOKI scenario IDs for bounded checks")
    parser.add_argument("--estimate", type=Path, help="Bounded manifest.json required before full preparation")
    parser.add_argument("--min-free-gb", type=float, default=20.)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
