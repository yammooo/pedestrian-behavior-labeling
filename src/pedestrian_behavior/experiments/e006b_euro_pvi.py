"""One-sequence-at-a-time adapter from Euro-PVI frames to E005 samples."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pedestrian_behavior.data.bbox import bbox_inputs, box_geometry
from pedestrian_behavior.data.features import features, normalize
from pedestrian_behavior.data.preparation import velocities
from pedestrian_behavior.experiments.e005_cache import DINO_BATCH, rgb_crop_box, rgb_tokens
from pedestrian_behavior.inspection.feature_probe import normalized_crop, relative_xyz, square_crop


IMAGE_SIZE = (1280, 806)
BOX_SOURCE = {"mask": 1, "seg": 2, "estimate": 3}
CLASS_ID = {"moving": 0, "stopped": 1, "waiting to cross": 2, "crossing": 3}
EMPTY_RGB = np.zeros((0, 772), dtype=np.float16)
EMPTY_STAGE3 = np.zeros((0, 77), dtype=np.float16)
EMPTY_STAGE5 = np.zeros((0, 221), dtype=np.float16)


def _model_xyz(camera_xyz):
    """Camera right/down/forward to the source models' forward/left/up axes."""
    xyz = np.asarray(camera_xyz, dtype=np.float32)
    return np.column_stack((xyz[:, 2], -xyz[:, 0], -xyz[:, 1]))


def _pose(yaw, x=0., y=0., z=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))
    result[:3, 3] = (x, y, z)
    return result


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _box_row(row):
    if row is None or row.get("bbox_src") not in BOX_SOURCE:
        return None
    try:
        pixels = np.array([float(row[k]) for k in ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")])
    except (TypeError, ValueError):
        return None
    if not np.isfinite(pixels).all() or pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        return None
    if min(pixels[2], IMAGE_SIZE[0]) <= max(pixels[0], 0) or min(pixels[3], IMAGE_SIZE[1]) <= max(pixels[1], 0):
        return None
    scale = np.tile(np.asarray(IMAGE_SIZE, dtype=np.float64), 2)
    return pixels / scale, BOX_SOURCE[row["bbox_src"]]


def _lidar_tokens(stage, center, world_from_native, anchor, width):
    features, coords = stage
    features, coords = np.asarray(features), np.asarray(coords)
    if features.ndim != 2 or features.shape[1] != width or coords.shape != (len(features), 3):
        raise ValueError("Tiny-Utonia stage has an unexpected shape")
    if not len(features):
        return EMPTY_STAGE3 if width == 72 else EMPTY_STAGE5
    relative = relative_xyz(coords, center, world_from_native, anchor)
    keep = np.square(relative[:, :2]).sum(1) <= 400.
    relative = relative[keep]
    selected = features[keep]
    cues = np.column_stack((relative / 20., np.square(relative[:, :2]).sum(1) / 400.,
                            np.zeros(len(relative), dtype=np.float32)))
    return np.concatenate((selected, cues), axis=1).astype(np.float16)


class EuroPVIAdapter:
    """Read/index native labels and encode each sequence's sweeps and image crops once."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        rows_path = self.root / "labels" / "frames.csv"
        grouped = defaultdict(lambda: defaultdict(list))
        with rows_path.open(newline="") as source:
            for row in csv.DictReader(source):
                sequence = f'{row["split"]}/{int(row["seq"]):04d}'
                grouped[sequence][str(row["inst_id"])].append(row)
        self._tracks = {sequence: tracks for sequence, tracks in grouped.items()}

    def sequences(self):
        return iter(sorted(self._tracks))

    def load_sequence(self, sequence_id):
        """Return one native sequence's JSON frame records and labeled tracks."""
        split, seq = sequence_id.split("/")
        sequence_id = f"{split}/{int(seq):04d}"
        if sequence_id not in self._tracks:
            raise KeyError(sequence_id)
        annotation_path = self.root / "sequences" / sequence_id / "annotations.json"
        saved = json.loads(annotation_path.read_text())
        frames = {}
        for frame in saved["frames"]:
            key = int(frame["frame"])
            frames[key] = frame | {"agents_by_id": {
                str(int(agent["inst_id"])): agent for agent in frame.get("agents", [])}}
        if len(frames) != len(saved["frames"]):
            raise ValueError(f"Repeated native frames in {sequence_id}")
        tracks = []
        for track_id, rows in sorted(self._tracks[sequence_id].items(), key=lambda item: int(item[0])):
            rows.sort(key=lambda row: int(row["frame"]))
            by_frame = {int(row["frame"]): row for row in rows}
            native_ids = list(by_frame)
            if len(by_frame) != len(rows) or native_ids != list(range(native_ids[0], native_ids[-1] + 1)):
                raise ValueError(f"Expected complete native labeled frames for {sequence_id}/{track_id}")
            first = native_ids[0]
            first_frame = frames.get(first)
            pose = first_frame.get("camera_pose_est") if first_frame else None
            yaw0 = _finite_float(pose.get("yaw")) if pose else np.nan
            frame_indices = list(range(first, native_ids[-1] + 1, 2))
            tracks.append({"track_id": track_id, "locator": json.dumps(
                {"dataset": "euro-pvi", "split": split, "seq": int(seq), "inst_id": int(track_id)},
                sort_keys=True, separators=(",", ":")), "group": sequence_id,
                "frame_indices": frame_indices, "rows": [by_frame.get(i) for i in frame_indices],
                "anchor_yaw": yaw0})
        return {"sequence_id": sequence_id, "annotation_path": annotation_path,
                "frames": frames, "tracks": tracks}

    def encode_sequence(self, sequence, dino, tiny, device="cuda"):
        """Encode each native frame once, then assemble 5 Hz per-track records."""
        tracks = sequence["tracks"]
        by_frame = defaultdict(list)
        records = []
        for track_index, track in enumerate(tracks):
            n = len(track["frame_indices"])
            records.append({"locator": track["locator"], "group": track["group"],
                "frame_indices": track["frame_indices"], "targets": np.full(n, -100, dtype=np.int64),
                "gt_valid": np.zeros(n, dtype=bool), "rgb_frames": [EMPTY_RGB] * n,
                "stage3_frames": [EMPTY_STAGE3] * n, "stage5_frames": [EMPTY_STAGE5] * n,
                "rgb_valid": np.zeros(n, dtype=bool), "lidar_valid": np.zeros(n, dtype=bool),
                "rows": track["rows"], "anchor_yaw": track["anchor_yaw"],
                "archive": str(sequence["annotation_path"])})
            for slot, (frame, row) in enumerate(zip(track["frame_indices"], track["rows"])):
                by_frame[frame].append((track_index, slot, row))

        for frame_index, assignments in sorted(by_frame.items()):
            native = sequence["frames"].get(frame_index, {})
            image_paths = {row["rgb"] for _, _, row in assignments if row and row.get("rgb")}
            lidar_paths = {row["lidar"] for _, _, row in assignments if row and row.get("lidar")}
            if len(image_paths) > 1 or len(lidar_paths) > 1:
                raise ValueError(f"Native modalities disagree at {sequence['sequence_id']}/{frame_index}")

            image_path = next(iter(image_paths), None) or native.get("rgb")
            crops, crop_refs = [], []
            if image_path:
                with Image.open(self.root / image_path) as image:
                    image = image.convert("RGB")
                    if image.size != IMAGE_SIZE:
                        raise ValueError(f"Unexpected Euro-PVI image size {image.size}")
                    for track_index, slot, row in assignments:
                        box = _box_row(row)
                        if box is None:
                            continue
                        normalized_box, _ = box
                        crop, detail = square_crop(image, normalized_box, expansion=3.)
                        if crop is not None:
                            crops.append(normalized_crop(crop))
                            crop_refs.append((track_index, slot,
                                rgb_crop_box(normalized_box, image.size, detail["bounds_pixels"])))
                for start in range(0, len(crops), DINO_BATCH):
                    batch = torch.stack(crops[start:start + DINO_BATCH])
                    _, patches = dino.descriptors(batch, patches=True)
                    for ref, patch in zip(crop_refs[start:start + DINO_BATCH], patches.reshape(-1, 256, 768)):
                        track_index, slot, box_crop = ref
                        record = records[track_index]
                        record["rgb_frames"][slot] = rgb_tokens(patch, box_crop)
                        record["rgb_valid"][slot] = True

            lidar_path = next(iter(lidar_paths), None) or native.get("lidar")
            if not lidar_path:
                continue
            points = np.load(self.root / lidar_path, mmap_mode="r")
            if points.ndim != 2 or points.shape[1] != 5 or not np.isfinite(points).all():
                raise ValueError(f"Expected finite Euro-PVI (N,5) camera sweep: {lidar_path}")
            native_points = np.zeros((len(points), 4), dtype=np.float32)
            native_points[:, :3] = _model_xyz(points[:, :3])
            stage3, stage5 = tiny.stages(native_points)
            pose = native.get("camera_pose_est")
            ego = native.get("ego_pos")
            yaw = _finite_float(pose.get("yaw")) if pose else np.nan
            if not np.isfinite(yaw):
                continue
            ego_xyz = np.asarray(ego, dtype=np.float64) if ego is not None else np.zeros(3)
            if ego_xyz.shape != (3,) or not np.isfinite(ego_xyz).all():
                ego_xyz = np.zeros(3)
            world_from_native = _pose(yaw, *ego_xyz)
            for track_index, slot, row in assignments:
                person = native.get("agents_by_id", {}).get(tracks[track_index]["track_id"])
                if person is None or not np.isfinite(records[track_index]["anchor_yaw"]):
                    continue
                center_cam = np.asarray(person.get("cam_pos"), dtype=np.float32)
                if center_cam.shape != (3,) or not np.isfinite(center_cam).all():
                    continue
                center = _model_xyz(center_cam[None])[0]
                anchor = _pose(float(records[track_index]["anchor_yaw"]))
                record = records[track_index]
                record["stage3_frames"][slot] = _lidar_tokens(stage3, center, world_from_native, anchor, 72)
                record["stage5_frames"][slot] = _lidar_tokens(stage5, center, world_from_native, anchor, 216)
                record["lidar_valid"][slot] = bool(len(record["stage3_frames"][slot]) + len(record["stage5_frames"][slot]))

        for track, record in zip(tracks, records):
            self._geometry(track, record, sequence["frames"])
        return records

    @staticmethod
    def _geometry(track, record, frames):
        rows, frame_indices, yaw0 = track["rows"], track["frame_indices"], track["anchor_yaw"]
        n = len(frame_indices)
        times = [frame * 100_000 for frame in frame_indices]
        ped_world = np.full((n, 2), np.nan, dtype=np.float64)
        ego_world = np.full((n, 2), np.nan, dtype=np.float64)
        corners = np.full((n, 4), np.nan, dtype=np.float64)
        box_sources = np.zeros(n, dtype=np.uint8)
        targets = np.full(n, -100, dtype=np.int64)
        gt_valid = np.zeros(n, dtype=bool)
        bbox_sources = []
        for i, (frame_index, row) in enumerate(zip(frame_indices, rows)):
            frame = frames.get(frame_index, {})
            ego = np.asarray(frame.get("ego_pos", []), dtype=np.float64)
            if ego.shape == (3,) and np.isfinite(ego).all():
                ego_world[i] = ego[:2]
            person = frame.get("agents_by_id", {}).get(track["track_id"])
            world = np.asarray(person.get("pos", []) if person else [], dtype=np.float64)
            if world.shape == (3,) and np.isfinite(world).all():
                ped_world[i] = world[:2]
            if row is not None:
                label = CLASS_ID.get(row.get("label", ""))
                if label is None:
                    raise ValueError(f"Unexpected Euro-PVI label {row.get('label')!r}")
                targets[i], gt_valid[i] = label, True
                box = _box_row(row)
                if box is not None:
                    corners[i], box_sources[i] = box
                bbox_sources.append(row.get("bbox_src") or None)
            else:
                bbox_sources.append(None)

        arrays = {"ped_position": np.full((n, 2), np.nan), "ped_position_valid": np.zeros(n, bool),
                  "ped_velocity": np.full((n, 2), np.nan), "ped_velocity_valid": np.zeros(n, bool),
                  "ego_position": np.full((n, 2), np.nan), "ego_pose_valid": np.zeros(n, bool),
                  "ego_velocity": np.full((n, 2), np.nan), "ego_velocity_valid": np.zeros(n, bool)}
        if np.isfinite(yaw0) and n and rows[0] is not None:
            origin = ego_world[0]
            if np.isfinite(origin).all():
                c, s = np.cos(yaw0), np.sin(yaw0)
                rotation = np.array([[c, s], [-s, c]])
                ped_valid = np.isfinite(ped_world).all(1)
                ego_valid = np.isfinite(ego_world).all(1)
                arrays["ped_position"][ped_valid] = (ped_world[ped_valid] - origin) @ rotation.T
                arrays["ego_position"][ego_valid] = (ego_world[ego_valid] - origin) @ rotation.T
                arrays["ped_position_valid"], arrays["ego_pose_valid"] = ped_valid, ego_valid
        arrays["ped_velocity"], arrays["ped_velocity_valid"] = velocities(
            arrays["ped_position"], arrays["ped_position_valid"], times)
        arrays["ego_velocity"], arrays["ego_velocity_valid"] = velocities(
            arrays["ego_position"], arrays["ego_pose_valid"], times)
        baseline_values, baseline_flags = features(arrays, "K+T+R")
        bbox = box_geometry(corners, times, box_sources)
        record.update(targets=targets, gt_valid=gt_valid,
                      baseline_values=baseline_values, baseline_flags=baseline_flags,
                      bbox=bbox, bbox_sources=box_sources, bbox_source_names=bbox_sources,
                      source_flags={"rgb_valid": record["rgb_valid"], "lidar_valid": record["lidar_valid"]},
                      prediction_metadata={"group": record["group"], "frame_indices": frame_indices,
                                           "bbox_sources": bbox_sources,
                                           "source_flags": {"rgb_valid": record["rgb_valid"],
                                                            "lidar_valid": record["lidar_valid"]}})


def sample_for(record, normalization):
    """Materialize one record as a CPU sample accepted by ``collate_e005``."""
    baseline = normalize(record["baseline_values"], record["baseline_flags"], normalization["baseline"])
    bbox = bbox_inputs(record["bbox"], "geometry", normalization["bbox"])
    geometry = torch.from_numpy(np.concatenate((baseline, bbox), axis=1).astype(np.float32))
    if geometry.shape != (len(record["targets"]), 20):
        raise ValueError("Euro-PVI E005 geometry must contain 20 input features")
    geometry_valid = geometry[:, [8, 9, 10, 11, 18, 19]].bool().any(-1)
    rgb_valid = torch.from_numpy(record["rgb_valid"])
    lidar_valid = torch.from_numpy(record["lidar_valid"])
    flags = torch.stack((geometry_valid, rgb_valid, lidar_valid), dim=-1).float()
    return {"geometry": geometry, "flags": flags,
            "rgb_frames": [torch.from_numpy(value) for value in record["rgb_frames"]],
            "lidar_levels": ([torch.from_numpy(value) for value in record["stage3_frames"]],
                             [torch.from_numpy(value) for value in record["stage5_frames"]]),
            "targets": torch.from_numpy(record["targets"]),
            "gt_valid": torch.from_numpy(record["gt_valid"]),
            "locator": record["locator"], "archive": record["archive"],
            "prediction_metadata": record["prediction_metadata"]}
