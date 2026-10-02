"""Read the ROAD-authoritative index and native Waymo v2 inspection data."""

import csv
import gzip
import json
from itertools import groupby
from pathlib import Path

import numpy as np


def annotation_rows(index: Path):
    with gzip.open(index / "pedestrians.csv.gz", "rt", newline="") as source:
        yield from csv.DictReader(source)


def read_tracks(index: Path, selected: list[tuple[str, str]]):
    tracks = {key: {} for key in selected}
    for row in annotation_rows(index):
        key = row["road_clip_id"], row["road_tube_uid"]
        if key not in tracks:
            continue
        if row["merged_agent_label"] != "Ped" or row["camera_name"] != "1":
            raise ValueError(f"{key}: expected a ROAD pedestrian in FRONT")
        if row["has_3d_box"] not in ("True", "False"):
            raise ValueError(f"{key}: invalid 3D mask")
        box3d = json.loads(row["waymo_box3d_center_size_heading_json"]) if row["has_3d_box"] == "True" else None
        if box3d is not None and (len(box3d) != 7 or not np.isfinite(box3d).all() or min(box3d[3:6]) <= 0):
            raise ValueError(f"{key}: invalid native 3D box")
        box2d = json.loads(row["road_box_normalized_json"])
        if (len(box2d) != 4 or not np.isfinite(box2d).all()
                or box2d[0] > box2d[2] or box2d[1] > box2d[3]):
            raise ValueError(f"{key}: invalid ROAD 2D box")
        observation = dict(
            frame=int(row["road_frame_1based"]), box2d=box2d,
            box3d=box3d, actions=json.loads(row["action_labels_json"]),
            locations=json.loads(row["loc_labels_json"]), original_type=row["waymo_3d_label"],
            disagreement=row["semantic_disagreement"] == "True",
        )
        ts = int(row["frame_timestamp_micros"])
        previous = tracks[key].get(ts)
        if previous and {k: v for k, v in previous.items() if k != "annotation_ids"} != observation:
            raise ValueError(f"{key} timestamp {ts}: conflicting repeated observation")
        if previous:
            previous["annotation_ids"].append(row["road_annotation_id"])
        else:
            tracks[key][ts] = dict(observation, annotation_ids=[row["road_annotation_id"]])
    return tracks


def component_path(scene: dict, component: str, waymo_root: Path | None = None):
    path = scene["component_paths"].get(component)
    if path is None:
        return None
    if waymo_root is not None:
        return waymo_root / scene["waymo_split"] / component / (scene["segment_context_name"] + ".parquet")
    return Path(path)


def component_rows(path: Path | None, scene_name: str):
    """Stream one component; reject unordered timestamps instead of misaligning it."""
    import pyarrow.parquet as pq

    if path is None or not path.is_file():
        return
    last = -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1, use_threads=False):
        row = {}
        for i, name in enumerate(batch.schema.names):
            value = batch.column(i)[0]
            # Keep large range/pose arrays in Arrow/NumPy instead of creating millions of Python floats.
            row[name] = value.values.to_numpy(zero_copy_only=False) if name.endswith(".values") and value.is_valid else value.as_py()
        ts = row.get("key.frame_timestamp_micros", last)
        if row["key.segment_context_name"] != scene_name or ts < last:
            raise ValueError(f"{path}: wrong scene or unordered timestamps")
        last = ts
        yield row


def frame_groups(rows):
    for timestamp, group in groupby(rows, key=lambda row: row["key.frame_timestamp_micros"]):
        yield timestamp, list(group)


def range_points(values, shape, calibration, pixel_pose=None, frame_pose=None):
    """Waymo range-image equations, including TOP per-pixel ego compensation.

    See official range_image_utils.py and v2/perception/utils/lidar_utils.py.
    Only positive finite ranges are retained; output is in the native vehicle frame.
    """
    if len(shape) != 3 or shape[2] != 4 or min(shape) <= 0:
        raise ValueError("Unsupported Waymo range image: expected H x W x 4")
    ranges = np.asarray(values, dtype=np.float64).reshape(shape)[..., 0]
    h, w = ranges.shape
    c = "[LiDARCalibrationComponent]."
    extrinsic = np.asarray(calibration[c + "extrinsic.transform"]).reshape(4, 4)
    inclination = calibration[c + "beam_inclination.values"]
    if inclination is None:
        low, high = (calibration[c + "beam_inclination." + k] for k in ("min", "max"))
        inclination = low + (np.arange(h) + .5) / h * (high - low)
    inclination = np.asarray(inclination)[::-1]
    if inclination.shape != (h,):
        raise ValueError("Beam inclination count does not match range image")
    azimuth = ((np.arange(w, 0, -1) - .5) / w * 2 - 1) * np.pi
    azimuth -= np.arctan2(extrinsic[1, 0], extrinsic[0, 0])
    valid = np.isfinite(ranges) & (ranges > 0)
    rows, cols = np.nonzero(valid)
    r, inc, az = ranges[valid], inclination[rows], azimuth[cols]
    points = np.column_stack((r * np.cos(inc) * np.cos(az),
                              r * np.cos(inc) * np.sin(az), r * np.sin(inc)))
    points = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    if pixel_pose is not None:
        if frame_pose is None:
            raise ValueError("TOP pixel poses require the matching vehicle frame pose")
        pose = np.asarray(pixel_pose, dtype=np.float64).reshape(h, w, 6)[valid]
        if not np.isfinite(pose).all():
            raise ValueError("Non-finite LiDAR pixel pose")
        roll, pitch, yaw = pose[:, :3].T
        x, y, z = points.T
        # Apply Rz(yaw) Ry(pitch) Rx(roll) without allocating per-point 4x4 matrices.
        y, z = np.cos(roll) * y - np.sin(roll) * z, np.sin(roll) * y + np.cos(roll) * z
        x, z = np.cos(pitch) * x + np.sin(pitch) * z, -np.sin(pitch) * x + np.cos(pitch) * z
        x, y = np.cos(yaw) * x - np.sin(yaw) * y, np.sin(yaw) * x + np.cos(yaw) * y
        world = np.column_stack((x, y, z)) + pose[:, 3:]
        inverse = np.linalg.inv(frame_pose)
        points = world @ inverse[:3, :3].T + inverse[:3, 3]
    if not np.isfinite(points).all():
        raise ValueError("Non-finite decoded LiDAR points")
    return points


def focus_centers(timestamps, observations, poses):
    """Interpolate view focus in world space; never interpolate annotation boxes."""
    known = []
    for ts in timestamps:
        box = observations.get(ts, {}).get("box3d")
        if box is not None:
            world = poses[ts] @ np.array([*box[:3], 1.])
            known.append((ts, world[:3]))
    if not known:
        return [(0., 0.) for _ in timestamps]
    times = np.asarray([k[0] for k in known]) - timestamps[0]
    xyz = np.asarray([k[1] for k in known])
    centers = []
    for ts in timestamps:
        world = [np.interp(ts - timestamps[0], times, xyz[:, axis]) for axis in range(3)]
        vehicle = np.linalg.inv(poses[ts]) @ np.array([*world, 1.])
        centers.append(tuple(vehicle[:2]))
    return centers
