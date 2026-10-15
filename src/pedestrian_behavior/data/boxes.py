"""Center/size features and their immediate-neighbor velocities."""

import json

import numpy as np

from .features import normalize
from .preparation import source_times, valid_pose, velocities

COLUMNS = ("x", "y", "z", "length", "width", "height", "vx", "vy", "vz",
           "u", "v", "width2d", "height2d", "velocity_u", "velocity_v",
           "velocity_width2d", "velocity_height2d", "velocity_length", "velocity_width", "velocity_height")
FLAGS = ("box3d_valid", "box3d_velocity_valid", "box2d_valid", "box2d_velocity_valid")
CONFIGURATIONS = {"box3d": 9, "box3d-2d": 17, "box3d-2d-size-velocity": 20}
INPUT_COLUMNS = {"boxes": 15, "boxes-2d-size-vel": 17, "boxes-2d-3d-size-vel": 20}


def column_validity(arrays):
    return np.column_stack([arrays[flag] for flag, count in zip(FLAGS, (6, 3, 4, 4)) for _ in range(count)]
                           + [arrays["box3d_velocity_valid"]]*3)


def box_features(world_centers, sizes, corners, times, anchor):
    """No interpolation, heading input, GT access or silent validity collapse."""
    n = len(times)
    world_centers, sizes, corners = (np.asarray(a, dtype=np.float64) for a in (world_centers, sizes, corners))
    if world_centers.shape != (n, 3) or sizes.shape != (n, 3) or corners.shape != (n, 4):
        raise ValueError("Invalid box alignment")
    values = np.full((n, 20), np.nan, dtype=np.float64)
    center_valid = np.isfinite(world_centers).all(axis=1) & bool(valid_pose(anchor))
    size_valid = np.isfinite(sizes).all(axis=1) & (sizes > 0).all(axis=1)
    # A common flag is valid only if no usable center or size would be discarded.
    if not np.array_equal(center_valid, size_valid):
        raise ValueError("3D center/size validity differs: separate flags require protocol review")
    if any(t is None and (c or s) for t, c, s in zip(times, center_valid, size_valid)):
        raise ValueError("3D box without selected source time")
    if valid_pose(anchor):
        yaw = np.arctan2(anchor[1, 0], anchor[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.]])
        values[center_valid, :3] = (world_centers[center_valid]-anchor[:3, 3]) @ rotation.T
    values[size_valid, 3:6] = sizes[size_valid]
    speed3d, velocity3d_valid = velocities(values[:, :6], center_valid, times)
    values[:, 6:9], values[:, 17:20] = speed3d[:, :3], speed3d[:, 3:6]
    box2d_valid = np.isfinite(corners).all(axis=1) & (corners[:, 2:] > corners[:, :2]).all(axis=1)
    if any(t is None and valid for t, valid in zip(times, box2d_valid)):
        raise ValueError("2D box without selected source time")
    values[box2d_valid, 9:13] = np.column_stack(((corners[:, :2]+corners[:, 2:])/2,
                                                               corners[:, 2:]-corners[:, :2]))[box2d_valid]
    values[:, 13:17], velocity2d_valid = velocities(values[:, 9:13], box2d_valid, times)
    result = {"values": values, "box3d_valid": center_valid, "box3d_velocity_valid": velocity3d_valid,
              "box2d_valid": box2d_valid, "box2d_velocity_valid": velocity2d_valid,
              "center3d_valid": center_valid.copy(), "size3d_valid": size_valid.copy()}
    valid = column_validity(result)
    if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
        raise ValueError("Nonfinite box computation or invalid missingness")
    return result


def box_inputs(arrays, configuration, statistics):
    # Keep preparation-era selections available without changing frozen cache policy.
    count = (CONFIGURATIONS | INPUT_COLUMNS)[configuration]
    numeric = normalize(arrays["values"], column_validity(arrays), statistics)[:, :count]
    flags = np.column_stack([arrays[name] for name in FLAGS])
    return np.concatenate((numeric, flags), axis=1).astype(np.float32)


def load_boxes(path, metadata, track_sha256):
    with np.load(path, allow_pickle=False) as saved:
        reference = json.loads(str(saved["metadata"]))
        arrays = {key: saved[key] for key in saved.files if key != "metadata"}
    for key, value in {"track_locator": metadata["track_locator"], "source_frames": metadata["source_frames"],
                       "source_time_us": source_times(metadata), "track_sha256": track_sha256}.items():
        if reference.get(key) != value:
            raise ValueError("E001c/frozen track reference mismatch")
    n = len(source_times(metadata))
    values = arrays["values"]
    if values.shape != (n, 20) or values.dtype != np.float64:
        raise ValueError("Invalid E001c numeric shape/dtype")
    for key in (*FLAGS, "center3d_valid", "size3d_valid", "native_3d", "loki_2d", "road_2d", "waymo_2d", "gt_valid"):
        if arrays[key].shape != (n,) or arrays[key].dtype != bool:
            raise ValueError(f"Invalid {key} shape/dtype")
    sizes = arrays["image_size"]
    if sizes.shape != (n, 2) or sizes.dtype != np.int64 or (sizes < 0).any():
        raise ValueError("Invalid image dimensions")
    if ((arrays["box3d_valid"] & ~arrays["native_3d"]).any()
            or (arrays["box2d_valid"] & (~(arrays["loki_2d"] | arrays["waymo_2d"]) | (sizes <= 0).any(axis=1))).any()):
        raise ValueError("Usable boxes lack native observation/image dimensions")
    valid = column_validity(arrays)
    if not np.isfinite(values[valid]).all() or not np.isnan(values[~valid]).all():
        raise ValueError("Invalid E001c numeric missingness")
    if ((arrays["box3d_velocity_valid"] & ~arrays["box3d_valid"]).any()
            or (arrays["box2d_velocity_valid"] & ~arrays["box2d_valid"]).any()
            or not np.array_equal(arrays["center3d_valid"], arrays["size3d_valid"])
            or not np.array_equal(arrays["box3d_valid"], arrays["center3d_valid"])):
        raise ValueError("Invalid E001c mask dependencies")
    if (values[arrays["box3d_valid"], 3:6] <= 0).any() or (values[arrays["box2d_valid"], 11:13] <= 0).any():
        raise ValueError("Invalid E001c dimensions")
    if arrays["targets"].shape != (n,) or arrays["targets"].dtype != np.int64 or not np.isin(arrays["targets"][arrays["gt_valid"]], range(4)).all():
        raise ValueError("Invalid frozen targets")
    missing = np.array([t is None for t in source_times(metadata)])
    if valid[missing].any():
        raise ValueError("Input in missing source slot")
    return reference, arrays
