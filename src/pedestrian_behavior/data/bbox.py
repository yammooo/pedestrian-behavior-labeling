"""Image-relative box geometry; independent masks and immediate-neighbor speeds."""

import json

import numpy as np

from .features import normalize
from .preparation import source_times, velocities

COLUMNS = ("center_u", "bottom_v", "width", "height", "velocity_u", "velocity_v")
SOURCES = {0: "missing", 1: "loki", 2: "road", 3: "waymo-front"}


def box_geometry(corners, times, sources):
    """Corners already use image-relative xyxy; never clip or fill observations."""
    corners = np.asarray(corners, dtype=np.float64)
    sources = np.asarray(sources)
    if corners.shape != (len(times), 4) or sources.shape != (len(times),) or not np.isin(sources, list(SOURCES)).all():
        raise ValueError("Invalid bbox alignment")
    present = sources != 0
    if any(t is None and p for t, p in zip(times, present)):
        raise ValueError("Box without a selected source frame")
    valid = present & np.isfinite(corners).all(axis=1)
    valid &= (corners[:, 2] > corners[:, 0]) & (corners[:, 3] > corners[:, 1])
    values = np.full((len(times), 6), np.nan, dtype=np.float64)
    values[valid, :4] = np.column_stack(((corners[:, 0]+corners[:, 2])/2, corners[:, 3],
                                       corners[:, 2]-corners[:, 0], corners[:, 3]-corners[:, 1]))[valid]
    speed_valid = np.zeros(len(times), dtype=bool)
    # Different annotation sources can have different box conventions at a switch.
    for source in SOURCES.keys() - {0}:
        speed, mask = velocities(values[:, :2], valid & (sources == source), times)
        values[mask, 4:] = speed[mask]
        speed_valid |= mask
    if not np.isfinite(values[valid, :4]).all() or not np.isfinite(values[speed_valid, 4:]).all():
        raise ValueError("Nonfinite bbox computation")
    return {"bbox_features": values, "bbox_valid": valid, "bbox_velocity_valid": speed_valid,
            "bbox_source": sources.astype(np.uint8)}


def normalization_sample(arrays):
    """Reuse the existing two-columns-per-mask moments operation."""
    return arrays["bbox_features"], np.column_stack((arrays["bbox_valid"], arrays["bbox_valid"],
                                                     arrays["bbox_velocity_valid"]))


def bbox_inputs(arrays, configuration, statistics):
    flags = np.column_stack((arrays["bbox_valid"], arrays["bbox_velocity_valid"]))
    if configuration == "availability":
        return flags.astype(np.float32)
    if configuration != "geometry":
        raise ValueError("Unknown bbox configuration")
    values, masks = normalization_sample(arrays)
    numeric = normalize(values, masks, statistics)[:, :6]
    return np.concatenate((numeric, flags), axis=1).astype(np.float32)


def load_bbox(path, metadata, track_sha256):
    """Reject stale, misaligned, malformed or numerically corrupted extensions."""
    with np.load(path, allow_pickle=False) as saved:
        reference = json.loads(str(saved["metadata"]))
        arrays = {k: saved[k] for k in saved.files if k != "metadata"}
    expected = {"track_locator": metadata["track_locator"], "source_frames": metadata["source_frames"],
                "source_time_us": source_times(metadata), "track_sha256": track_sha256}
    if any(reference.get(k) != v for k, v in expected.items()):
        raise ValueError("BBox/E001 track reference mismatch")
    n = len(metadata["source_frames"])
    for key in ("bbox_valid", "bbox_velocity_valid", "loki_2d", "road_2d", "waymo_2d"):
        if arrays[key].shape != (n,) or arrays[key].dtype != bool:
            raise ValueError(f"Invalid {key}")
    values = arrays["bbox_features"]
    if values.shape != (n, 6) or values.dtype != np.float64:
        raise ValueError("Invalid bbox features")
    source = arrays["bbox_source"]
    if source.shape != (n,) or source.dtype != np.uint8 or not np.isin(source, list(SOURCES)).all():
        raise ValueError("Invalid bbox source")
    sizes = arrays["image_size"]
    if sizes.shape != (n, 2) or sizes.dtype != np.int64 or (sizes < 0).any():
        raise ValueError("Invalid image dimensions")
    valid, speed = arrays["bbox_valid"], arrays["bbox_velocity_valid"]
    if (speed & ~valid).any() or (valid & ((source == 0) | (sizes <= 0).any(axis=1))).any():
        raise ValueError("Invalid bbox mask dependencies")
    if (values[valid, 2:4] <= 0).any():
        raise ValueError("Invalid bbox dimensions")
    for start, end, mask in ((0, 4, valid), (4, 6, speed)):
        if not np.isfinite(values[mask, start:end]).all() or not np.isnan(values[~mask, start:end]).all():
            raise ValueError("Invalid bbox missingness")
    for key in ("loki_2d", "road_2d", "waymo_2d"):
        expected_presence = np.array([bool(a.get(key, False)) for a in metadata["availability"]])
        if not np.array_equal(arrays[key], expected_presence):
            raise ValueError("BBox native presence mismatch")
    for i, time in enumerate(source_times(metadata)):
        if time is None and (valid[i] or source[i] != 0):
            raise ValueError("Box at missing source frame")
        if speed[i] and not any(0 <= j < n and valid[j] and source[j] == source[i] for j in (i-1, i+1)):
            raise ValueError("BBox velocity bridges a gap/source change")
    return reference, arrays
