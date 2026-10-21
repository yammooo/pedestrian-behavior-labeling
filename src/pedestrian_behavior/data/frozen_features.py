"""Validated frozen feature payloads and pedestrian readout geometry."""

import numpy as np

from .feature_cache import validate_arrays
from .preparation import valid_pose
from pedestrian_behavior.inspection.feature_probe import box_membership, relative_xyz


def validate_rgb(arrays):
    validate_arrays(arrays)
    if set(arrays) != {"cls", "valid", "source_keys", "times_us", "crop_info"}:
        raise ValueError("Unexpected RGB payload fields")
    n = len(arrays["valid"])
    if (arrays["cls"].shape != (n, 768) or arrays["cls"].dtype != np.float32 or
            arrays["valid"].shape != (n,) or arrays["valid"].dtype != bool or
            arrays["times_us"].shape != (n,) or arrays["times_us"].dtype != np.int64 or
            arrays["source_keys"].shape != (n,) or arrays["source_keys"].dtype.kind != "U" or
            arrays["crop_info"].shape != (n,) or arrays["crop_info"].dtype.kind != "U" or
            np.any(arrays["cls"][~arrays["valid"]] != 0)):
        raise ValueError("Misaligned/nonzero-missing RGB features")


def validate_points(arrays):
    validate_arrays(arrays)
    if set(arrays) != {"features", "coordinates", "native_indices", "world_from_native", "locators", "boxes", "anchors"}:
        raise ValueError("Unexpected point payload fields")
    n = len(arrays["features"])
    if (arrays["features"].shape != (n, 64) or arrays["features"].dtype != np.float32 or
            arrays["coordinates"].shape != (n, 3) or arrays["coordinates"].dtype != np.float32 or
            arrays["native_indices"].shape != (n,) or arrays["native_indices"].dtype != np.int64 or
            (arrays["native_indices"] < 0).any() or len(np.unique(arrays["native_indices"])) != n or
            arrays["world_from_native"].dtype != np.float64 or not valid_pose(arrays["world_from_native"]) or
            arrays["locators"].ndim != 1 or arrays["locators"].dtype.kind != "U" or
            len(set(arrays["locators"].tolist())) != len(arrays["locators"]) or
            arrays["anchors"].shape != (len(arrays["locators"]), 4, 4) or arrays["anchors"].dtype != np.float64 or
            any(not valid_pose(a) for a in arrays["anchors"]) or
            arrays["boxes"].shape != (len(arrays["locators"]), 7) or arrays["boxes"].dtype != np.float64 or
            (arrays["boxes"][:, 3:6] <= 0).any()):
        raise ValueError("Invalid/misaligned point features or references")


def point_union(points, features, selected, boxes, world_from_native, anchors, radius=20.):
    """Keep each voxel once in the union; per-track offsets are reconstructed on read."""
    if not valid_pose(world_from_native):
        raise ValueError("Missing required native transform")
    if (points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all() or
            features.shape != (len(selected), 64) or features.dtype != np.float32 or
            selected.dtype != np.int64 or (selected < 0).any() or (selected >= len(points)).any() or
            len(np.unique(selected)) != len(selected) or not np.isfinite(features).all()):
        raise ValueError("Invalid full-sweep encoder features/native indices")
    coordinates = points[selected, :3].astype(np.float32)
    keep = np.zeros(len(selected), bool)
    locators = sorted(boxes)
    native_boxes = []
    for locator in locators:
        center, size, yaw = boxes[locator]
        if not valid_pose(anchors[locator]):
            raise ValueError("Missing required initial ego anchor")
        box_membership(coordinates, center, size, yaw)  # Validate native geometry, even for empty scans.
        offsets = relative_xyz(coordinates, center, world_from_native, anchors[locator])
        keep |= np.linalg.norm(offsets[:, :2], axis=1) <= radius
        native_boxes.append([*center, *size, yaw])
    arrays = dict(features=features[keep].copy(), coordinates=coordinates[keep].copy(),
                  native_indices=selected[keep].copy(), world_from_native=np.asarray(world_from_native, np.float64),
                  locators=np.array(locators, dtype=str),
                  anchors=np.array([anchors[k] for k in locators], np.float64).reshape(-1,4,4), boxes=np.array(native_boxes, np.float64).reshape(-1, 7))
    validate_points(arrays)
    return arrays


def point_crop(arrays, locator, radius=20.):
    validate_points(arrays)
    matches = np.flatnonzero(arrays["locators"] == locator)
    if len(matches) != 1:
        raise ValueError("Requested pedestrian absent from frame union")
    box = arrays["boxes"][matches[0]]
    anchor = arrays["anchors"][matches[0]]
    offsets = relative_xyz(arrays["coordinates"], box[:3], arrays["world_from_native"], anchor)
    keep = np.linalg.norm(offsets[:, :2], axis=1) <= radius
    return {"features": arrays["features"][keep], "relative_xyz": offsets[keep],
            "inside_box": box_membership(arrays["coordinates"][keep], box[:3], box[3:6], box[6]),
            "native_indices": arrays["native_indices"][keep], "coordinates": arrays["coordinates"][keep]}


def empty_crop():
    return {"features": np.zeros((0,64), np.float32), "relative_xyz": np.zeros((0,3), np.float32),
            "inside_box": np.zeros(0, bool), "native_indices": np.zeros(0, np.int64),
            "coordinates": np.zeros((0,3), np.float32)}
