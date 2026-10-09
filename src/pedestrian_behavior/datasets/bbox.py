"""Native annotated boxes and image dimensions, without reading 3D geometry."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pedestrian_behavior.data.preparation import checksum
from .loki import _unique_json_object, frame_files
from .road_waymo import annotation_rows, component_path, preparation_rows, unique_row


def image_dimensions(size):
    values = np.asarray(size)
    if values.shape != (2,) or not np.isfinite(values).all() or (values <= 0).any() or (values != values.astype(np.int64)).any():
        raise ValueError("Invalid image dimensions")
    return values.astype(np.int64)


def read_loki_boxes(root, scene, requested, expected_hash):
    """Verify the original scene digest before retrieving selected native boxes."""
    digest, records, sizes = hashlib.sha256(), {}, set()
    for frame, paths in frame_files(Path(root) / scene):
        for kind in ("label2d", "label3d", "odometry"):
            digest.update((kind+frame).encode())
            if paths[kind]:
                digest.update(paths[kind].read_bytes())
        if frame not in requested:
            continue
        labels = json.loads(paths["label2d"].read_text(), object_pairs_hook=_unique_json_object) if paths["label2d"] else {}
        with Image.open(paths["image"]) as image:
            size = image_dimensions(image.size)
        sizes.add(tuple(map(int, size)))
        boxes = {}
        for track, row in labels.get("Pedestrian", {}).items():
            box = row["box"]
            left, top, width, height = (float(box[k]) for k in ("left", "top", "width", "height"))
            boxes[track] = np.array([left, top, left+width, top+height]) / np.tile(size, 2)
        records[frame] = {"image_size": size, "boxes": boxes}
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"{scene}: native LOKI scene differs from frozen E001")
    if set(records) != requested:
        raise ValueError(f"{scene}: missing selected native frame")
    return records, {"annotation_odometry_sha256": digest.hexdigest(), "image_sizes": [list(s) for s in sorted(sizes)],
                     "image_size_evidence": "Pillow header of every selected PNG; pixels are not decoded"}


def read_road_boxes(index, groups):
    """Read the frozen ROAD export once; retain only the eligible native IDs."""
    boxes = {scene: {} for scene in groups}
    for row in annotation_rows(Path(index)):
        scene, track = row["road_clip_id"], row["road_tube_uid"]
        if scene not in groups or track not in groups[scene]:
            continue
        if row["camera_name"] != "1" or row["merged_agent_label"] != "Ped" or row["timestamp_basis"] != "automatic_verified":
            raise ValueError("Unverified ROAD box camera/class/timestamp")
        corners = json.loads(row["road_box_normalized_json"])
        annotation = json.loads(row["road_annotation_json"])
        if annotation["tube_uid"] != track or corners != annotation["box"] or len(corners) != 4:
            raise ValueError("ROAD box/annotation identity or geometry mismatch")
        unique_row(boxes[scene], (track, int(row["frame_timestamp_micros"])), corners)
    return boxes


def read_waymo_boxes(scene, selected_ids, waymo_root=None):
    """FRONT camera labels (never projected LiDAR boxes) and calibration sizes."""
    name = scene["segment_context_name"]
    box_path, calibration = (component_path(scene, c, waymo_root) for c in ("camera_box", "camera_calibration"))
    if box_path is None or calibration is None or not box_path.is_file() or not calibration.is_file():
        raise ValueError("Missing native camera boxes or image dimensions")
    digest = checksum(box_path)
    if digest != scene["component_sha256"]["camera_box"]:
        raise ValueError("Native camera boxes differ from frozen E001")
    prefix = "[CameraCalibrationComponent]."
    sizes = []
    for row in preparation_rows(calibration, name, ["key.segment_context_name", "key.camera_name", prefix+"width", prefix+"height"]):
        if row["key.camera_name"] == 1:
            sizes.append(image_dimensions([row[prefix+"width"], row[prefix+"height"]]))
    if len(sizes) != 1:
        raise ValueError("Expected exactly one FRONT camera dimension record")
    size = sizes[0]
    prefix = "[CameraBoxComponent].box."
    columns = ["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name", "key.camera_object_id"]
    columns += [prefix+k for k in ("center.x", "center.y", "size.x", "size.y")]
    boxes = {}
    for row in preparation_rows(box_path, name, columns):
        if row["key.camera_name"] != 1 or row["key.camera_object_id"] not in selected_ids:
            continue
        x, y, width, height = (float(row[prefix+k]) for k in ("center.x", "center.y", "size.x", "size.y"))
        corners = (np.array([x-width/2, y-height/2, x+width/2, y+height/2]) / np.tile(size, 2)).tolist()
        unique_row(boxes, (row["key.camera_object_id"], row["key.frame_timestamp_micros"]), corners)
    return boxes, size, {"camera_box_path": str(box_path.resolve()), "camera_calibration_path": str(calibration.resolve()),
                         "camera_box_sha256": digest, "camera_calibration_sha256": checksum(calibration),
                         "image_sizes": [size.tolist()], "image_size_evidence": "Native FRONT CameraCalibrationComponent"}
