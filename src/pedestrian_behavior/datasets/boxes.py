"""Fresh native cuboids plus the accepted native-only camera box policy."""

from pathlib import Path

import numpy as np

from pedestrian_behavior.data.preparation import checksum, native_float, world_position
from .bbox import read_loki_boxes, read_waymo_boxes
from .loki import frame_files, read_frame
from .road_waymo import component_path, preparation_rows, unique_row


def read_loki_geometry(reader, scene, requested, frozen):
    camera, evidence = read_loki_boxes(reader.root, scene, requested, frozen["annotation_odometry_sha256"])
    records = reader.read_frames(scene, requested)
    paths = dict(frame_files(reader.root/scene))
    for frame_id, record in records.items():
        frame = read_frame(reader.root/scene, frame_id, paths[frame_id])
        for track, person in record["pedestrians"].items():
            row = frame.pedestrians[track].label3d
            person["size"] = [native_float(row[k]) for k in ("dim_x", "dim_y", "dim_z")] if row else None
        record["camera"] = camera[frame_id]
    return records, evidence


def read_waymo_geometry(scene, selected_ids, requested, waymo_root):
    boxes, image_size, evidence = read_waymo_boxes(scene, selected_ids, waymo_root)
    records, cuboids = {}, {}
    name = scene["segment_context_name"]
    for component in ("vehicle_pose", "lidar_box", "camera_to_lidar_box_association"):
        path = component_path(scene, component, waymo_root)
        if path is None or not Path(path).is_file() or checksum(path) != scene["component_sha256"][component]:
            raise ValueError(f"Native {component} differs from frozen E001")
        evidence[component+"_path"], evidence[component+"_sha256"] = str(path.resolve()), checksum(path)
    pose_path = component_path(scene, "vehicle_pose", waymo_root)
    for row in preparation_rows(pose_path, name):
        ts = row["key.frame_timestamp_micros"]
        if ts in requested:
            unique_row(records, ts, row)
    records = {ts: {"ego_pose": np.array(row["[VehiclePoseComponent].world_from_vehicle.transform"], dtype=np.float64).reshape(4, 4),
                    "pedestrians": {}, "camera": {"image_size": image_size,
                    "boxes": {track: boxes[track, ts] for track in selected_ids if (track, ts) in boxes}}}
               for ts, row in records.items()}
    # The accepted E001 association is frozen; never redo the native-ID join here.
    reverse = {scene["association_ids"][track]: track for track in selected_ids if scene["association_ids"][track] is not None}
    if len(reverse) != sum(scene["association_ids"][track] is not None for track in selected_ids):
        raise ValueError("Repeated frozen LiDAR association")
    prefix = "[LiDARBoxComponent].box."
    columns = ["key.segment_context_name", "key.frame_timestamp_micros", "key.laser_object_id"]
    columns += [prefix+field+axis for field in ("center.", "size.") for axis in "xyz"]
    for row in preparation_rows(component_path(scene, "lidar_box", waymo_root), name, columns):
        ts, laser = row["key.frame_timestamp_micros"], row["key.laser_object_id"]
        if ts in requested and laser in reverse:
            unique_row(cuboids, (reverse[laser], ts), row)
    for (track, ts), row in cuboids.items():
        record = records.setdefault(ts, {"ego_pose": None, "pedestrians": {}, "camera": {"image_size": image_size,
            "boxes": {t: boxes[t, ts] for t in selected_ids if (t, ts) in boxes}}})
        center, size = ([native_float(row[prefix+field+axis]) for axis in "xyz"] for field in ("center.", "size."))
        record["pedestrians"][track] = {"world_position": world_position(center, record["ego_pose"]), "size": size}
    for ts in requested:
        records.setdefault(ts, {"ego_pose": None, "pedestrians": {}, "camera": {"image_size": image_size,
            "boxes": {t: boxes[t, ts] for t in selected_ids if (t, ts) in boxes}}})
    return records, evidence
