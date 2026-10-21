"""Selected native scenes with aligned XYZ/strength, camera boxes and cuboids."""

from io import BytesIO

import numpy as np
from PIL import Image

from pedestrian_behavior.data.preparation import native_float, valid_pose
from pedestrian_behavior.inspection.render import read_points
from pedestrian_behavior.inspection.road_waymo import aligned_frames
from pedestrian_behavior.datasets import loki, road_waymo as road
from pedestrian_behavior.datasets.bbox import read_loki_boxes, read_waymo_boxes


def native_camera(reader, scene_id, keys, tracks, frozen):
    """Read selected images/native boxes without decoding LiDAR or fitting data."""
    keys = set(keys)
    if reader.dataset == "loki":
        paths = dict(loki.frame_files(reader.root/scene_id))
        image_keys = {key for key in keys if paths[key]["image"] is not None}
        camera, _ = read_loki_boxes(reader.root, scene_id, image_keys, frozen["annotation_odometry_sha256"])
        for key in sorted(keys, key=int):
            image = None
            if paths[key]["image"]:
                with Image.open(paths[key]["image"]) as raw:
                    image = raw.convert("RGB")
            yield key, image, camera.get(key, {}).get("boxes", {})
        return
    boxes, size, _ = read_waymo_boxes(frozen, set(tracks), reader.waymo_root)
    scene = reader.scenes[scene_id]
    rows = road.component_rows(road.component_path(scene, "camera_image", reader.waymo_root), scene["segment_context_name"])
    for key, group in zip(sorted(keys, key=int), aligned_frames(road.frame_groups(rows), sorted(keys, key=int))):
        image = None
        for row in group:
            if row["key.camera_name"] == 1 and row.get("[CameraImageComponent].image"):
                with Image.open(BytesIO(row["[CameraImageComponent].image"])) as raw:
                    image = raw.convert("RGB")
                if image.size != tuple(size):
                    raise ValueError("Camera calibration/image dimensions differ")
        yield key, image, {t: boxes[t, key] for t in tracks if (t, key) in boxes}


def native_scene(reader, scene_id, keys, tracks, frozen, *, include_rgb=True):
    """Yield one selected native frame at a time, with no behavior inputs."""
    keys = sorted(set(keys), key=int)
    if reader.dataset == "loki":
        camera = {}
        if include_rgb:
            camera, _ = read_loki_boxes(reader.root, scene_id, set(keys), frozen["annotation_odometry_sha256"])
        paths = dict(loki.frame_files(reader.root/scene_id))
        poses = reader.read_frames(scene_id, keys)
        for key in keys:
            frame = loki.read_frame(reader.root/scene_id, key, paths[key])
            points = np.array(list(read_points(frame.paths["pointcloud"])), dtype=np.float32).reshape(-1, 4) if frame.paths["pointcloud"] else None
            if points is not None and (not np.isfinite(points).all() or (points[:, 3] < 0).any() or (points[:, 3] > 1).any()):
                raise ValueError("LOKI normalized intensity contract violated")
            image = None
            if include_rgb and frame.paths["image"]:
                with Image.open(frame.paths["image"]) as raw:
                    image = raw.convert("RGB")
            boxes = {}
            for track in tracks:
                person = frame.pedestrians.get(track)
                if person and person.label3d:
                    row = person.label3d
                    center, size = ([native_float(row[p+a]) for a in "xyz"] for p in ("pos_", "dim_"))
                    boxes[track] = (np.array(center), np.array(size), native_float(row["yaw"]))
            pose = poses[key]["ego_pose"]
            yield {"key": key, "points": points, "image": image, "boxes": boxes,
                   "camera": camera[key]["boxes"] if include_rgb else {}, "world_from_native": pose @ reader.extrinsic if valid_pose(pose) else None,
                   "files": {k: str(v) if v else None for k, v in frame.paths.items()}}
        return
    scene = reader.scenes[scene_id]
    actual = reader.index_scene(scene_id)["provenance"]
    if actual["component_sha256"] != frozen["component_sha256"] or actual["association_ids"] != frozen["association_ids"]:
        raise ValueError("Native ROAD components/associations differ from frozen E001")
    camera, size = {}, None
    if include_rgb:
        camera, size, _ = read_waymo_boxes(frozen, set(tracks), reader.waymo_root)
    def rows(component):
        return road.component_rows(road.component_path(scene, component, reader.waymo_root), scene["segment_context_name"])
    image_groups = iter(aligned_frames(road.frame_groups(rows("camera_image")), keys)) if include_rgb else None
    clouds = iter(aligned_frames(road.frame_groups(rows("lidar")), keys))
    poses = iter(aligned_frames(road.frame_groups(rows("lidar_pose")), keys))
    calibration = {row["key.laser_name"]: row for row in rows("lidar_calibration")}
    for key in keys:
        image, parts = None, []
        for row in next(image_groups) if image_groups is not None else []:
            if row["key.camera_name"] == 1 and row.get("[CameraImageComponent].image"):
                with Image.open(BytesIO(row["[CameraImageComponent].image"])) as raw:
                    image = raw.convert("RGB")
                if image.size != tuple(size):
                    raise ValueError("Camera calibration/image dimensions differ")
        pixel = {row["key.laser_name"]: row for row in next(poses)}
        pose_row = reader.poses.get(key)
        pose = np.asarray(pose_row["[VehiclePoseComponent].world_from_vehicle.transform"]).reshape(4, 4) if pose_row else None
        for row in next(clouds):
            laser = row["key.laser_name"]
            if laser not in calibration or (laser == 1 and (laser not in pixel or pose is None)):
                raise ValueError("Missing native LiDAR calibration/motion pose")
            for ret in (1, 2):
                prefix = f"[LiDARComponent].range_image_return{ret}."
                if row[prefix+"values"] is None:
                    continue
                pp = pixel[laser]["[LiDARPoseComponent].range_image_return1.values"] if laser == 1 else None
                part = road.range_points(row[prefix+"values"], row[prefix+"shape"], calibration[laser],
                                         pp, pose if laser == 1 else None, with_strength=True)
                part[:, 3] = np.tanh(part[:, 3])
                parts.append(part.astype(np.float32))
        boxes = {}
        for track in tracks:
            b = reader.lidar.get((frozen["association_ids"][track], key))
            if b:
                p = "[LiDARBoxComponent].box."
                center, extent = (np.array([b[p+field+a] for a in "xyz"]) for field in ("center.", "size."))
                boxes[track] = (center, extent, b[p+"heading"])
        yield {"key": key, "points": np.concatenate(parts) if parts else None, "image": image,
               "boxes": boxes, "camera": {t: camera[t, key] for t in tracks if (t, key) in camera},
               "world_from_native": pose, "files": {c: str(road.component_path(scene, c, reader.waymo_root))
                   for c in ("camera_image", "lidar", "lidar_pose", "lidar_calibration")}}
