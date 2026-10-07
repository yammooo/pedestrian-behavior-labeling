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


def preparation_rows(path, scene_name, columns=None):
    """Column-projected Parquet reads; preparation never decodes image/LiDAR arrays."""
    import pyarrow.parquet as pq
    if path is None or not path.is_file():
        return
    last = -1
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns, use_threads=False):
        for row in batch.to_pylist():
            ts = row.get("key.frame_timestamp_micros", last)
            if row["key.segment_context_name"] != scene_name or ts < last:
                raise ValueError(f"{path}: wrong scene or unordered timestamps")
            last = ts
            yield row


def unique_row(rows, key, row):
    if key in rows and rows[key] != row:
        raise ValueError(f"Conflicting native duplicate: {key}")
    rows[key] = row


class TrackReader:
    """ROAD tube population, official native-ID extensions and same-frame poses."""

    dataset = "road-waymo"
    clock = {"basis": "physical", "unit": "microseconds", "rule": "Waymo frame_timestamp_micros",
             "qualification": "Frame pose defines native label coordinates; pose instant differs from frame start"}

    def __init__(self, index, waymo_root=None):
        from pedestrian_behavior.preparation import checksum
        self.index, self.waymo_root = Path(index), waymo_root
        manifest = json.loads((self.index / "scene_manifest.json").read_text())
        self.scenes = {s["road_clip_id"]: s for s in manifest}
        if len(self.scenes) != len(manifest):
            raise ValueError("Duplicate ROAD scene manifest entries")
        self.sources = {"index": str(self.index), "waymo_root": str(waymo_root) if waymo_root else None,
            "release": "ROAD-Waymo acquired trainval v1.0; Waymo release unknown",
            "vocabulary": json.loads((self.index / "road_label_definitions.json").read_text()),
            "sha256": {f: checksum(self.index / f) for f in
                       ("pedestrians.csv.gz", "scene_manifest.json", "road_label_definitions.json")},
            "evidence": ["https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/perception/box.py",
                         "https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/dataset.proto"]}
        self.road = {}
        self.frame_mapping = {}
        for row in annotation_rows(self.index):
            scene, track = row["road_clip_id"], row["road_tube_uid"]
            if scene not in self.scenes or row["merged_agent_label"] != "Ped" or row["camera_name"] != "1":
                raise ValueError("ROAD population/scene/camera mismatch")
            if row["segment_context_name"] != self.scenes[scene]["segment_context_name"]:
                raise ValueError("ROAD segment mismatch")
            if row["timestamp_basis"] != "automatic_verified":
                raise ValueError("Unverified ROAD timestamp interpretation")
            flags = {}
            for flag in ("has_3d_box", "semantic_disagreement"):
                if row[flag] not in ("True", "False"):
                    raise ValueError(f"Invalid {flag}")
                flags[flag] = row[flag] == "True"
            timestamp = int(row["frame_timestamp_micros"])
            frame_key = scene, int(row["road_frame_1based"])
            unique_row(self.frame_mapping, frame_key, timestamp)
            if flags["has_3d_box"] and not row["official_laser_object_id"]:
                raise ValueError("Exported 3D pair lacks an official identity")
            observation = {"annotation": {k: v for k, v in json.loads(row["road_annotation_json"]).items() if k != "box"},
                "actions": json.loads(row["action_labels_json"]), "locations": json.loads(row["loc_labels_json"]),
                "frame": int(row["road_frame_1based"]), "road_split": row["road_split"],
                "laser_id": row["official_laser_object_id"] or None,
                "flags": flags | {"association_status": row["association_status"],
                    "export_3d_type": row["waymo_3d_type"], "export_3d_label": row["waymo_3d_label"]}}
            timestamps = self.road.setdefault(scene, {}).setdefault(track, {})
            ts = int(row["frame_timestamp_micros"])
            previous = timestamps.get(ts)
            if previous is not None:
                if {k: v for k, v in previous.items() if k != "annotation_ids"} != observation:
                    raise ValueError(f"{scene}/{track}/{ts}: conflicting repeated observation")
                previous["annotation_ids"].append(row["road_annotation_id"])
            else:
                timestamps[ts] = observation | {"annotation_ids": [row["road_annotation_id"]]}
        self.scene_ids = sorted(self.road)
        if not self.scene_ids:
            raise ValueError("No ROAD candidate identities")
        self._scene_id = None

    def resolve_track(self, address):
        key = json.loads(address)
        if set(key) != {"clip", "tube_uid"} or key["tube_uid"] not in self.road.get(key["clip"], {}):
            raise ValueError("Invalid ROAD-Waymo track locator")
        return key

    def index_scene(self, scene_id):
        from pedestrian_behavior.preparation import checksum, locator
        scene = self.scenes[scene_id]
        name = scene["segment_context_name"]
        def rows(component, columns=None):
            return preparation_rows(component_path(scene, component, self.waymo_root), name, columns)
        # One scene at a time bounds native component memory.
        self.camera, self.lidar, self.poses = {}, {}, {}
        timestamps, images = set(), set()
        for r in rows("vehicle_pose"):
            ts = r["key.frame_timestamp_micros"]
            unique_row(self.poses, ts, r)
            timestamps.add(ts)
        image_columns = ["key.segment_context_name", "key.frame_timestamp_micros", "key.camera_name"]
        image_path = component_path(scene, "camera_image", self.waymo_root)
        image_known = image_path is not None and image_path.is_file()
        for r in rows("camera_image", image_columns):
            ts = r["key.frame_timestamp_micros"]
            timestamps.add(ts)
            if r["key.camera_name"] == 1:
                if ts in images:
                    raise ValueError("Duplicate FRONT image/frame")
                images.add(ts)
        for r in rows("camera_box"):
            if r["key.camera_name"] == 1:
                key = r["key.camera_object_id"], r["key.frame_timestamp_micros"]
                unique_row(self.camera, key, r)
                timestamps.add(key[1])
        for r in rows("lidar_box"):
            key = r["key.laser_object_id"], r["key.frame_timestamp_micros"]
            unique_row(self.lidar, key, r)
            timestamps.add(key[1])
        associations = {}
        for r in rows("camera_to_lidar_box_association"):
            if r["key.camera_name"] == 1:
                associations.setdefault(r["key.camera_object_id"], set()).add(r["key.laser_object_id"])
        tracks, laser_owner = {}, {}
        for track, road_observations in self.road[scene_id].items():
            exported_ids = {o["laser_id"] for o in road_observations.values() if o["laser_id"]}
            ids = exported_ids | associations.get(track, set())
            if len(ids) > 1:
                raise ValueError(f"{scene_id}/{track}: conflicting official associations")
            laser_id = next(iter(ids), None)
            if exported_ids and not exported_ids.issubset(associations.get(track, set())):
                raise ValueError(f"{scene_id}/{track}: exported association lacks official native evidence")
            if laser_id:
                if laser_id in laser_owner and laser_owner[laser_id] != track:
                    raise ValueError("One native LiDAR identity assigned to multiple ROAD tubes")
                laser_owner[laser_id] = track
            camera_times = {ts for (obj, ts) in self.camera if obj == track}
            lidar_times = {ts for (obj, ts) in self.lidar if obj == laser_id}
            native = set(road_observations) | camera_times | lidar_times
            timestamps.update(road_observations)
            tracks[track] = {"locator": locator(clip=scene_id, tube_uid=track), "extent_us": [min(native), max(native)],
                "laser_id": laser_id, "counts": {"observations": len(native), "road_2d": len(road_observations),
                "waymo_2d": len(camera_times), "3d": len(lidar_times), "annotations": len(road_observations),
                "duplicate_annotations": sum(len(o["annotation_ids"]) - 1 for o in road_observations.values())}}
        from collections import Counter
        native_action_sets = Counter(json.dumps(sorted(o["actions"])) for obs in self.road[scene_id].values() for o in obs.values())
        self._scene_id, self._tracks = scene_id, tracks
        return {"frames": [{"key": ts, "time_us": ts, "rgb": ts in images if image_known else None}
                           for ts in sorted(timestamps)], "tracks": tracks,
                "provenance": scene | {"component_sha256": {c: checksum(path)
                    for c in ("vehicle_pose", "camera_box", "lidar_box", "camera_to_lidar_box_association")
                    if (path := component_path(scene, c, self.waymo_root)) is not None and path.is_file()},
                    "native_action_sets": dict(native_action_sets), "association_ids": {k: v["laser_id"] for k, v in tracks.items()}}}

    def read_frames(self, scene_id, frame_ids):
        from pedestrian_behavior.preparation import world_position
        if self._scene_id != scene_id:
            self.index_scene(scene_id)
        result = {}
        for ts in frame_ids:
            pose_row = self.poses.get(ts)
            pose = np.asarray(pose_row["[VehiclePoseComponent].world_from_vehicle.transform"], dtype=np.float64).reshape(4, 4) if pose_row else None
            people = {}
            for track, candidate in self._tracks.items():
                observation = self.road[scene_id][track].get(ts)
                camera = self.camera.get((track, ts))
                box = self.lidar.get((candidate["laser_id"], ts))
                if observation is None and camera is None and box is None:
                    continue
                if observation and observation["flags"]["has_3d_box"] and box is None:
                    raise ValueError(f"{scene_id}/{track}/{ts}: exported pair missing from native source")
                semantic = {}
                if observation is not None:
                    semantic["road"] = {k: observation[k] for k in ("annotation", "actions", "locations", "annotation_ids")}
                center = [box["[LiDARBoxComponent].box.center." + a] for a in "xyz"] if box else None
                people[track] = {"world_position": world_position(center, pose), "annotations": semantic,
                    "availability": {"road_2d": observation is not None, "waymo_2d": camera is not None,
                                     "native_3d": box is not None},
                    "issues": [], "native_type": box.get("[LiDARBoxComponent].type") if box else None,
                    "export_flags": observation["flags"] if observation else None}
            result[ts] = {"ego_pose": pose, "pedestrians": people}
        return result
