"""On-demand same-frame context, shared across every model of a track."""

from io import BytesIO
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from pedestrian_behavior.data.preparation import checksum, load_track
from pedestrian_behavior.data.preparation import source_times
from pedestrian_behavior.datasets import loki, road_waymo as road
from pedestrian_behavior.datasets.bbox import read_road_boxes
from pedestrian_behavior.inspection.prepared_tracks import json_safe, preview
from pedestrian_behavior.inspection.render import read_points
from pedestrian_behavior.inspection.road_waymo import aligned_frames
from .prediction_index import write_json


def native_frames(reader, metadata):
    """Yield only selected source frames; large point clouds live for one frame."""
    identity = json.loads(metadata["track_locator"])
    keys = metadata["source_frames"]
    if reader.dataset == "loki":
        scenario = reader.root/identity["scenario"]
        paths = dict(loki.frame_files(scenario))
        for key in keys:
            image = points = box2d = box3d = None
            if key is not None:
                frame = loki.read_frame(scenario, key, paths[key])
                person = frame.pedestrians.get(identity["track_id"])
                if frame.paths["image"]:
                    with Image.open(frame.paths["image"]) as raw:
                        image = raw.convert("RGB")
                    if person and person.box:
                        b = person.box
                        box2d = [b["left"]/image.width, b["top"]/image.height,
                                 (b["left"]+b["width"])/image.width, (b["top"]+b["height"])/image.height]
                if frame.paths["pointcloud"]:
                    points = np.asarray(list(read_points(frame.paths["pointcloud"])), dtype=np.float64).reshape(-1, 4)[:, :3]
                if person:
                    box3d = person.label3d
            yield image, points, box2d, box3d, None, []
        return
    scene_id, track = identity["clip"], identity["tube_uid"]
    if getattr(reader, "_scene_id", None) != scene_id:
        reader.index_scene(scene_id)
    if getattr(reader, "_review_boxes_scene", None) != scene_id:
        reader._review_boxes = read_road_boxes(reader.index, {scene_id: set(reader.road[scene_id])})[scene_id]
        reader._review_boxes_scene = scene_id
    scene = reader.scenes[scene_id]
    def rows(component):
        return road.component_rows(road.component_path(scene, component, reader.waymo_root), scene["segment_context_name"])
    selected = [k for k in keys if k is not None]
    image_groups = iter(aligned_frames(road.frame_groups(rows("camera_image")), selected))
    clouds = iter(aligned_frames(road.frame_groups(rows("lidar")), selected))
    poses = iter(aligned_frames(road.frame_groups(rows("lidar_pose")), selected))
    calibration = {r["key.laser_name"]: r for r in rows("lidar_calibration")}
    laser_id = reader._tracks[track]["laser_id"]
    for key in keys:
        if key is None:
            yield None, None, None, None, None, ["No selected native frame"]
            continue
        image = box2d = box3d = road_box = None
        issues, parts = [], []
        for row in next(image_groups):
            if row["key.camera_name"] == 1 and row.get("[CameraImageComponent].image"):
                with Image.open(BytesIO(row["[CameraImageComponent].image"])) as raw:
                    image = raw.convert("RGB")
        pixel = {r["key.laser_name"]: r for r in next(poses)}
        for row in next(clouds):
            laser = row["key.laser_name"]
            pose = pixel.get(laser)
            if laser not in calibration or (laser == 1 and (pose is None or key not in reader.poses)):
                issues.append(f"Sensor {laser}: missing calibration/motion pose")
                continue
            for ret in (1, 2):
                p = f"[LiDARComponent].range_image_return{ret}."
                if row[p+"values"] is None:
                    continue
                frame_pose = np.asarray(reader.poses[key]["[VehiclePoseComponent].world_from_vehicle.transform"]).reshape(4, 4) if laser == 1 else None
                if laser == 1 and pose["[LiDARPoseComponent].range_image_return1.shape"] != [*row[p+"shape"][:2], 6]:
                    raise ValueError("TOP motion pose and range image shapes differ")
                parts.append(road.range_points(row[p+"values"], row[p+"shape"], calibration[laser],
                    pose["[LiDARPoseComponent].range_image_return1.values"] if laser == 1 else None, frame_pose))
        b = reader.camera.get((track, key))
        if b and image is not None:
            p = "[CameraBoxComponent].box."
            x, y, w, h = (b[p+k] for k in ("center.x", "center.y", "size.x", "size.y"))
            box2d = [(x-w/2)/image.width, (y-h/2)/image.height, (x+w/2)/image.width, (y+h/2)/image.height]
        road_box = reader._review_boxes.get((track, key))
        b = reader.lidar.get((laser_id, key))
        if b:
            p = "[LiDARBoxComponent].box."
            box3d = dict(zip(("pos_x", "pos_y", "dim_x", "dim_y", "yaw"),
                (b[p+k] for k in ("center.x", "center.y", "size.x", "size.y", "heading"))))
        yield image, np.concatenate(parts) if parts else None, box2d, box3d, road_box, issues


def valid_box(box):
    return box is not None and len(box) == 4 and np.isfinite(box).all() and box[2] > box[0] and box[3] > box[1]


def render_context_frame(frame, folder, number):
    image, points, box, box3d, road_box, issues = frame
    box = box if valid_box(box) else None
    combined_path = folder/f"{number:04d}-combined.jpg"
    preview(image, points, box, box3d, combined_path)
    with Image.open(combined_path) as combined:
        rgb = combined.crop((0, 100, 640, 503))
        rgb.save(folder/f"{number:04d}-rgb.jpg")
        combined.crop((640, 0, 1244, 604)).save(folder/f"{number:04d}-bev.jpg")
    combined_path.unlink()
    crop = Image.new("RGB", (256, 320), "#111")
    if image is not None and box is not None:
        corners = [max(0, min(1, v))*size for v, size in zip(box, (image.width, image.height)*2)]
        if corners[2] > corners[0] and corners[3] > corners[1]:
            fitted = ImageOps.contain(image.crop(corners), crop.size)
            crop.paste(fitted, ((crop.width-fitted.width)//2, (crop.height-fitted.height)//2))
    else:
        ImageDraw.Draw(crop).text((8, 12), "No native target crop", fill="white")
    crop.save(folder/f"{number:04d}-crop.jpg")
    if image is not None and valid_box(road_box):
        fitted = ImageOps.contain(image, rgb.size)
        ox, oy = (rgb.width-fitted.width)//2, (rgb.height-fitted.height)//2
        x1, y1, x2, y2 = road_box
        ImageDraw.Draw(rgb).rectangle((ox+x1*fitted.width, oy+y1*fitted.height,
                                      ox+x2*fitted.width, oy+y2*fitted.height), outline="cyan", width=2)
    rgb.save(folder/f"{number:04d}-rgb-road.jpg")
    return {"rgb": image is not None, "lidar": points is not None, "native_box2d": box is not None,
            "road_box2d": valid_box(road_box), "issues": issues}


class ContextCache:
    def __init__(self, outputs, cache, loki_root=None, road_index=None, waymo_root=None):
        self.outputs, self.cache = Path(outputs).resolve(), Path(cache).resolve()
        self.loki_root, self.road_index, self.waymo_root = loki_root, road_index, waymo_root
        self.readers = {}

    def archive(self, dataset, reference):
        path = Path(reference)
        if path.is_file():
            return path.resolve()
        candidate = self.outputs/"experiments/E001/reader-preparation"/dataset/"tracks"/path.name
        if not candidate.is_file():
            raise FileNotFoundError("Prepared track archive unavailable; check --outputs-root")
        return candidate

    def prepare(self, dataset, reference, locator, progress, expected=None):
        started = time.monotonic()
        archive = self.archive(dataset, reference)
        metadata, arrays = load_track(archive)
        if metadata["track_locator"] != locator:
            raise ValueError("Native context archive identity mismatch")
        if expected is not None:
            grid = (metadata["native_extent_us"][0]+np.arange(len(metadata["source_frames"]))*200000).tolist()
            if grid != expected["grid_times_us"] or source_times(metadata) != expected["source_times_us"]:
                raise ValueError("Native context timestamps differ from saved predictions")
        manifest_path = archive.parent.parent/"manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["dataset"] != dataset:
            raise ValueError("Native collection dataset mismatch")
        sources, identity = manifest["sources"], json.loads(locator)
        root = Path(self.loki_root or sources["root"]) if dataset == "loki" else Path(self.road_index or sources["index"])
        scene = identity.get("scenario", identity.get("clip"))
        files = []
        if dataset == "loki":
            selected = set(metadata["source_frames"])
            for key, paths in loki.frame_files(root/scene):
                if key in selected:
                    files.extend(p for p in paths.values() if p and p.is_file())
        else:
            rows = json.loads((root/"scene_manifest.json").read_text())
            record = next(r for r in rows if r["road_clip_id"] == scene)
            files = [road.component_path(record, c, self.waymo_root) for c in record["component_paths"]]
            files += [root/f for f in ("scene_manifest.json", "pedestrians.csv.gz", "road_label_definitions.json")]
            files = [p for p in files if p and p.is_file()]
        if not files:
            raise FileNotFoundError("Native storage unavailable; mount the dataset or set its root")
        if dataset == "road-waymo" and checksum(root/"pedestrians.csv.gz") != sources["sha256"]["pedestrians.csv.gz"]:
            raise ValueError("ROAD annotations differ from frozen collection")
        stamps = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(set(files))]
        evidence = {"archive_sha256": checksum(archive), "manifest_sha256": checksum(manifest_path),
                    "source_files": stamps, "renderer": {p.name: checksum(p) for p in
                    (Path(__file__), Path(__file__).with_name("prepared_tracks.py"), Path(__file__).with_name("render.py"))}}
        key = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
        destination = self.cache/"context"/key
        if (destination/"context.json").is_file():
            return json.loads((destination/"context.json").read_text()) | {"key": key, "cache_hit": True}
        reader_key = (dataset, str(root), str(self.waymo_root))
        if reader_key not in self.readers:
            self.readers[reader_key] = loki.TrackReader(root) if dataset == "loki" else road.TrackReader(root, self.waymo_root)
        reader = self.readers[reader_key]
        expected = manifest["scenes"][scene]
        actual = reader.index_scene(scene)["provenance"]
        if dataset == "loki":
            if actual["annotation_odometry_sha256"] != expected["annotation_odometry_sha256"]:
                raise ValueError("Native annotations differ from frozen collection")
        elif actual["component_sha256"] != expected["component_sha256"]:
            raise ValueError("Native association/box components differ from frozen collection")
        temporary = destination.with_suffix(".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        frames = []
        try:
            for i, frame in enumerate(native_frames(reader, metadata)):
                frames.append(render_context_frame(frame, temporary, i))
                progress(i+1, len(metadata["source_frames"]))
            result = {"locator": locator, "frames": frames, "evidence": evidence,
                      "seconds": time.monotonic()-started, "cache_hit": False,
                      "trajectory": json_safe({k: arrays[k] for k in ("ped_position", "ego_position")})}
            write_json(temporary/"context.json", result)
            temporary.rename(destination)
            return result | {"key": key}
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
