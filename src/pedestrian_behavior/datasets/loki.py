"""Read the released, flat LOKI scenario directories without remapping labels."""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


KINDS = {
    "image": ("image_", ".png"),
    "label2d": ("label2d_", ".json"),
    "label3d": ("label3d_", ".txt"),
    "pointcloud": ("pc_", ".ply"),
    "odometry": ("odom_", ".txt"),
}


@dataclass
class Pedestrian:
    track_id: str
    label2d: dict | None = None
    label3d: dict | None = None
    label3d_record_indices: list[int] = field(default_factory=list)

    @property
    def action(self) -> str | None:
        return self.label3d.get("intended_actions") if self.label3d else None

    @property
    def box(self) -> dict | None:
        return self.label2d.get("box") if self.label2d else None


@dataclass
class Frame:
    scenario: str
    frame_id: str
    paths: dict[str, Path | None]
    pedestrians: dict[str, Pedestrian]


def scenarios(root: Path):
    yield from sorted(p for p in root.glob("scenario_*") if p.is_dir())


def frame_files(scenario: Path):
    """Yield IDs from the union of available files, with missing paths as None."""
    files = {kind: {} for kind in KINDS}
    for kind, (prefix, suffix) in KINDS.items():
        for path in scenario.glob(f"{prefix}*{suffix}"):
            files[kind][path.name[len(prefix):-len(suffix)]] = path
    for frame_id in sorted(set().union(*(set(group) for group in files.values())), key=int):
        yield frame_id, {kind: group.get(frame_id) for kind, group in files.items()}


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_frame(scenario: Path, frame_id: str, paths: dict[str, Path | None]) -> Frame:
    people = {}
    if paths["label2d"]:
        data = json.loads(paths["label2d"].read_text(), object_pairs_hook=_unique_json_object)
        for track_id, row in data.get("Pedestrian", {}).items():
            people[track_id] = Pedestrian(track_id, label2d=row)
    if paths["label3d"]:
        with paths["label3d"].open(newline="") as source:
            for record_index, row in enumerate(csv.DictReader(source, skipinitialspace=True)):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(f"{scenario}/{frame_id}: malformed 3D row")
                if row["labels"] != "Pedestrian":
                    continue
                track_id = row["track_id"]
                person = people.setdefault(track_id, Pedestrian(track_id))
                if person.label3d is not None and person.label3d != row:
                    raise ValueError(f"{scenario}/{frame_id}/{track_id}: conflicting repeated 3D observation")
                person.label3d = row
                person.label3d_record_indices.append(record_index)
    return Frame(scenario.name, frame_id, paths, people)


def frames(scenario: Path):
    for frame_id, paths in frame_files(scenario):
        yield read_frame(scenario, frame_id, paths)


class TrackReader:
    """LOKI native union tracks; world interpretation requires explicit evidence."""

    dataset = "loki"
    clock = {"basis": "nominal", "unit": "microseconds", "rule": "integer suffix * 100000",
             "qualification": "Documented 5 Hz annotation cadence; physical timestamps unknown"}

    def __init__(self, root, transform_contract=None):
        from pedestrian_behavior.preparation import checksum, valid_pose
        import numpy as np

        self.root = Path(root)
        self.scene_ids = [s.name for s in scenarios(self.root)]
        if not self.scene_ids:
            raise ValueError("No LOKI scenarios")
        self.contract = json.loads(Path(transform_contract).read_text()) if transform_contract else None
        self.sources = {"root": str(self.root), "release": "unknown", "native_split": "unknown", "transform": self.contract,
                        "format_evidence": "https://usa.honda-ri.com/loki"}
        if self.contract:
            if (self.contract.get("units") != "metres-radians" or self.contract.get("axes") != "x-forward-y-left-z-up"
                    or self.contract.get("euler") != "Rz(yaw)Ry(pitch)Rx(roll)"
                    or not self.contract.get("evidence") or self.contract.get("verified") is not True):
                raise ValueError("Unverified LOKI transform interpretation")
            transform = np.asarray(self.contract["ego_from_pointcloud"], dtype=np.float64)
            if not valid_pose(transform):
                raise ValueError("Invalid LOKI pointcloud-to-ego calibration")
            self.extrinsic = transform
            self.sources["transform_sha256"] = checksum(transform_contract)

    def resolve_track(self, address):
        key = json.loads(address)
        if set(key) != {"scenario", "track_id"} or key["scenario"] not in self.scene_ids:
            raise ValueError("Invalid LOKI track locator")
        candidate = self.index_scene(key["scenario"])["tracks"].get(key["track_id"])
        if candidate is None:
            raise KeyError(address)
        return key

    def index_scene(self, scene_id):
        from pedestrian_behavior.preparation import locator
        if scene_id not in self.scene_ids:
            raise KeyError(scene_id)
        import hashlib
        digest = hashlib.sha256()
        catalogue, tracks = [], {}
        for frame in frames(self.root / scene_id):
            if int(frame.frame_id) % 2:
                raise ValueError("Unverified LOKI filename cadence: odd suffix")
            for kind in ("label2d", "label3d", "odometry"):
                path = frame.paths[kind]
                digest.update((kind + frame.frame_id).encode())
                if path:
                    digest.update(path.read_bytes())
            time = int(frame.frame_id) * 100_000
            catalogue.append({"key": frame.frame_id, "time_us": time, "rgb": bool(frame.paths["image"])})
            for track_id, person in frame.pedestrians.items():
                candidate = tracks.setdefault(track_id, {"locator": locator(scenario=scene_id, track_id=track_id),
                    "extent_us": [time, time], "counts": {"observations": 0, "2d": 0, "3d": 0, "annotations": 0, "duplicate_annotations": 0}})
                candidate["extent_us"][1] = time
                counts = candidate["counts"]
                counts["observations"] += 1
                counts["2d"] += person.label2d is not None
                counts["3d"] += person.label3d is not None
                counts["annotations"] += person.label3d is not None
                counts["duplicate_annotations"] += max(0, len(person.label3d_record_indices) - 1)
        return {"frames": sorted(catalogue, key=lambda f: f["time_us"]), "tracks": tracks,
                "provenance": {"scenario": scene_id, "annotation_odometry_sha256": digest.hexdigest()}}

    def read_frames(self, scene_id, frame_ids):
        from pedestrian_behavior.preparation import native_float, valid_pose, world_position
        import numpy as np
        if self.contract is None:
            raise ValueError("LOKI world transform unverified: require evidence-backed --loki-transform; no assumed extrinsic")
        if scene_id not in self.scene_ids:
            raise KeyError(scene_id)
        paths = dict(frame_files(self.root / scene_id))
        result = {}
        for frame_id in frame_ids:
            if frame_id not in paths:
                raise KeyError(frame_id)
            frame = read_frame(self.root / scene_id, frame_id, paths[frame_id])
            pose = None
            if frame.paths["odometry"]:
                values = frame.paths["odometry"].read_text().strip().split(",")
                if len(values) != 6:
                    raise ValueError("LOKI odometry must contain xyz, roll, pitch, yaw")
                x, y, z, roll, pitch, yaw = map(native_float, values)
                cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
                pose = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr, x],
                                 [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr, y],
                                 [-sp, cp*sr, cp*cr, z], [0., 0., 0., 1.]])
            people = {}
            for track_id, person in frame.pedestrians.items():
                semantic = {}
                if person.label2d is not None:
                    semantic["label2d"] = {k: v for k, v in person.label2d.items() if k != "box"}
                if person.label3d is not None:
                    semantic["label3d"] = {k: person.label3d[k] for k in ("intended_actions", "potential_destination", "stationary")
                                           if k in person.label3d}
                    semantic["label3d_record_indices"] = person.label3d_record_indices
                center = [native_float(person.label3d[k]) for k in ("pos_x", "pos_y", "pos_z")] if person.label3d else None
                transformed = world_position(center, pose @ self.extrinsic if valid_pose(pose) else None)
                people[track_id] = {"world_position": transformed, "annotations": semantic,
                    "availability": {"loki_2d": person.label2d is not None, "native_3d": person.label3d is not None},
                    "issues": [] if transformed is not None else ["missing-or-unusable-native-position"]}
            result[frame_id] = {"ego_pose": pose, "pedestrians": people, "availability_fields": ("loki_2d", "native_3d")}
        return result
