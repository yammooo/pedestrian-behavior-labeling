"""Read the released, flat LOKI scenario directories without remapping labels."""

import csv
import json
from dataclasses import dataclass
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
    for frame_id in sorted(set().union(*(set(group) for group in files.values()))):
        yield frame_id, {kind: group.get(frame_id) for kind, group in files.items()}


def read_frame(scenario: Path, frame_id: str, paths: dict[str, Path | None]) -> Frame:
    people = {}
    if paths["label2d"]:
        data = json.loads(paths["label2d"].read_text())
        for track_id, row in data.get("Pedestrian", {}).items():
            people[track_id] = Pedestrian(track_id, label2d=row)
    if paths["label3d"]:
        with paths["label3d"].open(newline="") as source:
            for row in csv.DictReader(source, skipinitialspace=True):
                if row["labels"] != "Pedestrian":
                    continue
                track_id = row["track_id"]
                person = people.setdefault(track_id, Pedestrian(track_id))
                person.label3d = row
    return Frame(scenario.name, frame_id, paths, people)


def frames(scenario: Path):
    for frame_id, paths in frame_files(scenario):
        yield read_frame(scenario, frame_id, paths)
