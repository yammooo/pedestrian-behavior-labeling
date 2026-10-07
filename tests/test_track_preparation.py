"""Human-readable acceptance scenarios for the combined native→saved-track flow."""

import csv
import gzip
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pedestrian_behavior.datasets.loki import TrackReader as LokiReader
from pedestrian_behavior.datasets.road_waymo import TrackReader as RoadReader
from pedestrian_behavior.preparation import (load_track, locator, prepare_collection, prepare_track,
    save_track, select_frames, source_times, velocities)


def pose(x=0., y=0., yaw=0., pitch=0., roll=0., z=0.):
    """Fixture constructor only, independent of both production readers."""
    c, s = math.cos(yaw), math.sin(yaw)
    cp, sp, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    p = np.eye(4)
    p[:3, :3], p[:3, 3] = rz @ ry @ rx, [x, y, z]
    return p


def native_readers(root, times, poses, people, *, actions=None, camera=None, road_times=None):
    """Write native CSV/JSON/odometry and native Waymo v2 Parquet components.

    Fixture semantic IDs are deliberately synthetic; no real labels are physical oracles.
    Each person maps source index→WORLD xyz (or None for an unusable center).
    """
    actions = actions or {}
    camera = camera if camera is not None else {p: set(v) for p, v in people.items()}
    loki_root, index = root / "loki", root / "road"
    scenario = loki_root / "scenario_000"
    scenario.mkdir(parents=True)
    index.mkdir()
    lidar, cam, assoc, vehicle, images, exported = [], [], [], [], [], []
    for i, (time, transform) in enumerate(zip(times, poses)):
        suffix = f"{i * 2:04d}"
        # LOKI's native nominal clock cannot encode an irregular physical clock.
        yaw = math.atan2(transform[1, 0], transform[0, 0])
        pitch = math.asin(-transform[2, 0])
        roll = math.atan2(transform[2, 1], transform[2, 2])
        (scenario / f"odom_{suffix}.txt").write_text(",".join(map(str, [*transform[:3, 3], roll, pitch, yaw])))
        rows3d, labels2d = [], {}
        key = {"key.segment_context_name": "scene", "key.frame_timestamp_micros": time}
        vehicle.append(key | {"[VehiclePoseComponent].world_from_vehicle.transform": transform.ravel().tolist()})
        images.append(key | {"key.camera_name": 1})
        for track, observations in people.items():
            native = None
            if i in observations:
                world = observations[i]
                native = [math.nan]*3 if world is None else (np.linalg.inv(transform) @ [*world, 1])[:3].tolist()
                rows3d.append(dict(labels="Pedestrian", track_id=track, stationary="dynamic",
                    pos_x=native[0], pos_y=native[1], pos_z=native[2], dim_x=1, dim_y=1, dim_z=2, yaw=0,
                    intended_actions=actions.get((track, i), "Moving"), potential_destination="destination-id"))
                lidar.append(key | {"key.laser_object_id": "laser-" + track,
                    **{"[LiDARBoxComponent].box.center." + a: v for a, v in zip("xyz", native)},
                    "[LiDARBoxComponent].type": 2})
            if i in camera.get(track, set()):
                labels2d[track] = {"box": {"left": 1, "top": 2, "width": 3, "height": 4},
                                  "not_in_lidar": native is None, "attributes": {"age": "Adult", "gender": "Female"}}
                cam.append(key | {"key.camera_name": 1, "key.camera_object_id": track})
                # Official native-ID link is available even when same-frame 3D is absent.
                assoc.append(key | {"key.camera_name": 1, "key.camera_object_id": track,
                                    "key.laser_object_id": "laser-" + track})
            if i in (road_times.get(track, set()) if road_times is not None else camera.get(track, set())):
                annotation = {"tube_uid": track, "action_ids": actions.get((track, i), [5]),
                              "loc_ids": [7], "box": [.1, .2, .3, .4]}
                exported.append(dict(road_clip_id="clip", road_tube_uid=track, road_frame_1based=i+1,
                    road_annotation_id=f"annotation-{track}-{i}", road_split="train", waymo_split="training",
                    segment_context_name="scene", camera_name="1", timestamp_basis="automatic_verified",
                    frame_timestamp_micros=time, merged_agent_label="Ped", has_3d_box=str(native is not None),
                    semantic_disagreement="False", official_laser_object_id="laser-" + track,
                    association_status="official_id_with_3d_box", waymo_3d_type="2", waymo_3d_label="Pedestrian",
                    road_annotation_json=json.dumps(annotation), action_labels_json=json.dumps(["Mov", "Xing"]),
                    loc_labels_json=json.dumps(["Pav"])))
        (scenario / f"label2d_{suffix}.json").write_text(json.dumps({"Pedestrian": labels2d}))
        if rows3d:
            with (scenario / f"label3d_{suffix}.txt").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows3d[0])); writer.writeheader(); writer.writerows(rows3d)
    components = {"vehicle_pose": vehicle, "camera_box": cam, "lidar_box": lidar,
                  "camera_to_lidar_box_association": assoc, "camera_image": images}
    paths = {}
    for component, rows in components.items():
        if rows:
            path = index / (component + ".parquet")
            pq.write_table(pa.Table.from_pylist(rows), path)
            paths[component] = str(path)
    (index / "scene_manifest.json").write_text(json.dumps([dict(road_clip_id="clip", waymo_split="training",
        camera_name=1, segment_context_name="scene", component_paths=paths)]))
    (index / "road_label_definitions.json").write_text(json.dumps({"all_action_labels": ["synthetic"]}))
    if not exported:
        raise ValueError("Fixture requires ROAD candidates")
    with gzip.open(index / "pedestrians.csv.gz", "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0])); writer.writeheader(); writer.writerows(exported)
    contract = root / "fixture-transform.json"
    contract.write_text(json.dumps({"verified": True, "evidence": "Synthetic fixture only: poses/axes declared here",
        "units": "metres-radians", "axes": "x-forward-y-left-z-up", "euler": "Rz(yaw)Ry(pitch)Rx(roll)",
        "ego_from_pointcloud": np.eye(4).tolist()}))
    return LokiReader(loki_root, contract), RoadReader(index)


class TrackPreparationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def prepare(self, reader, track="person", case="case", expected=None):
        scene = reader.scene_ids[0]
        catalogue = reader.index_scene(scene)
        candidate = catalogue["tracks"][track]
        selected = select_frames(catalogue["frames"], *candidate["extent_us"])
        records = reader.read_frames(scene, {f["key"] for f in selected if f})
        metadata, arrays = prepare_track(track, candidate, selected, records)
        path = self.root / (reader.dataset + "-" + case + ".npz")
        save_track(path, metadata, arrays)
        actual = load_track(path)
        report = os.environ.get("PREPARATION_ACCEPTANCE_OUTPUT")
        if report:
            output = Path(report); output.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copyfile(path, output / path.name)
            (output / (path.stem + "-expected.json")).write_text(json.dumps(expected or {}, indent=2))
        return actual

    def test_stationary_pedestrian_moving_turning_ego(self):
        readers = native_readers(self.root, [0, 200000, 400000],
            [pose(), pose(1, 0, math.pi/2), pose(1, 1, math.pi/2)],
            {"person": {i: [10, 5, 1] for i in range(3)}})
        expected = {"ped_position": [[10, 5]]*3, "ped_velocity": [[0, 0]]*3,
                    "ego_position": [[0, 0], [1, 0], [1, 1]], "ego_yaw": [0, math.pi/2, math.pi/2]}
        for reader in readers:
            with self.subTest(dataset=reader.dataset):
                _, arrays = self.prepare(reader, case="stationary", expected=expected)
                for k, v in expected.items():
                    np.testing.assert_allclose(arrays[k], v, atol=1e-12)

    def test_known_motion_and_uneven_timing(self):
        for number, times in enumerate(([0, 200000, 400000], [0, 210000, 400000])):
            root = self.root / str(number); root.mkdir()
            world = {i: [10 + (t/1e6)**2, 5, 1] for i, t in enumerate(times)}
            readers = native_readers(root, times, [pose()]*3, {"person": world})
            for reader in readers if number == 0 else readers[1:]:
                expected_v = [.2, .4, .6] if number == 0 else [.21, .42, .61]
                expected = {"ped_position": [[world[i][0], 5] for i in range(3)],
                            "ped_velocity": [[v, 0] for v in expected_v]}
                _, arrays = self.prepare(reader, case=f"acceleration-{number}", expected=expected)
                for k, v in expected.items():
                    np.testing.assert_allclose(arrays[k], v, atol=1e-12)
        # Constant 2 m/s has the same derivative at both endpoints and the center.
        for reader in native_readers(self.root / "constant", [0, 200000, 400000], [pose()]*3,
                                     {"person": {0: [10, 5, 1], 1: [10.4, 5, 1], 2: [10.8, 5, 1]}}):
            _, arrays = self.prepare(reader, case="constant", expected={"ped_velocity": [[2, 0]]*3})
            np.testing.assert_allclose(arrays["ped_velocity"], [[2, 0]]*3, atol=1e-12)

    def test_global_invariance_and_full_tilted_transform(self):
        base = [pose(), pose(1, 0, .3, pitch=.2, roll=-.1, z=2), pose(1, 1, .5, pitch=-.1, roll=.2, z=3)]
        world = [[10, 5, 4], [10.4, 5, 5], [10.8, 5, 6]]
        outputs = []
        for variant in range(2):
            root = self.root / str(variant); root.mkdir()
            global_transform = pose(123, -80, yaw=1.2, z=4) if variant else np.eye(4)
            transformed = [(global_transform @ [*p, 1])[:3] for p in world]
            readers = native_readers(root, [0, 200000, 400000], [global_transform @ p for p in base],
                                     {"person": dict(enumerate(transformed))})
            outputs.append([self.prepare(r, case=f"invariance-{variant}")[1] for r in readers])
        for original, transformed in zip(*outputs):
            for field in ("ped_position", "ped_velocity", "ego_position", "ego_velocity"):
                np.testing.assert_allclose(original[field], transformed[field], atol=1e-10)
            np.testing.assert_allclose(np.exp(1j*original["ego_yaw"]), np.exp(1j*transformed["ego_yaw"]), atol=1e-10)
            np.testing.assert_allclose(original["ped_position"], [[10, 5], [10.4, 5], [10.8, 5]], atol=1e-10)

    def test_complete_extent_internal_gaps_and_native_extensions(self):
        readers = native_readers(self.root, [0, 200000, 400000, 600000, 800000], [pose()]*5,
            {"person": {0: [10, 5, 1], 4: [12, 5, 1]}}, camera={"person": {1, 3}}, road_times={"person": {1}})
        for reader in readers:
            metadata, arrays = self.prepare(reader, case="gap", expected={"ped_position_valid": [True, False, False, False, True]})
            self.assertEqual(metadata["native_extent_us"], [0, 800000])
            np.testing.assert_equal(arrays["ped_position_valid"], [True, False, False, False, True])
            self.assertFalse(arrays["ped_velocity_valid"].any())
            if reader.dataset == "road-waymo":
                self.assertEqual([bool(x) for x in metadata["native_annotations"]], [False, True, False, False, False])
        # Identities are scoped by scenario, even if their native string is reused.
        self.assertNotEqual(locator(scenario="a", track_id="person"), locator(scenario="b", track_id="person"))

    def test_independent_missingness_and_no_later_anchor(self):
        readers = native_readers(self.root, [0, 200000, 400000], [pose()]*3,
            {"person": {0: [10, 5, 1]}, "unusable": {0: None}, "two-only": {}, "three-only": {1: [2, 3, 1]}},
            camera={"person": {0, 2}, "unusable": {0}, "two-only": {0, 1, 2}, "three-only": {0}},
            road_times={"person": {0}, "unusable": {0}, "two-only": {0}, "three-only": {0}})
        for reader in readers:
            catalogue = reader.index_scene(reader.scene_ids[0])
            self.assertEqual(set(catalogue["tracks"]), {"person", "unusable", "two-only", "three-only"})
            m, a = self.prepare(reader, case="single")
            self.assertTrue(a["ped_position_valid"][0]); self.assertFalse(a["ped_velocity_valid"].any())
            m, a = self.prepare(reader, "unusable", "unusable")
            self.assertFalse(a["ped_position_valid"].any()); self.assertTrue(m["availability"][0]["native_3d"])
            m, a = self.prepare(reader, "two-only", "two-only")
            self.assertFalse(a["ped_position_valid"].any()); self.assertTrue(a["ego_pose_valid"].all())
            m, a = self.prepare(reader, "three-only", "three-only")
            self.assertTrue(a["ped_position_valid"].any())
            candidate = catalogue["tracks"]["person"]
            selected = select_frames(catalogue["frames"], *candidate["extent_us"])
            records = reader.read_frames(reader.scene_ids[0], {f["key"] for f in selected if f})
            records[selected[0]["key"]]["ego_pose"] = None
            m, a = prepare_track("person", candidate, selected, records)
            self.assertFalse(m["anchor_valid"]); self.assertFalse(a["ego_pose_valid"].any())
            self.assertFalse(a["ped_position_valid"].any()); self.assertTrue(m["native_annotations"][0])

    def test_5hz_tolerance_ties_missing_and_fractional_endpoints(self):
        times = [0, 125000, 275000, 476000, 600000]
        _, reader = native_readers(self.root, times, [pose()]*5,
            {"person": {i: [10+2*t/1e6, 5, 1] for i, t in enumerate(times)}})
        metadata, arrays = self.prepare(reader, case="selection", expected={"source_frames": [0, 125000, None, 600000]})
        self.assertEqual(source_times(metadata), [0, 125000, None, 600000])
        np.testing.assert_allclose(arrays["ped_velocity"][:2], [[2, 0]]*2)
        self.assertFalse(arrays["ped_velocity_valid"][3])
        frames = [{"key": t, "time_us": t} for t in [0, 100000, 390000]]
        self.assertEqual(len(select_frames(frames, 0, 390000)), 2)
        with self.assertRaisesRegex(ValueError, "increasing"):
            select_frames([frames[1], frames[0]], 0, 390000)
        # Scene-first selection must not search a nearby pedestrian observation.
        root = self.root / "scene-first"; root.mkdir()
        _, reader = native_readers(root, [0, 160000, 250000, 400000], [pose()]*4,
            {"person": {0: [10, 5, 1], 2: [10.5, 5, 1], 3: [10.8, 5, 1]}},
            camera={"person": {0, 2, 3}})
        m, a = self.prepare(reader, case="scene-first", expected={"source_frames": [0, 160000, 400000]})
        self.assertEqual(m["source_frames"], [0, 160000, 400000])
        self.assertFalse(a["ped_position_valid"][1]); self.assertEqual(m["native_annotations"][1], {})
        self.assertFalse(m["availability"][1]["road_2d"])
        self.assertFalse(m["availability"][1]["native_3d"])

    def test_native_supervision_and_semantic_independence(self):
        readers = native_readers(self.root, [0, 200000, 400000], [pose()]*3,
            {"person": {0: [10, 5, 1], 1: [10.4, 5, 1], 2: [10.8, 5, 1]}})
        for reader in readers:
            metadata, arrays = self.prepare(reader, case="semantics")
            scene = reader.scene_ids[0]; catalogue = reader.index_scene(scene)
            selected = select_frames(catalogue["frames"], 0, 400000)
            records = reader.read_frames(scene, {f["key"] for f in selected})
            records[selected[0]["key"]]["pedestrians"]["person"]["annotations"] = {"empty": [], "null": None}
            m, changed = prepare_track("person", catalogue["tracks"]["person"], selected, records)
            for k, values in arrays.items():
                np.testing.assert_equal(values, changed[k])
            self.assertEqual(m["native_annotations"][0], {"empty": [], "null": None})
            if reader.dataset == "loki":
                self.assertEqual(metadata["native_annotations"][0]["label3d"]["potential_destination"], "destination-id")
                self.assertEqual(metadata["native_annotations"][0]["label2d"]["attributes"]["age"], "Adult")
            else:
                self.assertEqual(metadata["native_annotations"][0]["road"]["actions"], ["Mov", "Xing"])
                self.assertEqual(metadata["native_annotations"][0]["road"]["annotation"]["loc_ids"], [7])
                self.assertNotIn("box", metadata["native_annotations"][0]["road"]["annotation"])

    def test_identity_associations_duplicates_and_unverified_interpretation(self):
        loki, road = native_readers(self.root, [0, 200000], [pose()]*2,
            {"person": {0: [10, 5, 1], 1: [10.4, 5, 1]}})
        csv_path = road.index / "pedestrians.csv.gz"
        with gzip.open(csv_path, "rt", newline="") as f:
            rows = list(csv.DictReader(f))
        with gzip.open(csv_path, "at", newline="") as f:
            csv.DictWriter(f, fieldnames=list(rows[0])).writerow(rows[0] | {"road_annotation_id": "duplicate-id"})
        duplicate = RoadReader(road.index)
        m, _ = self.prepare(duplicate, case="duplicate")
        self.assertEqual(m["native_annotations"][0]["road"]["annotation_ids"], ["annotation-person-0", "duplicate-id"])
        self.assertEqual(len(m["source_frames"]), 2)
        with gzip.open(csv_path, "at", newline="") as f:
            csv.DictWriter(f, fieldnames=list(rows[0])).writerow(rows[0] | {"action_labels_json": '["Stop"]'})
        with self.assertRaisesRegex(ValueError, "conflicting repeated"):
            RoadReader(road.index)
        # Unused yaw/dimensions do not change kinematic usability.
        label = loki.root / "scenario_000" / "label3d_0000.txt"
        original = label.read_text()
        with label.open(newline="") as f: native_rows = list(csv.DictReader(f))
        native_rows[0].update(yaw="nan", dim_x="nan", dim_y="-1")
        with label.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(native_rows[0])); writer.writeheader(); writer.writerows(native_rows)
        _, usable = self.prepare(loki, case="unused-dimensions")
        self.assertTrue(usable["ped_position_valid"].all())
        native_rows[0]["pos_x"] = ""
        with label.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(native_rows[0])); writer.writeheader(); writer.writerows(native_rows)
        _, unusable = self.prepare(loki, case="empty-position")
        self.assertFalse(unusable["ped_position_valid"][0]); self.assertTrue(unusable["ped_position_valid"][1])
        label.write_text(original)
        r = loki.index_scene("scenario_000")["tracks"]["person"]
        self.assertEqual(loki.resolve_track(r["locator"])["track_id"], "person")
        self.assertEqual(json.loads(locator(track_id='a"\\b', scenario="x"))["track_id"], 'a"\\b')
        with self.assertRaisesRegex(ValueError, "unverified"):
            LokiReader(loki.root).read_frames("scenario_000", {"0000"})
        label = loki.root / "scenario_000" / "label3d_0000.txt"
        line = label.read_text().splitlines()[1]
        with label.open("a") as f: f.write(line + "\n")
        m, _ = self.prepare(loki, case="loki-duplicate")
        self.assertEqual(m["native_annotations"][0]["label3d_record_indices"], [0, 1])
        with label.open("a") as f: f.write(line.replace("Moving", "Stopped") + "\n")
        with self.assertRaisesRegex(ValueError, "conflicting repeated"):
            loki.index_scene("scenario_000")
        # A conflicting official native association fails before preparation.
        assoc_path = road.index / "camera_to_lidar_box_association.parquet"
        associations = pq.read_table(assoc_path).to_pylist()
        associations[1]["key.laser_object_id"] = "different-object"
        pq.write_table(pa.Table.from_pylist(associations), assoc_path)
        with self.assertRaisesRegex(ValueError, "conflicting official"):
            road.index_scene("clip")

    def test_persistence_reproducibility_and_complete_inventory(self):
        readers = native_readers(self.root, [0, 200000, 400000], [pose()]*3,
            {"person": {0: [10, 5, 1], 2: [10.8, 5, 1]}, "two-only": {}},
            camera={"person": {0, 1, 2}, "two-only": {0, 1, 2}})
        for reader in readers:
            first, second = self.root / (reader.dataset + "-a"), self.root / (reader.dataset + "-b")
            manifest, audit = prepare_collection(reader, first)
            manifest2, _ = prepare_collection(reader, second)
            self.assertEqual(manifest["status"], "complete"); self.assertEqual(len(audit["tracks"]), 2)
            self.assertEqual(manifest["tracks"], manifest2["tracks"])
            self.assertEqual(manifest["preparation_id"], manifest2["preparation_id"])
            reader.root.rename(reader.root.with_name("hidden-loki")) if reader.dataset == "loki" else reader.index.rename(reader.index.with_name("hidden-road"))
            for filename in manifest["tracks"]:
                m, a = load_track(first / "tracks" / filename)
                m2, a2 = load_track(second / "tracks" / filename)
                self.assertEqual(m, m2)
                for k in a: np.testing.assert_equal(a[k], a2[k])
                from pedestrian_behavior.inspection.prepared_tracks import write_inspector
                inspector = self.root / "standalone-inspector.html"
                data = write_inspector(inspector, [{"name": filename, "path": first/"tracks"/filename}])
                self.assertEqual(data[0]["metadata"], m)
                self.assertEqual(data[0]["arrays"]["ped_position"][0][0], float(a["ped_position"][0,0]) if a["ped_position_valid"][0] else None)
                self.assertIn('id="annotations"', inspector.read_text())
            with self.assertRaises(FileExistsError):
                prepare_collection(reader, first)


if __name__ == "__main__":
    unittest.main()
