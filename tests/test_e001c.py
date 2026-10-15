"""Acceptance: fresh boxes, frozen population/GT, masks and source statistics."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pedestrian_behavior.data.boxes import CONFIGURATIONS, FLAGS, INPUT_COLUMNS, box_features, box_inputs, column_validity, load_boxes
from pedestrian_behavior.data.features import fit_normalization
from pedestrian_behavior.data.preparation import checksum, load_track
from pedestrian_behavior.datasets.road_waymo import TrackReader as RoadReader
from pedestrian_behavior.experiments.e001c import prepare_boxes
import test_e001b
from test_track_preparation import pose


class E001cTest(unittest.TestCase):
    def test_centers_dimensions_velocities_irregular_times_and_coordinates(self):
        times = [0, 130000, 400000]
        seconds = np.array(times)/1e6
        anchor = pose(10, 20, np.pi/2, z=5)
        centers = np.column_stack((10+seconds**2, 20+2*seconds, 5+3*seconds))
        sizes = np.column_stack((1+seconds, 2+2*seconds, 3+3*seconds))
        corners = np.column_stack((seconds, seconds, 2+3*seconds, 4+5*seconds))
        result = box_features(centers, sizes, corners, times, anchor)
        np.testing.assert_allclose(result["values"][:, :3], np.column_stack((2*seconds, -seconds**2, 3*seconds)), atol=1e-14)
        # Unequal-neighbor centered derivative of t^2 is exactly 2t.
        np.testing.assert_allclose(result["values"][1, 6:9], [2, -.26, 3], atol=1e-13)
        np.testing.assert_allclose(result["values"][:, 17:], [[1, 2, 3]]*3, atol=1e-13)
        np.testing.assert_allclose(result["values"][0, 9:13], [1, 2, 2, 4])
        np.testing.assert_allclose(result["values"][:, 13:17], [[2, 3, 2, 4]]*3)
        for flag in FLAGS:
            self.assertTrue(result[flag].all())

    def test_gaps_singletons_and_validity_cannot_hide_usable_components(self):
        centers = np.array([[0, 0, 0], [np.nan]*3, [2, 0, 0], [3, 0, 0]])
        sizes = np.array([[1, 1, 2], [np.nan]*3, [1, 1, 2], [1, 1, 2]])
        corners = np.array([[0, 0, 1, 2], [np.nan]*4, [0, 0, 0, 2], [.2, 0, 1.2, 2]])
        result = box_features(centers, sizes, corners, [0, None, 400000, 600000], pose())
        self.assertEqual(result["box3d_velocity_valid"].tolist(), [False, False, True, True])
        self.assertEqual(result["box2d_valid"].tolist(), [True, False, False, True])
        self.assertFalse(result["box2d_velocity_valid"].any())
        sizes[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "validity differs"):
            box_features(centers, sizes, corners, [0, None, 400000, 600000], pose())
        sizes[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "validity differs"):
            box_features(centers, sizes, corners, [0, None, 400000, 600000], None)
        with self.assertRaisesRegex(ValueError, "source time"):
            box_features(centers[:1], sizes[:1], corners[:1], [None], pose())

    def test_source_statistics_missing_zero_and_identical_flags(self):
        arrays = box_features([[0, 0, 0], [2, 2, 2], [np.nan]*3], [[1, 1, 2], [1, 1, 2], [np.nan]*3],
                              [[0, 0, .2, .4], [.2, .2, .4, .6], [np.nan]*4], [0, 200000, 400000], pose())
        statistics = fit_normalization([(arrays["values"], column_validity(arrays))])
        self.assertEqual(statistics["count"], [2]*20)
        for config, width in CONFIGURATIONS.items():
            result = box_inputs(arrays, config, statistics)
            self.assertEqual(result.shape, (3, width+4))
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_array_equal(result[2], np.zeros(width+4))
            np.testing.assert_array_equal(result[:, -4:], np.column_stack([arrays[k] for k in FLAGS]))
        shifted = {key: value.copy() for key, value in arrays.items()}
        shifted["values"][:2, 0] += 10
        np.testing.assert_allclose(box_inputs(shifted, "box3d", statistics)[:2, 0], [9, 11])
        self.assertEqual(statistics["mean"][0], 1)

    def test_final_column_selection_is_15_17_20_plus_same_four_flags(self):
        arrays = box_features([[0, 0, 0]]*2, [[1, 1, 2]]*2, [[0, 0, 1, 2]]*2, [0, 200000], pose())
        arrays["values"][:] = np.arange(1, 21)
        statistics = {"mean": [0]*20, "scale": [1]*20}
        for configuration, count in INPUT_COLUMNS.items():
            actual = box_inputs(arrays, configuration, statistics)
            np.testing.assert_array_equal(actual[:, :count], [np.arange(1, count+1)]*2)
            np.testing.assert_array_equal(actual[:, count:], np.ones((2, 4)))
            self.assertEqual(actual.shape[1], count+4)

    def fixture(self, root, camera=None):
        helper = test_e001b.BboxTest()
        loki, road = helper.fixture(root, camera)
        path = road.index/"lidar_box.parquet"
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            row.update({"[LiDARBoxComponent].box.size."+axis: size for axis, size in zip("xyz", (1., 1., 2.))})
        pq.write_table(pa.Table.from_pylist(rows), path)
        return helper, (loki, RoadReader(road.index))

    def test_native_rebuild_both_datasets_frozen_targets_checksum_and_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper, readers = self.fixture(root)
            for reader in readers:
                with self.subTest(dataset=reader.dataset):
                    collection, setup, output = root/(reader.dataset+"-collection"), root/(reader.dataset+"-setup")/"setup.json", root/(reader.dataset+"-boxes")
                    helper.frozen_setup(reader, collection, setup)
                    native = reader.root if reader.dataset == "loki" else reader.index
                    transform = root/"fixture-transform.json" if reader.dataset == "loki" else None
                    record = prepare_boxes(collection, setup, native, output, loki_transform=transform)
                    self.assertEqual(record["status"], "complete")
                    self.assertTrue(record["e001_unchanged"]["all_sha256_match"])
                    self.assertTrue(record["mask_audit"]["four_flags_supported"])
                    frozen = json.loads(setup.read_text())
                    self.assertEqual([e["locator"] for e in record["tracks"]], [e["locator"] for e in frozen["tracks"] if not e["exclusions"]])
                    entry = next(e for e in record["tracks"] if json.loads(e["locator"]).get("track_id", json.loads(e["locator"]).get("tube_uid")) == "person")
                    metadata, _ = load_track(collection/"tracks"/entry["archive"])
                    reference, arrays = load_boxes(output/"tracks"/entry["archive"], metadata, entry["e001_sha256"])
                    self.assertEqual(arrays["box3d_valid"].tolist(), [True, False, True])
                    self.assertFalse(arrays["box3d_velocity_valid"].any())
                    self.assertTrue(arrays["box2d_valid"].all())
                    self.assertFalse(arrays["gt_valid"][1])
                    expected = [.0125, .04, .015, .04] if reader.dataset == "loki" else [.25, .3, .2, .2]
                    np.testing.assert_allclose(arrays["values"][:, 9:13], [expected]*3)
                    np.testing.assert_allclose(arrays["values"][[0, 2], 2], [1, 1])
                    with self.assertRaises(FileExistsError):
                        prepare_boxes(collection, setup, native, output, loki_transform=transform)
                    wrong = metadata | {"source_frames": metadata["source_frames"][::-1]}
                    with self.assertRaisesRegex(ValueError, "reference mismatch"):
                        load_boxes(output/"tracks"/entry["archive"], wrong, entry["e001_sha256"])
                    arrays["values"][0, 0] = np.inf
                    np.savez_compressed(output/"tracks"/entry["archive"], metadata=np.array(json.dumps(reference)), **arrays)
                    with self.assertRaisesRegex(ValueError, "missingness"):
                        load_boxes(output/"tracks"/entry["archive"], metadata, entry["e001_sha256"])

    def test_native_failure_retains_evidence_and_protects_frozen_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper, (_, reader) = self.fixture(root)
            collection, setup, output = root/"collection", root/"setup"/"setup.json", root/"failed"
            helper.frozen_setup(reader, collection, setup)
            frozen = checksum(collection/"manifest.json")
            path = reader.index/"lidar_box.parquet"
            path.write_bytes(path.read_bytes()+b"changed")
            with self.assertRaisesRegex(ValueError, "differs from frozen"):
                prepare_boxes(collection, setup, reader.index, output)
            failure = json.loads((output/"manifest.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertTrue(failure["e001_unchanged"]["all_sha256_match"])
            self.assertEqual(checksum(collection/"manifest.json"), frozen)

    def test_road_geometry_never_falls_back_to_road_and_mount_can_move(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            helper, (_, reader) = self.fixture(root, {"person": {1, 2}, "three-only": set(), "two-only": {1}})
            collection, setup = root/"collection", root/"setup"/"setup.json"
            helper.frozen_setup(reader, collection, setup)
            relocated = root/"waymo"
            for component in ("camera_box", "camera_calibration", "vehicle_pose", "lidar_box", "camera_to_lidar_box_association"):
                target = relocated/"training"/component/"scene.parquet"
                target.parent.mkdir(parents=True)
                (reader.index/(component+".parquet")).rename(target)
            record = prepare_boxes(collection, setup, reader.index, root/"boxes", waymo_root=relocated)
            entry = record["tracks"][0]
            metadata, _ = load_track(collection/"tracks"/entry["archive"])
            _, arrays = load_boxes(root/"boxes"/"tracks"/entry["archive"], metadata, entry["e001_sha256"])
            self.assertTrue(arrays["road_2d"][0])
            self.assertTrue(arrays["gt_valid"][0])
            self.assertFalse(arrays["box2d_valid"][0])
            self.assertTrue(np.isnan(arrays["values"][0, 9:17]).all())
            self.assertIn(str(relocated), record["scenes"]["clip"]["lidar_box_path"])

    def test_real_split_normalization_excludes_validation_and_test(self):
        from pedestrian_behavior.datasets.loki import TrackReader
        from pedestrian_behavior.data.preparation import prepare_collection
        from pedestrian_behavior.data.splits import split_groups
        from pedestrian_behavior.experiments.e001 import setup as make_setup
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, (reader, _) = self.fixture(root)
            for number in (1, 2):
                shutil.copytree(reader.root/"scenario_000", reader.root/f"scenario_{number:03d}")
            for scene, split in split_groups([f"scenario_{number:03d}" for number in range(3)]).items():
                if split != "training":
                    for path in (reader.root/scene).glob("label2d_*.json"):
                        data = json.loads(path.read_text())
                        for person in data["Pedestrian"].values():
                            person["box"]["left"] += 1000
                        path.write_text(json.dumps(data))
            reader = TrackReader(reader.root, root/"fixture-transform.json")
            collection, setup = root/"collection", root/"setup"/"setup.json"
            prepare_collection(reader, collection)
            make_setup(collection, reader.root, setup.parent, None)
            result = prepare_boxes(collection, setup, reader.root, root/"boxes", loki_transform=root/"fixture-transform.json")
            self.assertAlmostEqual(result["normalization"]["mean"][9], .0125)
            self.assertEqual(result["normalization"]["count"][9], 3)
            self.assertEqual(sum(row["full"]["tracks"] for row in result["coverage"].values()), 6)


if __name__ == "__main__":
    unittest.main()
