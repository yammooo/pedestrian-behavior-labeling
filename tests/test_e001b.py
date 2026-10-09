"""Independent bbox answers and native→extension→batch acceptance."""

import csv
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from pedestrian_behavior.data.bbox import bbox_inputs, box_geometry, load_bbox, normalization_sample
from pedestrian_behavior.data.features import fit_normalization
from pedestrian_behavior.data.loading import collate_tracks
from pedestrian_behavior.data.preparation import checksum, load_track, prepare_collection
from pedestrian_behavior.datasets.bbox import image_dimensions
from pedestrian_behavior.datasets.road_waymo import TrackReader as RoadReader
from pedestrian_behavior.experiments.e001 import dataset_from_setup, exclusion_reasons, setup, targets
from pedestrian_behavior.experiments.e001b import dataset_from_extension, prepare_bbox_extension
from test_track_preparation import native_readers, pose


class BboxTest(unittest.TestCase):
    def test_geometry_and_irregular_time_derivatives(self):
        # u=.2+time, bottom=.4+2*time, width=.2 and height=.2.
        times = [0, 150000, 400000]
        seconds = np.asarray(times)/1e6
        corners = np.column_stack((.1+seconds, .2+2*seconds, .3+seconds, .4+2*seconds))
        result = box_geometry(corners, times, [1, 1, 1])
        np.testing.assert_allclose(result["bbox_features"], np.column_stack((.2+seconds, .4+2*seconds,
            np.full(3, .2), np.full(3, .2), np.ones(3), np.full(3, 2.))), atol=1e-14)
        self.assertTrue(result["bbox_velocity_valid"].all())
        # A quadratic checks unequal-neighbor weighting, rather than an unweighted average.
        corners[:, [0, 2]] += seconds[:, None]**2
        result = box_geometry(corners, times, [1, 1, 1])
        self.assertAlmostEqual(result["bbox_features"][1, 4], 1.3)

    def test_gaps_singletons_source_switches_invalid_boxes(self):
        corners = [[.1, .2, .3, .4], [np.nan]*4, [.2, .3, .4, .5], [.3, .4, .5, .6], [.2, .3, .2, .5]]
        arrays = box_geometry(corners, [0, None, 400000, 600000, 800000], [1, 0, 2, 3, 3])
        self.assertEqual(arrays["bbox_valid"].tolist(), [True, False, True, True, False])
        self.assertFalse(arrays["bbox_velocity_valid"].any())
        self.assertTrue(np.isnan(arrays["bbox_features"][:, 4:]).all())
        with self.assertRaisesRegex(ValueError, "selected source"):
            box_geometry([[0, 0, 1, 1]], [None], [1])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            box_geometry([[0, 0, 1, 1]]*2, [1, 1], [1, 1])
        for size in ([0, 1], [1.5, 2], [np.nan, 2]):
            with self.assertRaisesRegex(ValueError, "dimensions"):
                image_dimensions(size)

    def test_masks_normalization_missing_and_source_statistics(self):
        arrays = box_geometry([[0, 0, .2, .4], [.2, .2, .4, .6], [np.nan]*4], [0, 200000, 400000], [1, 1, 0])
        statistics = fit_normalization([normalization_sample(arrays)])
        np.testing.assert_allclose(statistics["mean"], [.2, .5, .2, .4, 1, 1])
        self.assertEqual(statistics["count"], [2]*6)
        geometry = bbox_inputs(arrays, "geometry", statistics)
        availability = bbox_inputs(arrays, "availability", statistics)
        np.testing.assert_array_equal(geometry[:, -2:], availability)
        np.testing.assert_array_equal(geometry[2], np.zeros(8))
        self.assertEqual(geometry.dtype, np.float32)
        target = box_geometry([[10, 0, 10.2, .4]], [0], [1])
        target_inputs = bbox_inputs(target, "geometry", statistics)
        self.assertAlmostEqual(float(target_inputs[0, 0]), 99)
        self.assertEqual(statistics["count"], [2]*6)
        all_missing = box_geometry([[np.nan]*4], [0], [0])
        missing_statistics = fit_normalization([normalization_sample(all_missing)])
        np.testing.assert_array_equal(bbox_inputs(all_missing, "geometry", missing_statistics), np.zeros((1, 8)))

    def fixture(self, root, camera=None):
        people = {"person": {0: [10, 5, 1], 2: [12, 5, 1]}, "three-only": {0: [5, 1, 1]},
                  "two-only": {}}
        loki, road = native_readers(root, [0, 200000, 400000], [pose()]*3, people,
            camera=camera if camera is not None else {"person": {0, 1, 2}, "three-only": set(), "two-only": {1}},
            road_times={"person": {0, 2}})
        for i in range(3):
            Image.new("RGB", (200, 100)).save(loki.root/"scenario_000"/f"image_{i*2:04d}.png")
        camera_path = road.index/"camera_box.parquet"
        rows = pq.read_table(camera_path).to_pylist()
        for row in rows:
            row.update({"[CameraBoxComponent].box."+k: v for k, v in
                        zip(("center.x", "center.y", "size.x", "size.y"), (50., 30., 40., 20.))})
        pq.write_table(pa.Table.from_pylist(rows), camera_path)
        calibration = road.index/"camera_calibration.parquet"
        pq.write_table(pa.Table.from_pylist([{"key.segment_context_name": "scene", "key.camera_name": 1,
                    "[CameraCalibrationComponent].width": 200, "[CameraCalibrationComponent].height": 100}]), calibration)
        manifest_path = road.index/"scene_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest[0]["component_paths"]["camera_calibration"] = str(calibration)
        manifest_path.write_text(json.dumps(manifest))
        csv_path = road.index/"pedestrians.csv.gz"
        with gzip.open(csv_path, "rt", newline="") as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            row["road_box_normalized_json"] = json.dumps(json.loads(row["road_annotation_json"])["box"])
        with gzip.open(csv_path, "wt", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        return loki, RoadReader(road.index)

    def frozen_setup(self, reader, collection, setup_path):
        manifest, _ = prepare_collection(reader, collection)
        entries, samples = [], []
        from pedestrian_behavior.data.features import features
        for archive in manifest["tracks"]:
            metadata, arrays = load_track(collection/"tracks"/archive)
            y, gt = targets(metadata, reader.dataset)
            reasons = exclusion_reasons(metadata, arrays, gt)
            entry = {"archive": archive, "locator": metadata["track_locator"], "group": reader.scene_ids[0],
                     "exclusions": reasons, "split": None if reasons else "training", "slots": len(y),
                     "gt_frames": int(gt.sum()), "class_frames": [int((gt & (y == i)).sum()) for i in range(4)]}
            entries.append(entry)
            if not reasons:
                samples.append(features(arrays, "K+T+R"))
        setup_path.parent.mkdir()
        setup_path.write_text(json.dumps({"schema_version": 1, "experiment": "E001", "status": "complete",
            "dataset": reader.dataset, "collection": {"manifest_sha256": checksum(collection/"manifest.json")},
            "tracks": entries, "groups": {reader.scene_ids[0]: "training"}, "population": {"eligible": len(samples)},
            "normalization": {"K+T+R": fit_normalization(samples)}, "native_audit": {"verified_overrides": []}}))

    def test_native_extension_both_datasets_alignment_batches_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            readers = self.fixture(root)
            for reader in readers:
                with self.subTest(dataset=reader.dataset):
                    collection, setup_path, extension = root/(reader.dataset+"-collection"), root/(reader.dataset+"-setup")/"setup.json", root/(reader.dataset+"-bbox")
                    self.frozen_setup(reader, collection, setup_path)
                    native = reader.root if reader.dataset == "loki" else reader.index
                    record = prepare_bbox_extension(collection, setup_path, native, extension)
                    self.assertEqual(record["status"], "complete")
                    self.assertTrue(record["e001_unchanged"]["all_sha256_match"])
                    baseline_statistics = record["baseline_normalization"]
                    baseline = dataset_from_setup(collection, setup_path, "training", "K+T+R", baseline_statistics)
                    ds = [dataset_from_extension(collection, setup_path, extension, "training", config,
                        baseline_statistics, record["normalization"]) for config in ("availability", "geometry")]
                    for dataset, width in zip(ds, (14, 20)):
                        samples = [dataset[i] for i in range(len(dataset))]
                        batch = collate_tracks(samples)
                        self.assertEqual(batch["inputs"].shape[-1], width)
                        self.assertTrue(np.isfinite(batch["inputs"].numpy()).all())
                        for i, sample in enumerate(samples):
                            np.testing.assert_array_equal(sample["inputs"][:, :12], baseline[i]["inputs"])
                            np.testing.assert_array_equal(sample["targets"], baseline[i]["targets"])
                    for i in range(len(ds[0])):
                        np.testing.assert_array_equal(ds[0][i]["inputs"][:, -2:], ds[1][i]["inputs"][:, -2:])
                    person = next(e for e in record["tracks"] if json.loads(e["locator"]).get("track_id", json.loads(e["locator"]).get("tube_uid")) == "person")
                    metadata, _ = load_track(collection/"tracks"/person["archive"])
                    reference, arrays = load_bbox(extension/"tracks"/person["archive"], metadata, person["e001_sha256"])
                    self.assertEqual(arrays["bbox_valid"].tolist(), [True]*3)
                    # 2D survives the middle slot with no 3D or GT.
                    self.assertEqual(record["coverage"]["training"]["bbox-valid-without-3d"]["context_frames"], 1)
                    if reader.dataset == "road-waymo":
                        self.assertEqual(arrays["bbox_source"].tolist(), [3, 3, 3])
                        self.assertTrue(arrays["bbox_velocity_valid"].all())
                        np.testing.assert_allclose(arrays["bbox_features"][:, :4], [[.25, .4, .2, .2]]*3)
                        np.testing.assert_allclose(arrays["bbox_features"][:, 4:], 0)
                        self.assertEqual(arrays["road_2d"].tolist(), [True, False, True])
                    else:
                        np.testing.assert_allclose(arrays["bbox_features"][:, :4], [[.0125, .06, .015, .04]]*3)
                        self.assertEqual(record["coverage"]["training"]["track-never"]["tracks"], 1)
                    self.assertEqual(record["coverage"]["test"]["full"]["gt_frames"], 0)
                    with self.assertRaises(FileExistsError):
                        prepare_bbox_extension(collection, setup_path, native, extension)
                    wrong = dict(metadata, source_frames=list(reversed(metadata["source_frames"])))
                    with self.assertRaisesRegex(ValueError, "reference mismatch"):
                        load_bbox(extension/"tracks"/person["archive"], wrong, person["e001_sha256"])
                    arrays["bbox_features"][0, 0] = np.inf
                    np.savez_compressed(extension/"tracks"/person["archive"], metadata=np.array(json.dumps(reference)), **arrays)
                    with self.assertRaisesRegex(ValueError, "missingness"):
                        load_bbox(extension/"tracks"/person["archive"], metadata, person["e001_sha256"])

    def test_waymo_only_no_missing_or_invalid_fallback_and_old_policy_rejection(self):
        for mode in ("missing", "invalid"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                camera = {"person": {1, 2} if mode == "missing" else {0, 1, 2},
                          "three-only": set(), "two-only": {1}}
                _, reader = self.fixture(root, camera)
                if mode == "invalid":
                    path = reader.index/"camera_box.parquet"
                    rows = pq.read_table(path).to_pylist()
                    for row in rows:
                        if row["key.camera_object_id"] == "person" and row["key.frame_timestamp_micros"] == 0:
                            row["[CameraBoxComponent].box.size.x"] = 0.
                    pq.write_table(pa.Table.from_pylist(rows), path)
                    reader = RoadReader(reader.index)
                collection, setup_path, extension = root/"collection", root/"setup"/"setup.json", root/"bbox"
                self.frozen_setup(reader, collection, setup_path)
                record = prepare_bbox_extension(collection, setup_path, reader.index, extension)
                entry = next(e for e in record["tracks"] if json.loads(e["locator"])["tube_uid"] == "person")
                metadata, _ = load_track(collection/"tracks"/entry["archive"])
                _, arrays = load_bbox(extension/"tracks"/entry["archive"], metadata, entry["e001_sha256"])
                self.assertEqual(arrays["bbox_source"].tolist(), [0 if mode == "missing" else 3, 3, 3])
                self.assertEqual(arrays["bbox_valid"].tolist(), [False, True, True])
                self.assertEqual(arrays["bbox_velocity_valid"].tolist(), [False, True, True])
                self.assertTrue(arrays["road_2d"][0])
                self.assertTrue(np.isnan(arrays["bbox_features"][0]).all())
                baseline = dataset_from_setup(collection, setup_path, "training", "K+T+R", record["baseline_normalization"])
                ds = dataset_from_extension(collection, setup_path, extension, "training", "geometry",
                                            record["baseline_normalization"], record["normalization"])
                for i in range(len(ds)):
                    np.testing.assert_array_equal(ds[i]["targets"], baseline[i]["targets"])
                    np.testing.assert_array_equal(ds[i]["gt_valid"], baseline[i]["gt_valid"])
                sample = next(ds[i] for i in range(len(ds)) if ds.entries[i]["locator"] == entry["locator"])
                self.assertTrue(sample["gt_valid"][0])
                np.testing.assert_array_equal(sample["inputs"][0, 12:], np.zeros(8))
                record = json.loads((extension/"manifest.json").read_text())
                record["policy"]["road_source_rule"] = "ROAD when present; native FRONT camera box only when ROAD is absent"
                (extension/"manifest.json").write_text(json.dumps(record))
                with self.assertRaisesRegex(ValueError, "mismatched bbox extension"):
                    dataset_from_extension(collection, setup_path, extension, "training", "availability",
                                           record["baseline_normalization"], record["normalization"])

    def test_relocated_waymo_root_preserves_frozen_references_and_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, reader = self.fixture(root)
            collection, setup_path = root/"collection", root/"setup"/"setup.json"
            self.frozen_setup(reader, collection, setup_path)
            frozen_hash = checksum(collection/"manifest.json")
            waymo_root = root/"relocated-waymo"
            for component in ("camera_box", "camera_calibration"):
                target = waymo_root/"training"/component/"scene.parquet"
                target.parent.mkdir(parents=True)
                (reader.index/(component+".parquet")).rename(target)
            extension = root/"bbox"
            record = prepare_bbox_extension(collection, setup_path, reader.index, extension, waymo_root)
            self.assertEqual(record["waymo_root"], str(waymo_root.resolve()))
            self.assertEqual(record["scenes"]["clip"]["camera_box_path"],
                             str(waymo_root/"training/camera_box/scene.parquet"))
            self.assertEqual(checksum(collection/"manifest.json"), frozen_hash)
            ds = dataset_from_extension(collection, setup_path, extension, "training", "geometry",
                                        record["baseline_normalization"], record["normalization"])
            self.assertTrue(ds[0]["inputs"].isfinite().all())
            with self.assertRaisesRegex(ValueError, "separate"):
                prepare_bbox_extension(collection, setup_path, reader.index, waymo_root/"bbox", waymo_root)
            box_path = waymo_root/"training/camera_box/scene.parquet"
            rows = pq.read_table(box_path).to_pylist()
            rows[0]["[CameraBoxComponent].box.center.x"] += 1.
            pq.write_table(pa.Table.from_pylist(rows), box_path)
            with self.assertRaisesRegex(ValueError, "differ from frozen"):
                prepare_bbox_extension(collection, setup_path, reader.index, root/"failed-relocated", waymo_root)
            failure = json.loads((root/"failed-relocated/manifest.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertTrue(failure["e001_unchanged"]["all_sha256_match"])

    def test_real_setup_training_only_statistics_and_native_failure_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reader, _ = self.fixture(root)
            for i in (1, 2):
                shutil.copytree(reader.root/"scenario_000", reader.root/f"scenario_{i:03d}")
            from pedestrian_behavior.data.splits import split_groups
            for group, split in split_groups([f"scenario_{i:03d}" for i in range(3)]).items():
                if split != "training":
                    for path in (reader.root/group).glob("label2d_*.json"):
                        data = json.loads(path.read_text())
                        for person in data["Pedestrian"].values():
                            person["box"]["left"] += 1000
                        path.write_text(json.dumps(data))
            from pedestrian_behavior.datasets.loki import TrackReader
            reader = TrackReader(reader.root, root/"fixture-transform.json")
            collection, setup_path = root/"collection", root/"setup"/"setup.json"
            prepare_collection(reader, collection)
            setup(collection, reader.root, setup_path.parent, None)
            record = prepare_bbox_extension(collection, setup_path, reader.root, root/"bbox")
            self.assertEqual(record["normalization"]["count"], [3]*6)
            self.assertAlmostEqual(record["normalization"]["mean"][0], .0125)
            self.assertEqual(sum(v["full"]["tracks"] for v in record["coverage"].values()), 6)
            # Native changes must fail with retained local evidence, without rewriting E001.
            label = reader.root/"scenario_000"/"label2d_0000.json"
            label.write_text(label.read_text()+" ")
            with self.assertRaisesRegex(ValueError, "differs from frozen"):
                prepare_bbox_extension(collection, setup_path, reader.root, root/"failed")
            failed = json.loads((root/"failed"/"manifest.json").read_text())
            self.assertNotEqual(failed["status"], "complete")
            self.assertTrue(failed["failures"])
            self.assertTrue(failed["e001_unchanged"]["all_sha256_match"])


if __name__ == "__main__":
    unittest.main()
