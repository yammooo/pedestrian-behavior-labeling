"""Independent expected targets/features/splits/statistics/batches for E001."""

from copy import deepcopy
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from pedestrian_behavior.data.features import features, fit_normalization, normalize
from pedestrian_behavior.data.loading import collate_tracks, track_batches
from pedestrian_behavior.data.preparation import checksum, load_track, prepare_collection, save_track
from pedestrian_behavior.data.splits import split_groups
from pedestrian_behavior.datasets.loki import TrackReader
from pedestrian_behavior.experiments.e001 import (audit_native, dataset_from_setup, exclusion_reasons,
                                                 mapped_target, override_for, setup, targets)
from test_track_preparation import native_readers, pose


class E001DataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = []

    def tearDown(self):
        if output := os.environ.get("E001_ACCEPTANCE_OUTPUT"):
            Path(output).mkdir(parents=True, exist_ok=True)
            (Path(output) / (self._testMethodName + ".json")).write_text(json.dumps(self.evidence, indent=2) + "\n")

    def check(self, name, expected, actual):
        if isinstance(actual, (np.ndarray, torch.Tensor)):
            actual = actual.tolist()
        self.evidence.append({"check": name, "expected": expected, "actual": actual})
        self.assertEqual(expected, actual, name)

    def rejects(self, name, message, function, *args):
        with self.assertRaisesRegex(ValueError, message) as failure:
            function(*args)
        self.evidence.append({"check": name, "expected": "ValueError containing " + message,
                              "actual": str(failure.exception)})

    def arrays(self):
        nan = float("nan")
        return {"ped_position": np.array([[nan, nan], [10., 4.], [14., 6.]]),
                "ped_velocity": np.array([[nan, nan], [2., 1.], [4., 3.]]),
                "ego_position": np.array([[0., 0.], [nan, nan], [2., 1.]]),
                "ego_velocity": np.array([[nan, nan], [nan, nan], [1., 2.]]),
                "ped_position_valid": np.array([False, True, True]), "ped_velocity_valid": np.array([False, True, True]),
                "ego_pose_valid": np.array([True, False, True]), "ego_velocity_valid": np.array([False, False, True])}

    def test_targets_and_native_preservation(self):
        actions = [["MovTow", "MovAway", "Mov"], ["Mov", "Xing"], ["PushObj"], ["Stop", "PushObj"], ["xing"], []]
        self.check("ROAD projection (location xing supplies no action)", [0, 3, -100, 1, -100, -100],
                   [mapped_target(a, "road-waymo") for a in actions])
        self.check("LOKI current-frame labels", [0, 1, 2, 3],
                   [mapped_target([a], "loki") for a in ["Moving", "Stopped", "Waiting to cross", "Crossing the road"]])
        self.rejects("unresolved conflict", "Unresolved", mapped_target, ["Mov", "Stop"], "road-waymo")
        self.rejects("malformed native actions", "list of strings", mapped_target, "Mov", "road-waymo")
        metadata = {"track_locator": '{"scenario":"s","track_id":"p"}', "source_frames": [0, 200000, 400000],
                    "native_annotations": [{"label3d": {"intended_actions": "Stopped", "stationary": "true"}}, {},
                                           {"label3d": {"intended_actions": "Moving"}}], "anchor_valid": True}
        original = deepcopy(metadata)
        y, gt = targets(metadata, "loki")
        self.check("missing native supervision stays absent", [1, -100, 0], y)
        self.check("native annotations unchanged", original, metadata)
        arrays = self.arrays()
        arrays["ped_position_valid"] = np.array([False, False, True])
        self.check("GT and position may occupy different slots", [], exclusion_reasons(metadata, arrays, np.array([True, False, False])))
        metadata["anchor_valid"] = False
        arrays["ped_position_valid"][:] = False
        self.check("overlapping exclusions", ["no-usable-position", "no-accepted-gt", "invalid-anchor"],
                   exclusion_reasons(metadata, arrays, np.zeros(3, dtype=bool)))

    def road_audit_fixture(self):
        index = self.root / "index"; index.mkdir()
        override = {"clip": "clip", "tube_uid": "tube", "road_frames_inclusive": [1, 2],
                    "timestamps_inclusive_us": [100, 200], "expected_native_actions": ["MovTow", "Stop"],
                    "target": "MOVING", "observation_count": 2}
        policy = {"schema_version": 1, "experiment": "E001", "dataset": "road-waymo", "overrides": [override]}
        rows = [{"road_clip_id": "clip", "road_tube_uid": "tube", "road_frame_1based": i,
                 "frame_timestamp_micros": i*100, "merged_agent_label": "Ped", "camera_name": "1",
                 "action_labels_json": '["MovTow", "Stop"]', "road_annotation_id": str(i)} for i in (1, 2)]
        path = index / "pedestrians.csv.gz"
        with gzip.open(path, "wt", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        policy["native_csv_sha256"] = checksum(path)
        override_path = self.root / "overrides.json"
        override_path.write_text(json.dumps(policy))
        manifest = {"dataset": "road-waymo", "sources": {"sha256": {path.name: checksum(path)}}, "scenes": {"clip": {}}}
        return manifest, index, override_path, policy

    def test_override_audit_and_exact_application(self):
        manifest, index, path, policy = self.road_audit_fixture()
        audit = audit_native(manifest, index, path)
        self.check("complete native override count", 2, len(audit["verified_overrides"]))
        metadata = {"track_locator": '{"clip":"clip","tube_uid":"tube"}', "source_frames": [100, 200, 300],
                    "native_annotations": [{"road": {"actions": ["MovTow", "Stop"], "annotation_ids": [str(i)],
                                                      "annotation": {"tube_uid": "tube"}}} for i in (1, 2)] + [{}]}
        original = deepcopy(metadata)
        self.check("only verified native observations corrected", [0, 0, -100], targets(metadata, "road-waymo", audit["verified_overrides"])[0])
        self.check("override keeps native actions", original, metadata)
        stopped = deepcopy(audit["verified_overrides"])
        for observation in stopped:
            observation["target"] = 1
        self.check("accepted STOPPED correction uses its declared target", [1, 1, -100], targets(metadata, "road-waymo", stopped)[0])
        wrong = deepcopy(metadata); wrong["native_annotations"][0]["road"]["annotation_ids"] = ["other"]
        self.rejects("saved source mismatch", "source/scope", targets, wrong, "road-waymo", audit["verified_overrides"])
        wrong = deepcopy(metadata); wrong["track_locator"] = '{"clip":"clip","tube_uid":"other"}'
        self.rejects("different identity is not corrected", "Unresolved", targets, wrong, "road-waymo", audit["verified_overrides"])
        observation = {"clip": "clip", "tube_uid": "tube", "frame": 1, "timestamp": 300, "actions": ["MovTow", "Stop"]}
        self.rejects("frame/time disagreement", "scope mismatch", override_for, observation, policy["overrides"])
        observation["timestamp"] = 100; observation["actions"] = ["Mov"]
        self.rejects("native action disagreement", "actions mismatch", override_for, observation, policy["overrides"])
        policy["overrides"][0]["tube_uid"] = "other"; path.write_text(json.dumps(policy))
        self.rejects("unaccepted conflict cannot be hidden by resampling", "Unresolved", audit_native, manifest, index, path)
        policy["overrides"][0]["tube_uid"] = "tube"
        policy["overrides"][0]["observation_count"] = 3; path.write_text(json.dumps(policy))
        self.rejects("expected count mismatch", "bounds/count", audit_native, manifest, index, path)
        policy["overrides"][0]["observation_count"] = 2
        policy["overrides"][0]["road_frames_inclusive"] = [1, 3]
        policy["overrides"][0]["observation_count"] = 3
        path.write_text(json.dumps(policy))
        self.rejects("native observations must cover complete accepted scope", "identity/bounds/count", audit_native, manifest, index, path)
        policy["native_csv_sha256"] = "wrong"; path.write_text(json.dumps(policy))
        self.rejects("accepted source mismatch", "checksum mismatch", audit_native, manifest, index, path)

    def test_all_features_delayed_reference_and_missingness(self):
        expected = {
            "K": [[0, 0, 0], [2, 1, 1], [4, 3, 1]],
            "K+T": [[0]*6, [2, 1, 0, 0, 1, 1], [4, 3, 4, 2, 1, 1]],
            "K+T+R": [[0]*12, [2, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], [4, 3, 4, 2, 12, 5, 3, 1, 1, 1, 1, 1]],
            "RAW": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [10, 4, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0], [14, 6, 4, 3, 2, 1, 1, 2, 1, 1, 1, 1]],
        }
        for configuration, answer in expected.items():
            values, flags = features(self.arrays(), configuration)
            stats = {"mean": [0]*values.shape[1], "scale": [1]*values.shape[1]}
            self.check(configuration + " numeric columns then flags", answer, normalize(values, flags, stats))

    def test_split_and_normalization_guards(self):
        expected = {"7": "validation", "8": "validation", "1": "test", "5": "test", **{str(i): "training" for i in (3, 4, 2, 0, 9, 6)}}
        actual = split_groups([str(i) for i in range(10)]*2)
        self.check("fixed seed 0 clip assignment", expected, actual)
        self.check("input order/repetition independent", expected, split_groups(reversed([str(i) for i in range(10)])))
        values = np.array([[1., 7., 2., 5., np.nan, np.nan], [3., 7., 2.+1e-10, 5., np.nan, np.nan]])
        flags = np.array([[True, True, False]]*2)
        stats = fit_normalization([(values[:1], flags[:1]), (values[1:], flags[1:])])
        self.check("constant/near-constant/missing guarded columns", [1, 2, 3, 4, 5], [g["column"] for g in stats["guarded_columns"]])
        self.check("population scale and valid counts", [1., 1., 1., 1., 1., 1.], stats["scale"])
        self.check("unlabeled input slots participate", [2, 2, 2, 2, 0, 0], stats["count"])
        self.check("means for constant and missing inputs", [2., 7., 2.+.5e-10, 5., 0., 0.], stats["mean"])
        self.check("nonzero mean does not fill missing measurements", [[0., 0., 0.]],
                   normalize(np.array([[np.nan, np.nan]]), np.array([[False]]), {"mean": [4., 6.], "scale": [2., 3.]}))
        self.assertTrue(np.isfinite(normalize(values, flags, stats)).all())
        self.rejects("invalid statistics", "normalization", normalize, values, flags, {"mean": [0]*6, "scale": [0]*6})

    def sample(self, inputs, targets_, name):
        y = torch.tensor(targets_, dtype=torch.int64)
        return {"inputs": torch.tensor(inputs, dtype=torch.float32), "targets": y, "gt_valid": y != -100,
                "locator": name, "archive": name + ".npz"}

    def test_padding_internal_slots_dtypes_references_and_batches(self):
        a = self.sample([[2, 1, 1], [0, 0, 0], [4, 3, 1]], [0, -100, 3], "long")
        b = self.sample([[0, 0, 0]], [1], "short")
        batch = collate_tracks([a, b])
        self.check("lengths", [3, 1], batch["lengths"])
        self.check("only appended slots are padding", [[False, False, False], [False, True, True]], batch["padding_mask"])
        self.check("padding targets", [[0, -100, 3], [1, -100, -100]], batch["targets"])
        self.check("GT excludes internal gap and padding", [[True, False, True], [True, False, False]], batch["gt_valid"])
        self.check("padded inputs", [a["inputs"].tolist(), [[0, 0, 0]]*3], batch["inputs"])
        self.check("dtypes on CPU", ["torch.float32", "torch.int64", "torch.bool", "torch.int64", "torch.bool"],
                   [str(batch[k].dtype) for k in ("inputs", "targets", "gt_valid", "lengths", "padding_mask")])
        self.check("references", [["long", "short"], ["long.npz", "short.npz"]], [batch["locators"], batch["archives"]])
        self.assertTrue(all(t.device.type == "cpu" for t in batch.values() if isinstance(t, torch.Tensor)))
        samples = [self.sample([[0, 0, 0]], [0], str(i)) for i in range(65)]
        first, restart = track_batches(samples, training=True), track_batches(samples, training=True)
        epoch1 = list(first); epoch2 = list(first)
        self.check("whole-track batch 64 with incomplete final batch", [64, 1], [len(b["lengths"]) for b in epoch1])
        self.check("shuffle reproducible at restart", [b["locators"] for b in epoch1], [b["locators"] for b in restart])
        self.assertNotEqual(epoch1[0]["locators"], epoch2[0]["locators"])
        self.check("every eligible whole track once", sorted(str(i) for i in range(65)), sorted(n for b in epoch1 for n in b["locators"]))

    def test_setup_runtime_without_native_and_heldout_statistics(self):
        reader, _ = native_readers(self.root, [0, 200000, 400000], [pose()]*3,
                                   {"person": {0: [10, 5, 1], 2: [12, 5, 1]}, "two-only": {}},
                                   camera={"person": set(), "two-only": {0, 1, 2}})
        for i in range(1, 10):
            shutil.copytree(reader.root / "scenario_000", reader.root / f"scenario_{i:03}")
        reader = TrackReader(reader.root, self.root / "fixture-transform.json")
        collection = self.root / "collection"
        prepare_collection(reader, collection)
        output = self.root / "setup-a"
        record = setup(collection, reader.root, output, None)
        self.check("eligible 3D-only tracks and explicit 2D-only exclusions", [20, 10, 10, 10],
                   [record["population"][k] for k in ("candidates", "eligible", "excluded", "eligible_3d_only")])
        self.check("group/track split support", [6, 2, 2], [record["support"][s]["tracks"] for s in ("training", "validation", "test")])
        for t in record["tracks"]:
            if t["split"] in ("validation", "test"):
                path = collection / "tracks" / t["archive"]
                metadata, arrays = load_track(path)
                for field in ("ped_position", "ped_velocity", "ego_position", "ego_velocity"):
                    arrays[field] *= 1000
                save_track(path, metadata, arrays)
        other = setup(collection, reader.root, self.root / "setup-b", None)
        self.check("held-out values cannot affect source-training normalization", record["normalization"], other["normalization"])
        reader.root.rename(self.root / "hidden-native")
        dataset = dataset_from_setup(collection, output / "setup.json", "training", "RAW", record["normalization"]["RAW"])
        sample = dataset[0]
        self.check("on-the-fly archived sample preserves internal slot", [0, -100, 0], sample["targets"])
        self.check("runtime input dimensions", [3, 12], list(sample["inputs"].shape))
        self.assertEqual(sample["archive"], str(collection / "tracks" / Path(sample["archive"]).name))
        with self.assertRaises(FileExistsError):
            setup(collection, reader.root, output, None)

    def test_singleton_eligibility(self):
        reader, _ = native_readers(self.root, [0], [pose()], {"person": {0: [10, 5, 1]}},
                                   camera={"person": set()}, road_times={"person": {0}})
        collection = self.root / "single"
        manifest, _ = prepare_collection(reader, collection)
        metadata, arrays = load_track(collection / "tracks" / manifest["tracks"][0])
        y, gt = targets(metadata, "loki")
        self.check("singleton 3D-only survives with no usable velocity", [[], [0], [False]],
                   [exclusion_reasons(metadata, arrays, gt), y.tolist(), arrays["ped_velocity_valid"].tolist()])


if __name__ == "__main__":
    unittest.main()
