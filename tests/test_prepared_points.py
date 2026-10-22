"""Prepared inputs: read-only hits, complete populations/slots, costs and corruption."""

from collections import defaultdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pedestrian_behavior.data.feature_cache import FeatureCache
from pedestrian_behavior.data.indexed_points import PROTOCOL, ShardWriter
from pedestrian_behavior.data.prepared_points import PreparedPoints
from pedestrian_behavior.data.prepared_rgb import PreparedRGB
from pedestrian_behavior.experiments.e002a_indexed_cache import track_records
from pedestrian_behavior.inspection.pretrained import DITR_REVISION, DITR_SHA256, DITR_CONFIG_SHA256
from test_e002a_indexed_cache import Catalog
from test_indexed_points import fixture

ENCODER = {"revision": DITR_REVISION, "checkpoint": DITR_SHA256, "configuration": DITR_CONFIG_SHA256}


def prepared_fixture(root, catalog):
    root.mkdir(parents=True)
    groups = defaultdict(set)
    for t in catalog.tracks.values():
        groups[t["group"]]
    for group, key in catalog.frames:
        groups[group].add(key)
    scenes = {}
    for index, (group, keys) in enumerate(groups.items()):
        path = f"scenes/{index}"
        writer = ShardWriter(root/path, {"dataset": catalog.dataset, "group": group, "population": catalog.identity})
        for key in sorted(keys, key=int):
            members = sorted(set(catalog.frames[group, key]))
            a = fixture()
            a.update(locators=np.array(members), boxes=np.repeat(a["boxes"][:1], len(members), axis=0),
                     anchors=np.repeat(np.eye(4)[None], len(members), axis=0))
            writer.add(key, a)
        sha = writer.finish()
        writer.close()
        scenes[group] = {"path": path, "manifest_sha256": sha}
    record = {"format": "e002a-indexed-source-v1", "protocol": PROTOCOL, "status": "complete",
        "scope": "full-frozen-source-train-validation", "dataset": catalog.dataset,
        "population": catalog.identity, "encoder": ENCODER, "scenes": scenes,
        "tracks": track_records(catalog, groups)}
    (root/"manifest.json").write_text(json.dumps(record))
    return root


def open_points(root, catalog, **kwargs):
    return PreparedPoints(root, catalog.dataset, catalog.identity, catalog.tracks, ENCODER, **kwargs)


class PreparedInputsTest(unittest.TestCase):
    def test_scene_reader_budget_closes_and_reopens_without_losing_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = Catalog()
            catalog.tracks["b"]["group"] = "other"
            for key in ("1", "2"):
                catalog.frames["scene", key].remove("b")
                catalog.frames["other", key] = ["b"]
            root = prepared_fixture(Path(folder)/"points", catalog)
            points = open_points(root, catalog, max_open_scenes=1)
            for locator in ("a", "b", "a"):
                self.assertEqual(sum(len(f) for f in points.iter_track(locator)), points.track_cost(locator))
                self.assertEqual(points.snapshot()["open_scenes"], 1)
            self.assertEqual(points.snapshot()["scene_opens"], 3)
            self.assertEqual(points.snapshot()["requests"], 6)
            points.close()

    def test_packing_costs_use_manifests_and_reads_preserve_files(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = Catalog()
            root = prepared_fixture(Path(folder)/"points", catalog)
            before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
            with patch("pedestrian_behavior.data.prepared_points.IndexedShard", side_effect=AssertionError("payload opened")):
                points = open_points(root, catalog)
                self.assertGreater(points.track_cost("a"), 0)
                self.assertEqual(points.track_cost("c"), 0)
                points.close()
            points = open_points(root, catalog)
            crops = list(points.iter_track("a"))
            self.assertEqual(len(crops), 3)
            self.assertEqual(crops[-1].shape, (0, 68))
            self.assertEqual(crops[0].dtype, np.float32)
            self.assertEqual(sum(map(len, crops)), points.track_cost("a"))
            self.assertEqual(points.snapshot()["requests"], 2)
            points.close()
            self.assertEqual(before, {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in before})

    def test_incomplete_wrong_population_encoder_and_partial_tracks_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = Catalog()
            root = prepared_fixture(Path(folder)/"points", catalog)
            path = root/"manifest.json"
            original = json.loads(path.read_text())
            for field, value in (("status", "incomplete"), ("population", {}), ("encoder", {}), ("tracks", [])):
                path.write_text(json.dumps(original | {field: value}))
                with self.assertRaises(ValueError):
                    open_points(root, catalog)
            partial = original | {"scope": "bounded-native-check"}
            path.write_text(json.dumps(partial))
            with self.assertRaisesRegex(ValueError, "Full"):
                open_points(root, catalog)
            p = open_points(root, catalog, allow_partial=True)
            self.assertGreater(p.track_cost("a"), 0)
            p.close()
            # A partial manifest may retain a track, but cannot fabricate its missing usable slots.
            partial["tracks"][0]["point_frames"][0] = None
            path.write_text(json.dumps(partial))
            with self.assertRaises(ValueError):
                open_points(root, catalog, allow_partial=True)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                open_points(root, catalog)

    def test_partial_native_check_cannot_supply_an_uncached_complete_track(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = Catalog()
            catalog.tracks["a"]["source_keys"] = [1, 2, 3]
            catalog.tracks["a"]["times_us"] = [100, 200, 300]
            catalog.tracks["a"]["point_usable"][2] = True
            root = prepared_fixture(Path(folder)/"points", catalog)
            path = root/"manifest.json"
            record = json.loads(path.read_text())
            record["scope"] = "bounded-native-check"
            path.write_text(json.dumps(record))
            points = open_points(root, catalog, allow_partial=True)
            with self.assertRaisesRegex(ValueError, "complete selected track"):
                points.track_cost("a")
            points.close()

    def test_binary_mutation_and_source_manifest_mutation_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            catalog = Catalog()
            root = prepared_fixture(Path(folder)/"points", catalog)
            points = open_points(root, catalog)
            list(points.iter_track("a"))
            path = next(root.rglob("features.bin"))
            with path.open("r+b") as stream:
                stream.write(b"xx")
            with self.assertRaisesRegex(ValueError, "changed"):
                list(points.iter_track("a"))
            points.close()
            points = open_points(root, catalog)
            (root/"manifest.json").write_text((root/"manifest.json").read_text()+" ")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                list(points.iter_track("a"))
            points.close()

    def test_indexed_calibration_reads_new_points_without_legacy_point_fixtures(self):
        import torch
        from pedestrian_behavior.experiments.e002a_calibrate import fixture_samples
        class Dataset:
            entries = [{"locator": str(i)} for i in range(3)]
            def __getitem__(self, index):
                return {"locator": str(index), "inputs": torch.ones(3, 20), "targets": torch.zeros(3, dtype=torch.int64)}
        class Empty:
            entries = []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / "rgb.npz", cls=np.ones((3, 768), np.float32), valid=np.ones(3, bool))
            (root / "execution.json").write_text(json.dumps({"status": "complete", "tracks": [
                {"dataset": dataset, "name": dataset+str(i), "locator": str(i), "rgb_cache": "rgb.npz",
                 "point_frames": ["must-not-load"]*3}
                for dataset in ("loki", "road-waymo") for i in range(3)]}))
            sources = {d: Mock(iter_track=Mock(side_effect=lambda locator: iter([
                np.ones((2, 68), np.float32), np.zeros((0, 68), np.float32), np.ones((1, 68), np.float32)])))
                for d in ("loki", "road-waymo")}
            module = "pedestrian_behavior.experiments.e002a_calibrate."
            with patch(module + "e001b_run.assemble", return_value={"dataset": lambda d, split: Dataset() if split == "training" else Empty()}), \
                 patch(module + "loaded_track", side_effect=AssertionError("legacy point load")):
                samples = fixture_samples(root, sources)
            self.assertEqual(len(samples), 6)
            self.assertEqual(samples[0][1]["flags"][:, 2].tolist(), [1., 0., 1.])
            self.assertEqual([len(f) for f in samples[0][1]["points"]], [2, 0, 1])

    def test_rgb_hits_are_read_only_and_misses_never_call_producer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)/"rgb"
            with FeatureCache(root, min_free_bytes=0) as writer:
                writer.get({"track": "a"}, lambda: {"values": np.ones((2, 3), np.float32)})
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            cache = PreparedRGB(root)
            produce = Mock(side_effect=AssertionError("inference called"))
            np.testing.assert_array_equal(cache.get({"track": "a"}, produce)["values"], np.ones((2, 3)))
            with self.assertRaisesRegex(ValueError, "miss"):
                cache.get({"track": "absent"}, produce)
            produce.assert_not_called()
            cache.close()
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            path = next((root/"entries").glob("*.npz"))
            with path.open("r+b") as stream:
                stream.write(b"xx")
            cache = PreparedRGB(root)
            with self.assertRaisesRegex(ValueError, "checksum"):
                cache.get({"track": "a"}, produce)
            cache.close()
