"""Standalone builder acceptance: shared sweeps, source-only slots, readback and failures."""

from collections import defaultdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from pedestrian_behavior.data.indexed_points import IndexedShard, ShardWriter
from pedestrian_behavior.experiments.e002a_indexed_cache import (
    check_root, prepare_scene, run, track_records, verify_scene,
)
from test_indexed_points import fixture


class Catalog:
    def __init__(self, *args):
        self.dataset = "road-waymo"
        self.identity, self.protected, self.source_snapshots = {"frozen": True}, {}, {}
        self.tracks = {}
        self.frames = defaultdict(list)
        for name, center in (("a", .04), ("b", 1.04)):
            self.tracks[name] = {"locator": name, "group": "scene", "archive": name+".npz", "sha256": name,
                "track_id": name, "split": "training" if name == "a" else "validation", "slots": 3,
                "source_keys": [1, 2, None], "times_us": [100, 200, None], "point_usable": np.array([True, True, False])}
            for key in ("1", "2"):
                self.frames["scene", key].append(name)
        # A complete track without usable point slots must remain in the index.
        self.tracks["c"] = {**self.tracks["a"], "locator": "c", "point_usable": np.zeros(3, bool)}
        self.manifest = {"scenes": {"scene": {}}}
        self.reader = SimpleNamespace(read_frames=lambda group, keys: {k: {"ego_pose": np.eye(4)} for k in keys})

    def native_reader(self, group):
        return self.reader

    def source_identity(self, group):
        return "unchanged"


def frames(reader, group, keys, tracks, frozen, include_rgb):
    assert list(keys) == [1, 2]  # Native integer keys must not become manifest strings.
    assert not include_rgb
    data = fixture()
    for key in keys:
        yield {"key": key, "points": np.c_[data["coordinates"], np.ones(len(data["coordinates"]))].astype(np.float32),
            "boxes": {k: (b[:3], b[3:6], b[6]) for k, b in zip(("a", "b"), data["boxes"])},
            "world_from_native": np.eye(4)}


class Encoder:
    def __init__(self, *args):
        self.calls, self.loads, self.seconds = {"points": 0}, {"points": 1}, {"points": 0.}

    def __call__(self, kind, points):
        self.calls[kind] += 1
        return fixture()["features"], None, np.arange(len(points), dtype=np.int64), None

    def release(self):
        pass


class IndexedCacheBuilderTest(unittest.TestCase):
    def test_empty_native_sweeps_keep_references_without_encoder_calls(self):
        def empty(*args, **kwargs):
            for frame in frames(*args, **kwargs):
                frame["points"] = None
                yield frame
        with tempfile.TemporaryDirectory() as folder, patch(
                "pedestrian_behavior.experiments.e002a_indexed_cache.native_scene", empty):
            encoder = Encoder()
            result = prepare_scene(Catalog(), "scene", ["1", "2"], encoder,
                Path(folder)/"empty", 0, lambda *a: None)
            self.assertEqual(encoder.calls["points"], 0)
            self.assertEqual(result["verification"]["requests"], 4)
            self.assertEqual(result["verification"]["tokens"], 0)

    def test_shared_sweep_native_keys_missing_slots_and_full_file_readback(self):
        with tempfile.TemporaryDirectory() as folder, patch(
                "pedestrian_behavior.experiments.e002a_indexed_cache.native_scene", frames):
            catalog, encoder, progress = Catalog(), Encoder(), []
            root = Path(folder) / "scene"
            result = prepare_scene(catalog, "scene", ["1", "2"], encoder, root, 0,
                                   lambda key, meta: progress.append(key))
            self.assertEqual(encoder.calls["points"], 2)  # Not four pedestrian forwards.
            self.assertEqual(result["verification"]["requests"], 4)
            self.assertEqual(progress, ["1", "2"])
            records = track_records(catalog, {"scene": {"1", "2"}})
            self.assertEqual(records[0]["point_frames"], ["1", "2", None])
            self.assertEqual(records[2]["point_frames"], [None, None, None])
            reader = IndexedShard(root, result["manifest_sha256"])
            self.assertTrue(np.isfinite(reader.crop("1", "a")).all())
            reader.close()
            with (root / "features.bin").open("r+b") as stream:
                stream.write(b"xx")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_scene(root, result["manifest_sha256"], {"1": ["a", "b"], "2": ["a", "b"]})

    def test_floor_failure_does_not_publish_scene(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = ShardWriter(Path(folder) / "scene", {}, min_free_bytes=10**30)
            try:
                with self.assertRaisesRegex(OSError, "Insufficient"):
                    writer.add("1", fixture())
                self.assertFalse((writer.root / "manifest.json").exists())
                self.assertEqual((writer.root / "features.bin").stat().st_size, 0)
            finally:
                writer.close()

    def test_unsafe_roots_invalid_limits_and_existing_outputs_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            args = SimpleNamespace(outputs_root=base, output=base/"experiments/E002a/new",
                probe=base/"experiments/E002a/probe", min_free_gb=0, max_scenes=None, max_frames_per_scene=None)
            check_root(args, {})
            for output in (base/"experiments/E001", base/"experiments/E002a/rgb-cache/child", args.probe):
                with self.assertRaises(ValueError):
                    check_root(SimpleNamespace(**(vars(args) | {"output": output})), {})
            with self.assertRaises(ValueError):
                check_root(SimpleNamespace(**(vars(args) | {"max_scenes": 0})), {})
            args.output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run(args)

    def test_complete_run_and_failure_leave_distinct_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            probe = base/"experiments/E002a/probe"
            probe.mkdir(parents=True)
            for name in ("probe.json", "loki-transform.json", "weights", "config"):
                (probe/name).write_text("{}")
            config = {"ditr_weights": str(probe/"weights"), "ditr_config": str(probe/"config")}
            args = SimpleNamespace(outputs_root=base, output=base/"experiments/E002a/built", probe=probe,
                min_free_gb=0, max_scenes=None, max_frames_per_scene=None, device="cpu", dataset="road-waymo")
            module = "pedestrian_behavior.experiments.e002a_indexed_cache."
            with patch(module+"point_configuration", return_value=(config, probe/"loki-transform.json")), \
                 patch(module+"FeatureCatalog", Catalog), patch(module+"FrozenEncoders", Encoder), \
                 patch(module+"native_scene", frames):
                record = run(args)
                manifest = json.loads((args.output/"manifest.json").read_text())
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["scope"], "full-frozen-source-train-validation")
                self.assertEqual(len(manifest["tracks"]), 3)
                self.assertEqual(record["encoders"]["calls"]["points"], 2)
                args.output = base/"experiments/E002a/failed"
                with patch(module+"prepare_scene", side_effect=RuntimeError("inference failed")), \
                     self.assertRaisesRegex(RuntimeError, "inference failed"):
                    run(args)
                self.assertFalse((args.output/"manifest.json").exists())
                failed = json.loads((args.output/"cache.json").read_text())
                self.assertEqual(failed["status"], "failed")
                self.assertIn("inference failed", failed["failures"][0])
                args.output = base/"experiments/E002a/partial"
                args.max_scenes = 1
                run(args)
                self.assertEqual(json.loads((args.output/"manifest.json").read_text())["scope"], "bounded-native-check")

    def test_missing_box_missing_frame_and_changed_native_source_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            module = "pedestrian_behavior.experiments.e002a_indexed_cache."
            def missing(*args, **kwargs):
                for frame in frames(*args, **kwargs):
                    frame["boxes"].pop("a")
                    yield frame
            for i, factory in enumerate((missing, lambda *a, **kw: iter([]))):
                with patch(module+"native_scene", factory), self.assertRaises(ValueError):
                    prepare_scene(Catalog(), "scene", ["1", "2"], Encoder(), Path(folder)/str(i), 0, lambda *a: None)
            catalog = Catalog()
            with patch(module+"native_scene", frames), patch.object(
                    catalog, "source_identity", side_effect=["before", "after"]), \
                    self.assertRaisesRegex(ValueError, "changed"):
                prepare_scene(catalog, "scene", ["1", "2"], Encoder(), Path(folder)/"changed", 0, lambda *a: None)


if __name__ == "__main__":
    unittest.main()
