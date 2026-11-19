import tempfile
import unittest
from pathlib import Path

import numpy as np

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.experiments.e005_cache import FORMAT, E005Cache, E005Writer, atomic_json, rgb_tokens
from pedestrian_behavior.experiments.e005_road_cache import _capture_parity, _verify_parity, _write_clip
from pedestrian_behavior.experiments.e005_transfer import RoadSpatialFeatures


class RoadCacheTest(unittest.TestCase):
    def test_shared_sweep_union_and_missing_slots_match_online_features(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tracks = {
                "a": {"id": "a", "keys": [1, None], "point_usable": [True, False]},
                "b": {"id": "b", "keys": [1], "point_usable": [True]},
            }
            online = RoadSpatialFeatures(tracks, None, {}, None, None)
            world = np.eye(4)
            world[:3, 3] = (2., 3., 4.)
            online.anchors[1] = np.eye(4)
            coords = np.array([[0., 0., 0.], [15., 0., 80.], [25., 0., 0.], [50., 0., 0.]], np.float32)
            levels = tuple((np.arange(4 * width, dtype=np.float16).reshape(4, width), coords)
                           for width in (72, 216))
            online.sweeps[1] = (levels, world, {
                "a": (np.zeros(3), np.ones(3), 0.),
                "b": (np.array([25., 0., 0.]), np.ones(3), .1),
            })
            box = np.array([.25, .25, .75, .75], np.float32)
            online.rgb[1, "a"] = rgb_tokens(np.ones((256, 768), np.float16), box)

            writer = E005Writer(root, 0)
            try:
                writer.begin_scene("train_1")
                _write_clip(writer, online, list(tracks), {(1, "a"): box})
                samples = _capture_parity(writer, online, list(tracks))
                scene = writer.finish_scene("fixture")
                arrays, _ = writer.finish()
            finally:
                writer.close()
            self.assertEqual(scene["summary"]["stage3_tokens"], 3)
            self.assertEqual(scene["summary"]["stage3_selection_rows"], 4)

            tracks_path = root / "tracks.json"
            atomic_json(tracks_path, {"tracks": [{"locator": locator, "group": "train_1",
                "slots": len(track["keys"]), "source_keys": track["keys"]} for locator, track in tracks.items()]})
            atomic_json(root / "manifest.json", {"format": FORMAT,
                "status": "bounded-native-check", "tracks_sha256": checksum(tracks_path),
                "arrays": arrays, "scenes": {"train_1": scene}})
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                E005Cache(root)
            with E005Cache(root, allow_partial=True, verify_checksums=True) as cache:
                self.assertEqual(_verify_parity(cache, samples), {
                    "rgb": 1, "lidar": 1, "rgb_missing": 1, "lidar_missing": 1,
                })
                np.testing.assert_array_equal(cache.lidar_frame("b", 0)[0], online.lidar_frame("b", 0)[0])
