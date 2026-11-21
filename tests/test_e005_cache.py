import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pedestrian_behavior.experiments.e005_cache import (
    E005Cache,
    E005FeatureDataset,
    E005Writer,
    atomic_json,
    collate_e005,
    rgb_crop_box,
    rgb_patch_cues,
)
from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import predict, prediction_metrics


def make_cache(root):
    root.mkdir()
    writer = E005Writer(root, 0)
    writer.begin_scene("scenario_001")
    box_crop = np.array([.25, .25, .75, .75], np.float32)
    writer.add_rgb("2", "a", np.ones((256, 768), np.float16), box_crop)
    stage3 = (np.ones((2, 72), np.float16), np.array([[0, 0, 0], [1, 0, 0]], np.float32))
    stage5 = (np.ones((2, 216), np.float16), np.array([[0, 0, 0], [1, 0, 0]], np.float32))
    targets = {
        "a": {"box": [0, 0, 0, 2, 2, 2, 0], "anchor": np.eye(4),
              "indices3": np.array([0]), "indices5": np.array([0])},
        "b": {"box": [1, 0, 0, 2, 2, 2, 0], "anchor": np.eye(4),
              "indices3": np.array([1]), "indices5": np.array([1])},
    }
    writer.add_points("2", stage3, stage5, targets, np.eye(4))
    empty3 = (np.zeros((0, 72), np.float16), np.zeros((0, 3), np.float32))
    empty5 = (np.zeros((0, 216), np.float16), np.zeros((0, 3), np.float32))
    writer.add_points("4", empty3, empty5, {"a": {"box": targets["a"]["box"],
        "anchor": np.eye(4), "indices3": np.zeros(0, np.int64), "indices5": np.zeros(0, np.int64)}}, np.eye(4))
    scene = writer.finish_scene("native-source")
    arrays, payload_bytes = writer.finish()
    writer.close()
    tracks = {"format": "e005-loki-spatial-v1", "population": {"test": True}, "tracks": [
        {"locator": "a", "group": "scenario_001", "archive": "a.npz", "split": "training",
         "slots": 3, "source_keys": ["1", "2", "3"], "times_us": [0, 200000, 400000], "sha256": "a" * 64},
        {"locator": "b", "group": "scenario_001", "archive": "b.npz", "split": "training",
         "slots": 1, "source_keys": ["2"], "times_us": [200000], "sha256": "b" * 64},
    ]}
    atomic_json(root / "tracks.json", tracks)
    atomic_json(root / "manifest.json", {"format": "e005-loki-spatial-v1", "status": "complete",
        "tracks_sha256": checksum(root / "tracks.json"), "arrays": arrays,
        "scenes": {"scenario_001": scene}, "payload_bytes": payload_bytes})


class Baseline:
    entries = [{"locator": "a", "group": "scenario_001"}, {"locator": "b", "group": "scenario_001"}]
    rows = {
        "a": (3, "a", "a.npz"),
        "b": (1, "b", "b.npz"),
    }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        n, locator, archive = self.rows[self.entries[index]["locator"]]
        return {"inputs": torch.zeros((n, 20)), "targets": torch.zeros(n, dtype=torch.long),
                "gt_valid": torch.ones(n, dtype=torch.bool), "locator": locator, "archive": archive}


class E005CacheTest(unittest.TestCase):
    def test_crop_transform_and_patch_cues_include_padding_coordinates(self):
        box = rgb_crop_box([.4, .2, .6, .6], [200, 100], [60, 0, 140, 80])
        np.testing.assert_allclose(box, [.25, .25, .75, .75])
        cues = rgb_patch_cues(box).reshape(16, 16, 4)
        np.testing.assert_allclose(cues[0, 0], [-.46875, -.46875, 2 * .46875**2, 0])
        np.testing.assert_allclose(cues[7:9, 7:9, 3], 1.)

    def test_roundtrip_missing_slots_shared_tokens_batch_and_checksums(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "cache"
            make_cache(root)
            cache = E005Cache(root, verify_checksums=True)
            self.assertEqual(cache.rgb_frame("a", 0).shape, (0, 772))
            rgb = cache.rgb_frame("a", 1)
            self.assertEqual(rgb.shape, (256, 772))
            np.testing.assert_array_equal(rgb[:, :768], np.ones((256, 768), np.float16))
            self.assertEqual(cache.rgb_frame("b", 0).shape, (0, 772))
            self.assertEqual(cache.lidar_frame("a", 0)[0].shape, (0, 77))
            level3, level5 = cache.lidar_frame("a", 1)
            self.assertEqual(level3.shape, (1, 77))
            self.assertEqual(level5.shape, (1, 221))
            np.testing.assert_array_equal(level3[0, 72:], [0, 0, 0, 0, 1])
            np.testing.assert_array_equal(cache.lidar_frame("b", 0)[0][:, :72], np.ones((1, 72), np.float16))

            dataset = E005FeatureDataset(Baseline(), cache, "g-rgb-utonia")
            batch = collate_e005([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["geometry"].shape), (2, 3, 20))
            self.assertEqual(tuple(batch["flags"].shape), (2, 3, 3))
            self.assertEqual(batch["lengths"].device.type, "cpu")
            self.assertEqual(batch["lengths"].tolist(), [3, 1])
            self.assertEqual(len(batch["rgb_frames"]), 6)
            self.assertEqual(tuple(batch["rgb_frames"][1].shape), (256, 772))
            self.assertEqual(tuple(batch["rgb_frames"][3].shape), (0, 772))
            self.assertTrue(all(isinstance(frame, torch.Tensor) for frame in batch["lidar_levels"][0]))
            self.assertEqual(tuple(batch["lidar_levels"][0][3].shape), (1, 77))
            self.assertEqual(tuple(batch["lidar_levels"][1][4].shape), (0, 221))
            self.assertTrue(batch["flags"][1, 0, 2].item())
            tracks = predict(torch.nn.Identity(), [batch], torch.device("cpu"),
                             lambda _, b, device: torch.zeros((*b["targets"].shape, 4), device=device))
            self.assertEqual(prediction_metrics(tracks, 4)["denominators"]["groups"], 1)
            cache.close()

            with (root / "arrays" / "stage3_features.bin").open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\x00")
            corrupt = E005Cache(root)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                corrupt.verify()
            corrupt.close()


if __name__ == "__main__":
    unittest.main()
