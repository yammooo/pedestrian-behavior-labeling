import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pedestrian_behavior.experiments.e003b_embedding_diagnostic import _read_cache, _sample_validation


class EmbeddingSamplingTest(unittest.TestCase):
    def test_even_track_cap_and_equal_dataset_sample_count(self):
        def track(length, valid_indices, archive):
            valid = np.zeros(length, dtype=bool)
            valid[valid_indices] = True
            return {"gt_valid": valid, "targets": np.arange(length, dtype=np.int64) % 4,
                "archive": archive}

        tracks = {
            "road-waymo": {"a": track(20, np.arange(20), "a.npz"),
                "b": track(8, np.array([2, 6]), "b.npz")},
            "loki": {"c": track(12, np.arange(12), "c.npz")}}
        first = _sample_validation(tracks)
        second = _sample_validation(tracks)
        self.assertEqual(first, second)
        self.assertEqual(len(first["road-waymo"]), len(first["loki"]))
        for dataset, rows in first.items():
            counts = {}
            for locator, frame, _label, _archive in rows:
                counts[locator] = counts.get(locator, 0) + 1
                self.assertTrue(tracks[dataset][locator]["gt_valid"][frame])
            self.assertLessEqual(max(counts.values()), 10)

    def test_cache_accepts_unlabeled_context_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            np.savez_compressed(path, features=np.zeros((3, 128), dtype=np.float32),
                targets=np.array([0, -1, 1], dtype=np.int64),
                gt_valid=np.array([True, False, True]), offsets=np.array([0, 3], dtype=np.int64),
                locators=np.array(["track"]), archives=np.array(["track.npz"]),
                metadata=np.asarray(json.dumps({"source": "road-waymo", "target": "road-waymo",
                    "split": "validation"})))
            result = _read_cache(path, "road-waymo", "road-waymo")
            self.assertEqual(result["tracks"]["track"]["targets"].tolist(), [0, -1, 1])


if __name__ == "__main__":
    unittest.main()
