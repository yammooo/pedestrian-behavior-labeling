"""Hand-calculated checks for the bounded output-pooling diagnostic."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from pedestrian_behavior.experiments.e002a_pooling import diagnostic_payload, storage_measure
from pedestrian_behavior.inspection.output_pooling import pool_outputs


class OutputPoolingTest(unittest.TestCase):
    def test_means_negative_cells_and_preservation(self):
        xyz = np.array([[-.09, 0, 0], [-.01, 0, 0], [.01, 0, 0], [.09, 0, 0]], np.float32)
        features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.float32)
        coordinates, means, inverse, counts = pool_outputs(xyz, features, .1)
        np.testing.assert_allclose(coordinates[:, 0], [-.05, .05])
        np.testing.assert_array_equal(means, [[2, 3], [6, 7]])
        np.testing.assert_array_equal(inverse, [0, 0, 1, 1])
        np.testing.assert_array_equal(counts, [2, 2])
        preserve = np.array([False, False, True, True])
        coordinates, means, inverse, counts = pool_outputs(xyz, features, .1, preserve)
        np.testing.assert_array_equal(means[inverse][preserve], features[preserve])
        self.assertEqual(len(means), 3)
        self.assertEqual(counts.sum(), 4)

    def test_boundaries_empty_and_invalid(self):
        xyz = np.array([[0., 0, 0], [.125, 0, 0], [-.125, 0, 0]], np.float32)
        self.assertEqual(len(pool_outputs(xyz, xyz, .125)[0]), 3)
        result = pool_outputs(np.zeros((0, 3)), np.zeros((0, 64)), .1)
        self.assertEqual(result[1].shape, (0, 64))
        for size in [0, -1, float("inf"), float("nan")]:
            with self.assertRaises(ValueError):
                pool_outputs(xyz, xyz, size)
        for values in [np.full((3, 3), np.nan), np.full((3, 3), np.inf)]:
            with self.assertRaises(ValueError):
                pool_outputs(xyz, values, .1)

    def test_inverse_reconstructs_independent_cell_means(self):
        rng = np.random.default_rng(0)
        xyz = rng.uniform(-1, 1, (150, 3)).astype(np.float32)
        features = rng.normal(size=(150, 64)).astype(np.float32)
        coordinates, means, inverse, counts = pool_outputs(xyz, features, .2)
        for i in range(len(means)):
            members = inverse == i
            self.assertEqual(counts[i], members.sum())
            np.testing.assert_allclose(means[i], features[members].mean(0), atol=1e-6)
            np.testing.assert_allclose(coordinates[i], xyz[members].mean(0), atol=1e-6)

    def test_target_mixing_adaptive_preservation_and_storage_roundtrip(self):
        # Two points in the same 20cm cell, only the first inside the target box.
        xyz = np.array([[.04, 0, 0], [.14, 0, 0], [5.01, 0, 0], [5.09, 0, 0]], np.float32)
        arrays = {"coordinates": xyz, "features": np.arange(256, dtype=np.float32).reshape(4, 64),
            "native_indices": np.array([3, 7, 12, 14], np.int64), "world_from_native": np.eye(4),
            "boxes": np.array([[.04, 0, 0, .1, .1, .1, 0]]), "anchors": np.eye(4)[None],
            "locators": np.array(["test"])}
        _, pooled = diagnostic_payload(arrays, "20cm")
        self.assertEqual(pooled["targets"][0]["inside_tokens_mixed_with_outside"], 1)
        payload, adaptive = diagnostic_payload(arrays, "adaptive")
        self.assertEqual(adaptive["tokens"], 3)
        self.assertEqual(adaptive["targets"][0]["inside_tokens_mixed_with_outside"], 0)
        with tempfile.TemporaryDirectory() as folder:
            # The headroom guard uses the real filesystem; these tiny fixtures need <20KB.
            from unittest.mock import patch
            from types import SimpleNamespace
            with patch("pedestrian_behavior.experiments.e002a_pooling.shutil.disk_usage", return_value=SimpleNamespace(free=10**12)):
                storage = storage_measure(payload, Path(folder) / "temporary.npz")
            self.assertLess(storage["fp16"]["bytes"], storage["fp32"]["bytes"])
            self.assertFalse((Path(folder) / "temporary.npz").exists())


if __name__ == "__main__":
    unittest.main()
