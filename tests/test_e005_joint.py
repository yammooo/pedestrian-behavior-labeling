import unittest

import numpy as np

from pedestrian_behavior.data.features import fit_normalization
from pedestrian_behavior.experiments.e005_run import JointSpatialBatches, _merge_normalization


class _Dataset:
    def __init__(self, road_tracks, loki_tracks):
        self.source_indices = {
            "road-waymo": list(range(road_tracks)),
            "loki": list(range(road_tracks, road_tracks + loki_tracks)),
        }
        self.entries = [{} for _ in range(road_tracks + loki_tracks)]

    def __len__(self):
        return len(self.entries)


class E005JointTest(unittest.TestCase):
    def test_balanced_updates_cycle_across_epochs(self):
        dataset = _Dataset(3, 7)
        batches = JointSpatialBatches(dataset, {"max_tracks": 1}, training=True, seed=0,
                                      source_batch_size=2)
        for _ in range(2):
            updates = list(batches)
            self.assertEqual(len(updates), 3)
            self.assertEqual([len(update) for update in updates], [4, 4, 2])
            self.assertEqual(sum(i < 3 for update in updates for i in update), 5)
            self.assertEqual(sum(i >= 3 for update in updates for i in update), 5)
            for update in updates:
                self.assertEqual(sum(i < 3 for i in update), sum(i >= 3 for i in update))

    def test_pooled_moments_match_joint_training_features(self):
        road = (np.array([[0., 1.], [2., 0.], [4., 3.]], dtype=np.float64),
                np.array([[True, True], [True, False], [True, True]]))
        loki = (np.array([[10., 5.], [14., 0.]], dtype=np.float64),
                np.array([[True, True], [True, False]]))
        expected = fit_normalization([road, loki])
        actual = _merge_normalization([fit_normalization([road]), fit_normalization([loki])])
        for key in ("mean", "scale", "count", "std"):
            np.testing.assert_allclose(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
