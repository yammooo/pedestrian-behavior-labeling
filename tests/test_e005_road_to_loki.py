import tempfile
import unittest
from pathlib import Path

import numpy as np

from pedestrian_behavior.evaluation import save_predictions
from pedestrian_behavior.experiments.e005_road_to_loki import _matched_historical


def _track(locator, group, targets, logits):
    targets = np.asarray(targets, dtype=np.int64)
    return {"locator": locator, "archive": locator + ".npz", "group": group,
        "targets": targets, "gt_valid": np.ones(len(targets), dtype=bool),
        "logits": np.asarray(logits, dtype=np.float32), "conditions": {}, "source_flags": {}}


class E005RoadToLokiTest(unittest.TestCase):
    def test_matches_historical_predictions_by_partition_and_labels(self):
        dev = [_track("track-dev", "group-dev", [0, 1], [[4, 0, 0, 0], [0, 3, 0, 0]])]
        assessment = [_track("track-assess", "group-assess", [2, 3], [[0, 0, 4, 0], [0, 0, 0, 4]])]
        current = {row["locator"]: row for row in dev + assessment}
        partitions = {"development": dev, "assessment": assessment}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "zero-source-head").mkdir()
            save_predictions(root / "zero-source-head/development-predictions.npz", dev)
            save_predictions(root / "zero-source-head/assessment-predictions.npz", assessment)
            metrics, _, _ = _matched_historical(root, current, partitions)
            self.assertEqual(metrics["full_validation"]["track_weighted"]["macro_f1"], 1.)
            self.assertEqual(metrics["development"]["denominators"]["tracks"], 1)
            changed = dict(current)
            changed["track-dev"] = _track("track-dev", "group-dev", [1, 1], [[4, 0, 0, 0], [0, 3, 0, 0]])
            with self.assertRaisesRegex(ValueError, "targets changed"):
                _matched_historical(root, changed, partitions)


if __name__ == "__main__":
    unittest.main()
