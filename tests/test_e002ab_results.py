import unittest
from pathlib import Path

import numpy as np

from pedestrian_behavior.experiments.e002ab_results import (
    _metric_equal,
    _prediction_signature,
    _selected_row,
    _verify_alignment,
)


class E002ABResultsTests(unittest.TestCase):
    def test_selected_epoch_uses_last_best_improvement_and_checks_provenance(self):
        history = [
            {"epoch": 1, "selected": True, "training_ce": 1., "validation_ce": 1., "validation_f1": .2},
            {"epoch": 2, "selected": True, "training_ce": .8, "validation_ce": .7, "validation_f1": .4},
            {"epoch": 3, "selected": False, "training_ce": .6, "validation_ce": .6, "validation_f1": .3},
        ]
        selected = _selected_row(history, {"selection": {"epoch": 2}}, {"status": "complete"})
        self.assertEqual(selected["epoch"], 2)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            _selected_row(history, {"selection": {"epoch": 3}}, {"status": "complete"})

    def test_metric_reconstruction_comparison_checks_nested_numeric_values(self):
        _metric_equal({"a": [1., .2], "b": None}, {"a": [1.000000001, .2], "b": None})
        with self.assertRaisesRegex(ValueError, "metrics.a"):
            _metric_equal({"a": [1., .2]}, {"a": [1., .3]})

    def test_population_integrity_uses_gt_targets_and_saved_frame_times(self):
        base_track = {
            "locator": "track-a", "archive": "scene.npz", "group": "scene-1",
            "grid_times_us": [0, 200000], "source_times_us": [0, 200000],
            "targets": np.array([1, 2]), "gt_valid": np.array([True, True]),
            "conditions": {"duration": np.array([0, 0])},
            "source_flags": {"rgb_valid": np.array([True, False])},
        }
        same_track = base_track | {
            "archive": str(Path("scene.npz").resolve()),
            "targets": np.array([1, 2]), "gt_valid": np.array([True, True]),
            "conditions": {"duration": np.array([0, 0]), "rgb-observed": np.array([1, 0])},
            "source_flags": {"rgb_valid": np.array([True, False]), "lidar_valid": np.array([True, True])},
        }
        second_track = base_track | {
            "locator": "track-b", "archive": "scene.npz", "grid_times_us": [200000, 400000],
            "source_times_us": [200000, 400000], "targets": np.array([0, 3]),
            "gt_valid": np.array([True, False]), "conditions": {"duration": np.array([1, 1])},
            "source_flags": {"rgb_valid": np.array([False, False])},
        }
        same_second_track = second_track | {
            "archive": str(Path("scene.npz").resolve()),
            "conditions": {"duration": np.array([1, 1]), "rgb-observed": np.array([0, 0])},
            "source_flags": {"rgb_valid": np.array([False, False]), "lidar_valid": np.array([True, False])},
        }
        rows = [
            {"source": "loki", "run_path": "model-a"},
            {"source": "loki", "run_path": "model-b"},
        ]
        signatures = {
            "model-a": _prediction_signature([base_track, second_track]),
            "model-b": _prediction_signature([same_second_track, same_track]),
        }
        result = _verify_alignment(rows, signatures)["loki"]
        self.assertEqual(result["tracks"], 2)
        self.assertFalse(result["prediction_order_identical"])
        self.assertIn("duration", result["common_strata_axes"])
        self.assertEqual(result["available_modality_flags"], ["lidar_valid", "rgb_valid"])

        changed_time = same_track | {"grid_times_us": [0, 400000]}
        signatures["model-b"] = _prediction_signature([same_second_track, changed_time])
        with self.assertRaisesRegex(ValueError, "grid_times_us"):
            _verify_alignment(rows, signatures)

        changed_gt = same_track | {"targets": np.array([1, 3])}
        signatures["model-b"] = _prediction_signature([same_second_track, changed_gt])
        with self.assertRaisesRegex(ValueError, "targets misalignment"):
            _verify_alignment(rows, signatures)

        changed_rgb = same_track | {
            "source_flags": {"rgb_valid": np.array([False, False]), "lidar_valid": np.array([True, True])},
        }
        signatures["model-b"] = _prediction_signature([same_second_track, changed_rgb])
        with self.assertRaisesRegex(ValueError, "modality flag rgb_valid"):
            _verify_alignment(rows, signatures)

        signatures["model-b"] = _prediction_signature([same_track, same_track])
        with self.assertRaisesRegex(ValueError, "duplicate locator"):
            _verify_alignment(rows, signatures)


if __name__ == "__main__":
    unittest.main()
