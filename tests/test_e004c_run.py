import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pedestrian_behavior.experiments import e004c_run


def _physical_track(offset):
    return {"ped_position": np.asarray([[offset, 0.], [offset + 1., 0.]], dtype=np.float64),
        "ped_velocity": np.asarray([[1., 0.], [1., 0.]], dtype=np.float64),
        "ego_position": np.asarray([[0., 0.], [0., 0.]], dtype=np.float64),
        "ego_velocity": np.asarray([[0., 0.], [0., 0.]], dtype=np.float64),
        "ped_position_valid": np.ones(2, dtype=bool), "ped_velocity_valid": np.ones(2, dtype=bool),
        "ego_pose_valid": np.ones(2, dtype=bool), "ego_velocity_valid": np.ones(2, dtype=bool)}


class E004cRunTest(unittest.TestCase):
    def test_target_normalization_reads_selected_tracks_only(self):
        entries = [{"locator": locator, "archive": f"{locator}.npz", "e001_sha256": "sha",
                    "bbox_sha256": "sha"} for locator in ("a", "b", "excluded")]
        baseline = SimpleNamespace(entries=entries, collection=Path("collection"), extension=Path("extension"))
        selected = [{"locator": "a"}, {"locator": "b"}]
        archives = {"a.npz": _physical_track(1.), "b.npz": _physical_track(5.)}
        bbox = {"bbox_features": np.asarray([[1., 2., 3., 4., .1, .2], [2., 3., 4., 5., .2, .3]], dtype=np.float64),
            "bbox_valid": np.ones(2, dtype=bool), "bbox_velocity_valid": np.ones(2, dtype=bool)}
        with patch.object(e004c_run, "checksum", return_value="sha"), \
                patch.object(e004c_run, "load_track", side_effect=lambda path: (None, archives[Path(path).name])) as load_track, \
                patch.object(e004c_run, "load_bbox", return_value=(None, bbox)):
            result = e004c_run._target_normalization(selected, baseline)
        self.assertEqual([Path(call.args[0]).name for call in load_track.call_args_list], ["a.npz", "b.npz"])
        self.assertEqual(result["baseline"]["count"], [4] * 8)
        self.assertEqual(result["bbox"]["count"], [4] * 6)

    def test_target_dataset_sorts_baseline_before_wrapping_supply(self):
        class Baseline:
            collection, extension = Path("collection"), Path("extension")
            def __init__(self):
                self.entries = [{"locator": "z", "group": "g"}, {"locator": "a", "group": "g"}]
            def __len__(self):
                return len(self.entries)

        baseline = Baseline()
        supply = SimpleNamespace(catalogs={"loki": SimpleNamespace(
            tracks={"a": {"locator": "a"}, "z": {"locator": "z"}})})
        with patch.object(e004c_run.e001b, "dataset_from_extension", return_value=baseline):
            dataset = e004c_run._target_dataset(Path("outputs"), baseline,
                {"baseline": {}, "bbox": {}}, supply, "training")
        self.assertEqual([row["locator"] for row in dataset.entries], ["a", "z"])
        self.assertEqual([row["locator"] for row in dataset.baseline.entries], ["a", "z"])

    def test_saved_predictions_preserve_metrics_inputs(self):
        track = {"locator": "track-a", "archive": "track-a.npz", "group": "scenario-a",
            "logits": np.asarray([[1., 0., 0., 0.]], dtype=np.float32),
            "targets": np.asarray([0], dtype=np.int64), "gt_valid": np.asarray([True]),
            "conditions": {}, "source_flags": {"rgb_valid": np.asarray([True])}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            e004c_run._write_predictions(path, [track])
            restored = e004c_run.load_predictions(path)
            e004c_run._check_prediction_roundtrip([track], restored)


if __name__ == "__main__":
    unittest.main()
