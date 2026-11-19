import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pedestrian_behavior.evaluation import predict
from pedestrian_behavior.experiments import e005_run, e005_transfer as transfer
from pedestrian_behavior.experiments.e005_cache import E005FeatureDataset
from pedestrian_behavior.models.e005 import build_model


class OnlineTransferTest(unittest.TestCase):
    def test_one_sweep_for_two_tracks_and_missing_modalities(self):
        group = "synthetic-clip"
        a, b = [json.dumps({"clip": group, "tube_uid": tube}) for tube in ("a", "b")]
        track_index = {
            a: {"id": "a", "keys": [7, 8, None], "point_usable": [True, True, False]},
            b: {"id": "b", "keys": [7], "point_usable": [True]},
        }
        identity = np.eye(4)
        box = np.zeros(3), np.ones(3), 0.
        image = Image.new("RGB", (32, 32))

        class Baseline:
            entries = [{"group": group, "locator": a}, {"group": group, "locator": b}]

            def __getitem__(self, i):
                locator = self.entries[i]["locator"]
                n = len(track_index[locator]["keys"])
                return {"locator": locator, "archive": "example.npz", "inputs": torch.zeros(n, 20),
                        "targets": torch.zeros(n, dtype=torch.long), "gt_valid": torch.ones(n, dtype=torch.bool)}

        def native_points(_, clip, keys, ids, scene, *, include_rgb):
            self.assertEqual((clip, keys, ids, include_rgb), (group, {7, 8}, ["a", "b"], False))
            yield {"key": 7, "world_from_native": identity, "points": np.zeros((2, 4), np.float32),
                   "boxes": {"a": box, "b": (np.array([2., 0., 0.]), np.ones(3), 0.)}}
            yield {"key": 8, "world_from_native": identity, "points": None, "boxes": {"a": box}}

        def native_rgb(_, clip, keys, ids, scene):
            self.assertEqual(set(keys), {7, 8})
            yield 7, image, {"a": [.25, .25, .75, .75]}
            yield 8, None, {}

        lidar_calls, rgb_calls = [], []

        def stages(points):
            lidar_calls.append(len(points))
            xy3 = np.array([[0, 0, 0], [20, 0, 100], [21, 0, 0]], dtype=np.float32)
            xy5 = np.array([[1, 0, 0]], dtype=np.float32)
            return tuple((np.full((len(coords), width), value, np.float16)[:len(coords) if len(points) else 0],
                          coords[:len(coords) if len(points) else 0])
                         for coords, width, value in ((xy3, 72, 3), (xy5, 216, 5)))

        def descriptors(crops, *, patches):
            self.assertTrue(patches)
            self.assertEqual(crops.shape[1:], (3, 256, 256))
            rgb_calls.append(len(crops))
            return None, np.ones((len(crops), 16, 16, 768), np.float32)

        reader = SimpleNamespace(waymo_root=None, read_frames=lambda _, keys: {key: {"ego_pose": identity} for key in keys})
        model = build_model(transfer.ARM, e005_run.PROFILE)
        folded = build_model(transfer.ARM, e005_run.AALTO_FAST_PROFILE)
        folded.load_state_dict(model.state_dict())
        predictions = []
        with (patch.object(transfer.road_waymo, "component_path", return_value=Path(__file__)),
              patch.object(transfer, "native_scene", native_points),
              patch.object(transfer, "native_camera", native_rgb)):
            for prefetch, classifier in ((False, model), (True, folded)):
                features = transfer.RoadSpatialFeatures(track_index, reader, {group: {}},
                    SimpleNamespace(descriptors=descriptors), SimpleNamespace(stages=stages), prefetch=prefetch)
                dataset = E005FeatureDataset(Baseline(), features, transfer.ARM)
                predictions.append(predict(classifier, transfer._batches(dataset, features, prefetch=prefetch), "cpu",
                    forward_batch=lambda m, batch, _: m(batch, mode="ordinary") if prefetch else m(batch)))

        self.assertEqual((lidar_calls, rgb_calls), ([2, 0, 2, 0], [1, 1]))
        original, tracks = predictions
        for left, right in zip(original, tracks):
            self.assertEqual(left["locator"], right["locator"])
            np.testing.assert_allclose(left["logits"], right["logits"], atol=1e-6, rtol=1e-6)
        self.assertEqual([row["logits"].shape for row in tracks], [(3, 4), (1, 4)])
        np.testing.assert_array_equal(tracks[0]["source_flags"]["rgb_valid"], [True, False, False])
        np.testing.assert_array_equal(tracks[0]["source_flags"]["lidar_valid"], [True, False, False])
        np.testing.assert_array_equal(tracks[1]["source_flags"]["rgb_valid"], [False])
        level3, level5 = features.lidar_frame(a, 0)
        self.assertEqual((level3.shape, level5.shape), ((2, 77), (1, 221)))
        np.testing.assert_array_equal(level3[:, 0], [3., 3.])
        np.testing.assert_array_equal(level5[:, 0], [5.])
        self.assertEqual(level3[1, 74], 5.)  # Vertical height alone does not exclude the token.
        self.assertEqual(features.lidar_frame(b, 0)[0].shape, (3, 77))

    def test_prefetch_propagates_decoder_failure(self):
        closed = []

        def frames():
            try:
                yield 1
                raise ValueError("invalid native frame")
            finally:
                closed.append(True)

        decoded = transfer._one_ahead(frames())
        self.assertEqual(next(decoded), 1)
        with self.assertRaisesRegex(ValueError, "invalid native frame"):
            next(decoded)
        self.assertEqual(closed, [True])

    def test_in_progress_source_is_rejected_before_checkpoint_loading(self):
        with TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "status.json").write_text('{"status":"incomplete"}')
            (run / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "completed full LOKI E005"):
                transfer._load_source(run)


if __name__ == "__main__":
    unittest.main()
