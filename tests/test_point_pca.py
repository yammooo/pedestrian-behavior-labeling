"""Fixed PCA, training-only calibration, hierarchical correspondence and panels."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pedestrian_behavior.inspection.feature_video import PCA_VIDEO_SIZE, render_pca_frame
from pedestrian_behavior.inspection.point_pca import calibration_frames, fit_pca, pca_colors
from pedestrian_behavior.inspection.pretrained import UtoniaEncoder, utonia_features
from pedestrian_behavior.experiments.e002a_point_check import run


class PointPcaTest(unittest.TestCase):
    def test_pca_known_variance_and_fixed_transform_without_refitting(self):
        values = np.array([[a, b, c, 4.] for a in (-3., 3.) for b in (-2., 2.) for c in (-1., 1.)], dtype=np.float32)
        pca = fit_pca(values)
        np.testing.assert_allclose(pca["components"], np.eye(4)[:, :3], atol=1e-6)
        np.testing.assert_allclose(pca["explained_variance_ratio"], [9/14, 4/14, 1/14])
        baseline = pca_colors(torch.tensor(values), pca).numpy()
        np.testing.assert_allclose(baseline[0], [0, 0, 0])
        np.testing.assert_allclose(baseline[-1], [1, 1, 1])
        # Viewing a different set cannot change colors of the same point.
        extra = torch.tensor(np.array([[300., 200., -100., 4.], values[0]], dtype=np.float32))
        np.testing.assert_array_equal(pca_colors(extra, pca)[1].numpy(), baseline[0])
        for invalid in (values+np.nan, values[:2], np.zeros((4, 2))):
            with self.assertRaises(ValueError):
                fit_pca(invalid)
        with self.assertRaises(ValueError):
            pca_colors(torch.zeros(3, 5), pca)
        with self.assertRaises(ValueError):
            pca_colors(torch.full((3, 4), float("nan")), pca)

    def test_calibration_is_balanced_training_only_and_independent_of_order(self):
        frames = [{"dataset": d, "split": s, "group": str(g), "key": str(k)}
                  for d in ("loki", "road-waymo") for s in ("training", "validation", "test")
                  for g in range(5) for k in (1, 3, 2)]
        for dataset in ("loki", "road-waymo"):
            chosen = calibration_frames({"frames": list(reversed(frames))}, dataset)
            self.assertEqual(chosen, [(str(g), str(k)) for g in range(4) for k in (1, 2)])
        with self.assertRaises(ValueError):
            calibration_frames({"frames": [r for r in frames if r["split"] != "training"]}, "loki")

    @staticmethod
    def hierarchy(coord):
        first = {"coord": coord, "feat": torch.zeros(len(coord), 54)}
        current = first
        for level, width in enumerate((108, 216, 432, 576), 1):
            count = max(1, len(current["feat"])-1)
            inverse = torch.arange(len(current["feat"])) % count
            current = {"feat": torch.arange(count*width).reshape(count, width).float()+level,
                       "pooling_parent": current, "pooling_inverse": inverse}
        return current

    def test_gather_matches_author_full_upcast_including_repeated_indices(self):
        coord = torch.arange(18).reshape(6, 3).float()
        point = self.hierarchy(coord)
        values, actual_coord = utonia_features(point, torch.tensor([5, 0, 5, 2]))
        current = point
        full = current["feat"]
        for _ in range(2):
            parent = current["pooling_parent"]
            full = torch.cat((parent["feat"], full[current["pooling_inverse"]]), dim=-1)
            current = parent
        while "pooling_parent" in current:
            full = full[current["pooling_inverse"]]
            current = current["pooling_parent"]
        torch.testing.assert_close(values, full[[5, 0, 5, 2]])
        torch.testing.assert_close(actual_coord, coord)
        self.assertEqual(values.shape, (4, 1224))
        with self.assertRaises(ValueError):
            utonia_features({"feat": torch.ones(2, 4)}, torch.tensor([0]))

    def test_utonia_outdoor_scale_missing_channels_and_inverse_correspondence(self):
        encoder = UtoniaEncoder.__new__(UtoniaEncoder)
        encoder.device = "cpu"
        def model(data):
            torch.testing.assert_close(data["feat"][:, :3], data["coord"])
            torch.testing.assert_close(data["feat"][:, 3:6], torch.zeros_like(data["coord"]))
            torch.testing.assert_close(data["feat"][:, 6:], torch.zeros_like(data["coord"]))
            self.assertTrue(float(data["coord"].abs().max()) < 2.)
            return self.hierarchy(data["coord"])
        encoder.model = model
        points = np.array([[0, 0, 0, .5], [.001, 0, 0, .1], [1, 2, 3, .9], [4, 2, 3, .1]], dtype=np.float32)
        result = encoder.features(points, np.array([0, 1, 2]))
        np.testing.assert_array_equal(result[0], result[1])
        projected = encoder.features(points, np.array([0, 1, 2]), lambda x: x[:, :3])
        np.testing.assert_array_equal(projected, result[:, :3])

    def test_pca_panels_share_point_positions_and_keep_missing_inputs(self):
        frame = {"image": Image.new("RGB", (100, 100), "blue"), "camera": {"p": [.4, .2, .6, .8]},
                 "boxes": {"p": (np.zeros(3), np.array([2., 1., 2.]), 0.)}, "world_from_native": np.eye(4)}
        offsets = np.array([[3., 4., 0.]])
        colors = {"dditr": np.array([[1., 0., 0.]]), "utonia": np.array([[0., 1., 0.]])}
        image, evidence = render_pca_frame(frame, "p", np.eye(4), offsets, colors, "Known points")
        self.assertEqual(image.size, PCA_VIDEO_SIZE)
        self.assertEqual(image.getpixel((704+280, 90+290)), (255, 0, 0))
        self.assertEqual(image.getpixel((1383+280, 90+290)), (0, 255, 0))
        self.assertEqual(evidence["points_within_radii"], [1, 1, 1])
        _, missing = render_pca_frame(None, "p", np.eye(4), None, {}, "Missing")
        self.assertFalse(missing["rgb_valid"] or missing["lidar_valid"])

    def test_checkpoint_failure_propagates_and_retains_local_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"outputs/experiments/E002a"
            probe, videos, output = root/"probe", root/"videos", root/"comparison"
            probe.mkdir(parents=True)
            videos.mkdir()
            weights = root/"weights.pth"
            weights.write_bytes(b"invalid checkpoint")
            source = {"status": "complete", "protected": {}, "config": {
                "ditr_code": str(root), "ditr_weights": str(weights), "ditr_config": str(weights)}}
            (probe/"probe.json").write_text(json.dumps(source))
            old = json.dumps({"status": "complete", "tracks": [{}]*10})
            (videos/"videos.json").write_text(old)
            args = argparse.Namespace(probe=probe, videos=videos, output=output,
                                      utonia_code=root, utonia_weights=weights, device="cpu")
            with patch("pedestrian_behavior.experiments.e002a_point_check.PointEncoder", side_effect=ValueError("bad checkpoint")):
                with self.assertRaisesRegex(ValueError, "bad checkpoint"):
                    run(args)
            record = json.loads((output/"comparison.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertIn("bad checkpoint", record["failures"][0])
            self.assertEqual(record["changed_files"], [])
            self.assertEqual((videos/"videos.json").read_text(), old)
            self.assertTrue((output/"index.html").is_file())


if __name__ == "__main__":
    unittest.main()
