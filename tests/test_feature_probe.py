"""Native geometry/correspondence gates independent of pretrained downloads."""

import copy
import unittest

import numpy as np
from PIL import Image
import torch

from pedestrian_behavior.datasets.road_waymo import range_points
from pedestrian_behavior.experiments.e002a_probe import sample_groups, selected_frames
from pedestrian_behavior.inspection.feature_probe import (RGB_MEAN, ProbeReadout, box_membership,
    normalized_crop, relative_xyz, square_crop, voxel_sample)


class FeatureProbeTest(unittest.TestCase):
    def test_crop_boundary_outside_and_mean_padding(self):
        image = Image.new("RGB", (100, 100), "red")
        crop, info = square_crop(image, [0, 0, .2, .4])
        self.assertEqual(crop.shape, (256, 256, 3))
        self.assertEqual(info["bounds_pixels"], [-20., -10., 40., 50.])
        self.assertAlmostEqual(info["padding_fraction"], 1-2000/3600)
        np.testing.assert_allclose(crop[0, 0], RGB_MEAN)
        np.testing.assert_allclose(normalized_crop(crop)[:, 0, 0], 0, atol=1e-6)
        self.assertEqual(square_crop(image, [1, 0, 2, 1])[1]["reason"], "box-outside-image")
        self.assertFalse(square_crop(image, [0, 0, 0, 1])[1]["valid"])
        self.assertFalse(square_crop(None, None)[1]["valid"])
        self.assertEqual(square_crop(image, [-.1, 0, .1, .2])[1]["box_intersection_fraction"], .5)

    def test_rotated_cuboid_and_initial_heading_offsets(self):
        points = np.array([[0, 1.9, 0], [1.1, 0, 0], [0, 0, 1.1]])
        np.testing.assert_array_equal(box_membership(points, [0, 0, 0], [4, 2, 2], np.pi/2), [True, False, False])
        pose = np.eye(4)
        pose[:2, :2] = [[0, -1], [1, 0]]
        anchor = np.eye(4)
        np.testing.assert_allclose(relative_xyz(np.array([[1, 0, 2]]), [0, 0, 0], pose, anchor), [[0, 1, 2]])
        np.testing.assert_allclose(relative_xyz(np.array([[1, 0, 2]]), [0, 0, 0], pose, pose), [[1, 0, 2]], atol=1e-6)

    def test_voxels_preserve_feature_coordinate_pairs(self):
        points = np.array([[0, 0, 0, .1], [.01, 0, 0, .2], [-.051, 0, 0, .3], [1, 2, 3, .4]], dtype=np.float32)
        selected, inverse, grid = voxel_sample(points)
        self.assertEqual(len(selected), 3)
        self.assertEqual(inverse[0], inverse[1])
        self.assertIn(0, selected)
        expected = np.floor(points[:, :3].astype(np.float64)/.05).astype(int)
        expected -= expected.min(0)
        np.testing.assert_array_equal(grid[inverse], expected)
        np.testing.assert_array_equal(voxel_sample(points)[0], selected)
        with self.assertRaises(ValueError):
            voxel_sample(np.empty((0, 4)))
        points[0, 3] = np.nan
        with self.assertRaises(ValueError):
            voxel_sample(points)

    def test_strength_same_mask_as_native_coordinates(self):
        c = "[LiDARCalibrationComponent]."
        calibration = {c+"extrinsic.transform": np.eye(4).ravel(), c+"beam_inclination.values": [0],
                       c+"beam_inclination.min": 0., c+"beam_inclination.max": 0.}
        values = [2, 7, 0, -1, 0, 99, 0, -1, 4, 11, 0, -1]
        xyz = range_points(values, [1, 3, 4], calibration)
        both = range_points(values, [1, 3, 4], calibration, with_strength=True)
        np.testing.assert_array_equal(both[:, :3], xyz)
        np.testing.assert_array_equal(both[:, 3], [7, 11])

    def test_readout_pair_reordering_empty_and_gradients(self):
        torch.manual_seed(0)
        for seeds in (1, 4):
            model = ProbeReadout(32, seeds).eval()
            tokens = torch.randn(17, 68, requires_grad=True)
            first = model(tokens)
            torch.testing.assert_close(first, model(tokens[torch.randperm(17)]), atol=1e-6, rtol=1e-5)
            first.square().sum().backward()
            self.assertTrue(torch.isfinite(tokens.grad).all())
            self.assertEqual(model(tokens[:0]).shape, (256,))
            self.assertEqual(model(tokens[:0]).count_nonzero(), 0)

    def test_sample_never_uses_test_groups_or_changes_setup(self):
        setup = {"tracks": [], "groups": {}}
        for split, count in (("training", 12), ("validation", 3), ("test", 4)):
            for i in range(count):
                g = f"{split}-{i}"
                setup["groups"][g] = split
                setup["tracks"].append(dict(group=g, split=split, exclusions=[], slots=i+1, locator=g))
        before = copy.deepcopy(setup)
        grouped, sample, longest = sample_groups(setup)
        self.assertEqual(setup, before)
        self.assertFalse(any(g.startswith("test") for g in grouped))
        self.assertEqual(sum(s["random"] and s["split"] == "training" for s in sample.values()), 8)
        self.assertEqual(sum(s["random"] and s["split"] == "validation" for s in sample.values()), 2)
        self.assertEqual(longest["slots"], 12)
        self.assertEqual(sample_groups(setup)[1], sample)
        chosen, size = selected_frames({"a": {"source_frames": [None, "0000", "0002", "0004", "0006"]}}, True, [])
        self.assertEqual(len(chosen), 3)
        self.assertEqual(size, 4)


if __name__ == "__main__":
    unittest.main()
