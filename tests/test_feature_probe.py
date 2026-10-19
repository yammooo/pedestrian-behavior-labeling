"""Native geometry/correspondence gates independent of pretrained downloads."""

import copy
import unittest

import numpy as np
from PIL import Image
import torch

from pedestrian_behavior.datasets.road_waymo import range_points
from pedestrian_behavior.experiments.e002a_probe import sample_groups, selected_frames
from pedestrian_behavior.inspection.feature_probe import (RGB_MEAN, ProbeReadout, box_membership,
    crop_examples, normalized_crop, patch_maps, relative_xyz, square_crop, voxel_sample, temporal_probe)
from pedestrian_behavior.inspection.pretrained import ImageEncoder, image_state_from_hf


class FeatureProbeTest(unittest.TestCase):
    def test_gallery_valid_examples_precede_missing_and_keep_original_indices(self):
        tracks = [{"crop": {"valid": v}} for v in (False, False, False, True, False, True, True)]
        self.assertEqual(crop_examples(tracks, valid_limit=2), [3, 5, 0])
        self.assertEqual(crop_examples(tracks[:3]), [0])
        self.assertEqual(crop_examples([]), [])

    def test_patch_maps_cosine_pca_constant_and_invalid_inputs(self):
        features = np.zeros((3, 3, 4), dtype=np.float32)
        features[0, 0, 0], features[0, 1, 0], features[0, 2, 1] = 1, -2, 1
        colors, similarity = patch_maps(features, (0, 0))
        np.testing.assert_allclose(similarity[0], [1, -1, 0])
        self.assertEqual(colors.shape, (3, 3, 3))
        self.assertEqual(colors.dtype, np.float32)
        self.assertTrue(((colors >= 0) & (colors <= 1)).all())
        np.testing.assert_array_equal(colors, patch_maps(features, (0, 0))[0])
        constant, empty = patch_maps(np.zeros_like(features), (1, 1))
        np.testing.assert_allclose(constant, .5)
        np.testing.assert_allclose(empty, 0)
        with self.assertRaises(ValueError):
            patch_maps(features, (3, 0))
        with self.assertRaises(ValueError):
            patch_maps(features+np.nan, (0, 0))

    def test_image_adapter_returns_only_patch_tokens_and_preserves_cls_interface(self):
        class Backbone:
            def forward_features(self, images):
                return {"x_norm_clstoken": torch.ones(len(images), 768),
                        "x_norm_patchtokens": torch.arange(256*768).reshape(1, 256, 768).float().repeat(len(images), 1, 1),
                        "x_storage_tokens": torch.zeros(len(images), 4, 768)}
        encoder = ImageEncoder.__new__(ImageEncoder)
        encoder.model, encoder.device = Backbone(), "cpu"
        images = torch.zeros(2, 3, 256, 256)
        cls, patches = encoder.descriptors(images, patches=True)
        np.testing.assert_array_equal(cls, encoder(images))
        self.assertEqual(patches.shape, (2, 16, 16, 768))
        self.assertEqual(patches[0, 1, 0, 0], 16*768)
        with self.assertRaises(ValueError):
            encoder.descriptors(images[:, :, :128], patches=True)

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

    def test_crop_expansion_preserves_center_resolution_and_default(self):
        image = Image.new("RGB", (100, 100), "red")
        box = [0, 0, .2, .4]
        baseline, _ = square_crop(image, box)
        for expansion, side in ((1.5, 60), (2., 80), (3., 120)):
            pixels, info = square_crop(image, box, expansion=expansion)
            bounds = np.array(info["bounds_pixels"])
            np.testing.assert_allclose((bounds[:2]+bounds[2:])/2, [10, 20])
            np.testing.assert_allclose(bounds[2:]-bounds[:2], side)
            self.assertEqual(pixels.shape, (256, 256, 3))
            np.testing.assert_allclose(pixels[0, 0], RGB_MEAN)
            if expansion == 1.5:
                np.testing.assert_array_equal(pixels, baseline)
        for invalid in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                square_crop(image, box, expansion=invalid)

    def test_hf_checkpoint_conversion_retains_all_learned_values(self):
        expected = {"blocks.0.attn.qkv.weight": torch.zeros(6, 2), "blocks.0.attn.qkv.bias": torch.zeros(6),
                    "blocks.0.attn.qkv.bias_mask": torch.tensor([1, 1, 0, 0, 1, 1]),
                    "rope_embed.periods": torch.ones(1), "mask_token": torch.zeros(1, 2),
                    "blocks.0.ls1.gamma": torch.zeros(2)}
        state = {"model.layer.0.attention.q_proj.weight": torch.ones(2, 2),
                 "model.layer.0.attention.k_proj.weight": torch.ones(2, 2)*2,
                 "model.layer.0.attention.v_proj.weight": torch.ones(2, 2)*3,
                 "model.layer.0.attention.q_proj.bias": torch.ones(2),
                 "model.layer.0.attention.v_proj.bias": torch.ones(2)*3,
                 "embeddings.mask_token": torch.ones(1, 1, 2)*4,
                 "model.layer.0.layer_scale1.lambda1": torch.ones(2)*5}
        converted = image_state_from_hf(state, expected)
        torch.testing.assert_close(converted["blocks.0.attn.qkv.bias"], torch.tensor([1., 1., 0., 0., 3., 3.]))
        self.assertEqual(converted["mask_token"].shape, (1, 2))
        self.assertEqual(converted["blocks.0.attn.qkv.weight"][4, 0], 3)
        with self.assertRaises(ValueError):
            image_state_from_hf(state | {"unexpected": torch.ones(1)}, expected)

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

    def test_temporal_padding_isolation(self):
        torch.manual_seed(0)
        layer = torch.nn.TransformerEncoderLayer(256, 4, 1024, .1, batch_first=True, norm_first=True)
        temporal = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False).eval()
        values = torch.randn(2, 6, 256, requires_grad=True)
        lengths = torch.tensor([3, 6])
        first = temporal_probe(temporal, values, lengths)
        other = values.detach().clone()
        other[0, 3:] = float("nan")
        torch.testing.assert_close(first, temporal_probe(temporal, other, lengths))
        self.assertEqual(first[0, 3:].count_nonzero(), 0)
        first.square().sum().backward()
        self.assertEqual(values.grad[0, 3:].count_nonzero(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_readout_finite_backward_and_pair_order(self):
        torch.manual_seed(0)
        model = ProbeReadout(128, 4).cuda().eval()
        tokens = torch.randn(257, 68, device="cuda", requires_grad=True)
        result = model(tokens)
        torch.testing.assert_close(result, model(tokens.flip(0)), atol=1e-5, rtol=1e-4)
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(tokens.grad).all())


if __name__ == "__main__":
    unittest.main()
