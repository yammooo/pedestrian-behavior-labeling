"""E002b frozen model variants preserve masks, shapes and checkpoint structure."""

import io
import unittest

import torch

from pedestrian_behavior.data.multimodal import collate_multimodal
from pedestrian_behavior.experiments.e002b import ARMS, VARIANTS, build_model, model_spec
from pedestrian_behavior.models.multimodal import ARM_FLAGS, MultimodalClassifier


PROFILE = {"frame_chunk": 2, "frame_tokens": 64, "ordinary_tokens": 256}


def batch_for(arm):
    torch.manual_seed(17)
    names = ARM_FLAGS[arm]
    full_flags = [
        {"geometry_valid": 1, "rgb_valid": 0, "lidar_valid": 1},
        {"geometry_valid": 0, "rgb_valid": 1, "lidar_valid": 0},
        {"geometry_valid": 1, "rgb_valid": 1, "lidar_valid": 1},
    ]
    second = [
        {"geometry_valid": 1, "rgb_valid": 1, "lidar_valid": 0},
        {"geometry_valid": 0, "rgb_valid": 0, "lidar_valid": 1},
    ]
    samples = []
    for index, rows in enumerate((full_flags, second)):
        count = len(rows)
        geometry = torch.randn(count, 20)
        flags = torch.tensor([[row[name] for name in names] for row in rows], dtype=torch.float32)
        sample = {"geometry": geometry, "flags": flags,
                  "targets": torch.arange(count, dtype=torch.long) % 4,
                  "gt_valid": torch.ones(count, dtype=torch.bool),
                  "locator": f"track-{index}", "archive": f"track-{index}.npz"}
        if "rgb_valid" in names:
            sample["rgb"] = torch.randn(count, 768)
        if "lidar_valid" in names:
            sample["points"] = [torch.randn(5 + j, 68) if row["lidar_valid"] else torch.zeros(0, 68)
                                for j, row in enumerate(rows)]
        samples.append(sample)
    return collate_multimodal(samples)


class E002bModelTest(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous)

    def test_variant_shapes_post_norm_masks_checkpoint_and_backward(self):
        expected = {
            "norm": (64, 128, 128, 256, 1024),
            "temporal-small": (64, 128, 128, 128, 512),
            "branches-small": (32, 64, 64, 256, 1024),
        }
        for variant, dims in expected.items():
            geometry_width, rgb_width, spatial_width, temporal_width, temporal_ffn = dims
            for arm in ARMS:
                with self.subTest(variant=variant, arm=arm):
                    spec = model_spec(arm, variant)
                    model = build_model(arm, variant, PROFILE).eval()
                    batch = batch_for(arm)
                    batch["flags"][1, 2:] = 1.
                    batch["geometry"][1, 2:] = 9.
                    if "rgb" in batch:
                        batch["rgb"][1, 2:] = -7.
                    self.assertEqual(model.geometry[0].out_features, geometry_width)
                    self.assertEqual(model.geometry[-2].out_features, geometry_width)
                    self.assertEqual(model.fusion.in_features, spec["fusion"][0])
                    self.assertEqual(model.fusion.out_features, temporal_width)
                    self.assertEqual(model.temporal[0].ffn[0].out_features, temporal_ffn)
                    self.assertEqual(model.head.in_features, temporal_width)
                    self.assertIsNotNone(model.geometry_norm)
                    if model.rgb is not None:
                        self.assertEqual(model.rgb.out_features, rgb_width)
                    if model.spatial is not None:
                        self.assertEqual(model.spatial.project.out_features, spatial_width)

                    for module in (model.geometry_norm, model.rgb_norm, model.spatial_norm):
                        if module is not None:
                            with torch.no_grad():
                                module.bias.fill_(.375)

                    fused = []
                    hook = model.fusion.register_forward_pre_hook(
                        lambda _module, args: fused.append(args[0].detach().clone()))
                    logits = model(batch)
                    hook.remove()
                    self.assertEqual(tuple(logits.shape), (2, 3, 4))
                    self.assertTrue(torch.equal(logits[1, 2:], torch.zeros_like(logits[1, 2:])))

                    valid = torch.arange(batch["geometry"].shape[1])[None, :] < batch["lengths"][:, None]
                    flags = batch["flags"].masked_fill(~valid[..., None], 0.)
                    cursor = 0
                    for index, name in enumerate(ARM_FLAGS[arm]):
                        width = {"geometry_valid": geometry_width, "rgb_valid": rgb_width,
                                 "lidar_valid": spatial_width}[name]
                        norm = {"geometry_valid": model.geometry_norm,
                                "rgb_valid": model.rgb_norm,
                                "lidar_valid": model.spatial_norm}[name]
                        self.assertIsNotNone(norm)
                        descriptor = fused[0][..., cursor:cursor + width]
                        absent = flags[..., index] == 0
                        self.assertTrue(torch.equal(descriptor[absent], torch.zeros_like(descriptor[absent])))
                        cursor += width
                    torch.testing.assert_close(fused[0][..., cursor:], flags, rtol=0, atol=0)

                    # Save after setting nonzero affine bias so reload exercises normalized parameters too.
                    model.zero_grad(set_to_none=True)
                    model(batch).square().sum().backward()
                    for module in (model.geometry_norm, model.rgb_norm, model.spatial_norm):
                        if module is not None:
                            self.assertIsNotNone(module.weight.grad)
                            self.assertIsNotNone(module.bias.grad)

                    checkpoint = io.BytesIO()
                    torch.save(model.state_dict(), checkpoint)
                    checkpoint.seek(0)
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    restored = build_model(arm, variant, PROFILE).eval()
                    restored.load_state_dict(state, strict=True)
                    torch.testing.assert_close(model(batch), restored(batch), rtol=0, atol=0)

    def test_e002a_default_dimensions_and_state_dict_names_remain_unchanged(self):
        model = MultimodalClassifier("g-rgb-dditr", PROFILE)
        self.assertEqual((model.geometry[0].out_features, model.geometry[-2].out_features), (64, 64))
        self.assertEqual(model.rgb.out_features, 128)
        self.assertEqual(model.spatial.project.out_features, 128)
        self.assertEqual((model.fusion.in_features, model.fusion.out_features), (323, 256))
        self.assertEqual((model.temporal[0].qkv.in_features, model.temporal[0].qkv.out_features), (256, 768))
        self.assertEqual((model.head.in_features, model.head.out_features), (256, 4))
        self.assertIsNone(model.geometry_norm)
        self.assertIsNone(model.rgb_norm)
        self.assertIsNone(model.spatial_norm)
        self.assertFalse(any("_norm" in name for name in model.state_dict()
                             if name.startswith(("geometry_norm", "rgb_norm", "spatial_norm"))))
        restored = MultimodalClassifier("g-rgb-dditr", PROFILE)
        restored.load_state_dict(model.state_dict(), strict=True)

    def test_factory_rejects_unfrozen_arms_and_variants(self):
        self.assertEqual(set(VARIANTS), {"norm", "temporal-small", "branches-small"})
        with self.assertRaises(ValueError):
            model_spec("g-rgb-dditr", "norm")
        with self.assertRaises(ValueError):
            model_spec("g", "custom")


if __name__ == "__main__":
    unittest.main()
