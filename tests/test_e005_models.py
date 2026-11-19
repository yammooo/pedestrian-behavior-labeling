import unittest
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from pedestrian_behavior.models.e005 import ARMS, build_model


PROFILE = {"frame_chunk": 1, "frame_tokens": 1024, "ordinary_tokens": 1}


def tokens(rows, width):
    return torch.randn(rows, width, dtype=torch.float16) if rows else torch.zeros(0, width, dtype=torch.float16)


def batch_for(arm):
    torch.manual_seed(20)
    flags = torch.tensor([
        [1, 1, 1], [1, 0, 0], [1, 1, 0],
        [1, 1, 1], [float("nan"), float("nan"), float("nan")],
        [float("nan"), float("nan"), float("nan")],
    ], dtype=torch.float32)
    flags = flags.reshape(2, 3, 3)
    result = {"geometry": torch.randn(2, 3, 20),
              "flags": flags[..., :2] if arm == "g-rgb" else flags,
              "lengths": torch.tensor([3, 1], dtype=torch.int64)}
    result["geometry"][1, 1:] = float("nan")
    result["rgb_frames"] = [tokens(256, 772), tokens(0, 772), tokens(256, 772),
                            tokens(256, 772), tokens(0, 772), tokens(0, 772)]
    if arm == "g-rgb-utonia":
        result["lidar_levels"] = ([tokens(2, 77), tokens(0, 77), tokens(0, 77), tokens(3, 77), tokens(0, 77), tokens(0, 77)],
                                  [tokens(3, 221), tokens(0, 221), tokens(0, 221), tokens(2, 221), tokens(0, 221), tokens(0, 221)])
    return result


class E005ModelTest(unittest.TestCase):
    def setUp(self):
        old = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, old)

    def test_forward_backward_masks_and_cached_fp16(self):
        for arm in ARMS:
            with self.subTest(arm=arm):
                model = build_model(arm, PROFILE).eval()
                batch = batch_for(arm)
                captured = []
                hook = model.fusion.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
                with patch("pedestrian_behavior.models.e005.checkpoint", wraps=torch_checkpoint) as calls:
                    logits = model(batch)
                    self.assertEqual(calls.call_count, 0)
                hook.remove()
                self.assertEqual(tuple(logits.shape), (2, 3, 4))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.equal(logits[1, 1:], torch.zeros_like(logits[1, 1:])))
                single = {"geometry": batch["geometry"][1:2, :1], "flags": batch["flags"][1:2, :1],
                          "lengths": torch.tensor([1]), "rgb_frames": [batch["rgb_frames"][3]]}
                if model.lidar is not None:
                    single["lidar_levels"] = ([batch["lidar_levels"][0][3]], [batch["lidar_levels"][1][3]])
                self.assertTrue(torch.allclose(logits[1, 0], model(single)[0, 0], atol=1e-6, rtol=1e-6))
                fused = captured[0]
                self.assertTrue(torch.equal(fused[0, 1, 64:192], torch.zeros(128)))
                if arm == "g-rgb-utonia":
                    self.assertTrue(torch.equal(fused[0, 1, 192:320], torch.zeros(128)))
                    self.assertTrue(torch.equal(fused[0, 2, 192:320], torch.zeros(128)))
                model.train()
                with patch("pedestrian_behavior.models.e005.checkpoint", wraps=torch_checkpoint) as calls:
                    model(batch, mode="ordinary")
                    self.assertEqual(calls.call_count, 0)
                    model(batch).square().mean().backward()
                    self.assertGreater(calls.call_count, 0)
                self.assertIsNotNone(model.rgb.seed.grad)
                self.assertGreater(model.rgb.seed.grad.abs().sum().item(), 0.)
                self.assertGreater(model.rgb.project.weight.grad.abs().sum().item(), 0.)
                if model.lidar is not None:
                    self.assertGreater(model.lidar.seed.grad.abs().sum().item(), 0.)
                    self.assertGreater(model.lidar.stage3.weight.grad.abs().sum().item(), 0.)
                    self.assertGreater(model.lidar.stage5.weight.grad.abs().sum().item(), 0.)

    def test_valid_lidar_uses_available_level_tokens(self):
        model = build_model("g-rgb-utonia", PROFILE).eval()
        batch = batch_for("g-rgb-utonia")
        batch["lidar_levels"][0][0] = tokens(0, 77)
        self.assertTrue(torch.isfinite(model(batch)).all())

    def test_recompute_matches_ordinary_outputs_and_gradients(self):
        model = build_model("g-rgb", PROFILE).train()
        batch = batch_for("g-rgb")
        torch.manual_seed(31)
        ordinary = model(batch, mode="ordinary")
        ordinary.square().mean().backward()
        expected = ordinary.detach().clone()
        expected_grads = [parameter.grad.detach().clone() for parameter in model.parameters()]

        model.zero_grad(set_to_none=True)
        torch.manual_seed(31)
        recomputed = model(batch, mode="recompute")
        recomputed.square().mean().backward()
        self.assertTrue(torch.allclose(recomputed, expected, atol=1e-6, rtol=1e-6))
        for actual, expected in zip((p.grad for p in model.parameters()), expected_grads):
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_folded_attention_preserves_full_model_outputs_and_gradients(self):
        arm = "g-rgb-utonia"
        batch = batch_for(arm)
        profile = PROFILE | {"frame_chunk": 4}
        standard = build_model(arm, profile).train()
        folded = build_model(arm, profile | {"fold_single_query": True}).train()
        folded.load_state_dict(standard.state_dict())
        torch.manual_seed(5)
        expected = standard(batch)
        expected.square().mean().backward()
        torch.manual_seed(5)
        actual = folded(batch)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for candidate, reference in zip(folded.parameters(), standard.parameters()):
            torch.testing.assert_close(candidate.grad, reference.grad, atol=2e-6, rtol=2e-5)

    def test_packed_lidar_preserves_outputs_and_gradients(self):
        arm = "g-rgb-utonia"
        batch = batch_for(arm)
        batch["lidar_levels"][0][0] = tokens(0, 77)
        profile = PROFILE | {"frame_chunk": 4, "fold_single_query": True}
        standard = build_model(arm, profile).train()
        packed = build_model(arm, profile | {"pack_lidar_projection": True}).train()
        packed.load_state_dict(standard.state_dict())
        torch.manual_seed(5)
        expected = standard(batch)
        expected.square().mean().backward()
        torch.manual_seed(5)
        actual = packed(batch)
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for candidate, reference in zip(packed.parameters(), standard.parameters()):
            torch.testing.assert_close(candidate.grad, reference.grad, atol=2e-6, rtol=2e-5)

    def test_lidar_stage_tokens_remain_in_stage3_then_stage5_order(self):
        model = build_model("g-rgb-utonia", PROFILE).eval()
        batch = batch_for("g-rgb-utonia")
        batch["flags"] = torch.tensor([[[1., 1., 1.]]])
        batch["geometry"] = batch["geometry"][:1, :1]
        batch["lengths"] = torch.tensor([1], dtype=torch.int64)
        batch["rgb_frames"] = batch["rgb_frames"][:1]
        batch["lidar_levels"] = ([batch["lidar_levels"][0][0]], [batch["lidar_levels"][1][0]])
        expected = model.lidar.project_frames(
            [batch["lidar_levels"][0][0].float()], [batch["lidar_levels"][1][0].float()])[0]
        seen = []
        hook = model.lidar.k.register_forward_pre_hook(lambda _m, args: seen.append(args[0].detach().clone()))
        model(batch)
        hook.remove()
        self.assertEqual(tuple(seen[0].shape), (1, 5, 128))
        self.assertTrue(torch.allclose(seen[0][0], expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
