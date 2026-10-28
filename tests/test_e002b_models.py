"""E002b frozen model variants preserve masks, shapes and checkpoint structure."""

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from pedestrian_behavior.data.multimodal import collate_multimodal
from pedestrian_behavior.experiments.e002b import ARMS, VARIANTS, build_model, model_spec
from pedestrian_behavior.experiments.e002b_independent import attempt_command, attempts
from pedestrian_behavior.experiments import e002b_run
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
        self.assertFalse(any(name.startswith("rgb_branch_dropout") for name in model.state_dict()))
        restored = MultimodalClassifier("g-rgb-dditr", PROFILE)
        restored.load_state_dict(model.state_dict(), strict=True)

    def test_factory_rejects_unfrozen_arms_and_variants(self):
        self.assertEqual(set(VARIANTS), {"norm", "temporal-small", "branches-small"})
        with self.assertRaises(ValueError):
            model_spec("g-rgb-dditr", "norm")
        with self.assertRaises(ValueError):
            model_spec("g", "custom")

    def test_rgb_regularization_changes_only_temporal_dropout(self):
        for variant in ("norm", "temporal-small"):
            for dropout in (.3, .5):
                model = build_model("g-rgb", variant, PROFILE, dropout)
                self.assertEqual(model_spec("g-rgb", variant, dropout)["dropout"], dropout)
                self.assertTrue(all(block.dropout.p == block.ffn[2].p == dropout
                                    for block in model.temporal))
                self.assertIsNone(model.spatial)
                self.assertEqual(tuple(model(batch_for("g-rgb")).shape), (2, 3, 4))

    def test_rgb_branch_dropout_is_time_locked_per_track_and_preserves_state_keys(self):
        batch = batch_for("g-rgb")
        model = build_model("g-rgb", "temporal-small", PROFILE, dropout=0., rgb_branch_dropout=.5).train()
        baseline = build_model("g-rgb", "temporal-small", PROFILE, dropout=0.).train()
        self.assertEqual(set(model.state_dict()), set(baseline.state_dict()))
        baseline.load_state_dict(model.state_dict(), strict=True)

        captured, fused = [], []
        hook = model.rgb_branch_dropout.register_forward_hook(
            lambda _module, inputs, output: captured.append((inputs[0].detach(), output.detach())))
        fusion_hook = model.fusion.register_forward_pre_hook(
            lambda _module, inputs: fused.append(inputs[0].detach()))
        model(batch)
        hook.remove()
        fusion_hook.remove()

        before, after = captured[0]
        lengths = batch["lengths"]
        keep = []
        for track, length in enumerate(lengths.tolist()):
            track_keep = []
            for channel in range(before.shape[1]):
                observed = before[track, channel, :length].abs() > 1e-6
                ratios = after[track, channel, :length][observed] / before[track, channel, :length][observed]
                self.assertGreater(len(ratios), 0)
                self.assertTrue(torch.all((ratios == 0.) | (ratios == 2.)))
                self.assertTrue(torch.all(ratios == ratios[0]))
                track_keep.append(bool(ratios[0]))
            keep.append(torch.tensor(track_keep))
        self.assertTrue((keep[0] != keep[1]).any())

        valid = torch.arange(batch["geometry"].shape[1])[None, :] < lengths[:, None]
        geometry = batch["geometry"].masked_fill(~valid[..., None], 0.)
        expected_geometry = model.geometry_norm(model.geometry(geometry))
        expected_geometry = expected_geometry * batch["flags"][..., :1].masked_fill(~valid[..., None], 0.)
        torch.testing.assert_close(fused[0][..., :64], expected_geometry)
        torch.testing.assert_close(fused[0][..., 64 + 128:],
                                   batch["flags"].masked_fill(~valid[..., None], 0.))

        eval_rows = []
        hook = model.rgb_branch_dropout.register_forward_hook(
            lambda _module, inputs, output: eval_rows.append((inputs[0].detach(), output.detach())))
        model.eval()(batch)
        hook.remove()
        self.assertTrue(torch.equal(eval_rows[0][0], eval_rows[0][1]))

    def test_independent_runner_requires_current_code_and_exact_model_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / "profile.json"
            row = {"status": "complete", "experiment": "E002b", "device": "cpu",
                   "code": e002b_run.resource_code(), "torch": str(torch.__version__),
                   "cuda": torch.version.cuda, "profile": {"max_tracks": 64, "temporal_cells": 4096,
                       "microbatch_tokens": 100, "ordinary_tokens": 100, "frame_chunk": 1,
                       "frame_tokens": 10},
                   "model_specs": {"g/norm": model_spec("g", "norm")}, "model_coverage": ["g/norm"]}
            profile_path.write_text(json.dumps(row))
            args = SimpleNamespace(profile=profile_path, arm="g", model_variant="norm", device="cpu")
            self.assertEqual(e002b_run._profile_for(args)["model_coverage"], ["g/norm"])
            args = SimpleNamespace(profile=profile_path, arm="g-rgb", model_variant="temporal-small",
                                   device="cpu", rgb_branch_dropout=.5)
            row["model_specs"] = {"g-rgb/temporal-small": model_spec("g-rgb", "temporal-small")}
            row["model_coverage"] = ["g-rgb/temporal-small"]
            profile_path.write_text(json.dumps(row))
            self.assertEqual(e002b_run._profile_for(args)["model_coverage"], ["g-rgb/temporal-small"])
            row["code"] = {"stale": "hash"}
            profile_path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "code changed"):
                e002b_run._profile_for(args)

    def test_e002b_validator_confines_outputs_to_new_experiment_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(outputs_root=root, output=root / "experiments/E002b/runs/attempt",
                                   rgb_cache_root=root / "experiments/E002a/rgb-cache",
                                   point_cache_root=root / "experiments/E002a/indexed-loki")
            e002b_run.validate_roots(args, {})
            args.rgb_branch_dropout = .3
            args.output = root / "experiments/E002c/runs/attempt"
            e002b_run.validate_roots(args, {})
            args.output = root / "experiments/E002b/runs/attempt"
            with self.assertRaisesRegex(ValueError, "E002c evidence"):
                e002b_run.validate_roots(args, {})
            del args.rgb_branch_dropout
            args.output = root / "experiments/E002a/runs/attempt"
            with self.assertRaisesRegex(ValueError, "E002b evidence"):
                e002b_run.validate_roots(args, {})

    def test_independent_launcher_matrix_and_command_keep_lidar_grouped(self):
        plan = attempts("road-waymo", e002b_run.ARMS, tuple(VARIANTS))
        self.assertEqual(len(plan), 6)
        self.assertEqual({row["arm"] for row in plan}, {"g", "g-rgb"})
        self.assertEqual({row["variant"] for row in plan}, set(VARIANTS))
        command = attempt_command(plan[0], Path("outputs/experiments/E002b/runs/pilot"), "cuda", ["--profile", "profile.json"])
        self.assertIn(plan[0]["arm"], command)
        self.assertIn(plan[0]["variant"], command)
        self.assertIn("scripts/train-e002b.py", command)
        regularization = attempts("loki", ["g-rgb"], ["norm", "temporal-small"], (.3, .5))
        self.assertEqual(len(regularization), 4)
        self.assertEqual(len({row["name"] for row in regularization}), 4)
        self.assertEqual({row["dropout"] for row in regularization}, {.3, .5})
        self.assertIn("0.5", attempt_command(regularization[-1], Path("outputs/experiments/E002b/runs/probe"),
                                               "cuda", []))


if __name__ == "__main__":
    unittest.main()
