"""E002a tensor flow, full context, accumulation and memory-path acceptance."""

from copy import deepcopy
from functools import partial
import os
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from pedestrian_behavior.accumulation import train_accumulated_epoch, warmup_lr
from pedestrian_behavior.data.multimodal import LogicalBatches, collate_multimodal, pack_tracks
from pedestrian_behavior.evaluation import track_cross_entropy
from pedestrian_behavior.experiments.e002a_run import validate_profile
from pedestrian_behavior.experiments.e002a_camera_cache import CameraReuse
from pedestrian_behavior.experiments.e002a_sweep import attempts, attempt_command
from pedestrian_behavior.models.multimodal import MultimodalClassifier, frame_chunks, rotary
from pedestrian_behavior.training import seed_run


PROFILE = {"max_tracks": 8, "temporal_cells": 100000, "microbatch_tokens": 10000,
           "ordinary_tokens": 1000, "frame_chunk": 2, "frame_tokens": 1000}


def sample(n=3, name="track", points=True):
    geometry, rgb = torch.randn(n, 20), torch.randn(n, 768)
    flags = torch.ones(n, 3)
    flags[1::2] = 0
    frames = [torch.randn(7 + i, 68) if flags[i, 2] and points else torch.zeros(0, 68) for i in range(n)]
    if not points:
        flags[:, 2] = 0
    return {"geometry": geometry, "rgb": rgb, "flags": flags, "points": frames,
            "targets": torch.arange(n) % 4, "gt_valid": torch.ones(n, dtype=torch.bool),
            "locator": name, "archive": name + ".npz"}


class Samples:
    def __init__(self, values):
        self.values = values
        self.entries = [{"slots": len(v["targets"])} for v in values]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def cost(self, i):
        return sum(len(f) for f in self.values[i]["points"])

    def release_encoder(self):
        pass


def memory_equivalence(device, concurrency=2):
    seed_run(8)
    batch = collate_multimodal([sample(3), sample(2, "second")])
    model = MultimodalClassifier("g-rgb-dditr", PROFILE | {"frame_chunk": concurrency}).to(device)
    initial = deepcopy(model.state_dict())
    reference, errors = None, {}
    for mode in ("ordinary", "recompute", "offload"):
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
        seed_run(27)
        logits = model(batch, mode=mode)
        losses = track_cross_entropy(logits, batch["targets"].to(device), batch["gt_valid"].to(device), batch["padding_mask"].to(device))
        losses.mean().backward()
        gradients = {k: p.grad.detach().cpu().clone() if p.grad is not None else None for k, p in model.named_parameters()}
        optimizer.step()
        item = (logits.detach().cpu(), gradients, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if reference is None:
            reference = item
        torch.testing.assert_close(item[0], reference[0], rtol=1e-4, atol=1e-5)
        maximum = 0.
        for actual, expected in zip(item[1:], reference[1:]):
            for k, value in actual.items():
                if value is None or expected[k] is None:
                    if value is not expected[k]:
                        raise AssertionError("Gradient presence changed")
                else:
                    torch.testing.assert_close(value, expected[k], rtol=1e-4, atol=1e-5, msg=f"{mode}/{k}")
                    maximum = max(maximum, float((value - expected[k]).abs().max()))
        errors[mode] = maximum
    return errors


class MultimodalTest(unittest.TestCase):
    def setUp(self):
        previous = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous)
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, previous_determinism)
        seed_run(0)

    def test_shapes_missing_disabled_branches_shared_initialization_and_no_gt_inputs(self):
        batch = collate_multimodal([sample(3), sample(2, "b")])
        state = None
        for arm in ("g", "g-rgb", "g-rgb-dditr"):
            seed_run(0)
            model = MultimodalClassifier(arm, PROFILE).eval()
            if state is None:
                state = deepcopy(model.state_dict())
            for k, v in model.state_dict().items():
                torch.testing.assert_close(v, state[k], rtol=0, atol=0)
            fused = []
            handle = model.fusion.register_forward_pre_hook(lambda module, args: fused.append(args[0].detach()))
            logits = model(batch)
            self.assertEqual(logits.shape, (2, 3, 4))
            self.assertTrue((logits[1, 2:] == 0).all())
            self.assertTrue((fused[0][0, 1, :320] == 0).all())
            if arm == "g":
                self.assertTrue((fused[0][..., 64:320] == 0).all())
                self.assertFalse(model.rgb.weight.requires_grad)
            if arm != "g-rgb-dditr":
                self.assertTrue((fused[0][..., 192:320] == 0).all())
                self.assertFalse(model.spatial.seed.requires_grad)
            changed = batch | {"targets": batch["targets"] + 99, "gt_valid": ~batch["gt_valid"]}
            torch.testing.assert_close(logits, model(changed), rtol=0, atol=0)
            handle.remove()

    def test_padding_future_context_and_internal_missing_slots(self):
        model = MultimodalClassifier("g-rgb", PROFILE).eval()
        a, b = sample(3, points=False), sample(5, "long", points=False)
        solo, padded = collate_multimodal([a]), collate_multimodal([a, b])
        for k in ("geometry", "rgb", "flags"):
            padded[k][0, 3:] = float("nan")
        torch.testing.assert_close(model(solo)[0], model(padded)[0, :3], rtol=1e-5, atol=1e-6)
        before = model(solo).detach()
        solo["geometry"][0, -1] += 5
        self.assertGreater(float((before[0, 0] - model(solo)[0, 0]).abs().max()), 1e-5)
        self.assertTrue(torch.isfinite(model(solo)).all())
        self.assertEqual(model.temporal[0].qkv.out_features, 768)
        self.assertIsNone(model.spatial.q.bias)

    def test_rotary_and_chunk_boundaries_do_not_remove_tokens(self):
        x = torch.randn(2, 4, 5, 64)
        torch.testing.assert_close(rotary(x).norm(dim=-1), x.norm(dim=-1))
        frames = [torch.zeros(n, 68) for n in (0, 5, 6, 0, 3)]
        chunks = list(frame_chunks(frames, 3, 10))
        self.assertEqual([[i for i, _ in c] for c in chunks], [[1], [2], [4]])
        with self.assertRaises(ValueError):
            list(frame_chunks(frames, 3, 5))

    def test_memory_paths_preserve_dropout_outputs_all_gradients_and_update(self):
        for concurrency in (1, 2, 4):
            self.assertLess(max(memory_equivalence("cpu", concurrency).values()), 1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_memory_paths_preserve_dropout_outputs_gradients_and_update(self):
        for concurrency in (1, 2, 4):
            self.assertLess(max(memory_equivalence("cuda", concurrency).values()), 1e-4)

    def test_accumulation_unequal_physical_and_final_partial_updates(self):
        data = Samples([sample(3, str(i), points=False) for i in range(7)])
        a = MultimodalClassifier("g", PROFILE)
        b = deepcopy(a)
        for model in (a, b):
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.
            # Functional attention dropout follows training; eval leaves gradients enabled.
            model.train = lambda mode=True, model=model: torch.nn.Module.train(model, False)
            model.eval()
        oa, ob = (torch.optim.AdamW(m.parameters(), lr=.0003, weight_decay=.0001) for m in (a, b))
        limits = PROFILE | {"max_tracks": 3}
        loader = LogicalBatches(data, limits, training=True)
        indices = list(next(iter(LogicalBatches(data, limits, training=True))))
        reference = collate_multimodal([data[i] for i in indices])
        losses = track_cross_entropy(b(reference), reference["targets"], reference["gt_valid"], reference["padding_mask"])
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_(b.parameters(), 1.)
        ob.step()
        result = train_accumulated_epoch(a, loader, oa, torch.device("cpu"), 1., peak_lr=.0003, warmup_steps=1)
        self.assertEqual((result["optimizer_steps"], result["processed_tracks"], result["physical_microbatches"]), (1, 7, 3))
        self.assertAlmostEqual(result["training_ce"], float(losses.detach().double().mean()), places=6)
        for name, p in a.named_parameters():
            torch.testing.assert_close(p, dict(b.named_parameters())[name], rtol=1e-4, atol=1e-5, msg=name)
        long = Samples([sample(1, str(i), points=False) for i in range(65)])
        logical = LogicalBatches(long, PROFILE, training=True)
        self.assertEqual([len(i) for i in logical], [64, 1])

    def test_packing_logical_order_warmup_and_bad_profile(self):
        data = Samples([sample(1, str(i), points=False) for i in range(65)])
        a, b = (LogicalBatches(data, PROFILE, training=True) for _ in range(2))
        first = list(a)
        self.assertEqual(first, list(b))
        self.assertNotEqual(first, list(a))
        groups = pack_tracks([0, 1, 2], {0: 3, 1: 5, 2: 1}, {0: 2, 1: None, 2: 3}, PROFILE)
        self.assertEqual(groups, [[1], [0, 2]])
        for n, expected in ((1, .0003 / 402), (402, .0003), (403, .0003)):
            self.assertAlmostEqual(warmup_lr(.0003, n, 402), expected, places=16)
        with self.assertRaises(ValueError):
            validate_profile(PROFILE | {"max_tracks": 0})
        with self.assertRaises(ValueError):
            pack_tracks([0], {0: 1000}, {0: None}, PROFILE)

    def test_nonfinite_empty_gt_and_bad_point_masks_fail(self):
        model = MultimodalClassifier("g-rgb-dditr", PROFILE)
        batch = collate_multimodal([sample(3)])
        batch["flags"][0, 0, 2] = 0
        with self.assertRaisesRegex(ValueError, "validity"):
            model(batch)
        batch["flags"][0, 0, 2] = 1
        batch["points"][0][0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "spatial"):
            model(batch)

    def test_preparation_image_reuse_keeps_sources_and_bounds_ram(self):
        from PIL import Image
        tracks = {"a": {"group": "g", "source_keys": ["1", "2"], "track_id": "a"},
                  "b": {"group": "g", "source_keys": ["2", "3"], "track_id": "b"}}
        catalogs = {"loki": SimpleNamespace(tracks=tracks)}
        reader = SimpleNamespace(dataset="loki")
        decoded = []
        def native(reader, group, keys, selected, frozen):
            decoded.append(list(keys))
            for key in keys:
                yield key, Image.new("RGB", (2, 2), int(key)), {t: key + t for t in selected}
        with patch("pedestrian_behavior.experiments.e002a_camera_cache.native_camera", native):
            cache = CameraReuse(catalogs, [("loki", "a"), ("loki", "b")], capacity_bytes=36)
            first = list(cache(reader, "g", ["1", "2"], ["a"], {}))
            second = list(cache(reader, "g", ["2", "3"], ["b"], {}))
            self.assertEqual(decoded, [["1", "2", "3"]])
            self.assertEqual(first[1][2], {"a": "2a"})
            self.assertEqual(second[0][2], {"b": "2b"})
            self.assertLessEqual(cache.snapshot()["peak_decoded_bytes"], 36)

    def test_six_attempt_launcher_names_and_limits(self):
        plan = attempts()
        self.assertEqual(len(plan), 6)
        self.assertEqual(len({t["name"] for t in plan}), 6)
        command = attempt_command(plan[0], Path("outputs/new"), "cuda", ["--profile", "profile.json"])
        self.assertIn("e002a-road-waymo-g-s0", " ".join(command))
        self.assertEqual({t["seed"] for t in plan}, {0})
        from pedestrian_behavior.experiments.e002a_run import run
        with self.assertRaisesRegex(ValueError, "rgb-preparation"):
            run(SimpleNamespace(smoke=False, rgb_preparation=None))

    def test_rgb_cache_readback_support_excludes_unlabeled_context(self):
        from pedestrian_behavior.experiments.e002a_rgb_audit import support
        row = support(np.array([0, 1, -100]), np.array([True, True, False]), np.array([True, False, True]))
        self.assertEqual(row["class_gt_frames"], [1, 1, 0, 0])
        self.assertEqual(row["rgb_class_gt_frames"], [1, 0, 0, 0])
        self.assertEqual((row["rgb_slots"], row["rgb_gt_frames"], row["rgb_tracks"]), (2, 1, 1))

    def test_paired_summary_uses_saved_predictions_and_checks_availability(self):
        from pedestrian_behavior.evaluation import save_predictions
        from pedestrian_behavior.experiments.e002a_summary import summarize
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for row in attempts():
                run = root / row["name"]
                destination = run / (row["source"] + "-validation")
                destination.mkdir(parents=True)
                (run / "status.json").write_text(json.dumps({"status": "complete"}))
                track = {"locator": "track", "archive": "reference", "group": "group",
                         "targets": np.array([0, 1]), "gt_valid": np.ones(2, bool),
                         "logits": np.array([[2, 0, 0, 0], [0, 2, 0, 0]], np.float32),
                         "conditions": {"observed": np.zeros(2, np.int64)},
                         "source_flags": {"rgb_valid": np.ones(2, bool), "lidar_valid": np.ones(2, bool),
                                          "point_tokens": np.ones(2, np.int64)}}
                save_predictions(destination / "predictions.npz", [track])
            result = summarize(root)
            self.assertTrue((root / "validation-comparison.svg").is_file())
            pairs = result["sources"]["loki"]["paired_track_changes"]
            self.assertEqual(len(pairs), 3)
            self.assertTrue(all(p["accuracy_change"] == 0 for p in pairs))
            self.assertIn(("g-rgb-dditr", "g-rgb"), {(p["arm"], p["reference_arm"]) for p in pairs})

    def test_shared_attempt_multimodal_validation_checkpoint_and_logging_failure(self):
        from test_e001b_training import fixture, FakeRun
        from pedestrian_behavior.experiments import e002a_run
        from pedestrian_behavior.evaluation import load_predictions, prediction_metrics
        from pedestrian_behavior.inspection.pretrained import DINO_PTH_SHA256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root / "experiments")
            probe = root / "probe"
            probe.mkdir()
            (probe / "probe.json").write_text("{}")
            config = {"rgb_checkpoint_sha256": DINO_PTH_SHA256}
            for key in ("dino_weights", "ditr_weights", "ditr_config"):
                path = probe / key
                path.write_text("fixture-only asset")
                config[key] = str(path)
            profile = root / "profile.json"
            profile.write_text(json.dumps({"status": "complete", "device": "cpu", "torch": str(torch.__version__), "cuda": torch.version.cuda,
                                           "code": e002a_run.resource_code(), "profile": PROFILE}))
            setup = json.loads((root / "experiments/E001/data-setup/loki/setup.json").read_text())
            selection = root / "selection.json"
            selection.write_text(json.dumps({"tracks": [{"dataset": "loki", "locator": t["locator"]} for t in setup["tracks"]
                                                          if not t["exclusions"] and t["split"] in ("training", "validation")]}))
            def rgb(supply, dataset, key):
                n = supply.catalogs[dataset].tracks[key]["slots"]
                return {"cls": np.ones((n, 768), np.float32), "valid": np.ones(n, bool)}
            def points(supply, dataset, key):
                for _ in range(supply.catalogs[dataset].tracks[key]["slots"]):
                    yield {"features": np.ones((2, 64), np.float32), "relative_xyz": np.ones((2, 3), np.float32), "inside_box": np.ones(2, bool)}
            for fail in (None, "final"):
                events = []
                args = SimpleNamespace(source="loki", arm="g-rgb-dditr", device="cpu", smoke=True,
                    outputs_root=root, output=root / "experiments/E002a" / ("failed" if fail else "complete"),
                    probe=probe, profile=profile, smoke_selection=selection,
                    rgb_cache_root=root / "experiments/E002a/rgb", point_cache_root=root / "experiments/E002a/points",
                    point_cache_gb=450, min_free_gb=0)
                with patch.object(e002a_run, "load_configuration", return_value=(config, root / "experiments/fixture-transform.json")), \
                     patch.object(e002a_run.FeatureSupply, "rgb", rgb), patch.object(e002a_run.FeatureSupply, "iter_points", points), \
                     patch.object(e002a_run.FeatureSupply, "point_cost", return_value={"tokens": 2, "inside_box_tokens": 2}), \
                     patch("pedestrian_behavior.experiments.run.wandb.init", return_value=FakeRun(events, fail)):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "logging"):
                            e002a_run.run(args)
                    else:
                        e002a_run.run(args)
                status = json.loads((args.output / "status.json").read_text())
                self.assertEqual(status["status"], "failed" if fail else "complete")
                self.assertFalse(any(args.output.glob("*-test")))
                checkpoint = torch.load(args.output / "best.pt", weights_only=True)
                self.assertEqual(checkpoint["config"]["experiment"], "E002a")
                self.assertFalse(any("backbone" in k for k in checkpoint["model_state"]))
                destination = args.output / "loki-validation-smoke"
                actual = prediction_metrics(load_predictions(destination / "predictions.npz"), 4)
                self.assertEqual(actual, json.loads((destination / "metrics.json").read_text())["primary"])
                step = next(e for e in events if "train/loss_step" in e)
                self.assertIn("train/learning_rate", step)
                self.assertEqual(json.loads((args.output / "protected.json").read_text())["changed"], [])


if __name__ == "__main__":
    unittest.main()
