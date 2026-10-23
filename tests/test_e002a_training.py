"""E002a tensor flow, full context, accumulation and memory-path acceptance."""

from copy import deepcopy
from functools import partial
import os
import json
from pathlib import Path
import tempfile
import unittest
import threading
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
from pedestrian_behavior.models.multimodal import MultimodalClassifier, _all_finite_point_tokens, frame_chunks, rotary
from pedestrian_behavior.training import seed_run


PROFILE = {"max_tracks": 8, "temporal_cells": 100000, "microbatch_tokens": 10000,
           "ordinary_tokens": 1000, "frame_chunk": 2, "frame_tokens": 1000}


def sample(n=3, name="track", points=True, arm="g-rgb-dditr"):
    geometry, rgb = torch.randn(n, 20), torch.randn(n, 768)
    flags = torch.ones(n, 3)
    flags[1::2] = 0
    frames = [torch.randn(7 + i, 68) if flags[i, 2] and points else torch.zeros(0, 68) for i in range(n)]
    if not points:
        flags[:, 2] = 0
    result = {"geometry": geometry, "flags": flags[:, :{"g": 1, "g-rgb": 2, "g-rgb-dditr": 3}[arm]],
            "targets": torch.arange(n) % 4, "gt_valid": torch.ones(n, dtype=torch.bool),
            "locator": name, "archive": name + ".npz"}
    if arm != "g":
        result["rgb"] = rgb
    if arm == "g-rgb-dditr":
        result["points"] = frames
    return result


class Samples:
    def __init__(self, values):
        self.values = values
        self.entries = [{"slots": len(v["targets"])} for v in values]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def cost(self, i):
        return sum(len(f) for f in self.values[i].get("points", ()))

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

    def _prefetch_equivalence(self, device, *, recompute=False):
        data = Samples([sample(3+i%2, str(i)) for i in range(5)])
        profile = PROFILE | {"max_tracks": 2, "ordinary_tokens": 1 if recompute else PROFILE["ordinary_tokens"]}
        seed_run(4)
        initial = MultimodalClassifier("g-rgb-dditr", profile).state_dict()
        results = []
        for enabled in (False, True):
            rows, outputs = [], []
            loader = LogicalBatches(data, profile, training=True, seed=0, prefetch=enabled,
                on_pack=lambda row: rows.append(row["physical_indices"]))
            model = MultimodalClassifier("g-rgb-dditr", profile).to(device)
            model.load_state_dict(initial)
            hook = model.register_forward_hook(lambda m, args, out: outputs.append(out.detach().cpu()))
            optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
            seed_run(9)
            result = train_accumulated_epoch(model, loader, optimizer, torch.device(device), 1., peak_lr=.0003, warmup_steps=1)
            hook.remove()
            results.append((rows, outputs, result["training_ce"],
                {k: p.grad.detach().cpu().clone() for k,p in model.named_parameters()},
                {k: p.detach().cpu().clone() for k,p in model.state_dict().items()}))
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[0][2], results[1][2])
        for a,b in zip(results[0][1], results[1][1]): torch.testing.assert_close(a,b,rtol=0,atol=0)
        for i in (3,4):
            for key in results[0][i]:
                torch.testing.assert_close(results[0][i][key],results[1][i][key],rtol=0,atol=0)

    def test_prefetch_preserves_dropout_predictions_gradients_updates_and_order(self):
        self._prefetch_equivalence("cpu")
        self._prefetch_equivalence("cpu", recompute=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_prefetch_preserves_dropout_predictions_gradients_updates_and_order(self):
        self._prefetch_equivalence("cuda")
        self._prefetch_equivalence("cuda", recompute=True)

    def test_prefetch_is_one_batch_ahead_joins_and_propagates_producer_failure(self):
        data = Samples([sample(3, str(i)) for i in range(4)])
        started, release = threading.Event(), threading.Event()
        reads = []
        class Blocking(Samples):
            def __getitem__(self, i):
                reads.append(i)
                if i==1:
                    started.set()
                    if not release.wait(5): raise RuntimeError("test producer timeout")
                return super().__getitem__(i)
        loader = LogicalBatches(Blocking(data.values), PROFILE | {"max_tracks": 1}, prefetch=True)
        iterator = loader.microbatches([0,1,2,3])
        self.assertEqual(next(iterator)["locators"], ["0"])
        self.assertTrue(started.wait(5))
        self.assertEqual(reads, [0,1])
        release.set(); iterator.close()
        self.assertFalse(any(t.name.startswith("prepared-input") for t in threading.enumerate()))
        class Broken(Samples):
            def __getitem__(self, i):
                if i==1: raise ValueError("producer failed")
                return super().__getitem__(i)
        loader = LogicalBatches(Broken(data.values), PROFILE | {"max_tracks": 1}, prefetch=True)
        with self.assertRaisesRegex(ValueError,"producer failed"):
            list(loader.microbatches([0,1,2]))
        self.assertFalse(any(t.name.startswith("prepared-input") for t in threading.enumerate()))

    def test_model_failure_closes_prefetch_before_attempt_resources(self):
        data = Samples([sample(3, str(i)) for i in range(3)])
        model = MultimodalClassifier("g-rgb-dditr", PROFILE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
        model.forward = lambda batch: (_ for _ in ()).throw(ValueError("model failed"))
        loader = LogicalBatches(data, PROFILE | {"max_tracks":1}, prefetch=True)
        with self.assertRaisesRegex(ValueError,"model failed"):
            train_accumulated_epoch(model, loader, optimizer, torch.device("cpu"), 1., peak_lr=.0003, warmup_steps=1)
        self.assertFalse(any(t.name.startswith("prepared-input") for t in threading.enumerate()))

    def test_enabled_only_fusion_shapes_missingness_checkpoint_and_no_gt_inputs(self):
        from pedestrian_behavior.experiments.e002a_run import model_spec
        for arm, width, count in (("g", 65, 1), ("g-rgb", 194, 2), ("g-rgb-dditr", 323, 3)):
            batch = collate_multimodal([sample(3, arm=arm), sample(2, "b", arm=arm)])
            seed_run(0)
            model = MultimodalClassifier(arm, PROFILE).eval()
            self.assertEqual(model.fusion.in_features, width)
            self.assertEqual(model_spec(arm)["fusion"], [width, 256])
            self.assertEqual(len(model_spec(arm)["availability"]), count)
            fused = []
            handle = model.fusion.register_forward_pre_hook(lambda module, args: fused.append(args[0].detach()))
            logits = model(batch)
            self.assertEqual(logits.shape, (2, 3, 4))
            self.assertTrue((logits[1, 2:] == 0).all())
            self.assertEqual(fused[0].shape[-1], width)
            self.assertTrue((fused[0][0, 1, :width - count] == 0).all())
            torch.testing.assert_close(fused[0][..., -count:], batch["flags"])
            self.assertEqual(model.rgb is None, arm == "g")
            self.assertEqual(model.spatial is None, arm != "g-rgb-dditr")
            absent = {key: object() for key in ("rgb", "points") if key not in batch}
            changed = batch | {"targets": batch["targets"] + 99, "gt_valid": ~batch["gt_valid"]}
            torch.testing.assert_close(logits, model(changed | absent), rtol=0, atol=0)
            restored = MultimodalClassifier(arm, PROFILE).eval()
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(logits, restored(batch), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "shapes"):
                model(batch | {"flags": torch.ones(2, 3, count + 1)})
            handle.remove()

    def test_padding_future_context_and_internal_missing_slots(self):
        model = MultimodalClassifier("g-rgb", PROFILE).eval()
        a, b = sample(3, arm="g-rgb"), sample(5, "long", arm="g-rgb")
        solo, padded = collate_multimodal([a]), collate_multimodal([a, b])
        for k in ("geometry", "rgb", "flags"):
            padded[k][0, 3:] = float("nan")
        torch.testing.assert_close(model(solo)[0], model(padded)[0, :3], rtol=1e-5, atol=1e-6)
        before = model(solo).detach()
        solo["geometry"][0, -1] += 5
        self.assertGreater(float((before[0, 0] - model(solo)[0, 0]).abs().max()), 1e-5)
        self.assertTrue(torch.isfinite(model(solo)).all())
        self.assertEqual(model.temporal[0].qkv.out_features, 768)
        self.assertIsNone(model.spatial)

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
    def test_cuda_all_arm_widths_forward_and_backward(self):
        for arm in ("g", "g-rgb", "g-rgb-dditr"):
            batch = collate_multimodal([sample(3, arm=arm), sample(2, "second", arm=arm)])
            model = MultimodalClassifier(arm, PROFILE).eval()
            expected = model(batch).detach()
            model.cuda()
            actual = model(batch)
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)
            loss = track_cross_entropy(actual, batch["targets"].cuda(), batch["gt_valid"].cuda(), batch["padding_mask"].cuda()).mean()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_memory_paths_preserve_dropout_outputs_gradients_and_update(self):
        for concurrency in (1, 2, 4):
            self.assertLess(max(memory_equivalence("cuda", concurrency).values()), 1e-4)

    def test_accumulation_unequal_physical_and_final_partial_updates(self):
        data = Samples([sample(3, str(i), arm="g") for i in range(7)])
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

    def test_cpu_finite_check_accepts_lazy_negative_view(self):
        value = torch._neg_view(torch.ones(3, 68))
        self.assertTrue(value.is_neg())
        self.assertTrue(_all_finite_point_tokens(value))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_nonfinite_cuda_point_tokens_still_fail(self):
        model = MultimodalClassifier("g-rgb-dditr", PROFILE).cuda()
        batch = collate_multimodal([sample(3)])
        batch["points"][0] = batch["points"][0].cuda()
        batch["points"][0][0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "spatial"):
            model(batch)

    def test_dataset_never_requests_disabled_features_or_point_costs(self):
        from pedestrian_behavior.experiments.e002a_run import FeatureDataset
        from unittest.mock import Mock
        value = sample(3, arm="g")
        value["inputs"] = value.pop("geometry")
        value["inputs"][:, [8, 9, 10, 11, 18, 19]] = 0
        value["inputs"][0, 8] = 1
        value["inputs"][2, 18] = 1
        class Baseline:
            entries = [{"locator": "track", "slots": 3}]
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return deepcopy(value)
        for arm in ("g", "g-rgb"):
            supply = Mock()
            supply.catalogs = {"loki": SimpleNamespace(tracks={"track": Baseline.entries[0]})}
            supply.identity = {"fixture": True}
            supply.point_cost.side_effect = AssertionError("Disabled LiDAR cost read")
            supply.iter_points.side_effect = AssertionError("Disabled LiDAR feature read")
            supply.rgb.return_value = {"cls": np.ones((3, 768), np.float32), "valid": np.array([True, False, True])}
            if arm == "g":
                supply.rgb.side_effect = AssertionError("Disabled RGB feature read")
            dataset = FeatureDataset(Baseline(), supply if arm != "g" else None, "loki", arm)
            result = dataset[0]
            self.assertEqual(dataset.cost(0), 0)
            self.assertNotIn("points", result)
            self.assertEqual("rgb" in result, arm != "g")
            torch.testing.assert_close(result["flags"][:, 0], torch.tensor([1., 0., 1.]))
            self.assertEqual(set(result["prediction_metadata"]["source_flags"]), {"rgb_valid"} if arm != "g" else set())
            self.assertFalse(supply.point_cost.called or supply.iter_points.called)
            self.assertEqual(supply.rgb.call_count, int(arm != "g"))
            batch = collate_multimodal([result])
            self.assertTrue(torch.isfinite(MultimodalClassifier(arm, PROFILE)(batch)).all())
        with self.assertRaisesRegex(ValueError, "Mixed modality"):
            collate_multimodal([sample(2, arm="g"), sample(2, arm="g-rgb")])

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
        for row in plan:
            command = attempt_command(row, Path("outputs/new"), "cuda", [], indexed_root=Path("outputs/prepared"))
            self.assertEqual("--indexed-point-cache" in command, row["arm"] == "g-rgb-dditr")
            if row["arm"] == "g-rgb-dditr":
                self.assertIn("outputs/prepared/indexed-" + row["source"], command)
            self.assertNotIn("--point-cache-gb", command)
        from pedestrian_behavior.experiments.e002a_run import run
        with self.assertRaisesRegex(ValueError, "rgb-preparation"):
            run(SimpleNamespace(smoke=False, arm="g-rgb", rgb_preparation=None))

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
                from pedestrian_behavior.experiments.e002a_run import FUSION_PROTOCOL
                (run / "config.json").write_text(json.dumps({"fusion_protocol": FUSION_PROTOCOL}))
                track = {"locator": "track", "archive": "reference", "group": "group",
                         "targets": np.array([0, 1]), "gt_valid": np.ones(2, bool),
                         "logits": np.array([[2, 0, 0, 0], [0, 2, 0, 0]], np.float32),
                         "conditions": {"observed": np.zeros(2, np.int64)},
                         "source_flags": {}}
                if row["arm"] != "g":
                    track["source_flags"]["rgb_valid"] = np.ones(2, bool)
                if row["arm"] == "g-rgb-dditr":
                    track["source_flags"].update(lidar_valid=np.ones(2, bool), point_tokens=np.ones(2, np.int64))
                save_predictions(destination / "predictions.npz", [track])
            result = summarize(root)
            self.assertTrue((root / "validation-comparison.svg").is_file())
            pairs = result["sources"]["loki"]["paired_track_changes"]
            self.assertEqual(len(pairs), 3)
            self.assertTrue(all(p["accuracy_change"] == 0 for p in pairs))
            self.assertIn(("g-rgb-dditr", "g-rgb"), {(p["arm"], p["reference_arm"]) for p in pairs})
            self.assertEqual(result["fusion_protocol"], FUSION_PROTOCOL)
            point_config = root / "e002a-road-waymo-g-rgb-dditr-s0/config.json"
            original_config = point_config.read_text()
            point_config.write_text(json.dumps({"fusion_protocol": FUSION_PROTOCOL, "point_storage_protocol": "pooled"}))
            with self.assertRaisesRegex(ValueError, "point storage protocols"):
                summarize(root)
            point_config.write_text(original_config)
            destination = root / "e002a-loki-g-rgb-dditr-s0/loki-validation/predictions.npz"
            from pedestrian_behavior.evaluation import load_predictions
            tracks = load_predictions(destination)
            tracks[0]["source_flags"]["rgb_valid"][0] = False
            save_predictions(destination, tracks)
            with self.assertRaisesRegex(ValueError, "availability differs"):
                summarize(root)
            config = root / "e002a-road-waymo-g-rgb-s0/config.json"
            config.write_text("{}")
            with self.assertRaisesRegex(ValueError, "fusion protocols"):
                summarize(root)
            for row in attempts():
                run = root / row["name"]
                (run / "config.json").write_text("{}")
                destination = run / (row["source"] + "-validation/predictions.npz")
                tracks = load_predictions(destination)
                tracks[0]["source_flags"] = {"rgb_valid": np.ones(2, bool), "lidar_valid": np.ones(2, bool),
                                             "point_tokens": np.ones(2, np.int64)}
                save_predictions(destination, tracks)
            self.assertEqual(summarize(root)["fusion_protocol"], "shared-availability-v1")

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
            from test_prepared_points import prepared_fixture
            from pedestrian_behavior.experiments.e002a_features import FeatureCatalog, extraction_identity
            from pedestrian_behavior.data.feature_cache import FeatureCache
            catalog = FeatureCatalog(root, "loki", config, root / "experiments/fixture-transform.json")
            points_root = prepared_fixture(root / "experiments/E002a/indexed-loki", catalog)
            rgb_root = root / "experiments/E002a/prepared-rgb"
            with FeatureCache(rgb_root, min_free_bytes=0) as cache:
                cache.get({"fixture": True}, lambda: {"values": np.ones(1, np.float32)})
            rgb_preparation = root / "rgb-preparation.json"
            rgb_preparation.write_text(json.dumps({"status": "complete", "extraction": extraction_identity("cpu", config),
                "config": {"rgb_cache_root": str(rgb_root)},
                "tracks": [{"dataset": "loki", "locator": k} for k in catalog.tracks]}))
            profile = root / "profile.json"
            profile.write_text(json.dumps({"status": "complete", "device": "cpu", "torch": str(torch.__version__), "cuda": torch.version.cuda,
                                           "code": e002a_run.resource_code(), "profile": PROFILE,
                                           "point_storage_protocol": e002a_run.POINT_PROTOCOL}))
            setup = json.loads((root / "experiments/E001/data-setup/loki/setup.json").read_text())
            selection = root / "selection.json"
            selection.write_text(json.dumps({"tracks": [{"dataset": "loki", "locator": t["locator"]} for t in setup["tracks"]
                                                          if not t["exclusions"] and t["split"] in ("training", "validation")]}))
            def rgb(supply, dataset, key):
                n = supply.catalogs[dataset].tracks[key]["slots"]
                return {"cls": np.ones((n, 768), np.float32), "valid": np.ones(n, bool)}
            for arm, fail in (("g", None), ("g-rgb", None), ("g-rgb-dditr", None), ("g-rgb-dditr", "final")):
                events = []
                args = SimpleNamespace(source="loki", arm=arm, device="cpu", smoke=True,
                    outputs_root=root, output=root / "experiments/E002a" / ("failed" if fail else arm),
                    probe=probe, profile=profile, smoke_selection=selection,
                    rgb_cache_root=rgb_root, rgb_preparation=rgb_preparation, indexed_point_cache=points_root)
                with patch.object(e002a_run, "load_configuration", return_value=(config, root / "experiments/fixture-transform.json")) as load_config, \
                     patch.object(e002a_run.FeatureSupply, "rgb", side_effect=rgb, autospec=True) as rgb_read, \
                     patch.object(e002a_run.PreparedSupply, "point_tensors", autospec=True,
                                  side_effect=lambda supply, source, key: supply.points.iter_track(key)) as point_read, \
                     patch.object(e002a_run.PreparedSupply, "track_cost", autospec=True,
                                  side_effect=lambda supply, source, key: supply.points.track_cost(key)) as cost_read, \
                     patch("pedestrian_behavior.experiments.run.wandb.init", return_value=FakeRun(events, fail)):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "logging"):
                            e002a_run.run(args)
                    else:
                        e002a_run.run(args)
                    self.assertEqual(rgb_read.called, arm != "g")
                    self.assertEqual(load_config.called, arm != "g")
                    self.assertEqual(point_read.called, arm == "g-rgb-dditr")
                    self.assertEqual(cost_read.called, arm == "g-rgb-dditr")
                self.assertFalse((root / "experiments/E002a/point-cache").exists())
                status = json.loads((args.output / "status.json").read_text())
                self.assertEqual(status["status"], "failed" if fail else "complete")
                self.assertFalse(any(args.output.glob("*-test")))
                checkpoint = torch.load(args.output / "best.pt", weights_only=True)
                self.assertEqual(checkpoint["config"]["experiment"], "E002a")
                self.assertEqual(checkpoint["config"]["fusion_protocol"], e002a_run.FUSION_PROTOCOL)
                self.assertEqual(checkpoint["config"]["model"], json.loads(json.dumps(e002a_run.model_spec(arm))))
                self.assertFalse(any("backbone" in k for k in checkpoint["model_state"]))
                destination = args.output / "loki-validation-smoke"
                actual = prediction_metrics(load_predictions(destination / "predictions.npz"), 4)
                self.assertEqual(actual, json.loads((destination / "metrics.json").read_text())["primary"])
                tracks = load_predictions(destination / "predictions.npz")
                self.assertEqual("rgb-observed" in tracks[0]["conditions"], arm != "g")
                self.assertEqual("lidar-observed" in tracks[0]["conditions"], arm == "g-rgb-dditr")
                step = next(e for e in events if "train/loss_step" in e)
                self.assertIn("train/learning_rate", step)
                self.assertEqual(json.loads((args.output / "protected.json").read_text())["changed"], [])
                provenance = json.loads((args.output / "provenance.json").read_text())
                self.assertEqual(provenance["runtime"]["encoder_calls"], {})
                self.assertEqual(checkpoint["config"]["feature_supply"], "prepared-read-only-no-inference")
                self.assertEqual(checkpoint["config"]["point_storage_protocol"],
                                 e002a_run.POINT_PROTOCOL if arm == "g-rgb-dditr" else None)
            saved_profile = profile.read_text()
            changed_profile = json.loads(saved_profile)
            changed_profile["profile"]["frame_tokens"] = 1
            profile.write_text(json.dumps(changed_profile))
            args.output = root / "experiments/E002a/oversize-frame"
            with patch.object(e002a_run, "load_configuration", return_value=(config, root / "experiments/fixture-transform.json")), \
                 patch("pedestrian_behavior.experiments.run.wandb.init") as initialize, \
                 self.assertRaisesRegex(ValueError, "token ceiling"):
                e002a_run.run(args)
            initialize.assert_not_called()
            profile.write_text(saved_profile)
            # In-progress cache roots must fail before creating a cloud run or falling back.
            (points_root / "manifest.json").unlink()
            args.output = root / "experiments/E002a/missing-manifest"
            with patch.object(e002a_run, "load_configuration", return_value=(config, root / "experiments/fixture-transform.json")), \
                 patch("pedestrian_behavior.experiments.run.wandb.init") as initialize, \
                 self.assertRaises(FileNotFoundError):
                e002a_run.run(args)
            initialize.assert_not_called()
            self.assertEqual(json.loads((args.output / "status.json").read_text())["status"], "failed")
            self.assertFalse((root / "experiments/E002a/point-cache").exists())


if __name__ == "__main__":
    unittest.main()
