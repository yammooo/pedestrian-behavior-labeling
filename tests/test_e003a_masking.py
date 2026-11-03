"""E003a deterministic masking, branch gating and input-only packing."""

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pedestrian_behavior.data.multimodal import collate_multimodal
from pedestrian_behavior.data.multimodal import LogicalBatches
from pedestrian_behavior.experiments.e002a_features import FeatureSupply
from pedestrian_behavior.experiments.e002b import build_model
from pedestrian_behavior.experiments.e003a import (BRANCHES, FeatureDataset, MaskSampler,
    _empty_gaps, _rgb_identity, _rgb_validity, mask_seed)


class FakeBaseline:
    def __init__(self, natural, targets, gt_valid):
        self.entries = [{"locator": '["scene","track"]', "archive": "track.npz", "slots": len(targets)}]
        self.collection, self.extension = Path("/unused"), Path("/unused")
        self.configuration = "K+T+R"
        self.inputs = torch.zeros(len(targets), 20)
        for index, valid in enumerate(natural[:, 0]):
            if valid:
                self.inputs[index, 8] = 1.
        self.targets, self.gt_valid = torch.tensor(targets), torch.tensor(gt_valid)
        self.reads = 0

    def __len__(self):
        return 1

    def __getitem__(self, index):
        self.reads += 1
        return {"inputs": self.inputs.clone(), "targets": self.targets.clone(),
                "gt_valid": self.gt_valid.clone(), "locator": self.entries[index]["locator"],
                "archive": "/unused/track.npz"}


class FakePoints:
    def __init__(self, counts):
        self.counts = {"[\"scene\",\"track\"]": np.asarray(counts, dtype=np.int64)}
        self.reads, self.masks = 0, []

    def iter_track(self, locator, keep_mask=None):
        keep_mask = np.asarray(keep_mask, bool)
        self.masks.append(keep_mask.copy())
        for keep, count in zip(keep_mask, self.counts[locator]):
            if keep:
                self.reads += int(count > 0)
                yield np.ones((count, 68), np.float32)
            else:
                yield np.zeros((0, 68), np.float32)


class FakeSupply:
    def __init__(self, natural):
        self.points = FakePoints(natural[:, 2].astype(np.int64))
        self.rgb_supply = object()
        self.identity = {"rgb": {"identity": "frozen"}, "points": {"identity": "frozen"}}
        self.natural = natural

    def rgb(self, source, locator):
        return {"valid": self.natural[:, 1].copy(),
                "cls": np.ones((len(self.natural), 768), np.float32)}


def fake_view(natural, branch="rgb"):
    targets = [0, 1, 2, 3][:len(natural)]
    gt = [True] * len(natural)
    baseline = FakeBaseline(natural, targets, gt)
    supply = FakeSupply(natural)
    base = SimpleNamespace(baseline=baseline, supply=supply, entries=baseline.entries)
    data = FeatureDataset(base, "loki", "g-rgb-dditr", mode="branch", branch=branch)
    data._natural_flags = lambda _: natural.copy()
    return data, baseline, supply


class E003aMaskingTest(unittest.TestCase):
    def test_counter_rng_is_deterministic_independent_and_bounded(self):
        sampler = MaskSampler(0)
        for slots in (1, 2, 4, 5, 9, 21, 37):
            natural = np.ones((slots, 3), dtype=bool)
            natural[::4, 1] = False
            original = natural.copy()
            for epoch in (1, 2, 3):
                mask, details = sampler.draw("loki", epoch, f"track-{slots}", natural)
                repeated, repeated_details = sampler.draw("loki", epoch, f"track-{slots}", natural)
                np.testing.assert_array_equal(mask, repeated)
                self.assertEqual(details, repeated_details)
                self.assertLessEqual(details["candidate_draws"], 3)
                self.assertTrue(np.all(mask <= natural))
                if details["augmented"] and not details["fallback"]:
                    self.assertGreaterEqual(mask.any(axis=1).sum(), np.ceil(.25 * natural.any(axis=1).sum()))
                    self.assertTrue(all(gap["newly_hidden"] <= min(10, slots // 4)
                                        for gap in details["empty_gaps"]))
                for attempt in details["attempts"]:
                    if slots == 1:
                        self.assertEqual(attempt["block_length"], slots)
                    else:
                        self.assertIn(attempt["block_length"], [v for v in (1, 5, 10, 20) if v < slots] + [slots])
            np.testing.assert_array_equal(natural, original)
        self.assertNotEqual(mask_seed(0, "loki", 1, "x"), mask_seed(0, "loki", 2, "x"))
        self.assertNotEqual(mask_seed(0, "loki", 1, "x"), mask_seed(0, "road-waymo", 1, "x"))

    def test_rng_does_not_consume_global_numpy_or_torch_state(self):
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        natural = np.ones((12, 3), dtype=bool)
        MaskSampler(0).draw("road-waymo", 3, "one", natural)
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))

    def test_all_naturally_empty_track_remains_an_empty_view(self):
        mask, details = MaskSampler(0).draw("loki", 1, "empty", np.zeros((4, 3), dtype=bool))
        self.assertFalse(mask.any())
        self.assertEqual(details["natural_observable_slots"], 0)

    def test_failed_candidates_fall_back_to_same_track_after_three_attempts(self):
        class RejectRng:
            def random(self):
                return .9 if not hasattr(self, "started") else .5
            def choice(self, values, size=None, replace=True, p=None):
                self.started = True
                if size == 0:
                    return np.zeros(0, dtype=np.int64)
                if list(values) == [1]:
                    return 1
                if tuple(values) == (0, 1, 2, 3):
                    return 0
                return 1

        natural = np.ones((2, 3), dtype=bool)
        with patch("pedestrian_behavior.experiments.e003a.np.random.default_rng", return_value=RejectRng()):
            mask, details = MaskSampler(0).draw("loki", 1, "same", natural)
        np.testing.assert_array_equal(mask, natural)
        self.assertEqual(details["candidate_draws"], 3)
        self.assertTrue(details["fallback"])
        self.assertEqual(details["locator"], "same")
        self.assertTrue(all(attempt["rejected"] for attempt in details["attempts"]))

    def test_empty_gaps_bridge_natural_gaps_and_count_only_added_observations(self):
        natural = np.zeros((6, 3), dtype=bool)
        natural[[0, 2, 4], 0] = True
        effective = np.zeros_like(natural)
        effective[5, 2] = True
        self.assertEqual(_empty_gaps(natural, effective), [{"start": 0, "length": 5, "newly_hidden": 3}])

    def test_geometry_zeroing_preserves_gt_and_skips_masked_point_reads(self):
        natural = np.array([[False, True, True], [True, True, True], [False, True, True]], dtype=bool)
        data, baseline, supply = fake_view(natural, "rgb")
        cost = data.cost(0)
        self.assertEqual(cost, 0)
        self.assertEqual(baseline.reads, 0, "cost/mask selection must precede feature loading")
        sample = data[0]
        self.assertEqual(baseline.reads, 1)
        self.assertEqual(supply.points.reads, 0)
        self.assertFalse(supply.points.masks[-1].any())
        self.assertEqual(int(sample["targets"].numel()), len(natural))
        np.testing.assert_array_equal(sample["targets"].numpy(), baseline.targets.numpy())
        np.testing.assert_array_equal(sample["gt_valid"].numpy(), baseline.gt_valid.numpy())
        self.assertTrue(torch.equal(sample["geometry"], torch.zeros_like(sample["geometry"])))
        self.assertTrue(torch.equal(baseline.inputs[0], torch.zeros(20)))
        self.assertEqual(float(baseline.inputs[1, 8]), 1., "masking changed the frozen input sample")

    def test_real_logical_batches_accept_masked_feature_view(self):
        natural = np.array([[True, True, True], [False, True, False], [True, False, True]], dtype=bool)
        data, baseline, supply = fake_view(natural, "geometry")
        logical = LogicalBatches(data, {"max_tracks": 1, "temporal_cells": 9,
            "microbatch_tokens": 100, "ordinary_tokens": 100}, training=True)
        batches = list(logical.microbatches([0]))
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["lengths"].tolist(), [3])
        self.assertEqual(baseline.reads, 1)
        self.assertEqual(supply.points.reads, 0)

    def test_layernorm_outputs_are_zero_after_effective_branch_gate(self):
        natural = np.array([[False, True, True], [True, True, True], [False, True, True]], dtype=bool)
        data, _, _ = fake_view(natural, "geometry")
        batch = collate_multimodal([data[0]])
        profile = {"frame_chunk": 2, "frame_tokens": 32, "ordinary_tokens": 100,
                   "temporal_cells": 16, "microbatch_tokens": 100, "max_tracks": 2}
        model = build_model("g-rgb-dditr", "temporal-small", profile, rgb_branch_dropout=.5).eval()
        received = []
        handle = model.fusion.register_forward_pre_hook(lambda _module, args: received.append(args[0].detach().clone()))
        try:
            model(batch)
        finally:
            handle.remove()
        fused = received[0]
        self.assertEqual(fused.shape, (1, 3, 64 + 128 + 128 + 3))
        for slot in (0, 2):
            torch.testing.assert_close(fused[0, slot], torch.zeros_like(fused[0, slot]))

    def test_rgb_identity_matches_frozen_feature_supply_key(self):
        class Cache:
            def valid(self, identity):
                self.valid_identity = identity
                return np.array([True], dtype=bool)

            def get(self, identity, produce, validate):
                self.identity = identity
                return {"valid": np.array([True], dtype=bool)}
        cache = Cache()
        catalog = SimpleNamespace(identity={"catalog": "sha"}, tracks={"loc": {"group": "scene", "sha256": "track"}},
                                  source_identity=lambda _: "native")
        supply = FeatureSupply({"loki": catalog}, cache, None, None,
                               {"backend": "frozen", "rgb_checkpoint_sha256": "weights"})
        supply.rgb("loki", "loc")
        self.assertEqual(_rgb_identity(SimpleNamespace(rgb_supply=supply), "loki", "loc"), cache.identity)
        np.testing.assert_array_equal(_rgb_validity(SimpleNamespace(rgb_supply=supply), "loki", "loc", 1), [True])
        self.assertEqual(cache.valid_identity, cache.identity)


if __name__ == "__main__":
    unittest.main()
