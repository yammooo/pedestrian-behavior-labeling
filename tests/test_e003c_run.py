import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pedestrian_behavior.data.loading import collate_tracks
from pedestrian_behavior.evaluation import save_predictions
from pedestrian_behavior.experiments import e003b_run
from pedestrian_behavior.experiments.e003c_run import (TemporalHead, _check_adapted_predictions,
    _canonical_records, _load_attempt_cache, _model, _smoke_subset, _warmup_epoch, optimizer_factory)
from pedestrian_behavior.models.multimodal import TemporalBlock
from pedestrian_behavior.training import fit, seed_run, train_epoch


class SourceStub:
    def __init__(self):
        self.front = nn.Linear(4, 4)
        self.temporal = nn.ModuleList([TemporalBlock(4, 8, .1), TemporalBlock(4, 8, .1)])
        self.final_norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 4)


def _prediction_tracks():
    tracks = []
    for label in range(4):
        logits = np.full((1, 4), -1., dtype=np.float32)
        logits[0, label] = 1.
        tracks.append({"logits": logits, "targets": np.array([label], dtype=np.int64),
            "gt_valid": np.array([True]), "locator": f"track-{label}", "archive": f"track-{label}.npz",
            "group": f"group-{label}", "conditions": {}, "source_flags": {}})
    return tracks


class E003cInitializationTest(unittest.TestCase):
    def test_failed_feature_preparation_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.json").write_text(json.dumps({"status": "failed"}))
            with self.assertRaisesRegex(ValueError, "must be complete"):
                _load_attempt_cache(root, "training", stage="pre-temporal-fusion",
                    source_normalization={}, source_sha="fixture")

    def test_smoke_selection_filters_the_wrapped_feature_dataset(self):
        class Baseline:
            def __init__(self, locators):
                self.entries = [{"locator": locator} for locator in locators]

            def __len__(self):
                return len(self.entries)

            def __getitem__(self, index):
                return self.entries[index]

        baselines = {"training": Baseline(["first", "selected"]),
            "validation": Baseline(["first-val", "selected-val"])}

        class FeatureDataset:
            def __init__(self, baseline):
                self.baseline = baseline
                self.entries = list(baseline.entries)

            def __len__(self):
                return len(self.baseline)

            def __getitem__(self, index):
                return self.baseline[index]

        datasets = {split: FeatureDataset(baseline) for split, baseline in baselines.items()}
        selection = {"tracks": [
            {"dataset": "loki", "split": "training", "locator": "selected"},
            {"dataset": "loki", "split": "validation", "locator": "selected-val"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(json.dumps(selection))
            selected = _smoke_subset(path, baselines)
        for split, dataset in datasets.items():
            dataset.entries = [entry for entry in dataset.entries if entry["locator"] in selected[split]]
        self.assertEqual(selected["training"], {"selected"})
        self.assertEqual(len(datasets["training"]), 1)
        self.assertEqual(datasets["training"][0]["locator"], "selected")
        self.assertEqual(datasets["validation"][0]["locator"], "selected-val")

    def test_cache_order_is_canonical_before_same_seeded_loader(self):
        records = [{"features": np.zeros((1, 128), dtype=np.float32),
            "targets": np.array([i % 4], dtype=np.int64), "gt_valid": np.array([True]),
            "locator": f"track-{i}", "archive": f"track-{i}.npz", "group": "group",
            "source_flags": {}} for i in range(8)]
        first = e003b_run.CachedTracks(_canonical_records(records[::-1]))
        second = e003b_run.CachedTracks(_canonical_records(records))
        first_batches = e003b_run._batches(first, training=True, seed=0)
        second_batches = e003b_run._batches(second, training=True, seed=0)

        def epoch_order(batches):
            return [locator for batch in batches for locator in batch["locators"]]

        self.assertEqual([epoch_order(first_batches) for _ in range(2)],
                         [epoch_order(second_batches) for _ in range(2)])

    def test_epoch_zero_candidate_survives_worse_continuation(self):
        model = nn.Linear(1, 1, bias=False)
        model.weight.data.zero_()

        def validate(current):
            initial = current.weight.item() == 0.
            return {"track_ce": .5 if initial else .8,
                "track_weighted": {"macro_f1": .6 if initial else .4}}

        def continue_epoch(current, _batches, _optimizer, _device, _clip, on_step=None, step_offset=0):
            with torch.no_grad():
                current.weight.fill_(1.)
            return {"training_ce": .7, "training_f1": .5, "optimizer_steps": 1,
                "processed_tracks": 1, "context_slots": 1, "gt_slots": 1,
                "padding_slots": 0, "batch_slots": 1, "seconds": 0.}

        with tempfile.TemporaryDirectory() as directory:
            result = fit(model, [], validate, torch.optim.AdamW(model.parameters()), "cpu", directory,
                {"config": {}, "normalization": {}}, lambda *_: None, epochs=1, patience=1,
                clip_norm=1., epoch_operation=continue_epoch, initial_candidate=True)
            checkpoint = torch.load(Path(directory) / "best.pt", map_location="cpu", weights_only=True)
        self.assertEqual(result["selection"], {"epoch": 0, "f1": .6, "ce": .5})
        self.assertEqual(checkpoint["epoch"], 0)
        self.assertEqual(checkpoint["selection"]["epoch"], 0)
        self.assertEqual(float(checkpoint["model_state"]["weight"].item()), 0.)

    def test_temporal_wrapper_reenables_only_temporal_and_disables_dropout(self):
        source = SourceStub()
        for parameter in source.front.parameters():
            parameter.requires_grad_(False)
        for parameter in source.temporal.parameters():
            parameter.requires_grad_(False)
        for parameter in source.final_norm.parameters():
            parameter.requires_grad_(False)
        for parameter in source.head.parameters():
            parameter.requires_grad_(False)
        weights = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        bias = torch.arange(4, dtype=torch.float32)
        model = _model("temporal-head", source, {"model_state": {"linear.weight": weights,
            "linear.bias": bias}}, "cpu")
        model.train()
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in source.front.parameters()))
        self.assertTrue(torch.equal(model.head.weight, weights))
        self.assertTrue(torch.equal(model.head.bias, bias))
        self.assertTrue(all(layer.p == 0. for layer in model.modules() if isinstance(layer, nn.Dropout)))

    def test_optimizer_groups_keep_head_to_temporal_lr_ratio_during_warmup(self):
        model = TemporalHead(nn.ModuleList([TemporalBlock(8, 16, .1), TemporalBlock(8, 16, .1)]),
                             nn.LayerNorm(8), nn.Linear(8, 4))
        optimizer = optimizer_factory("temporal-head")(model)
        batch = collate_tracks([{"inputs": torch.randn(2, 8), "targets": torch.tensor([0, 1]),
            "gt_valid": torch.tensor([True, True]), "locator": "track", "archive": "track.npz"}])
        _warmup_epoch("temporal-head", 3)(model, [batch], optimizer, torch.device("cpu"), 1.)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"] / optimizer.param_groups[1]["lr"], 10.)
        head_only = optimizer_factory("head")(e003b_run.AdaptationHead())
        self.assertEqual(len(head_only.param_groups), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA full-shape acceptance runs on aalto")
    def test_cuda_temporal_head_full_shape_forward_backward_and_update(self):
        seed_run(0)
        device = torch.device("cuda")
        model = TemporalHead(nn.ModuleList([TemporalBlock(128, 512, .1), TemporalBlock(128, 512, .1)]),
            nn.LayerNorm(128), nn.Linear(128, 4)).to(device)
        model.train()
        batch_size, timesteps = 64, 102
        batch = {"inputs": torch.randn(batch_size, timesteps, 128, device=device),
            "lengths": torch.full((batch_size,), timesteps, dtype=torch.int64, device=device),
            "targets": torch.randint(0, 4, (batch_size, timesteps), device=device),
            "gt_valid": torch.ones((batch_size, timesteps), dtype=torch.bool, device=device),
            "padding_mask": torch.zeros((batch_size, timesteps), dtype=torch.bool, device=device)}
        optimizer = optimizer_factory("temporal-head")(model)
        result = train_epoch(model, [batch], optimizer, device, 1.)
        torch.cuda.synchronize(device)

        self.assertEqual(result["optimizer_steps"], 1)
        self.assertEqual(result["processed_tracks"], batch_size)
        self.assertEqual(result["context_slots"], batch_size * timesteps)
        self.assertTrue(np.isfinite(result["training_ce"]))
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
            self.assertTrue(torch.isfinite(parameter).all().item(), name)

    def test_epoch_zero_logits_reproduce_saved_adapted_predictions(self):
        tracks = _prediction_tracks()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            save_predictions(path, tracks)
            metrics = _check_adapted_predictions(tracks, path)
            self.assertEqual(metrics["denominators"]["tracks"], 4)
            changed = [dict(row, logits=row["logits"] + .1) for row in tracks]
            with self.assertRaises(ValueError):
                _check_adapted_predictions(changed, path)


if __name__ == "__main__":
    unittest.main()
