import json
from pathlib import Path
import tempfile
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from pedestrian_behavior.evaluation import predict, prediction_metrics, track_cross_entropy
from pedestrian_behavior.experiments import e003b_run
from pedestrian_behavior.training import fit, reload_checkpoint, seed_run


class TinyFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(20, 128)
        self.dropout = nn.Dropout(.8)
        self.final_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 4)

    def forward(self, batch):
        device = self.head.weight.device
        geometry = batch["geometry"].to(device)
        valid = torch.arange(geometry.shape[1], device=device)[None, :] < batch["lengths"].to(device)[:, None]
        features = self.final_norm(self.dropout(self.project(geometry)))
        return self.head(features).masked_fill(~valid[..., None], 0.)


class FakeBaseline:
    def __init__(self, split, names):
        self.entries = [{"locator": json.dumps({"scenario": name}), "archive": name + ".npz",
                         "slots": 3, "split": split} for name in names]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        geometry = torch.zeros((3, 20))
        geometry[:, 8] = 1.
        return {"inputs": geometry, "targets": torch.tensor([0, 1, 2]),
            "gt_valid": torch.tensor([True, False, True]), "locator": entry["locator"],
            "archive": entry["archive"]}


class FakeSupply:
    def __init__(self, tracks):
        self.catalogs = {"loki": SimpleNamespace(tracks=tracks)}
        self.identity = {"fixture": True}

    def track_cost(self, source, locator):
        return 1

    def rgb(self, source, locator):
        return {"cls": np.zeros((3, 768), np.float32), "valid": np.array([True, False, True])}

    def point_tensors(self, source, locator):
        return [np.zeros((1, 68), np.float32) for _ in range(3)]


def fixture_records():
    records = []
    for index, length in enumerate((2, 3, 4)):
        gt = np.array([True] + [False] * (length - 1), dtype=bool)
        targets = np.arange(length, dtype=np.int64) % 4
        records.append({"features": np.random.default_rng(index).normal(size=(length, 128)).astype(np.float32),
            "source_logits": np.zeros((length, 4), np.float32), "targets": targets, "gt_valid": gt,
            "locator": json.dumps({"scenario": f"g{index}"}), "archive": f"g{index}.npz",
            "group": f"g{index}", "source_flags": {"rgb_valid": np.ones(length, bool),
                "lidar_valid": np.arange(length) % 2 == 0, "point_tokens": np.arange(length, dtype=np.int64)}})
    return records


class E003bTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_frozen_hook_features_reproduce_original_logits(self):
        torch.manual_seed(7)
        model = TinyFrozen()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        e003b_run._freeze_source(model)
        self.assertFalse(model.training)
        self.assertFalse(model.dropout.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        tracks = {entry["locator"]: entry for entry in FakeBaseline("validation", ["a", "b"]).entries}
        dataset = e003b_run.e002a_run.FeatureDataset(FakeBaseline("validation", ["a", "b"]),
            FakeSupply(tracks), "loki", "g-rgb-dditr")
        profile = {"max_tracks": 64, "temporal_cells": 100, "microbatch_tokens": 64,
            "ordinary_tokens": 64, "frame_chunk": 2, "frame_tokens": 10}
        records = e003b_run._extract(dataset, model, profile, torch.device("cpu"), "validation", self.root)
        self.assertEqual(len(records), 2)
        for record in records:
            expected = model.head(torch.from_numpy(record["features"])).detach().numpy()
            np.testing.assert_allclose(record["source_logits"], expected, rtol=1e-6, atol=1e-6)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_source_normalization_is_used_for_both_target_splits(self):
        normalization = {"baseline": {"mean": [1.]}, "bbox": {"mean": [2.]}}
        reference = {"setup_path": "setup", "collection_path": "collection",
                     "bbox": {"manifest_path": "bbox/manifest.json"}}
        returned = {"training": FakeBaseline("training", ["a"]),
                    "validation": FakeBaseline("validation", ["b"])}
        assembly = {"datasets": {"loki": reference}}
        with patch.object(e003b_run.e001b, "dataset_from_extension",
                          side_effect=lambda *_args: returned[_args[3]]) as build:
            actual, _ = e003b_run._baselines(assembly, "loki", normalization)
        self.assertEqual(actual, returned)
        self.assertEqual([call.args[-2:] for call in build.call_args_list],
                         [(normalization["baseline"], normalization["bbox"])] * 2)

    def test_smoke_subset_keeps_full_population_identity_separate(self):
        datasets = {"training": FakeBaseline("training", ["a", "b"]),
                    "validation": FakeBaseline("validation", ["c", "d"])}
        full = e003b_run._population(datasets)
        selection = self.root / "selection.json"
        selection.write_text(json.dumps({"tracks": [
            {"dataset": "loki", "split": "training", "locator": datasets["training"].entries[0]["locator"]},
            {"dataset": "loki", "split": "validation", "locator": datasets["validation"].entries[0]["locator"]}]}))
        e003b_run._smoke_subset(selection, "loki", datasets)
        self.assertEqual(full, {json.dumps({"scenario": name}) for name in ("a", "b", "c", "d")})
        self.assertEqual(len(e003b_run._population(datasets)), 2)

    def test_pickle_free_cache_roundtrip(self):
        records, identity = fixture_records(), {"checkpoint_sha256": "abc", "split": "training"}
        path = self.root / "features.npz"
        e003b_run._save_cache(path, records, identity)
        restored = e003b_run._load_cache(path, identity)
        self.assertEqual([row["locator"] for row in restored], [row["locator"] for row in records])
        for old, new in zip(records, restored):
            np.testing.assert_array_equal(new["features"], old["features"])
            np.testing.assert_array_equal(new["targets"], old["targets"])
            np.testing.assert_array_equal(new["gt_valid"], old["gt_valid"])
            np.testing.assert_array_equal(new["source_flags"]["lidar_valid"], old["source_flags"]["lidar_valid"])
        with self.assertRaises(ValueError):
            e003b_run._load_cache(path, identity | {"checkpoint_sha256": "changed"})

    def test_fresh_head_fit_padding_denominators_and_checkpoint_reload(self):
        records = fixture_records()
        seed_run(0)
        head = e003b_run.AdaptationHead()
        self.assertEqual(sum(parameter.numel() for parameter in head.parameters()), 516)
        train = e003b_run._batches(e003b_run.CachedTracks(records), training=True, seed=0)
        validation = e003b_run._batches(e003b_run.CachedTracks(records), seed=0)
        batch = next(iter(validation))
        logits = head(batch["inputs"], batch["lengths"])
        losses = track_cross_entropy(logits, batch["targets"], batch["gt_valid"], batch["padding_mask"])
        for i, length in enumerate(batch["lengths"].tolist()):
            accepted = batch["gt_valid"][i, :length]
            expected = torch.nn.functional.cross_entropy(logits[i, :length][accepted],
                batch["targets"][i, :length][accepted])
            torch.testing.assert_close(losses[i], expected)
        optimizer = torch.optim.AdamW(head.parameters(), **e003b_run.SETTINGS["optimizer"])
        fit_output = self.root / "fit"
        fit_output.mkdir()
        result = fit(head, train, lambda model: prediction_metrics(predict(model, validation, "cpu"), 4),
            optimizer, torch.device("cpu"), fit_output, {"config": {"head": 516}}, lambda *_: None,
            epochs=2, patience=2, clip_norm=1.,
            epoch_operation=partial(e003b_run._warmup_epoch, warmup_steps=3))
        self.assertEqual(result["history"][-1]["processed_tracks"], 3)
        self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))
        loaded = e003b_run.AdaptationHead()
        checkpoint = reload_checkpoint(self.root / "fit/best.pt", loaded, "cpu")
        self.assertEqual(checkpoint["selection"], result["selection"])
        tracks = predict(loaded, validation, "cpu")
        metrics = prediction_metrics(tracks, 4)
        self.assertEqual(metrics["denominators"]["tracks"], 3)
        self.assertEqual(metrics["denominators"]["gt_frames"], sum(row["gt_valid"].sum() for row in records))
        self.assertEqual(metrics["denominators"]["context_frames"], sum(len(row["targets"]) for row in records))
        self.assertTrue(all("rgb_valid" in row["source_flags"] and "lidar_valid" in row["source_flags"] for row in tracks))

    def test_checkpoint_prediction_paths_and_launcher_case_order(self):
        root = self.root / "outputs/experiments/E003a/runs/run-road"
        path = e003b_run._source_prediction_path(root, "road-waymo", "loki", "e003a", self.root)
        self.assertEqual(path, root / "loki-validation/predictions.npz")
        path = e003b_run._source_prediction_path(root, "road-waymo", "loki", "e002d", self.root / "outputs")
        self.assertEqual(path, self.root / "outputs/experiments/E002/transfer-validation-20261102/cells/road-waymo/geometry+rgb+lidar/predictions.npz")
        launcher = Path(__file__).resolve().parents[1] / "scripts/run-e003b-sequential.sh"
        calls = [line.strip() for line in launcher.read_text().splitlines() if line.startswith("run_case ")]
        self.assertEqual(calls, ["run_case e002d road-waymo loki", "run_case e003a road-waymo loki",
            "run_case e002d loki road-waymo", "run_case e003a loki road-waymo"])


if __name__ == "__main__":
    unittest.main()
