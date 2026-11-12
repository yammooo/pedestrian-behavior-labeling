import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from pedestrian_behavior.experiments import e004a_run


def _record(locator, group, label=0):
    return {"features": np.zeros((1, 128), dtype=np.float32),
        "source_logits": np.zeros((1, 4), dtype=np.float32),
        "targets": np.asarray([label], dtype=np.int64), "gt_valid": np.ones(1, dtype=bool),
        "locator": locator, "archive": locator + ".npz", "group": group, "source_flags": {}}


class RecordingLoader:
    def __init__(self):
        self.iterations = 0

    def __iter__(self):
        marker = self.iterations
        self.iterations += 1
        yield {"inputs": torch.full((1, 1, 128), float(marker)),
            "lengths": torch.ones(1, dtype=torch.long),
            "targets": torch.zeros((1, 1), dtype=torch.long),
            "gt_valid": torch.ones((1, 1), dtype=torch.bool),
            "padding_mask": torch.zeros((1, 1), dtype=torch.bool)}


class RecordingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 4)
        self.seen = []

    def forward(self, inputs, lengths):
        self.seen.extend(inputs[:, 0, 0].detach().cpu().tolist())
        return self.linear(inputs)


class FakeRun:
    def define_metric(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class E004aRunTest(unittest.TestCase):
    def test_recording_draw_is_deterministic_nested_and_round_robin(self):
        records = [_record(f"{group}-{index}", group, index % 4)
            for group, count in (("scenario-a", 4), ("scenario-b", 2), ("scenario-c", 1))
            for index in range(count)]
        first = e004a_run.recording_round_robin(records, 3)
        second = e004a_run.recording_round_robin(records, 3)
        self.assertEqual([row["locator"] for row in first], [row["locator"] for row in second])
        self.assertEqual(len({row["group"] for row in first[:3]}), 3)
        for smaller, larger in zip((1, 3, 6), (3, 6, 7)):
            self.assertEqual([row["locator"] for row in first[:smaller]],
                             [row["locator"] for row in first[:larger]][:smaller])

    def test_validation_group_split_is_disjoint_46_47(self):
        records = [_record(f"track-{group}-{index}", f"scenario-{group:02d}", index % 4)
            for group in range(93) for index in range(2)]
        development, assessment, identity = e004a_run.split_development_assessment(records)
        dev_groups = {row["group"] for row in development}
        assessment_groups = {row["group"] for row in assessment}
        self.assertEqual((len(dev_groups), len(assessment_groups)), (46, 47))
        self.assertFalse(dev_groups & assessment_groups)
        self.assertEqual(len(development) + len(assessment), len(records))
        self.assertEqual(identity["seed"], 0)

    def test_source_initialization_copies_all_516_parameters(self):
        state = {"weight": np.arange(512, dtype=np.float32).reshape(4, 128),
                 "bias": np.arange(4, dtype=np.float32)}
        model = e004a_run._new_head(state, torch.device("cpu"))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 516)
        torch.testing.assert_close(model.linear.weight, torch.from_numpy(state["weight"]))
        torch.testing.assert_close(model.linear.bias, torch.from_numpy(state["bias"]))

    def test_update_intervals_keep_one_loader_and_monotone_steps(self):
        loader, model = RecordingLoader(), RecordingHead()
        steps, checks = [], []
        with tempfile.TemporaryDirectory() as directory:
            result = e004a_run.fit(model, loader,
                lambda _model: {"track_weighted": {"macro_f1": .5}, "track_ce": 1.},
                torch.optim.AdamW(model.parameters(), lr=3e-4), torch.device("cpu"), directory,
                {"config": {}, "normalization": {}}, lambda row, _history: checks.append(row["epoch"]),
                epochs=2, patience=15, clip_norm=1., on_step=lambda step, _loss: steps.append(step),
                epoch_operation=e004a_run._fixed_update_epoch(100), initial_candidate=True)
        self.assertEqual(checks, [0, 1, 2])
        self.assertEqual(steps, list(range(1, 201)))
        self.assertEqual(loader.iterations, 200)
        self.assertEqual(model.seen, [float(value) for value in range(200)])
        self.assertEqual(result["history"][-1]["optimizer_steps"], 100)

    def test_assessment_tracks_are_first_predicted_after_selection(self):
        from pedestrian_behavior.evaluation import predict as real_predict

        training, development, assessment = [_record("train", "train")], [_record("dev", "dev")], [_record("audit", "audit")]
        cache = {"root": Path("cache"),
            "head": {"weight": np.zeros((4, 128), dtype=np.float32), "bias": np.zeros(4, dtype=np.float32)},
            "manifest": {"normalization": {}, "source_head": {"state_sha256": "test-head"},
                "derived_features": {"training": {"sha256": "train-cache"},
                                     "validation": {"sha256": "validation-cache"}}}}
        plan = {"population": {"development": {"tracks": 1, "groups": 1, "gt_timesteps": 1,
                "class_gt_counts": [1, 0, 0, 0]}, "assessment": {"tracks": 1, "groups": 1,
                "gt_timesteps": 1, "class_gt_counts": [1, 0, 0, 0]}}}
        plan_row = {"budget": 1, "draw": 0}
        predicted_ids = []

        def observe_predict(model, batches, device):
            rows = real_predict(model, batches, device)
            predicted_ids.append({row["locator"] for row in rows})
            return rows

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(e004a_run.wandb, "init", return_value=FakeRun()), \
                patch.object(e004a_run.wandb, "Table", side_effect=lambda **kwargs: kwargs), \
                patch.object(e004a_run.wandb, "Image", side_effect=lambda path: path), \
                patch.object(e004a_run, "predict", side_effect=observe_predict):
            e004a_run._run_one(Path(directory), cache, training, development, assessment,
                plan_row, plan, torch.device("cpu"), True)
        self.assertEqual(predicted_ids[:4], [{"dev"}] * 4)
        self.assertEqual(predicted_ids[4], {"audit"})


if __name__ == "__main__":
    unittest.main()
