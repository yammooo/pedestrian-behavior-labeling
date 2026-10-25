"""Shared physical batches preserve isolated single-model training semantics."""

from copy import deepcopy
import json
import unittest
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pedestrian_behavior.accumulation import train_accumulated_epoch
from pedestrian_behavior.data.multimodal import LogicalBatches, PredictionBatches, collate_multimodal
from pedestrian_behavior.experiments.e002b import build_model
from pedestrian_behavior.experiments import e002b_all, e002b_group
from pedestrian_behavior.shared_accumulation import rng_state, train_shared_accumulated_epoch
from pedestrian_behavior.training import seed_run


PROFILE = {"max_tracks": 8, "temporal_cells": 1000, "microbatch_tokens": 1000,
           "ordinary_tokens": 1000, "frame_chunk": 2, "frame_tokens": 64}


class Samples:
    def __init__(self, values):
        self.values = values
        self.reads = 0
        self.entries = [{"slots": len(value["targets"])} for value in values]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        self.reads += 1
        return self.values[index]

    def cost(self, index):
        return sum(len(frame) for frame in self.values[index].get("points", ()))


class TinyDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(.35)
        self.output = nn.Linear(20, 4)

    def forward(self, batch):
        x = batch["geometry"].to(self.output.weight.device)
        return self.output(self.dropout(x)).masked_fill(batch["padding_mask"].to(x.device)[..., None], 0.)


class RecordingAdamW(torch.optim.AdamW):
    def step(self, closure=None):
        self.last_gradients = [None if parameter.grad is None else parameter.grad.detach().clone()
                               for group in self.param_groups for parameter in group["params"]]
        return super().step(closure)


def _geometry_values(count):
    torch.manual_seed(71)
    result = []
    for index in range(count):
        slots = (index % 3) + 1
        geometry = torch.randn(slots, 20)
        result.append({"geometry": geometry, "flags": torch.ones(slots, 1),
            "targets": torch.arange(slots, dtype=torch.long) % 4,
            "gt_valid": torch.ones(slots, dtype=torch.bool), "locator": f"track-{index}",
            "archive": f"track-{index}.npz"})
    return result


def _point_values():
    torch.manual_seed(91)
    result = []
    for index in range(2):
        slots = 3
        result.append({"geometry": torch.randn(slots, 20), "flags": torch.ones(slots, 2),
            "points": [torch.randn(4 + frame, 68) for frame in range(slots)],
            "targets": torch.arange(slots, dtype=torch.long) % 4,
            "gt_valid": torch.ones(slots, dtype=torch.bool), "locator": f"point-{index}",
            "archive": f"point-{index}.npz"})
    return result


def _clone_rng(state):
    return state[0].clone(), None if state[1] is None else state[1].clone()


def _assert_state_equal(test, left, right):
    test.assertEqual(left.keys(), right.keys())
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0, msg=name)


def _single_reference(model, dataset, profile, state):
    optimizer = RecordingAdamW(model.parameters(), lr=.002, weight_decay=.01)
    reference_loader = LogicalBatches(dataset, profile, training=True, seed=13)
    torch.set_rng_state(state[0])
    result = train_accumulated_epoch(model, reference_loader, optimizer, torch.device("cpu"), 1.,
        peak_lr=.002, warmup_steps=5)
    return model, optimizer, result, rng_state("cpu")


def _assert_optimizer_equal(test, left, right):
    a, b = left.state_dict(), right.state_dict()
    test.assertEqual(a["param_groups"], b["param_groups"])
    test.assertEqual(a["state"].keys(), b["state"].keys())
    for parameter in a["state"]:
        test.assertEqual(a["state"][parameter].keys(), b["state"][parameter].keys())
        for key, value in a["state"][parameter].items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, b["state"][parameter][key], rtol=0, atol=0)
            else:
                test.assertEqual(value, b["state"][parameter][key])


class SharedAccumulationTest(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    def tearDown(self):
        torch.set_num_threads(self.previous_threads)

    def test_isolated_dropout_partial_updates_order_and_inputs_match_independent_runs(self):
        values = _geometry_values(65)
        source = Samples(values)
        frozen = [{key: value.clone() if torch.is_tensor(value) else value for key, value in row.items()}
                  for row in values]
        initial, initial_rng = {}, {}
        for name, seed in (("a", 3), ("b", 7)):
            seed_run(seed)
            model = TinyDropout()
            initial[name] = deepcopy(model.state_dict())
            initial_rng[name] = rng_state("cpu")

        references = {}
        for name in initial:
            model = TinyDropout()
            model.load_state_dict(initial[name])
            references[name] = _single_reference(model, source, PROFILE, initial_rng[name])
        expected = {name: item[0].state_dict() for name, item in references.items()}

        outcomes = []
        for order in (("a", "b"), ("b", "a")):
            models, optimizers = {}, {}
            for name in initial:
                model = TinyDropout()
                model.load_state_dict(initial[name])
                models[name] = model
                optimizers[name] = RecordingAdamW(model.parameters(), lr=.002, weight_decay=.01)
            states = {name: _clone_rng(initial_rng[name]) for name in initial}
            loader = LogicalBatches(source, PROFILE, training=True, seed=13)
            prepared_before = source.reads
            rows, states = train_shared_accumulated_epoch(models, loader, optimizers, "cpu", 1.,
                rng_states=states, peak_lr=.002, warmup_steps=5, order=order)
            self.assertEqual(source.reads - prepared_before, len(source))
            self.assertEqual({name: (rows[name]["optimizer_steps"], rows[name]["processed_tracks"])
                              for name in rows}, {"a": (2, 65), "b": (2, 65)})
            self.assertEqual(rows["a"]["learning_rate"], rows["b"]["learning_rate"])
            for name in initial:
                _assert_state_equal(self, expected[name], models[name].state_dict())
                torch.testing.assert_close(torch.tensor(rows[name]["training_ce"]),
                    torch.tensor(references[name][2]["training_ce"]), rtol=0, atol=1e-7)
                torch.testing.assert_close(states[name][0], references[name][3][0], rtol=0, atol=0)
                _assert_optimizer_equal(self, references[name][1], optimizers[name])
                for expected_grad, actual_grad in zip(references[name][1].last_gradients,
                                                       optimizers[name].last_gradients):
                    if expected_grad is not None:
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
            outcomes.append({name: deepcopy(model.state_dict()) for name, model in models.items()})

        # Stop model A after epoch one; B continues with the same persistent shuffle/RNG stream.
        reference_model = TinyDropout()
        reference_model.load_state_dict(initial["b"])
        reference_optimizer = RecordingAdamW(reference_model.parameters(), lr=.002, weight_decay=.01)
        torch.set_rng_state(initial_rng["b"][0])
        reference_loader = LogicalBatches(source, PROFILE, training=True, seed=13)
        for epoch in range(2):
            train_accumulated_epoch(reference_model, reference_loader, reference_optimizer, torch.device("cpu"), 1.,
                peak_lr=.002, warmup_steps=5, step_offset=epoch * 2)

        stopped_models, stopped_optimizers = {}, {}
        for name in initial:
            model = TinyDropout()
            model.load_state_dict(initial[name])
            stopped_models[name] = model
            stopped_optimizers[name] = RecordingAdamW(model.parameters(), lr=.002, weight_decay=.01)
        stopped_states = {name: _clone_rng(state) for name, state in initial_rng.items()}
        stopped_loader = LogicalBatches(source, PROFILE, training=True, seed=13)
        offsets = {name: 0 for name in initial}
        first_rows, stopped_states = train_shared_accumulated_epoch(stopped_models, stopped_loader,
            stopped_optimizers, "cpu", 1., rng_states=stopped_states, peak_lr=.002, warmup_steps=5,
            step_offsets=offsets)
        for name in initial:
            offsets[name] += first_rows[name]["optimizer_steps"]
        second_rows, stopped_states = train_shared_accumulated_epoch({"b": stopped_models["b"]}, stopped_loader,
            {"b": stopped_optimizers["b"]}, "cpu", 1., rng_states={"b": stopped_states["b"]},
            peak_lr=.002, warmup_steps=5, step_offsets={"b": offsets["b"]})
        self.assertEqual(second_rows["b"]["processed_tracks"], 65)
        _assert_state_equal(self, reference_model.state_dict(), stopped_models["b"].state_dict())
        _assert_optimizer_equal(self, reference_optimizer, stopped_optimizers["b"])

        for name in initial:
            _assert_state_equal(self, outcomes[0][name], outcomes[1][name])
        for before, after in zip(frozen, source.values):
            for key in before:
                if torch.is_tensor(before[key]):
                    self.assertTrue(torch.equal(before[key], after[key]), key)

    def test_e002b_recompute_dropout_and_final_update_match_independent_model(self):
        profile = PROFILE | {"max_tracks": 2, "temporal_cells": 32, "microbatch_tokens": 100,
                            "ordinary_tokens": 1, "frame_tokens": 64, "frame_chunk": 2}
        data = Samples(_point_values())
        seed_run(19)
        original = build_model("g-dditr", "branches-small", profile)
        state_dict, model_rng = deepcopy(original.state_dict()), rng_state("cpu")
        reference = deepcopy(original)
        ref_optimizer = RecordingAdamW(reference.parameters(), lr=.001, weight_decay=.0001)
        ref_rows = train_accumulated_epoch(reference, LogicalBatches(data, profile, training=True, seed=0),
            ref_optimizer, torch.device("cpu"), 1., peak_lr=.001, warmup_steps=3)

        models, optimizers = {}, {}
        for name in ("first", "second"):
            model = build_model("g-dditr", "branches-small", profile)
            model.load_state_dict(state_dict)
            models[name] = model
            optimizers[name] = RecordingAdamW(model.parameters(), lr=.001, weight_decay=.0001)
        states = {name: _clone_rng(model_rng) for name in models}
        before = []
        for row in data.values:
            snapshot = {key: value.clone() if torch.is_tensor(value) else value for key, value in row.items()}
            if "points" in row:
                snapshot["points"] = [frame.clone() for frame in row["points"]]
            before.append(snapshot)
        rows, _ = train_shared_accumulated_epoch(models, LogicalBatches(data, profile, training=True, seed=0),
            optimizers, "cpu", 1., rng_states=states, peak_lr=.001, warmup_steps=3)
        self.assertEqual(rows["first"]["processed_tracks"], 2)
        self.assertEqual(rows["first"]["optimizer_steps"], ref_rows["optimizer_steps"])
        torch.testing.assert_close(torch.tensor(rows["first"]["training_ce"]),
            torch.tensor(ref_rows["training_ce"]), rtol=1e-5, atol=1e-6)
        _assert_state_equal(self, reference.state_dict(), models["first"].state_dict())
        _assert_state_equal(self, models["first"].state_dict(), models["second"].state_dict())
        for optimizer in optimizers.values():
            _assert_optimizer_equal(self, ref_optimizer, optimizer)
            for expected_grad, actual_grad in zip(ref_optimizer.last_gradients, optimizer.last_gradients):
                if expected_grad is not None:
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
        for before_row, after_row in zip(before, data.values):
            for key in before_row:
                if torch.is_tensor(before_row[key]):
                    self.assertTrue(torch.equal(before_row[key], after_row[key]), key)
            for before_frame, after_frame in zip(before_row.get("points", ()), after_row.get("points", ())):
                self.assertTrue(torch.equal(before_frame, after_frame), "points")

    def test_group_failure_finishes_every_open_run_and_marks_attempts_failed(self):
        class FailedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, _batch):
                raise RuntimeError("injected forward failure")

        class FakeRun:
            def __init__(self):
                self.finish_codes = []
                self.id = f"fake-{len(runs)}"
                self.url = "https://example.invalid/"

            def define_metric(self, *_args, **_kwargs):
                pass

            def log(self, *_args, **_kwargs):
                pass

            def finish(self, exit_code=0):
                self.finish_codes.append(exit_code)

        values = _geometry_values(2)
        dataset = Samples(values)
        runs = []
        def fake_init(**_kwargs):
            run = FakeRun()
            runs.append(run)
            return run

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence_root = root / "outputs/experiments/E002b/shared"
            evidence_root.parent.mkdir(parents=True)
            profile = root / "profile.json"
            profile.write_text("{}\n")
            prepared = {"config": {"profile": PROFILE, "warmup_updates": 1}, "normalization": {},
                "datasets": {"road-waymo": {}}, "dataset": lambda _source, _split: dataset,
                "batches": lambda d, training=False, seed=0: LogicalBatches(d, PROFILE, training, seed)
                    if training else PredictionBatches(LogicalBatches(d, PROFILE, training, seed)),
                "conditions": lambda *_args: ({}, {}), "close": lambda: None}
            args = SimpleNamespace(source="road-waymo", device="cpu", output=evidence_root,
                outputs_root=root / "outputs", profile=profile, smoke=False, smoke_selection=None)
            with patch.object(e002b_group, "profile_for", return_value={}), \
                 patch.object(e002b_group, "assembly", return_value=prepared), \
                 patch.object(e002b_group.e002b, "build_model", side_effect=lambda *_args: FailedModel()), \
                 patch.object(e002b_group.wandb, "init", side_effect=fake_init):
                with self.assertRaisesRegex(RuntimeError, "injected forward failure"):
                    e002b_group.run(args)
            self.assertEqual(len(runs), 3)
            self.assertTrue(all(run.finish_codes == [1] for run in runs))
            self.assertEqual(json.loads((evidence_root / "status.json").read_text())["status"], "failed")
            for variant in e002b_group.VARIANTS:
                status = json.loads((evidence_root / variant / "status.json").read_text())
                self.assertEqual(status["status"], "failed")
                self.assertIn("injected forward failure", status["failures"][0]["message"])

    def test_all_models_launcher_checks_shared_profile_before_starting_independent_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs/experiments/E002b/all"
            args = SimpleNamespace(output=output, outputs_root=root / "outputs", device="cuda",
                shared_profile=root / "stale-profile.json")
            with patch.object(e002b_group, "profile_for", side_effect=ValueError("profile changed")), \
                 patch.object(e002b_all.e002b_independent, "run") as independent:
                with self.assertRaisesRegex(ValueError, "profile changed"):
                    e002b_all.run(args)
            independent.assert_not_called()
            self.assertFalse(output.exists())

    def test_final_logging_failure_preserves_local_outputs_and_fails_all_runs(self):
        class FakeRun:
            def __init__(self, index):
                self.finish_codes = []
                self.id, self.url = f"fake-{index}", "https://example.invalid/"

            def define_metric(self, *_args, **_kwargs):
                pass

            def log(self, *_args, **_kwargs):
                pass

            def finish(self, exit_code=0):
                self.finish_codes.append(exit_code)

        values = _geometry_values(2)
        for index, value in enumerate(values):
            value["locator"] = json.dumps({"clip": f"clip-{index}", "track_id": index})
        dataset, runs = Samples(values), []
        def fake_init(**_kwargs):
            run = FakeRun(len(runs))
            runs.append(run)
            return run

        def decorate(tracks, *_args):
            for track in tracks:
                track["conditions"] = {}
                track["source_flags"] = {}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "outputs/experiments/E002b/shared"
            evidence.parent.mkdir(parents=True)
            profile = root / "profile.json"
            profile.write_text("{}\n")
            prepared = {"config": {"profile": PROFILE, "warmup_updates": 1}, "normalization": {},
                "datasets": {"road-waymo": {}}, "dataset": lambda _source, _split: dataset,
                "batches": lambda d, training=False, seed=0: LogicalBatches(d, PROFILE, training, seed)
                    if training else PredictionBatches(LogicalBatches(d, PROFILE, training, seed)),
                "conditions": lambda *_args: ({}, {}), "close": lambda: None}
            args = SimpleNamespace(source="road-waymo", device="cpu", output=evidence,
                outputs_root=root / "outputs", profile=profile, smoke=False, smoke_selection=None)
            settings = {"seed": 0, "epochs": 1, "patience": 1, "clip_norm": 1.,
                "optimizer": {"lr": .002, "weight_decay": .001, "betas": [.9, .999], "eps": 1e-8}}
            with patch.object(e002b_group, "SETTINGS", settings), \
                 patch.object(e002b_group, "profile_for", return_value={}), \
                 patch.object(e002b_group, "assembly", return_value=prepared), \
                 patch.object(e002b_group.e002b, "build_model", side_effect=lambda *_args: TinyDropout()), \
                 patch.object(e002b_group, "AXES", {}), \
                 patch.object(e002b_group, "_decorate", side_effect=decorate), \
                 patch.object(e002b_group, "plots", return_value=[]), \
                 patch.object(e002b_group, "log_final", side_effect=RuntimeError("injected final log failure")), \
                 patch.object(e002b_group.wandb, "init", side_effect=fake_init):
                with self.assertRaisesRegex(RuntimeError, "injected final log failure"):
                    e002b_group.run(args)
            self.assertEqual(len(runs), 3)
            self.assertTrue(all(run.finish_codes == [1] for run in runs))
            for variant in e002b_group.VARIANTS:
                attempt = evidence / variant
                self.assertEqual(json.loads((attempt / "status.json").read_text())["status"], "failed")
                self.assertTrue((attempt / "steps.jsonl").is_file())
                self.assertTrue((attempt / "history.json").is_file())
                self.assertTrue((attempt / "best.pt").is_file())
                self.assertTrue((attempt / "last.pt").is_file())
                self.assertTrue((attempt / "road-waymo-validation/predictions.npz").is_file())
                self.assertTrue((attempt / "road-waymo-validation/metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
