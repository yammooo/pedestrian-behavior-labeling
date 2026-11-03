import gzip
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pedestrian_behavior.evaluation import load_predictions, prediction_metrics, save_predictions
from pedestrian_behavior.experiments import e003a_run, run as shared_run
from pedestrian_behavior.training import train_epoch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = torch.nn.Linear(1, 4)
        self.forward_modes = []
        self.loaded = []

    def forward(self, inputs, lengths):
        self.forward_modes.append(self.training)
        return self.readout(inputs)

    def load_state_dict(self, state_dict, strict=True):
        self.loaded.append(strict)
        return super().load_state_dict(state_dict, strict=strict)


class FakeFeatureDataset:
    def __init__(self, base, source, arm, *, mode="natural", branch=None, sampler=None,
                 seed=0, fixed_masks=None):
        self.base, self.source, self.arm = base, source, arm
        self.mode, self.branch, self.sampler, self.seed = mode, branch, sampler, seed
        self.fixed_masks = fixed_masks
        self.finish_calls = 0
        self.locators = [json.dumps({"clip": "a"}), json.dumps({"clip": "b"})]

    def finish_epoch(self):
        self.finish_calls += 1
        return ({"tracks": 1, "clean_tracks": 1, "augmented": 0, "changed": 0, "fallback": 0,
                 "clean_fraction": 1., "changed_augmented_fraction": 0., "fallback_fraction": 0.,
                 "natural_observable_slots": 2, "effective_observable_slots": 2,
                 "newly_hidden_observable_slots": 0, "effective_observable_fraction": 1.,
                 "branch_retention_fraction": {"geometry": 1., "rgb": 1., "lidar": 1.}},
                [{"locator": "fixture", "changed": False}])


def _flags(dataset, locator):
    natural = {
        "a": {"geometry": np.array([False, False]), "rgb": np.array([True, True]),
              "lidar": np.array([False, True])},
        "b": {"geometry": np.array([True, True]), "rgb": np.array([False, True]),
              "lidar": np.array([True, True])},
    }[json.loads(locator)["clip"]]
    if dataset.mode == "branch":
        effective = {name: values if name == dataset.branch else np.zeros(2, dtype=bool)
                     for name, values in natural.items()}
    elif dataset.mode == "fixedmixed":
        proposal = ({"geometry": np.zeros(2, bool), "rgb": np.zeros(2, bool), "lidar": np.zeros(2, bool)}
                    if json.loads(locator)["clip"] == "a" else
                    {"geometry": np.array([True, False]), "rgb": np.zeros(2, bool),
                     "lidar": np.array([False, True])})
        key = (dataset.source, locator)
        effective = dataset.fixed_masks.setdefault(key, {key: value.copy() for key, value in proposal.items()})
    else:
        effective = natural
    return {name + "_valid": values.copy() for name, values in natural.items()} | {
        "effective_" + name + "_valid": values.copy() for name, values in effective.items()}


class FakeLogical:
    def __init__(self, dataset, training=False, seed=0):
        self.dataset, self.training, self.seed, self.iteration = dataset, training, seed, 1

    def __iter__(self):
        metadata = [{"source_flags": _flags(self.dataset, locator)} for locator in self.dataset.locators]
        yield {"inputs": torch.ones((2, 2, 1)), "lengths": torch.tensor([2, 2]),
               "targets": torch.tensor([[0, 1], [1, 2]]), "gt_valid": torch.ones((2, 2), dtype=torch.bool),
               "padding_mask": torch.zeros((2, 2), dtype=torch.bool), "locators": self.dataset.locators,
               "archives": ["a.npz", "b.npz"], "prediction_metadata": metadata}


class E003aRunTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def assembly_fixture(self, patches, *, output, smoke=True):
        source, opposite = "road-waymo", "loki"
        source_training, source_validation = "road-train", json.dumps({"clip": "source-val-a"})
        source_validation_2 = json.dumps({"clip": "source-val-b"})
        target_a, target_b = json.dumps({"scenario": "target-a"}), json.dumps({"scenario": "target-b"})
        source_catalog = SimpleNamespace(tracks={
            source_training: {"split": "training"}, source_validation: {"split": "validation"},
            source_validation_2: {"split": "validation"}})
        supply = SimpleNamespace(catalogs={source: source_catalog},
                                 rgb_supply=SimpleNamespace(rgb_cache="read-only-rgb"))
        source_bases = {split: SimpleNamespace(source=source, split=split, supply=supply)
                        for split in ("training", "validation")}
        normalization = {"baseline": {"mean": [1.]}, "bbox": {"mean": [2.]}}
        base_close_calls = []
        original_batch_calls = []

        def original_batches(data, training=False, seed=0):
            original_batch_calls.append((data, training, seed))
            return FakeLogical(data, training, seed)

        def source_conditions(metadata, arrays, track, dataset):
            return {"route": np.ones(2, dtype=np.int64)}, {"owner": np.ones(2, dtype=np.int64)}

        prepared = {"config": {"experiment": "E002a"}, "normalization": normalization,
            "datasets": {source: {"fixture": True}},
            "dataset": lambda name, split: source_bases[split] if name == source else None,
            "batches": original_batches, "epoch_operation": train_epoch,
            "conditions": source_conditions, "build_model": TinyModel, "close": lambda: base_close_calls.append(True),
            "forward_batch": None}
        policy_record = {}

        def base_assembly(args, *, policy):
            policy_record.update(policy)
            return prepared

        profile = {"temporal_cells": 64, "frame_tokens": 10}
        control_ids, target_dataset_call, points_instances = [], [], []
        target_baseline = SimpleNamespace(entries=[
            {"locator": target_a, "archive": "target-a.npz", "bbox_sha256": "digest", "slots": 2},
            {"locator": target_b, "archive": "target-b.npz", "bbox_sha256": "digest", "slots": 2}])
        target_catalog = SimpleNamespace(identity={"catalog": "target"}, tracks={target_a: {}, target_b: {}}, protected={})

        class FakePoints:
            def __init__(self, *args, **kwargs):
                self.protected = {"point-manifest": "digest"}
                self.identity = {"points": "target"}
                self.counts = {target_a: np.array([1, 0]), target_b: np.array([0, 1])}
                self.required = []
                self.closed = 0
                points_instances.append(self)

            def require_track(self, locator):
                self.required.append(locator)

            def close(self):
                self.closed += 1

        selection = self.root / "smoke.json"
        selection.write_text(json.dumps({"tracks": [
            {"dataset": opposite, "split": "validation", "locator": target_a},
            {"dataset": opposite, "split": "training", "locator": target_b},
            {"dataset": source, "split": "validation", "locator": source_validation}] }))
        args = SimpleNamespace(source=source, output=output, outputs_root=self.root / "outputs",
            profile=self.root / "profile.json", device="cpu", control_run=self.root / "outputs/experiments/E002d/runs/fusion-20261030-140937-road-waymo-rgbdrop50",
            probe=self.root / "probe", rgb_cache_root=self.root / "rgb", rgb_preparation=self.root / "rgb-source.json",
            opposite_rgb_preparation=self.root / "rgb-target.json", indexed_point_cache=None,
            opposite_indexed_point_cache=None, point_workers=1, smoke=smoke,
            smoke_selection=selection if smoke else None)
        if smoke:
            output.mkdir(parents=True, exist_ok=True)

        def control_validator(control_path, got_source, got_normalization, expected):
            control_ids.append((got_source, got_normalization, set(expected)))
            return {"model_state": TinyModel().state_dict()}, {"path": str(control_path), "sha256": "control"}

        def target_dataset_from_extension(*args):
            target_dataset_call.append(args)
            return target_baseline

        def target_owner_assemble(*args, **kwargs):
            return {"conditions": lambda metadata, arrays, track, dataset:
                    ({"route": np.full(2, 2, dtype=np.int64)}, {"owner": np.full(2, 2, dtype=np.int64)})}

        def control_transfer(*args):
            return {"path": "fixture", "manifest_sha256": "manifest", "results_sha256": "results",
                    "target_predictions": "predictions.npz", "target_predictions_sha256": "predictions",
                    "target_support": {"tracks": 2}, "target": {"target_tracks": 2, "target_f1": .5, "target_ce": 1.}}

        patches.enter_context(patch.object(e003a_run, "_profile_for_assembly", return_value={"profile": profile}))
        patches.enter_context(patch.object(e003a_run, "_validate_roots"))
        patches.enter_context(patch.object(e003a_run.e002a_run, "assembly", side_effect=base_assembly))
        patches.enter_context(patch.object(e003a_run.e002a_run, "load_configuration", return_value=({}, None)))
        patches.enter_context(patch.object(e003a_run, "FeatureCatalog", return_value=target_catalog))
        patches.enter_context(patch.object(e003a_run, "extraction_identity", return_value={"rgb": "identity"}))
        patches.enter_context(patch.object(e003a_run, "_validate_rgb_preparation"))
        patches.enter_context(patch.object(e003a_run, "PreparedPoints", FakePoints))
        patches.enter_context(patch.object(e003a_run.e002a_run, "PreparedSupply",
                                            side_effect=lambda *args: SimpleNamespace(identity={"target": True})))
        patches.enter_context(patch.object(e003a_run.e001b, "dataset_from_extension", side_effect=target_dataset_from_extension))
        patches.enter_context(patch.object(e003a_run.e001b_run, "assemble", side_effect=target_owner_assemble))
        patches.enter_context(patch.object(e003a_run.e002a_run, "FeatureDataset",
                                            side_effect=lambda baseline, supply, source, arm:
                                                SimpleNamespace(baseline=baseline, supply=supply, source=source)))
        patches.enter_context(patch.object(e003a_run, "FeatureDataset", FakeFeatureDataset))
        patches.enter_context(patch.object(e003a_run, "_validate_control", side_effect=control_validator))
        patches.enter_context(patch.object(e003a_run, "checksum", return_value="digest"))
        if not smoke:
            patches.enter_context(patch.object(e003a_run, "_control_transfer_result", side_effect=control_transfer))
        assembly = e003a_run.assembly(args)
        return SimpleNamespace(args=args, prepared=assembly, source=source, opposite=opposite,
            source_validation=source_validation, source_validation_2=source_validation_2,
            target_a=target_a, target_b=target_b, control_ids=control_ids,
            target_dataset_call=target_dataset_call, points_instances=points_instances,
            source_bases=source_bases, normalization=normalization, original_batch_calls=original_batch_calls,
            base_close_calls=base_close_calls, policy=policy_record, source_catalog=source_catalog)

    def test_assembly_routes_bounded_smoke_and_wraps_captured_hooks(self):
        with ExitStack() as patches:
            fixture = self.assembly_fixture(patches, output=self.root / "run", smoke=True)
            prepared, args = fixture.prepared, fixture.args

            self.assertEqual(prepared["evaluation_splits"], [("road-waymo", "validation-smoke"),
                                                               ("loki", "validation-smoke")])
            self.assertEqual(prepared["config"]["opposite_validation"],
                             {"source": "loki", "normalization_source": "road-waymo", "tracks": 1,
                              "split": "validation-smoke"})
            self.assertEqual(fixture.control_ids, [("road-waymo", fixture.normalization,
                {fixture.source_validation, fixture.source_validation_2})])
            self.assertEqual(fixture.target_dataset_call[0][5:],
                             (fixture.normalization["baseline"], fixture.normalization["bbox"]))
            target_baseline = prepared["dataset"]("loki", "validation-smoke").base.baseline
            self.assertEqual([row["locator"] for row in target_baseline.entries], [fixture.target_a])
            self.assertEqual(fixture.points_instances[0].required, [fixture.target_a, fixture.target_b])
            self.assertEqual(args.smoke_selection, fixture.args.smoke_selection)
            self.assertEqual(prepared["config"]["control"]["path"], str(args.control_run.resolve()))

            source_conditions = prepared["conditions"]({}, {}, {"source_flags": {}}, "road-waymo")
            observed_flags = {"rgb_valid": np.array([True, False]), "lidar_valid": np.array([False, True])}
            target_conditions = prepared["conditions"]({}, {}, {"source_flags": observed_flags}, "loki")
            self.assertEqual((source_conditions[0]["route"][0], target_conditions[0]["route"][0]), (1, 2))
            np.testing.assert_array_equal(target_conditions[0]["rgb-observed"], [1, 0])
            np.testing.assert_array_equal(target_conditions[0]["lidar-observed"], [0, 1])

            routed = prepared["evaluation_predictions"](TinyModel(), "loki", "validation-smoke", torch.device("cpu"))
            self.assertEqual(fixture.original_batch_calls[-1][0].source, "loki")
            self.assertEqual(len(routed), 2)

            training = prepared["dataset"]("road-waymo", "training")
            logical = prepared["batches"](training, training=True, seed=17)
            self.assertEqual(fixture.original_batch_calls[-1], (training, True, 17))
            self.assertEqual(training.epoch(), logical.iteration)
            model = TinyModel()
            row = prepared["epoch_operation"](model, logical, torch.optim.AdamW(model.parameters()), torch.device("cpu"), 1.)
            self.assertEqual(row["masking"]["tracks"], 1)
            self.assertEqual(training.finish_calls, 1)
            self.assertEqual(json.loads((args.output / "masks/epoch-001-summary.json").read_text()), row["masking"])
            with gzip.open(args.output / "masks/epoch-001-tracks.jsonl.gz", "rt") as records:
                self.assertEqual(json.loads(records.readline()), {"locator": "fixture", "changed": False})

            prepared["close"]()
            self.assertEqual((fixture.base_close_calls, fixture.points_instances[0].closed), ([True], 1))
            self.assertTrue((args.output / "opposite-protected.json").exists())

    def test_post_evaluation_keeps_shared_masks_and_no_input_tracks_in_eval_mode(self):
        with ExitStack() as patches:
            fixture = self.assembly_fixture(patches, output=self.root / "run", smoke=True)
            prepared = fixture.prepared
            models = []
            patches.enter_context(patch.object(e003a_run.e002b, "build_model",
                                                side_effect=lambda *args, **kwargs: models.append(TinyModel()) or models[-1]))
            patches.enter_context(patch.object(e003a_run, "load_track",
                return_value=({"native_extent_us": [0, 200000]}, {})))
            patches.enter_context(patch.object(e003a_run, "source_times", return_value=[0, 200000]))
            model = TinyModel()
            model.train()
            prepared["post_evaluation"](model, {}, {}, fixture.args.output, torch.device("cpu"), None)

            root = fixture.args.output / "diagnostics/road-waymo"
            new = load_predictions(root / "fixedmixed/e003a/predictions.npz")
            control = load_predictions(root / "fixedmixed/e002d-control/predictions.npz")
            for left, right in zip(new, control):
                for branch in ("geometry", "rgb", "lidar"):
                    np.testing.assert_array_equal(left["source_flags"]["effective_" + branch + "_valid"],
                                                  right["source_flags"]["effective_" + branch + "_valid"])
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["fixedmixed"]["e003a"]["no_input_tracks"], 1)
            self.assertEqual(summary["fixedmixed"]["e002d-control"]["no_input_tracks"], 1)
            self.assertEqual(summary["fixedmixed"]["e003a"]["no_input"]["denominators"]["tracks"], 1)
            self.assertEqual(summary["geometry-only"]["e003a"]["no_input_tracks"], 1)
            self.assertTrue(model.forward_modes and all(mode is False for mode in model.forward_modes))
            self.assertTrue(models and models[0].loaded == [True])
            self.assertTrue(models[0].forward_modes and all(mode is False for mode in models[0].forward_modes))

    def test_local_mask_history_survives_epoch_wandb_failure(self):
        with ExitStack() as patches:
            fixture = self.assembly_fixture(patches, output=self.root / "outputs/experiments/E003a/run", smoke=False)
            epoch_payloads = []

            class FakeRun:
                id, url = "fixture", "fixture"
                config = SimpleNamespace(update=lambda *args, **kwargs: None)

                def define_metric(self, *args, **kwargs):
                    pass

                def log(self, row, **kwargs):
                    if "epoch" in row:
                        epoch_payloads.append(row)
                        raise RuntimeError("epoch W&B failure")

                def finish(self, **kwargs):
                    pass

            patches.enter_context(patch.object(shared_run.wandb, "init", return_value=FakeRun()))
            with self.assertRaisesRegex(RuntimeError, "epoch W&B failure"):
                shared_run.run_attempt("road-waymo", e003a_run.ARM, e003a_run.VARIANT, fixture.args.output, "cpu",
                    settings=e003a_run.SETTINGS | {"epochs": 1},
                    assemble=lambda *_: fixture.prepared, classes=["a", "b", "c", "d"], axes={})

            history = json.loads((fixture.args.output / "history.json").read_text())
            self.assertEqual((history[0]["epoch"], history[0]["masking"]["tracks"]), (1, 1))
            self.assertTrue((fixture.args.output / "masks/epoch-001-summary.json").exists())
            with gzip.open(fixture.args.output / "masks/epoch-001-tracks.jsonl.gz", "rt") as records:
                self.assertEqual(json.loads(records.readline())["locator"], "fixture")
            self.assertEqual(json.loads((fixture.args.output / "status.json").read_text())["status"], "failed")
            self.assertTrue((fixture.args.output / "best.pt").exists())
            mask_scalars = {key: value for key, value in epoch_payloads[0].items()
                            if key.startswith("details/masking/") and isinstance(value, (int, float))}
            self.assertTrue({"details/masking/clean_fraction", "details/masking/changed_augmented_fraction",
                             "details/masking/fallback_fraction", "details/masking/effective_observable_fraction",
                             "details/masking/branch_retention/geometry"}.issubset(mask_scalars))
            self.assertEqual(fixture.base_close_calls, [True])

    def test_control_transfer_result_validates_checkpoint_population_and_metrics(self):
        path = self.root / "transfer"
        cells = path / "cells/road-waymo/geometry+rgb+lidar"
        cells.mkdir(parents=True)
        (path / "status.json").write_text(json.dumps({"status": "complete"}))

        def predictions(locators, labels):
            tracks = []
            for locator, y in zip(locators, labels):
                logits = np.full((len(y), 4), -5., dtype=np.float32)
                logits[np.arange(len(y)), y] = 5.
                tracks.append({"locator": locator, "archive": locator + ".npz", "group": locator,
                    "logits": logits, "targets": np.asarray(y, dtype=np.int64),
                    "gt_valid": np.ones(len(y), dtype=bool), "conditions": {}, "source_flags": {}})
            return tracks

        source_tracks = predictions(["source-a", "source-b"], [[0, 1], [2, 3]])
        target_locators = ["target-a", "target-b"]
        target_tracks = predictions(target_locators, [[1, 2], [3, 0]])
        source_metrics = prediction_metrics(source_tracks, 4)
        target_metrics = prediction_metrics(target_tracks, 4)
        save_predictions(cells / "predictions.npz", target_tracks)
        checkpoint_sha = "canonical-control-checkpoint"
        model = {"training_source": "road-waymo", "input": "geometry+rgb+lidar",
                 "checkpoint_sha256": checkpoint_sha,
                 "source_validation": {"track_weighted_macro_f1": source_metrics["track_weighted"]["macro_f1"],
                                       "track_ce": source_metrics["track_ce"]}}
        target = {"training_source": "road-waymo", "target_source": "loki", "input": "geometry+rgb+lidar",
                  "checkpoint_sha256": checkpoint_sha, "target_tracks": len(target_tracks),
                  "target_f1": target_metrics["track_weighted"]["macro_f1"], "target_ce": target_metrics["track_ce"]}
        manifest = {"status": "frozen-before-cross-dataset-validation-inference",
                    "evaluation": {"held_out_tests_accessed": False}, "models": [model]}
        manifest_path = path / "transfer-comparison.json"
        results_path = path / "results.json"
        manifest_path.write_text(json.dumps(manifest))
        results_path.write_text(json.dumps([target]))
        control = {"sha256": checkpoint_sha, "validation_metrics": source_metrics}

        result = e003a_run._control_transfer_result(path, "road-waymo", "loki", control, set(target_locators))
        self.assertEqual(result["target_predictions_sha256"], e003a_run.checksum(cells / "predictions.npz"))
        self.assertEqual(result["target_support"], target_metrics["denominators"])
        self.assertEqual(result["target"], target)

        changed = dict(target, checkpoint_sha256="different-checkpoint")
        results_path.write_text(json.dumps([changed]))
        with self.assertRaisesRegex(ValueError, "canonical E002d control checkpoint"):
            e003a_run._control_transfer_result(path, "road-waymo", "loki", control, set(target_locators))

        results_path.write_text(json.dumps([target]))
        save_predictions(cells / "predictions.npz", target_tracks[:1])
        with self.assertRaisesRegex(ValueError, "do not cover the target population"):
            e003a_run._control_transfer_result(path, "road-waymo", "loki", control, set(target_locators))

        save_predictions(cells / "predictions.npz", target_tracks)
        bad_metrics = dict(target, target_ce=target["target_ce"] + 1.)
        results_path.write_text(json.dumps([bad_metrics]))
        with self.assertRaisesRegex(ValueError, "metrics disagree with local predictions"):
            e003a_run._control_transfer_result(path, "road-waymo", "loki", control, set(target_locators))


if __name__ == "__main__":
    unittest.main()
