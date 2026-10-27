from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from pedestrian_behavior.data.preparation import checksum
from pedestrian_behavior.evaluation import prediction_metrics
from pedestrian_behavior.experiments import e002b_checkpoint_evaluation as evaluation
from pedestrian_behavior.experiments.e002b_group import AXES


class CheckpointEvaluationTest(unittest.TestCase):
    def test_profiled_singleton_may_exceed_total_tokens_but_not_shape_bounds(self):
        profile = {"max_tracks": 64, "temporal_cells": 665856,
                   "microbatch_tokens": 1460024, "frame_tokens": 524288}
        self.assertTrue(evaluation._validate_packed_shape(1, 101**2, 1499487, 15856, profile))
        with self.assertRaisesRegex(ValueError, "resource bounds"):
            evaluation._validate_packed_shape(2, 2*101**2, 1499487, 15856, profile)
        with self.assertRaisesRegex(ValueError, "resource bounds"):
            evaluation._validate_packed_shape(1, profile["temporal_cells"] + 1, 100, 10, profile)
        with self.assertRaisesRegex(ValueError, "resource bounds"):
            evaluation._validate_packed_shape(1, 101**2, 100, profile["frame_tokens"] + 1, profile)

    def test_checkpoint_metrics_require_full_confusion_and_loss_parity(self):
        metrics = {
            "track_ce": .75,
            "track_weighted": {"macro_f1": .5, "confusion": [[3, 1], [2, 4]]},
            "denominators": {"tracks": 2},
        }
        result = evaluation.verify_checkpoint_metrics(metrics, {
            "track_ce": .75 + 1e-10,
            "track_weighted": {"macro_f1": .5, "confusion": [[3, 1], [2, 4]]},
            "denominators": {"tracks": 2},
        })
        self.assertTrue(result["matches_checkpoint"])
        self.assertAlmostEqual(result["track_weighted_macro_f1"], .5)
        with self.assertRaisesRegex(ValueError, "track_weighted.confusion"):
            evaluation.verify_checkpoint_metrics(metrics, {
                "track_ce": .75,
                "track_weighted": {"macro_f1": .5, "confusion": [[4, 0], [2, 4]]},
                "denominators": {"tracks": 2},
            })
        with self.assertRaisesRegex(ValueError, "track_ce"):
            evaluation.verify_checkpoint_metrics(metrics | {"track_ce": .9}, metrics)

    def test_evaluation_writes_local_exports_before_finishing_fresh_wnab_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"outputs"
            original = root/"experiments/E002b/runs"/evaluation.EXPECTED_ORIGINAL
            original.mkdir(parents=True)
            (original/"provenance.json").write_text("{}\n")
            (original/"status.json").write_text('{"status":"incomplete"}\n')
            output = root/"experiments/E002b/runs/checkpoint-evaluation-test"
            source_file = Path(temporary)/"source-manifest.json"
            source_file.write_text("frozen source\n")

            normalization = {"baseline": {"center": 1.}, "bbox": {"center": 2.}}
            source_reference = {"setup_sha256": "s", "collection_sha256": "c", "policy_sha256": "p",
                "native_policy": {}, "population": {}, "support": {},
                "bbox": {"manifest_sha256": "b", "policy_sha256": "bp", "coverage": {}}}
            point_identity = {"protocol": "indexed-v1", "manifest_sha256": "point-manifest", "root": "/read-only/points"}
            profile = {"max_tracks": 64, "temporal_cells": 665856, "microbatch_tokens": 1460024,
                "ordinary_tokens": 1810000, "frame_chunk": 4, "frame_tokens": 524288}
            digest = "calibration-sha"

            raw_track = {"logits": np.array([[2., 0., 0., 0.]], np.float32),
                "targets": np.array([0], np.int64), "gt_valid": np.array([True]),
                "locator": '{"scenario":"s","track_id":"t"}', "archive": "track.npz", "group": "scenario"}
            expected = prediction_metrics([raw_track.copy()], 4)
            checkpoints, configs, histories, origins = {}, {}, {}, {}
            for variant, epoch in evaluation.BEST_EPOCHS.items():
                checkpoints[variant] = {"model_state": {}, "normalization": normalization,
                    "epoch": epoch, "selection": {"epoch": epoch, "ce": expected["track_ce"],
                        "f1": expected["track_weighted"]["macro_f1"]}, "validation": expected}
                configs[variant] = {"experiment": "E002b", "source": "loki", "arm": "g-dditr",
                    "model_variant": variant, "model": {"variant": variant}, "settings": {"seed": 0,
                        "epochs": 100, "patience": 15}, "profile": profile, "profile_sha256": digest,
                    "point_storage_protocol": "indexed-v1", "point_loading": {"open_scenes": 8,
                        "metadata_limit_bytes": 1000, "frame_workers": 1},
                    "indexed_point_cache": point_identity, "held_out_tests": False}
                histories[variant] = [{"epoch": i, "training_ce": 1., "validation_ce": 1.,
                    "validation_f1": .1, "selected": i == epoch} for i in range(1, 24)]
                origins[variant] = {"run_id": "original-"+variant, "run_url": "https://wandb.ai/original/"+variant,
                    "best_checkpoint_sha256": "best-"+variant, "last_checkpoint_sha256": "last-"+variant,
                    "selected_epoch": epoch, "completed_training_epochs": 23}
            old_provenance = {"profile_sha256": digest, "execution_profile": profile,
                "source_statistics": normalization, "datasets": {"loki": source_reference},
                "code_sha256": {}}
            old_status = {"status": "incomplete", "execution": "shared-g-dditr", "failures": []}
            point_reader = Mock()
            point_reader.identity = point_identity
            point_reader.snapshot.return_value = {"read_only": True, "track_reads": 1}
            catalog = Mock()
            catalog.identity = {"dataset": "loki", "tracks_sha256": "tracks"}
            prepared = {"normalization": normalization, "datasets": {"loki": source_reference},
                "catalog": catalog, "points": point_reader, "dataset": object(),
                "packing": {"physical_batches": 1}, "source_files_sha256": {str(source_file): checksum(source_file)}}
            logical = Mock()
            logical.stats = {"prepared_seconds": .1}

            class Model:
                def to(self, device):
                    return self

                def load_state_dict(self, state, strict=True):
                    return None

                def eval(self):
                    return self

            class FakeRun:
                def __init__(self, index):
                    self.id = f"evaluation-{index}"
                    self.url = f"https://wandb.ai/evaluation/{index}"
                    self.define_metric = Mock()
                    self.finish = Mock()

            runs = [FakeRun(i) for i in range(3)]

            def decorate(tracks, prepared, cache, source):
                for track in tracks:
                    track.update({"conditions": {axis: np.zeros(1, dtype=np.int64) for axis in AXES},
                        "source_flags": {"lidar_valid": np.ones(1, dtype=bool)},
                        "source_times_us": [0], "native_extent_us": [0, 200000], "grid_times_us": [0]})

            def make_tracks(models, batches, device):
                return {variant: [raw_track.copy()] for variant in evaluation.BEST_EPOCHS}

            def save_plots(directory, history, selection, evaluations, classes):
                directory = Path(directory)
                directory.mkdir(parents=True)
                image = directory/"validation.png"
                image.write_bytes(b"plot")
                return [image]

            group_provenance = old_provenance | {"wandb": {v: {"id": origins[v]["run_id"],
                "url": origins[v]["run_url"]} for v in origins}}
            snapshot = {str(path): checksum(path) for path in
                (original/"status.json", original/"provenance.json")}
            with ExitStack() as stack:
                patches = [
                    patch.object(evaluation, "_load_original", return_value=(original, old_status,
                        group_provenance, checkpoints, configs, histories, origins)),
                    patch.object(evaluation, "_snapshot_original", return_value=snapshot),
                    patch.object(evaluation, "_verify_frozen_code", return_value={}),
                    patch.object(evaluation, "_validate_calibration", return_value=(profile,
                        {"gpu": "fake-gpu", "memory_preflight": {"estimated_group_peak_bytes": 0}},
                        Path(temporary)/"profile.json", digest)),
                    patch.object(evaluation, "_prepare_validation", return_value=prepared),
                    patch.object(evaluation, "EXPECTED_VALIDATION_TRACKS", 1),
                    patch.object(evaluation, "seed_run", return_value={"seed": 0}),
                    patch.object(evaluation, "LogicalBatches", return_value=logical),
                    patch.object(evaluation, "predict_shared", side_effect=make_tracks),
                    patch.object(evaluation, "_decorate", side_effect=decorate),
                    patch.object(evaluation, "plots", side_effect=save_plots),
                    patch.object(evaluation, "log_final"),
                    patch.object(evaluation.e002b, "build_model", side_effect=lambda *args: Model()),
                    patch.object(evaluation.torch, "set_num_threads"),
                    patch.object(evaluation.torch.cuda, "is_available", return_value=True),
                    patch.object(evaluation.torch.cuda, "get_device_name", return_value="fake-gpu"),
                    patch.object(evaluation.torch.cuda, "mem_get_info", return_value=(20*2**30, 24*2**30)),
                    patch.object(evaluation.torch.cuda, "reset_peak_memory_stats"),
                    patch.object(evaluation.torch.cuda, "synchronize"),
                    patch.object(evaluation.torch.cuda, "max_memory_allocated", return_value=123),
                    patch.object(evaluation.torch.cuda, "max_memory_reserved", return_value=456),
                    patch.object(evaluation.wandb, "init", side_effect=runs),
                ]
                active = [stack.enter_context(item) for item in patches]
                log_final, wandb_init = active[11], active[-1]
                result = evaluation.run_evaluation(outputs_root=root, original_root=original,
                    profile_path=Path(temporary)/"profile.json", output_root=output)
                self.assertEqual(set(result["wandb"]), set(evaluation.BEST_EPOCHS))
                self.assertEqual(wandb_init.call_count, 3)
                self.assertEqual(log_final.call_count, 3)
                self.assertTrue(all(run.finish.call_count == 1 and any(call.args[0] == "details/*" and
                    call.kwargs.get("hidden") for call in run.define_metric.call_args_list) and
                    any(call.args[0] == "train/macro_f1" and call.kwargs.get("step_metric") == "epoch"
                        for call in run.define_metric.call_args_list) for run in runs))

                failed_output = root/"experiments/E002b/runs/checkpoint-evaluation-log-failure"
                failed_runs = [FakeRun(i+3) for i in range(3)]
                wandb_init.reset_mock(side_effect=True)
                wandb_init.side_effect = failed_runs
                log_final.reset_mock(side_effect=True)

                def fail_on_second_run(*args, **kwargs):
                    if log_final.call_count == 2:
                        raise RuntimeError("mocked W&B logging failure")

                log_final.side_effect = fail_on_second_run
                with self.assertRaisesRegex(RuntimeError, "mocked W&B logging failure"):
                    evaluation.run_evaluation(outputs_root=root, original_root=original,
                        profile_path=Path(temporary)/"profile.json", output_root=failed_output)

            self.assertEqual(json.loads((output/"status.json").read_text())["status"], "complete")
            self.assertEqual(json.loads((original/"status.json").read_text()), {"status": "incomplete"})
            for variant in evaluation.BEST_EPOCHS:
                child = output/variant
                self.assertTrue((child/"loki-validation/predictions.npz").is_file())
                self.assertTrue((child/"loki-validation/metrics.json").is_file())
                self.assertTrue((child/"plots/validation.png").is_file())
                self.assertEqual(json.loads((child/"status.json").read_text())["status"], "complete")
                self.assertTrue(json.loads((child/"provenance.json").read_text())["evaluation_only"])
                self.assertFalse((child/"best.pt").exists())
                failed_child = failed_output/variant
                self.assertTrue((failed_child/"loki-validation/predictions.npz").is_file())
                self.assertTrue((failed_child/"loki-validation/metrics.json").is_file())
                self.assertTrue((failed_child/"plots/validation.png").is_file())
                self.assertEqual(json.loads((failed_child/"status.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((failed_output/"status.json").read_text())["status"], "failed")
            self.assertEqual(json.loads((failed_output/"evaluation-evidence.json").read_text())["status"], "wandb-failed")

    def test_output_must_be_a_fresh_child_of_e002b_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/"outputs"
            original = root/"experiments/E002b/runs"/evaluation.EXPECTED_ORIGINAL
            with self.assertRaisesRegex(ValueError, "fresh children"):
                evaluation.run_evaluation(outputs_root=root, original_root=original,
                    profile_path=Path(temporary)/"profile.json", output_root=Path(temporary)/"outside")


if __name__ == "__main__":
    unittest.main()
