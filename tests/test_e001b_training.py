"""E001b source-only inputs, shared-run evidence, bbox slices and seed matrix."""

import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from pedestrian_behavior.data.loading import track_batches
from pedestrian_behavior.data.preparation import checksum, prepare_collection
from pedestrian_behavior.datasets.loki import TrackReader as LokiReader
from pedestrian_behavior.datasets.road_waymo import TrackReader as RoadReader
from pedestrian_behavior.evaluation import load_predictions, predict, prediction_metrics, stratified_metrics
from pedestrian_behavior.experiments.e001 import STRATA as E001_STRATA, setup
from pedestrian_behavior.experiments.e001b import dataset_from_extension, prepare_bbox_extension
from pedestrian_behavior.experiments.e001b_run import STRATA, bbox_conditions, build_model, run_attempt
from pedestrian_behavior.experiments.e001b_sweep import attempts, attempt_command
from pedestrian_behavior.training import reload_checkpoint
import test_e001b
import test_e001_sweep


class FakeRun:
    id, url = "fixture", "fixture"
    def __init__(self, events, fail=None):
        self.events, self.fail = events, fail
        self.config = SimpleNamespace(update=lambda *a, **k: None)
    def define_metric(self, *a, **k):
        pass
    def log(self, value, **kwargs):
        self.events.append(value)
        if self.fail == "final" and any(k.startswith("eval/") for k in value):
            raise RuntimeError("final logging failure")
    def finish(self, **kwargs):
        pass


def fixture(root):
    loki, road = test_e001b.BboxTest().fixture(root)
    for i in (1, 2):
        shutil.copytree(loki.root/"scenario_000", loki.root/f"scenario_{i:03d}")
    manifest_path = road.index/"scene_manifest.json"
    scene = json.loads(manifest_path.read_text())[0]
    # Synthetic clips share components only to check wiring; no real-data comparison.
    manifest_path.write_text(json.dumps([scene | {"road_clip_id": f"clip{i}"} for i in range(3)]))
    csv_path = road.index/"pedestrians.csv.gz"
    with gzip.open(csv_path, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with gzip.open(csv_path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(row | {"road_clip_id": f"clip{i}"} for i in range(3) for row in rows)
    policy = root/"road-policy.json"
    policy.write_text(json.dumps({"schema_version": 1, "experiment": "E001", "dataset": "road-waymo",
        "native_csv_sha256": checksum(csv_path), "overrides": []}))
    base, extended = root/"E001", root/"E001b"
    for reader in (LokiReader(loki.root, root/"fixture-transform.json"), RoadReader(road.index)):
        d = reader.dataset
        native = reader.root if d == "loki" else reader.index
        collection, setup_dir = base/"reader-preparation"/d, base/"data-setup"/d
        prepare_collection(reader, collection)
        setup(collection, native, setup_dir, None if d == "loki" else policy)
        prepare_bbox_extension(collection, setup_dir/"setup.json", native, extended/"bbox-native"/d)
    return base, extended


class E001bTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = []
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)
    def tearDown(self):
        if output := os.environ.get("E001_ACCEPTANCE_OUTPUT"):
            Path(output).mkdir(parents=True, exist_ok=True)
            (Path(output)/(self._testMethodName+".json")).write_text(json.dumps(self.evidence, indent=2)+"\n")
    def check(self, name, expected, actual):
        self.evidence.append({"check": name, "expected": expected, "actual": actual})
        self.assertEqual(expected, actual)

    def test_bbox_diagnostics_and_twenty_attempt_matrix(self):
        for mask, cohort in (([False, False], 0), ([True, False], 1), ([True, True], 2)):
            cond = bbox_conditions({"bbox_valid": np.array(mask), "bbox_velocity_valid": np.zeros(2, dtype=bool)})
            self.check(f"bbox track cohort {cohort}", [cohort]*2, cond["track-bbox"].tolist())
        self.check("E001 axes preserved", E001_STRATA, {k: STRATA[k] for k in E001_STRATA})
        track = {"logits": np.eye(4, dtype=np.float32)[[0, 3, 1]], "targets": np.array([0, 3, 1]),
                 "gt_valid": np.ones(3, dtype=bool), "group": "fixture"}
        track["conditions"] = bbox_conditions({"bbox_valid": np.array([False, True, False]),
                                               "bbox_velocity_valid": np.zeros(3, dtype=bool)})
        slices = stratified_metrics([track], 4, {k: STRATA[k] for k in track["conditions"]})
        self.check("verified box slice supports crossing only", [1., [3], [0, 0, 0, 1]],
                   [slices["frame-bbox"]["valid"]["track_weighted"]["macro_f1"],
                    slices["frame-bbox"]["valid"]["track_weighted"]["supported_classes"],
                    slices["frame-bbox"]["valid"]["denominators"]["class_frames"]])
        self.check("empty velocity slice remains unscored", None, slices["bbox-velocity"]["valid"]["track_weighted"]["macro_f1"])
        plan = attempts(range(5))
        self.check("twenty unique BiLSTM attempts", [20, 20, ["B"]],
                   [len(plan), len({r["name"] for r in plan}), sorted({r["variant"] for r in plan})])
        options = dict(zip(attempt_command(plan[-1], Path("/tmp/runs"), "cuda")[2::2],
                           attempt_command(plan[-1], Path("/tmp/runs"), "cuda")[3::2]))
        self.check("explicit seed and budget", ["4", "75", "8", "geometry"],
                   [options[k] for k in ("--seed", "--epochs", "--patience", "--configuration")])

    def test_full_attempt_source_statistics_reload_predictions_and_wandb(self):
        base, extended = fixture(self.root)
        caches = {d: json.loads((extended/"bbox-native"/d/"manifest.json").read_text()) for d in ("loki", "road-waymo")}
        before = {p: checksum(p) for p in base.rglob("*") if p.is_file()}
        for source in caches:
            for configuration, width in (("availability", 14), ("geometry", 20)):
                with self.subTest(source=source, configuration=configuration):
                    output = self.root/(source+"-"+configuration)
                    events, init_args = [], []
                    def initialize(**kwargs):
                        init_args.append(kwargs)
                        return FakeRun(events)
                    with patch("pedestrian_behavior.experiments.e001b_run.E001_ROOT", base), \
                         patch("pedestrian_behavior.experiments.e001b_run.E001B_ROOT", extended), \
                         patch("pedestrian_behavior.experiments.run.wandb.init", side_effect=initialize):
                        fitted = run_attempt(source, configuration, output, "cpu", seed=3, epochs=2, patience=8)
                    self.check(source+configuration+" source selection", [2, "budget"], [len(fitted["history"]), fitted["stopping_reason"]])
                    checkpoint = reload_checkpoint(output/"best.pt", build_model(configuration).eval(), "cpu")
                    expected_statistics = {"baseline": caches[source]["baseline_normalization"], "bbox": caches[source]["normalization"]}
                    self.check("checkpoint retains both SOURCE statistics", expected_statistics, checkpoint["normalization"])
                    self.assertEqual(checkpoint["config"]["model"]["input_dim"], width)
                    self.assertRegex(init_args[0]["name"], rf"^E001b-{source}-{configuration}-BiLSTM-seed3-\d{{8}}-\d{{6}}$")
                    self.assertEqual(init_args[0]["config"]["experiment"], "E001b")
                    provenance = json.loads((output/"provenance.json").read_text())
                    self.assertEqual(provenance["source_statistics"], expected_statistics)
                    self.assertTrue(all("bbox" in d for d in provenance["datasets"].values()))
                    model = build_model(configuration).eval()
                    reload_checkpoint(output/"best.pt", model, "cpu")
                    for d in caches:
                        destination = output/(d+"-test")
                        tracks = load_predictions(destination/"predictions.npz")
                        metrics = json.loads((destination/"metrics.json").read_text())
                        self.check("saved predictions regenerate all metrics", [metrics["primary"], metrics["strata"]],
                                   [prediction_metrics(tracks, 4), stratified_metrics(tracks, 4, STRATA)])
                        ds = dataset_from_extension(base/"reader-preparation"/d, base/"data-setup"/d/"setup.json",
                             extended/"bbox-native"/d, "test", configuration, expected_statistics["baseline"], expected_statistics["bbox"])
                        replay = predict(model, track_batches(ds, seed=3), torch.device("cpu"))
                        for old, new in zip(tracks, replay):
                            np.testing.assert_array_equal(old["logits"], new["logits"])
                            self.assertEqual(set(old["conditions"]), set(STRATA))
                            self.assertTrue({"bbox_valid", "bbox_velocity_valid", "bbox_source"} <= set(old["source_flags"]))
                            self.assertTrue(Path(old["bbox_archive"]).is_file())
                        self.assertTrue((output/"plots"/(d+"-frame-bbox.svg")).exists())
                    self.check("one record per epoch and optimizer update", [2, 2],
                               [sum("epoch" in r for r in events), sum("train/loss_step" in r for r in events)])
                    final = events[-1]
                    self.assertTrue(all(k.startswith(("eval/", "strata/", "details/")) for k in final))
                    self.assertFalse(any("predictions" in k or ".pt" in k for k in final))
                    self.assertTrue(any(k.endswith("/track-bbox") for k in final))
                    with self.assertRaises(FileExistsError):
                        run_attempt(source, configuration, output, "cpu")
        self.check("frozen E001 files unchanged", True, all(checksum(p) == h for p, h in before.items()))

    def test_e001b_wrapper_runs_bounded_child_processes(self):
        from pedestrian_behavior.experiments.e001b_sweep import run_sweep
        plan = attempts([0])[:2]
        def command(row, destination, device):
            return [sys.executable, "-c", test_e001_sweep.CHILD, str(destination/row["name"]), "0"]
        output = self.root/"sweep"
        with patch("pedestrian_behavior.experiments.e001b_sweep.attempts", return_value=plan), \
             patch("pedestrian_behavior.experiments.e001b_sweep.attempt_command", side_effect=command):
            state = run_sweep(output, jobs=2, seeds=[0], device="cpu")
        self.check("E001b sweep uses shared child scheduler", ["complete", 2, 2],
                   [state["status"], state["jobs"], len(list((output/"logs").glob("*.log")))])

    def test_cache_and_wandb_failures_preserve_local_evidence(self):
        base, extended = fixture(self.root)
        for stage in ("init", "final", "old-target-policy"):
            output = self.root/stage
            target = extended/"bbox-native/road-waymo/manifest.json"
            original = target.read_text()
            if stage == "old-target-policy":
                record = json.loads(original);record["policy"]["road_source_rule"] = "ROAD preferred"
                target.write_text(json.dumps(record))
            events = []
            def initialize(**kwargs):
                if stage == "init":
                    raise RuntimeError("init logging failure")
                return FakeRun(events, stage)
            try:
                with patch("pedestrian_behavior.experiments.e001b_run.E001_ROOT", base), \
                     patch("pedestrian_behavior.experiments.e001b_run.E001B_ROOT", extended), \
                     patch("pedestrian_behavior.experiments.run.wandb.init", side_effect=initialize) as external:
                    with self.assertRaisesRegex(ValueError if stage == "old-target-policy" else RuntimeError,
                                                "mismatched bbox" if stage == "old-target-policy" else "logging failure"):
                        run_attempt("loki", "geometry", output, "cpu", epochs=1)
                    if stage == "old-target-policy":
                        self.assertFalse(external.called)
                self.check(stage+" retained failure", "failed", json.loads((output/"status.json").read_text())["status"])
                if stage == "final":
                    self.assertTrue(all((output/name).is_file() for name in ("best.pt", "last.pt", "history.json", "steps.jsonl")))
                    self.assertTrue(all((output/(d+"-test")/"predictions.npz").is_file() for d in ("loki", "road-waymo")))
            finally:
                target.write_text(original)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA acceptance runs on aalto")
    def test_cuda_smoke_uses_source_validation_only(self):
        base, extended = fixture(self.root)
        output = self.root/"smoke"
        events = []
        with patch("pedestrian_behavior.experiments.e001b_run.E001_ROOT", base), \
             patch("pedestrian_behavior.experiments.e001b_run.E001B_ROOT", extended), \
             patch("pedestrian_behavior.experiments.run.wandb.init", return_value=FakeRun(events)):
            fitted = run_attempt("road-waymo", "geometry", output, "cuda", smoke=True)
        self.check("smoke stopped after one epoch", [1, "budget"], [len(fitted["history"]), fitted["stopping_reason"]])
        self.check("held-out tests untouched by smoke", ["road-waymo-validation-smoke"],
                   sorted(p.name for p in output.iterdir() if p.is_dir() and (p/"metrics.json").is_file()))
