"""Thin E001c integration: frozen data, source statistics, replay and launcher."""

import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import numpy as np
import torch

from pedestrian_behavior.data.boxes import FLAGS, INPUT_COLUMNS, load_boxes
from pedestrian_behavior.data.loading import track_batches
from pedestrian_behavior.data.preparation import checksum, load_track, prepare_collection
from pedestrian_behavior.datasets.loki import TrackReader as LokiReader
from pedestrian_behavior.datasets.road_waymo import TrackReader as RoadReader
from pedestrian_behavior.evaluation import load_predictions, predict, prediction_metrics, stratified_metrics
from pedestrian_behavior.experiments.e001 import setup
from pedestrian_behavior.experiments.e001c import dataset_from_cache, load_cache, prepare_boxes
from pedestrian_behavior.experiments.e001c_run import INPUT_DIMENSIONS, STRATA, build_model, run_attempt
from pedestrian_behavior.experiments.e001c_sweep import attempts, attempt_command, run_sweep
from pedestrian_behavior.training import reload_checkpoint
import test_e001c
import test_e001_sweep
from test_e001b_training import FakeRun


def fixture(root):
    _, (loki, road) = test_e001c.E001cTest().fixture(root)
    for i in (1, 2):
        shutil.copytree(loki.root/"scenario_000", loki.root/f"scenario_{i:03d}")
    manifest_path = road.index/"scene_manifest.json"
    scene = json.loads(manifest_path.read_text())[0]
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
    base, cache = root/"E001", root/"E001c"
    for reader in (LokiReader(loki.root, root/"fixture-transform.json"), RoadReader(road.index)):
        dataset = reader.dataset
        native = reader.root if dataset == "loki" else reader.index
        collection, setup_dir = base/"reader-preparation"/dataset, base/"data-setup"/dataset
        prepare_collection(reader, collection)
        setup(collection, native, setup_dir, None if dataset == "loki" else policy)
        prepare_boxes(collection, setup_dir/"setup.json", native, cache/"box-cache"/dataset,
                      loki_transform=root/"fixture-transform.json" if dataset == "loki" else None)
    return base, cache


class E001cTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, threads)

    def test_thirty_attempts_explicit_settings_and_shared_scheduler(self):
        plan = attempts(range(5))
        self.assertEqual((len(plan), len({row["name"] for row in plan})), (30, 30))
        self.assertEqual({r["variant"] for r in plan}, {"B"})
        self.assertEqual(INPUT_DIMENSIONS, {"boxes": 19, "boxes-2d-size-vel": 21, "boxes-2d-3d-size-vel": 24})
        options = dict(zip(attempt_command(plan[-1], Path("/tmp/runs"), "cuda")[2::2],
                           attempt_command(plan[-1], Path("/tmp/runs"), "cuda")[3::2]))
        self.assertEqual([options[k] for k in ("--seed", "--epochs", "--patience", "--configuration")],
                         ["4", "75", "8", "boxes-2d-3d-size-vel"])
        def command(row, destination, device):
            return [sys.executable, "-c", test_e001_sweep.CHILD, str(destination/row["name"]), "0"]
        with patch("pedestrian_behavior.experiments.e001c_sweep.attempts", return_value=plan[:2]), \
             patch("pedestrian_behavior.experiments.e001c_sweep.attempt_command", side_effect=command):
            state = run_sweep(self.root/"sweep", jobs=2, seeds=[0], device="cpu")
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(list((self.root/"sweep/logs").glob("*.log"))), 2)

    def test_all_sources_and_configurations_checkpoints_metrics_and_wandb(self):
        base, cache = fixture(self.root)
        manifests = {d: json.loads((cache/"box-cache"/d/"manifest.json").read_text()) for d in ("loki", "road-waymo")}
        before = {p: checksum(p) for root in (base, cache) for p in root.rglob("*") if p.is_file()}
        for source in manifests:
            for configuration, width in INPUT_DIMENSIONS.items():
                with self.subTest(source=source, configuration=configuration):
                    output = self.root/(source+"-"+configuration)
                    events, initialized = [], []
                    def initialize(**kwargs):
                        initialized.append(kwargs)
                        return FakeRun(events)
                    with patch("pedestrian_behavior.experiments.e001c_run.E001_ROOT", base), \
                         patch("pedestrian_behavior.experiments.e001c_run.E001C_ROOT", cache), \
                         patch("pedestrian_behavior.experiments.run.wandb.init", side_effect=initialize), \
                         patch("pedestrian_behavior.experiments.run.plots", return_value=[]):
                        result = run_attempt(source, configuration, output, "cpu", seed=3, epochs=2, patience=8)
                    self.assertEqual((len(result["history"]), result["stopping_reason"]), (2, "budget"))
                    statistics = manifests[source]["normalization"]
                    model = build_model(configuration).eval()
                    checkpoint = reload_checkpoint(output/"best.pt", model, "cpu")
                    self.assertEqual(checkpoint["normalization"], statistics)
                    self.assertEqual(checkpoint["config"]["model"]["input_dim"], width)
                    self.assertEqual(tuple(checkpoint["config"]["inputs"]["flags"]), FLAGS)
                    self.assertEqual(len(checkpoint["config"]["inputs"]["numeric_columns"]), INPUT_COLUMNS[configuration])
                    self.assertRegex(initialized[0]["name"], rf"^E001c-{source}-{configuration}-BiLSTM-seed3-\d{{8}}-\d{{6}}$")
                    provenance = json.loads((output/"provenance.json").read_text())
                    self.assertEqual(provenance["source_statistics"], statistics)
                    self.assertTrue(all("boxes" in d for d in provenance["datasets"].values()))
                    for dataset in manifests:
                        destination = output/(dataset+"-test")
                        tracks = load_predictions(destination/"predictions.npz")
                        metrics = json.loads((destination/"metrics.json").read_text())
                        self.assertEqual(metrics["primary"], prediction_metrics(tracks, 4))
                        self.assertEqual(metrics["strata"], stratified_metrics(tracks, 4, STRATA))
                        ds = dataset_from_cache(base/"reader-preparation"/dataset, base/"data-setup"/dataset/"setup.json",
                                                cache/"box-cache"/dataset, "test", configuration, statistics)
                        self.assertEqual(ds.statistics, statistics)
                        sample = ds[0]
                        metadata, _ = load_track(sample["archive"])
                        entry = ds.entries[0]
                        _, arrays = load_boxes(cache/"box-cache"/dataset/"tracks"/entry["archive"], metadata, entry["e001_sha256"])
                        expected = (arrays["values"]-np.array(statistics["mean"]))/np.array(statistics["scale"])
                        expected = np.nan_to_num(expected[:, :INPUT_COLUMNS[configuration]], nan=0).astype(np.float32)
                        np.testing.assert_array_equal(sample["inputs"][:, :width-4], expected)
                        replay = predict(model, track_batches(ds, seed=3), torch.device("cpu"))
                        for saved, actual in zip(tracks, replay):
                            np.testing.assert_array_equal(saved["logits"], actual["logits"])
                            self.assertEqual(set(saved["conditions"]), set(STRATA))
                            self.assertTrue(set(FLAGS) <= set(saved["source_flags"]))
                            self.assertTrue(Path(saved["boxes_archive"]).is_file())
                    self.assertEqual(sum("epoch" in e for e in events), 2)
                    self.assertEqual(sum("train/loss_step" in e for e in events), 2)
                    self.assertTrue(all(k.startswith(("eval/", "strata/", "details/")) for k in events[-1]))
                    self.assertTrue(any("strata/track-bbox" in k for k in events[-1]))
                    self.assertFalse(any("prediction" in k or ".pt" in k for k in events[-1]))
                    with self.assertRaises(FileExistsError):
                        run_attempt(source, configuration, output, "cpu")
        self.assertTrue(all(checksum(path) == digest for path, digest in before.items()))

    def test_cache_and_external_logging_failures_keep_local_evidence(self):
        base, cache = fixture(self.root)
        path = cache/"box-cache/road-waymo/manifest.json"
        original = path.read_text()
        for stage in ("policy", "population", "init", "final"):
            output = self.root/stage
            record = json.loads(original)
            if stage == "policy":
                record["policy"]["sources2d"] = "ROAD fallback"
            if stage == "population":
                record["tracks"][0]["split"] = "training" if record["tracks"][0]["split"] != "training" else "validation"
            path.write_text(json.dumps(record))
            def initialize(**kwargs):
                if stage == "init":
                    raise RuntimeError("init logging failure")
                return FakeRun([], stage)
            try:
                with patch("pedestrian_behavior.experiments.e001c_run.E001_ROOT", base), \
                     patch("pedestrian_behavior.experiments.e001c_run.E001C_ROOT", cache), \
                     patch("pedestrian_behavior.experiments.run.wandb.init", side_effect=initialize) as external, \
                     patch("pedestrian_behavior.experiments.run.plots", return_value=[]):
                    with self.assertRaisesRegex(ValueError if stage in ("policy", "population") else RuntimeError,
                                                "E001c cache|population" if stage in ("policy", "population") else "logging failure"):
                        run_attempt("loki", "boxes", output, "cpu", epochs=1)
                    if stage in ("policy", "population"):
                        self.assertFalse(external.called)
                self.assertEqual(json.loads((output/"status.json").read_text())["status"], "failed")
                if stage == "final":
                    self.assertTrue((output/"best.pt").is_file())
                    self.assertTrue((output/"road-waymo-test/predictions.npz").is_file())
            finally:
                path.write_text(original)

    def test_archive_corruption_and_cached_target_disagreement_fail(self):
        base, cache = fixture(self.root)
        collection, setup_path, folder = base/"reader-preparation/loki", base/"data-setup/loki/setup.json", cache/"box-cache/loki"
        record = load_cache(collection, setup_path, folder)
        ds = dataset_from_cache(collection, setup_path, folder, "training", "boxes", record["normalization"])
        entry = ds.entries[0]
        path = folder/"tracks"/entry["archive"]
        metadata, _ = load_track(collection/"tracks"/entry["archive"])
        reference, arrays = load_boxes(path, metadata, entry["e001_sha256"])
        arrays["targets"][arrays["gt_valid"]] = (arrays["targets"][arrays["gt_valid"]]+1)%4
        np.savez_compressed(path, metadata=np.array(json.dumps(reference)), **arrays)
        with self.assertRaisesRegex(ValueError, "archive changed"):
            ds[0]
        entry["boxes_sha256"] = checksum(path)
        with self.assertRaisesRegex(ValueError, "targets differ"):
            ds[0]

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA acceptance runs on aalto")
    def test_cuda_smoke_does_not_evaluate_tests(self):
        base, cache = fixture(self.root)
        output = self.root/"smoke"
        with patch("pedestrian_behavior.experiments.e001c_run.E001_ROOT", base), \
             patch("pedestrian_behavior.experiments.e001c_run.E001C_ROOT", cache), \
             patch("pedestrian_behavior.experiments.run.wandb.init", return_value=FakeRun([])), \
             patch("pedestrian_behavior.experiments.run.plots", return_value=[]):
            run_attempt("road-waymo", "boxes", output, "cuda", smoke=True)
        self.assertTrue((output/"road-waymo-validation-smoke/metrics.json").is_file())
        self.assertFalse(any(output.glob("*-test")))


if __name__ == "__main__":
    unittest.main()
