import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pedestrian_behavior.experiments import e004a_run, e005_head


def _record(locator, group):
    return {"features": np.eye(4, 128, dtype=np.float32), "targets": np.arange(4),
        "gt_valid": np.ones(4, dtype=bool), "locator": locator, "archive": locator + ".npz",
        "group": group, "source_flags": {}}


class E005HeadTest(unittest.TestCase):
    def test_full_plan_uses_five_nested_recording_diverse_draws(self):
        training = [_record(f"track-{index:04d}", f"clip-{index % 40}") for index in range(1000)]
        validation = [_record(f"val-{index}-{track}", f"val-clip-{index}")
                      for index in range(78) for track in range(2)]
        development, assessment = e005_head._split_validation(validation)
        plan = e005_head._plan(training, development, assessment, "source", smoke=False)
        splits = plan["validation_split"]
        self.assertEqual(len(splits["development_groups"]), 39)
        self.assertEqual(len(splits["assessment_groups"]), 39)
        self.assertTrue(set(splits["development_groups"]).isdisjoint(splits["assessment_groups"]))
        self.assertEqual(len(plan["runs"]), 30)
        first_subsets = set()
        for draw in range(5):
            runs = [row for row in plan["runs"] if row["draw"] == draw]
            self.assertEqual([row["budget"] for row in runs], [25, 50, 100, 250, 500, 1000])
            self.assertEqual(len({row["group"] for row in runs[0]["selected_tracks"]}), 25)
            first_subsets.add(tuple(row["locator"] for row in runs[0]["selected_tracks"]))
            for smaller, larger in zip(runs, runs[1:]):
                self.assertEqual(smaller["selected_tracks"], larger["selected_tracks"][:smaller["budget"]])
        self.assertEqual(len(first_subsets), 5)

    def test_cached_smoke_writes_zero_and_fit_reports_without_wandb(self):
        training = [_record(f"train-{index:02d}", f"clip-{index % 2}") for index in range(26)]
        validation = [_record(f"val-{index}", f"val-clip-{index}") for index in range(2)]
        head = {"weight": np.eye(4, 128, dtype=np.float32), "bias": np.zeros(4, dtype=np.float32)}
        predictions = [{key: row[key] for key in ("locator", "archive", "group", "targets", "gt_valid", "source_flags")}
            | {"logits": row["features"] @ head["weight"].T + head["bias"]} for row in validation]
        manifest = {"source_checkpoint_sha256": "e005-checkpoint", "normalization": {"dataset": "loki"},
            "source_head": {"state_sha256": "e005-head"},
            "derived_features": {split: {"sha256": split} for split in ("training", "validation")}}
        with tempfile.TemporaryDirectory() as directory, patch.object(e004a_run.wandb, "init") as wandb_init:
            output = Path(directory) / "run"
            cache = {"records": {"training": training, "validation": validation}, "head": head,
                "source_validation": predictions, "root": Path(directory) / "cache", "manifest": manifest}
            summary = e005_head.run(output, cache, smoke=True)
            wandb_init.assert_not_called()
            self.assertEqual(summary["fits"], 1)
            self.assertEqual(summary["results"], 2)
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "complete")
            fit = output / "budget-0025-draw-00"
            config = json.loads((fit / "config.json").read_text())
            self.assertEqual((config["source"], config["target"], config["normalization_source"]),
                             ("loki", "road-waymo", "loki"))
            self.assertEqual(config["source_checkpoint_sha256"], "e005-checkpoint")
            self.assertEqual(config["settings"]["check_updates"], [0, 1, 5, 10, 25, 50, 100, 200])
            self.assertEqual(len((fit / "steps.jsonl").read_text().splitlines()), 200)
            report = json.loads((output / "report/budget-report.json").read_text())
            self.assertIsNone(report["all_tracks"])
            zero, adapted = report["summary"]
            self.assertEqual(zero["macro_f1_mean"], 1.0)
            self.assertEqual(adapted["classes"]["WAITING_TO_CROSS"]["f1"]["n"], 1)
            risk = adapted["targets"]["0.05"]
            self.assertIn("assessment_error_mean_nonempty", risk)
            self.assertIn("gt_class_retention", risk["classes"]["CROSSING"])
            self.assertTrue((output / "report/budget-coverage.png").is_file())


if __name__ == "__main__":
    unittest.main()
