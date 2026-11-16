import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pedestrian_behavior.experiments.e004c_analysis import make_report


def _coverage(empty=False, error=0.0, coverage=0.4):
    return {"targets": [{
        "error_target": target, "threshold": None if empty else 0.8, "reject_all": empty,
        "calibration": {"coverage": 0.0 if empty else 0.5, "error": None if empty else 0.0},
        "assessment": {"coverage": 0.0 if empty else coverage, "error": None if empty else error,
                       "accepted_frames": 0 if empty else 4, "error_frames": 0 if empty else 1,
                       "accepted_mass": 0.0 if empty else coverage, "error_mass": 0.0 if empty else error * coverage,
                       "classes": []},
    } for target in (0.01, 0.05, 0.10, 0.20)]}


def _row(method, budget, draw, macro_f1, class_f1, coverage, **extra):
    return {"method": method, "budget": budget, "draw": draw, "macro_f1": macro_f1,
            "class_f1": class_f1, "track_ce": 0.8,
            "coverage": coverage,
            "annotation_access": {"development_calibration": {"tracks": 930, "gt_timesteps": 29626}},
            **extra}


class E004cAnalysisTest(unittest.TestCase):
    def test_method_budget_report_keeps_reject_all_and_class_variation(self):
        rows = [_row("source", 0, None, 0.3, [0.2, 0.3, 0.4, 0.5], _coverage())]
        rows.extend(
            _row("head", 25, draw, 0.4 + draw / 100,
                 [0.2 + draw / 100, 0.3 + draw / 100, 0.4 + draw / 100, 0.5 + draw / 100],
                 _coverage(empty=(draw == 0), error=draw / 10, coverage=draw / 10),
                 fit_dir=f"fit-{draw}", selected_update=10, stopping_reason="patience",
                 assessment_metrics={"support": [1, 2, 3, 4]})
            for draw in range(5)
        )

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            report = make_report(rows, output)

            head = next(row for row in report["summary"] if row["method"] == "head")
            self.assertEqual(head["n"], 5)
            self.assertAlmostEqual(head["macro_f1_mean"], np.mean([0.4 + draw / 100 for draw in range(5)]))
            self.assertAlmostEqual(head["macro_f1_sd"], np.std([0.4 + draw / 100 for draw in range(5)], ddof=1))
            self.assertAlmostEqual(head["class_f1_mean"][0], 0.22)
            self.assertAlmostEqual(head["class_f1_sd"][0], np.std([0.2 + draw / 100 for draw in range(5)], ddof=1))

            target = head["targets"]["0.05"]
            self.assertAlmostEqual(target["coverage_mean"], 0.2)
            self.assertEqual(target["reject_all_count"], 1)
            self.assertEqual(target["empty_assessment_count"], 1)
            self.assertEqual(target["assessment_error_n_nonempty"], 4)
            self.assertEqual(target["assessment_error_n_undefined"], 1)
            self.assertAlmostEqual(target["assessment_error_mean_nonempty"], 0.25)
            self.assertEqual(target["overrun_count"], 4)
            source = next(row for row in report["summary"] if row["method"] == "source")
            self.assertIsNone(source["macro_f1_sd"])
            self.assertIsNone(source["class_f1_sd"][0])
            self.assertIn("930 tracks / 29,626 GT timesteps", report["caption"])
            self.assertIn("fixed training seed 0", report["caption"])
            self.assertEqual(report["runs"][1]["assessment_metrics"]["support"], [1, 2, 3, 4])

            with (output / "budget-summary.csv").open(newline="", encoding="utf-8") as stream:
                summary_rows = list(csv.DictReader(stream))
            self.assertIn("target_0.01_reject_all_count", summary_rows[0])
            raw = json.loads((output / "budget-report.json").read_text())
            self.assertEqual(raw["runs"][1]["fit_dir"], "fit-0")
            for name in ("budget-f1", "budget-class-f1", "budget-coverage"):
                for suffix in ("png", "svg"):
                    self.assertTrue((output / f"{name}.{suffix}").is_file())
            for name in ("budget-summary.csv", "budget-runs.csv", "budget-report.json"):
                self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
