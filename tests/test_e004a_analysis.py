import tempfile
import unittest
from pathlib import Path

import numpy as np

from pedestrian_behavior.experiments.e004a_analysis import analyze_checkpoint, make_report


def _track(locator, probabilities, predictions, targets, group=None):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.int64)
    logits = np.log(np.tile((1 - probabilities)[:, None] / 3, (1, 4)))
    logits[np.arange(len(predictions)), predictions] = np.log(probabilities)
    return {"locator": locator, "group": group or locator, "logits": logits,
            "targets": np.asarray(targets, dtype=np.int64), "gt_valid": np.ones(len(targets), dtype=bool)}


def _coverage(empty=False, error=0.0):
    return {"targets": [{"error_target": target, "threshold": None if empty else 0.8,
        "reject_all": empty, "calibration": {"coverage": 0.0 if empty else 0.5, "error": None if empty else 0.0},
        "assessment": {"coverage": 0.0 if empty else 0.4, "error": None if empty else error,
            "accepted_frames": 0 if empty else 4, "error_frames": 0 if empty else 1,
            "accepted_mass": 0.0 if empty else 0.4, "error_mass": 0.0 if empty else error * 0.4,
            "classes": []}} for target in (0.01, 0.05, 0.10, 0.20)]}


class E004aAnalysisTest(unittest.TestCase):
    def analyze(self, calibration, assessment, target):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        result = analyze_checkpoint(calibration, assessment, Path(temp.name), (target,))
        return result["targets"][0], Path(temp.name)

    def test_track_weights_stay_fixed_after_partial_acceptance(self):
        calibration = [_track("cal", [0.9], [0], [0]) | {"source_flags": {
            "rgb_valid": np.ones(1, dtype=bool), "point_tokens": np.asarray([80], dtype=np.int64)}}]
        assessment = [
            _track("long", [0.95, 0.8], [0, 1], [0, 0]) | {"source_flags": {
                "rgb_valid": np.ones(2, dtype=bool), "point_tokens": np.asarray([80, 86], dtype=np.int64)}},
            _track("short", [0.7], [0], [0]) | {"source_flags": {
                "rgb_valid": np.ones(1, dtype=bool), "point_tokens": np.asarray([0], dtype=np.int64)}},
        ]
        selected, output = self.analyze(calibration, assessment, 0.0)
        measured = selected["assessment"]
        self.assertAlmostEqual(selected["threshold"], 0.9)
        self.assertEqual(measured["accepted_frames"], 1)
        self.assertEqual(measured["total_mass"], 2.0)
        self.assertEqual(measured["accepted_mass"], 0.5)
        self.assertEqual(measured["coverage"], 0.25)
        self.assertAlmostEqual(measured["raw_coverage"], 1 / 3)
        unavailable = measured["natural_availability_cohorts"]["rgb_valid"]["unavailable"]
        self.assertIsNone(unavailable["coverage"])
        self.assertIsNone(unavailable["raw_coverage"])
        self.assertEqual(set(measured["natural_availability_cohorts"]), {"rgb_valid"})
        with np.load(output / "calibration-curve.npz", allow_pickle=False) as curve:
            self.assertEqual(curve["threshold"].shape, (1,))

    def test_ties_and_nonmonotone_prefix_risk_are_all_considered(self):
        calibration = [_track("cal", [0.9, 0.8, 0.8, 0.7], [1, 0, 1, 0], [0, 0, 0, 0])]
        selected, output = self.analyze(calibration, calibration, 0.5)
        self.assertAlmostEqual(selected["threshold"], 0.7)
        with np.load(output / "calibration-curve.npz", allow_pickle=False) as curve:
            np.testing.assert_allclose(curve["threshold"], [0.9, 0.8, 0.7])
            np.testing.assert_array_equal(curve["accepted_frames"], [1, 3, 4])
            np.testing.assert_allclose(curve["error"], [1.0, 2 / 3, 0.5])

    def test_reject_all_stays_reject_all_on_assessment(self):
        calibration = [_track("cal", [0.99], [1], [0])]
        assessment = [_track("assessment", [0.999], [0], [0])]
        selected, _ = self.analyze(calibration, assessment, 0.01)
        self.assertTrue(selected["reject_all"])
        self.assertIsNone(selected["threshold"])
        self.assertEqual(selected["assessment"]["accepted_frames"], 0)
        self.assertIsNone(selected["assessment"]["error"])

    def test_assessment_risk_is_measured_without_changing_cutoff(self):
        calibration = [_track("cal", [0.8], [0], [0])]
        assessment = [_track("assessment", [0.99], [1], [0])]
        selected, _ = self.analyze(calibration, assessment, 0.0)
        self.assertAlmostEqual(selected["threshold"], 0.8)
        self.assertEqual(selected["assessment"]["error"], 1.0)

    def test_class_retention_keeps_original_class_denominator(self):
        calibration = [_track("cal", [0.9] * 8 + [0.7] * 2, [0] * 8 + [1] * 2, [0] * 10)]
        selected, _ = self.analyze(calibration, calibration, 0.0)
        class_zero = selected["assessment"]["classes"][0]
        self.assertEqual(class_zero["original_gt_frames"], 10)
        self.assertAlmostEqual(class_zero["gt_class_retention"], 0.8)
        self.assertAlmostEqual(class_zero["accepted_precision"], 1.0)
        self.assertAlmostEqual(class_zero["automated_correct_fraction"], 0.8)
        self.assertIsNone(selected["assessment"]["classes"][1]["gt_class_retention"])

    def test_report_anchors_empty_risk_and_sample_sd(self):
        rows = [{"budget": 0, "draw": None, "macro_f1": 0.2, "track_ce": 1.0,
                 "coverage": _coverage(empty=True), "annotation_access": {"calibration_tracks": 2}}]
        rows.extend({"budget": 5, "draw": draw, "macro_f1": 0.4 + draw / 100,
                     "track_ce": 0.8, "coverage": _coverage(empty=draw == 0, error=0.1)} for draw in range(5))
        rows.append({"budget": 8574, "draw": None, "macro_f1": 0.7, "track_ce": 0.5,
                     "coverage": _coverage(empty=False, error=0.0)})
        with tempfile.TemporaryDirectory() as temp:
            report = make_report(rows, Path(temp))
            self.assertIn("annotation_access record", report["caption"])
            by_budget = {row["budget"]: row for row in report["summary"]}
            self.assertIsNone(by_budget[0]["macro_f1_sd"])
            self.assertIsNone(by_budget[8574]["macro_f1_sd"])
            self.assertAlmostEqual(by_budget[5]["macro_f1_sd"], np.std([0.4 + i / 100 for i in range(5)], ddof=1))
            target = by_budget[5]["targets"]["0.05"]
            self.assertAlmostEqual(target["coverage_mean"], 0.32)
            self.assertEqual(target["empty_assessment_count"], 1)
            self.assertEqual(target["assessment_error_n_nonempty"], 4)
            self.assertEqual(target["overrun_count"], 4)
            for name in ("budget-f1.png", "budget-f1.svg", "budget-coverage.png", "budget-coverage.svg",
                         "budget-summary.csv", "budget-runs.csv", "budget-report.json"):
                self.assertTrue((Path(temp) / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
