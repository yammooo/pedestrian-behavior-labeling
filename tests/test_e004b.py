import unittest

import numpy as np

from pedestrian_behavior.experiments import e004a_run, e004b_run


def _record(locator, group, labels, gt_valid=None):
    targets = np.asarray(labels, dtype=np.int64)
    if gt_valid is None:
        gt_valid = targets >= 0
    return {"locator": locator, "group": group, "targets": targets,
        "gt_valid": np.asarray(gt_valid, dtype=bool)}


def _role_pool():
    rows = []
    for class_id in range(4):
        labels = np.full(8, class_id, dtype=np.int64)
        gt_valid = np.asarray([True, False, True, False, True, False, True, True])
        rows.append(_record(f"anchor-{class_id}", f"scenario-anchor-{class_id}", labels, gt_valid))
    for index in range(8):
        rows.append(_record(f"other-{index}", f"scenario-other-{index}", [index % 4]))
    return rows


def _full_pool():
    rows = []
    for index in range(e004b_run.ALL_TRACKS):
        if index < 4:
            class_id = index
            labels = np.full(7, class_id, dtype=np.int64)
            gt_valid = np.asarray([True, False, True, False, True, True, True])
        else:
            labels, gt_valid = [index % 4], [True]
        rows.append(_record(f"track-{index:05d}", f"scenario-{index:05d}", labels, gt_valid))
    return rows


class E004bSamplerTest(unittest.TestCase):
    def test_four_roles_use_ge5_accepted_gt_in_distinct_scenarios(self):
        order, metadata = e004b_run.class_covered_order(_role_pool(), 2)
        self.assertEqual(len(metadata["anchors"]), 4)
        self.assertEqual(len({row["group"] for row in metadata["anchors"]}), 4)
        self.assertCountEqual([row["class_id"] for row in metadata["anchors"]], range(4))
        for anchor in metadata["anchors"]:
            self.assertEqual(anchor["role_gt_timesteps"], 5)
            self.assertEqual(anchor["class_gt_counts"][anchor["class_id"]], 5)
        self.assertEqual(len({row["locator"] for row in order}), len(_role_pool()))

    def test_seed_and_canonical_input_order_fix_anchors_and_exact_remainder(self):
        records = _role_pool()
        shuffled = list(np.random.default_rng(71).permutation(records))
        first, details = e004b_run.class_covered_order(records, 4)
        second, shuffled_details = e004b_run.class_covered_order(shuffled, 4)
        self.assertEqual([row["locator"] for row in first], [row["locator"] for row in second])
        self.assertEqual(details, shuffled_details)
        anchor_ids = {row["locator"] for row in details["anchors"]}
        old_order = e004a_run.recording_round_robin(records, 4)
        expected_remainder = [row["locator"] for row in old_order if row["locator"] not in anchor_ids]
        self.assertEqual([row["locator"] for row in first[:4]],
                         [row["locator"] for row in details["anchors"]])
        self.assertEqual([row["locator"] for row in first[4:]], expected_remainder)
        self.assertEqual([row["locator"] for row in first[:5]],
                         [row["locator"] for row in first[:10]][:5])

    def test_unlabeled_values_do_not_make_an_anchor_eligible(self):
        rows = _role_pool()
        labels = np.full(8, 0, dtype=np.int64)
        gt_valid = np.zeros(8, dtype=bool)
        gt_valid[:4] = True
        rows[0] = _record("weak-moving", "scenario-weak", labels, gt_valid)
        with self.assertRaisesRegex(ValueError, "no >=5-GT MOVING track"):
            e004b_run.class_covered_order(rows, 0)

    def test_infeasible_role_scenarios_fail_without_fallback(self):
        rows = [_record(f"class-{class_id}", "same-scenario", [class_id] * 5)
            for class_id in range(4)]
        with self.assertRaisesRegex(ValueError, "infeasible"):
            e004b_run.class_covered_order(rows, 0)

    def test_plan_has_40_nested_draw_fits_and_one_canonical_full_fit(self):
        training = _full_pool()
        development = [_record("validation-dev", "dev-scenario", [0])]
        assessment = [_record("validation-assess", "audit-scenario", [1])]
        plan = e004b_run.build_plan(training, development, assessment)
        self.assertEqual(len(plan["runs"]), 41)
        self.assertEqual(sum(row["budget"] != e004b_run.ALL_TRACKS for row in plan["runs"]), 40)
        self.assertNotIn(1, plan["protocol"]["budget_grid"])
        self.assertEqual(plan["protocol"]["budget_grid"], [5, 10, 25, 50, 100, 250, 500, 1000, 8574])
        self.assertEqual(plan["screening_access"]["accounting"]["candidate_tracks"], 8574)
        self.assertEqual(len(plan["screening_access"]["candidate_track_class_support"]), 8574)
        training_ids = {row["locator"] for row in training}
        validation_ids = {row["locator"] for row in development + assessment}
        for row in plan["runs"]:
            selected_ids = [item["locator"] for item in row["selected_tracks"]]
            self.assertTrue(set(selected_ids).issubset(training_ids))
            self.assertFalse(set(selected_ids) & validation_ids)
            if row["budget"] == e004b_run.ALL_TRACKS:
                self.assertEqual(row["draw"], None)
                self.assertEqual(selected_ids, sorted(training_ids))
        for draw in range(5):
            rows = sorted((row for row in plan["runs"] if row["draw"] == draw), key=lambda row: row["budget"])
            previous = []
            for row in rows:
                selected = [item["locator"] for item in row["selected_tracks"]]
                self.assertEqual(selected[:len(previous)], previous)
                previous = selected

    def test_smoke_plan_is_two_budgets_with_full_check_schedule(self):
        plan = e004b_run.build_plan(_full_pool(), [_record("dev", "dev", [0])],
            [_record("audit", "audit", [1])], smoke=True)
        self.assertEqual([(row["budget"], row["draw"]) for row in plan["runs"]], [(5, 0), (10, 0)])
        self.assertEqual(plan["protocol"]["check_updates"], [0, 1, 5, 10, 25, 50, 100, 200])
        self.assertTrue(plan["protocol"]["early_checks"])


if __name__ == "__main__":
    unittest.main()
