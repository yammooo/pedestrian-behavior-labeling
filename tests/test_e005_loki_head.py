import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pedestrian_behavior.experiments import e004a_run, e005_loki_head


def _track(index, group=None):
    target = index % 4
    return {"locator": f"track-{index:04d}", "archive": f"track-{index:04d}.npz",
        "group": group or f"scenario-{index % 50:02d}",
        "targets": np.asarray([target, target], dtype=np.int64),
        "gt_valid": np.asarray([True, True]),
        "source_flags": {"rgb_valid": np.asarray([True, False])}}


class E005LokiHeadTest(unittest.TestCase):
    def test_reference_uses_exact_nested_e004a_subsets_and_checks_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            starts = (0, 500, 1000, 1500, 2099)
            training = [_track(index) for index in range(3099)]
            entries = [{"locator": row["locator"], "group": row["group"]} for row in training]
            runs = []
            for draw, start in enumerate(starts):
                ordered = training[start:start + 1000]
                for budget in e005_loki_head.BUDGETS:
                    runs.append({"budget": budget, "draw": draw,
                        "selected_tracks": [e004a_run._track_label_record(row) for row in ordered[:budget]]})
            development, assessment = [_track(4000, "dev")], [_track(4001, "assessment")]
            groups = {"development_groups": ["dev"], "assessment_groups": ["assessment"]}
            protocol = {"early_checks": True, "check_updates": [0, *e004a_run.EARLY_CHECK_UPDATES],
                "patience_updates": e004a_run.EARLY_PATIENCE_UPDATES, "fit_seed": 0, "training_seed": 0,
                "max_optimizer_updates": e004a_run.SETTINGS["max_updates"],
                "optimizer": e004a_run.SETTINGS["optimizer"]}
            plan = {"experiment": "E004a", "smoke": False, "protocol": protocol,
                "validation_split": groups, "runs": runs,
                "population": {"development": e004a_run._population_summary(development),
                    "assessment": e004a_run._population_summary(assessment)}}
            (root / "subset-plans.json").write_text(json.dumps(plan))

            saved, selected, union = e005_loki_head._reference_plan(root, entries, development, assessment)
            self.assertEqual(len(selected), 30)
            self.assertEqual(len(union), 3099)
            e005_loki_head._verify_selected_labels(selected, training)
            changed = [dict(row) for row in training]
            changed[0] = changed[0] | {"targets": np.asarray([3, 3])}
            with self.assertRaisesRegex(ValueError, "GT labels changed"):
                e005_loki_head._verify_selected_labels(selected, changed)

    def test_direct_replay_requires_same_labels_flags_and_logits(self):
        logits = np.asarray([[1., 2., 3., 4.]], dtype=np.float32)
        targets = np.asarray([3], dtype=np.int64)
        gt = np.asarray([True])
        flags = {"rgb_valid": np.asarray([True])}
        common = {"locator": "track", "archive": "track.npz", "group": "scenario",
            "targets": targets, "gt_valid": gt, "source_flags": flags}
        actual = [common | {"source_logits": logits, "features": np.zeros((1, 128), dtype=np.float32)}]
        expected = [common | {"logits": logits}]
        matched, delta = e005_loki_head.e005_transfer_check(actual, expected)
        self.assertEqual(matched[0]["locator"], "track")
        self.assertEqual(delta, 0.)
        actual[0]["source_logits"] = logits + 1
        with self.assertRaisesRegex(ValueError, "replay differs"):
            e005_loki_head.e005_transfer_check(actual, expected)


if __name__ == "__main__":
    unittest.main()
