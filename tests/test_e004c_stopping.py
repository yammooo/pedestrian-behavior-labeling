import tempfile
import unittest
from itertools import count
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from pedestrian_behavior.training import fit


def _fake_epoch(model, _batches, optimizer, _device, _clip_norm,
                on_step=None, step_offset=0, updates=None):
    return {"training_ce": 1., "training_f1": .5, "optimizer_steps": updates,
            "processed_tracks": updates, "context_slots": updates, "gt_slots": updates,
            "padding_slots": 0, "batch_slots": updates, "seconds": 0.}


def _validation(f1=.5, ce=1.):
    return {"track_ce": ce, "track_weighted": {"macro_f1": f1}}


class _CountingUpdates:
    def __init__(self):
        self.iterations = 0
        self.produced = 0

    def __iter__(self):
        self.iterations += 1
        return self

    def __next__(self):
        self.produced += 1
        return self.produced


class E004cStoppingTest(unittest.TestCase):
    def test_unbounded_fit_consumes_lazy_schedule_past_10000_checks(self):
        schedule = _CountingUpdates()
        model, optimizer = nn.Linear(1, 1), object()
        callbacks = []
        with tempfile.TemporaryDirectory() as directory, \
                patch("pedestrian_behavior.training.save_checkpoint"):
            result = fit(model, [], lambda _model: _validation(), optimizer, "cpu", directory,
                {}, lambda row, _history: callbacks.append(row), epochs=None, patience=1,
                clip_norm=1., epoch_operation=_fake_epoch, initial_candidate=True,
                check_updates=schedule, patience_updates=10001)

        self.assertEqual(schedule.iterations, 1)
        self.assertEqual(schedule.produced, 10001)
        self.assertEqual(callbacks[0]["optimizer_step"], 0)
        self.assertTrue(callbacks[0]["selected"])
        self.assertEqual(result["history"][-1]["optimizer_step"], 10001)
        self.assertEqual(result["stopping_reason"], "patience")

    def test_step_zero_selection_and_strict_f1_patience_ignore_ce_ties(self):
        validations = iter((_validation(.5, 1.), _validation(.5, .8), _validation(.6, .9),
                            _validation(.6, .7), _validation(.6, .8), _validation(.6, .8)))
        callbacks = []
        with tempfile.TemporaryDirectory() as directory, \
                patch("pedestrian_behavior.training.save_checkpoint"):
            result = fit(nn.Linear(1, 1), [], lambda _model: next(validations), object(), "cpu", directory,
                {}, lambda row, _history: callbacks.append(row), epochs=None, patience=1,
                clip_norm=1., epoch_operation=_fake_epoch, initial_candidate=True,
                check_updates=iter((1, 100, 900, 1599, 1600)), patience_updates=1500)

        self.assertEqual(callbacks[0]["optimizer_step"], 0)
        self.assertTrue(callbacks[0]["selected"])
        by_step = {row.get("optimizer_step"): row for row in callbacks}
        self.assertTrue(by_step[1]["selected"])  # CE tie changes the selected checkpoint.
        self.assertTrue(by_step[900]["selected"])
        self.assertIsNone(by_step[1599]["stopping_reason"])  # F1 improvement at 100 reset patience.
        self.assertEqual(by_step[1600]["stopping_reason"], "patience")  # CE tie at 900 did not reset it.
        self.assertEqual((result["selection"]["epoch"], result["selection"]["optimizer_step"],
                          result["selection"]["ce"]), (3, 900, .7))

    def test_unbounded_fit_rejects_nonincreasing_and_exhausted_schedules(self):
        for schedule, message in ((iter((5, 4)), "strictly increasing"),
                                  (iter((1,)), "ended before patience")):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory, \
                    patch("pedestrian_behavior.training.save_checkpoint"):
                with self.assertRaisesRegex(ValueError, message):
                    fit(nn.Linear(1, 1), [], lambda _model: _validation(), object(), "cpu", directory,
                        {}, lambda *_: None, epochs=None, patience=1, clip_norm=1.,
                        epoch_operation=_fake_epoch, initial_candidate=True,
                        check_updates=schedule, patience_updates=100)


if __name__ == "__main__":
    unittest.main()
