import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pedestrian_behavior.experiments import e002a_run, e002e_run


class E002eWeightDecayTests(unittest.TestCase):
    def test_settings_change_only_weight_decay_without_mutating_reference(self):
        original = e002a_run.SETTINGS["optimizer"].copy()
        for value in (.01, .1):
            settings = e002e_run.attempt_settings(value)
            self.assertEqual(settings["optimizer"], original | {"weight_decay": value})
            self.assertEqual(settings["epochs"], e002a_run.SETTINGS["epochs"])
            self.assertEqual(settings["patience"], e002a_run.SETTINGS["patience"])
        self.assertEqual(e002a_run.SETTINGS["optimizer"], original)
        self.assertIsNot(e002e_run.attempt_settings(.01)["optimizer"], e002a_run.SETTINGS["optimizer"])
        with self.assertRaisesRegex(ValueError, "weight decay"):
            e002e_run.attempt_settings(.001)

    def test_run_passes_effective_settings_and_e002e_assembly(self):
        args = SimpleNamespace(source="road-waymo", arm="g-rgb", model_variant="temporal-small",
            dropout=.1, rgb_branch_dropout=0., lidar_branch_dropout=0., smoke=False,
            weight_decay=.1, output=Path("outputs/experiments/E002e/runs/run"), device="cuda")
        captured = {}

        def fake_attempt(*call_args, **kwargs):
            captured.update(args=call_args, kwargs=kwargs)
            kwargs["assemble"]()
            return "ok"

        prepared = {"config": {}}
        with patch.object(e002e_run, "run_attempt", side_effect=fake_attempt), \
             patch.object(e002e_run.e002b_run, "_profile_for"), \
             patch.object(e002e_run.e002a_run, "assembly", side_effect=lambda args, policy: (
                 captured.update(policy=policy) or prepared)):
            self.assertEqual(e002e_run.run(args), "ok")

        settings = captured["kwargs"]["settings"]
        self.assertEqual(settings["optimizer"]["lr"], .0003)
        self.assertEqual(settings["optimizer"]["weight_decay"], .1)
        self.assertEqual(settings["optimizer"]["betas"], [.9, .999])
        policy = captured["policy"]
        self.assertEqual(policy["experiment"], "E002e")
        self.assertEqual(policy["run_name"], "e002e-road-waymo-g-rgb-temporal-small-wd0p1-s0")
        self.assertEqual(policy["model_spec"]("g-rgb", "temporal-small")["temporal"]["width"], 128)
        self.assertEqual(prepared["config"]["weight_decay_sensitivity"]["control"], .0001)

    def test_root_guard_requires_fresh_e002e_location_separate_from_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(outputs_root=root, output=root / "experiments/E002e/runs/run",
                rgb_cache_root=root / "experiments/E002a/rgb-cache",
                point_cache_root=root / "experiments/E002a/indexed-road-waymo")
            e002e_run.validate_roots(args, {})
            args.output = root / "experiments/E002b/runs/run"
            with self.assertRaisesRegex(ValueError, "E002e evidence"):
                e002e_run.validate_roots(args, {})


if __name__ == "__main__":
    unittest.main()
