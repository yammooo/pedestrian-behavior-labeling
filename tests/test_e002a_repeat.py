import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pedestrian_behavior.experiments import e002a_repeat, e002a_run


class E002aRepeatTest(unittest.TestCase):
    def test_eight_seeded_commands(self):
        plan = e002a_repeat.attempts()
        self.assertEqual([(row["source"], row["seed"]) for row in plan],
                         [(source, seed) for source in ("road-waymo", "loki") for seed in (1, 2, 3, 4)])
        self.assertEqual(len({row["name"] for row in plan}), 8)
        command = e002a_repeat.attempt_command(plan[-1], Path("outputs/run"), "cuda", [])
        self.assertEqual(command[command.index("--seed") + 1], "4")
        self.assertEqual(command[command.index("--arm") + 1], "g-rgb")
        self.assertEqual(command[command.index("--output") + 1], "outputs/run/e002a-loki-g-rgb-s4")

    def test_single_run_seed_defaults_to_zero(self):
        parser = argparse.ArgumentParser()
        e002a_run.add_arguments(parser)
        required = ["--source", "loki", "--arm", "g-rgb", "--device", "cpu",
                    "--output", "run", "--profile", "profile.json"]
        self.assertEqual(parser.parse_args(required).seed, 0)
        self.assertEqual(parser.parse_args(required + ["--seed", "3"]).seed, 3)
        with self.assertRaisesRegex(ValueError, "Seed must be"):
            e002a_run.run(SimpleNamespace(seed=-1))

    def test_unmounted_native_root_fails_before_sweep(self):
        with tempfile.TemporaryDirectory() as temporary:
            probe = Path(temporary)
            (probe / "probe.json").write_text(json.dumps({"config": {
                "waymo_root": str(probe / "unmounted"), "loki_root": str(probe / "loki")}}))
            with self.assertRaisesRegex(FileNotFoundError, "Native source root unavailable"):
                e002a_repeat.check_native_mount(probe)
            (probe / "unmounted").mkdir()
            (probe / "loki").mkdir()
            e002a_repeat.check_native_mount(probe)


if __name__ == "__main__":
    unittest.main()
