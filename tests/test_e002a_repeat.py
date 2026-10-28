import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from pedestrian_behavior.experiments import e002a_repeat, e002a_run


PROFILE = {"max_tracks": 8, "temporal_cells": 100000, "microbatch_tokens": 10000,
           "ordinary_tokens": 1000, "frame_chunk": 2, "frame_tokens": 1000}


class E002aRepeatTest(unittest.TestCase):
    def test_fixed_plan_and_seeded_child_command(self):
        plan = e002a_repeat.attempts()
        self.assertEqual(len(plan), 8)
        self.assertEqual([row["source"] for row in plan], ["road-waymo"] * 4 + ["loki"] * 4)
        self.assertEqual([row["seed"] for row in plan], [1, 2, 3, 4] * 2)
        self.assertEqual([row["name"] for row in plan],
                         [f"e002a-{source}-g-rgb-s{seed}" for source in e002a_repeat.SOURCES for seed in e002a_repeat.SEEDS])
        command = e002a_repeat.attempt_command(plan[-1], Path("outputs/run"), "cuda", ["--profile", "profile.json"])
        self.assertEqual(command[command.index("--seed") + 1], "4")
        self.assertEqual(command[command.index("--arm") + 1], "g-rgb")
        self.assertEqual(command[command.index("--output") + 1], "outputs/run/e002a-loki-g-rgb-s4")

    def test_single_run_cli_accepts_seed_and_keeps_zero_default(self):
        parser = e002a_repeat.build_parser()
        args = parser.parse_args(["--profile", "profile.json", "--rgb-preparation", "rgb.json"])
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.profile, Path("profile.json"))
        self.assertEqual(args.output.parent, Path("outputs/experiments/E002a/runs"))

        single = argparse.ArgumentParser()
        e002a_run.add_arguments(single)
        required = ["--source", "loki", "--arm", "g-rgb", "--device", "cpu",
                    "--output", "run", "--profile", "profile.json"]
        self.assertEqual(single.parse_args(required).seed, 0)
        self.assertEqual(single.parse_args(required + ["--seed", "3"]).seed, 3)
        with self.assertRaisesRegex(ValueError, "Seed must be"):
            e002a_run.run(SimpleNamespace(seed=-1))

    def test_preflight_fails_before_creating_output_when_profile_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            output = root / "experiments/E002a/runs/repeat"
            args = SimpleNamespace(outputs_root=root, output=output, device="cpu",
                                   profile=Path(temporary) / "missing-profile.json")
            with self.assertRaisesRegex(FileNotFoundError, "Calibration profile missing"):
                e002a_repeat.validate_inputs(args)
            self.assertFalse(output.exists())

    def test_preflight_rejects_missing_rgb_cache_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "outputs"
            output = root / "experiments/E002a/runs/repeat"
            profile_path = Path(temporary) / "profile.json"
            profile_path.write_text(json.dumps({"status": "complete", "device": "cpu", "code": {"fixture": 1},
                "torch": str(torch.__version__), "cuda": torch.version.cuda, "profile": PROFILE}))
            cache_root = Path(temporary) / "rgb-cache"
            prep_path = Path(temporary) / "rgb-preparation.json"
            tracks = []
            for source in e002a_repeat.SOURCES:
                setup_path = root / "experiments/E001/data-setup" / source / "setup.json"
                setup_path.parent.mkdir(parents=True)
                population = [{"locator": source + "-train", "split": "training", "exclusions": []},
                              {"locator": source + "-validation", "split": "validation", "exclusions": []},
                              {"locator": source + "-test", "split": "test", "exclusions": []}]
                setup_path.write_text(json.dumps({"status": "complete", "tracks": population}))
                tracks.extend({"dataset": source, "locator": row["locator"], "rgb_valid": 1}
                              for row in population[:2])
            prep_path.write_text(json.dumps({"status": "complete", "extraction": {"fixture": 1},
                "config": {"rgb_cache_root": str(cache_root)}, "tracks": tracks}))
            args = SimpleNamespace(outputs_root=root, output=output, device="cpu", profile=profile_path,
                rgb_preparation=prep_path, rgb_cache_root=cache_root, probe=Path(temporary) / "probe")
            with patch.object(e002a_repeat, "resource_code", return_value={"fixture": 1}), \
                 patch.object(e002a_repeat, "load_configuration", return_value=({}, Path("transform.json"))), \
                 patch.object(e002a_repeat, "extraction_identity", return_value={"fixture": 1}), \
                 self.assertRaisesRegex(FileNotFoundError, "Prepared RGB cache index/entries missing"):
                e002a_repeat.validate_inputs(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
