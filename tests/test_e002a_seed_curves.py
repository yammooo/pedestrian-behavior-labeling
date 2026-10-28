import json
from pathlib import Path
import tempfile
import unittest

from pedestrian_behavior.experiments.e002a_seed_curves import load_histories, plot_histories


class E002aSeedCurvesTest(unittest.TestCase):
    def test_loads_complete_repeats_and_exports_both_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            for source in ("road-waymo", "loki"):
                for seed in (1, 2, 3, 4):
                    run = root / f"e002a-{source}-g-rgb-s{seed}"
                    run.mkdir(parents=True)
                    rows = [
                        {"epoch": 1, "training_ce": 1., "validation_ce": 1.2,
                         "training_f1": .2, "validation_f1": .3, "selected": True},
                        {"epoch": 2, "training_ce": .8, "validation_ce": 1.1,
                         "training_f1": .4, "validation_f1": .5, "selected": True},
                    ]
                    (run / "status.json").write_text(json.dumps({"status": "complete"}))
                    (run / "history.json").write_text(json.dumps(rows))
                    (run / "provenance.json").write_text(json.dumps({"selection": {"epoch": 2}}))
            histories = load_histories(root)
            self.assertEqual(histories["road-waymo"][1][1], 2)
            output = Path(temporary) / "figures"
            plot_histories(root, output)
            for source in ("road-waymo", "loki"):
                for extension in ("png", "svg"):
                    self.assertTrue((output / f"{source}-g-rgb-seed-curves.{extension}").is_file())


if __name__ == "__main__":
    unittest.main()
