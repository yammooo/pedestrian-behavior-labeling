import unittest

import torch

from pedestrian_behavior.experiments.e001 import build_model as build_e001_model
from pedestrian_behavior.experiments.e001_run import SETTINGS as E001_SETTINGS
from pedestrian_behavior.experiments.e001d_run import MODEL, SETTINGS, build_model
from pedestrian_behavior.experiments.e001d_sweep import SEEDS, SOURCES, attempts


class E001dTest(unittest.TestCase):
    def test_matched_capacity_model_and_fixed_matrix(self):
        model = build_model().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 30342)
        self.assertEqual((MODEL["input_dim"], MODEL["recurrent_hidden"], MODEL["framewise_block"]), (12, None, [64, 194, 64]))
        self.assertEqual((SETTINGS["epochs"], SETTINGS["patience"], SETTINGS["optimizer"]["lr"],
                          SETTINGS["optimizer"]["weight_decay"]), (75, 8, .001, .0001))
        self.assertEqual((E001_SETTINGS["epochs"], E001_SETTINGS["patience"], E001_SETTINGS["batch_size"]), (30, 5, 64))
        self.assertEqual(sum(p.numel() for p in build_e001_model("K+T+R", "B").parameters()), 30340)

        rows = attempts()
        self.assertEqual([(row["source"], row["seed"]) for row in rows],
                         [(source, seed) for seed in SEEDS for source in SOURCES])
        self.assertEqual(len({row["name"] for row in rows}), 10)

        inputs = torch.randn(2, 4, 12)
        lengths = torch.tensor([4, 3])
        original = model(inputs, lengths)
        self.assertEqual(tuple(original.shape), (2, 4, 4))
        changed = inputs.clone()
        changed[:, 1] += 1
        altered = model(changed, lengths)
        torch.testing.assert_close(original[:, (0, 2, 3)], altered[:, (0, 2, 3)])
        self.assertTrue(torch.equal(original[1, 3], torch.zeros(4)))

        model.train()
        model(inputs, lengths).square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
