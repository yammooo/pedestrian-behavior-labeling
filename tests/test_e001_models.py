"""Hand-calculated nonlinear/recurrent outputs and padding-independent predictions."""

import json
import math
import os
from pathlib import Path
import unittest

import torch

from pedestrian_behavior.data.loading import collate_tracks
from pedestrian_behavior.experiments.e001 import build_model


class E001ModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.evidence = []

    def tearDown(self):
        if output := os.environ.get("E001_ACCEPTANCE_OUTPUT"):
            Path(output).mkdir(parents=True, exist_ok=True)
            (Path(output)/(self._testMethodName+".json")).write_text(json.dumps(self.evidence, indent=2)+"\n")

    def check(self, name, expected, actual):
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().tolist()
        self.evidence.append({"check": name, "expected": expected, "actual": actual})
        self.assertEqual(expected, actual, name)

    def close(self, name, expected, actual):
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=actual.dtype, device=actual.device),
                                   rtol=1e-6, atol=1e-6)
        self.evidence.append({"check": name, "expected": expected, "actual": actual.detach().cpu().tolist()})

    def sample(self, values, labels, name):
        y = torch.tensor(labels, dtype=torch.int64)
        return {"inputs": torch.tensor(values, dtype=torch.float32), "targets": y, "gt_valid": y != -100,
                "locator": name, "archive": name+".npz"}

    def test_dimensions_parameter_counts_and_independent_weights(self):
        expected = {"K": (3, 4676, 29764), "K+T": (6, 4868, 29956),
                    "K+T+R": (12, 5252, 30340), "RAW": (12, 5252, 30340)}
        for feature, (dimension, frame_count, temporal_count) in expected.items():
            for variant, count in (("A", frame_count), ("B", temporal_count)):
                model = build_model(feature, variant).eval()
                output = model(torch.zeros(2, 3, dimension), torch.tensor([1, 3]))
                self.check(feature+variant+" parameter count", count, sum(p.numel() for p in model.parameters()))
                self.check(feature+variant+" shape/dtype/finite", [[2, 3, 4], "torch.float32", True],
                           [list(output.shape), str(output.dtype), bool(torch.isfinite(output).all())])
                self.check(feature+variant+" zero padding logits", [[0.]*4]*2, output[0, 1:])
                if variant == "B":
                    self.check(feature+variant+" recurrent dimensions", [64, 32, 1, True, 0.],
                               [model.recurrent.input_size, model.recurrent.hidden_size, model.recurrent.num_layers,
                                model.recurrent.bidirectional, model.recurrent.dropout])
        a, b = build_model("K", "A"), build_model("K", "B")
        self.check("A and B do not share encoder/head parameters", True,
                   not ({p.data_ptr() for p in a.parameters()} & {p.data_ptr() for p in b.parameters()}))

    def test_hand_calculated_nonlinear_frame_outputs(self):
        model = build_model("K", "A").eval()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.encoder[0].weight[0, :2] = torch.tensor([1., -1.])
            model.encoder[3].weight[0, 0] = 2.
            model.encoder[3].bias[0] = -1.
            model.head[1].weight[:3, 0] = torch.tensor([1., -1., .5])
            model.head[1].bias[3] = .25
        inputs = torch.tensor([[[2., 0., 1.], [0., 0., 0.], [0., 2., 1.]]])
        # z = ReLU(2*ReLU(x0-x1)-1); scores = [z,-z,z/2,1/4].
        self.close("two nonlinear layers, linear readout; zero internal slot is real",
                   [[[3., -3., 1.5, .25], [0., 0., 0., .25], [0., 0., 0., .25]]],
                   model(inputs, torch.tensor([3])))

    def test_hand_calculated_bidirectional_states_and_internal_gap(self):
        model = build_model("K", "B").eval()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.encoder[0].weight[0, 0] = model.encoder[3].weight[0, 0] = 1.
            model.recurrent.weight_ih_l0[64, 0] = model.recurrent.weight_ih_l0_reverse[64, 0] = 1.
            model.head[1].weight[0, 0] = model.head[1].weight[1, 32] = 1.
        batch = collate_tracks([self.sample([[1., 0., 1.]], [0], "short"),
                                self.sample([[1., 0., 1.], [0., 0., 0.], [2., 0., 1.]], [0, -100, 3], "long")])
        # One active cell per direction: i=f=o=1/2, g=tanh(x0), c'=c/2+g/2, h=tanh(c')/2.
        a, b = math.tanh(1)/2, math.tanh(2)/2
        h = lambda c: math.tanh(c)/2
        expected = [[[h(a), h(a), 0., 0.], [0.]*4, [0.]*4],
                    [[h(a), h(a+b/4), 0., 0.], [h(a/2), h(b/2), 0., 0.], [h(a/4+b), h(b), 0., 0.]]]
        self.close("per-slot forward/backward output, unsorted singleton and decaying internal gap",
                   expected, model(batch["inputs"], batch["lengths"]))
        self.check("GT gap is not removed from recurrence", [3, False],
                   [int(batch["lengths"][1]), bool(batch["padding_mask"][1, 1])])

    def test_padding_batch_composition_reset_and_backward(self):
        batch = collate_tracks([self.sample([[1., 2., 1.]], [1], "short"),
                               self.sample([[2., 1., 1.], [0., 0., 0.], [4., 3., 1.]], [0, -100, 3], "long")])
        for variant in ("A", "B"):
            model = build_model("K", variant).eval()
            original = model(batch["inputs"], batch["lengths"])
            padded = torch.full((2, 7, 3), float("nan"))
            padded[:, :3] = batch["inputs"]
            padded[0, 1:] = float("nan")
            longer = model(padded, batch["lengths"])
            for i, length in enumerate(batch["lengths"].tolist()):
                single = model(batch["inputs"][i:i+1, :length], torch.tensor([length]))
                error = float((single[0]-original[i, :length]).abs().max().detach())
                self.assertLessEqual(error, 1e-6)
                torch.testing.assert_close(longer[i, :length], original[i, :length], rtol=1e-6, atol=1e-6)
                self.evidence.append({"check": variant+f" track {i}: alone vs batch vs seven-slot NaN padding",
                                      "expected": "max error <= 1e-6", "actual": error})
            model(torch.ones_like(batch["inputs"]), batch["lengths"])
            self.check(variant+" no state carry between calls", True,
                       torch.equal(original, model(batch["inputs"], batch["lengths"])))
            model.train()
            x = padded.clone().requires_grad_()
            logits = model(x, batch["lengths"])
            # Only a differentiability smoke check; equal-track loss is the next increment.
            loss = torch.nn.functional.cross_entropy(logits[:, :3][batch["gt_valid"]],
                                                     batch["targets"][batch["gt_valid"]])
            loss.backward()
            self.check(variant+" finite gradients through encoder/recurrent/head", True,
                       all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            padding = torch.arange(7)[None, :] >= batch["lengths"][:, None]
            self.check(variant+" padding has zero input gradient", True, bool((x.grad[padding] == 0).all()))
            before = model.head[1].weight.detach().clone()
            torch.optim.AdamW(model.parameters(), lr=.001).step()
            self.check(variant+" optimizer can update readout", False, torch.equal(before, model.head[1].weight))

    def test_invalid_lengths_shapes_and_configuration(self):
        for variant in ("A", "B"):
            model = build_model("K", variant)
            for lengths in (torch.tensor([0]), torch.tensor([4]), torch.tensor([2.]), torch.tensor([1, 2])):
                with self.assertRaisesRegex(ValueError, "lengths"):
                    model(torch.zeros(1, 3, 3), lengths)
            for inputs in (torch.zeros(1, 3), torch.zeros(1, 3, 6), torch.zeros(0, 3, 3)):
                with self.assertRaisesRegex(ValueError, "inputs"):
                    model(inputs, torch.tensor([1]))
        for arguments in (("unknown", "A"), ("K", "unknown")):
            with self.assertRaisesRegex(ValueError, "Unknown"):
                build_model(*arguments)
        self.check("invalid length/shape/configuration fails explicitly", True, True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA acceptance runs on aalto")
    def test_cuda_predictions_padding_and_gradients(self):
        lengths = torch.tensor([1, 3])
        for feature, dimension in (("K", 3), ("K+T", 6), ("K+T+R", 12), ("RAW", 12)):
            for variant in ("A", "B"):
                model = build_model(feature, variant).cuda().eval()
                inputs = torch.randn(2, 3, dimension, device="cuda")
                original = model(inputs, lengths)
                padded = torch.full((2, 7, dimension), float("nan"), device="cuda")
                padded[:, :3] = inputs
                longer = model(padded, lengths)
                for i, length in enumerate(lengths.tolist()):
                    single = model(inputs[i:i+1, :length], torch.tensor([length]))
                    torch.testing.assert_close(original[i, :length], single[0], rtol=1e-6, atol=1e-6)
                    torch.testing.assert_close(original[i, :length], longer[i, :length], rtol=1e-6, atol=1e-6)
                model.train()
                loss = model(padded, lengths).square().sum()
                loss.backward()
                self.check(feature+variant+" CUDA finite forward/backward and padding-independent predictions", True,
                           bool(torch.isfinite(original).all()) and
                           all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
