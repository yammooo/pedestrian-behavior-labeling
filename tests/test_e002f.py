import unittest

import torch

from pedestrian_behavior.experiments.e002f_run import TemporalAdaptationHead, _adaptation_model
from pedestrian_behavior.models.e002f import ARCHITECTURES, BiLSTMTemporalBlock, build_model, model_spec


PROFILE = {"max_tracks": 64, "temporal_cells": 665856, "microbatch_tokens": 1460024,
    "ordinary_tokens": 1512760, "frame_chunk": 2, "frame_tokens": 524288}


def _batch(lengths, padded_value=0.):
    batch, slots = len(lengths), max(lengths)
    geometry = torch.randn(batch, slots, 20)
    rgb = torch.randn(batch, slots, 768)
    flags = torch.zeros(batch, slots, 3)
    for i, length in enumerate(lengths):
        flags[i, :length, :2] = 1.
        geometry[i, length:] = padded_value
        rgb[i, length:] = padded_value
    return {"geometry": geometry, "rgb": rgb, "flags": flags,
        "points": [torch.zeros(0, 68) for _ in range(batch * slots)],
        "lengths": torch.tensor(lengths, dtype=torch.long)}


class E002fModelTests(unittest.TestCase):
    def test_architectures_have_declared_temporal_parameter_count(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = build_model(architecture, PROFILE)
                count = sum(p.numel() for name, p in model.named_parameters()
                    if name.startswith(("temporal.", "final_norm.", "head.")))
                self.assertEqual(count, 199044)
                self.assertEqual(sum(p.numel() for p in model.parameters()), 552196)
                self.assertEqual(model_spec(architecture)["temporal"]["type"],
                    "transformer" if architecture == "transformer-1layer" else "bidirectional-lstm")

    def test_bilstm_packs_complete_tracks_and_zeros_padding(self):
        torch.manual_seed(3)
        module = BiLSTMTemporalBlock(.1).eval()
        valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        source = torch.randn(2, 4, 128, requires_grad=True)
        changed_suffix = source.detach().clone()
        changed_suffix[1, 2:] = 99.
        actual = module(source, valid)
        altered = module(changed_suffix, valid)
        self.assertTrue(torch.allclose(actual[valid], altered[valid], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(actual[~valid], torch.zeros_like(actual[~valid])))
        actual.sum().backward()
        self.assertTrue(torch.equal(source.grad[~valid], torch.zeros_like(source.grad[~valid])))

    def test_adaptation_arms_start_from_same_source_logits_and_disable_dropout(self):
        torch.manual_seed(7)
        for architecture in ARCHITECTURES:
            source = build_model(architecture, PROFILE).eval()
            batch = _batch([5, 3])
            captured = []
            hook = source.fusion.register_forward_hook(lambda _m, _i, output: captured.append(output.detach()))
            with torch.no_grad():
                source_logits = source(batch)
            hook.remove()
            features = captured[0]
            temporal = features
            valid = torch.arange(features.shape[1])[None, :] < batch["lengths"][:, None]
            for block in source.temporal:
                temporal = block(temporal, valid)
            expected = source.head(source.final_norm(temporal)).masked_fill(~valid[..., None], 0.)
            self.assertTrue(torch.allclose(source_logits, expected, atol=1e-6, rtol=1e-6))
            postnorm = source.final_norm(temporal).masked_fill(~valid[..., None], 0.)
            for arm in ("head", "temporal-head"):
                model = _adaptation_model(arm, source, torch.device("cpu"))
                actual = model(postnorm if arm == "head" else features, batch["lengths"])
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))
                if arm == "head":
                    self.assertTrue(torch.equal(model.linear.weight, source.head.weight))
                    self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()
                        if parameter is not model.linear.weight and parameter is not model.linear.bias))
                else:
                    self.assertTrue(all(module.p == 0. for module in model.modules()
                        if isinstance(module, torch.nn.Dropout)))
                    self.assertEqual(sum(p.numel() for p in model.parameters()), 199044)


if __name__ == "__main__":
    unittest.main()
