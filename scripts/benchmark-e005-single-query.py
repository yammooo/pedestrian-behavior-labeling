"""Benchmark candidate for E005's exact single-query spatial readout."""

import torch

from pedestrian_behavior.models.e005 import SpatialReadout

original_forward = SpatialReadout.forward


def folded_forward(self, frames, seed=None):
    return original_forward(self, frames, seed, folded=True)


def check():
    torch.set_num_threads(2)
    torch.manual_seed(5)
    layer = SpatialReadout().eval()
    frames = [torch.randn(n, 128, requires_grad=True) for n in (1, 17, 257, 8191)]
    args = frames + list(layer.parameters())
    reference = layer(frames)
    folded = folded_forward(layer, frames)
    grads_ref = torch.autograd.grad(reference.square().mean(), args)
    grads_folded = torch.autograd.grad(folded.square().mean(), args)
    torch.testing.assert_close(folded, reference, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(grads_folded, grads_ref):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    print("Max eval output difference:", (folded - reference).abs().max().item())
    print("Max input/parameter gradient difference:",
          max((a - b).abs().max().item() for a, b in zip(grads_folded, grads_ref)))

    layer.train()
    torch.manual_seed(7)
    reference = layer(frames)
    torch.manual_seed(7)
    folded = folded_forward(layer, frames)
    torch.testing.assert_close(folded, reference, atol=2e-6, rtol=2e-5)
    print("Max seeded training output difference:", (folded - reference).abs().max().item())


if __name__ == "__main__":
    check()
