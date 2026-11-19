"""Benchmark candidate for E005's exact single-query spatial readout."""

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from pedestrian_behavior.models.e005 import SpatialReadout


def folded_forward(self, frames, seed=None):
    seed = self.seed if seed is None else seed
    tokens = pad_sequence(frames, batch_first=True)
    valid = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < torch.tensor(
        [len(frame) for frame in frames], device=tokens.device)[:, None]
    query = self.q(self.seed_norm(seed)).reshape(4, 32)
    # A single query can absorb K before scoring. Linearity lets V follow the weighted sum.
    query = torch.einsum("hd,hdw->hw", query, self.k.weight.reshape(4, 32, 128)) * 32 ** -.5
    scores = F.linear(tokens, query).transpose(1, 2)
    weights = scores.masked_fill(~valid[:, None, :], float("-inf")).softmax(dim=-1)
    pooled = F.dropout(weights, p=.1, training=self.training).bmm(tokens)
    values = torch.einsum("bhw,hdw->bhd", pooled, self.v.weight.reshape(4, 32, 128))
    u = seed + self.dropout(self.out(values.reshape(-1, 128)))
    return u + self.dropout(self.ffn(self.norm_ffn(u)))


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
