"""Temporary E002a resource graph; synthetic objective, not a behavior model."""

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


MODES = ("ordinary", "recompute", "offload")


def select_tracks(entries):
    """Minimum, median and maximum lengths in development groups; no GT ranking."""
    ordered = sorted(entries, key=lambda e: (e["slots"], e["locator"]))
    if len(ordered) < 3:
        raise ValueError("At least three development tracks required")
    return [(name, ordered[i]) for name, i in
            (("short", 0), ("typical", len(ordered)//2), ("long", len(ordered)-1))]


def crop_tokens(features, points, offsets, inside, radius=20):
    """Retain every aligned encoder voxel in the horizontal cylinder."""
    if (features.shape != (len(points), 64) or offsets.shape != (len(points), 3)
            or inside.shape != (len(points),) or not np.isfinite(features).all()
            or not np.isfinite(offsets).all()):
        raise ValueError("Invalid feature/site pairing")
    keep = (offsets[:, :2]**2).sum(1) <= radius**2
    tokens = np.c_[features[keep], offsets[keep]/radius, inside[keep]].astype(np.float32)
    return tokens, keep


def rotary(x):
    positions = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)
    frequency = 10000**(-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype)/x.shape[-1])
    angle = positions[:, None]*frequency
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a*angle.cos()-b*angle.sin(), a*angle.sin()+b*angle.cos()), -1).flatten(-2)


class TemporalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.qkv = nn.Linear(256, 768)
        self.out = nn.Linear(256, 256)
        self.norm_ffn = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(.1), nn.Linear(1024, 256))
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, t, 3, 4, 64).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rotary(q), rotary(k), v,
                                           dropout_p=.1 if self.training else 0)
        x = x+self.dropout(self.out(y.transpose(1, 2).reshape(b, t, 256)))
        return x+self.dropout(self.ffn(self.norm_ffn(x)))


class ExecutionGraph(nn.Module):
    """Agreed widths/RoPE, zero geometry and saved RGB; no behavior labels/loss."""
    def __init__(self):
        super().__init__()
        self.feature_norm = nn.LayerNorm(64)
        self.project = nn.Linear(68, 128)
        self.point_norm = nn.LayerNorm(128)
        self.seed = nn.Parameter(torch.randn(1, 128)*.02)
        self.seed_norm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(128, 4, dropout=0, batch_first=True)
        self.out_norm = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128))
        self.geometry = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.rgb = nn.Linear(768, 128)
        self.fusion = nn.Linear(323, 256)
        self.temporal = nn.Sequential(TemporalBlock(), TemporalBlock())
        self.final_norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 4)

    def spatial(self, cpu_input, seed):
        x = cpu_input.to(seed.device)
        h = self.project(torch.cat((self.feature_norm(x[:, :64]), x[:, 64:]), -1))
        h = self.point_norm(h).unsqueeze(0)
        q = self.seed_norm(seed).unsqueeze(0)
        u = seed+self.attention(q, h, h, need_weights=False)[0][0]
        return (u+self.ffn(self.out_norm(u)))[0]

    def forward(self, frames, rgb, valid, *, mode="ordinary", tracks=1):
        if mode not in MODES or tracks < 1 or not frames:
            raise ValueError("Invalid resource workload")
        device = self.seed.device
        fused = []
        for _ in range(tracks):
            descriptors = []
            for frame in frames:
                if frame.ndim != 2 or frame.shape[1] != 68 or not torch.isfinite(frame).all():
                    raise ValueError("Invalid point tokens")
                if not len(frame):
                    descriptors.append(self.seed.new_zeros(128))
                elif mode == "recompute":
                    # CUDA seed argument makes checkpoint preserve the CUDA RNG state.
                    descriptors.append(checkpoint(self.spatial, frame, self.seed, use_reentrant=False))
                else:
                    scope = torch.autograd.graph.save_on_cpu(pin_memory=device.type == "cuda") if mode == "offload" else nullcontext()
                    with scope:
                        descriptors.append(self.spatial(frame, self.seed))
            t = len(frames)
            g = self.geometry(torch.zeros(t, 20, device=device))*0
            r = self.rgb(rgb.to(device))*valid.to(device)[:, None]
            lidar = torch.stack(descriptors)
            flags = torch.stack((valid.new_zeros(t), valid, valid.new_tensor([len(f)>0 for f in frames])), -1).to(device)
            fused.append(self.fusion(torch.cat((g, r, lidar, flags), -1)))
        return self.head(self.final_norm(self.temporal(torch.stack(fused))))


def compare_gradients(frames, rgb, valid, device):
    """Check outputs, all parameter gradients and one AdamW update across paths."""
    reference = None
    errors = {}
    for mode in MODES:
        torch.manual_seed(0)
        graph = ExecutionGraph().to(device)
        optimizer = torch.optim.AdamW(graph.parameters(), lr=.001, weight_decay=.0001)
        value = graph(frames, rgb, valid, mode=mode)
        value.square().mean().backward()
        gradients = {k: p.grad.detach().cpu().clone() if p.grad is not None else None for k, p in graph.named_parameters()}
        optimizer.step()
        state = {k: v.detach().cpu().clone() for k, v in graph.state_dict().items()}
        item = {"output": value.detach().cpu(), "gradients": gradients, "state": state}
        if reference is None:
            reference = item
        maximum = 0.
        for section in ("gradients", "state"):
            for k, actual in item[section].items():
                expected = reference[section][k]
                if actual is None or expected is None:
                    if actual is not expected:
                        raise ValueError("Gradient presence differs")
                    continue
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                maximum = max(maximum, float((actual-expected).abs().max()))
        torch.testing.assert_close(item["output"], reference["output"], rtol=1e-4, atol=1e-5)
        errors[mode] = maximum
    return errors
