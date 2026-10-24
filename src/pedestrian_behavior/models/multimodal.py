"""Frozen-feature multimodal readout; no dataset, target or cache policy."""

from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint


ARM_FLAGS = {"g": ("geometry_valid",), "g-rgb": ("geometry_valid", "rgb_valid"),
             "g-dditr": ("geometry_valid", "lidar_valid"),
             "g-rgb-dditr": ("geometry_valid", "rgb_valid", "lidar_valid")}


def _all_finite_point_tokens(value):
    return np.isfinite(value.detach().resolve_neg().numpy()).all() if value.device.type == "cpu" else torch.isfinite(value).all()


def rotary(x):
    positions = torch.arange(x.shape[-2], device=x.device, dtype=x.dtype)
    frequency = 10000 ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=x.dtype) / x.shape[-1])
    angle = positions[:, None] * frequency
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1).flatten(-2)


class TemporalBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(256)
        self.qkv = nn.Linear(256, 768)
        self.out = nn.Linear(256, 256)
        self.norm_ffn = nn.LayerNorm(256)
        self.ffn = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(.1), nn.Linear(1024, 256))
        self.dropout = nn.Dropout(.1)

    def forward(self, x, valid):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, t, 3, 4, 64).permute(2, 0, 3, 1, 4)
        # FP32 math avoids fused-backend mask-alignment limits on irregular track lengths.
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(rotary(q), rotary(k), v, attn_mask=valid[:, None, None, :],
                                               dropout_p=.1 if self.training else 0.)
        x = x + self.dropout(self.out(y.transpose(1, 2).reshape(b, t, 256)))
        return x + self.dropout(self.ffn(self.norm_ffn(x)))


class SpatialReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_norm = nn.LayerNorm(64)
        self.project = nn.Linear(68, 128)
        self.point_norm = nn.LayerNorm(128)
        self.seed = nn.Parameter(torch.randn(1, 128) * .02)
        self.seed_norm = nn.LayerNorm(128)
        self.q = nn.Linear(128, 128, bias=False)
        self.k = nn.Linear(128, 128, bias=False)
        self.v = nn.Linear(128, 128, bias=False)
        self.out = nn.Linear(128, 128, bias=False)
        self.norm_ffn = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 128))
        self.dropout = nn.Dropout(.1)

    def forward(self, cpu_tokens, lengths, seed):
        x = cpu_tokens.to(seed.device)
        h = self.project(torch.cat((self.feature_norm(x[:, :64]), x[:, 64:]), -1))
        h = pad_sequence(h.split(lengths), batch_first=True)
        valid = torch.arange(h.shape[1], device=h.device)[None, :] < torch.tensor(lengths, device=h.device)[:, None]
        h = self.point_norm(h)
        b, n, _ = h.shape
        q = self.q(self.seed_norm(seed)).reshape(1, 1, 4, 32).transpose(1, 2).expand(b, -1, -1, -1)
        k, v = (linear(h).reshape(b, n, 4, 32).transpose(1, 2) for linear in (self.k, self.v))
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                               dropout_p=.1 if self.training else 0.)
        u = seed + self.dropout(self.out(y.transpose(1, 2).reshape(b, 128)))
        return u + self.dropout(self.ffn(self.norm_ffn(u)))


def frame_chunks(frames, max_frames, max_padded_tokens):
    """No tokens removed: bound concurrent padded spatial work, not temporal context."""
    chunk, maximum = [], 0
    for index, tokens in enumerate(frames):
        if not len(tokens):
            continue
        if len(tokens) > max_padded_tokens:
            raise ValueError("Single spatial frame exceeds calibrated token ceiling")
        nmax = max(maximum, len(tokens))
        if chunk and (len(chunk) == max_frames or (len(chunk) + 1) * nmax > max_padded_tokens):
            yield chunk
            chunk, maximum = [], 0
        chunk.append((index, tokens))
        maximum = max(maximum, len(tokens))
    if chunk:
        yield chunk


class MultimodalClassifier(nn.Module):
    """The agreed 64/128/128 -> 256, one-seed, full-track bidirectional model."""

    def __init__(self, arm, profile):
        super().__init__()
        if arm not in ARM_FLAGS:
            raise ValueError("Unknown multimodal arm")
        self.geometry = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.rgb = nn.Linear(768, 128) if "rgb_valid" in ARM_FLAGS[arm] else None
        self.spatial = SpatialReadout() if "lidar_valid" in ARM_FLAGS[arm] else None
        self.fusion = nn.Linear(64 + 128 * (len(ARM_FLAGS[arm]) - 1) + len(ARM_FLAGS[arm]), 256)
        self.temporal = nn.ModuleList([TemporalBlock(), TemporalBlock()])
        self.final_norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 4)
        self.arm, self.profile = arm, dict(profile)

    def forward(self, batch, *, mode=None):
        device = self.head.weight.device
        geometry, flags = (batch[k].to(device) for k in ("geometry", "flags"))
        lengths = batch["lengths"]
        b, t, _ = geometry.shape
        if geometry.shape != (b, t, 20) or flags.shape != (b, t, len(ARM_FLAGS[self.arm])):
            raise ValueError("Invalid multimodal shapes")
        valid = torch.arange(t, device=device)[None, :] < lengths.to(device)[:, None]
        if (lengths <= 0).any() or (lengths > t).any() or len(lengths) != b:
            raise ValueError("Invalid complete-track lengths")
        if any(not torch.isfinite(x[valid]).all() for x in (geometry, flags)) or not ((flags[valid] == 0) | (flags[valid] == 1)).all():
            raise ValueError("Nonfinite features or invalid branch flags")
        geometry, flags = (x.masked_fill(~valid[..., None], 0.) for x in (geometry, flags))
        g = self.geometry(geometry) * flags[..., :1]
        descriptors = [g]
        if self.rgb is not None:
            rgb = batch["rgb"].to(device)
            if rgb.shape != (b, t, 768) or not torch.isfinite(rgb[valid]).all():
                raise ValueError("Invalid/nonfinite RGB features")
            rgb = rgb.masked_fill(~valid[..., None], 0.)
            descriptors.append(self.rgb(rgb) * flags[..., 1:2])
        if self.spatial is not None:
            lidar = g.new_zeros(b * t, 128)
            frames = batch["points"]
            if len(frames) != b * t:
                raise ValueError("Point slots do not match temporal padding")
            if any(x.ndim != 2 or x.shape[1] != 68 or x.dtype != torch.float32 or not _all_finite_point_tokens(x) for x in frames):
                raise ValueError("Invalid/nonfinite spatial tokens")
            observed = torch.tensor([len(x) > 0 for x in frames], device=device).reshape(b, t)
            lidar_flag = ARM_FLAGS[self.arm].index("lidar_valid")
            if not torch.equal(observed, flags[..., lidar_flag].bool() & valid):
                raise ValueError("LiDAR validity and point tokens disagree")
            chunks = list(frame_chunks(frames, self.profile["frame_chunk"], self.profile["frame_tokens"]))
            tokens = sum(len(chunk) * max(len(x) for _, x in chunk) for chunk in chunks)
            mode = mode or ("ordinary" if tokens <= self.profile["ordinary_tokens"] else "recompute")
            if mode not in ("ordinary", "recompute", "offload"):
                raise ValueError("Unknown spatial execution mode")
            for chunk in chunks:
                indices, values = zip(*chunk)
                inputs = torch.cat(values)
                sizes = tuple(len(x) for x in values)
                if mode == "recompute" and self.training and torch.is_grad_enabled():
                    # CPU inputs bound retained VRAM; CUDA seed preserves CUDA dropout RNG.
                    descriptor = checkpoint(self.spatial, inputs, sizes, self.spatial.seed, use_reentrant=False)
                else:
                    scope = torch.autograd.graph.save_on_cpu(pin_memory=device.type == "cuda") if mode == "offload" else nullcontext()
                    with scope:
                        descriptor = self.spatial(inputs, sizes, self.spatial.seed)
                lidar = lidar.index_copy(0, torch.tensor(indices, device=device), descriptor)
            descriptors.append(lidar.reshape(b, t, 128))
        x = self.fusion(torch.cat((*descriptors, flags), -1))
        for layer in self.temporal:
            x = layer(x, valid)
        return self.head(self.final_norm(x)).masked_fill(~valid[..., None], 0.)
