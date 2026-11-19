"""E005 frozen DINO patch and Utonia level readouts."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint

from .e002f import BiLSTMTemporalBlock


ARMS = ("g-rgb", "g-rgb-utonia")
BRANCH_FLAGS = {"g-rgb": ("geometry_valid", "rgb_valid"),
                "g-rgb-utonia": ("geometry_valid", "rgb_valid", "lidar_valid")}


def _frame_chunks(counts, max_frames, max_padded_tokens):
    """Bound padded tokens while retaining each complete frame's token set."""
    chunk, maximum = [], 0
    for index, count in enumerate(counts):
        if not count:
            continue
        if count > max_padded_tokens:
            raise ValueError("Single spatial frame exceeds calibrated token ceiling")
        nmax = max(maximum, count)
        if chunk and (len(chunk) == max_frames or (len(chunk) + 1) * nmax > max_padded_tokens):
            yield chunk
            chunk, maximum = [], 0
        chunk.append(index)
        maximum = max(maximum, count)
    if chunk:
        yield chunk


class SpatialReadout(nn.Module):
    """One learned query cross-attends to one frame's projected spatial tokens."""

    def __init__(self, *, folded_attention=False):
        super().__init__()
        self.folded_attention = folded_attention
        self.seed = nn.Parameter(torch.randn(1, 128) * .02)
        self.seed_norm = nn.LayerNorm(128)
        self.q = nn.Linear(128, 128, bias=False)
        self.k = nn.Linear(128, 128, bias=False)
        self.v = nn.Linear(128, 128, bias=False)
        self.out = nn.Linear(128, 128, bias=False)
        self.norm_ffn = nn.LayerNorm(128)
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 128))
        self.dropout = nn.Dropout(.1)

    def forward(self, frames, seed=None, *, folded=None):
        seed = self.seed if seed is None else seed
        h = pad_sequence(frames, batch_first=True)
        valid = torch.arange(h.shape[1], device=h.device)[None, :] < torch.tensor(
            [len(frame) for frame in frames], device=h.device)[:, None]
        b, n, _ = h.shape
        query = self.q(self.seed_norm(seed))
        if folded is None:
            folded = self.folded_attention
        if folded:
            # The single query absorbs K, and linear V commutes with the weighted token sum.
            q = torch.einsum("hd,hdw->hw", query.reshape(4, 32), self.k.weight.reshape(4, 32, 128)) * 32 ** -.5
            scores = F.linear(h, q).transpose(1, 2)
            weights = scores.masked_fill(~valid[:, None, :], float("-inf")).softmax(dim=-1)
            pooled = F.dropout(weights, p=.1, training=self.training).bmm(h)
            y = torch.einsum("bhw,hdw->bhd", pooled, self.v.weight.reshape(4, 32, 128))
        else:
            q = query.reshape(1, 1, 4, 32).transpose(1, 2).expand(b, -1, -1, -1)
            k, v = (linear(h).reshape(b, n, 4, 32).transpose(1, 2) for linear in (self.k, self.v))
            with sdpa_kernel(SDPBackend.MATH):
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :],
                                                   dropout_p=.1 if self.training else 0.).transpose(1, 2)
        u = seed + self.dropout(self.out(y.reshape(b, 128)))
        return u + self.dropout(self.ffn(self.norm_ffn(u)))


class RGBPatchReadout(SpatialReadout):
    def __init__(self, *, folded_attention=False):
        super().__init__(folded_attention=folded_attention)
        self.feature_norm = nn.LayerNorm(768)
        self.project = nn.Linear(772, 128)
        self.token_norm = nn.LayerNorm(128)

    def project_frames(self, frames):
        joined = torch.cat(frames)
        x = torch.cat((self.feature_norm(joined[:, :768]), joined[:, 768:]), -1)
        projected = self.token_norm(self.project(x))
        return list(projected.split([len(frame) for frame in frames]))


class UtoniaReadout(SpatialReadout):
    def __init__(self, *, folded_attention=False):
        super().__init__(folded_attention=folded_attention)
        self.stage3_norm = nn.LayerNorm(72)
        self.stage5_norm = nn.LayerNorm(216)
        self.stage3 = nn.Linear(77, 128)
        self.stage5 = nn.Linear(221, 128)
        self.level = nn.Parameter(torch.randn(2, 128) * .02)
        self.token_norm = nn.LayerNorm(128)

    def project_frames(self, stage3, stage5):
        frames = []
        for x3, x5 in zip(stage3, stage5):
            h3 = self.stage3(torch.cat((self.stage3_norm(x3[:, :72]), x3[:, 72:]), -1)) + self.level[0]
            h5 = self.stage5(torch.cat((self.stage5_norm(x5[:, :216]), x5[:, 216:]), -1)) + self.level[1]
            frames.append(self.token_norm(torch.cat((h3, h5), 0)))
        return frames


class E005Classifier(nn.Module):
    def __init__(self, arm, profile):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown E005 arm: {arm}")
        if (not isinstance(profile, dict) or profile.get("frame_chunk", 0) < 1
                or profile.get("frame_tokens", 0) < 1 or profile.get("ordinary_tokens", 0) < 1):
            raise ValueError("Expected positive frame_chunk, frame_tokens and ordinary_tokens profile limits")
        self.arm, self.profile = arm, dict(profile)
        folded = profile.get("fold_single_query", False)
        if type(folded) is not bool:
            raise ValueError("Expected a boolean fold_single_query setting")
        self.geometry = nn.Sequential(nn.Linear(20, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.geometry_norm = nn.LayerNorm(64)
        self.rgb = RGBPatchReadout(folded_attention=folded)
        self.rgb_norm = nn.LayerNorm(128)
        self.rgb_branch_dropout = nn.Dropout1d(.5)
        self.lidar = UtoniaReadout(folded_attention=folded) if arm == "g-rgb-utonia" else None
        self.lidar_norm = nn.LayerNorm(128) if self.lidar is not None else None
        self.fusion = nn.Linear(64 + 128 + 128 * int(self.lidar is not None) + len(BRANCH_FLAGS[arm]), 128)
        self.temporal = BiLSTMTemporalBlock(.1)
        self.final_norm = nn.LayerNorm(128)
        self.head = nn.Linear(128, 4)

    @staticmethod
    def _check_tokens(frames, width, valid, *, patch=False, allow_empty=False):
        if len(frames) != len(valid):
            raise ValueError("Spatial token slots do not match temporal padding")
        for tokens, present in zip(frames, valid):
            if (not isinstance(tokens, torch.Tensor) or tokens.device.type != "cpu"
                    or tokens.ndim != 2 or tokens.shape[1] != width
                    or tokens.dtype not in (torch.float16, torch.float32)):
                raise ValueError("Invalid cached spatial tokens")
            if (not present and tokens.shape[0] > 0
                    or present and tokens.shape[0] == 0 and not allow_empty
                    or patch and present and tokens.shape[0] != 256):
                raise ValueError("Spatial validity flags and token sets disagree")

    def _padded_tokens(self, counts):
        return sum(len(indices) * max(counts[i] for i in indices) for indices in
                   _frame_chunks(counts, self.profile["frame_chunk"], self.profile["frame_tokens"]))

    def _readout(self, module, counts, inputs, project, mode):
        device = self.head.weight.device
        result = self.head.weight.new_zeros((len(counts), 128))
        for indices in _frame_chunks(counts, self.profile["frame_chunk"], self.profile["frame_tokens"]):
            cpu_inputs = inputs(indices)

            def encode(*values, frame_indices=tuple(indices)):
                return module(project(frame_indices, values[:-1]), values[-1])

            descriptor = (checkpoint(encode, *cpu_inputs, module.seed, use_reentrant=False)
                          if mode == "recompute" and self.training and torch.is_grad_enabled()
                          else encode(*cpu_inputs, module.seed))
            result = result.index_copy(0, torch.tensor(indices, device=device), descriptor)
        return result

    def forward(self, batch, *, mode=None):
        if mode not in (None, "ordinary", "recompute"):
            raise ValueError("Unknown E005 spatial execution mode")
        device = self.head.weight.device
        geometry, flags = (batch[key].to(device=device, dtype=torch.float32) for key in ("geometry", "flags"))
        lengths = batch["lengths"]
        if (geometry.ndim != 3 or geometry.shape[-1] != 20 or geometry.shape[0] == 0
                or lengths.device.type != "cpu" or lengths.dtype != torch.int64
                or lengths.shape != (geometry.shape[0],)):
            raise ValueError("Invalid geometry/lengths shapes")
        b, t, _ = geometry.shape
        if flags.shape != (b, t, len(BRANCH_FLAGS[self.arm])):
            raise ValueError("Invalid E005 validity flags")
        if (lengths <= 0).any() or (lengths > t).any():
            raise ValueError("Invalid complete-track lengths")
        valid = torch.arange(t, device=device)[None, :] < lengths.to(device)[:, None]
        if (not torch.isfinite(geometry[valid]).all() or not torch.isfinite(flags[valid]).all()
                or not ((flags[valid] == 0) | (flags[valid] == 1)).all()):
            raise ValueError("Nonfinite features or invalid branch flags")
        geometry = geometry.masked_fill(~valid[..., None], 0.)
        flags = flags.masked_fill(~valid[..., None], 0.)
        g = self.geometry_norm(self.geometry(geometry)) * flags[..., :1]

        rgb_frames = batch["rgb_frames"]
        rgb_valid = (flags[..., 1].bool() & valid).reshape(-1).tolist()
        self._check_tokens(rgb_frames, 772, rgb_valid, patch=True)
        rgb_counts = [len(x) for x in rgb_frames]
        if self.lidar is not None:
            stage3, stage5 = batch["lidar_levels"]
            lidar_valid = (flags[..., 2].bool() & valid).reshape(-1).tolist()
            self._check_tokens(stage3, 77, lidar_valid, allow_empty=True)
            self._check_tokens(stage5, 221, lidar_valid, allow_empty=True)
            lidar_counts = [len(x) + len(y) for x, y in zip(stage3, stage5)]
            if any(present and count == 0 for present, count in zip(lidar_valid, lidar_counts)):
                raise ValueError("LiDAR validity flags and token levels disagree")
        else:
            lidar_counts = []
        mode = mode or ("ordinary" if self._padded_tokens(rgb_counts) + self._padded_tokens(lidar_counts)
                        <= self.profile["ordinary_tokens"] else "recompute")
        rgb = self._readout(self.rgb, rgb_counts,
            lambda indices: [rgb_frames[i] for i in indices],
            lambda indices, values: self.rgb.project_frames(
                [value.to(device=device, dtype=torch.float32) for value in values]), mode)
        rgb = self.rgb_branch_dropout(self.rgb_norm(rgb.reshape(b, t, 128)).transpose(1, 2)).transpose(1, 2)
        rgb = rgb * flags[..., 1:2]
        descriptors = [g, rgb]

        if self.lidar is not None:
            lidar = self._readout(self.lidar, lidar_counts,
                lambda indices: [value for i in indices for value in (stage3[i], stage5[i])],
                lambda indices, values: self.lidar.project_frames(
                    [values[2 * j].to(device=device, dtype=torch.float32) for j in range(len(indices))],
                    [values[2 * j + 1].to(device=device, dtype=torch.float32) for j in range(len(indices))]), mode)
            lidar = self.lidar_norm(lidar.reshape(b, t, 128)) * flags[..., 2:3]
            descriptors.append(lidar)

        x = self.fusion(torch.cat((*descriptors, flags), -1))
        x = self.temporal(x, valid)
        return self.head(self.final_norm(x)).masked_fill(~valid[..., None], 0.)


def build_model(arm, profile):
    return E005Classifier(arm, profile)
