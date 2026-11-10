"""E002f temporal backbones on the shared E002d multimodal branches."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .multimodal import MultimodalClassifier, TemporalBlock


ARCHITECTURES = ("transformer-1layer", "bilstm-96")


class BiLSTMTemporalBlock(nn.Module):
    """One packed bidirectional LSTM and projection; padded suffixes are never recurrent inputs."""

    def __init__(self, dropout=.1):
        super().__init__()
        self.lstm = nn.LSTM(128, 96, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(192, 128)

    def forward(self, x, valid):
        lengths = valid.sum(-1)
        if (lengths <= 0).any() or not torch.all(valid == (torch.arange(
                x.shape[1], device=x.device)[None, :] < lengths[:, None])):
            raise ValueError("BiLSTM requires nonempty contiguous complete-track masks")
        packed = pack_padded_sequence(x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        encoded = self.projection(self.dropout(encoded))
        return encoded.masked_fill(~valid[..., None], 0.)


class E002fClassifier(MultimodalClassifier):
    """E002d's trainable branches/fusion with one of two 128-D temporal backbones."""

    def __init__(self, architecture, profile):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown E002f architecture: {architecture}")
        super().__init__("g-rgb-dditr", profile, geometry_width=64, rgb_width=128, spatial_width=128,
            temporal_width=128, temporal_ffn=512, normalize_branches=True,
            temporal_dropout=.1, rgb_branch_dropout=.5)
        self.architecture = architecture
        if architecture == "transformer-1layer":
            self.temporal = nn.ModuleList([TemporalBlock(128, 512, .1)])
        else:
            self.temporal = nn.ModuleList([BiLSTMTemporalBlock(.1)])


def model_spec(architecture):
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown E002f architecture: {architecture}")
    temporal = ({"type": "transformer", "width": 128, "heads": 4, "layers": 1,
                 "ffn": 512, "attention_bias": True, "rope_base": 10000,
                 "dropout": .1, "attention_backend": "torch-SDPA-math"}
                if architecture == "transformer-1layer" else
                {"type": "bidirectional-lstm", "input": 128, "hidden_per_direction": 96,
                 "layers": 1, "bias": True, "packed_complete_tracks": True,
                 "output_dropout": .1, "projection": [192, 128]})
    return {"architecture": architecture, "availability": ["geometry_valid", "rgb_valid", "lidar_valid"],
        "geometry_mlp": [20, 64, 64], "branch_layer_norm": True,
        "rgb_projection": [768, 128], "rgb_descriptor_dropout": .5,
        "spatial_readout": {"input": 68, "width": 128, "heads": 4, "seeds": 1,
            "layers": 1, "ffn": 512, "dropout": .1},
        "fusion": [323, 128], "temporal": temporal,
        "final_norm": {"type": "LayerNorm", "width": 128, "eps": 1e-5},
        "classifier": [128, 4], "precision": "float32",
        "trainable_temporal_norm_head_parameters": 199044}


def build_model(architecture, profile):
    model_spec(architecture)
    model = E002fClassifier(architecture, profile)
    count = sum(parameter.numel() for name, parameter in model.named_parameters()
        if name.startswith(("temporal.", "final_norm.", "head.")))
    if count != 199044:
        raise ValueError(f"E002f temporal/norm/head parameter count changed: {count}")
    return model
