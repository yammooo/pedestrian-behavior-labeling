"""Framewise or bidirectional whole-track classification with a common MLP/readout."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class KinematicClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_classes, dropout, *, recurrent_hidden=None):
        super().__init__()
        if (min(input_dim, embedding_dim, num_classes) < 1
                or recurrent_hidden is not None and recurrent_hidden < 1):
            raise ValueError("Model dimensions must be positive")
        self.input_dim = input_dim
        self.encoder = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(embedding_dim, embedding_dim), nn.ReLU())
        self.recurrent = (nn.LSTM(embedding_dim, recurrent_hidden, num_layers=1, batch_first=True,
                                 bidirectional=True, dropout=0.) if recurrent_hidden is not None else None)
        output_dim = 2*recurrent_hidden if self.recurrent is not None else embedding_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(output_dim, num_classes))

    def forward(self, inputs, lengths):
        """Return [B,T,C] logits; lengths are CPU int64, internal gaps are real slots.

        Padding logits are zero and unscored. Recurrent state starts at zero on every call.
        """
        if inputs.ndim != 3 or inputs.shape[0] == 0 or inputs.shape[2] != self.input_dim:
            raise ValueError("Expected nonempty inputs[B,T,input_dim]")
        if (lengths.device.type != "cpu" or lengths.dtype != torch.int64
                or lengths.shape != (inputs.shape[0],)
                or (lengths < 1).any() or (lengths > inputs.shape[1]).any()):
            raise ValueError("Expected CPU int64 lengths[B] between 1 and T")
        padding = torch.arange(inputs.shape[1], device=inputs.device)[None, :] >= lengths.to(inputs.device)[:, None]
        # Mask before the encoder so even arbitrary/NaN padding cannot contaminate gradients.
        embeddings = self.encoder(inputs.masked_fill(padding[..., None], 0.))
        if self.recurrent is not None:
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
            packed, _ = self.recurrent(packed)
            embeddings, _ = pad_packed_sequence(packed, batch_first=True, total_length=inputs.shape[1])
        return self.head(embeddings).masked_fill(padding[..., None], 0.)
