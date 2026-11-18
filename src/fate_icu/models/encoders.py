"""Modality encoders: each maps one input to a single ``d``-dimensional token (Stage 3).

* ``StaticEncoder``: MLP over the encoded internship features and their missing indicators.
* ``LSTMEncoder``: 2-layer LSTM over ``[values, mask, delta]`` per hour (MedFuse's choice).
* ``TransformerTSEncoder``: the ablation, a small Transformer with learned time embeddings.
* ``ImageTokenEncoder``: MLP over the frozen image feature vector (1024-d features plus
  pathology logits).
"""

from __future__ import annotations

import torch
from torch import nn


class StaticEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d), nn.LayerNorm(d)
        )

    def forward(self, static: torch.Tensor) -> torch.Tensor:
        return self.net(static)


def _ts_input(values: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return torch.cat([values, mask, delta], dim=-1)


class LSTMEncoder(nn.Module):
    """Final hidden state of the top layer over the hourly grid ``[B, T, 3C]``."""

    def __init__(self, channels: int, d: int = 256, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            3 * channels, d, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(_ts_input(values, mask, delta))
        return self.norm(hidden[-1])


class TransformerTSEncoder(nn.Module):
    """Ablation: one token per hour plus a learned time embedding, mean-pooled."""

    def __init__(
        self, channels: int, hours: int, d: int = 256, layers: int = 2, heads: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        width = min(d, 64)
        self.inp = nn.Linear(3 * channels, width)
        self.time = nn.Embedding(hours, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.Linear(width, d), nn.LayerNorm(d))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        x = self.inp(_ts_input(values, mask, delta))
        x = x + self.time(torch.arange(x.shape[1], device=x.device))[None]
        return self.out(self.encoder(x).mean(dim=1))


class ImageTokenEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image)
