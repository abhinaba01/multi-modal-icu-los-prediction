"""The model family: single-modality, concatenation and attention fusion (Stage 3).

One class covers every neural arm; ``ModelSpec`` says which modalities it reads and how it
fuses them:

* ``fusion="none"``: one modality, token -> MLP (B1 static, B2 time series, B4 image);
* ``fusion="concat"``: tokens concatenated with presence flags -> MLP (B3, B6);
* ``fusion="attention"``: ``[CLS, static, ts, image]`` plus modality embeddings -> a
  2-layer Transformer encoder (d = 256, 4 heads, dropout 0.1); an absent modality is removed
  with ``key_padding_mask``, so it cannot influence the output (M, our model).

Two heads on the fused vector make the internship's two stages one model: a long-stay logit
and ``log1p(LOS days)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from fate_icu.models.encoders import (
    ImageTokenEncoder,
    LSTMEncoder,
    StaticEncoder,
    TransformerTSEncoder,
)

MODALITIES = ("static", "ts", "image")


@dataclass(frozen=True)
class ModelSpec:
    modalities: tuple[str, ...] = MODALITIES
    fusion: str = "attention"  # none | concat | attention
    ts_encoder: str = "lstm"  # lstm | transformer
    d: int = 256
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not self.modalities or any(m not in MODALITIES for m in self.modalities):
            raise ValueError(f"modalities must be a non-empty subset of {MODALITIES}")
        if self.fusion not in ("none", "concat", "attention"):
            raise ValueError(f"unknown fusion {self.fusion!r}")
        if (self.fusion == "none") != (len(self.modalities) == 1):
            raise ValueError("fusion 'none' is for exactly one modality, and only then")
        if self.ts_encoder not in ("lstm", "transformer"):
            raise ValueError(f"unknown time-series encoder {self.ts_encoder!r}")
        if self.d % self.heads:
            raise ValueError("d must be divisible by the number of heads")


class FusionNet(nn.Module):
    """See the module docstring. ``dims`` comes from ``StayData.dims``."""

    def __init__(self, spec: ModelSpec, dims: dict[str, int]) -> None:
        super().__init__()
        self.spec = spec
        d = spec.d
        self.encoders = nn.ModuleDict()
        for name in spec.modalities:
            if name == "static":
                self.encoders[name] = StaticEncoder(dims["static"], d, spec.dropout)
            elif name == "ts" and spec.ts_encoder == "lstm":
                self.encoders[name] = LSTMEncoder(dims["ts_channels"], d, 2, spec.dropout)
            elif name == "ts":
                self.encoders[name] = TransformerTSEncoder(
                    dims["ts_channels"], dims["hours"], d, 2, spec.heads, spec.dropout
                )
            else:
                self.encoders[name] = ImageTokenEncoder(dims["image"], d, spec.dropout)
        n = len(spec.modalities)
        if spec.fusion == "attention":
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            self.modality_embedding = nn.Embedding(n + 1, d)  # 0 = CLS
            layer = nn.TransformerEncoderLayer(
                d, spec.heads, 2 * d, spec.dropout, batch_first=True, norm_first=True, activation="gelu"
            )
            self.transformer = nn.TransformerEncoder(layer, spec.layers, enable_nested_tensor=False)
            self.trunk = nn.Identity()
        else:
            width = d * n + (n if spec.fusion == "concat" else 0)
            self.trunk = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Dropout(spec.dropout))
        self.norm = nn.LayerNorm(d)
        self.logit_head = nn.Linear(d, 1)
        self.los_head = nn.Linear(d, 1)

    def tokens(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Each modality's token ``[B, d]`` (computed for every row, absent or not)."""
        out = {}
        for name in self.spec.modalities:
            if name == "static":
                out[name] = self.encoders[name](batch["static"])
            elif name == "ts":
                out[name] = self.encoders[name](batch["values"], batch["mask"], batch["delta"])
            else:
                out[name] = self.encoders[name](batch["image"])
        return out

    def forward(
        self, batch: dict[str, torch.Tensor], present: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """``logit`` and ``log_los`` (both ``[B]``); ``present[name]`` is a bool ``[B]`` per modality."""
        tokens = self.tokens(batch)
        names = self.spec.modalities
        if self.spec.fusion == "attention":
            batch_size = next(iter(tokens.values())).shape[0]
            stack = [(self.cls + self.modality_embedding.weight[0]).expand(batch_size, -1, -1)]
            for k, name in enumerate(names, start=1):
                stack.append(tokens[name][:, None, :] + self.modality_embedding.weight[k])
            x = torch.cat(stack, dim=1)
            absent = torch.stack([~present[name] for name in names], dim=1)
            padding = torch.cat([torch.zeros_like(absent[:, :1]), absent], dim=1)
            fused = self.transformer(x, src_key_padding_mask=padding)[:, 0]
        else:
            parts = [tokens[name] * present[name][:, None].to(tokens[name].dtype) for name in names]
            if self.spec.fusion == "concat":
                parts += [present[name][:, None].to(parts[0].dtype) for name in names]
            fused = self.trunk(torch.cat(parts, dim=1))
        fused = self.norm(fused)
        return {"logit": self.logit_head(fused).squeeze(-1), "log_los": self.los_head(fused).squeeze(-1)}
