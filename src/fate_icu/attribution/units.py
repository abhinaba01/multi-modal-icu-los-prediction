"""Evidence units and the scorer that removes any set of them (Stage 4).

An evidence unit is anything we can remove and re-score (CLAUDE.md A8):

* ``timeseries``: one variable over one 4-hour block (the last block is shorter when 24 is not
  a multiple of the block length);
* ``static``: one feature group of the contract's ``fate_static`` set;
* ``image``: one pathology concept, or ``whole_image`` (the X-ray is left out).

``UnitSpace`` fixes the order of the units, so a score vector from any method and a removal
mask from any test line up. ``StayScorer`` re-scores one stay with chosen units removed, through
the gates of ``gates.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fate_icu.attribution.gates import GatedLogit, GateLayout, stay_tensors
from fate_icu.models.data import StayData

WHOLE_IMAGE = "whole_image"
BLOCK_HOURS = 4


@dataclass(frozen=True)
class Unit:
    modality: str  # timeseries | static | image
    key: str  # channel, static group, image concept or whole_image
    start_hour: int | None = None  # timeseries only; the block is [start_hour, end_hour)
    end_hour: int | None = None

    @property
    def block(self) -> str | None:
        return None if self.start_hour is None else f"hours {self.start_hour}-{self.end_hour}"


class UnitSpace:
    """The ordered list of units: time-series blocks, then static groups, then image units."""

    def __init__(self, layout: GateLayout, block_hours: int = BLOCK_HOURS, whole_image: bool = True) -> None:
        if block_hours < 1:
            raise ValueError("block_hours must be at least 1")
        self.layout = layout
        self.block_hours = block_hours
        edges = list(range(0, layout.hours, block_hours)) + [layout.hours]
        self.blocks = list(zip(edges[:-1], edges[1:], strict=True))
        units: list[Unit] = []
        for channel in layout.channels:
            for start, end in self.blocks:
                units.append(Unit("timeseries", channel, start, end))
        units += [Unit("static", group) for group in layout.groups]
        units += [Unit("image", concept) for concept in layout.concepts]
        if whole_image:
            units.append(Unit("image", WHOLE_IMAGE))
        self.units = units
        self.index = {unit: i for i, unit in enumerate(units)}
        self.n_ts = len(layout.channels) * len(self.blocks)
        self.n_static = len(layout.groups)
        self.n_concepts = layout.n_concepts
        self.has_whole_image = whole_image

    def __len__(self) -> int:
        return len(self.units)

    @property
    def ts_slice(self) -> slice:
        return slice(0, self.n_ts)

    @property
    def static_slice(self) -> slice:
        return slice(self.n_ts, self.n_ts + self.n_static)

    @property
    def concept_slice(self) -> slice:
        return slice(self.n_ts + self.n_static, self.n_ts + self.n_static + self.n_concepts)

    @property
    def whole_image_index(self) -> int | None:
        return len(self.units) - 1 if self.has_whole_image else None

    def ts_unit(self, channel: str, block: int) -> int:
        return self.layout.channels.index(channel) * len(self.blocks) + block

    def hourly_to_blocks(self, hourly: np.ndarray) -> np.ndarray:
        """Sum hourly ``[..., T, C]`` values into block scores ``[..., n_blocks, C]``."""
        return np.stack([hourly[..., s:e, :].sum(axis=-2) for s, e in self.blocks], axis=-2)

    def blocks_to_units(self, blocks: np.ndarray) -> np.ndarray:
        """Block scores ``[n_blocks, C]`` -> the time-series part of a unit score vector."""
        return np.asarray(blocks).T.reshape(-1)

    def gates(self, removed: np.ndarray) -> dict[str, torch.Tensor]:
        """Gates for removal masks ``removed`` (bool ``[B, n_units]``; True = the unit is removed)."""
        removed = np.asarray(removed, dtype=bool)
        if removed.ndim != 2 or removed.shape[1] != len(self):
            raise ValueError(f"removed must have shape [B, {len(self)}]")
        n = removed.shape[0]
        ts = np.ones((n, self.layout.hours, len(self.layout.channels)), dtype=np.float32)
        ts_removed = removed[:, self.ts_slice].reshape(n, len(self.layout.channels), len(self.blocks))
        for b, (start, end) in enumerate(self.blocks):
            ts[:, start:end, :] = 1.0 - ts_removed[:, :, b][:, None, :]
        out = {
            "ts": torch.from_numpy(ts),
            "static": torch.from_numpy(1.0 - removed[:, self.static_slice].astype(np.float32)),
            "concept": torch.from_numpy(1.0 - removed[:, self.concept_slice].astype(np.float32)),
        }
        out["image_removed"] = (
            torch.from_numpy(removed[:, self.whole_image_index])
            if self.has_whole_image
            else torch.zeros(n, dtype=torch.bool)
        )
        return out


class StayScorer:
    """Re-scores one stay with any set of its evidence units removed."""

    def __init__(self, model: GatedLogit, space: UnitSpace, data: StayData, row: int) -> None:
        self.model, self.space, self.row = model, space, row
        self.fixed = stay_tensors(data, np.array([row]))
        self.has_image = bool(data.has_image[row])

    def expand(self, n: int) -> dict[str, torch.Tensor]:
        return {k: v.expand(n, *v.shape[1:]) for k, v in self.fixed.items()}

    @torch.no_grad()
    def logits(self, removed: np.ndarray, chunk: int = 256) -> np.ndarray:
        """Logit for each removal mask in ``removed`` (``[B, n_units]``)."""
        removed = np.atleast_2d(np.asarray(removed, dtype=bool))
        out = []
        for start in range(0, len(removed), chunk):
            part = removed[start : start + chunk]
            gates = self.space.gates(part)
            fixed = self.expand(len(part))
            present = fixed["image_present"] & ~gates["image_removed"]
            out.append(
                self.model(
                    gates["ts"], gates["static"], gates["concept"], present,
                    fixed["values"], fixed["mask"], fixed["delta"], fixed["static"], fixed["image"],
                ).numpy()
            )  # fmt: skip
        return np.concatenate(out)

    def base_logit(self) -> float:
        return float(self.logits(np.zeros((1, len(self.space)), dtype=bool))[0])

    def removal_delta(self, units: np.ndarray | None = None) -> np.ndarray:
        """Δ for each unit: ``logit(x) - logit(x without the unit)`` (positive: pushed longer).

        Units that do not exist for this stay (a concept or the whole image without an X-ray)
        get Δ = 0 because removing them changes nothing.
        """
        n = len(self.space)
        units = np.arange(n) if units is None else np.asarray(units)
        removed = np.zeros((len(units), n), dtype=bool)
        removed[np.arange(len(units)), units] = True
        return self.base_logit() - self.logits(removed)
