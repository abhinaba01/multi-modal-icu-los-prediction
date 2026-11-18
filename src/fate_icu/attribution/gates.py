"""The long-stay logit as a function of presence gates (Stage 4).

Every attribution method works in the same space: a *gate* is 1 where the evidence is
present and 0 where it is removed. A gate of 0 means exactly what training taught the model
"removed" means (``fate_icu.models.removal``), so occlusion, Integrated Gradients, the mask
methods and the deletion scores of driver selection all measure the same thing:

* time series: a gate per (hour, variable); a removed cell has value 0, mask 0 and the
  largest delta;
* static: a gate per feature group; a removed group takes the train median and its missing
  indicators become 1;
* image concepts: a gate per pathology concept; a removed concept's logit takes the mean over
  train images (a neutral value). The 1,024 pooled features are not a concept and stay fixed;
* the whole image: present or not (not differentiable, so it is scored by ablation only).

``GatedLogit.forward`` takes the gates first and the stay's fixed tensors after them, so a
method can attribute one gate and hold everything else fixed through captum's
``additional_forward_args``. All gates are continuous in [0, 1], which Integrated Gradients and
the learned masks need; at 0 or 1 the result equals the removal functions used in training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from fate_icu.models.data import StayData
from fate_icu.models.fusion import FusionNet
from fate_icu.models.image import FEATURE_WIDTH


@dataclass
class GateLayout:
    """What the gates refer to: names, sizes and the fill values of removed evidence."""

    channels: list[str]
    hours: int
    groups: list[str]
    concepts: list[str]
    group_of_column: np.ndarray  # [F + K]: static input column -> group index
    removed_static: np.ndarray  # [F + K]: value of a removed column (median; 1 for indicators)
    delta_fill: float
    concept_fill: np.ndarray  # [K] logit of a removed concept
    n_values: int  # F
    feature_width: int = FEATURE_WIDTH  # pooled image features before the concept logits

    @property
    def n_concepts(self) -> int:
        return len(self.concepts)


def build_layout(data: StayData, concepts: list[str], feature_width: int = FEATURE_WIDTH) -> GateLayout:
    """The gate layout for ``data``; concept fills come from the train stays that have an image."""
    if data.static_fill is None:
        raise ValueError("StayData has no static_fill: build it with models.data.assemble")
    groups = list(data.static_groups)
    n_values = data.static.shape[1]
    n_inputs = n_values + data.static_missing.shape[1]
    group_of_column = np.full(n_inputs, -1, dtype=np.int64)
    removed = np.zeros(n_inputs, dtype=np.float32)
    removed[:n_values] = data.static_fill
    for g, name in enumerate(groups):
        for column in data.static_groups[name]:
            group_of_column[column] = g
    for k, column in enumerate(data.indicator_columns):
        group_of_column[n_values + k] = group_of_column[column]
        removed[n_values + k] = 1.0
    if (group_of_column < 0).any():
        raise ValueError("every static column must belong to a feature group")
    if data.image.shape[1] != feature_width + len(concepts):
        raise ValueError(
            f"image vectors hold {data.image.shape[1]} values, expected {feature_width + len(concepts)}"
        )
    train_images = data.rows("train", paired_only=True)
    if len(train_images) == 0:
        raise ValueError("no train stay has an image: concept removal needs a train mean")
    fill = data.image[train_images][:, feature_width:].mean(axis=0)
    delta_fill = float(data.delta.max()) if data.delta.size else 1.0
    return GateLayout(
        list(data.channels), data.values.shape[1], groups, list(concepts), group_of_column, removed,
        delta_fill, fill.astype(np.float32), n_values, feature_width,
    )  # fmt: skip


class GatedLogit(nn.Module):
    """``forward(ts_gate, static_gate, concept_gate, image_present, values, mask, delta, static, image)``."""

    def __init__(self, net: FusionNet, layout: GateLayout) -> None:
        super().__init__()
        self.net = net.eval()
        self.layout = layout
        n_inputs = len(layout.group_of_column)
        one_hot = np.zeros((len(layout.groups), n_inputs), dtype=np.float32)
        one_hot[layout.group_of_column, np.arange(n_inputs)] = 1.0
        self.register_buffer("group_to_column", torch.from_numpy(one_hot))
        self.register_buffer("removed_static", torch.from_numpy(layout.removed_static))
        self.register_buffer("concept_fill", torch.from_numpy(layout.concept_fill))
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        ts_gate: torch.Tensor,
        static_gate: torch.Tensor,
        concept_gate: torch.Tensor,
        image_present: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        delta: torch.Tensor,
        static: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """Long-stay logit ``[B]`` with the given evidence kept (gate 1) or removed (gate 0)."""
        column_gate = static_gate @ self.group_to_column
        static_in = static * column_gate + self.removed_static * (1 - column_gate)
        image_in = image.clone()
        width = self.layout.feature_width
        logits = image[:, width:]
        image_in[:, width:] = logits * concept_gate + self.concept_fill * (1 - concept_gate)
        batch = {
            "static": static_in,
            "values": values * ts_gate,
            "mask": mask * ts_gate,
            "delta": delta * ts_gate + self.layout.delta_fill * (1 - ts_gate),
            "image": image_in,
        }
        n = values.shape[0]
        everyone = torch.ones(n, dtype=torch.bool, device=values.device)
        present = {"static": everyone, "ts": everyone, "image": image_present.bool()}
        return self.net(batch, present)["logit"]


def stay_tensors(data: StayData, rows: np.ndarray) -> dict[str, torch.Tensor]:
    """The fixed (gate-free) inputs of the stays ``rows``, as float tensors."""
    return {
        "values": torch.from_numpy(data.values[rows]),
        "mask": torch.from_numpy(data.mask[rows]),
        "delta": torch.from_numpy(data.delta[rows]),
        "static": torch.from_numpy(np.concatenate([data.static[rows], data.static_missing[rows]], axis=1)),
        "image": torch.from_numpy(data.image[rows]),
        "image_present": torch.from_numpy(data.has_image[rows]),
    }


def ones_gates(layout: GateLayout, n: int) -> dict[str, torch.Tensor]:
    """Gates that keep everything."""
    return {
        "ts": torch.ones(n, layout.hours, len(layout.channels)),
        "static": torch.ones(n, len(layout.groups)),
        "concept": torch.ones(n, layout.n_concepts),
    }


def logit(
    model: GatedLogit,
    fixed: dict[str, torch.Tensor],
    *,
    ts: torch.Tensor | None = None,
    static: torch.Tensor | None = None,
    concept: torch.Tensor | None = None,
    image_present: torch.Tensor | None = None,
) -> torch.Tensor:
    """The logit with any of the gates replaced (the others are all ones)."""
    n = fixed["values"].shape[0]
    base = ones_gates(model.layout, n)
    present = fixed["image_present"] if image_present is None else image_present
    return model(
        base["ts"] if ts is None else ts,
        base["static"] if static is None else static,
        base["concept"] if concept is None else concept,
        present,
        fixed["values"], fixed["mask"], fixed["delta"], fixed["static"], fixed["image"],
    )  # fmt: skip
