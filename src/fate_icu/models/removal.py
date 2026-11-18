"""What "this evidence was removed" means as a model input (Stage 3, reused by Stages 4 and 7).

Training removes random evidence units so that removal becomes an ordinary input; the
deletion tests of Stages 4 and 7 then measure reliance, not distribution shift
(docs/stages.md, Stage 3: robustness training). Both uses call the functions here, so the
two meanings of "removed" cannot drift apart.

* a time-series unit is one variable over one block of hours: its values become 0 (the
  train mean after standardisation), its mask 0 and its delta the largest delta, which is
  what a channel that was never observed looks like;
* a static group's columns take the train median, and the missing indicator of each of
  its columns that has one is set to 1;
* a modality is removed by marking it absent (``present`` False): the fusion model then
  leaves it out entirely.
"""

from __future__ import annotations

import torch


def block_unit_mask(
    rows: int, hours: int, channels: int, block_hours: int, fraction: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Bool ``[rows, hours, channels]``: True where a (block x variable) unit is removed.

    Args:
        rows: Number of stays.
        hours: Length of the hourly grid.
        channels: Number of variables.
        block_hours: Hours per unit (4 in the spec); the last block may be shorter.
        fraction: ``[rows]`` removal probability per stay.
        generator: Source of randomness.
    """
    n_blocks = -(-hours // block_hours)
    draw = torch.rand(rows, n_blocks, channels, generator=generator)
    units = draw < fraction[:, None, None]
    return units.repeat_interleave(block_hours, dim=1)[:, :hours]


def remove_ts_units(
    values: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor, units: torch.Tensor, delta_fill: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Copies of ``values``, ``mask`` and ``delta`` with the ``units`` removed."""
    return (
        values.masked_fill(units, 0.0),
        mask.masked_fill(units, 0.0),
        delta.masked_fill(units, float(delta_fill)),
    )


def remove_static_groups(
    static: torch.Tensor,
    removed: torch.Tensor,
    group_columns: list[list[int]],
    fill: torch.Tensor,
    indicator_columns: list[int],
) -> torch.Tensor:
    """Copy of the ``[B, F + K]`` static input with groups removed.

    Args:
        static: Values ``[B, F]`` followed by the ``K`` missing indicators.
        removed: Bool ``[B, n_groups]``: True removes that group for that stay.
        group_columns: Encoded column positions of each group.
        fill: ``[F]`` values a removed column takes (the train median).
        indicator_columns: The column each of the ``K`` indicators belongs to.
    """
    out = static.clone()
    n_values = fill.shape[0]
    position_of = {column: k for k, column in enumerate(indicator_columns)}
    for g, columns in enumerate(group_columns):
        rows = removed[:, g]
        if not rows.any():
            continue
        for column in columns:
            out[rows, column] = fill[column]
            if column in position_of:
                out[rows, n_values + position_of[column]] = 1.0
    return out
