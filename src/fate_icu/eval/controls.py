"""The two control inputs of the ablation grid (Stage 7).

* **C1: summary of abnormal values.** Instead of the evidence packet, the language model gets a deterministic
  summary of every time-series variable that left its reference range during the window, with no
  attribution. If a model given this names more units that the predictor never used (``unattributed`` in
  H2), the packet is what keeps explanations faithful. ``abnormal_values`` and ``summary_text`` build it
  from the raw hourly values.
* **C2: random drivers.** The packet keeps the prediction, the number of drivers and their effect sizes, but
  the units are drawn at random from a pool of truthfully described units of the stay. Explanations of such
  a packet are faithful to the packet and not to the model, which the claim-deletion test of faithfulness
  must catch.

Both take what they need as arguments (arrays, reference ranges, a pool), so they run on synthetic data
here and on real stays on the GCP VM.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Abnormal:
    name: str
    label: str
    unit: str
    side: str  # "above" or "below"
    extreme: float  # the highest value (above) or lowest value (below) in the window
    bound: float  # the reference limit that was crossed
    hours: int  # hours with a value outside the range


def abnormal_values(
    hourly: Mapping[str, np.ndarray],
    ranges: Mapping[str, tuple[float | None, float | None]],
    labels: Mapping[str, tuple[str, str]],
) -> list[Abnormal]:
    """Variables with values outside their reference range, most hours outside first (NaN = not measured).

    ``ranges[name] = (low, high)`` (either may be ``None``); ``labels[name] = (label, unit)``.
    """
    out = []
    for name, series in hourly.items():
        values = np.asarray(series, dtype=float)
        values = values[~np.isnan(values)]
        low, high = ranges.get(name, (None, None))
        if not values.size:
            continue
        label, unit = labels[name]
        above = values[values > high] if high is not None else values[:0]
        below = values[values < low] if low is not None else values[:0]
        if above.size >= below.size and above.size:
            out.append(Abnormal(name, label, unit, "above", float(above.max()), float(high), int(above.size)))
        elif below.size:
            out.append(Abnormal(name, label, unit, "below", float(below.min()), float(low), int(below.size)))
    return sorted(out, key=lambda a: (-a.hours, a.name))


def _number(x: float) -> str:
    return f"{x:g}" if abs(x) < 100 else f"{x:.0f}"


def summary_text(items: Sequence[Abnormal], window_hours: int = 24) -> str:
    """The C1 input: one line per abnormal variable, and a line saying so when nothing was abnormal."""
    if not items:
        return f"No measured variable left its reference range in the first {window_hours} hours."
    lines = [f"Values outside the reference range in the first {window_hours} hours:"]
    for a in items:
        unit = f" {a.unit}" if a.unit else ""
        word = "highest" if a.side == "above" else "lowest"
        lines.append(
            f"- {a.label}: {a.side} {_number(a.bound)}{unit} in {a.hours} h "
            f"({word} value {_number(a.extreme)}{unit})"
        )
    return "\n".join(lines)


def random_packet(packet: Any, pool: Sequence[Mapping[str, Any]], rng: random.Random) -> dict:
    """C2: a valid packet with the same prediction, count and effect sizes, with random units from ``pool``.

    Each pool entry is a driver dict without ``id`` and ``effect`` (``modality``, ``key``, ``label``,
    ``observed``, ``direction`` and the modality's other fields). Units already used by the original packet
    are allowed: the point is that the choice does not depend on the model.
    """
    raw = packet.model_dump() if hasattr(packet, "model_dump") else dict(packet)
    sizes = [abs(float(d["effect"])) for d in raw["drivers"]]
    if len(pool) < len(sizes):
        raise ValueError("the pool has fewer units than the packet has drivers")
    chosen = rng.sample(list(pool), len(sizes))
    drivers = []
    for number, (entry, size) in enumerate(zip(chosen, sizes, strict=True), start=1):
        drivers.append(
            {**entry, "id": f"D{number}", "effect": size if entry["direction"] == "longer" else -size}
        )
    return {**raw, "drivers": drivers}
