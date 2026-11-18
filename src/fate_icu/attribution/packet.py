"""The evidence packet: the only patient information the LLM ever sees (Stage 4).

One JSON per stay: the prediction and at most five drivers. Every ``observed`` string comes
from the deterministic templates below, never from an LLM, and every driver carries its
canonical ``key`` (a channel, a static group or an image concept) so that retrieval (Stage 5,
D-038) and the evaluation harness (Stage 7) can look it up without reading free text.

``effect`` is the removal effect Δ on the long-stay logit: positive when the unit pushed toward
a longer stay, negative toward a shorter one, and ``direction`` always agrees with its sign.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fate_icu.data.contract import Contract
from fate_icu.data.timeseries.variables import Variable

MAX_DRIVERS = 5
SCHEMA_VERSION = 1
NOT_RECORDED = "not recorded"
CONCEPT_NAMES = {
    "Atelectasis": "atelectasis",
    "Consolidation": "consolidation",
    "Pneumothorax": "pneumothorax",
    "Edema": "pulmonary edema",
    "Effusion": "pleural effusion",
    "Pneumonia": "pneumonia",
    "Cardiomegaly": "cardiomegaly",
    "Lung Lesion": "lung lesion",
    "Fracture": "fracture",
    "Lung Opacity": "lung opacity",
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum",
    "whole_image": "the chest X-ray",
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PacketPrediction(_Strict):
    p_long_stay: float = Field(ge=0.0, le=1.0)
    predicted_los_days: float = Field(ge=0.0)
    long_stay_threshold_days: float = Field(gt=0.0)
    window_hours: int = Field(gt=0)


class PacketDriver(_Strict):
    id: str = Field(pattern=r"^D[1-5]$")
    modality: Literal["timeseries", "static", "image"]
    key: str = Field(min_length=1)
    label: str = Field(min_length=1)
    time: str | None = None  # time series only
    observed: str | None = None
    reference_range: str | None = None  # time series only
    region: str | None = None  # image only
    direction: Literal["longer", "shorter"]
    effect: float
    reported_finding: bool | None = None  # image concepts only

    @model_validator(mode="after")
    def _fields_fit_the_modality(self) -> PacketDriver:
        if self.modality != "timeseries" and (self.time or self.reference_range):
            raise ValueError("time and reference_range belong to time-series drivers")
        if self.modality != "image" and (self.region or self.reported_finding is not None):
            raise ValueError("region and reported_finding belong to image drivers")
        if (self.direction == "longer") != (self.effect > 0):
            raise ValueError("direction must agree with the sign of the effect")
        return self

    @field_validator("effect")
    @classmethod
    def _finite_and_nonzero(cls, value: float) -> float:
        if not math.isfinite(value) or value == 0.0:
            raise ValueError("effect must be finite and not zero")
        return value


class EvidencePacket(_Strict):
    schema_version: int = SCHEMA_VERSION
    note: str | None = None  # synthetic fixtures say so here
    prediction: PacketPrediction
    drivers: list[PacketDriver] = Field(max_length=MAX_DRIVERS)

    @model_validator(mode="after")
    def _ordered(self) -> EvidencePacket:
        if [d.id for d in self.drivers] != [f"D{i}" for i in range(1, len(self.drivers) + 1)]:
            raise ValueError("driver ids must run D1, D2, ... in order")
        sizes = [abs(d.effect) for d in self.drivers]
        if sizes != sorted(sizes, reverse=True):
            raise ValueError("drivers must be sorted by the size of the effect, largest first")
        keys = [(d.modality, d.key, d.time) for d in self.drivers]
        if len(set(keys)) != len(keys):
            raise ValueError("a driver may appear once")
        return self


def check_keys(packet: EvidencePacket, allowed: dict[str, set[str]]) -> None:
    """Every driver key must be a known key of its modality (``rag.drivers.driver_keys``)."""
    for d in packet.drivers:
        if d.key not in allowed.get(d.modality, set()):
            raise ValueError(f"driver {d.id}: {d.key!r} is not a {d.modality} key")


# ---- deterministic text -------------------------------------------------------------------------


def format_number(x: float) -> str:
    """Three significant figures at most, no trailing zeros ("1.1", "98", "0.62")."""
    if not math.isfinite(x):
        return NOT_RECORDED
    if x != 0 and abs(x) < 0.01:
        return f"{x:.2g}"
    digits = 0 if abs(x) >= 100 else 1 if abs(x) >= 10 else 2
    text = f"{x:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def block_label(start_hour: int, end_hour: int) -> str:
    return f"hours {start_hour}-{end_hour}"


def _range_note(value: float, variable: Variable) -> str:
    low, high = variable.reference_range.low, variable.reference_range.high
    if high is not None and value > high:
        return ", above the reference range"
    if low is not None and value < low:
        return ", below the reference range"
    return ""


def describe_timeseries(variable: Variable, raw: np.ndarray, start_hour: int, end_hour: int) -> str:
    """What a variable did in a block, from its raw hourly values (NaN: not measured).

    Compared with the mean of the earlier hours when there are any: a change of 10 % or more is
    "rose" or "fell"; a smaller one is "stayed near".
    """
    raw = np.asarray(raw, dtype=float)
    block = raw[start_hour:end_hour]
    block = block[~np.isnan(block)]
    if block.size == 0:
        return "not measured in this period"
    now = float(block.mean())
    earlier = raw[:start_hour]
    earlier = earlier[~np.isnan(earlier)]
    unit = f" {variable.unit}" if variable.unit else ""
    note = _range_note(now, variable)
    if earlier.size == 0:
        return f"averaged {format_number(now)}{unit}{note}"
    before = float(earlier.mean())
    if abs(now - before) >= 0.10 * max(abs(before), 1e-9):
        verb = "rose" if now > before else "fell"
        return f"{verb} from {format_number(before)} to {format_number(now)}{unit}{note}"
    return f"stayed near {format_number(now)}{unit}{note}"


def describe_static(contract: Contract, group: str, values: dict[str, object], limit: int = 3) -> str:
    """Up to ``limit`` features of a group, in contract order, as ``label: value``."""
    names = [f.name for f in contract.features if f.group == group and f.name in values]
    parts = []
    for name in names[:limit]:
        value = values[name]
        missing = value is None or (isinstance(value, float) and math.isnan(value))
        if missing:
            text = NOT_RECORDED
        elif isinstance(value, (int, float, np.integer, np.floating)):
            text = format_number(float(value))
        else:
            text = str(value)
        parts.append(f"{contract.feature(name).label}: {text}")
    return "; ".join(parts) if parts else NOT_RECORDED


def describe_image(concept: str, probability: float | None, reported: bool) -> str:
    """Label of an image driver. A finding is worded as an image-model estimate, never a diagnosis."""
    name = CONCEPT_NAMES.get(concept, concept)
    if concept == "whole_image":
        return "chest X-ray (overall appearance, image model)"
    if probability is None:
        return f"image model signal for {name}"
    if reported:
        return f"{name} (image model estimate {probability:.2f})"
    return f"image model signal for {name} (estimate {probability:.2f}, below its reporting threshold)"
