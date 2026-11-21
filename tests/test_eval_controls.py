"""The control inputs C1 (summary of abnormal values) and C2 (random drivers)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from fate_icu.attribution.packet import EvidencePacket
from fate_icu.eval.controls import abnormal_values, random_packet, summary_text
from fate_icu.llm.prompt import EXAMPLES

NAN = float("nan")
RANGES = {"heart_rate": (60.0, 100.0), "creatinine": (0.6, 1.3), "spo2": (94.0, None)}
LABELS = {
    "heart_rate": ("heart rate", "bpm"),
    "creatinine": ("serum creatinine", "mg/dL"),
    "spo2": ("SpO2", "%"),
}


def test_abnormal_values_finds_values_outside_the_range_and_ignores_missing_hours():
    hourly = {
        "heart_rate": np.array([90, 118, 130, NAN, 95]),  # 2 hours above 100, the highest is 130
        "creatinine": np.array([0.9, 1.0, NAN]),  # normal
        "spo2": np.array([97, 92, 91, 96]),  # 2 hours below 94, the lowest is 91
    }
    items = abnormal_values(hourly, RANGES, LABELS)
    assert [(a.name, a.side, a.extreme, a.bound, a.hours) for a in items] == [
        ("heart_rate", "above", 130.0, 100.0, 2),
        ("spo2", "below", 91.0, 94.0, 2),
    ]


def test_a_variable_never_measured_is_not_abnormal_and_the_order_is_by_hours_outside():
    hourly = {
        "heart_rate": np.array([NAN, NAN]),
        "creatinine": np.array([2.0, 2.2, 2.4]),
        "spo2": np.array([90.0]),
    }
    items = abnormal_values(hourly, RANGES, LABELS)
    assert [a.name for a in items] == ["creatinine", "spo2"]  # 3 hours outside, then 1


def test_the_summary_text_is_deterministic_and_says_when_nothing_is_abnormal():
    items = abnormal_values({"heart_rate": np.array([118.0, 130.0])}, RANGES, LABELS)
    text = summary_text(items)
    assert text == (
        "Values outside the reference range in the first 24 hours:\n"
        "- heart rate: above 100 bpm in 2 h (highest value 130 bpm)"
    )
    assert summary_text(items) == text
    assert "No measured variable" in summary_text([])


POOL = [
    {
        "modality": "timeseries", "key": "lactate", "label": "lactate", "time": "hours 4-8",
        "observed": "rose from 1.1 to 3.4 mmol/L, above the reference range", "direction": "longer",
    },
    {
        "modality": "timeseries", "key": "map", "label": "mean arterial pressure", "time": "hours 0-4",
        "observed": "stayed near 82 mmHg", "direction": "shorter",
    },
    {
        "modality": "static", "key": "demographics", "label": "age",
        "observed": "age: 61", "direction": "longer",
    },
    {
        "modality": "static", "key": "labs", "label": "laboratory results (mean of first 24 h)",
        "observed": "mean serum sodium: 139", "direction": "shorter",
    },
]  # fmt: skip


def test_a_random_packet_keeps_the_prediction_the_count_and_the_effect_sizes():
    original = EXAMPLES[0]["packet"]  # three drivers
    packet = random_packet(original, POOL, random.Random(0))
    EvidencePacket.model_validate(packet)  # valid: ids in order, sorted by |effect|, no duplicate unit
    assert packet["prediction"] == original["prediction"]
    assert [abs(d["effect"]) for d in packet["drivers"]] == [abs(d["effect"]) for d in original["drivers"]]
    assert all((d["effect"] > 0) == (d["direction"] == "longer") for d in packet["drivers"])


def test_the_random_choice_is_reproducible_and_depends_on_the_seed():
    original = EXAMPLES[0]["packet"]
    a = random_packet(original, POOL, random.Random(1))
    b = random_packet(original, POOL, random.Random(1))
    seen = {
        tuple(d["key"] for d in random_packet(original, POOL, random.Random(s))["drivers"]) for s in range(20)
    }
    assert a == b and len(seen) > 1


def test_a_pool_smaller_than_the_packet_is_an_error():
    with pytest.raises(ValueError, match="fewer units"):
        random_packet(EXAMPLES[0]["packet"], POOL[:2], random.Random(0))
