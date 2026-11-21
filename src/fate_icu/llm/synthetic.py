"""Synthetic evidence packets and passages, for smoke runs and tests of the explainer (Stage 6).

Nothing here comes from patient data: drivers are drawn from a fixed vocabulary of made-up but
plausible findings, and passages are generic one-line statements marked as synthetic. They exercise the
sampling, validation, training and generation code on the laptop, on Kaggle or on Colab (public data
only); the real packets and passages are made on the GCP VM and never leave it.
"""

from __future__ import annotations

import random
from typing import Any

from fate_icu.attribution.packet import EvidencePacket

# (modality, key, label, text of what was observed at a value, unit)
_STATIC = (
    ("sofa_static", "SOFA score (worst of first 24 h)", "total SOFA score (worst, first 24 h): {}", (1, 14)),
    ("demographics", "demographics", "age (years): {}", (22, 92)),
    ("labs", "laboratory results (mean of first 24 h)", "mean serum creatinine (mg/dL): {}", (0.5, 4.2)),
    ("vitals", "vital signs (mean of first 24 h)", "mean respiratory rate (breaths/min): {}", (11, 32)),
)
_SERIES = (
    ("heart_rate", "heart rate", "bpm", (55, 135)),
    ("creatinine", "serum creatinine", "mg/dL", (0.5, 4.0)),
    ("mbp", "mean arterial pressure", "mmHg", (50, 95)),
    ("lactate", "lactate", "mmol/L", (0.8, 6.0)),
    ("spo2", "oxygen saturation", "%", (84, 99)),
)
_IMAGE = (("Effusion", "pleural effusion"), ("Edema", "pulmonary edema"), ("Atelectasis", "atelectasis"))
_PASSAGES = (
    "Synthetic passage: a higher early organ-failure score is associated with a longer ICU stay.",
    "Synthetic passage: persistent abnormal vital signs in the first day are associated with longer stays.",
    "Synthetic passage: elevated creatinine early in the ICU stay is associated with longer stays.",
    "Synthetic passage: older age is associated with longer ICU stays in many cohorts.",
)


def _value(rng: random.Random, low: float, high: float) -> float:
    return round(rng.uniform(low, high), 1 if high < 20 else 0)


def _number(x: float) -> str:
    return f"{x:g}"


def random_packet(rng: random.Random) -> dict[str, Any]:
    """One valid evidence packet (as a dict) with 2 to 5 drivers sorted by effect size."""
    count = rng.randint(2, 5)
    days = round(rng.uniform(0.8, 12.0), 1)
    threshold = 3.7
    p_long = min(0.97, max(0.03, round(0.15 + 0.07 * days + rng.uniform(-0.08, 0.08), 2)))
    sizes = sorted((round(rng.uniform(0.15, 1.6), 2) for _ in range(count)), reverse=True)
    kinds = rng.sample(["static"] * len(_STATIC) + ["timeseries"] * len(_SERIES) + ["image"], count)
    used_static: set[str] = set()
    used_series: set[str] = set()
    drivers: list[dict[str, Any]] = []
    for index, (kind, size) in enumerate(zip(kinds, sizes, strict=True), start=1):
        effect = size if rng.random() < 0.65 else -size
        base = {
            "id": f"D{index}",
            "modality": kind,
            "effect": effect,
            "direction": "longer" if effect > 0 else "shorter",
        }
        if kind == "static":
            key, label, template, (low, high) = rng.choice([s for s in _STATIC if s[0] not in used_static])
            used_static.add(key)
            base |= {"key": key, "label": label, "observed": template.format(_number(_value(rng, low, high)))}
        elif kind == "timeseries":
            key, label, unit, (low, high) = rng.choice([s for s in _SERIES if s[0] not in used_series])
            used_series.add(key)
            start = rng.choice([0, 4, 8, 12, 16, 20])
            before, now = _value(rng, low, high), _value(rng, low, high)
            verb = "rose" if now > before else "fell" if now < before else "stayed near"
            what = (
                f"{verb} from {_number(before)} to {_number(now)} {unit}"
                if verb != "stayed near"
                else f"stayed near {_number(now)} {unit}"
            )
            base |= {"key": key, "label": label, "time": f"hours {start}-{start + 4}", "observed": what}
        else:
            key, name = rng.choice(_IMAGE)
            estimate = round(rng.uniform(0.2, 0.9), 2)
            base |= {
                "key": key,
                "label": f"{name} (image model estimate {estimate:.2f})",
                "region": rng.choice(["left lower lung", "right lower lung", "both lungs"]),
                "reported_finding": estimate >= 0.5,
            }
        drivers.append(base)
    packet = {
        "note": "synthetic",
        "prediction": {
            "p_long_stay": p_long,
            "predicted_los_days": days,
            "long_stay_threshold_days": threshold,
            "window_hours": 24,
        },
        "drivers": drivers,
    }
    return EvidencePacket.model_validate(packet).model_dump(exclude_none=True)


def random_passages(rng: random.Random, packet: dict[str, Any]) -> list[dict[str, str]]:
    """Zero to two passages with ids R1, R2 (zero for some packets, as in real retrieval)."""
    count = rng.choice([0, 1, 1, 2])
    chosen = rng.sample(_PASSAGES, count)
    return [{"rid": f"R{i}", "title": "synthetic", "text": text} for i, text in enumerate(chosen, start=1)]


def random_items(n: int, seed: int = 0) -> list[tuple[dict[str, Any], list[dict[str, str]]]]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        packet = random_packet(rng)
        items.append((packet, random_passages(rng, packet)))
    return items
