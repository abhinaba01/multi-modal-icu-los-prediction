"""The prompt of the explainer (Stage 6).

The language model sees one thing: the instructions below, the evidence packet (prediction and drivers) and
the retrieved passages, rendered as plain text. It never sees raw records, notes or the image. Gemma has no
separate system role, so everything goes in one user turn.

``build_messages`` is the chat the model is given. For sampling from the base model it carries two worked
examples (``few_shot=True``); for fine-tuning and for the tuned model the examples are dropped
(``few_shot=False``), because the weights have learned the format and a short prompt leaves room for the
passages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fate_icu.llm.validate import MAX_SENTENCES, MIN_SENTENCES

INSTRUCTIONS = f"""You explain an ICU length-of-stay prediction to a clinician, using only the evidence below.

Rules:
- Write {MIN_SENTENCES} to {MAX_SENTENCES} sentences of plain clinical English.
- The first sentence states the prediction: the expected stay in days and the probability of a long
  stay.
- Every sentence ends with at least one citation: [D1], [D2], ... for a driver or [R1], [R2], ... for a
  passage.
- Cite only ids that appear in the evidence. Use only numbers that appear in the evidence, in the
  prediction or in the passage you cite.
- Describe each driver with its own direction: "longer" if it points toward a longer stay, "shorter" if toward
  a shorter one. Say nothing about a driver that is not listed.
- Passages are background from the literature: cite one only for what it says. Do not recommend, start or stop
  any treatment.
- A finding labelled "image model" is the model's estimate, not a diagnosis."""

EXAMPLES: tuple[dict[str, Any], ...] = (
    {
        "packet": {
            "prediction": {
                "p_long_stay": 0.81,
                "predicted_los_days": 7.4,
                "long_stay_threshold_days": 3.7,
                "window_hours": 24,
            },
            "drivers": [
                {
                    "id": "D1",
                    "modality": "static",
                    "key": "sofa",
                    "label": "SOFA score (worst of first 24 h)",
                    "observed": "total SOFA score: 11; renal SOFA sub-score: 3",
                    "direction": "longer",
                    "effect": 1.4,
                },
                {
                    "id": "D2",
                    "modality": "timeseries",
                    "key": "heart_rate",
                    "label": "heart rate",
                    "time": "hours 12-16",
                    "observed": "rose from 82 to 118 bpm, above the reference range",
                    "direction": "longer",
                    "effect": 0.9,
                },
                {
                    "id": "D3",
                    "modality": "static",
                    "key": "demographics",
                    "label": "age",
                    "observed": "age: 41",
                    "direction": "shorter",
                    "effect": -0.3,
                },
            ],
        },
        "passages": [
            {
                "rid": "R1",
                "text": "Higher SOFA scores in the first 24 hours are associated with longer ICU stays.",
            }
        ],
        "text": (
            "The model predicts an ICU stay of about 7.4 days; the probability of a long stay (more than 3.7 "
            "days) is 81%, mainly because of the SOFA score (worst of first 24 h) [D1] and heart rate [D2]. "
            "A total SOFA score of 11 with a renal sub-score of 3 points toward a longer stay [D1], in line "
            "with the literature linking higher early SOFA scores to longer stays [R1]. "
            "Heart rate rose from 82 to 118 bpm in hours 12-16, above the reference range, which also points "
            "toward a longer stay [D2]. "
            "Age 41 points toward a shorter stay [D3]."
        ),
    },
    {
        "packet": {
            "prediction": {
                "p_long_stay": 0.12,
                "predicted_los_days": 1.9,
                "long_stay_threshold_days": 3.7,
                "window_hours": 24,
            },
            "drivers": [
                {
                    "id": "D1",
                    "modality": "static",
                    "key": "sofa",
                    "label": "SOFA score (worst of first 24 h)",
                    "observed": "total SOFA score: 1",
                    "direction": "shorter",
                    "effect": -1.1,
                },
                {
                    "id": "D2",
                    "modality": "timeseries",
                    "key": "creatinine",
                    "label": "serum creatinine",
                    "time": "hours 8-12",
                    "observed": "stayed near 0.8 mg/dL",
                    "direction": "shorter",
                    "effect": -0.5,
                },
            ],
        },
        "passages": [],
        "text": (
            "The model predicts an ICU stay of about 1.9 days; the probability of a long stay (more than 3.7 "
            "days) is 12%, mainly because of the SOFA score (worst of first 24 h) [D1] and serum creatinine "
            "[D2]. "
            "A total SOFA score of 1 points toward a shorter stay [D1]. "
            "Serum creatinine stayed near 0.8 mg/dL in hours 8-12, which points toward a shorter stay [D2]."
        ),
    },
)


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _plain(obj: Any) -> Any:
    """A pydantic model or a mapping as a plain dict (the packet and passages arrive in either form)."""
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def render_evidence(packet: Any, passages: Sequence[Any] = ()) -> str:
    """The packet and the passages as text, in a fixed layout (the same input always gives the same text)."""
    packet = _plain(packet)
    prediction = _get(packet, "prediction")
    lines = [
        "PREDICTION",
        f"- expected ICU stay: {_get(prediction, 'predicted_los_days'):g} days",
        f"- probability of a long stay (more than {_get(prediction, 'long_stay_threshold_days'):g} days): "
        f"{round(float(_get(prediction, 'p_long_stay')) * 100)}%",
        f"- made from the first {_get(prediction, 'window_hours')} hours of data",
        "",
        "DRIVERS (largest effect first)",
    ]
    for d in _get(packet, "drivers"):
        parts = [f"[{_get(d, 'id')}] {_get(d, 'label')}"]
        if _get(d, "time"):
            parts.append(f"period: {_get(d, 'time')}")
        if _get(d, "observed"):
            parts.append(f"observed: {_get(d, 'observed')}")
        if _get(d, "reference_range"):
            parts.append(f"reference range: {_get(d, 'reference_range')}")
        if _get(d, "region"):
            parts.append(f"region: {_get(d, 'region')}")
        parts.append(f"points toward a {_get(d, 'direction')} stay")
        lines.append("- " + " | ".join(parts))
    if passages:
        lines += ["", "PASSAGES (background from the literature)"]
        for p in passages:
            p = _plain(p)
            title = _get(p, "title")
            head = f"[{_get(p, 'rid')}]" + (f" {title}:" if title else "")
            lines.append(f"- {head} {_get(p, 'text')}")
    return "\n".join(lines)


def _example_block(example: Mapping[str, Any], number: int) -> str:
    return (
        f"Example {number}\n{render_evidence(example['packet'], example['passages'])}\n\n"
        f"Explanation:\n{example['text']}"
    )


def build_messages(
    packet: Any, passages: Sequence[Any] = (), *, few_shot: bool = False
) -> list[dict[str, str]]:
    """The chat for one packet: a single user turn (Gemma has no system role)."""
    blocks = [INSTRUCTIONS]
    if few_shot:
        blocks += [_example_block(e, i) for i, e in enumerate(EXAMPLES, start=1)]
        blocks.append("Now the real case.")
    blocks.append(render_evidence(packet, passages))
    blocks.append("Explanation:")
    return [{"role": "user", "content": "\n\n".join(blocks)}]
