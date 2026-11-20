"""The deterministic explanation of an evidence packet (Stage 6).

No model is involved: the sentences are filled in from the packet, so the text can only say what the
packet says (drivers, directions, observed values) and always meets the contract of ``validate.check``.
It is the fallback when a language model fails the contract three times, and the "template explanation"
row (E4) of the Stage 7 grid. Layout: one sentence for the prediction, one for each of the three
strongest drivers, one that gathers the rest.
"""

from __future__ import annotations

from typing import Any

from fate_icu.llm.validate import Validation, check

MAX_DRIVER_SENTENCES = 3


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name)


def _days(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".") if x < 10 else f"{x:.0f}"


def _number(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:g}"


def _capital(text: str) -> str:
    return text[:1].upper() + text[1:]


def explain(packet: Any) -> str:
    """The explanation of ``packet`` (an ``EvidencePacket`` or its dict form)."""
    prediction, drivers = _get(packet, "prediction"), list(_get(packet, "drivers"))
    if not drivers:
        raise ValueError("an explanation needs at least one driver in the packet")
    days = float(_get(prediction, "predicted_los_days"))
    percent = round(float(_get(prediction, "p_long_stay")) * 100)
    threshold = float(_get(prediction, "long_stay_threshold_days"))
    first_two = [f"{_get(d, 'label')} [{_get(d, 'id')}]" for d in drivers[:2]]
    sentences = [
        f"The model predicts an ICU stay of about {_days(days)} days; the probability of a long stay "
        f"(more than {_number(threshold)} days) is {percent}%, mainly because of {' and '.join(first_two)}."
    ]
    for d in drivers[:MAX_DRIVER_SENTENCES]:
        observed = _get(d, "observed")
        detail = f"{observed}, which points" if observed else "which points"
        sentences.append(
            f"{_capital(_get(d, 'label'))}: {detail} toward a {_get(d, 'direction')} stay [{_get(d, 'id')}]."
        )
    rest = drivers[MAX_DRIVER_SENTENCES:]
    if rest:
        parts = [f"{_get(d, 'label')} toward a {_get(d, 'direction')} stay [{_get(d, 'id')}]" for d in rest]
        sentences.append(f"Smaller contributions came from {' and '.join(parts)}.")
    if len(sentences) < 3:  # a packet with one driver: say what the estimate is, still citing the driver
        only = drivers[0]
        window = _number(float(_get(prediction, "window_hours")))
        sentences.append(
            f"This model estimate, made from the first {window} hours of data, leans "
            f"toward a {_get(only, 'direction')} stay [{_get(only, 'id')}]."
        )
    return " ".join(sentences)


def explain_checked(packet: Any) -> tuple[str, Validation]:
    """The template text and its validation; the template is expected to pass (a test enforces it)."""
    text = explain(packet)
    return text, check(text, packet)
