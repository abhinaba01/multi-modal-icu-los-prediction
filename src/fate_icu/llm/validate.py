"""The output contract of an explanation (Stage 6).

An explanation is 3 to 5 sentences of plain clinical English that use only what is in the evidence packet
(and the retrieved passages, when there are any). ``check`` returns every rule an explanation breaks; an
explanation is accepted only when it breaks none. Rules:

* **sentence_count**: 3 to 5 sentences;
* **first_sentence_prediction**: the first sentence states the prediction (the predicted days or the
  probability of a long stay);
* **citation_missing**: every sentence cites at least one ``[D#]`` (a driver) or ``[R#]`` (a passage);
* **unknown_id**: only IDs that exist in the packet or the passage list are cited;
* **number_not_in_packet**: every number in a sentence is the prediction's or belongs to a driver or
  passage *that sentence cites*, so a made-up figure, or one borrowed from another driver, is caught;
* **direction**: after the first sentence, a sentence that cites drivers says "longer" or "shorter" as
  each cited driver does, and uses no direction word that none of its cited drivers has;
* **treatment_advice**: no treatment recommendations.

The first sentence is exempt from the direction rule: it states the prediction, not a driver's effect.
Failures are reported by rule name, so retry and fallback rates can be counted (never hidden).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MIN_SENTENCES, MAX_SENTENCES = 3, 5
CITATION = re.compile(r"\[(D\d+|R\d+)\]")
NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[])")
TREATMENT = re.compile(
    r"\b(recommend\w*|prescrib\w*|administer\w*|initiat\w*|should\s+(?:be\s+)?(?:given|treated|started|"
    r"administered|receive|undergo)|consider\s+(?:giving|starting|treating|adding|using)|"
    r"(?:start|begin|stop|discontinue)\s+(?:the\s+)?(?:treatment|therapy|medication|antibiotics?|drug))\b",
    re.IGNORECASE,
)
DIRECTION_WORDS = ("longer", "shorter")


@dataclass(frozen=True)
class Validation:
    ok: bool
    failures: tuple[str, ...]

    def __bool__(self) -> bool:
        return self.ok


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text.strip()) if s.strip()]


def _numbers(text: str) -> set[float]:
    without_citations = CITATION.sub(" ", text)
    return {round(float(m), 6) for m in NUMBER.findall(without_citations)}


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name)


def prediction_numbers(packet: Any) -> set[float]:
    """Every way the explanation may print the prediction: days, percent, the threshold, the window."""
    prediction = _get(packet, "prediction")
    p, days = float(_get(prediction, "p_long_stay")), float(_get(prediction, "predicted_los_days"))
    thr, window = float(_get(prediction, "long_stay_threshold_days")), float(_get(prediction, "window_hours"))
    out = {round(p * 100, 6), round(round(p * 100), 6), round(round(p * 100, 1), 6)}
    out |= {round(days, 6), round(round(days, 1), 6), round(round(days), 6)}
    out |= {round(thr, 6), round(round(thr, 1), 6), round(round(thr), 6), round(window, 6)}
    return out


def driver_numbers(driver: Any) -> set[float]:
    """The numbers in one driver's text fields (label, observed value, time, reference range, region)."""
    out: set[float] = set()
    for field in ("label", "observed", "time", "reference_range", "region"):
        value = _get(driver, field)
        if value:
            out |= _numbers(str(value))
    return out


def unsupported_numbers(sentence: str, packet: Any, passages: Sequence[Any] = ()) -> set[float]:
    """The numbers of one sentence that belong neither to the prediction nor to a driver or passage it cites.

    This is the number rule of ``check`` for a single sentence; Stage 7 uses it for numeric consistency.
    """
    drivers = {_get(d, "id"): d for d in _get(packet, "drivers")}
    passage_nums = {_get(p, "rid"): _numbers(str(_get(p, "text"))) for p in passages}
    allowed = set(prediction_numbers(packet))
    for i in CITATION.findall(sentence):
        if i in drivers:
            allowed |= driver_numbers(drivers[i])
        allowed |= passage_nums.get(i, set())
    return _numbers(sentence) - allowed


def check(text: str, packet: Any, passages: Sequence[Any] = ()) -> Validation:
    """Every rule the text breaks (empty when it meets the contract)."""
    failures: list[str] = []
    sentences = split_sentences(text)
    if not MIN_SENTENCES <= len(sentences) <= MAX_SENTENCES:
        failures.append("sentence_count")
    drivers = {_get(d, "id"): d for d in _get(packet, "drivers")}
    passage_ids = {_get(p, "rid") for p in passages}

    if sentences:
        prediction = _get(packet, "prediction")
        days, percent = (
            float(_get(prediction, "predicted_los_days")),
            float(_get(prediction, "p_long_stay")) * 100,
        )
        stated = {round(v, 6) for v in (round(days, 1), round(days), round(percent), round(percent, 1))}
        if not _numbers(sentences[0]) & stated:
            failures.append("first_sentence_prediction")

    driver_nums = {i: driver_numbers(d) for i, d in drivers.items()}
    passage_nums = {_get(p, "rid"): _numbers(str(_get(p, "text"))) for p in passages}
    prediction_nums = prediction_numbers(packet)
    missing = unknown = wrong_direction = bad_number = False
    for index, sentence in enumerate(sentences):
        ids = CITATION.findall(sentence)
        allowed = set(prediction_nums)
        for i in ids:
            allowed |= driver_nums.get(i, set()) | passage_nums.get(i, set())
        if _numbers(sentence) - allowed:
            bad_number = True
        if not ids:
            missing = True
            continue
        if any(i not in drivers and i not in passage_ids for i in ids):
            unknown = True
        if index == 0:
            continue
        cited = [drivers[i] for i in ids if i in drivers]
        if not cited:
            continue  # a sentence that cites only passages makes no claim about a driver
        lowered = sentence.lower()
        words = {w for w in DIRECTION_WORDS if re.search(rf"\b{w}\b", lowered)}
        expected = {_get(d, "direction") for d in cited}
        if words != expected:
            wrong_direction = True
    if bad_number:
        failures.append("number_not_in_packet")
    if missing:
        failures.append("citation_missing")
    if unknown:
        failures.append("unknown_id")
    if wrong_direction:
        failures.append("direction")
    if TREATMENT.search(text):
        failures.append("treatment_advice")
    return Validation(ok=not failures, failures=tuple(dict.fromkeys(failures)))
