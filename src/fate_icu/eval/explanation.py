"""Explanation faithfulness: does the text name what actually moves the model? (Stage 7)

The text of an explanation is compared with its evidence packet by a deterministic lexicon, no language model:

* ``Lexicon`` finds *mentions* of evidence units in a text: time-series channels (label and synonyms from
  ``configs/variables.yaml``), static feature groups (the group label and the feature labels of the contract),
  image concepts and the whole image. Matching is case-insensitive on whole words, longest phrase first, and a
  span is used once, so "mean heart rate (beats/min)" is the static vitals group and never also the channel
  "heart rate". A phrase that two units share is an error when the lexicon is built.
* ``driver_precision``, ``driver_recall_at_k``, ``stated_directions`` and ``direction_accuracy`` compare the
  mentions and the direction words with the packet.
* ``numeric_consistency`` is the share of sentences whose numbers all belong to the prediction or to a
  cited driver
  or passage (the number rule of the output contract).
* ``claim_deletion_validity`` re-scores the model without each named driver: the claim holds if the logit
  moves
  in the stated direction by at least tau. ``delta_of`` is supplied by the caller, because it needs the model.
* ``packet_sensitivity`` drops a driver from the packet, or adds a decoy, regenerates the text and checks
  that the
  text follows. ``generate`` is supplied by the caller.

Every function takes one explanation; the aggregation over stays (with patient-level bootstrap) is in
``run.py``.
A metric that is undefined for a text (no entities mentioned, no direction stated) returns ``None``, which the
caller counts and excludes; it is never turned into a 0 or a 1.
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fate_icu.attribution.packet import CONCEPT_NAMES
from fate_icu.llm.validate import CITATION, DIRECTION_WORDS, split_sentences, unsupported_numbers

WHOLE_IMAGE_PHRASES = ("chest x-ray", "chest radiograph", "chest film", "radiograph", "x-ray")


@dataclass(frozen=True)
class Entity:
    modality: str  # "timeseries" | "static" | "image"
    key: str


class Lexicon:
    """Phrases of evidence units and a longest-first, non-overlapping matcher."""

    def __init__(self, phrases: Mapping[Entity, Sequence[str]]):
        owner: dict[str, Entity] = {}
        for entity, texts in phrases.items():
            for text in texts:
                norm = " ".join(text.casefold().split())
                if not norm:
                    continue
                if owner.get(norm, entity) != entity:
                    raise ValueError(f"phrase {norm!r} belongs to both {owner[norm]} and {entity}")
                owner[norm] = entity
        self._owner = owner
        self._patterns = {
            norm: re.compile(r"(?<!\w)" + re.escape(norm).replace(r"\ ", r"\s+") + r"(?!\w)")
            for norm in owner
        }
        self._order = sorted(owner, key=lambda n: (-len(n), n))

    @property
    def entities(self) -> set[Entity]:
        return set(self._owner.values())

    def find(self, text: str) -> list[tuple[Entity, int, int]]:
        """Every mention as ``(entity, start, end)`` in text order; each character belongs to one mention."""
        lowered = text.casefold()
        taken = [False] * len(lowered)
        found: list[tuple[Entity, int, int]] = []
        for norm in self._order:
            for m in self._patterns[norm].finditer(lowered):
                if any(taken[m.start() : m.end()]):
                    continue
                for i in range(m.start(), m.end()):
                    taken[i] = True
                found.append((self._owner[norm], m.start(), m.end()))
        return sorted(found, key=lambda f: f[1])

    def entities_in(self, text: str) -> set[Entity]:
        return {e for e, _, _ in self.find(text)}


def build_lexicon(variables=None, contract=None, concepts: Sequence[str] | None = None) -> Lexicon:
    """The lexicon of every unit a driver can name, from the variable set and the contract."""
    from fate_icu.data.contract import load_contract
    from fate_icu.data.timeseries.variables import load_variables
    from fate_icu.rag.drivers import STATIC_FEATURE_SET

    variables = variables or load_variables()
    contract = contract or load_contract()
    phrases: dict[Entity, list[str]] = {}
    for v in variables.variables:
        phrases[Entity("timeseries", v.name)] = [v.label, *v.synonyms]
    allowed = set(contract.feature_names(STATIC_FEATURE_SET))
    for group in contract.feature_sets[STATIC_FEATURE_SET].groups:
        labels = [contract.feature_groups[group].label]
        labels += [f.label for f in contract.features if f.group == group and f.name in allowed]
        phrases[Entity("static", group)] = labels
    for concept in concepts or ():
        names = [CONCEPT_NAMES.get(concept, concept), concept]
        phrases[Entity("image", concept)] = names
    phrases[Entity("image", "whole_image")] = list(WHOLE_IMAGE_PHRASES)
    return Lexicon(phrases)


# ---- helpers on the packet ------------------------------------------------------------------------


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _drivers(packet: Any) -> list[Any]:
    return list(_get(packet, "drivers"))


def driver_entities(packet: Any) -> dict[str, Entity]:
    """``{driver id: entity}`` for the drivers of a packet."""
    return {_get(d, "id"): Entity(_get(d, "modality"), _get(d, "key")) for d in _drivers(packet)}


# ---- mention metrics ------------------------------------------------------------------------------


def driver_precision(text: str, packet: Any, lexicon: Lexicon) -> float | None:
    """Share of the distinct entities named in the text that are drivers of the packet (``None``: none
    named)."""
    named = lexicon.entities_in(text)
    if not named:
        return None
    return len(named & set(driver_entities(packet).values())) / len(named)


def driver_recall_at_k(text: str, packet: Any, lexicon: Lexicon, k: int = 3) -> float | None:
    """Share of the ``k`` largest drivers (by |effect|) that the text names (``None``: the packet has
    none)."""
    ranked = sorted(_drivers(packet), key=lambda d: -abs(float(_get(d, "effect"))))[:k]
    if not ranked:
        return None
    named = lexicon.entities_in(text)
    return sum(Entity(_get(d, "modality"), _get(d, "key")) in named for d in ranked) / len(ranked)


def stated_directions(text: str, packet: Any, lexicon: Lexicon) -> dict[str, set[str]]:
    """For each driver the text refers to (by name or by its [D#] citation): the direction words of those
    sentences.

    A driver the text never refers to is absent; a driver referred to in sentences without a direction word
    has an
    empty set.
    """
    by_entity = {e: i for i, e in driver_entities(packet).items()}
    out: dict[str, set[str]] = {}
    for sentence in split_sentences(text):
        ids = {i for i in CITATION.findall(sentence) if i in driver_entities(packet)}
        ids |= {by_entity[e] for e in lexicon.entities_in(sentence) if e in by_entity}
        words = {w for w in DIRECTION_WORDS if re.search(rf"\b{w}\b", sentence.lower())}
        for i in ids:
            out.setdefault(i, set()).update(words)
    return out


def direction_accuracy(text: str, packet: Any, lexicon: Lexicon) -> float | None:
    """Share of the drivers with a stated direction whose statement is exactly the packet's direction."""
    truth = {_get(d, "id"): _get(d, "direction") for d in _drivers(packet)}
    stated = {i: w for i, w in stated_directions(text, packet, lexicon).items() if w}
    if not stated:
        return None
    return sum(words == {truth[i]} for i, words in stated.items()) / len(stated)


def numeric_consistency(text: str, packet: Any, passages: Sequence[Any] = ()) -> float | None:
    """Share of the sentences whose numbers all come from the prediction or a cited driver or passage."""
    sentences = split_sentences(text)
    if not sentences:
        return None
    return sum(not unsupported_numbers(s, packet, passages) for s in sentences) / len(sentences)


# ---- tests that need the model or the generator -------------------------------------------------------


def claim_deletion_validity(
    text: str, packet: Any, lexicon: Lexicon, delta_of: Callable[[str], float], tau: float
) -> float | None:
    """Share of the named drivers whose claim survives deletion.

    ``delta_of(driver_id)`` is the model's logit with the stay's evidence minus its logit with that
    driver's unit
    removed (positive: the unit pushes toward a longer stay). The text states "longer" or "shorter" for the
    driver;
    the claim is valid if ``delta`` has that sign and ``|delta| >= tau``. Drivers with no stated direction,
    or both,
    are not tested.
    """
    tested = [(i, next(iter(w))) for i, w in stated_directions(text, packet, lexicon).items() if len(w) == 1]
    if not tested:
        return None
    ok = 0
    for driver_id, word in tested:
        delta = float(delta_of(driver_id))
        ok += (delta >= tau) if word == "longer" else (delta <= -tau)
    return ok / len(tested)


def drop_driver(packet: Any, driver_id: str) -> dict:
    """A copy of the packet (as a dict) without one driver; the remaining ids are renumbered D1, D2, ..."""
    raw = copy.deepcopy(packet.model_dump() if hasattr(packet, "model_dump") else dict(packet))
    kept = [d for d in raw["drivers"] if d["id"] != driver_id]
    for number, d in enumerate(kept, start=1):
        d["id"] = f"D{number}"
    raw["drivers"] = kept
    return raw


def add_decoy(packet: Any, decoy: Mapping[str, Any]) -> dict:
    """A copy of the packet with one more driver appended as the *smallest* (so ids and order stay valid).

    ``decoy`` is a driver dict without ``id`` and ``effect``; its effect is set just below the smallest,
    with the sign
    of its direction.
    """
    raw = copy.deepcopy(packet.model_dump() if hasattr(packet, "model_dump") else dict(packet))
    smallest = min((abs(float(d["effect"])) for d in raw["drivers"]), default=0.1)
    size = smallest * 0.9
    new = {
        **decoy,
        "id": f"D{len(raw['drivers']) + 1}",
        "effect": size if decoy["direction"] == "longer" else -size,
    }
    raw["drivers"].append(new)
    return raw


def packet_sensitivity(
    packet: Any,
    lexicon: Lexicon,
    generate: Callable[[Any], str],
    decoy: Mapping[str, Any],
    *,
    drop_id: str = "D1",
) -> dict[str, bool | None]:
    """Does the text follow the packet when it changes?

    * ``drop_followed``: after ``drop_id`` is deleted, the regenerated text no longer names its entity
      (``None`` if the original text did not name it, so there is nothing to follow).
    * ``decoy_followed``: after the decoy is added, the regenerated text names the decoy's entity.
    """
    entities = driver_entities(packet)
    target = entities.get(drop_id)
    original = lexicon.entities_in(generate(packet))
    dropped_text = generate(drop_driver(packet, drop_id))
    drop_followed = None if target not in original else target not in lexicon.entities_in(dropped_text)
    decoy_entity = Entity(decoy["modality"], decoy["key"])
    decoy_followed = decoy_entity in lexicon.entities_in(generate(add_decoy(packet, decoy)))
    return {"drop_followed": drop_followed, "decoy_followed": decoy_followed}


def per_stay_metrics(
    text: str, packet: Any, passages: Sequence[Any], lexicon: Lexicon, *, k: int = 3
) -> dict[str, float | None]:
    """The model-free metrics of one explanation (the ones that need no re-scoring or regeneration)."""
    return {
        "driver_precision": driver_precision(text, packet, lexicon),
        "unattributed_rate": None
        if driver_precision(text, packet, lexicon) is None
        else 1.0 - driver_precision(text, packet, lexicon),
        f"driver_recall_at_{k}": driver_recall_at_k(text, packet, lexicon, k),
        "direction_accuracy": direction_accuracy(text, packet, lexicon),
        "numeric_consistency": numeric_consistency(text, packet, passages),
    }


def random_decoy(packet: Any, pool: Sequence[Mapping[str, Any]], rng: random.Random) -> Mapping[str, Any]:
    """A decoy driver (from ``pool``) whose entity is not already in the packet."""
    taken = set(driver_entities(packet).values())
    options = [d for d in pool if Entity(d["modality"], d["key"]) not in taken]
    if not options:
        raise ValueError("every pool entry is already a driver of this packet")
    return rng.choice(options)
