"""Retrieval queries: one fixed template over curated driver labels (Stage 5).

    "{label} associated with {direction} intensive care unit length of stay"

is applied to every driver key (:func:`fate_icu.rag.drivers.driver_keys`, 50 keys) in
both directions (longer, shorter): 100 queries. The labels live in
``configs/rag/queries.yaml``; a label that differs from its source label (variables.yaml,
the contract's feature groups, the image concept name) must say why. No query can hold a
patient value: labels are fixed text, checked to contain no numbers, percentages or units.
LLM query rewriting is an ablation, not the default.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from fate_icu.rag.drivers import IMAGE_MODEL_WEIGHTS, DriverKey, Modality, driver_keys
from fate_icu.utils.env import REPO_ROOT

DEFAULT_QUERIES_PATH = REPO_ROOT / "configs" / "rag" / "queries.yaml"
DIRECTIONS = ("longer", "shorter")
# a digit not inside a word ("SpO2" is fine, "24 h" is not), a percent sign, or a unit of
# measurement ("first day" is fine: without digits a duration names no patient value)
_PATIENT_VALUE = re.compile(r"(?<![A-Za-z])\d|%|\b(?:mg|mmol|mmhg|kg|ml|dl|bpm|mcg|ug|meq)\b", re.IGNORECASE)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class QueryLabel(_Strict):
    modality: Modality
    key: str
    label: str
    source_label: str
    reason: str | None = None


class QuerySpec(_Strict):
    version: int
    template: str
    directions: list[str]
    image_model: str
    labels: list[QueryLabel]

    @property
    def sha256(self) -> str:
        text = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def label_for(self, modality: str, key: str) -> QueryLabel:
        for label in self.labels:
            if label.modality == modality and label.key == key:
                return label
        raise KeyError(f"no query label for {modality}:{key}")


class QueryDef(_Strict):
    query_id: str
    modality: Modality
    key: str
    direction: str
    text: str


def check_no_patient_values(text: str) -> None:
    """Raise if ``text`` holds a number, a percentage or a unit (a possible patient value)."""
    match = _PATIENT_VALUE.search(text)
    if match:
        raise ValueError(f"query text may hold a patient value ({match.group(0)!r}): {text!r}")


def load_query_spec(
    path: str | Path = DEFAULT_QUERIES_PATH, *, keys: list[DriverKey] | None = None
) -> QuerySpec:
    """Load and check ``configs/rag/queries.yaml`` against the driver keys.

    Raises:
        ValueError: Missing or extra keys, a changed label without a reason, a label that may
            hold a patient value, or a template without exactly ``{label}`` and ``{direction}``.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    keys = driver_keys() if keys is None else keys
    fields = sorted({name for _, name, _, _ in string.Formatter().parse(raw["template"]) if name is not None})
    if fields != ["direction", "label"]:
        raise ValueError(f"the template must use exactly {{label}} and {{direction}}, not {fields}")
    check_no_patient_values(raw["template"].replace("{label}", "").replace("{direction}", ""))
    if list(raw["directions"]) != list(DIRECTIONS):
        raise ValueError(f"directions must be {list(DIRECTIONS)}")
    if raw["image_model"] != IMAGE_MODEL_WEIGHTS:
        raise ValueError(
            f"the labels are written for {raw['image_model']}, the image model is {IMAGE_MODEL_WEIGHTS}"
        )
    entries = raw["labels"]
    given = {(modality, str(key)) for modality, block in entries.items() for key in (block or {})}
    expected = {(k.modality, k.key) for k in keys}
    if given != expected:
        missing = sorted(f"{m}:{k}" for m, k in expected - given)
        extra = sorted(f"{m}:{k}" for m, k in given - expected)
        raise ValueError(f"query labels do not match the driver keys: missing {missing}, extra {extra}")
    labels = []
    for k in keys:
        entry = entries[k.modality][k.key]
        label = str(entry["label"]).strip()
        reason = entry.get("reason")
        if label != k.source_label and not reason:
            raise ValueError(
                f"{k.modality}:{k.key}: {label!r} differs from {k.source_label!r}; give a reason"
            )
        check_no_patient_values(label)
        labels.append(
            QueryLabel(
                modality=k.modality, key=k.key, label=label, source_label=k.source_label, reason=reason
            )
        )
    return QuerySpec(
        version=raw["version"],
        template=raw["template"],
        directions=list(raw["directions"]),
        image_model=raw["image_model"],
        labels=labels,
    )


def query_text(spec: QuerySpec, modality: str, key: str, direction: str) -> str:
    if direction not in spec.directions:
        raise ValueError(f"unknown direction {direction!r}")
    return spec.template.format(label=spec.label_for(modality, key).label, direction=direction)


def build_queries(spec: QuerySpec) -> list[QueryDef]:
    """Every query: driver-key order, then direction order."""
    return [
        QueryDef(
            query_id=f"{label.modality}:{label.key}:{direction}",
            modality=label.modality,
            key=label.key,
            direction=direction,
            text=spec.template.format(label=label.label, direction=direction),
        )
        for label in spec.labels
        for direction in spec.directions
    ]
