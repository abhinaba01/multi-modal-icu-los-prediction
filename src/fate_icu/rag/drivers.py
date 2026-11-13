"""The evidence keys a driver can name: one retrieval query per key and direction (Stage 5).

A driver in an evidence packet (Stage 4) points at one evidence unit by modality and key:

* ``timeseries``: a channel of ``configs/variables.yaml`` (e.g. ``creatinine``);
* ``static``: a feature group of the contract's ``fate_static`` set (e.g. ``sofa_static``);
* ``image``: a pathology concept of the frozen torchxrayvision image model, or
  ``whole_image``.

Retrieval queries are built from this list and fixed labels, never from a packet's
free-text fields (``label``, ``observed``, ``time``), which can hold patient values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fate_icu.data.contract import Contract, load_contract
from fate_icu.data.cxr.preprocess import _xrv
from fate_icu.data.timeseries.variables import VariableSet, load_variables

Modality = Literal["timeseries", "static", "image"]
MODALITIES: tuple[str, ...] = ("timeseries", "static", "image")
IMAGE_MODEL_WEIGHTS = "densenet121-res224-chex"  # Stage 3's frozen image encoder
WHOLE_IMAGE = "whole_image"
STATIC_FEATURE_SET = "fate_static"


@dataclass(frozen=True)
class DriverKey:
    """One evidence key and the label its source definition gives it."""

    modality: Modality
    key: str
    source_label: str


def image_concepts(weights: str = IMAGE_MODEL_WEIGHTS) -> list[str]:
    """The non-empty pathology labels of a torchxrayvision model, in model order.

    Read from torchxrayvision's model table, so no weights are downloaded.
    """
    labels = _xrv().models.model_urls[weights]["labels"]
    return [label for label in labels if label]


def driver_keys(
    variables: VariableSet | None = None,
    contract: Contract | None = None,
    concepts: list[str] | None = None,
) -> list[DriverKey]:
    """Every key a driver can name: channels, then static groups, then image concepts.

    Args:
        variables: The time-series channels (default ``configs/variables.yaml``).
        contract: The contract whose ``fate_static`` groups are the static units.
        concepts: Image concepts (default :func:`image_concepts`).
    """
    variables = variables or load_variables()
    contract = contract or load_contract()
    concepts = image_concepts() if concepts is None else list(concepts)
    keys = [DriverKey("timeseries", v.name, v.label) for v in variables.variables]
    keys += [
        DriverKey("static", group, contract.feature_groups[group].label)
        for group in contract.feature_sets[STATIC_FEATURE_SET].groups
    ]
    keys += [DriverKey("image", concept, concept) for concept in concepts]
    keys.append(DriverKey("image", WHOLE_IMAGE, "whole chest radiograph"))
    return keys
