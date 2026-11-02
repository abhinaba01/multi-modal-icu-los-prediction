"""The hourly time-series variables (``configs/variables.yaml``) as a validated model.

One :class:`Variable` per tensor channel, in channel order. Later stages import it for
labels, units, reference ranges and synonyms instead of hard-coding them::

    from fate_icu.data.timeseries.variables import load_variables
    variables = load_variables()
    variables.names                   # channel order
    variables.get("creatinine").reference_range.text
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from fate_icu.utils.env import REPO_ROOT
from fate_icu.utils.hashing import file_sha256

DEFAULT_VARIABLES_PATH = REPO_ROOT / "configs" / "variables.yaml"

Source = Literal["chartevents", "labevents", "outputevents", "norepinephrine_equivalent_dose", "sofa"]
# Raw MIMIC tables: rows carry itemids and a storetime (the availability rule applies).
EVENT_SOURCES = ("chartevents", "labevents", "outputevents")
# mimic-code derived concepts: no storetime.
DERIVED_SOURCES = ("norepinephrine_equivalent_dose", "sofa")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ValidRange(_Strict):
    """Plausibility limits; a value outside them is dropped before hourly aggregation."""

    gt: float | None = None
    ge: float | None = None
    lt: float | None = None
    le: float | None = None

    @model_validator(mode="after")
    def _one_bound_per_side(self) -> ValidRange:
        if self.gt is not None and self.ge is not None:
            raise ValueError("use gt or ge, not both")
        if self.lt is not None and self.le is not None:
            raise ValueError("use lt or le, not both")
        return self

    @property
    def lower(self) -> float | None:
        return self.gt if self.gt is not None else self.ge

    def contains(self, values: np.ndarray) -> np.ndarray:
        """Boolean mask of finite values inside the limits."""
        values = np.asarray(values, dtype=float)
        ok = np.isfinite(values)
        if self.gt is not None:
            ok &= values > self.gt
        if self.ge is not None:
            ok &= values >= self.ge
        if self.lt is not None:
            ok &= values < self.lt
        if self.le is not None:
            ok &= values <= self.le
        return ok


class ReferenceRange(_Strict):
    low: float | None = None
    high: float | None = None
    text: str

    @model_validator(mode="after")
    def _ordered(self) -> ReferenceRange:
        if self.low is not None and self.high is not None and self.low > self.high:
            raise ValueError("reference range low must not exceed high")
        return self


class Variable(_Strict):
    name: str
    label: str
    group: Literal["vitals", "respiratory", "labs", "output", "medication", "sofa"]
    unit: str
    source: Source
    itemids: list[int] = Field(default_factory=list)
    column: str | None = None
    combine: Literal["none", "sum_cocharted"] = "none"
    negate_itemids: list[int] = Field(default_factory=list)
    fraction_to_percent: bool = False
    hourly: Literal["mean", "min", "max", "sum"]
    floor: float | None = None
    transform: Literal["none", "log1p"]
    impute: Literal["median", "zero"] = "median"
    valid: ValidRange
    reference_range: ReferenceRange
    synonyms: list[str] = Field(min_length=1)
    mapping: Literal["internship", "mimic-code"]
    note: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Variable:
        if self.source in EVENT_SOURCES and not self.itemids:
            raise ValueError(f"{self.name}: a {self.source} variable needs itemids")
        if self.source in DERIVED_SOURCES and self.itemids:
            raise ValueError(f"{self.name}: derived sources take no itemids")
        if self.source == "sofa" and not self.column:
            raise ValueError(f"{self.name}: a sofa variable needs its component column")
        if self.source != "sofa" and self.column:
            raise ValueError(f"{self.name}: only sofa variables take a column")
        if len(set(self.itemids)) != len(self.itemids):
            raise ValueError(f"{self.name}: duplicate itemids")
        if not set(self.negate_itemids) <= set(self.itemids):
            raise ValueError(f"{self.name}: negate_itemids must be among its itemids")
        if self.combine == "sum_cocharted" and self.source != "chartevents":
            raise ValueError(f"{self.name}: sum_cocharted needs chartevents")
        if self.transform == "log1p":
            floored = self.floor is not None and self.floor >= 0
            lower = self.valid.lower
            if not floored and (self.negate_itemids or lower is None or lower < 0):
                raise ValueError(f"{self.name}: log1p needs values >= 0 (set valid >= 0 or a floor)")
        return self


class VariableSet(_Strict):
    version: int
    window_rule: str
    variables: list[Variable]

    _sha256: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _unique(self) -> VariableSet:
        names = [v.name for v in self.variables]
        if len(names) != len(set(names)):
            raise ValueError("variable names must be unique")
        labels = [v.label.casefold() for v in self.variables]
        if len(labels) != len(set(labels)):
            raise ValueError("variable labels must be unique")
        owner: dict[str, str] = {}
        for variable in self.variables:
            for synonym in variable.synonyms:
                key = synonym.casefold()
                if key in owner and owner[key] != variable.name:
                    raise ValueError(f"synonym {synonym!r} is used by {owner[key]} and {variable.name}")
                owner[key] = variable.name
        for source in EVENT_SOURCES:
            seen: dict[int, str] = {}
            for variable in self.by_source(source):
                for itemid in variable.itemids:
                    if itemid in seen:
                        raise ValueError(f"{source} itemid {itemid} is in {seen[itemid]} and {variable.name}")
                    seen[itemid] = variable.name
        columns = [v.column for v in self.by_source("sofa")]
        if len(columns) != len(set(columns)):
            raise ValueError("each sofa component may appear once")
        return self

    @property
    def sha256(self) -> str | None:
        """LF-normalised SHA-256 of the YAML file this set was loaded from."""
        return self._sha256

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def get(self, name: str) -> Variable:
        for variable in self.variables:
            if variable.name == name:
                return variable
        raise KeyError(f"unknown variable {name!r}")

    def index(self, name: str) -> int:
        return self.names.index(name)

    def by_source(self, source: str) -> list[Variable]:
        return [v for v in self.variables if v.source == source]

    def itemids(self, source: str) -> list[int]:
        """Sorted itemids read from one raw table."""
        return sorted({i for v in self.by_source(source) for i in v.itemids})

    def itemid_map(self, source: str) -> dict[int, str]:
        """``itemid -> variable name`` for one raw table."""
        return {i: v.name for v in self.by_source(source) for i in v.itemids}


def _load(path: Path) -> VariableSet:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    variables = VariableSet.model_validate(raw)
    variables._sha256 = file_sha256(path)
    return variables


@lru_cache(maxsize=8)
def _load_cached(path: str) -> VariableSet:
    return _load(Path(path))


def load_variables(path: str | Path | None = None) -> VariableSet:
    """Load and validate the variables (cached per path); defaults to ``configs/variables.yaml``."""
    return _load_cached(str(Path(path or DEFAULT_VARIABLES_PATH).resolve()))
