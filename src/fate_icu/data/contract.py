"""The frozen FaTE-ICU contract (``configs/contract.yaml``) as a validated pydantic model.

Later code imports the cohort, window, target, feature lists and splits from here
instead of hard-coding them::

    from fate_icu.data.contract import load_contract
    contract = load_contract()
    contract.window.hours                    # W
    contract.feature_names("fate_static")    # model inputs available at t0
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from fate_icu.utils.env import REPO_ROOT
from fate_icu.utils.hashing import file_sha256

DEFAULT_CONTRACT_PATH = REPO_ROOT / "configs" / "contract.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Provenance(_Strict):
    legacy_dir: str
    legacy_tag: str
    legacy_commit: str | None
    legacy_tree_sha256: str
    original_metrics_available: bool
    note: str


class MimicSpec(_Strict):
    version: str
    demo_version: str
    mimic_code_tag: dict[str, str]
    schemas: dict[Literal["hosp", "icu", "derived"], str]
    tables: dict[str, Literal["hosp", "icu", "derived"]]

    def qualified(self, table: str) -> str:
        """Schema-qualified name of a legacy table key, e.g. ``mimiciv_icu.icustays``."""
        if table not in self.tables:
            raise KeyError(f"table {table!r} is not in the contract (known: {sorted(self.tables)})")
        return f"{self.schemas[self.tables[table]]}.{table}"


class VitalRange(_Strict):
    low: float
    high: float
    inclusive: bool


class CohortSpec(_Strict):
    unit: Literal["icu_stay"]
    first_stay_only: bool
    readmission_exclusion_hours: int
    readmission_rule: str
    exclude_death_in_icu: bool
    death_rule: str
    los_days_min_exclusive: float
    los_days_max_inclusive: float
    min_anchor_age: int
    drop_duplicate_rows: bool
    complete_case_feature_set: str
    vital_mean_ranges: dict[str, VitalRange]
    stays_ending_before_t0: str
    selection_uses_post_t0_information: list[str]


class WindowDef(_Strict):
    """A time window relative to ICU ``intime`` (hours)."""

    description: str
    start_hours: float | None = None
    start_inclusive: bool | None = None
    end_hours: float | None = None  # None: after t0 / time not recorded
    end_inclusive: bool | None = None
    n_subwindows: int | None = None
    subwindow_hours: int | None = None

    def ends_by(self, hours: float) -> bool:
        """True if every event in the window is at or before ``intime + hours``."""
        return self.end_hours is not None and self.end_hours <= hours


class WindowSpec(_Strict):
    hours: int
    t0: str
    event_time_column: dict[str, str]
    storetime_checked: bool
    definitions: dict[str, WindowDef]


class LongStaySpec(_Strict):
    rule: str
    method: Literal["train_q3", "fixed"]
    quantile: float = Field(gt=0, lt=1)
    threshold_days: float | None = Field(default=None, gt=0)
    frozen_from: str | None
    legacy_method: str


class TargetSpec(_Strict):
    kind: Literal["total_icu_los"]
    source: str
    unit: Literal["days"]
    regression_rounding_dp: int
    long_stay: LongStaySpec


class FeatureGroup(_Strict):
    label: str
    available_at_t0: bool
    description: str


class Feature(_Strict):
    name: str
    label: str
    group: str
    kind: Literal["numeric", "categorical", "binary"]
    source: str
    window: str
    aggregation: str


class FeatureSet(_Strict):
    description: str
    groups: list[str]
    t0_only: bool


class PatientHashSplit(_Strict):
    method: Literal["patient_hash"]
    group_column: str
    seed: int
    fractions: dict[Literal["train", "val", "test"], float]
    rule: str

    @model_validator(mode="after")
    def _fractions(self) -> PatientHashSplit:
        if set(self.fractions) != {"train", "val", "test"}:
            raise ValueError("fractions must define train, val and test")
        if any(v <= 0 for v in self.fractions.values()):
            raise ValueError("every split fraction must be positive")
        if abs(sum(self.fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


class LegacySplit(_Strict):
    method: Literal["repeated_random_stay_level"]
    n_repeats: int
    test_size: float
    seed: int
    patient_level: Literal[False]


class Splits(_Strict):
    fate: PatientHashSplit
    legacy: LegacySplit


class B0Spec(_Strict):
    prior: Literal["XGBoost", "Random forest"]
    xgb_n_estimators: int
    rf_n_estimators: int
    random_state: int
    stage1_threshold: Literal["gmean_on_train_scores"]
    stage2: Literal["regress_on_predicted_short_train_stays"]
    hard_label_threshold: float


class Tolerance(_Strict):
    counts: int
    q3_days: float
    long_prevalence: float


class ReproductionTargets(_Strict):
    mimic_version: str
    cohort_flow: dict[str, int]
    whole_cohort_q3_days: float
    long_prevalence: float
    tolerance: Tolerance


class Contract(_Strict):
    contract_version: int
    provenance: Provenance
    mimic: MimicSpec
    cohort: CohortSpec
    window: WindowSpec
    target: TargetSpec
    feature_groups: dict[str, FeatureGroup]
    features: list[Feature]
    feature_sets: dict[str, FeatureSet]
    splits: Splits
    b0: B0Spec
    reproduction_targets: ReproductionTargets

    _sha256: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _consistent(self) -> Contract:
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("feature names must be unique")
        labels = [f.label for f in self.features]
        if len(labels) != len(set(labels)):
            raise ValueError("feature labels must be unique")
        for feature in self.features:
            if feature.group not in self.feature_groups:
                raise ValueError(f"feature {feature.name}: unknown group {feature.group!r}")
            if feature.window not in self.window.definitions:
                raise ValueError(f"feature {feature.name}: unknown window {feature.window!r}")
            window = self.window.definitions[feature.window]
            if self.feature_groups[feature.group].available_at_t0 and not window.ends_by(self.window.hours):
                raise ValueError(
                    f"feature {feature.name} is in a group marked available at t0 but its window "
                    f"{feature.window!r} ends after t0"
                )
        for group in self.feature_groups:
            if not any(f.group == group for f in self.features):
                raise ValueError(f"feature group {group!r} has no features")
        for set_name, fset in self.feature_sets.items():
            for group in fset.groups:
                if group not in self.feature_groups:
                    raise ValueError(f"feature set {set_name}: unknown group {group!r}")
                if fset.t0_only and not self.feature_groups[group].available_at_t0:
                    raise ValueError(f"feature set {set_name} is t0_only but includes {group!r}")
        if self.cohort.complete_case_feature_set not in self.feature_sets:
            raise ValueError("cohort.complete_case_feature_set must name a feature set")
        if self.target.long_stay.method == "fixed" and self.target.long_stay.threshold_days is None:
            raise ValueError("a fixed long-stay threshold needs threshold_days")
        if self.cohort.los_days_min_exclusive * 24 < self.window.hours:
            raise ValueError("the cohort would include stays that end before t0")
        return self

    # --- convenience ------------------------------------------------------------------

    @property
    def sha256(self) -> str | None:
        """LF-normalised SHA-256 of the YAML file this contract was loaded from."""
        return self._sha256

    def feature(self, name: str) -> Feature:
        for feature in self.features:
            if feature.name == name:
                return feature
        raise KeyError(f"unknown feature {name!r}")

    def feature_names(self, feature_set: str) -> list[str]:
        """Feature names of a named set, in contract (= legacy column) order."""
        if feature_set not in self.feature_sets:
            raise KeyError(f"unknown feature set {feature_set!r} (known: {sorted(self.feature_sets)})")
        groups = set(self.feature_sets[feature_set].groups)
        return [f.name for f in self.features if f.group in groups]

    def available_at_t0(self, name: str) -> bool:
        return self.feature_groups[self.feature(name).group].available_at_t0

    def group_of(self, names: list[str]) -> dict[str, str]:
        """Map each feature name to its group."""
        return {name: self.feature(name).group for name in names}

    def categorical(self, names: list[str]) -> list[str]:
        return [n for n in names if self.feature(n).kind == "categorical"]

    def schema(self, key: Literal["hosp", "icu", "derived"]) -> str:
        return self.mimic.schemas[key]

    def t0(self, intime: pd.Series | pd.Timestamp) -> pd.Series | pd.Timestamp:
        """Prediction time t0 = ICU intime + W."""
        return pd.to_datetime(intime) + pd.Timedelta(hours=self.window.hours)


def _load(path: Path) -> Contract:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    contract = Contract.model_validate(raw)
    contract._sha256 = file_sha256(path)
    return contract


@lru_cache(maxsize=8)
def _load_cached(path: str) -> Contract:
    return _load(Path(path))


def load_contract(path: str | Path | None = None) -> Contract:
    """Load and validate the contract (cached per path).

    Args:
        path: Contract YAML; defaults to ``configs/contract.yaml``.

    Returns:
        The validated, immutable contract.
    """
    return _load_cached(str(Path(path or DEFAULT_CONTRACT_PATH).resolve()))
