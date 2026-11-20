"""The demo model: a small linear model with exact reasons, and a validated explanation sentence.

For a patient entered by hand (only some of the fields are filled in; the rest are "not recorded")
it returns the prediction, the strongest reasons and the explanation text. The model is a ridge-
regularised logistic regression for the long-stay question and a ridge regression for log(LOS),
fitted on the train split of the open MIMIC-IV demo (a few dozen stays), so every value the user
types can contribute. Its explanation is exact: a feature's effect on the long-stay log-odds is its
coefficient times how far the entered value is from the training average (a value left out
contributes nothing). Effects are summed per contract feature group, the unit Stage 4 uses for
static evidence, and become an evidence packet; the text is the Stage 6 template, checked by the
Stage 6 contract.

This is a stand-in for the real tabular explainer (TreeSHAP over the internship's XGBoost, which
needs thousands of stays to use many features). Nothing here is trained on credentialed data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fate_icu.attribution.packet import EvidencePacket, format_number
from fate_icu.data.contract import Contract
from fate_icu.data.targets import resolve_long_stay_threshold
from fate_icu.llm.template import explain
from fate_icu.llm.validate import check, split_sentences
from fate_icu.tabular.api import TabularTables, load_splits

MIN_FIELDS = 3
MAX_DRIVERS = 5
SMALL_EFFECT = 0.02  # log-odds: below this a reason is noise, unless nothing else is left
VITALS = "Vital signs (mean of the first 24 hours)"
LABS = "Laboratory results (mean of the first 24 hours)"
SOFA = "Organ function (SOFA, worst of the first 24 hours)"


@dataclass(frozen=True)
class FormField:
    """One form field: the contract feature it fills, how it is entered and what is plausible."""

    name: str
    section: str
    kind: str  # number | choice
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    example: float | str | None = None  # an example value for the placeholder


FIELDS: tuple[FormField, ...] = (
    FormField("anchor_age", "Patient", "number", 18, 110, 1, 65),
    FormField("gender", "Patient", "choice"),
    FormField("first_careunit", "Patient", "choice"),
    FormField("admission_type", "Patient", "choice"),
    FormField("admission_location", "Patient", "choice"),
    FormField("heart_rate", VITALS, "number", 20, 250, 1, 88),
    FormField("respiration", VITALS, "number", 4, 60, 1, 19),
    FormField("saturation", VITALS, "number", 50, 100, 0.5, 97),
    FormField("temperature", VITALS, "number", 85, 110, 0.1, 98.4),
    FormField("gcs_eye", VITALS, "number", 1, 4, 0.25, 4),
    FormField("gcs_verbal", VITALS, "number", 1, 5, 0.25, 5),
    FormField("gcs_motor", VITALS, "number", 1, 6, 0.25, 6),
    FormField("creatinine", LABS, "number", 0.1, 20, 0.1, 0.9),
    FormField("urea_nitrogen", LABS, "number", 1, 250, 1, 17),
    FormField("sodium", LABS, "number", 100, 180, 1, 138),
    FormField("potassium", LABS, "number", 1.5, 9, 0.1, 4),
    FormField("glucose", LABS, "number", 20, 1000, 1, 130),
    FormField("bicarbonate", LABS, "number", 5, 50, 1, 24),
    FormField("hemoglobin", LABS, "number", 3, 22, 0.1, 11),
    FormField("white_blood_cells", LABS, "number", 0.1, 100, 0.1, 10),
    FormField("platelet_count", LABS, "number", 5, 1500, 1, 220),
    FormField("sofa_total", SOFA, "number", 0, 24, 1, 4),
    FormField("sofa_resp", SOFA, "number", 0, 4, 1, 0),
    FormField("sofa_cardio", SOFA, "number", 0, 4, 1, 0),
    FormField("sofa_renal", SOFA, "number", 0, 4, 1, 0),
    FormField("urine_24h", SOFA, "number", 0, 15000, 50, 1800),
)
FIELD_BY_NAME = {f.name: f for f in FIELDS}


@dataclass
class DemoModel:
    contract: Contract
    categories: dict[str, list[str]]  # the options of each choice field, as seen in training
    typical: dict[str, float]  # the training median of each number field
    mean: dict[str, float]  # training mean of each number field, and of each choice level ("name=level")
    scale: dict[str, float]  # training standard deviation of each number field
    classifier: Any  # sklearn LogisticRegression on the design vector
    regressor: Any  # sklearn Ridge on log1p(LOS days)
    columns: list[str]  # design vector order: number fields, then "name=level"
    threshold_days: float
    n_train: int
    labels: dict[str, str] = field(default_factory=dict)  # contract label of each feature

    def options(self) -> list[dict[str, Any]]:
        """The form description the page is built from."""
        out = []
        for f in FIELDS:
            item: dict[str, Any] = {
                "name": f.name, "section": f.section, "kind": f.kind, "label": self.labels[f.name],
            }  # fmt: skip
            if f.kind == "choice":
                item["options"] = self.categories[f.name]
            else:
                item |= {"min": f.minimum, "max": f.maximum, "step": f.step, "example": f.example}
                item["typical"] = self.typical.get(f.name)
            out.append(item)
        return out

    def design(self, values: dict[str, Any]) -> np.ndarray:
        """The model's input for one patient. A value that was not entered is the training average, so it adds
        nothing to the prediction or to the reasons."""
        row = []
        for column in self.columns:
            if "=" in column:
                name, level = column.split("=", 1)
                given = values.get(name)
                row.append(self.mean[column] if given is None else float(given == level))
            else:
                given = values.get(column)
                row.append(0.0 if given is None else (float(given) - self.mean[column]) / self.scale[column])
        return np.asarray(row, dtype=float)

    def centre(self) -> np.ndarray:
        """The design vector of an average patient: 0 per z-scored number, the level frequency per choice."""
        return np.asarray([self.mean[c] if "=" in c else 0.0 for c in self.columns], dtype=float)

    def effects(self, values: dict[str, Any]) -> dict[str, float]:
        """Each entered feature's effect on the long-stay log-odds: coefficient x (value - average)."""
        parts = np.asarray(self.classifier.coef_[0], dtype=float) * (self.design(values) - self.centre())
        out: dict[str, float] = {}
        for column, part in zip(self.columns, parts, strict=True):
            name = column.split("=", 1)[0]
            if name in values:
                out[name] = out.get(name, 0.0) + float(part)
        return out


def _design_matrix(
    frame: pd.DataFrame, columns: list[str], mean: dict[str, float], scale: dict[str, float]
) -> np.ndarray:
    rows = []
    for _, record in frame.iterrows():
        row = []
        for column in columns:
            if "=" in column:
                name, level = column.split("=", 1)
                given = record[name]
                row.append(mean[column] if pd.isna(given) else float(str(given) == level))
            else:
                given = record[column]
                row.append(0.0 if pd.isna(given) else (float(given) - mean[column]) / scale[column])
        rows.append(row)
    return np.asarray(rows, dtype=float)


def train_demo_model(
    tables: TabularTables, contract: Contract, legacy_cfg: dict[str, Any] | None = None
) -> DemoModel:
    """Fit the demo model on the train split of ``tables`` (the open demo in real use).

    ``legacy_cfg`` is unused; it keeps the call shape of the tabular trainers.
    """
    from sklearn.linear_model import LogisticRegression, Ridge

    split = load_splits(tables.cohort, contract)
    threshold, _source = resolve_long_stay_threshold(tables.los, split, contract, require_frozen=False)
    names = [f.name for f in FIELDS]
    train_rows = (split == "train").reindex(tables.features.index).fillna(False).to_numpy()
    train = tables.features.loc[train_rows, names]
    los = tables.los.loc[train.index].astype(float)
    y = (los.round(contract.target.regression_rounding_dp) > threshold).astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        raise ValueError("the train stays need both short and long stays")

    categories = {
        f.name: sorted(train[f.name].dropna().astype(str).unique()) for f in FIELDS if f.kind == "choice"
    }
    numbers = [f.name for f in FIELDS if f.kind == "number"]
    mean: dict[str, float] = {}
    scale: dict[str, float] = {}
    typical: dict[str, float] = {}
    for name in numbers:
        col = pd.to_numeric(train[name], errors="coerce")
        mean[name] = float(col.mean()) if col.notna().any() else 0.0
        sd = float(col.std(ddof=0)) if col.notna().sum() > 1 else 1.0
        scale[name] = sd if sd > 1e-9 else 1.0
        typical[name] = float(col.median()) if col.notna().any() else mean[name]
    columns = list(numbers)
    for name, levels in categories.items():
        for level in levels:
            key = f"{name}={level}"
            columns.append(key)
            mean[key] = float((train[name].astype(str) == level).mean())
    X = _design_matrix(train, columns, mean, scale)
    classifier = LogisticRegression(C=0.2, max_iter=2000).fit(X, y)
    regressor = Ridge(alpha=5.0).fit(X, np.log1p(los.to_numpy()))
    labels = {f.name: contract.feature(f.name).label for f in FIELDS}
    return DemoModel(
        contract, categories, typical, mean, scale, classifier, regressor, columns,
        float(threshold), int(len(train)), labels,
    )  # fmt: skip


def parse_inputs(demo: DemoModel, raw: dict[str, Any]) -> dict[str, Any]:
    """Check what was typed: known fields, plausible numbers, listed choices. Blank is not recorded."""
    clean: dict[str, Any] = {}
    unknown = sorted(set(raw) - set(FIELD_BY_NAME))
    if unknown:
        raise ValueError(f"unknown field(s): {', '.join(unknown)}")
    for name, value in raw.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        f = FIELD_BY_NAME[name]
        label = demo.labels[name]
        if f.kind == "choice":
            if str(value) not in demo.categories[name]:
                raise ValueError(f"{label}: choose one of the listed options")
            clean[name] = str(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label}: enter a number") from error
        if not math.isfinite(number) or not f.minimum <= number <= f.maximum:
            raise ValueError(f"{label}: enter a value between {f.minimum:g} and {f.maximum:g}")
        clean[name] = number
    if len(clean) < MIN_FIELDS:
        raise ValueError(f"enter at least {MIN_FIELDS} values so there is something to explain")
    return clean


def _shown(f: FormField, value: Any) -> str:
    return str(value) if f.kind == "choice" else format_number(float(value))


def _observed(demo: DemoModel, names: list[str], values: dict[str, Any], effects: dict[str, float]) -> str:
    """The user's own values for the features of one reason, strongest first (at most two)."""
    parts = []
    for name in sorted(names, key=lambda n: -abs(effects[n]))[:2]:
        f = FIELD_BY_NAME[name]
        text = f"{demo.labels[name]} {_shown(f, values[name])}"
        typical = demo.typical.get(name)
        if f.kind == "number" and typical is not None and abs(float(values[name]) - typical) > 1e-9:
            text += f" (typical {format_number(typical)})"
        parts.append(text)
    return "; ".join(parts)


def explain_patient(demo: DemoModel, raw: dict[str, Any]) -> dict[str, Any]:
    """Prediction, reasons and the explanation text for one patient entered by hand."""
    values = parse_inputs(demo, raw)
    x = demo.design(values)[None, :]
    p_long = float(np.clip(demo.classifier.predict_proba(x)[0, 1], 0.0, 1.0))
    days = float(max(np.expm1(demo.regressor.predict(x)[0]), 0.5))
    if p_long >= 0.5:  # a stay the model calls long is at least as long as the threshold
        days = max(days, demo.threshold_days)
    effects = demo.effects(values)
    contract = demo.contract
    by_group: dict[str, list[str]] = {}
    for name in values:
        by_group.setdefault(contract.feature(name).group, []).append(name)
    group_effect = {g: float(sum(effects[n] for n in names)) for g, names in by_group.items()}
    ranked = sorted(group_effect, key=lambda g: -abs(group_effect[g]))
    kept = [g for g in ranked if abs(group_effect[g]) >= SMALL_EFFECT][:MAX_DRIVERS] or ranked[:1]
    drivers = [
        {
            "id": f"D{i}",
            "modality": "static",
            "key": g,
            "label": contract.feature_groups[g].label,
            "observed": _observed(demo, by_group[g], values, effects),
            "direction": "longer" if group_effect[g] > 0 else "shorter",
            "effect": round(group_effect[g], 4) or (SMALL_EFFECT if group_effect[g] >= 0 else -SMALL_EFFECT),
        }
        for i, g in enumerate(kept, start=1)
    ]
    packet = EvidencePacket.model_validate(
        {
            "note": "Demo packet built from entered values; not patient data.",
            "prediction": {
                "p_long_stay": round(p_long, 4),
                "predicted_los_days": round(days, 2),
                "long_stay_threshold_days": round(demo.threshold_days, 2),
                "window_hours": contract.window.hours,
            },
            "drivers": drivers,
        }
    )
    text = explain(packet)
    verdict = check(text, packet)
    top = sorted(values, key=lambda n: -abs(effects[n]))[:8]
    return {
        "prediction": packet.prediction.model_dump(),
        "drivers": [d.model_dump(exclude_none=True) for d in packet.drivers],
        "text": text,
        "sentences": split_sentences(text),
        "valid": verdict.ok,
        "failures": list(verdict.failures),
        "feature_effects": [
            {
                "label": demo.labels[n],
                "value": _shown(FIELD_BY_NAME[n], values[n]),
                "effect": round(effects[n], 4),
            }
            for n in top
        ],
        "entered": len(values),
        "not_recorded": len(FIELDS) - len(values),
        "trained_on": demo.n_train,
    }


def save(demo: DemoModel, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(demo, path)


def load(path: Path) -> DemoModel:
    import joblib

    return joblib.load(path)
