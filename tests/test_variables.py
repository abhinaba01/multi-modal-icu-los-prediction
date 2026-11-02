"""configs/variables.yaml: complete, consistent, and faithful to the internship's item mappings."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from fate_icu.data.timeseries.variables import DEFAULT_VARIABLES_PATH, ValidRange, VariableSet
from fate_icu.tabular._legacy import legacy


@pytest.fixture()
def raw_variables():
    with DEFAULT_VARIABLES_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _legacy_itemids() -> dict[str, set[int]]:
    """Stage 2 variable -> itemids the internship used for the same variable."""
    vitals = legacy("features.vitals").VITAL_ITEMIDS
    labs = legacy("features.labs").LAB_ITEMIDS
    sofa = legacy("features.sofa_static")

    def by_name(mapping, name):
        return {itemid for itemid, column in mapping.items() if column == name}

    return {
        "heart_rate": by_name(vitals, "heart_rate"),
        "resp_rate": by_name(vitals, "respiration"),
        "spo2": by_name(vitals, "saturation"),
        "temperature": by_name(vitals, "temperature"),
        "gcs": set(sofa.CHART_GCS),
        "mbp": set(sofa.CHART_MAP),
        "fio2": {sofa.CHART_FIO2},
        "pao2": {sofa.LAB_PAO2},
        "bilirubin": {sofa.LAB_BILIRUBIN},
        "creatinine": by_name(labs, "creatinine"),
        "bun": by_name(labs, "urea_nitrogen"),
        "glucose": by_name(labs, "glucose"),
        "sodium": by_name(labs, "sodium"),
        "potassium": by_name(labs, "potassium"),
        "haemoglobin": by_name(labs, "hemoglobin"),
        "platelets": by_name(labs, "platelet_count"),
        "wbc": by_name(labs, "white_blood_cells"),
        "urine_output": set(sofa.URINE_ITEMIDS),
    }


def test_the_stage2_channel_set(variables):
    assert variables.sha256 and len(variables.sha256) == 64
    assert len(variables.variables) == 30
    groups = [v.group for v in variables.variables]
    assert groups.count("vitals") == 8 and groups.count("respiratory") == 4
    assert groups.count("labs") == 10 and groups.count("sofa") == 6
    assert variables.by_source("norepinephrine_equivalent_dose")[0].impute == "zero"
    assert {v.column for v in variables.by_source("sofa")} == {
        "respiration",
        "coagulation",
        "liver",
        "cardiovascular",
        "cns",
        "renal",
    }


def test_internship_mappings_are_reused_exactly(variables):
    expected = _legacy_itemids()
    internship = {v.name for v in variables.variables if v.mapping == "internship"}
    assert internship == set(expected)
    for name, itemids in expected.items():
        assert set(variables.get(name).itemids) == itemids, name


def test_every_channel_is_documented(variables):
    for v in variables.variables:
        assert v.label and v.unit and v.reference_range.text and v.synonyms, v.name
    log1p = {v.name for v in variables.variables if v.transform == "log1p"}
    assert log1p == {"creatinine", "bun", "bilirubin", "wbc", "lactate", "glucose", "urine_output", "ne_dose"}


def test_internship_gcs_convention(variables):
    gcs = variables.get("gcs")
    assert gcs.combine == "sum_cocharted" and gcs.hourly == "min"
    assert gcs.valid.contains(np.array([3, 15, 2, 16])).tolist() == [True, True, False, False]


def test_valid_range_semantics():
    rng = ValidRange(gt=0, le=100)
    assert rng.contains(np.array([0.0, 0.1, 100.0, 100.1, np.nan])).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]
    with pytest.raises(ValidationError):
        ValidRange(gt=0, ge=0)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda v: v["variables"][1]["itemids"].append(220045), "itemid 220045"),
        (lambda v: v["variables"][1]["synonyms"].append("pulse"), "synonym 'pulse'"),
        (lambda v: v["variables"][1].update(label="heart rate"), "labels must be unique"),
        (lambda v: v["variables"][22].pop("floor"), "log1p needs values >= 0"),
        (lambda v: v["variables"][24].pop("column"), "needs its component column"),
        (lambda v: v["variables"][0].update(itemids=[]), "needs itemids"),
        (lambda v: v["variables"][0].update(mapping="new"), "Input should be"),
    ],
)
def test_validators_reject_bad_variables(raw_variables, mutate, message):
    bad = copy.deepcopy(raw_variables)
    mutate(bad)
    with pytest.raises(ValidationError, match=message):
        VariableSet.model_validate(bad)
