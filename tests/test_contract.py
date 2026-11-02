"""The contract matches the legacy code it freezes, and its validators bite."""

from __future__ import annotations

import copy
import re

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from fate_icu.data.contract import DEFAULT_CONTRACT_PATH, Contract
from fate_icu.tabular._legacy import LEGACY_ROOT, check_legacy_config_matches_contract, legacy


@pytest.fixture()
def raw_contract():
    with DEFAULT_CONTRACT_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_contract_loads_with_hash(contract):
    assert contract.contract_version == 1
    assert contract.sha256 and len(contract.sha256) == 64
    assert contract.window.hours == 24
    assert contract.provenance.legacy_commit is None
    assert contract.provenance.original_metrics_available is False


def test_feature_sets_match_the_legacy_tables(contract):
    bd = legacy("features.build_dataset")
    sofa_static = legacy("features.sofa_static")
    dyn = legacy("features.sofa_dynamics")
    dynamics = dyn.dynamics_feature_columns()
    assert contract.feature_names("legacy_hempel") == bd.BASE_FEATURES
    assert contract.feature_names("legacy_hempel_sofa") == bd.BASE_FEATURES + sofa_static.SOFA_STATIC_COLUMNS
    assert contract.feature_names("legacy_hempel_sofa_dynamics") == (
        bd.BASE_FEATURES + dynamics["continuous"] + dynamics["indicator"] + dyn.interaction_feature_columns()
    )
    assert len(contract.features) == 79


def test_categorical_features_match_legacy(contract):
    bd = legacy("features.build_dataset")
    every = [f.name for f in contract.features]
    assert contract.categorical(every) == bd.BASE_CATEGORICAL


def test_itemids_in_feature_sources_match_legacy(contract):
    vitals = legacy("features.vitals").VITAL_ITEMIDS
    labs = legacy("features.labs").LAB_ITEMIDS
    for itemid_map, group in ((vitals, "vitals"), (labs, "labs")):
        by_name = {name: itemid for itemid, name in itemid_map.items()}
        features = [f for f in contract.features if f.group == group]
        assert {f.name for f in features} == set(by_name)
        for feature in features:
            assert int(re.search(r"itemid (\d+)", feature.source).group(1)) == by_name[feature.name]


def test_legacy_config_matches_contract(contract):
    cfg = legacy("config").load_config(LEGACY_ROOT / "config.yaml")
    check_legacy_config_matches_contract(cfg, contract)
    assert contract.splits.legacy.seed == cfg["project"]["random_state"]


def test_diagnoses_are_flagged_and_kept_out_of_fate_inputs(contract):
    assert contract.feature_groups["diagnoses"].available_at_t0 is False
    assert not contract.available_at_t0("Diagnosis_1")
    fate = contract.feature_names("fate_static")
    assert "Diagnosis_1" not in fate and "Diagnosis_2" not in fate
    assert "Diagnosis_1" in contract.feature_names("legacy_hempel")


def test_fate_static_uses_only_features_available_at_t0(contract):
    for name in contract.feature_names("fate_static"):
        feature = contract.feature(name)
        assert contract.available_at_t0(name)
        assert contract.window.definitions[feature.window].ends_by(contract.window.hours), name
    assert len(contract.feature_names("fate_static")) == 77


def test_threshold_not_yet_frozen(contract):
    assert contract.target.long_stay.method == "train_q3"
    assert contract.target.long_stay.threshold_days is None


def test_t0_and_qualified_names(contract):
    assert contract.t0(pd.Timestamp("2150-01-01 08:00")) == pd.Timestamp("2150-01-02 08:00")
    assert contract.mimic.qualified("icustays") == "mimiciv_icu.icustays"
    assert contract.mimic.qualified("labevents") == "mimiciv_hosp.labevents"
    with pytest.raises(KeyError):
        contract.mimic.qualified("notes")


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["splits"]["fate"]["fractions"].update(test=0.3), "sum to 1"),
        (lambda c: c["feature_sets"]["fate_static"]["groups"].append("diagnoses"), "t0_only"),
        (lambda c: c["feature_groups"]["diagnoses"].update(available_at_t0=True), "ends after t0"),
        (lambda c: c["features"][0].update(group="nope"), "unknown group"),
        (lambda c: c["features"][1].update(label=c["features"][0]["label"]), "labels must be unique"),
        (lambda c: c["cohort"].update(los_days_min_exclusive=0.5), "end before t0"),
        (lambda c: c.update(surprise=1), "Extra inputs"),
    ],
)
def test_validators_reject_bad_contracts(raw_contract, mutate, message):
    bad = copy.deepcopy(raw_contract)
    mutate(bad)
    with pytest.raises(ValidationError, match=message):
        Contract.model_validate(bad)
