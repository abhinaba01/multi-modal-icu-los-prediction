"""The synthetic generator: real schemas, determinism, and planted cohort answers."""

from __future__ import annotations

import pandas as pd

from fate_icu.data.synthetic import SCHEMAS, generate_synthetic_mimic
from fate_icu.tabular._legacy import legacy, legacy_config
from fate_icu.tabular.loaders import FrameLoader


def test_tables_have_the_mimic_columns(synth):
    for key, frame in synth.tables.items():
        assert list(frame.columns) == SCHEMAS[key], key


def test_generation_is_deterministic():
    a = generate_synthetic_mimic(n_patients=5, seed=7)
    b = generate_synthetic_mimic(n_patients=5, seed=7)
    for key in a.tables:
        pd.testing.assert_frame_equal(a.tables[key], b.tables[key])
    assert a.roles == b.roles


def test_legacy_build_cohort_finds_the_planted_answers(synth, synth_dir, contract):
    cfg = legacy_config(contract, gem_path=synth.gem_path(synth_dir))
    cohort = legacy("features.cohort").build_cohort(FrameLoader(synth.tables), cfg)
    assert set(cohort["stay_id"]) == synth.expected_base_stays()


def test_final_cohort_applies_every_contract_rule(synth, synth_tables):
    final = set(synth_tables.cohort.index)
    assert final == synth.expected_final_stays()
    for role in (
        "missing_lab",
        "vital_out_of_range",
        "readmit_within_48h",
        "death_in_icu",
        "los_exactly_1",
        "los_above_21",
        "minor",
    ):
        assert synth.roles[role] not in final, role
    flow = synth_tables.flow
    assert flow["icustays"] == len(synth.tables["icustays"])
    assert flow["after_readmission_48h"] == flow["icustays"] - 1
    assert flow["after_death_in_icu"] == flow["after_readmission_48h"] - 1
    assert flow["after_los_1_21"] == flow["after_death_in_icu"] - 3
    assert flow["after_age"] == flow["after_los_1_21"] - 1
    assert flow["final"] == flow["after_age"] - 2 == len(final)


def test_every_final_stay_ends_after_t0(synth_tables):
    cohort = synth_tables.cohort
    assert (cohort["outtime"] > cohort["t0"]).all()
