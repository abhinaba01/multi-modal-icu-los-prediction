"""The FaTE split is patient-level, deterministic, correctly sized and stable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fate_icu.data.splits import assign_patient_splits, fate_splits, legacy_splits, patient_hash_unit
from fate_icu.tabular._legacy import legacy


def test_hash_unit_is_pinned():
    # Changing the hash rule would silently move patients between splits.
    assert patient_hash_unit(10000001, 666) == pytest.approx(0.6732317427611995, abs=0)
    assert patient_hash_unit(10000002, 666) == pytest.approx(0.6460363031010615, abs=0)
    assert patient_hash_unit(10000001, 667) != patient_hash_unit(10000001, 666)


def test_fractions_on_many_patients(contract):
    subjects = pd.Series(np.arange(10_000_000, 10_020_000))
    labels = assign_patient_splits(subjects, contract.splits.fate)
    shares = labels.value_counts(normalize=True)
    for name, fraction in contract.splits.fate.fractions.items():
        assert shares[name] == pytest.approx(fraction, abs=0.01)


def test_all_stays_of_a_patient_share_a_split(synth_tables, contract):
    split = fate_splits(synth_tables.cohort, contract)
    per_patient = split.groupby(synth_tables.cohort["subject_id"]).nunique()
    assert (per_patient == 1).all()
    assert (synth_tables.cohort["subject_id"].value_counts() > 1).any()  # the fixture has repeat patients


def test_deterministic_and_stable_under_subsetting(synth_tables, contract):
    cohort = synth_tables.cohort
    full = fate_splits(cohort, contract)
    again = fate_splits(cohort.sample(frac=1.0, random_state=3), contract)
    pd.testing.assert_series_equal(full, again.loc[full.index])
    subset = cohort.iloc[::3]
    pd.testing.assert_series_equal(fate_splits(subset, contract), full.loc[subset.index])


def test_legacy_splits_are_the_internship_splits(contract):
    expected = legacy("evaluation.experiment").make_splits(50, 10, 0.2, 666)
    for (a_train, a_test), (b_train, b_test) in zip(legacy_splits(50, contract), expected, strict=True):
        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_test, b_test)
