"""Tests for cohort helper logic (48h-readmission exclusion)."""

from __future__ import annotations

import pandas as pd

from src.features.cohort import _non_readmission_stays


def test_non_readmission_excludes_within_48h_same_hadm():
    base = pd.Timestamp("2020-01-01")
    icustays = pd.DataFrame([
        # hadm 100: stay 1 discharged day1; stay 2 readmitted +0.5d (< 48h) -> EXCLUDE;
        #           stay 3 readmitted +3d after stay 2's discharge (>= 48h) -> keep.
        {"hadm_id": 100, "stay_id": 1, "intime": base, "outtime": base + pd.Timedelta(days=1)},
        {"hadm_id": 100, "stay_id": 2, "intime": base + pd.Timedelta(days=1.5), "outtime": base + pd.Timedelta(days=2)},
        {"hadm_id": 100, "stay_id": 3, "intime": base + pd.Timedelta(days=5), "outtime": base + pd.Timedelta(days=6)},
        # hadm 200: single stay -> always kept.
        {"hadm_id": 200, "stay_id": 4, "intime": base, "outtime": base + pd.Timedelta(days=1)},
    ])
    keep = _non_readmission_stays(icustays, hours=48)
    assert keep == {1, 3, 4}


def test_first_stay_always_kept():
    base = pd.Timestamp("2020-01-01")
    icustays = pd.DataFrame([
        {"hadm_id": 1, "stay_id": 10, "intime": base, "outtime": base + pd.Timedelta(hours=6)},
    ])
    assert _non_readmission_stays(icustays, hours=48) == {10}
