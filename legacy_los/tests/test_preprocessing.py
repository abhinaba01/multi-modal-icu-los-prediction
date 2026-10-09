"""Tests for the faithful preprocessing (label-encode, min-max, Q3 target, cleaning)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing.prepare import prepare_dataset

CONFIG = {
    "classification": {"threshold": "q3", "positive_label": "long"},
    "vital_mean_ranges": {
        "heart_rate": {"low": 25, "high": 225, "inclusive": True},
        "respiration": {"low": 7, "high": 40, "inclusive": False},
        "saturation": {"low": 50, "high": 120, "inclusive": False},
        "temperature": {"low": 86, "high": 113, "inclusive": True},
    },
}


def _frame():
    data = {
        "los": [2.0, 3.0, 5.0, 8.0, 1.5, 4.0, 10.0, 3.5],
        "first_careunit": ["MICU", "MICU", "SICU", "CCU", "MICU", "SICU", "MICU", "CCU"],
        "admission_type": ["EW EMER.", "URGENT", "EW EMER.", "ELECTIVE", "URGENT", "EW EMER.", "URGENT", "ELECTIVE"],
        "admission_location": ["ER", "ER", "TRANSFER", "CLINIC", "ER", "ER", "TRANSFER", "CLINIC"],
        "Diagnosis_1": ["I00-I99", "K00-K93", "I00-I99", "J00-J99", "I00-I99", "K00-K93", "I00-I99", "J00-J99"],
        "Diagnosis_2": ["E00-E90", "E00-E90", "N00-N99", "E00-E90", "N00-N99", "E00-E90", "N00-N99", "E00-E90"],
        "gender": ["M", "F", "M", "F", "M", "F", "M", "F"],
        "anchor_age": [60, 55, 70, 45, 65, 50, 80, 40],
        "heart_rate": [80, 90, 75, 100, 85, 95, 300, 88],   # row idx 6 (hr=300) out of range
        "respiration": [18, 20, 16, 22, 19, 17, 21, 18],
        "saturation": [98, 97, 99, 96, 98, 97, 95, 98],
        "temperature": [98.6, 99.0, 98.0, 100.0, 98.5, 97.5, 98.8, 99.1],
        "glucose": [100, 110, 90, 120, 105, 95, 130, np.nan],  # row idx 7 NaN -> complete-case drop
    }
    return pd.DataFrame(data, index=pd.Index(range(1, 9), name="stay_id"))


def test_prepare_drops_incomplete_and_out_of_range_rows():
    prepared = prepare_dataset(_frame(), CONFIG)
    # Start 8, drop the NaN-glucose row and the HR=300 row -> 6 remain.
    assert len(prepared.X) == 6
    assert 7 not in prepared.X.index   # HR out of range
    assert 8 not in prepared.X.index   # glucose NaN


def test_prepare_min_max_scales_all_features_to_unit_interval():
    prepared = prepare_dataset(_frame(), CONFIG)
    assert list(prepared.X.columns) == prepared.feature_columns
    assert "los" not in prepared.feature_columns
    values = prepared.X.to_numpy()
    assert np.nanmin(values) >= 0.0 - 1e-9
    assert np.nanmax(values) <= 1.0 + 1e-9


def test_prepare_labels_categoricals_as_numbers():
    prepared = prepare_dataset(_frame(), CONFIG)
    # gender was categorical; after encode + min-max it is numeric in [0,1].
    assert pd.api.types.is_numeric_dtype(prepared.X["gender"])
    assert set(np.unique(prepared.X["gender"].to_numpy())).issubset({0.0, 1.0})


def test_prepare_threshold_is_third_quartile_and_binary_target():
    prepared = prepare_dataset(_frame(), CONFIG)
    remaining_los = [2.0, 3.0, 5.0, 8.0, 1.5, 4.0]
    expected_q3 = float(np.quantile(remaining_los, 0.75))
    assert abs(prepared.threshold - expected_q3) < 1e-9
    # long = los > Q3(~4.75): only 5.0 and 8.0
    assert prepared.y_class.sum() == 2
    assert set(prepared.y_class.unique()).issubset({0, 1})
