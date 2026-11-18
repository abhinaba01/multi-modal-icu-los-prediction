"""Stage 3 model inputs: train-only fitting, alignment with the contract split, image pairing."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from fate_icu.data.targets import resolve_long_stay_threshold
from fate_icu.models.data import StaticEncoding, StayData, assemble

FEATURES = ["first_careunit", "anchor_age", "heart_rate", "gender"]


@pytest.fixture(scope="module")
def tables(synth_tables):
    return synth_tables


@pytest.fixture(scope="module")
def stays(tables, synth_split, ts_build, contract):
    threshold, _ = resolve_long_stay_threshold(tables.los, synth_split, contract, require_frozen=False)
    tensors = {
        "stay_ids": ts_build.stay_ids,
        "split": ts_build.split,
        "channels": ts_build.channels,
        **ts_build.arrays,
    }
    return tensors, threshold


def build(tables, synth_split, stays, contract, image_features=None, image_dim=7):
    tensors, threshold = stays
    return assemble(
        cohort=tables.cohort,
        features=tables.features[contract.feature_names("fate_static")],
        split=synth_split,
        tensors=tensors,
        image_features=image_features or {},
        image_dim=image_dim,
        contract=contract,
        threshold_days=threshold,
        require_signal=False,  # the synthetic tables have constant SOFA onset columns
    )


def test_assembled_arrays_follow_the_tensor_order_and_the_contract_split(
    tables, synth_split, stays, contract
):
    data, encoding = build(tables, synth_split, stays, contract)
    tensors, _ = stays
    assert np.array_equal(data.stay_ids, tensors["stay_ids"])
    assert np.array_equal(data.split, synth_split.loc[data.stay_ids].to_numpy(dtype=str))
    assert data.values.shape == tensors["values"].shape
    assert data.static.shape == (len(data), encoding.n_features)
    assert data.static_missing.shape == (len(data), len(encoding.indicator_columns))
    assert np.isfinite(data.static).all() and not np.isnan(data.static_missing).any()
    # patients never cross splits
    by_patient = pd.DataFrame({"p": data.subject_ids, "s": data.split}).groupby("p")["s"].nunique()
    assert (by_patient == 1).all()


def test_labels_use_the_contract_rule_and_the_given_threshold(tables, synth_split, stays, contract):
    data, _ = build(tables, synth_split, stays, contract)
    _, threshold = stays
    rounded = tables.los.loc[data.stay_ids].round(contract.target.regression_rounding_dp).to_numpy()
    assert np.array_equal(data.y_long, (rounded > threshold).astype(np.float32))
    assert np.allclose(data.los_days, rounded)
    assert data.threshold_days == threshold


def test_static_encoding_is_fitted_on_train_stays_only(tables, contract):
    cols = tables.features[contract.feature_names("fate_static")]
    cohort = tables.cohort.loc[cols.index]
    train = np.arange(len(cols)) % 3 != 0
    base = StaticEncoding.fit(cols, train, contract, require_signal=False)
    changed = cols.copy()
    numeric = [c for c in cols.columns if pd.api.types.is_numeric_dtype(cols[c])]
    changed.loc[~train, numeric] = changed.loc[~train, numeric].astype(float) * 1000 + 99999
    other = StaticEncoding.fit(changed, train, contract, require_signal=False)
    assert np.array_equal(base.fill, other.fill)
    assert base.indicator_columns == other.indicator_columns
    pd.testing.assert_series_equal(base.preprocessor.mins, other.preprocessor.mins)
    pd.testing.assert_series_equal(base.preprocessor.spans, other.preprocessor.spans)
    assert len(cohort) == len(cols)


def test_a_column_missing_in_train_gets_an_indicator_and_a_train_median_fill(tables, contract):
    cols = tables.features[contract.feature_names("fate_static")].copy()
    train = np.ones(len(cols), dtype=bool)
    numeric = next(c for c in cols.columns if pd.api.types.is_numeric_dtype(cols[c]))
    cols.iloc[:3, cols.columns.get_loc(numeric)] = np.nan
    encoding = StaticEncoding.fit(cols, train, contract, require_signal=False)
    position = encoding.columns.index(numeric)
    assert position in encoding.indicator_columns
    values, missing = encoding.transform(cols)
    assert np.isfinite(values).all()
    assert (values[:3, position] == np.float32(encoding.fill[position])).all()
    assert missing[:3, encoding.indicator_columns.index(position)].tolist() == [1.0, 1.0, 1.0]


def test_a_feature_with_no_train_value_stops_the_run(tables, contract):
    cols = tables.features[contract.feature_names("fate_static")].copy()
    numeric = next(c for c in cols.columns if pd.api.types.is_numeric_dtype(cols[c]))
    cols[numeric] = np.nan
    with pytest.raises(ValueError, match="no train value"):
        StaticEncoding.fit(cols, np.ones(len(cols), dtype=bool), contract)


def test_groups_cover_every_column_once(tables, contract):
    cols = tables.features[contract.feature_names("fate_static")]
    encoding = StaticEncoding.fit(cols, np.ones(len(cols), dtype=bool), contract, require_signal=False)
    flat = sorted(j for js in encoding.groups.values() for j in js)
    assert flat == list(range(len(encoding.columns)))
    assert set(encoding.groups) <= set(contract.feature_groups)


def test_only_the_stays_given_an_image_have_one(tables, synth_split, stays, contract):
    ids = stays[0]["stay_ids"]
    chosen = {int(ids[0]): np.ones(7, np.float32), int(ids[5]): np.full(7, 2.0, np.float32)}
    data, _ = build(tables, synth_split, stays, contract, chosen)
    assert data.has_image.sum() == 2 and data.has_image[[0, 5]].all()
    assert (data.image[~data.has_image] == 0).all() and (data.image[5] == 2).all()


def test_a_wrong_image_shape_or_a_stranger_stay_is_an_error(tables, synth_split, stays, contract):
    ids = stays[0]["stay_ids"]
    with pytest.raises(ValueError, match="shape"):
        build(tables, synth_split, stays, contract, {int(ids[0]): np.ones(3, np.float32)})
    with pytest.raises(ValueError, match="outside the cohort"):
        build(tables, synth_split, stays, contract, {-5: np.ones(7, np.float32)})


def test_tensors_with_a_different_split_are_refused(tables, synth_split, stays, contract):
    tensors, threshold = copy.deepcopy(stays[0]), stays[1]
    tensors["split"] = np.where(tensors["split"] == "train", "test", "train")
    with pytest.raises(ValueError, match="split labels"):
        assemble(
            cohort=tables.cohort,
            features=tables.features[contract.feature_names("fate_static")],
            split=synth_split,
            tensors=tensors,
            image_features={},
            image_dim=7,
            contract=contract,
            threshold_days=threshold,
            require_signal=False,
        )


def test_stay_data_checks_its_own_consistency():
    from stay_data import random_stay_data

    data = random_stay_data(n=10)
    with pytest.raises(ValueError, match="rows, expected"):
        StayData(**{**data.__dict__, "y_long": data.y_long[:5]})
    leaked = data.image.copy()
    leaked[~data.has_image] = 1.0
    with pytest.raises(ValueError, match="all-zero"):
        StayData(**{**data.__dict__, "image": leaked})
    assert set(data.dims) == {"static", "ts_channels", "hours", "image"}


def test_a_feature_without_variation_in_train_stops_a_full_run_but_not_a_smoke_run(tables, contract):
    cols = tables.features[contract.feature_names("fate_static")]
    everything = np.ones(len(cols), dtype=bool)
    with pytest.raises(ValueError, match="single train value"):
        StaticEncoding.fit(cols, everything, contract)
    smoke = StaticEncoding.fit(cols, everything, contract, require_signal=False)
    assert smoke.constant_columns and set(smoke.constant_columns) <= set(cols.columns)
