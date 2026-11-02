"""No input may come from after t0 = ICU intime + W (CLAUDE.md A4).

The legacy extraction does not report what it read, so the adapter records the latest
event time each stay's features used (``provenance``). These tests assert it is never
after t0, check the planted boundary and sentinel events, and show that adding extreme
events after every t0 changes no feature at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fate_icu.data.synthetic import (
    BILIRUBIN,
    CREATININE,
    HEART_RATE,
    SPO2,
    URINE_FOLEY,
    add_post_t0_sentinels,
)
from fate_icu.tabular.api import PROVENANCE_SOURCES, load_tables
from fate_icu.tabular.loaders import FrameLoader


def test_provenance_never_after_t0(synth_tables):
    t0 = synth_tables.cohort["t0"]
    prov = synth_tables.provenance
    assert list(prov.columns) == list(PROVENANCE_SOURCES)
    for source in PROVENANCE_SOURCES:
        used = prov[source].dropna()
        assert not used.empty, source
        assert (used <= t0.loc[used.index]).all(), source


def test_features_ignore_everything_after_t0(synth, synth_dir, synth_tables, contract):
    perturbed = load_tables(
        loader=FrameLoader(add_post_t0_sentinels(synth)),
        contract=contract,
        gem_path=synth.gem_path(synth_dir),
        cache=False,
    )
    pd.testing.assert_frame_equal(perturbed.features, synth_tables.features)
    pd.testing.assert_frame_equal(perturbed.cohort, synth_tables.cohort)
    pd.testing.assert_frame_equal(perturbed.provenance, synth_tables.provenance)


def _events(table, stay_or_hadm, key, itemid):
    rows = table[(table[key] == stay_or_hadm) & (table["itemid"] == itemid)]
    return rows


def test_planted_boundary_and_sentinel_events(synth, synth_tables, contract):
    stay = synth.roles["leakage"]
    row = synth_tables.cohort.loc[stay]
    intime, t0 = row["intime"], row["t0"]
    feats = synth_tables.features.loc[stay]
    chart = synth.tables["chartevents"]

    def window_mean(itemid):  # legacy mean window (intime, t0]
        ev = _events(chart, stay, "stay_id", itemid)
        sel = (ev["charttime"] > intime) & (ev["charttime"] <= t0)
        return ev.loc[sel, "valuenum"].mean()

    # heart rate: the value at intime (50) is excluded, the one at t0 (100) included,
    # the 999 one minute after t0 excluded.
    assert feats["heart_rate"] == pytest.approx(window_mean(HEART_RATE))
    hr = _events(chart, stay, "stay_id", HEART_RATE)
    assert 100.0 in hr.loc[hr["charttime"] == t0, "valuenum"].tolist()
    assert feats["heart_rate"] < 200
    # SpO2 at exactly t0 counts in the vital mean ...
    assert feats["saturation"] == pytest.approx(window_mean(SPO2))
    # ... and the creatinine 99 / bilirubin 50 / urine 99,999 after t0 never count.
    labs = synth.tables["labevents"]
    cr = _events(labs, row["hadm_id"], "hadm_id", CREATININE)
    in_window = cr[(cr["charttime"] > intime) & (cr["charttime"] <= t0)]
    assert feats["creatinine"] == pytest.approx(in_window["valuenum"].mean())
    bili = _events(labs, row["hadm_id"], "hadm_id", BILIRUBIN)
    bili_in = bili[(bili["charttime"] >= intime) & (bili["charttime"] < t0)]
    assert feats["bilirubin_max"] == pytest.approx(bili_in["valuenum"].max())
    urine = _events(synth.tables["outputevents"], stay, "stay_id", URINE_FOLEY)
    urine_in = urine[(urine["charttime"] >= intime) & (urine["charttime"] < t0)]
    assert feats["urine_24h"] == pytest.approx(urine_in["value"].sum())


def test_late_stored_lab_is_still_used(synth, synth_tables):
    """Documents the known gap: legacy features ignore storetime (see the audit)."""
    stay = synth.roles["leakage"]
    latest_lab = synth_tables.provenance.loc[stay, "labs"]
    t0 = synth_tables.cohort.loc[stay, "t0"]
    assert latest_lab == t0 - pd.Timedelta(hours=1)  # WBC drawn at t0-1h, stored at t0+2h


def test_model_inputs_are_all_available_at_t0(contract):
    unavailable = [n for n in contract.feature_names("fate_static") if not contract.available_at_t0(n)]
    assert unavailable == []
    hours = contract.window.hours
    assert all(
        contract.window.definitions[contract.feature(n).window].ends_by(hours)
        for n in contract.feature_names("fate_static")
    )
    assert np.isfinite(hours)
