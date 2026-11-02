"""Stage 2 tensors on synthetic data: bins, mask, planted cases, fill rules, determinism."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fate_icu.data.synthetic import FIO2, GU_IRRIGANT_IN, GU_IRRIGANT_OUT, HEART_RATE, URINE_FOLEY
from fate_icu.data.timeseries.build import TimeSeriesConfig, build, load_tensors, run
from fate_icu.data.timeseries.report import legacy_comparison
from fate_icu.data.timeseries.sources import FrameEventSource
from fate_icu.data.timeseries.tensors import Normaliser, assign_bins, tensor_sha256

H = 24


def _pos(ts_build, stay):
    return int(np.flatnonzero(ts_build.stay_ids == stay)[0])


def _ch(variables, name):
    return variables.index(name)


def _bin(times, intime):
    seconds = (pd.to_datetime(times) - intime).dt.total_seconds()
    return (np.ceil(seconds / 3600) - 1).astype(int)


# --- bins and shapes ---------------------------------------------------------------------


def test_bins_at_the_boundaries():
    intime = pd.Timestamp("2150-01-01 08:00:00")
    offsets = [0, 1, 3600, 3601, H * 3600, H * 3600 + 1, -1]
    times = pd.Series([intime + pd.Timedelta(seconds=s) for s in offsets])
    bins = assign_bins(times, pd.Series([intime] * len(offsets)), H)
    assert bins.tolist() == [-1, 0, 0, 1, H - 1, -1, -1]


def test_shapes_types_and_order(ts_build, synth_tables, variables):
    n = len(synth_tables.cohort)
    for name in ("values", "mask", "delta"):
        array = ts_build.arrays[name]
        assert array.shape == (n, H, 30) and array.dtype == np.float32, name
    assert ts_build.raw.shape == (n, H, 30)
    assert ts_build.stay_ids.tolist() == sorted(synth_tables.cohort.index)
    assert ts_build.channels == variables.names
    values, mask, delta = (ts_build.arrays[k] for k in ("values", "mask", "delta"))
    assert np.isfinite(values).all()
    assert set(np.unique(mask)) <= {0.0, 1.0}
    assert (delta[mask == 1] == 0).all() and (delta[mask == 0] >= 1).all() and delta.max() <= 24
    assert (mask == ~np.isnan(ts_build.raw)).all()


# --- the mask equals an independent recomputation from the raw tables ---------------------


def _expected_cells(synth, cohort, variables, name) -> set[tuple[int, int]]:
    """(stay_id, hour) cells with an accepted value, recomputed directly from the tables."""
    v = variables.get(name)
    keys = cohort.reset_index()[["stay_id", "subject_id", "intime"]]
    keys["t0"] = keys["intime"] + pd.Timedelta(hours=H)
    if v.source == "sofa":
        rows = synth.tables["sofa"].merge(keys, on="stay_id")
        rows = rows[
            (rows["endtime"] > rows["intime"]) & (rows["endtime"] <= rows["t0"]) & rows[v.column].notna()
        ]
        rows = rows[v.valid.contains(rows[v.column].to_numpy(dtype=float))]
        return set(zip(rows["stay_id"], _bin(rows["endtime"], rows["intime"]), strict=True))
    if v.source == "norepinephrine_equivalent_dose":
        rows = synth.tables["norepinephrine_equivalent_dose"].merge(keys, on="stay_id")
        cells = set()
        for r in rows.itertuples(index=False):
            for h in range(H):
                start, end = r.intime + pd.Timedelta(hours=h), r.intime + pd.Timedelta(hours=h + 1)
                if (
                    r.starttime < end
                    and r.endtime > start
                    and v.valid.contains(np.array([r.norepinephrine_equivalent_dose]))[0]
                ):
                    cells.add((r.stay_id, h))
        return cells
    table = synth.tables[v.source]
    value = "value" if v.source == "outputevents" else "valuenum"
    if v.source == "labevents":
        rows = table.merge(keys, on="subject_id")
    else:
        rows = table.merge(keys.drop(columns="subject_id"), on="stay_id")
    rows = rows[rows["itemid"].isin(v.itemids)]
    rows = rows[(rows["charttime"] > rows["intime"]) & (rows["charttime"] <= rows["t0"])]
    rows = rows[rows["storetime"].notna() & (rows["storetime"] <= rows["t0"]) & rows[value].notna()]
    if v.combine == "sum_cocharted":
        counts = rows.groupby(["stay_id", "charttime"])["itemid"].nunique()
        totals = rows.groupby(["stay_id", "charttime"])[value].sum()[counts == len(v.itemids)]
        totals = totals[v.valid.contains(totals.to_numpy())].reset_index()
        totals = totals.merge(keys[["stay_id", "intime"]], on="stay_id")
        return set(zip(totals["stay_id"], _bin(totals["charttime"], totals["intime"]), strict=True))
    vals = rows[value].to_numpy(dtype=float)
    if v.fraction_to_percent:
        vals = np.where(vals <= 1, vals * 100, vals)
    rows = rows[v.valid.contains(vals)]
    return set(zip(rows["stay_id"], _bin(rows["charttime"], rows["intime"]), strict=True))


def test_mask_matches_the_raw_tables_for_every_channel(synth, ts_build, synth_tables, variables):
    cohort = synth_tables.cohort.sort_index()
    for name in variables.names:
        c = _ch(variables, name)
        pos, hour = np.nonzero(ts_build.arrays["mask"][:, :, c])
        got = set(zip(ts_build.stay_ids[pos].tolist(), hour.tolist(), strict=True))
        assert got == _expected_cells(synth, cohort, variables, name), name


# --- planted cases ---------------------------------------------------------------------


def _chart(synth, stay, itemids, lo, hi):
    ce = synth.tables["chartevents"]
    return ce[
        (ce["stay_id"] == stay)
        & ce["itemid"].isin(itemids)
        & (ce["charttime"] > lo)
        & (ce["charttime"] <= hi)
    ]


def test_planted_rows_follow_the_rules(synth, ts_build, synth_tables, variables):
    stay = synth.roles["stage2_planted"]
    p, intime = _pos(ts_build, stay), synth_tables.cohort.loc[stay, "intime"]
    raw = ts_build.raw

    def hour(h):
        return intime + pd.Timedelta(hours=h), intime + pd.Timedelta(hours=h + 1)

    # stored after t0 (250 at 12.25 h), no storetime (240 at 13.25 h), implausible (0 at 8.25 h)
    for h, planted in ((12, 250.0), (13, 240.0), (8, 0.0)):
        rows = _chart(synth, stay, [HEART_RATE], *hour(h))
        kept = rows[rows["valuenum"] != planted]["valuenum"]
        assert raw[p, h, _ch(variables, "heart_rate")] == pytest.approx(kept.mean(), rel=1e-6)
    # MAP 801 at 7.25 h is implausible and dropped
    mbp_items = variables.get("mbp").itemids
    rows = _chart(synth, stay, mbp_items, *hour(7))
    assert (rows["valuenum"] == 801.0).any()
    assert raw[p, 7, _ch(variables, "mbp")] == pytest.approx(
        rows[rows["valuenum"] != 801.0]["valuenum"].mean(), rel=1e-6
    )
    # FiO2 0.5 at 9.25 h is a fraction: 50 %
    assert _chart(synth, stay, [FIO2], *hour(9))["valuenum"].tolist() == [0.5]
    assert raw[p, 9, _ch(variables, "fio2")] == pytest.approx(50.0)
    # creatinine at 20 h stored after t0: hour 19 has no creatinine at all
    assert np.isnan(raw[p, 19, _ch(variables, "creatinine")])
    # urine hour 5: Foley + irrigant out (700) - irrigant in (500)
    oe = synth.tables["outputevents"]
    lo, hi = hour(5)
    sel = oe[(oe["stay_id"] == stay) & (oe["charttime"] > lo) & (oe["charttime"] <= hi)]
    foley = sel[sel["itemid"] == URINE_FOLEY]["value"].sum()
    assert set(sel["itemid"]) == {URINE_FOLEY, GU_IRRIGANT_IN, GU_IRRIGANT_OUT}
    assert raw[p, 5, _ch(variables, "urine_output")] == pytest.approx(foley + 700.0 - 500.0, rel=1e-6)


def test_window_edges_on_the_leakage_stay(synth, ts_build, synth_tables, variables):
    stay = synth.roles["leakage"]
    p, intime = _pos(ts_build, stay), synth_tables.cohort.loc[stay, "intime"]
    t0 = intime + pd.Timedelta(hours=H)
    hr = ts_build.raw[p, :, _ch(variables, "heart_rate")]
    first = _chart(synth, stay, [HEART_RATE], intime, intime + pd.Timedelta(hours=1))
    assert hr[0] == pytest.approx(first["valuenum"].mean())  # the 50 bpm exactly at intime is excluded
    # the 100 bpm charted exactly at t0 was stored 5 min later: the internship's mean keeps
    # it, Stage 2's availability rule drops it
    last = _chart(synth, stay, [HEART_RATE], t0 - pd.Timedelta(hours=1), t0)
    at_t0 = last[last["charttime"] == t0]
    assert at_t0["valuenum"].tolist() == [100.0] and (at_t0["storetime"] > t0).all()
    stored_in_time = last[last["storetime"] <= t0]["valuenum"]
    assert hr[H - 1] == pytest.approx(stored_in_time.mean())
    # SOFA: rows end on the hour; one ends exactly at intime (excluded), one exactly at t0 (last hour)
    sofa = synth.tables["sofa"]
    rows = sofa[sofa["stay_id"] == stay]
    assert (rows["endtime"] == intime).any() and (rows["endtime"] == t0).any()
    at_t0 = rows[rows["endtime"] == t0].iloc[0]
    got = ts_build.raw[p, H - 1, _ch(variables, "sofa_cardiovascular")]
    assert (np.isnan(got) and pd.isna(at_t0["cardiovascular"])) or got == at_t0["cardiovascular"]


def test_vasopressor_intervals_and_zero_fill(synth, ts_build, synth_tables, variables):
    stay, c = synth.roles["leakage"], _ch(variables, "ne_dose")
    p = _pos(ts_build, stay)
    dose = ts_build.raw[p, :, c]
    assert (dose[0:3] >= 0.15).all()  # the interval from intime - 2 h to intime + 3 h
    assert (dose[22:24] >= 0.2).all()  # the interval from 22 h, clipped at t0
    stats = ts_build.normaliser.channels[c]
    zero_value = np.float32((0.0 - stats.mean) / stats.std)
    unobserved = ts_build.arrays["mask"][:, :, c] == 0
    assert np.all(ts_build.arrays["values"][:, :, c][unobserved] == zero_value)  # never forward-filled


# --- normaliser: forward fill only, delta, train-only fitting ------------------------------


def _toy_raw(variables, rng):
    """Stay 0 (train): every channel observed; stay 1 (test): heart rate at hours 5 and 10."""
    raw = np.full((2, H, len(variables.variables)), np.nan)
    raw[0] = rng.uniform(1.0, 4.0, size=(H, len(variables.variables)))
    raw[0, :, 0] = rng.uniform(50.0, 150.0, size=H)  # heart rate: the test values lie inside this range
    raw[1, 5, 0], raw[1, 10, 0] = 80.0, 120.0
    return raw, np.array([True, False])


def test_forward_fill_only_and_delta(variables):
    raw, train = _toy_raw(variables, np.random.RandomState(0))
    norm = Normaliser.fit(raw, train, variables, delta_cap_hours=6)
    out = norm.transform(raw, variables)
    hr, stats = out["values"][1, :, 0], norm.channels[0]
    median_value = np.float32((stats.median - stats.mean) / stats.std)

    def scaled(x):
        return np.float32((np.clip(x, stats.clip_low, stats.clip_high) - stats.mean) / stats.std)

    assert np.all(hr[:5] == median_value)  # before the first value: train median, not hour 5's value
    assert np.all(hr[5:10] == scaled(80.0)) and np.all(hr[10:] == scaled(120.0))
    assert out["mask"][1, :, 0].tolist() == [1.0 if t in (5, 10) else 0.0 for t in range(H)]
    assert out["delta"][1, :6, 0].tolist() == [1, 2, 3, 4, 5, 0]
    assert out["delta"][1, 6:11, 0].tolist() == [1, 2, 3, 4, 0]
    assert out["delta"][1, 11:, 0].max() == 6  # capped
    later = raw.copy()
    later[1, 10, 0] = 30.0
    changed = norm.transform(later, variables)["values"][1, :, 0]
    assert np.array_equal(changed[:10], hr[:10])  # a later value never reaches earlier hours


def test_normaliser_fits_on_train_only(ts_build, variables):
    raw = ts_build.raw.astype(float)
    train = ts_build.split == "train"
    fitted = Normaliser.fit(raw, train, variables).to_dict()
    perturbed = raw.copy()
    perturbed[~train] = perturbed[~train] * 3 + 7
    assert Normaliser.fit(perturbed, train, variables).to_dict() == fitted


def test_log1p_clip_and_standardise(ts_build, variables):
    c = _ch(variables, "creatinine")
    stats = ts_build.normaliser.channels[c]
    observed = ts_build.arrays["mask"][:, :, c] == 1
    raw = ts_build.raw[:, :, c][observed].astype(float)
    expected = (np.clip(np.log1p(raw), stats.clip_low, stats.clip_high) - stats.mean) / stats.std
    np.testing.assert_allclose(ts_build.arrays["values"][:, :, c][observed], expected, rtol=1e-5, atol=1e-5)


def test_require_signal_fails_loudly(variables):
    raw, train = _toy_raw(variables, np.random.RandomState(1))
    raw[0, :, 3] = np.nan  # no train value for one channel
    with pytest.raises(ValueError, match="no observed value"):
        Normaliser.fit(raw, train, variables, require_signal=True)
    with pytest.warns(UserWarning):
        assert Normaliser.fit(raw, train, variables, require_signal=False).warnings


# --- determinism, batching, artifacts -----------------------------------------------------


def test_build_is_deterministic_and_batch_invariant(synth, ts_build, synth_tables, synth_split, variables):
    again = build(
        FrameEventSource(synth.tables, H), synth_tables.cohort, synth_split, variables, H, batch_size=7
    )
    assert again.sha256 == ts_build.sha256
    for name in ("values", "mask", "delta"):
        assert np.array_equal(again.arrays[name], ts_build.arrays[name]), name
    assert np.array_equal(again.raw, ts_build.raw, equal_nan=True)
    assert tensor_sha256(ts_build.arrays) == ts_build.sha256


def test_internship_comparison_sees_the_dropped_values(synth, ts_build, synth_tables):
    report = legacy_comparison(ts_build.extraction, synth_tables.features)
    assert report["heart_rate"]["internship_feature"] == "heart_rate"
    stay = synth.roles["stage2_planted"]
    m = ts_build.extraction.measurements
    stage2 = m[(m["stay_id"] == stay) & (m["channel"] == "heart_rate")]["value"].mean()
    assert stage2 != pytest.approx(synth_tables.features.loc[stay, "heart_rate"])


def test_run_writes_tensors_and_aggregates(synth, synth_tables, contract, tmp_path, monkeypatch):
    monkeypatch.setenv("FATE_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("FATE_DB_URL", "postgresql+psycopg://alice:s3cret@localhost:5432/mimiciv_demo")
    config = TimeSeriesConfig(run_name="unit", source="postgres", require_signal=True)
    report = run(config, contract, tables=synth_tables, source=FrameEventSource(synth.tables, H))
    out = tmp_path / "timeseries" / report["run_id"]
    for name in ("tensors.npz", "normaliser.json", "report.json", "run_manifest.json"):
        assert (out / name).exists(), name
    loaded = load_tensors(out)
    assert loaded["values"].shape == (len(synth_tables.cohort), H, 30)
    assert loaded["channels"][0] == "heart_rate" and loaded["normaliser"].channels[0].name == "heart_rate"
    manifest = (out / "run_manifest.json").read_text(encoding="utf-8")
    assert "s3cret" not in manifest and json.loads(manifest)["data"]["variables_sha256"]
    assert report["tensor_sha256"] == json.loads(manifest)["extra"]["tensor_sha256"]
    for counts in report["rows_by_source"].values():
        assert all(isinstance(v, str) or v >= 10 for v in counts.values())
