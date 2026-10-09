"""Tests for the single-pass multi-sink scanner (optimization A).

Proves that reading a table once with several sinks yields exactly what running
each sink independently would -- so building all three feature tables from one
scan cannot change any feature value. Also checks the two window boundary rules
(mean uses (intime, +24h]; raw uses [intime, +24h)).
"""

from __future__ import annotations

import pandas as pd
import pandas.testing as pdt

from src.features.aggregate import MeanSink, RawSink, stream_multi

CONFIG = {"observation_window_hours": 24}
T0 = pd.Timestamp("2020-01-01")


class FakeLoader:
    """Minimal loader that yields in-memory chunks (no disk / MIMIC needed)."""

    def __init__(self, df: pd.DataFrame, chunksize: int = 2):
        self.df = df.reset_index(drop=True)
        self.chunksize = chunksize

    def iter_chunks(self, table, usecols=None, chunksize=None, dtype=None):
        size = chunksize or self.chunksize
        for start in range(0, len(self.df), size):
            chunk = self.df.iloc[start:start + size]
            yield (chunk[usecols] if usecols else chunk).copy()


def _events():
    """chartevents-like rows for one stay, spanning both window boundaries."""
    def row(itemid, hours, value):
        return {"stay_id": 1, "itemid": itemid, "charttime": T0 + pd.Timedelta(hours=hours), "valuenum": value}
    return pd.DataFrame([
        row(111, 1, 10.0), row(111, 24, 20.0),         # mean 'a': (10+20)/2 over (0,24]
        row(222, 0, 5.0), row(222, 5, 15.0), row(222, 24, 25.0),
        row(333, 2, 7.0),
        row(999, 3, 100.0),                             # in no sink -> ignored
    ])


COHORT = pd.DataFrame([{"stay_id": 1, "intime": T0}])
MEAN = MeanSink("m", {111: "a", 222: "b"})              # (intime, +24h]
RAW = RawSink("r", {222, 333})                          # [intime, +24h)


def _sorted_raw(df):
    return df.sort_values(["itemid", "hours"]).reset_index(drop=True)


def test_mean_window_is_open_lower_closed_upper():
    loader = FakeLoader(_events(), chunksize=2)  # split across chunks on purpose
    res = stream_multi(loader, CONFIG, "chartevents", "stay_id", "valuenum", "charttime", COHORT, [MEAN])
    m = res["m"].loc[1]
    assert m["a"] == 15.0                        # 10 @1h and 20 @24h (24h included)
    assert m["b"] == 20.0                        # 15 @5h and 25 @24h; 5 @0h EXCLUDED


def test_raw_window_is_closed_lower_open_upper():
    loader = FakeLoader(_events(), chunksize=2)
    res = stream_multi(loader, CONFIG, "chartevents", "stay_id", "valuenum", "charttime", COHORT, [RAW])
    raw = _sorted_raw(res["r"])
    # 222 @0h INCLUDED, 222 @24h EXCLUDED, 333 @2h included -> 3 rows, values {5,15,7}
    assert list(raw["itemid"]) == [222, 222, 333]
    assert list(raw["value"]) == [5.0, 15.0, 7.0]
    assert list(raw["hours"]) == [0.0, 5.0, 2.0]


def test_multi_sink_equals_independent_runs():
    loader_multi = FakeLoader(_events(), chunksize=2)
    combined = stream_multi(loader_multi, CONFIG, "chartevents", "stay_id", "valuenum", "charttime",
                            COHORT, [MEAN, RAW])
    mean_only = stream_multi(FakeLoader(_events(), 2), CONFIG, "chartevents", "stay_id", "valuenum",
                             "charttime", COHORT, [MEAN])
    raw_only = stream_multi(FakeLoader(_events(), 2), CONFIG, "chartevents", "stay_id", "valuenum",
                            "charttime", COHORT, [RAW])
    # The second sink must not perturb the first, and vice versa.
    pdt.assert_frame_equal(combined["m"], mean_only["m"])
    pdt.assert_frame_equal(_sorted_raw(combined["r"]), _sorted_raw(raw_only["r"]))
