"""Stage 2 on the local open demo database (``demo`` marker, laptop only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text

from fate_icu.data.synthetic import SCHEMAS
from fate_icu.data.timeseries.build import build
from fate_icu.data.timeseries.sources import FrameEventSource, SqlEventSource
from fate_icu.data.timeseries.variables import EVENT_SOURCES
from fate_icu.tabular.api import load_splits, load_tables

pytestmark = pytest.mark.demo


@pytest.fixture(scope="module")
def engine(settings):
    eng = create_engine(settings.db_url)
    with eng.connect() as conn:  # fails loudly if FATE_DB_URL is set but unreachable
        conn.execute(text("SELECT 1"))
    return eng


@pytest.fixture(scope="module")
def demo_cohort(settings, contract):
    return load_tables("postgres", contract=contract, settings=settings, cache=True).cohort.sort_index()


@pytest.fixture(scope="module")
def demo_build(engine, demo_cohort, contract, variables):
    hours = contract.window.hours
    split = load_splits(demo_cohort, contract)
    return build(
        SqlEventSource(engine, contract, hours), demo_cohort, split, variables, hours, require_signal=False
    )


def _read(engine, sql, params=None):
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def test_variable_itemids_exist_in_the_demo(engine, variables):
    chart = set(_read(engine, "SELECT itemid FROM mimiciv_icu.d_items")["itemid"])
    labs = set(_read(engine, "SELECT itemid FROM mimiciv_hosp.d_labitems")["itemid"])
    for v in variables.by_source("chartevents") + variables.by_source("outputevents"):
        assert set(v.itemids) <= chart, v.name
    for v in variables.by_source("labevents"):
        assert set(v.itemids) <= labs, v.name


def test_derived_vasopressor_table_has_the_synthetic_schema(engine):
    columns = _read(
        engine,
        "SELECT column_name FROM information_schema.columns WHERE table_schema = "
        "'mimiciv_derived' AND table_name = 'norepinephrine_equivalent_dose'",
    )["column_name"]
    assert set(SCHEMAS["norepinephrine_equivalent_dose"]) <= set(columns)


def test_sql_and_in_memory_sources_agree(engine, demo_cohort, contract, variables):
    hours = contract.window.hours
    tables = {
        "chartevents": _read(
            engine,
            "SELECT stay_id, itemid, charttime, storetime, valuenum "
            "FROM mimiciv_icu.chartevents WHERE itemid = ANY(:i)",
            {"i": variables.itemids("chartevents")},
        ),
        "labevents": _read(
            engine,
            "SELECT subject_id, itemid, charttime, storetime, valuenum "
            "FROM mimiciv_hosp.labevents WHERE itemid = ANY(:i)",
            {"i": variables.itemids("labevents")},
        ),
        "outputevents": _read(
            engine,
            "SELECT stay_id, itemid, charttime, storetime, value "
            "FROM mimiciv_icu.outputevents WHERE itemid = ANY(:i)",
            {"i": variables.itemids("outputevents")},
        ),
        "norepinephrine_equivalent_dose": _read(
            engine, "SELECT * FROM mimiciv_derived.norepinephrine_equivalent_dose"
        ),
        "sofa": _read(engine, "SELECT * FROM mimiciv_derived.sofa"),
    }
    sql, frame = SqlEventSource(engine, contract, hours), FrameEventSource(tables, hours)
    for table in EVENT_SOURCES:
        itemids = variables.itemids(table)
        pd.testing.assert_frame_equal(
            sql.events(table, demo_cohort, itemids), frame.events(table, demo_cohort, itemids), obj=table
        )
    pd.testing.assert_frame_equal(sql.intervals(demo_cohort), frame.intervals(demo_cohort))
    pd.testing.assert_frame_equal(sql.sofa(demo_cohort), frame.sofa(demo_cohort))


def _sql_bounds(column, valid):
    parts = []
    for op, bound in ((">", valid.gt), (">=", valid.ge), ("<", valid.lt), ("<=", valid.le)):
        if bound is not None:
            parts.append(f"{column} {op} {float(bound)}")
    return " AND ".join(parts) or "TRUE"


SIMPLE = [
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "resp_rate",
    "spo2",
    "temperature",
    "pao2",
    "paco2",
    "ph",
    "creatinine",
    "bun",
    "bilirubin",
    "platelets",
    "wbc",
    "lactate",
    "glucose",
    "sodium",
    "potassium",
    "haemoglobin",
]


@pytest.mark.parametrize("name", SIMPLE)
def test_mask_matches_an_independent_sql_count(engine, demo_cohort, demo_build, variables, name):
    """Observed (stay, hour) cells recomputed entirely in SQL with the Stage 2 rules."""
    v = variables.get(name)
    if v.source == "labevents":
        source, join = "mimiciv_hosp.labevents", "i.subject_id = e.subject_id"
    else:
        source, join = "mimiciv_icu.chartevents", "i.stay_id = e.stay_id"
    sql = f"""
        SELECT DISTINCT i.stay_id,
               CAST(CEIL(EXTRACT(EPOCH FROM (e.charttime - i.intime)) / 3600.0) AS integer) - 1 AS hour
        FROM {source} AS e JOIN mimiciv_icu.icustays AS i ON {join}
        WHERE i.stay_id = ANY(:stays) AND e.itemid = ANY(:items)
          AND e.charttime > i.intime AND e.charttime <= i.intime + INTERVAL '24 hours'
          AND e.storetime IS NOT NULL AND e.storetime <= i.intime + INTERVAL '24 hours'
          AND e.valuenum IS NOT NULL AND {_sql_bounds("e.valuenum", v.valid)}"""
    expected = _read(engine, sql, {"stays": [int(s) for s in demo_cohort.index], "items": v.itemids})
    c = variables.index(name)
    pos, hour = np.nonzero(demo_build.arrays["mask"][:, :, c])
    got = set(zip(demo_build.stay_ids[pos].tolist(), hour.tolist(), strict=True))
    assert got == set(zip(expected["stay_id"].tolist(), expected["hour"].tolist(), strict=True))


def test_demo_build_end_to_end(demo_build, demo_cohort):
    n = len(demo_cohort)
    assert demo_build.arrays["values"].shape == (n, 24, 30)
    assert np.isfinite(demo_build.arrays["values"]).all()
    assert (demo_build.arrays["mask"] == ~np.isnan(demo_build.raw)).all()
    t0 = demo_cohort["t0"]
    for column in demo_build.extraction.provenance.columns:
        used = demo_build.extraction.provenance[column].dropna()
        assert (used <= t0.loc[used.index]).all(), column
