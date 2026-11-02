"""Loaders give the legacy code identical inputs, whatever the source."""

from __future__ import annotations

import pandas as pd
import pytest
from sqlalchemy import text

from fate_icu.data.synthetic import SCHEMAS
from fate_icu.tabular._legacy import legacy_config
from fate_icu.tabular.api import build_tables, load_tables
from fate_icu.tabular.loaders import (
    FrameLoader,
    PostgresLoader,
    csv_loader,
    describe_source,
    legacy_itemid_filters,
)


def _assert_tables_equal(a, b):
    pd.testing.assert_frame_equal(a.cohort, b.cohort, check_dtype=False)
    pd.testing.assert_frame_equal(a.features, b.features, check_dtype=False)
    pd.testing.assert_frame_equal(a.provenance, b.provenance, check_dtype=False)
    assert a.flow == b.flow


def test_itemid_prefilter_changes_nothing(synth, synth_dir, synth_tables, contract):
    filtered = load_tables(
        loader=FrameLoader(synth.tables, itemid_filters=legacy_itemid_filters()),
        contract=contract,
        gem_path=synth.gem_path(synth_dir),
        cache=False,
    )
    _assert_tables_equal(filtered, synth_tables)


def test_legacy_csv_loader_matches_frame_loader(synth, synth_dir, synth_tables, contract, monkeypatch):
    monkeypatch.delenv("MIMIC_RAW_DIR", raising=False)
    cfg = legacy_config(contract, raw_dir=synth_dir, gem_path=synth.gem_path(synth_dir))
    from_csv = build_tables(csv_loader(cfg), cfg, contract)
    _assert_tables_equal(from_csv, synth_tables)


def test_csv_loader_refuses_a_conflicting_env_override(synth_dir, contract, monkeypatch, tmp_path):
    monkeypatch.setenv("MIMIC_RAW_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="MIMIC_RAW_DIR"):
        csv_loader(legacy_config(contract, raw_dir=synth_dir))


def test_postgres_query_is_schema_qualified_and_safe(contract):
    loader = PostgresLoader(
        "unused", contract, engine=object(), itemid_filters={"chartevents": frozenset({220045, 220210})}
    )
    sql, params = loader.build_query("chartevents", ["stay_id", "itemid", "charttime", "valuenum"])
    assert sql == (
        'SELECT "stay_id", "itemid", "charttime", "valuenum" FROM "mimiciv_icu"."chartevents"'
        ' WHERE "itemid" = ANY(:itemids)'
    )
    assert params == {"itemids": [220045, 220210]}
    assert loader.build_query("patients", None) == ('SELECT * FROM "mimiciv_hosp"."patients"', {})
    with pytest.raises(ValueError, match="unsafe"):
        loader.build_query("patients", ["subject_id; DROP TABLE x"])


def test_describe_source_has_no_credentials(contract):
    from sqlalchemy import create_engine

    engine = create_engine("postgresql+psycopg://alice:s3cret@localhost:5432/mimiciv_demo")
    described = describe_source(PostgresLoader("unused", contract, engine=engine))
    assert described == "postgres://localhost:5432/mimiciv_demo"
    assert "s3cret" not in described and "alice" not in described


# --- local demo database (laptop) ----------------------------------------------------


@pytest.fixture(scope="module")
def demo_loader(settings, contract):
    loader = PostgresLoader(settings.db_url, contract, itemid_filters=legacy_itemid_filters())
    with loader.engine.connect() as conn:  # fails loudly if FATE_DB_URL is set but unreachable
        conn.execute(text("SELECT 1"))
    return loader


@pytest.mark.demo
def test_demo_has_the_synthetic_schemas(demo_loader, contract):
    for table in contract.mimic.tables:
        real = demo_loader.columns(table)
        assert real, f"{contract.mimic.qualified(table)} is missing"
        assert set(SCHEMAS[table]) <= set(real), (table, sorted(set(SCHEMAS[table]) - set(real)))


@pytest.mark.demo
def test_demo_has_derived_sofa(demo_loader, contract):
    schema = contract.schema("derived")
    with demo_loader.engine.connect() as conn:
        columns = [
            r[0]
            for r in conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = :s AND table_name = 'sofa'"
                ),
                {"s": schema},
            )
        ]
        n = conn.execute(text(f'SELECT COUNT(*) FROM "{schema}"."sofa"')).scalar_one()
    assert set(SCHEMAS["sofa"]) <= set(columns)
    assert n > 0


@pytest.mark.demo
def test_demo_postgres_matches_legacy_csv_reader(demo_loader, settings, contract, monkeypatch):
    if settings.mimic_demo_dir is None or settings.icd9_gem_path is None:
        pytest.fail("set MIMIC_DEMO_DIR and ICD9_GEM_PATH in .env (docs/laptop_setup.md)")
    monkeypatch.delenv("MIMIC_RAW_DIR", raising=False)
    cfg = legacy_config(contract, raw_dir=settings.mimic_demo_dir, gem_path=settings.icd9_gem_path)
    from_csv = build_tables(csv_loader(cfg), cfg, contract)
    from_pg = build_tables(demo_loader, cfg, contract)
    _assert_tables_equal(from_pg, from_csv)
    assert from_pg.flow["final"] > 0
