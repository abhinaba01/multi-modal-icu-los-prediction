"""Table loaders with the legacy ``MIMICLoader`` interface (``load_table``, ``iter_chunks``).

The legacy feature code only calls these two methods, so any object providing them can
feed it unchanged:

* :class:`PostgresLoader` reads the schema-qualified tables named in the contract
  (``mimiciv_demo`` on the laptop, full MIMIC-IV in the lab);
* :class:`FrameLoader` serves in-memory DataFrames (synthetic fixtures in tests);
* the legacy CSV ``MIMICLoader`` itself (:func:`csv_loader`).

``itemid_filters`` optionally drops event rows whose ``itemid`` the legacy code never
uses. The legacy code filters by itemid itself, so features are identical with or
without it (tested); it only saves time on the full database.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from fate_icu.data.contract import Contract
from fate_icu.tabular._legacy import legacy

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def legacy_itemid_filters() -> dict[str, frozenset[int]]:
    """Every itemid the legacy feature code reads, per event table."""
    vitals = legacy("features.vitals")
    labs = legacy("features.labs")
    sofa = legacy("features.sofa_static")
    return {
        "chartevents": frozenset(vitals.VITAL_ITEMIDS) | frozenset(sofa.CHART_ITEMIDS),
        "labevents": frozenset(labs.LAB_ITEMIDS) | frozenset(sofa.LAB_ITEMIDS),
        "inputevents": frozenset(sofa.VASO_ITEMIDS),
        "outputevents": frozenset(sofa.URINE_ITEMIDS),
    }


def _finish(frame: pd.DataFrame, parse_dates: list[str] | None, dtype: dict[str, Any] | None) -> pd.DataFrame:
    for column in parse_dates or []:
        frame[column] = pd.to_datetime(frame[column])
    if dtype:
        frame = frame.astype(dtype)
    return frame


class FrameLoader:
    """Serve in-memory tables through the legacy loader interface."""

    def __init__(
        self,
        tables: Mapping[str, pd.DataFrame],
        chunksize: int = 5_000,
        itemid_filters: Mapping[str, frozenset[int]] | None = None,
    ):
        self.tables = dict(tables)
        self.chunksize = chunksize
        self.itemid_filters = dict(itemid_filters or {})

    def _table(self, name: str, usecols: list[str] | None) -> pd.DataFrame:
        if name not in self.tables:
            raise KeyError(f"table {name!r} not provided")
        frame = self.tables[name]
        if name in self.itemid_filters:
            frame = frame[frame["itemid"].isin(self.itemid_filters[name])]
        return (frame[usecols] if usecols else frame).reset_index(drop=True).copy()

    def load_table(self, table_name, usecols=None, parse_dates=None, dtype=None) -> pd.DataFrame:
        return _finish(self._table(table_name, usecols), parse_dates, dtype)

    def iter_chunks(self, table_name, usecols=None, chunksize=None, dtype=None) -> Iterator[pd.DataFrame]:
        frame = self._table(table_name, usecols)
        size = chunksize or self.chunksize
        for start in range(0, max(len(frame), 1), size):
            yield _finish(frame.iloc[start : start + size].copy(), None, dtype)


class PostgresLoader:
    """Read MIMIC-IV tables from PostgreSQL through the legacy loader interface."""

    def __init__(
        self,
        db_url: str,
        contract: Contract,
        chunksize: int = 500_000,
        itemid_filters: Mapping[str, frozenset[int]] | None = None,
        engine: Any | None = None,
    ):
        if engine is None:
            from sqlalchemy import create_engine

            engine = create_engine(db_url, pool_pre_ping=True)
        self.engine = engine
        self.contract = contract
        self.chunksize = chunksize
        self.itemid_filters = dict(itemid_filters or {})

    def build_query(self, table_name: str, usecols: list[str] | None) -> tuple[str, dict[str, Any]]:
        """SQL text and bound parameters for one table read.

        Identifiers come from the contract and the legacy code; they are still checked
        against a strict pattern and quoted.
        """
        qualified = self.contract.mimic.qualified(table_name)
        for part in qualified.split("."):
            if not _IDENTIFIER.match(part):
                raise ValueError(f"unsafe identifier {part!r}")
        columns = "*"
        if usecols:
            for column in usecols:
                if not _IDENTIFIER.match(column):
                    raise ValueError(f"unsafe column name {column!r}")
            columns = ", ".join(f'"{c}"' for c in usecols)
        schema, table = qualified.split(".")
        sql = f'SELECT {columns} FROM "{schema}"."{table}"'
        params: dict[str, Any] = {}
        if table_name in self.itemid_filters:
            sql += ' WHERE "itemid" = ANY(:itemids)'
            params["itemids"] = sorted(int(i) for i in self.itemid_filters[table_name])
        return sql, params

    def load_table(self, table_name, usecols=None, parse_dates=None, dtype=None) -> pd.DataFrame:
        from sqlalchemy import text

        sql, params = self.build_query(table_name, usecols)
        with self.engine.connect() as conn:
            frame = pd.read_sql(text(sql), conn, params=params)
        return _finish(frame, parse_dates, dtype)

    def iter_chunks(self, table_name, usecols=None, chunksize=None, dtype=None) -> Iterator[pd.DataFrame]:
        from sqlalchemy import text

        sql, params = self.build_query(table_name, usecols)
        rows = chunksize or self.chunksize
        with self.engine.connect().execution_options(stream_results=True, max_row_buffer=rows) as conn:
            for chunk in pd.read_sql(text(sql), conn, params=params, chunksize=rows):
                yield _finish(chunk, None, dtype)

    def columns(self, table_name: str) -> list[str]:
        """Column names of a contract table, from ``information_schema``."""
        from sqlalchemy import text

        schema, table = self.contract.mimic.qualified(table_name).split(".")
        sql = text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = :schema AND table_name = :table ORDER BY ordinal_position"
        )
        with self.engine.connect() as conn:
            return [row[0] for row in conn.execute(sql, {"schema": schema, "table": table})]


def csv_loader(config: dict[str, Any]):
    """The legacy CSV ``MIMICLoader`` (reads ``config['data']['raw_dir']``).

    The legacy code lets the ``MIMIC_RAW_DIR`` environment variable override the
    configured folder. An empty value (as python-dotenv sets for ``MIMIC_RAW_DIR=``) is
    removed; a non-empty value pointing elsewhere is an error rather than a silent switch.
    """
    env_value = os.environ.get("MIMIC_RAW_DIR")
    if env_value is not None and not env_value.strip():
        del os.environ["MIMIC_RAW_DIR"]
    elif env_value is not None:
        configured = Path(config["data"]["raw_dir"]).resolve()
        if Path(env_value).resolve() != configured:
            raise ValueError(
                f"MIMIC_RAW_DIR={env_value} would override the requested CSV folder {configured} "
                "in the legacy loader; unset it for this run."
            )
    return legacy("data.loader").MIMICLoader(config)


def describe_source(loader: Any) -> str:
    """A credential-free description of a loader's data source (for cache keys/manifests)."""
    if isinstance(loader, PostgresLoader):
        url = loader.engine.url
        return f"postgres://{url.host}:{url.port}/{url.database}"
    if isinstance(loader, FrameLoader):
        return "in-memory"
    raw = getattr(loader, "config", {}).get("data", {}).get("raw_dir")
    return f"csv:{Path(raw).name}" if raw else type(loader).__name__
