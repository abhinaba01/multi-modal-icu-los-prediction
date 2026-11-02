"""Raw rows for the time-series variables, from PostgreSQL or from in-memory tables.

Both sources apply only the time window and return identical frames:

* ``events(table, ...)`` for chartevents, labevents and outputevents: ``stay_id, itemid,
  charttime, storetime, value`` with ``intime < charttime <= t0`` (labevents matched on
  ``subject_id`` and time, as Stage 2 specifies);
* ``intervals(...)``: norepinephrine-equivalent intervals overlapping ``(intime, t0]``;
* ``sofa(...)``: hourly SOFA rows with ``intime < endtime <= t0``.

The availability (storetime) and plausibility rules are applied afterwards, once, in
:mod:`fate_icu.data.timeseries.extract`. The SQL lives in ``sql/timeseries/``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from fate_icu.data.contract import Contract
from fate_icu.utils.env import REPO_ROOT

SQL_DIR = REPO_ROOT / "sql" / "timeseries"
EVENT_COLUMNS = ["stay_id", "itemid", "charttime", "storetime", "value"]
INTERVAL_COLUMNS = ["stay_id", "starttime", "endtime", "value"]
SOFA_COMPONENTS = ["respiration", "coagulation", "liver", "cardiovascular", "cns", "renal"]
SOFA_COLUMNS = ["stay_id", "endtime", *SOFA_COMPONENTS]
NE_TABLE = "norepinephrine_equivalent_dose"
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")


def _standardise(frame: pd.DataFrame, columns: list[str], times: Sequence[str]) -> pd.DataFrame:
    """Fixed column order and dtypes, rows sorted, so both sources compare equal."""
    out = frame[columns].copy()
    out["stay_id"] = out["stay_id"].astype("int64")
    if "itemid" in out.columns:
        out["itemid"] = out["itemid"].astype("int64")
    for column in times:
        out[column] = pd.to_datetime(out[column]).astype("datetime64[ns]")
    for column in columns:
        if column not in ("stay_id", "itemid", *times):
            out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    sort_cols = [c for c in columns if c in ("stay_id", "charttime", "starttime", "endtime", "itemid")]
    sort_cols += [c for c in columns if c not in sort_cols]
    return out.sort_values(sort_cols, kind="mergesort", na_position="first").reset_index(drop=True)


class FrameEventSource:
    """Time-windowed rows from in-memory tables (synthetic fixtures, tests).

    Args:
        tables: Frames keyed ``chartevents``, ``labevents``, ``outputevents``,
            ``norepinephrine_equivalent_dose`` and ``sofa``, with MIMIC-IV columns.
        hours: The window W.
    """

    def __init__(self, tables: Mapping[str, pd.DataFrame], hours: int):
        self.tables = dict(tables)
        self.hours = int(hours)

    def _with_window(self, frame: pd.DataFrame, cohort: pd.DataFrame, key: str) -> pd.DataFrame:
        keys = cohort.reset_index()[["stay_id", "subject_id", "intime"]]
        if key == "subject_id":
            frame = frame.drop(columns=["stay_id"], errors="ignore")
            merged = frame.merge(keys, on="subject_id", how="inner")
        else:
            merged = frame.merge(keys[["stay_id", "intime"]], on="stay_id", how="inner")
        merged["t0"] = pd.to_datetime(merged["intime"]) + pd.Timedelta(hours=self.hours)
        return merged

    def events(self, table: str, cohort: pd.DataFrame, itemids: Sequence[int]) -> pd.DataFrame:
        frame = self.tables[table]
        frame = frame[frame["itemid"].isin(list(itemids))]
        value = "value" if table == "outputevents" else "valuenum"
        merged = self._with_window(frame, cohort, "subject_id" if table == "labevents" else "stay_id")
        charttime = pd.to_datetime(merged["charttime"])
        merged = merged[(charttime > merged["intime"]) & (charttime <= merged["t0"])]
        if value != "value":  # the numeric column replaces the text one
            merged = merged.drop(columns=["value"], errors="ignore").rename(columns={value: "value"})
        return _standardise(merged, EVENT_COLUMNS, ("charttime", "storetime"))

    def intervals(self, cohort: pd.DataFrame) -> pd.DataFrame:
        merged = self._with_window(self.tables[NE_TABLE], cohort, "stay_id")
        start, end = pd.to_datetime(merged["starttime"]), pd.to_datetime(merged["endtime"])
        merged = merged[(start < merged["t0"]) & (end > merged["intime"])]
        merged = merged.rename(columns={"norepinephrine_equivalent_dose": "value"})
        return _standardise(merged, INTERVAL_COLUMNS, ("starttime", "endtime"))

    def sofa(self, cohort: pd.DataFrame) -> pd.DataFrame:
        merged = self._with_window(self.tables["sofa"], cohort, "stay_id")
        end = pd.to_datetime(merged["endtime"])
        merged = merged[(end > merged["intime"]) & (end <= merged["t0"])]
        return _standardise(merged, SOFA_COLUMNS, ("endtime",))


class SqlEventSource:
    """Time-windowed rows from the MIMIC-IV PostgreSQL database (``sql/timeseries/*.sql``).

    Args:
        engine: SQLAlchemy engine for the database named in ``FATE_DB_URL``.
        contract: Supplies the schema names.
        hours: The window W.
    """

    def __init__(self, engine: Any, contract: Contract, hours: int):
        self.engine = engine
        self.hours = int(hours)
        schemas = {key: contract.schema(key) for key in ("hosp", "icu", "derived")}
        for name in schemas.values():
            if not _IDENTIFIER.match(name):
                raise ValueError(f"unsafe schema name {name!r}")
        self.schemas = schemas

    def _query(self, name: str, params: dict[str, Any]) -> pd.DataFrame:
        from sqlalchemy import text

        sql = (SQL_DIR / f"{name}.sql").read_text(encoding="utf-8").format(**self.schemas)
        with self.engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=params)

    def _params(self, cohort: pd.DataFrame) -> dict[str, Any]:
        return {"stay_ids": [int(s) for s in cohort.index], "hours": self.hours}

    def events(self, table: str, cohort: pd.DataFrame, itemids: Sequence[int]) -> pd.DataFrame:
        params = {**self._params(cohort), "itemids": [int(i) for i in itemids]}
        return _standardise(self._query(table, params), EVENT_COLUMNS, ("charttime", "storetime"))

    def intervals(self, cohort: pd.DataFrame) -> pd.DataFrame:
        frame = self._query(NE_TABLE, self._params(cohort))
        return _standardise(frame, INTERVAL_COLUMNS, ("starttime", "endtime"))

    def sofa(self, cohort: pd.DataFrame) -> pd.DataFrame:
        return _standardise(self._query("sofa", self._params(cohort)), SOFA_COLUMNS, ("endtime",))
