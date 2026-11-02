"""From windowed raw rows to accepted measurements, with the Stage 2 rules applied once.

For the raw tables (chartevents, labevents, outputevents), in this order:

1. **availability**: drop rows without a ``storetime`` or stored after t0 (their result
   was not available at t0, even though ``charttime`` is inside the window);
2. drop non-numeric values;
3. per variable: convert FiO2 fractions to percent, drop values outside the
   plausibility limits (``valid``), negate subtracted items (GU irrigant in);
4. GCS: sum eye + verbal + motor charted at the same time (all three required), then the
   limits on the total.

The derived tables have no ``storetime``: norepinephrine-equivalent intervals are kept
when they overlap ``(intime, t0]`` (clipped at t0 later), SOFA rows when
``intime < endtime <= t0``.

Every run also records **provenance**: per stay and source, the latest event time and
the latest ``storetime`` it used, so tests (and every build) can assert both are <= t0.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fate_icu.data.timeseries.variables import EVENT_SOURCES, Variable, VariableSet

MEASUREMENT_COLUMNS = ["stay_id", "channel", "time", "storetime", "value"]
INTERVAL_COLUMNS = ["stay_id", "channel", "starttime", "endtime", "value"]
COUNT_KEYS = (
    "rows_in_window",
    "dropped_no_storetime",
    "dropped_stored_after_t0",
    "dropped_non_numeric",
    "dropped_implausible",
    "dropped_incomplete_gcs",
    "accepted",
)


@dataclass
class Extraction:
    """Accepted measurements for one cohort.

    Attributes:
        measurements: One row per accepted value (a GCS row is one co-charted total):
            ``stay_id, channel, time, storetime, value`` (``storetime`` is NaT for SOFA).
        intervals: Vasopressor intervals ``stay_id, channel, starttime, endtime, value``.
        counts: Rows seen and dropped per source and rule.
        dropped_by_channel: Implausible values dropped per channel.
        provenance: Per stay: latest event time and storetime used, per source.
    """

    measurements: pd.DataFrame
    intervals: pd.DataFrame
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    dropped_by_channel: dict[str, int] = field(default_factory=dict)
    provenance: pd.DataFrame = field(default_factory=pd.DataFrame)


def _t0(cohort: pd.DataFrame, hours: int) -> pd.Series:
    return pd.to_datetime(cohort["intime"]) + pd.Timedelta(hours=hours)


def _plausible(variable: Variable, values: np.ndarray) -> np.ndarray:
    return variable.valid.contains(values)


def _process_raw(
    table: str,
    raw: pd.DataFrame,
    variables: VariableSet,
    t0: pd.Series,
    counts: dict[str, int],
    dropped: dict[str, int],
) -> pd.DataFrame:
    counts["rows_in_window"] += len(raw)
    stay_t0 = raw["stay_id"].map(t0)
    no_store = raw["storetime"].isna()
    counts["dropped_no_storetime"] += int(no_store.sum())
    late = ~no_store & (raw["storetime"] > stay_t0)
    counts["dropped_stored_after_t0"] += int(late.sum())
    rows = raw[~no_store & ~late]
    non_numeric = rows["value"].isna()
    counts["dropped_non_numeric"] += int(non_numeric.sum())
    rows = rows[~non_numeric]

    parts = []
    for variable in variables.by_source(table):
        sub = rows[rows["itemid"].isin(variable.itemids)].copy()
        if sub.empty:
            continue
        if variable.fraction_to_percent:
            sub.loc[sub["value"] <= 1, "value"] = sub.loc[sub["value"] <= 1, "value"] * 100.0
        if variable.combine == "sum_cocharted":
            sub = _sum_cocharted(variable, sub, counts)
        else:
            ok = _plausible(variable, sub["value"].to_numpy())
            dropped[variable.name] = dropped.get(variable.name, 0) + int((~ok).sum())
            counts["dropped_implausible"] += int((~ok).sum())
            sub = sub[ok]
            if variable.negate_itemids:
                negate = sub["itemid"].isin(variable.negate_itemids)
                sub.loc[negate, "value"] = -sub.loc[negate, "value"]
        if variable.combine == "sum_cocharted":
            ok = _plausible(variable, sub["value"].to_numpy())
            dropped[variable.name] = dropped.get(variable.name, 0) + int((~ok).sum())
            counts["dropped_implausible"] += int((~ok).sum())
            sub = sub[ok]
        sub = sub.assign(channel=variable.name).rename(columns={"charttime": "time"})
        parts.append(sub[MEASUREMENT_COLUMNS])
    accepted = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    counts["accepted"] += len(accepted)
    return accepted


def _sum_cocharted(variable: Variable, rows: pd.DataFrame, counts: dict[str, int]) -> pd.DataFrame:
    """Total of components charted at the same time; all components required (internship rule)."""
    per_time = rows.pivot_table(
        index=["stay_id", "charttime"], columns="itemid", values="value", aggfunc="mean"
    )
    per_time = per_time.reindex(columns=variable.itemids)
    stored = rows.groupby(["stay_id", "charttime"])["storetime"].max()
    complete = per_time.notna().all(axis=1)
    counts["dropped_incomplete_gcs"] += int((~complete).sum())
    per_time = per_time[complete]
    total = per_time[variable.itemids].sum(axis=1).rename("value")
    out = total.reset_index()
    out["storetime"] = stored.reindex(total.index).to_numpy()
    return out


def _process_intervals(
    raw: pd.DataFrame, variable: Variable, counts: dict[str, int], dropped: dict[str, int]
) -> pd.DataFrame:
    counts["rows_in_window"] += len(raw)
    non_numeric = raw["value"].isna()
    counts["dropped_non_numeric"] += int(non_numeric.sum())
    rows = raw[~non_numeric]
    ok = _plausible(variable, rows["value"].to_numpy())
    dropped[variable.name] = dropped.get(variable.name, 0) + int((~ok).sum())
    counts["dropped_implausible"] += int((~ok).sum())
    rows = rows[ok].assign(channel=variable.name)
    counts["accepted"] += len(rows)
    return rows[INTERVAL_COLUMNS]


def _process_sofa(
    raw: pd.DataFrame, variables: VariableSet, counts: dict[str, int], dropped: dict[str, int]
) -> pd.DataFrame:
    counts["rows_in_window"] += len(raw)
    parts = []
    for variable in variables.by_source("sofa"):
        sub = raw[["stay_id", "endtime", variable.column]].rename(
            columns={"endtime": "time", variable.column: "value"}
        )
        sub = sub[sub["value"].notna()]  # a component not scored that hour is unobserved
        ok = _plausible(variable, sub["value"].to_numpy())
        dropped[variable.name] = dropped.get(variable.name, 0) + int((~ok).sum())
        counts["dropped_implausible"] += int((~ok).sum())
        sub = sub[ok].assign(channel=variable.name, storetime=pd.NaT)
        parts.append(sub[MEASUREMENT_COLUMNS])
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    counts["accepted"] += len(out)
    return out


def _provenance(
    measurements: pd.DataFrame,
    intervals: pd.DataFrame,
    variables: VariableSet,
    cohort: pd.DataFrame,
    t0: pd.Series,
) -> pd.DataFrame:
    """Latest event time and storetime used per stay and source (intervals clipped at t0)."""
    prov = pd.DataFrame(index=cohort.index.copy())
    source_of = {v.name: v.source for v in variables.variables}
    if not measurements.empty:
        by_source = measurements.assign(source=measurements["channel"].map(source_of))
        for source, group in by_source.groupby("source"):
            prov[f"{source}_time"] = group.groupby("stay_id")["time"].max().reindex(prov.index)
            if source in EVENT_SOURCES:
                prov[f"{source}_storetime"] = group.groupby("stay_id")["storetime"].max().reindex(prov.index)
    if not intervals.empty:
        clipped_end = np.minimum(intervals["endtime"].to_numpy(), intervals["stay_id"].map(t0).to_numpy())
        used = pd.Series(clipped_end, index=intervals.index)
        prov["norepinephrine_equivalent_dose_time"] = (
            used.groupby(intervals["stay_id"]).max().reindex(prov.index)
        )
    return prov


def _batches(index: pd.Index, size: int | None) -> Iterable[pd.Index]:
    if not size or size >= len(index):
        yield index
        return
    for start in range(0, len(index), size):
        yield index[start : start + size]


def extract(
    source: Any, cohort: pd.DataFrame, variables: VariableSet, hours: int, batch_size: int | None = None
) -> Extraction:
    """Accepted measurements and intervals for a cohort, processed in batches of stays.

    Args:
        source: :class:`~fate_icu.data.timeseries.sources.SqlEventSource` or
            :class:`~fate_icu.data.timeseries.sources.FrameEventSource`.
        cohort: Index ``stay_id``; columns ``subject_id`` and ``intime``.
        variables: The channel definitions.
        hours: The window W.
        batch_size: Stays per query (bounds memory on the full database).

    Returns:
        The extraction, with counts and provenance.
    """
    cohort = cohort.sort_index()
    t0 = _t0(cohort, hours)
    tables = [t for t in EVENT_SOURCES if variables.by_source(t)]
    counts = {t: dict.fromkeys(COUNT_KEYS, 0) for t in [*tables, "norepinephrine_equivalent_dose", "sofa"]}
    dropped: dict[str, int] = {}
    measurement_parts, interval_parts = [], []
    ne_vars = variables.by_source("norepinephrine_equivalent_dose")
    for batch in _batches(cohort.index, batch_size):
        part = cohort.loc[batch]
        for table in tables:
            raw = source.events(table, part, variables.itemids(table))
            measurement_parts.append(_process_raw(table, raw, variables, t0, counts[table], dropped))
        for variable in ne_vars:
            interval_parts.append(
                _process_intervals(
                    source.intervals(part), variable, counts["norepinephrine_equivalent_dose"], dropped
                )
            )
        if variables.by_source("sofa"):
            measurement_parts.append(_process_sofa(source.sofa(part), variables, counts["sofa"], dropped))

    measurements = (
        pd.concat([p for p in measurement_parts if not p.empty], ignore_index=True)
        if any(not p.empty for p in measurement_parts)
        else pd.DataFrame(columns=MEASUREMENT_COLUMNS)
    )
    intervals = (
        pd.concat([p for p in interval_parts if not p.empty], ignore_index=True)
        if any(not p.empty for p in interval_parts)
        else pd.DataFrame(columns=INTERVAL_COLUMNS)
    )
    measurements = measurements.astype({"stay_id": "int64", "value": "float64"})
    measurements["time"] = pd.to_datetime(measurements["time"])
    measurements["storetime"] = pd.to_datetime(measurements["storetime"])
    measurements = measurements.sort_values(["stay_id", "channel", "time", "value"], kind="mergesort")
    intervals = intervals.astype({"stay_id": "int64", "value": "float64"})
    intervals = intervals.sort_values(
        ["stay_id", "channel", "starttime", "endtime", "value"], kind="mergesort"
    )
    return Extraction(
        measurements=measurements.reset_index(drop=True),
        intervals=intervals.reset_index(drop=True),
        counts=counts,
        dropped_by_channel=dropped,
        provenance=_provenance(measurements, intervals, variables, cohort, t0),
    )


def assert_no_future(extraction: Extraction, cohort: pd.DataFrame, hours: int) -> None:
    """Raise if any provenance time or storetime is after t0 (run on every build)."""
    t0 = _t0(cohort.sort_index(), hours)
    for column in extraction.provenance.columns:
        used = extraction.provenance[column].dropna()
        late = used > t0.loc[used.index]
        if late.any():
            raise RuntimeError(f"time-series input after t0 in {column} for {int(late.sum())} stays")
