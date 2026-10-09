"""Chunked first-24h aggregation over the large MIMIC-IV event tables.

The core is :func:`stream_multi`: it makes ONE streaming pass over an event table
and feeds every "sink" attached to it. A sink is either

* a :class:`MeanSink` -- per-stay mean of a set of itemids (the baseline vitals /
  labs), or
* a :class:`RawSink` -- the raw in-window rows for a set of itemids (the SOFA
  inputs, later reduced with min/max/sum/windowing).

This lets ``build_features.py`` read ``chartevents`` (and ``labevents``) exactly
once while producing *both* the baseline mean features and the SOFA raw inputs --
instead of re-scanning the 3.3 GB file for each. Because each sink keeps its own
itemid set and window rule, the outputs are identical to running the sinks
separately (see ``tests/test_aggregate.py``).

:func:`chunked_windowed_mean` is a thin single-``MeanSink`` wrapper, so the
single-table build path is unchanged by construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.loader import MIMICLoader

LOGGER = logging.getLogger(__name__)


@dataclass
class MeanSink:
    """Per-stay mean of ``itemid_map`` over the window.

    Attributes:
        name: Result key.
        itemid_map: ``itemid -> output column name``.
        lower_inclusive: Include events exactly at ``intime`` (``>=`` vs ``>``).
        upper_inclusive: Include events exactly at ``intime + window`` (``<=`` vs ``<``).
    """

    name: str
    itemid_map: dict[int, str]
    lower_inclusive: bool = False   # baseline: (intime, intime+24h]
    upper_inclusive: bool = True
    itemids: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self.itemids = set(self.itemid_map)


@dataclass
class RawSink:
    """Raw in-window rows for ``itemids`` (for later SOFA reduction).

    Attributes:
        name: Result key.
        itemids: Itemids to keep.
        lower_inclusive: Include events exactly at ``intime``.
        upper_inclusive: Include events exactly at ``intime + window``.
    """

    name: str
    itemids: set[int]
    lower_inclusive: bool = True    # SOFA: [intime, intime+24h)
    upper_inclusive: bool = False


_RAW_COLUMNS = ["stay_id", "itemid", "value", "charttime", "hours"]


def stream_multi(
    loader: MIMICLoader,
    config: dict[str, Any],
    table: str,
    join_key: str,
    value_col: str,
    time_col: str,
    cohort_time_df: pd.DataFrame,
    sinks: list[MeanSink | RawSink],
    window_hours: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Stream ``table`` once and evaluate every sink against each chunk.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration (for the default window).
        table: Raw table key to stream.
        join_key: Column to join events to the cohort on (``stay_id`` / ``hadm_id``).
        value_col: Numeric value column in the table (``valuenum`` / ``rate`` / ``value``).
        time_col: Event-time column (``charttime`` / ``starttime``).
        cohort_time_df: DataFrame with ``[join_key, 'stay_id', 'intime']``.
        sinks: Sinks to evaluate (mean and/or raw); each keeps its own itemids/window.
        window_hours: Observation window; defaults to ``config['observation_window_hours']``.

    Returns:
        Mapping ``sink.name -> result`` (mean sinks give a per-stay mean frame;
        raw sinks give an in-window row frame with columns
        ``stay_id, itemid, value, charttime, hours``).
    """
    hours = window_hours or int(config["observation_window_hours"])
    all_itemids: set[int] = set()
    for sink in sinks:
        all_itemids |= sink.itemids

    key_cols = ["stay_id", "intime"] if join_key == "stay_id" else [join_key, "stay_id", "intime"]
    keys = cohort_time_df[key_cols].copy()
    keys[join_key] = pd.to_numeric(keys[join_key], errors="coerce").astype("Int64")
    keys["intime"] = pd.to_datetime(keys["intime"], errors="coerce")

    mean_parts: dict[str, list[pd.DataFrame]] = {s.name: [] for s in sinks if isinstance(s, MeanSink)}
    raw_parts: dict[str, list[pd.DataFrame]] = {s.name: [] for s in sinks if isinstance(s, RawSink)}
    usecols = list(dict.fromkeys([join_key, "itemid", time_col, value_col]))

    n_chunks = 0
    for chunk in loader.iter_chunks(table, usecols=usecols):
        n_chunks += 1
        chunk = chunk[chunk["itemid"].isin(all_itemids)]
        if chunk.empty:
            continue
        chunk = chunk[[join_key, "itemid", time_col, value_col]].copy()
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        chunk = chunk[chunk[value_col].notna()]
        if chunk.empty:
            continue
        chunk[join_key] = pd.to_numeric(chunk[join_key], errors="coerce").astype("Int64")
        chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
        merged = chunk.merge(keys, on=join_key, how="inner")
        if merged.empty:
            continue
        hrs = (merged[time_col] - merged["intime"]).dt.total_seconds() / 3600.0

        for sink in sinks:
            lower = hrs >= 0 if sink.lower_inclusive else hrs > 0
            upper = hrs <= hours if sink.upper_inclusive else hrs < hours
            selection = lower & upper & merged["itemid"].isin(sink.itemids)
            if not selection.any():
                continue
            sub = merged[selection]
            if isinstance(sink, MeanSink):
                agg = sub.groupby(["stay_id", "itemid"])[value_col].agg(["sum", "count"])
                mean_parts[sink.name].append(agg)
            else:
                raw_parts[sink.name].append(pd.DataFrame({
                    "stay_id": sub["stay_id"].to_numpy(),
                    "itemid": sub["itemid"].to_numpy(),
                    "value": sub[value_col].to_numpy(),
                    "charttime": sub[time_col].to_numpy(),
                    "hours": hrs[selection].to_numpy(),
                }))

    LOGGER.info("%s: processed %d chunks for %d sink(s)", table, n_chunks, len(sinks))

    results: dict[str, pd.DataFrame] = {}
    for sink in sinks:
        if isinstance(sink, MeanSink):
            results[sink.name] = _finalize_mean(mean_parts[sink.name], sink.itemid_map)
        else:
            parts = raw_parts[sink.name]
            results[sink.name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=_RAW_COLUMNS)
    return results


def _finalize_mean(parts: list[pd.DataFrame], itemid_map: dict[int, str]) -> pd.DataFrame:
    """Combine per-chunk (sum, count) aggregates into per-stay means.

    Args:
        parts: Per-chunk grouped ``(sum, count)`` frames indexed by (stay_id, itemid).
        itemid_map: ``itemid -> output column name``.

    Returns:
        Per-stay mean frame indexed by ``stay_id`` with the mapped, ordered columns.
    """
    if not parts:
        return pd.DataFrame(columns=list(itemid_map.values())).rename_axis("stay_id")
    totals = pd.concat(parts).groupby(level=["stay_id", "itemid"]).sum()
    means = (totals["sum"] / totals["count"]).unstack("itemid").rename(columns=itemid_map)
    ordered = [name for name in dict.fromkeys(itemid_map.values()) if name in means.columns]
    return means[ordered]


def chunked_windowed_mean(
    loader: MIMICLoader,
    config: dict[str, Any],
    table: str,
    itemid_map: dict[int, str],
    cohort_time_df: pd.DataFrame,
    join_key: str,
    usecols: list[str] | None = None,
    window_hours: int | None = None,
) -> pd.DataFrame:
    """Per-stay mean of ``itemid_map`` over ``(intime, intime+window]`` (single sink).

    Thin wrapper over :func:`stream_multi` used by the vitals/labs builders; kept
    so the single-table path is identical to the multi-sink path.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        table: Raw table key.
        itemid_map: ``itemid -> output column name``.
        cohort_time_df: DataFrame with ``[join_key, 'stay_id', 'intime']``.
        join_key: Join column.
        usecols: Ignored (retained for call-site compatibility).
        window_hours: Observation window override.

    Returns:
        Per-stay mean frame indexed by ``stay_id``.
    """
    sink = MeanSink(name="mean", itemid_map=itemid_map, lower_inclusive=False, upper_inclusive=True)
    result = stream_multi(loader, config, table, join_key, "valuenum", "charttime",
                          cohort_time_df, [sink], window_hours=window_hours)
    return result["mean"]
