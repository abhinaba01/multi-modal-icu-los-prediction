"""Windowed SOFA raw inputs, ported from ``sql/07_extract_sofa_windows.sql``.

Emits one row per ``(stay_id, win)`` with the RAW worst-in-window physiological
values (NOT scored, NOT coalesced to 0) so the pure-Python scorer in
``sofa_dynamics.py`` can (a) apply the same thresholds as the static SOFA and
(b) tell "not measured" (NaN) apart from "measured and normal" for
last-observation-carried-forward.

Renal is creatinine-only here (urine-output SOFA thresholds are defined on 24h
totals, not meaningful over 6h windows) -- exactly as the reference SQL notes.
Consumes the shared event collection from ``sofa_static.collect_sofa_events``.
"""

from __future__ import annotations

import pandas as pd

from src.features.sofa_static import (
    CHART_FIO2, CHART_MAP, CHART_SPO2, LAB_BILIRUBIN, LAB_CREATININE,
    LAB_PAO2, LAB_PLATELETS, VASO_DOPA, VASO_EPI, VASO_NOREPI, VASO_VASOPRESSIN,
    _gcs_min, inp_map,
)

RAW_WINDOW_COLUMNS = [
    "pao2", "spo2", "fio2_chart", "rate_norepi", "rate_epi", "rate_dopa",
    "on_vaso", "map_mean", "bilirubin_max", "platelets_min", "creatinine_max", "gcs_min",
]


def build_sofa_windows_raw(events: dict[str, pd.DataFrame], n_windows: int = 4, window_hours: int = 6) -> pd.DataFrame:
    """Build the long per-(stay, window) raw SOFA-input table.

    Args:
        events: Output of ``sofa_static.collect_sofa_events``.
        n_windows: Number of first-24h sub-windows.
        window_hours: Hours per window (``n_windows * window_hours`` should be 24).

    Returns:
        Long DataFrame with ``stay_id``, ``win`` and the columns in
        :data:`RAW_WINDOW_COLUMNS` (NaN where a component was not measured).
    """
    chart = _with_win(events["chart"], window_hours, n_windows)
    lab = _with_win(events["lab"], window_hours, n_windows)
    inp = _with_win(events["input"], window_hours, n_windows)

    pieces = [
        _agg_win(lab[lab["itemid"] == LAB_PAO2], "min", "pao2"),
        _agg_win(chart[(chart["itemid"] == CHART_SPO2) & chart["value"].between(50, 100)], "min", "spo2"),
        _agg_win(chart[chart["itemid"] == CHART_FIO2], "max", "fio2_chart"),
        _agg_win(inp[inp["itemid"].isin(VASO_NOREPI)], "max", "rate_norepi"),
        _agg_win(inp[inp["itemid"].isin(VASO_EPI)], "max", "rate_epi"),
        _agg_win(inp[inp["itemid"].isin(VASO_DOPA)], "max", "rate_dopa"),
        _on_vaso_win(inp[inp["itemid"].isin(VASO_VASOPRESSIN)]),
        _agg_win(_with_win(inp_map(events["chart"]), window_hours, n_windows), "mean", "map_mean"),
        _agg_win(lab[lab["itemid"] == LAB_BILIRUBIN], "max", "bilirubin_max"),
        _agg_win(lab[lab["itemid"] == LAB_PLATELETS], "min", "platelets_min"),
        _agg_win(lab[(lab["itemid"] == LAB_CREATININE) & lab["value"].between(0.1, 20)], "max", "creatinine_max"),
        _gcs_min_win(chart),
    ]

    result: pd.DataFrame | None = None
    for piece in pieces:
        if piece is None or piece.empty:
            continue
        result = piece if result is None else result.merge(piece, on=["stay_id", "win"], how="outer")

    if result is None:
        return pd.DataFrame(columns=["stay_id", "win", *RAW_WINDOW_COLUMNS])
    for column in RAW_WINDOW_COLUMNS:
        if column not in result.columns:
            result[column] = pd.NA
    return result[["stay_id", "win", *RAW_WINDOW_COLUMNS]].sort_values(["stay_id", "win"]).reset_index(drop=True)


def _with_win(df: pd.DataFrame, window_hours: int, n_windows: int) -> pd.DataFrame:
    """Attach a 0-based ``win`` index and drop rows beyond the last window.

    Args:
        df: Event frame with an ``hours`` column.
        window_hours: Hours per window.
        n_windows: Number of windows.

    Returns:
        Copy with an integer ``win`` column, restricted to ``win < n_windows``.
    """
    if df.empty:
        return df.assign(win=pd.Series(dtype=int))
    out = df.copy()
    out["win"] = (out["hours"] // window_hours).astype(int)
    return out[out["win"] < n_windows]


def _agg_win(df: pd.DataFrame, how: str, name: str) -> pd.DataFrame | None:
    """Aggregate ``value`` per ``(stay_id, win)`` into a named column.

    Args:
        df: Windowed event frame.
        how: Reducer name.
        name: Output column name.

    Returns:
        DataFrame ``[stay_id, win, name]`` or ``None`` if empty.
    """
    if df.empty:
        return None
    grouped = df.groupby(["stay_id", "win"])["value"].agg(how).rename(name).reset_index()
    return grouped


def _on_vaso_win(df: pd.DataFrame) -> pd.DataFrame | None:
    """Per-window binary vasopressin flag.

    Args:
        df: Windowed vasopressin event frame.

    Returns:
        DataFrame ``[stay_id, win, on_vaso]`` or ``None`` if empty.
    """
    if df.empty:
        return None
    grouped = df.groupby(["stay_id", "win"])["value"].size().rename("on_vaso").reset_index()
    grouped["on_vaso"] = (grouped["on_vaso"] > 0).astype(int)
    return grouped


def _gcs_min_win(chart: pd.DataFrame) -> pd.DataFrame | None:
    """Per-window worst total GCS.

    Args:
        chart: Windowed chartevents (must carry a ``win`` column).

    Returns:
        DataFrame ``[stay_id, win, gcs_min]`` or ``None`` if unavailable.
    """
    series = _gcs_min(chart, group_cols=["win"])
    if series.empty:
        return None
    return series.rename("gcs_min").reset_index()
