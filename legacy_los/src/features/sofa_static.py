"""Static (worst-in-24h) SOFA score, ported from ``sql/06_extract_sofa.sql``.

This module (a) collects the raw physiological inputs SOFA needs from the raw
MIMIC-IV event CSVs -- ``collect_sofa_events`` is shared with the windowed
extractor -- and (b) scores the six SOFA subsystems using the *exact* thresholds
and item ids of the reference SQL (``sql/06``), including the urine-output
criterion in the renal score.

Component scores default to 0 when a subsystem was not measured (standard
clinical convention: absence of vasopressors / dysfunction = 0), matching the
final ``COALESCE(..., 0)`` in ``sql/06``. A quality guard flags any component
that carries no signal across the cohort (the symptom of a missing source
table), so the SOFA scripts fail loudly rather than train on constants.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.data.loader import MIMICLoader

LOGGER = logging.getLogger(__name__)

# --- item ids (mirror sql/06 & sql/07) --------------------------------------
# chartevents itemids needed by SOFA and their role.
CHART_SPO2 = 220277
CHART_FIO2 = 223835
CHART_MAP = {220052, 220181, 225312}
CHART_GCS = {220739: "gcs_eye", 223900: "gcs_verbal", 223901: "gcs_motor"}
CHART_ITEMIDS = {CHART_SPO2, CHART_FIO2, *CHART_MAP, *CHART_GCS}

# labevents itemids: PaO2, bilirubin, platelets, creatinine.
LAB_PAO2 = 50821
LAB_BILIRUBIN = 50885
LAB_PLATELETS = 51265
LAB_CREATININE = 50912
LAB_ITEMIDS = {LAB_PAO2, LAB_BILIRUBIN, LAB_PLATELETS, LAB_CREATININE}

# inputevents vasopressor itemids by drug.
VASO_NOREPI = {221906, 30047, 30120}
VASO_EPI = {221289, 30044}
VASO_DOPA = {221662, 30043}
VASO_DOBU = {221653, 30042}
VASO_VASOPRESSIN = {222315}
VASO_ITEMIDS = VASO_NOREPI | VASO_EPI | VASO_DOPA | VASO_DOBU | VASO_VASOPRESSIN

# outputevents urine itemids (mirror sql/06 renal urine list).
URINE_ITEMIDS = {226559, 226560, 226561, 226584, 226563, 226564, 226565, 226567,
                 226557, 226558, 227488, 227489}

SOFA_COMPONENT_COLUMNS = ["sofa_resp", "sofa_cardio", "sofa_hepatic", "sofa_coag", "sofa_renal", "sofa_neuro"]
SOFA_STATIC_COLUMNS = [*SOFA_COMPONENT_COLUMNS, "sofa_total", "urine_24h", "bilirubin_max"]
SOFA_RAW_SIGNALS = ["urine_24h", "bilirubin_max"]


def collect_sofa_events(loader: MIMICLoader, config: dict[str, Any], cohort_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Stream the four raw event tables once and return the in-window SOFA inputs.

    Shared by the static and windowed SOFA builders so the big tables are read at
    most once per SOFA run. The observation window is ``[intime, intime+24h)`` and
    every returned frame carries an ``hours`` column (hours since ICU ``intime``)
    so the windowed builder can derive 6h windows.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        cohort_df: Cohort with ``stay_id``, ``hadm_id`` and ``intime`` (datetime).

    Returns:
        Dict with keys ``chart``, ``lab``, ``input``, ``output`` -- filtered,
        cohort-joined event frames.
    """
    hours_window = int(config["observation_window_hours"])
    stay_keys = cohort_df[["stay_id", "intime"]].copy()
    stay_keys["stay_id"] = pd.to_numeric(stay_keys["stay_id"], errors="coerce").astype("Int64")
    stay_keys["intime"] = pd.to_datetime(stay_keys["intime"], errors="coerce")
    hadm_keys = cohort_df[["hadm_id", "stay_id", "intime"]].copy()
    hadm_keys["hadm_id"] = pd.to_numeric(hadm_keys["hadm_id"], errors="coerce").astype("Int64")
    hadm_keys["intime"] = pd.to_datetime(hadm_keys["intime"], errors="coerce")

    chart = _stream_events(loader, "chartevents", CHART_ITEMIDS, stay_keys, "stay_id",
                           ["stay_id", "itemid", "charttime", "valuenum"], "charttime", "valuenum", hours_window)
    lab = _stream_events(loader, "labevents", LAB_ITEMIDS, hadm_keys, "hadm_id",
                         ["hadm_id", "itemid", "charttime", "valuenum"], "charttime", "valuenum", hours_window)
    inputs = _stream_events(loader, "inputevents", VASO_ITEMIDS, stay_keys, "stay_id",
                            ["stay_id", "itemid", "starttime", "rate"], "starttime", "rate", hours_window)
    outputs = _stream_events(loader, "outputevents", URINE_ITEMIDS, stay_keys, "stay_id",
                             ["stay_id", "itemid", "charttime", "value"], "charttime", "value", hours_window)
    return {"chart": chart, "lab": lab, "input": inputs, "output": outputs}


def _stream_events(loader, table, itemids, keys, join_key, usecols, time_col, value_col, hours_window):
    """Stream one event table, keep in-window rows for the requested itemids.

    Args:
        loader: MIMIC loader.
        table: Raw table key.
        itemids: Itemids to keep.
        keys: Cohort key frame containing ``join_key``, ``stay_id`` and ``intime``.
        join_key: ``stay_id`` or ``hadm_id``.
        usecols: Columns to read (must include join_key, itemid, time_col, value_col).
        time_col: Event time column.
        value_col: Value column (``valuenum``, ``rate`` or ``value``).
        hours_window: Observation window in hours.

    Returns:
        Concatenated in-window events with columns ``stay_id, itemid, <value as
        'value'>, charttime, hours``.
    """
    parts: list[pd.DataFrame] = []
    key_cols = ["stay_id", "intime"] if join_key == "stay_id" else [join_key, "stay_id", "intime"]
    keys = keys[key_cols]
    for chunk in loader.iter_chunks(table, usecols=usecols):
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue
        chunk = chunk.copy()
        chunk[value_col] = pd.to_numeric(chunk[value_col], errors="coerce")
        chunk[join_key] = pd.to_numeric(chunk[join_key], errors="coerce").astype("Int64")
        chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
        merged = chunk.merge(keys, on=join_key, how="inner")
        hours = (merged[time_col] - merged["intime"]).dt.total_seconds() / 3600.0
        in_window = (hours >= 0) & (hours < hours_window)
        merged = merged[in_window]
        if merged.empty:
            continue
        out = pd.DataFrame({
            "stay_id": merged["stay_id"].to_numpy(),
            "itemid": merged["itemid"].to_numpy(),
            "value": merged[value_col].to_numpy(),
            "charttime": merged[time_col].to_numpy(),
            "hours": hours[in_window].to_numpy(),
        })
        parts.append(out)
    if not parts:
        return pd.DataFrame(columns=["stay_id", "itemid", "value", "charttime", "hours"])
    return pd.concat(parts, ignore_index=True)


# --- static scoring (mirrors sql/06 CASE thresholds) -------------------------

def build_sofa_static(events: dict[str, pd.DataFrame], cohort_stay_ids: pd.Index | pd.Series,
                      strict: bool = True) -> pd.DataFrame:
    """Compute the worst-in-24h SOFA components + total for the cohort.

    Args:
        events: Output of :func:`collect_sofa_events`.
        cohort_stay_ids: Universe of stay ids to reindex onto (missing -> 0).
        strict: Raise if any component is degenerate (see :func:`check_sofa_quality`).

    Returns:
        DataFrame indexed by ``stay_id`` with :data:`SOFA_STATIC_COLUMNS`.
    """
    ids = pd.Index(pd.unique(pd.Series(cohort_stay_ids).astype("int64")), name="stay_id")
    chart, lab, inp, out = events["chart"], events["lab"], events["input"], events["output"]

    # Respiratory inputs
    pao2 = _agg(lab[lab["itemid"] == LAB_PAO2], "min")
    spo2 = _agg(chart[(chart["itemid"] == CHART_SPO2) & chart["value"].between(50, 100)], "min")
    fio2 = _agg(chart[chart["itemid"] == CHART_FIO2], "max")

    # Cardiovascular inputs
    rate_norepi = _agg(inp[inp["itemid"].isin(VASO_NOREPI)], "max")
    rate_epi = _agg(inp[inp["itemid"].isin(VASO_EPI)], "max")
    rate_dopa = _agg(inp[inp["itemid"].isin(VASO_DOPA)], "max")
    on_vaso = (_count(inp[inp["itemid"].isin(VASO_VASOPRESSIN)]) > 0).astype(int)
    map_mean = _agg(inp_map(chart), "mean")

    # Hepatic / coagulation / renal inputs
    bilirubin_max = _agg(lab[lab["itemid"] == LAB_BILIRUBIN], "max")
    platelets_min = _agg(lab[lab["itemid"] == LAB_PLATELETS], "min")
    creatinine_max = _agg(lab[(lab["itemid"] == LAB_CREATININE) & lab["value"].between(0.1, 20)], "max")
    urine_24h = _agg(out, "sum")

    # Neurological input: worst (min) total GCS from co-timed eye+verbal+motor.
    gcs_min = _gcs_min(chart)

    frame = pd.DataFrame(index=ids)
    frame["sofa_resp"] = _score_resp(ids, pao2, spo2, fio2)
    frame["sofa_cardio"] = _score_cardio(ids, rate_norepi, rate_epi, rate_dopa, on_vaso, map_mean)
    frame["sofa_hepatic"] = _select_desc(ids, bilirubin_max, [(12.0, 4), (6.0, 3), (2.0, 2), (1.2, 1)], ascending_bad=True)
    frame["sofa_coag"] = _select_desc(ids, platelets_min, [(20, 4), (50, 3), (100, 2), (150, 1)], ascending_bad=False)
    frame["sofa_renal"] = _score_renal(ids, creatinine_max, urine_24h)
    frame["sofa_neuro"] = _select_desc(ids, gcs_min, [(6, 4), (10, 3), (13, 2), (15, 1)], ascending_bad=False)
    frame["sofa_total"] = frame[SOFA_COMPONENT_COLUMNS].sum(axis=1)
    frame["urine_24h"] = urine_24h.reindex(ids)
    frame["bilirubin_max"] = bilirubin_max.reindex(ids)

    # Quality guard runs on the un-filled raw signals so it can still see an
    # entirely-missing source table; only then do we 0-fill the two raw signals
    # so they never introduce NA (which would drop rows in the complete-case step
    # and desynchronise the ablation cohorts).
    check_sofa_quality(frame, strict=strict)
    frame["urine_24h"] = frame["urine_24h"].fillna(0.0)
    frame["bilirubin_max"] = frame["bilirubin_max"].fillna(0.0)
    return frame[SOFA_STATIC_COLUMNS]


def inp_map(chart: pd.DataFrame) -> pd.DataFrame:
    """Return MAP chart rows filtered to a plausible range (0-300).

    Args:
        chart: Collected chartevents.

    Returns:
        MAP rows with ``value`` in [0, 300].
    """
    return chart[chart["itemid"].isin(CHART_MAP) & chart["value"].between(0, 300)]


def _agg(df: pd.DataFrame, how: str) -> pd.Series:
    """Aggregate ``value`` per stay by the named reducer.

    Args:
        df: Event rows with ``stay_id`` and ``value``.
        how: One of ``min``, ``max``, ``mean``, ``sum``.

    Returns:
        Series indexed by ``stay_id``.
    """
    if df.empty:
        return pd.Series(dtype=float, name="value")
    return df.groupby("stay_id")["value"].agg(how)


def _count(df: pd.DataFrame) -> pd.Series:
    """Count rows per stay (used for the binary vasopressin flag).

    Args:
        df: Event rows with ``stay_id``.

    Returns:
        Series of counts indexed by ``stay_id``.
    """
    if df.empty:
        return pd.Series(dtype=int)
    return df.groupby("stay_id")["value"].size()


def _gcs_min(chart: pd.DataFrame, group_cols: list[str] | None = None) -> pd.Series:
    """Worst (minimum) total GCS from co-charted eye+verbal+motor scores.

    GCS components are charted together at the same ``charttime`` in MIMIC-IV; we
    pivot on ``charttime`` (per stay), require all three present, sum, and take the
    minimum. This reproduces the intent of the ``sql/06`` co-timed join.

    Args:
        chart: Collected chartevents.
        group_cols: Extra grouping columns (e.g. ``['win']`` for the windowed
            variant). Defaults to ``[]`` (whole 24h).

    Returns:
        Series of minimum total GCS indexed by ``stay_id`` (plus any group cols).
    """
    group_cols = group_cols or []
    gcs = chart[chart["itemid"].isin(CHART_GCS)]
    if gcs.empty:
        return pd.Series(dtype=float)
    gcs = gcs.assign(component=gcs["itemid"].map(CHART_GCS))
    index_cols = ["stay_id", *group_cols, "charttime"]
    per_time = gcs.pivot_table(index=index_cols, columns="component", values="value", aggfunc="mean")
    needed = ["gcs_eye", "gcs_verbal", "gcs_motor"]
    per_time = per_time.dropna(subset=[c for c in needed if c in per_time.columns])
    if per_time.empty or not all(c in per_time.columns for c in needed):
        return pd.Series(dtype=float)
    total = per_time[needed].sum(axis=1)
    return total.groupby(level=["stay_id", *group_cols]).min()


def _reindex(series: pd.Series, ids: pd.Index) -> np.ndarray:
    """Reindex a per-stay series onto ``ids`` as a float array (NaN where absent).

    Args:
        series: Per-stay values.
        ids: Target stay-id index.

    Returns:
        Float numpy array aligned to ``ids``.
    """
    return series.reindex(ids).to_numpy(dtype=float)


def _score_resp(ids, pao2, spo2, fio2) -> np.ndarray:
    """Respiratory sub-score from PaO2/FiO2 (SpO2/FiO2 fallback), mirrors sql/06.

    Args:
        ids: Stay-id index.
        pao2: Worst PaO2 per stay.
        spo2: Worst SpO2 per stay.
        fio2: Highest FiO2 per stay (percent or fraction).

    Returns:
        Integer sub-score array (null P/F -> 0).
    """
    pao2, spo2, fio2 = _reindex(pao2, ids), _reindex(spo2, ids), _reindex(fio2, ids)
    fio2_frac = np.where(np.isnan(fio2), np.nan, np.where(fio2 > 1.0, fio2 / 100.0, fio2))
    fio2_frac = np.where(fio2_frac == 0, np.nan, fio2_frac)
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = np.where(~np.isnan(pao2), pao2 / fio2_frac,
                      np.where((~np.isnan(spo2)) & (spo2 <= 97), ((spo2 / fio2_frac) - 64.0) / 0.84, np.nan))
    conditions = [pf < 100, pf < 200, pf < 300, pf < 400]
    return np.select(conditions, [4, 3, 2, 1], default=0).astype(int)


def _score_cardio(ids, norepi, epi, dopa, on_vaso, map_mean) -> np.ndarray:
    """Cardiovascular sub-score from vasopressors + MAP, mirrors sql/06 (null=false).

    Args:
        ids: Stay-id index.
        norepi, epi, dopa: Max infusion rates per stay.
        on_vaso: Binary vasopressin flag per stay.
        map_mean: Mean MAP per stay.

    Returns:
        Integer sub-score array.
    """
    norepi, epi, dopa = _reindex(norepi, ids), _reindex(epi, ids), _reindex(dopa, ids)
    on_vaso = on_vaso.reindex(ids).fillna(0).to_numpy()
    map_mean = _reindex(map_mean, ids)
    tier4 = (dopa > 15) | (epi > 0.1) | (norepi > 0.1)
    tier3 = (dopa > 5) | (epi <= 0.1) | (norepi <= 0.1)
    tier2 = (dopa <= 5) | (on_vaso == 1)
    tier1 = map_mean < 70
    return np.select([tier4, tier3, tier2, tier1], [4, 3, 2, 1], default=0).astype(int)


def _score_renal(ids, creatinine_max, urine_24h) -> np.ndarray:
    """Renal sub-score from creatinine and 24h urine output, mirrors sql/06.

    Args:
        ids: Stay-id index.
        creatinine_max: Worst creatinine per stay.
        urine_24h: 24h urine total per stay.

    Returns:
        Integer sub-score array.
    """
    cr = _reindex(creatinine_max, ids)
    ur = _reindex(urine_24h, ids)
    tier4 = (cr >= 5.0) | (ur < 200)
    tier3 = (cr >= 3.5) | (ur < 500)
    tier2 = cr >= 2.0
    tier1 = cr >= 1.2
    return np.select([tier4, tier3, tier2, tier1], [4, 3, 2, 1], default=0).astype(int)


def _select_desc(ids, series, breakpoints, ascending_bad) -> np.ndarray:
    """Threshold a per-stay value onto an integer sub-score (null -> 0).

    Args:
        ids: Stay-id index.
        series: Per-stay input value.
        breakpoints: ``(cutoff, points)`` in descending severity.
        ascending_bad: If True higher is worse (``>= cutoff``); else lower is
            worse (``< cutoff``).

    Returns:
        Integer sub-score array.
    """
    values = _reindex(series, ids)
    conditions = [(values >= cutoff if ascending_bad else values < cutoff) for cutoff, _ in breakpoints]
    choices = [points for _, points in breakpoints]
    return np.select(conditions, choices, default=0).astype(int)


def check_sofa_quality(sofa: pd.DataFrame, strict: bool) -> None:
    """Flag SOFA components/raw signals that carry no usable signal.

    Args:
        sofa: SOFA feature matrix.
        strict: Raise on dead features when True; otherwise warn.

    Raises:
        ValueError: If ``strict`` and any component is constant/all-null or a raw
            signal is entirely missing (symptom of an absent source table).
    """
    dead: list[str] = []
    for column in SOFA_COMPONENT_COLUMNS:
        values = pd.to_numeric(sofa[column], errors="coerce")
        if values.notna().sum() == 0:
            dead.append(f"{column} (all null)")
        elif values.nunique(dropna=True) <= 1:
            dead.append(f"{column} (constant = {values.dropna().iloc[0]})")
    for column in SOFA_RAW_SIGNALS:
        if pd.to_numeric(sofa[column], errors="coerce").notna().sum() == 0:
            dead.append(f"{column} (all null)")
    if not dead:
        LOGGER.info("SOFA quality check passed; all components carry signal.")
        return
    message = (
        "SOFA features carry no usable signal: " + "; ".join(dead) + ". "
        "This usually means a source table is missing/empty (cardiovascular needs "
        "inputevents; renal urine needs outputevents) or an item id matched no rows. "
        "Provide the required tables (or pass strict=False to bypass)."
    )
    if strict:
        raise ValueError(message)
    LOGGER.warning(message)
