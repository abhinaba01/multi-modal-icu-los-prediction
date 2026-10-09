"""Dynamic SOFA features: trajectory/velocity, organ-failure onset sequence, and
sub-score interaction terms -- the three representations added by
``run_hempel_sofa_dynamics.py`` on top of the Hempel baseline.

The three feature groups (exactly the three the user specified):

1. **Trajectory / velocity** -- delta and OLS slope of each subsystem score and of
   the SOFA total across the first-24h 6h windows (deteriorating vs stabilising).
2. **Organ-failure onset sequence** -- the window in which each organ first crosses
   its dysfunction threshold, which organ(s) failed first, count of failed organs,
   and time-to-first-failure (encodes e.g. renal-first vs respiratory-first).
3. **Sub-score interactions** -- products of clinically coupled organ sub-scores
   (cardiopulmonary, cardiorenal, hepatorenal, ...) so the collapsed 0-24 total is
   decomposed into interaction terms.

The pure-Python scorer (:func:`score_sofa_windows`) mirrors the thresholds in
``sql/06``/``sofa_static.py`` exactly; the windowed renal score is creatinine-only
(urine SOFA thresholds are 24h totals). This module does only arithmetic on the
scores -- no clinical thresholds beyond the scorer -- so it is fully unit-testable
without a database. (Reused, verified, from the sibling project's
``src/features/sofa_dynamics.py``.)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# numpy>=2.0 renamed np.trapz -> np.trapezoid. Use trapezoid when present.
_trapz = getattr(np, "trapezoid", None)
if _trapz is None:  # pragma: no cover - numpy < 2.0 only
    _trapz = np.trapz

SOFA_COMPONENTS = ["resp", "cardio", "hepatic", "coag", "renal", "neuro"]

# Clinically coupled organ pairs for interaction terms.
SOFA_INTERACTION_PAIRS = [
    ("resp", "cardio"),    # cardiopulmonary coupling (shock + hypoxaemia)
    ("cardio", "renal"),   # cardiorenal syndrome
    ("renal", "hepatic"),  # hepatorenal axis
    ("neuro", "resp"),     # airway / sedation vs respiratory failure
    ("coag", "hepatic"),   # coagulopathy of liver failure
    ("resp", "renal"),     # ARDS + AKI
]

DEFAULT_N_WINDOWS = 4
DEFAULT_WINDOW_HOURS = 6
DEFAULT_ONSET_THRESHOLD = 1  # SOFA >= 1 counts as any organ dysfunction


def dynamics_feature_columns(n_windows: int = DEFAULT_N_WINDOWS) -> dict[str, list[str]]:
    """Return the dynamic-SOFA feature column names, split by scaling treatment.

    Args:
        n_windows: Number of windows (documentation only; not in column names).

    Returns:
        Mapping with ``"continuous"`` and ``"indicator"`` column lists.
    """
    continuous: list[str] = []
    for component in SOFA_COMPONENTS:
        continuous += [f"sofa_{component}_delta", f"sofa_{component}_slope", f"sofa_{component}_onset_win"]
    continuous += [
        "sofa_total_first", "sofa_total_last", "sofa_total_max", "sofa_total_delta",
        "sofa_total_slope", "sofa_total_auc", "sofa_n_failed_organs",
        "sofa_time_to_first_failure_win",
    ]
    indicator = [f"sofa_first_fail_{component}" for component in SOFA_COMPONENTS]
    return {"continuous": continuous, "indicator": indicator}


def interaction_feature_columns() -> list[str]:
    """Return the SOFA interaction-term column names.

    Returns:
        List of ``sofa_<a>_x_<b>`` product-feature names.
    """
    return [f"sofa_{a}_x_{b}" for a, b in SOFA_INTERACTION_PAIRS]


RAW_WINDOW_COLUMNS = [
    "pao2", "spo2", "fio2_chart", "rate_norepi", "rate_epi", "rate_dopa",
    "on_vaso", "map_mean", "bilirubin_max", "platelets_min", "creatinine_max", "gcs_min",
]


def _threshold_score(values: pd.Series, breakpoints: list[tuple[float, int]], ascending_bad: bool) -> pd.Series:
    """Map a numeric series onto integer SOFA sub-scores, preserving NaN.

    Args:
        values: Numeric input series.
        breakpoints: ``(cutoff, score)`` in descending score order.
        ascending_bad: If True, higher is worse (``>= cutoff``); else lower is
            worse (``< cutoff``).

    Returns:
        Integer-valued (nullable float) score series; NaN where input is NaN.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    score = pd.Series(np.nan, index=numeric.index, dtype=float)
    measured = numeric.notna()
    score[measured] = 0.0
    for cutoff, points in reversed(breakpoints):
        hit = measured & (numeric >= cutoff if ascending_bad else numeric < cutoff)
        score[hit] = float(points)
    return score


def score_sofa_windows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Score raw per-window physiological values into SOFA component sub-scores.

    Mirrors the thresholds in ``sofa_static.py``/``sql/06`` exactly. The windowed
    renal score is creatinine-only.

    Args:
        raw_df: DataFrame with ``stay_id``, ``win`` and the columns in
            :data:`RAW_WINDOW_COLUMNS` (missing columns treated as all-NaN).

    Returns:
        DataFrame with ``stay_id``, ``win`` and ``sofa_resp`` ... ``sofa_neuro``
        (NaN where the component had no measurement in that window).
    """
    frame = raw_df.copy()
    for column in RAW_WINDOW_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    out = frame[["stay_id", "win"]].copy()

    fio2_frac = frame["fio2_chart"].where(frame["fio2_chart"].isna() | (frame["fio2_chart"] <= 1.0),
                                          frame["fio2_chart"] / 100.0)
    fio2_frac = fio2_frac.replace(0.0, np.nan)
    pf = pd.Series(np.nan, index=frame.index, dtype=float)
    have_pao2 = frame["pao2"].notna() & fio2_frac.notna()
    pf[have_pao2] = frame["pao2"][have_pao2] / fio2_frac[have_pao2]
    have_spo2 = (~have_pao2) & frame["spo2"].notna() & (frame["spo2"] <= 97) & fio2_frac.notna()
    pf[have_spo2] = ((frame["spo2"][have_spo2] / fio2_frac[have_spo2]) - 64.0) / 0.84
    resp_measured = frame["fio2_chart"].notna() & (frame["pao2"].notna() | frame["spo2"].notna())
    resp = pd.Series(np.nan, index=frame.index, dtype=float)
    resp[resp_measured] = 0.0
    resp[resp_measured & (pf < 400)] = 1.0
    resp[resp_measured & (pf < 300)] = 2.0
    resp[resp_measured & (pf < 200)] = 3.0
    resp[resp_measured & (pf < 100)] = 4.0
    out["sofa_resp"] = resp

    cardio_inputs = ["rate_norepi", "rate_epi", "rate_dopa", "on_vaso", "map_mean"]
    cardio_measured = frame[cardio_inputs].notna().any(axis=1)
    dopa, epi, norepi = frame["rate_dopa"], frame["rate_epi"], frame["rate_norepi"]
    on_vaso = frame["on_vaso"].fillna(0)
    map_mean = frame["map_mean"]
    cardio = pd.Series(np.nan, index=frame.index, dtype=float)
    cardio[cardio_measured] = 0.0
    cardio[cardio_measured & (map_mean < 70)] = 1.0
    cardio[cardio_measured & ((dopa <= 5) | (on_vaso == 1))] = 2.0
    cardio[cardio_measured & ((dopa > 5) | (epi <= 0.1) | (norepi <= 0.1))] = 3.0
    cardio[cardio_measured & ((dopa > 15) | (epi > 0.1) | (norepi > 0.1))] = 4.0
    out["sofa_cardio"] = cardio

    out["sofa_hepatic"] = _threshold_score(
        frame["bilirubin_max"], [(12.0, 4), (6.0, 3), (2.0, 2), (1.2, 1)], ascending_bad=True)
    out["sofa_coag"] = _threshold_score(
        frame["platelets_min"], [(20, 4), (50, 3), (100, 2), (150, 1)], ascending_bad=False)
    out["sofa_renal"] = _threshold_score(
        frame["creatinine_max"], [(5.0, 4), (3.5, 3), (2.0, 2), (1.2, 1)], ascending_bad=True)
    out["sofa_neuro"] = _threshold_score(
        frame["gcs_min"], [(6, 4), (10, 3), (13, 2), (15, 1)], ascending_bad=False)
    return out


def build_sofa_dynamics(
    windows_df: pd.DataFrame,
    cohort_stay_ids: pd.Index | pd.Series | None = None,
    n_windows: int = DEFAULT_N_WINDOWS,
    window_hours: int = DEFAULT_WINDOW_HOURS,
    onset_threshold: int = DEFAULT_ONSET_THRESHOLD,
) -> pd.DataFrame:
    """Compute trajectory/velocity and onset-sequence features from windowed SOFA.

    Args:
        windows_df: Long table with ``stay_id``, ``win`` and the six ``sofa_*``
            component scores (NaN where nothing measured in that window).
        cohort_stay_ids: Optional universe of stay ids to reindex onto.
        n_windows: Number of first-24h sub-windows.
        window_hours: Hours per window (interprets sentinels only).
        onset_threshold: Component score at/above which an organ counts as failed.

    Returns:
        DataFrame indexed by ``stay_id`` with the columns from
        :func:`dynamics_feature_columns`.
    """
    components = [f"sofa_{component}" for component in SOFA_COMPONENTS]
    frame = windows_df.copy()
    if "stay_id" not in frame.columns or "win" not in frame.columns:
        raise KeyError("windows_df must contain 'stay_id' and 'win' columns")
    for column in components:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce").astype("Int64")
    frame["win"] = pd.to_numeric(frame["win"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["stay_id", "win"])

    records = [
        _dynamics_for_stay(int(stay_id), group, components, n_windows, onset_threshold)
        for stay_id, group in frame.groupby("stay_id", sort=True)
    ]
    columns = dynamics_feature_columns(n_windows)["continuous"] + dynamics_feature_columns(n_windows)["indicator"]
    if records:
        result = pd.DataFrame.from_records(records).set_index("stay_id")
    else:
        result = pd.DataFrame(columns=["stay_id", *columns]).set_index("stay_id")
    result = result.reindex(columns=columns)

    if cohort_stay_ids is not None:
        result = _reindex_and_fill(result, cohort_stay_ids, n_windows)
    return result


def _dynamics_for_stay(stay_id, group, components, n_windows, onset_threshold) -> dict[str, float]:
    """Compute all dynamic features for a single stay's windowed scores.

    Args:
        stay_id: Stay identifier.
        group: Rows for this stay (one per populated window).
        components: Component score column names.
        n_windows: Full window count to reindex onto.
        onset_threshold: Failure threshold on component scores.

    Returns:
        Flat feature dictionary including ``stay_id``.
    """
    grid = (group.set_index("win")[components].reindex(range(n_windows)).apply(pd.to_numeric, errors="coerce"))
    win_index = np.arange(n_windows, dtype=float)

    row: dict[str, float] = {"stay_id": stay_id}
    total_filled = np.zeros(n_windows, dtype=float)
    first_fail_win = n_windows
    n_failed = 0

    for component in components:
        raw = grid[component].to_numpy(dtype=float)
        filled = _locf_fill(raw)
        total_filled = total_filled + filled

        short = component.replace("sofa_", "")
        row[f"sofa_{short}_delta"] = float(filled[-1] - filled[0])
        row[f"sofa_{short}_slope"] = _ols_slope(win_index, filled)

        onset = _first_crossing(raw, onset_threshold)
        row[f"sofa_{short}_onset_win"] = float(onset) if onset is not None else float(n_windows)
        if onset is not None:
            n_failed += 1
            first_fail_win = min(first_fail_win, onset)

    row["sofa_total_first"] = float(total_filled[0])
    row["sofa_total_last"] = float(total_filled[-1])
    row["sofa_total_max"] = float(total_filled.max())
    row["sofa_total_delta"] = float(total_filled[-1] - total_filled[0])
    row["sofa_total_slope"] = _ols_slope(win_index, total_filled)
    row["sofa_total_auc"] = float(_trapz(total_filled, dx=1.0))
    row["sofa_n_failed_organs"] = float(n_failed)
    row["sofa_time_to_first_failure_win"] = float(first_fail_win)

    for component in components:
        short = component.replace("sofa_", "")
        onset = _first_crossing(grid[component].to_numpy(dtype=float), onset_threshold)
        row[f"sofa_first_fail_{short}"] = int(onset is not None and onset == first_fail_win)
    return row


def _locf_fill(values: np.ndarray) -> np.ndarray:
    """Carry the last observed score forward, then fill leading gaps with 0.

    Args:
        values: 1-D array of per-window scores, possibly containing NaN.

    Returns:
        Same-length array with NaNs replaced (0 before the first observation).
    """
    out = values.astype(float).copy()
    last = 0.0
    seen = False
    for i, value in enumerate(out):
        if np.isnan(value):
            out[i] = last if seen else 0.0
        else:
            last = value
            seen = True
    return out


def _first_crossing(values: np.ndarray, threshold: int) -> int | None:
    """Return the first window index whose observed score is >= threshold.

    Args:
        values: 1-D array of per-window scores (NaN skipped).
        threshold: Crossing threshold.

    Returns:
        Zero-based window index of first crossing, or ``None``.
    """
    for i, value in enumerate(values):
        if not np.isnan(value) and value >= threshold:
            return i
    return None


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """OLS slope of ``y`` on ``x``, robust to degenerate input.

    Args:
        x: Window indices.
        y: Per-window values (already gap-filled).

    Returns:
        Slope in score-per-window, or 0.0 if it cannot be estimated.
    """
    if len(x) < 2 or np.allclose(y, y[0]):
        return 0.0
    variance = np.var(x)
    if variance == 0:
        return 0.0
    return float(np.cov(x, y, bias=True)[0, 1] / variance)


def build_sofa_interactions(sofa_static_df: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Build organ sub-score interaction (product) terms from static SOFA.

    Args:
        sofa_static_df: DataFrame indexed by ``stay_id`` (or with a ``stay_id``
            column) with ``sofa_resp`` ... ``sofa_neuro`` columns.
        pairs: Optional override of organ pairs.

    Returns:
        DataFrame indexed by ``stay_id`` with ``sofa_<a>_x_<b>`` product columns.

    Raises:
        KeyError: If a required component column is missing.
    """
    pairs = pairs or SOFA_INTERACTION_PAIRS
    frame = sofa_static_df.copy()
    if "stay_id" in frame.columns:
        frame = frame.set_index("stay_id")
    result = pd.DataFrame(index=frame.index)
    for a, b in pairs:
        col_a, col_b = f"sofa_{a}", f"sofa_{b}"
        if col_a not in frame.columns or col_b not in frame.columns:
            raise KeyError(f"interaction needs {col_a} and {col_b}; present: {list(frame.columns)}")
        result[f"sofa_{a}_x_{b}"] = pd.to_numeric(frame[col_a], errors="coerce") * pd.to_numeric(frame[col_b], errors="coerce")
    return result


def _reindex_and_fill(result: pd.DataFrame, cohort_stay_ids: pd.Index | pd.Series, n_windows: int) -> pd.DataFrame:
    """Reindex onto the full cohort and fill defaults for stays with no rows.

    Args:
        result: Feature frame indexed by ``stay_id``.
        cohort_stay_ids: Universe of stay ids.
        n_windows: Sentinel value for onset columns.

    Returns:
        Reindexed and default-filled feature frame.
    """
    ids = pd.Index(pd.unique(pd.Series(cohort_stay_ids)), name="stay_id")
    out = result.reindex(ids)
    onset_columns = [c for c in out.columns if c.endswith("_onset_win")] + ["sofa_time_to_first_failure_win"]
    for column in out.columns:
        default = float(n_windows) if column in onset_columns else 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)
    return out
