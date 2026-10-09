"""Tests for SOFA scoring: static (sql/06 port), windowed scorer, dynamics, interactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.sofa_dynamics import build_sofa_dynamics, build_sofa_interactions, score_sofa_windows
from src.features.sofa_static import build_sofa_static


def _events_single_stay():
    """One synthetic stay with known worst-in-24h values -> known SOFA scores."""
    def row(itemid, value, hours=1.0):
        return {"stay_id": 1, "itemid": itemid, "value": value,
                "charttime": pd.Timestamp("2020-01-01") + pd.Timedelta(hours=hours), "hours": hours}

    chart = pd.DataFrame([
        row(223835, 50.0),   # FiO2 = 50%
        row(220277, 95.0),   # SpO2 = 95 (unused since PaO2 present)
        row(220052, 60.0),   # MAP = 60 (<70)
        row(220739, 1.0, 2), # GCS eye (co-timed at hours=2)
        row(223900, 1.0, 2), # GCS verbal
        row(223901, 2.0, 2), # GCS motor  -> total 4 (<6 -> neuro 4)
    ])
    lab = pd.DataFrame([
        row(50821, 90.0),    # PaO2 = 90 -> P/F = 90/0.5 = 180 -> resp 3
        row(50885, 7.0),     # bilirubin 7 -> hepatic 3
        row(51265, 30.0),    # platelets 30 -> coag 3
        row(50912, 6.0),     # creatinine 6 -> renal 4
    ])
    inp = pd.DataFrame([row(221906, 0.2)])   # norepinephrine 0.2 -> cardio 4
    out = pd.DataFrame([row(226559, 100.0)]) # urine 100 mL
    return {"chart": chart, "lab": lab, "input": inp, "output": out}


def test_build_sofa_static_scores():
    sofa = build_sofa_static(_events_single_stay(), cohort_stay_ids=[1], strict=False)
    row = sofa.loc[1]
    assert row["sofa_resp"] == 3
    assert row["sofa_cardio"] == 4
    assert row["sofa_hepatic"] == 3
    assert row["sofa_coag"] == 3
    assert row["sofa_renal"] == 4
    assert row["sofa_neuro"] == 4
    assert row["sofa_total"] == 21
    assert row["urine_24h"] == 100.0
    assert row["bilirubin_max"] == 7.0


def test_build_sofa_static_missing_stay_defaults_to_zero():
    sofa = build_sofa_static(_events_single_stay(), cohort_stay_ids=[1, 999], strict=False)
    assert sofa.loc[999, "sofa_total"] == 0
    assert sofa.loc[999, "urine_24h"] == 0.0


def test_score_sofa_windows_thresholds():
    raw = pd.DataFrame([{
        "stay_id": 1, "win": 0,
        "pao2": 90.0, "spo2": np.nan, "fio2_chart": 50.0,
        "rate_norepi": 0.2, "rate_epi": np.nan, "rate_dopa": np.nan, "on_vaso": np.nan,
        "map_mean": 60.0, "bilirubin_max": 7.0, "platelets_min": 30.0,
        "creatinine_max": 6.0, "gcs_min": 4.0,
    }])
    scored = score_sofa_windows(raw).iloc[0]
    assert scored["sofa_resp"] == 3
    assert scored["sofa_cardio"] == 4
    assert scored["sofa_hepatic"] == 3
    assert scored["sofa_coag"] == 3
    assert scored["sofa_renal"] == 4
    assert scored["sofa_neuro"] == 4


def test_score_sofa_windows_preserves_nan_when_unmeasured():
    raw = pd.DataFrame([{"stay_id": 1, "win": 0}])
    scored = score_sofa_windows(raw).iloc[0]
    # Nothing measured -> all components NaN (so downstream LOCF can distinguish).
    assert scored[["sofa_resp", "sofa_cardio", "sofa_hepatic", "sofa_coag", "sofa_renal", "sofa_neuro"]].isna().all()


def test_build_sofa_dynamics_trajectory_and_onset():
    # sofa_resp rises 0->1->2->3 across four windows.
    windows = pd.DataFrame([
        {"stay_id": 1, "win": w, "sofa_resp": float(w),
         "sofa_cardio": 0.0, "sofa_hepatic": 0.0, "sofa_coag": 0.0, "sofa_renal": 0.0, "sofa_neuro": 0.0}
        for w in range(4)
    ])
    dyn = build_sofa_dynamics(windows, cohort_stay_ids=[1], n_windows=4).loc[1]
    assert dyn["sofa_resp_delta"] == 3.0
    assert abs(dyn["sofa_resp_slope"] - 1.0) < 1e-9
    assert dyn["sofa_resp_onset_win"] == 1.0     # first win with score >= 1
    assert dyn["sofa_n_failed_organs"] == 1.0
    assert dyn["sofa_first_fail_resp"] == 1
    assert dyn["sofa_total_delta"] == 3.0


def test_build_sofa_interactions_products():
    static = pd.DataFrame({
        "sofa_resp": [2], "sofa_cardio": [3], "sofa_hepatic": [1],
        "sofa_coag": [0], "sofa_renal": [4], "sofa_neuro": [1],
    }, index=pd.Index([1], name="stay_id"))
    inter = build_sofa_interactions(static).loc[1]
    assert inter["sofa_resp_x_cardio"] == 6
    assert inter["sofa_cardio_x_renal"] == 12
    assert inter["sofa_renal_x_hepatic"] == 4
