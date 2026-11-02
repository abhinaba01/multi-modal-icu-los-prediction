"""Aggregate-only reports for a time-series build (safe to paste back from the lab).

* missingness: share of stay-hours and of stays with an accepted value, per channel;
* comparison with the internship: the per-stay mean of the accepted measurements against
  the frozen internship feature of the same variable. The two differ where Stage 2's
  rules drop values the internship kept (stored after t0, missing storetime,
  implausible), which docs/decisions.md D-016 asks us to quantify.

Counts below 10 are suppressed (CLAUDE.md A3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fate_icu.data.timeseries.extract import Extraction
from fate_icu.data.timeseries.variables import VariableSet
from fate_icu.utils.aggregates import MIN_CELL, safe_share, suppress

# Stage 2 channel -> frozen internship feature with the same itemids and window.
LEGACY_EQUIVALENTS = {
    "heart_rate": "heart_rate",
    "resp_rate": "respiration",
    "spo2": "saturation",
    "temperature": "temperature",
    "creatinine": "creatinine",
    "bun": "urea_nitrogen",
    "glucose": "glucose",
    "sodium": "sodium",
    "potassium": "potassium",
    "haemoglobin": "hemoglobin",
    "platelets": "platelet_count",
    "wbc": "white_blood_cells",
}


def missingness(raw: np.ndarray, split: np.ndarray, variables: VariableSet) -> dict[str, Any]:
    """Observed shares per channel, overall and in train."""
    observed = ~np.isnan(raw)
    n, w, _ = raw.shape
    train = np.asarray(split) == "train"
    out = {}
    for k, variable in enumerate(variables.variables):
        cells = observed[:, :, k]
        out[variable.name] = {
            "label": variable.label,
            "observed_share_of_stay_hours": safe_share(int(cells.sum()), n * w),
            "share_of_stays_with_a_value": safe_share(int(cells.any(axis=1).sum()), n),
            "train_observed_share_of_stay_hours": safe_share(int(cells[train].sum()), int(train.sum()) * w),
        }
    return out


def legacy_comparison(extraction: Extraction, legacy_features: pd.DataFrame) -> dict[str, Any]:
    """Per-stay window means of accepted values vs the internship's 24 h mean features."""
    if extraction.measurements.empty:
        return {}
    means = extraction.measurements.groupby(["stay_id", "channel"])["value"].mean().unstack("channel")
    out: dict[str, Any] = {}
    for channel, legacy_name in LEGACY_EQUIVALENTS.items():
        if channel not in means.columns or legacy_name not in legacy_features.columns:
            continue
        both = pd.concat(
            [means[channel].rename("stage2"), legacy_features[legacy_name].rename("legacy")],
            axis=1,
            join="inner",
        ).dropna()
        n = len(both)
        different = ~np.isclose(both["stage2"], both["legacy"], rtol=1e-9, atol=1e-9)
        n_diff = int(different.sum())
        abs_diff = (both["stage2"] - both["legacy"]).abs()[different]
        out[channel] = {
            "internship_feature": legacy_name,
            "stays_compared": suppress(n),
            "share_of_stays_differing": safe_share(n_diff, n),
            "median_abs_difference_where_differing": (
                round(float(abs_diff.median()), 4) if n_diff >= MIN_CELL else f"n<{MIN_CELL}"
            ),
            "pearson_r": round(float(both["stage2"].corr(both["legacy"])), 4)
            if n >= MIN_CELL
            else f"n<{MIN_CELL}",
        }
    return out
