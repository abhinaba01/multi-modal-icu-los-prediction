"""Tests for the ablation comparison (reuses per-repeat metrics; no refitting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ablation.compare import run_ablation_study
from src.evaluation.metrics import CLASSIFICATION_METRICS, REGRESSION_METRICS


def _stage(models, metrics, base, delta, n=6):
    """Build a per-repeat stage dict: {model -> DataFrame(rows=repeat, cols=metrics)}."""
    out = {}
    for m in models:
        data = {metric: np.full(n, base) + (delta if metric == metrics[0] else 0.0) for metric in metrics}
        out[m] = pd.DataFrame(data)
    return out


def _bundle(signature, class_delta, reg_delta, n_repeats=6):
    return {
        "classification": _stage(["A", "B"], CLASSIFICATION_METRICS, 0.7, class_delta, n_repeats),
        "regression_full": _stage(["A", "B"], REGRESSION_METRICS, 2.0, reg_delta, n_repeats),
        "regression_short": _stage(["A", "B"], REGRESSION_METRICS, 1.0, reg_delta, n_repeats),
        "meta": {"index_signature": signature, "n_repeats": n_repeats},
    }


def test_delta_is_augmented_minus_baseline():
    base = _bundle("SIG", class_delta=0.0, reg_delta=0.0)
    aug = _bundle("SIG", class_delta=0.05, reg_delta=-0.10)
    tables = run_ablation_study(base, aug)
    # first classification metric (accuracy) up by 0.05
    assert abs(tables["classification_delta"].loc["A", CLASSIFICATION_METRICS[0]] - 0.05) < 1e-9
    # first regression metric (rmse) down by 0.10
    assert abs(tables["regression_delta"].loc["A", REGRESSION_METRICS[0]] - (-0.10)) < 1e-9


def test_consistent_difference_is_significant():
    base = _bundle("SIG", class_delta=0.0, reg_delta=0.0)
    aug = _bundle("SIG", class_delta=0.05, reg_delta=0.0)
    p = run_ablation_study(base, aug)["classification_pvalue"].loc["A", CLASSIFICATION_METRICS[0]]
    assert p < 0.05  # a consistent +0.05 across repeats is significant


def test_pairing_guard_rejects_mismatched_cohorts():
    base = _bundle("SIG_A", 0.0, 0.0)
    aug = _bundle("SIG_B", 0.05, 0.0)   # different index signature
    with pytest.raises(ValueError, match="paired"):
        run_ablation_study(base, aug, "hempel", "hempel_sofa")


def test_pairing_guard_rejects_mismatched_repeats():
    base = _bundle("SIG", 0.0, 0.0, n_repeats=6)
    aug = _bundle("SIG", 0.05, 0.0, n_repeats=4)
    with pytest.raises(ValueError, match="n_repeats"):
        run_ablation_study(base, aug)
