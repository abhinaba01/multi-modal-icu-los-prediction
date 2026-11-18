"""Predictive metrics with patient-level bootstrap confidence intervals (Stage 3).

Classification: AUROC, AUPRC, Brier score and expected calibration error (ECE). Regression:
MAE and RMSE in days. The bootstrap resamples *patients*, not stays, so a patient with
several stays counts once per draw (CLAUDE.md A4: patient-level evaluation). A resample that
holds a single class has no AUROC or AUPRC; it is skipped and counted, never silently
replaced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ECE_BINS = 10


def _check_binary(y: np.ndarray, name: str) -> None:
    if len(np.unique(y)) < 2:
        raise ValueError(f"{name} needs both classes in the labels")


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    _check_binary(y, "AUROC")
    return float(roc_auc_score(y, p))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    _check_binary(y, "AUPRC")
    return float(average_precision_score(y, p))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def ece(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    """Expected calibration error: the weighted gap between confidence and accuracy per bin.

    Equal-width bins on [0, 1]; a probability of exactly 1 falls in the last bin.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        in_bin = index == b
        if in_bin.any():
            total += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(total)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))))


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)) ** 2)))


CLASSIFICATION = {"auroc": auroc, "auprc": auprc, "brier": brier, "ece": ece}
REGRESSION = {"mae_days": mae, "rmse_days": rmse}


def patient_bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    score: np.ndarray,
    patients: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 666,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Point estimate and percentile CI of ``metric`` with patients as the resampling unit.

    Returns:
        ``value``, ``low``, ``high``, the number of ``draws`` used and ``skipped`` (resamples
        for which the metric is undefined, e.g. one class only). ``low`` and ``high`` are NaN
        when no draw was usable.
    """
    y, score, patients = np.asarray(y), np.asarray(score), np.asarray(patients)
    if not len(y) == len(score) == len(patients):
        raise ValueError("y, score and patients must have the same length")
    value = metric(y, score)
    unique, inverse = np.unique(patients, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(len(unique))]
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    skipped = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, len(unique), size=len(unique))
        rows = np.concatenate([rows_of[k] for k in chosen])
        try:
            draws.append(metric(y[rows], score[rows]))
        except ValueError:
            skipped += 1
    if draws:
        low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    else:
        low = high = float("nan")
    return {"value": value, "low": float(low), "high": float(high), "draws": len(draws), "skipped": skipped}


def evaluate_population(
    y_long: np.ndarray,
    p_long: np.ndarray,
    los_days: np.ndarray,
    pred_los_days: np.ndarray,
    patients: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 666,
    classification: Mapping[str, Callable] = CLASSIFICATION,
    regression: Mapping[str, Callable] = REGRESSION,
) -> dict[str, dict[str, float | int]]:
    """Every metric with a patient-level CI for one population of stays."""
    out: dict[str, dict[str, float | int]] = {}
    for name, fn in classification.items():
        out[name] = patient_bootstrap(fn, y_long, p_long, patients, n_boot=n_boot, seed=seed)
    for name, fn in regression.items():
        out[name] = patient_bootstrap(
            lambda a, b, fn=fn: fn(a, b), los_days, pred_los_days, patients, n_boot=n_boot, seed=seed
        )
    return out
