"""Statistics for the evaluation harness (Stage 7).

* ``paired_patient_bootstrap``: the difference of a metric between two methods scored on the *same* stays,
  with patients as the resampling unit, so a patient with several stays counts once per draw. A draw in
  which the metric is undefined (one class only) is skipped and counted, never replaced.
* ``paired_mean_difference``: the same for per-stay numbers (e.g. the count of unattributed drivers).
* ``holm_bonferroni``: step-down correction across the pre-registered hypotheses.
* ``seed_summary``: mean and standard deviation over seeds (5 for neural models).

The p-value of the bootstrap is two-sided: twice the smaller share of draws on either side of zero, with one
added to numerator and denominator so it is never exactly 0 (``(count + 1) / (draws + 1)``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def _patient_rows(patients: np.ndarray) -> list[np.ndarray]:
    unique, inverse = np.unique(patients, return_inverse=True)
    return [np.flatnonzero(inverse == k) for k in range(len(unique))]


def _bootstrap_summary(
    difference: float, draws: list[float], skipped: int, alpha: float
) -> dict[str, float | int]:
    if draws:
        d = np.asarray(draws)
        low, high = np.quantile(d, [alpha / 2, 1 - alpha / 2])
        below, above = int((d <= 0).sum()), int((d >= 0).sum())
        p = min(1.0, 2 * (min(below, above) + 1) / (len(d) + 1))
    else:
        low = high = p = float("nan")
    return {
        "difference": float(difference),
        "low": float(low),
        "high": float(high),
        "p_value": float(p),
        "draws": len(draws),
        "skipped": skipped,
    }


def paired_patient_bootstrap(
    metric: Metric,
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    patients: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 666,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """``metric(y, score_a) - metric(y, score_b)`` with a percentile CI and a two-sided bootstrap p-value.

    Both scores must be for the same stays in the same order. Positive means that ``a`` is higher.
    """
    y, a, b, patients = np.asarray(y), np.asarray(score_a), np.asarray(score_b), np.asarray(patients)
    if not len(y) == len(a) == len(b) == len(patients):
        raise ValueError("y, both scores and patients must have the same length")
    difference = metric(y, a) - metric(y, b)
    rows_of = _patient_rows(patients)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    skipped = 0
    for _ in range(n_boot):
        chosen = rng.integers(0, len(rows_of), size=len(rows_of))
        rows = np.concatenate([rows_of[k] for k in chosen])
        try:
            draws.append(metric(y[rows], a[rows]) - metric(y[rows], b[rows]))
        except ValueError:
            skipped += 1
    return _bootstrap_summary(difference, draws, skipped, alpha)


def paired_mean_difference(
    values_a: np.ndarray,
    values_b: np.ndarray,
    patients: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 666,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """The mean over stays of ``values_a - values_b``, with the same patient-level bootstrap."""
    a, b, patients = (
        np.asarray(values_a, dtype=float),
        np.asarray(values_b, dtype=float),
        np.asarray(patients),
    )
    if not len(a) == len(b) == len(patients):
        raise ValueError("both value arrays and patients must have the same length")
    diff = a - b
    rows_of = _patient_rows(patients)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(rows_of), size=len(rows_of))
        draws.append(float(diff[np.concatenate([rows_of[k] for k in chosen])].mean()))
    return _bootstrap_summary(float(diff.mean()), draws, 0, alpha)


def holm_bonferroni(p_values: Mapping[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """Holm's step-down adjustment: sort the p-values, multiply the i-th smallest by ``m - i``, make the
    adjusted values non-decreasing, cap at 1. ``reject`` is ``adjusted <= alpha``."""
    names = sorted(p_values, key=lambda n: p_values[n])
    m = len(names)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(names):
        running = max(running, min(1.0, (m - rank) * float(p_values[name])))
        adjusted[name] = running
    return {
        n: {"p": float(p_values[n]), "p_adjusted": adjusted[n], "reject": adjusted[n] <= alpha}
        for n in p_values
    }


def seed_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Mean and sample standard deviation (ddof = 1; ``nan`` for a single seed) over seeds."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        raise ValueError("no values to summarise")
    return {
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else float("nan"),
        "n": int(v.size),
    }


def mean_ci(
    values: Sequence[float | None],
    patients: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 666,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Mean of per-stay values with a patient-level percentile CI.

    ``None`` and ``nan`` (undefined for that stay) are left out and counted in ``missing``;
    they are never turned into numbers.
    """
    v = np.asarray([np.nan if x is None else x for x in values], dtype=float)
    patients = np.asarray(patients)
    if len(v) != len(patients):
        raise ValueError("values and patients must have the same length")
    keep = np.isfinite(v)
    v, patients = v[keep], patients[keep]
    if not len(v):
        return {
            "mean": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "n": 0,
            "missing": int((~keep).sum()),
        }
    rows_of = _patient_rows(patients)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(rows_of), size=len(rows_of))
        draws.append(float(v[np.concatenate([rows_of[k] for k in chosen])].mean()))
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(v.mean()),
        "low": float(low),
        "high": float(high),
        "n": len(v),
        "missing": int((~keep).sum()),
    }


def ratio_ci(
    numerators: np.ndarray,
    denominators: np.ndarray,
    patients: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 666,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Sum of numerators over sum of denominators (e.g. supported claims over claims) with a patient-level CI.

    A resample whose denominators sum to 0 is skipped and counted."""
    num, den, patients = np.asarray(numerators, float), np.asarray(denominators, float), np.asarray(patients)
    if not len(num) == len(den) == len(patients):
        raise ValueError("numerators, denominators and patients must have the same length")
    if den.sum() == 0:
        return {
            "ratio": float("nan"),
            "low": float("nan"),
            "high": float("nan"),
            "draws": 0,
            "skipped": n_boot,
        }
    rows_of = _patient_rows(patients)
    rng = np.random.default_rng(seed)
    draws, skipped = [], 0
    for _ in range(n_boot):
        chosen = rng.integers(0, len(rows_of), size=len(rows_of))
        rows = np.concatenate([rows_of[k] for k in chosen])
        total = den[rows].sum()
        if total == 0:
            skipped += 1
        else:
            draws.append(float(num[rows].sum() / total))
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2]) if draws else (float("nan"), float("nan"))
    return {
        "ratio": float(num.sum() / den.sum()),
        "low": float(low),
        "high": float(high),
        "draws": len(draws),
        "skipped": skipped,
    }
