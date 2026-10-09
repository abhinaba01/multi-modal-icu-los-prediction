"""Stepwise (two-stage) helpers: G-mean threshold selection + confusion counts.

The paper's practical framework classifies each stay as short/long first, then
regresses LOS only on the *predicted-short* stays. The prior classifiers (random
forest and XGBoost) are turned into hard short/long calls by choosing the
probability threshold that maximises the geometric mean of sensitivity and
specificity, ``sqrt(TPR * TNR)`` -- the paper's stated selection rule.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_curve


def gmean_threshold(y_true, y_score) -> float:
    """Return the probability threshold maximising ``sqrt(TPR * TNR)``.

    Args:
        y_true: True labels (0 short, 1 long).
        y_score: Predicted probability of the positive class (long).

    Returns:
        The threshold on ``y_score`` (predict long when ``score >= threshold``)
        that maximises the G-mean. Falls back to 0.5 if it cannot be estimated.
    """
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    gmeans = np.sqrt(np.clip(tpr, 0, None) * np.clip(1.0 - fpr, 0, None))
    best = int(np.argmax(gmeans))
    threshold = thresholds[best]
    # roc_curve's first threshold is +inf; guard against a degenerate pick.
    if not np.isfinite(threshold):
        return 0.5
    return float(threshold)


def confusion_counts(y_true, y_pred_long) -> dict[str, int]:
    """Confusion counts in the paper's Table-5 orientation.

    Args:
        y_true: True labels (0 short, 1 long).
        y_pred_long: Boolean/int array, 1 where predicted long.

    Returns:
        Dict with keys ``actual_long_pred_long``, ``actual_long_pred_short``,
        ``actual_short_pred_long``, ``actual_short_pred_short``.
    """
    y_true = np.asarray(y_true).astype(int)
    pred_long = np.asarray(y_pred_long).astype(int)
    actual_long = y_true == 1
    actual_short = y_true == 0
    return {
        "actual_long_pred_long": int(np.sum(actual_long & (pred_long == 1))),
        "actual_long_pred_short": int(np.sum(actual_long & (pred_long == 0))),
        "actual_short_pred_long": int(np.sum(actual_short & (pred_long == 1))),
        "actual_short_pred_short": int(np.sum(actual_short & (pred_long == 0))),
    }
