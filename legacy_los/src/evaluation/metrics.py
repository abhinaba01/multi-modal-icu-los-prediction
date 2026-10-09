"""Evaluation metrics -- faithful to the paper's definitions and the R code.

Classification (paper section 2.6): accuracy, F1, balanced accuracy, MCC, ROC-AUC,
with the positive class = "long" stay (label 1).

Regression (reference ``evaluation_regression.R``): RMSE, MAE, MAPE, and R2 where
**R2 is the SQUARED PEARSON CORRELATION** ``cor(actual, predicted)^2`` -- NOT
sklearn's ``r2_score`` (1 - SS_res/SS_tot). This is a deliberate faithfulness
choice; see README.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)

CLASSIFICATION_METRICS = ["accuracy", "f1", "balanced_accuracy", "mcc", "auc"]
REGRESSION_METRICS = ["rmse", "mae", "mape", "r2"]


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Compute the paper's classification metrics (positive class = long = 1).

    Args:
        y_true: True labels (0 short, 1 long).
        y_pred: Predicted labels at the 0.5 threshold.
        y_score: Predicted probability of the positive class.

    Returns:
        Dict with accuracy, F1, balanced accuracy, MCC, AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute the paper's regression metrics (R2 = squared Pearson correlation).

    Args:
        y_true: True LOS (days).
        y_pred: Predicted LOS (days).

    Returns:
        Dict with RMSE, MAE, MAPE (%), and R2 (= corr^2).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = float(np.mean(np.abs(error) / y_true) * 100.0)
    r2 = _squared_pearson(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def _squared_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Squared Pearson correlation between truth and prediction (R code's R2).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        ``corr(y_true, y_pred) ** 2``, or 0.0 when either side is constant.
    """
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr ** 2)


def gmean(tpr: float, tnr: float) -> float:
    """Geometric mean of sensitivity and specificity.

    Args:
        tpr: True positive rate (sensitivity).
        tnr: True negative rate (specificity).

    Returns:
        ``sqrt(tpr * tnr)``.
    """
    return float(np.sqrt(max(tpr, 0.0) * max(tnr, 0.0)))


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of metric dicts key-wise (ignoring NaNs).

    Args:
        rows: Per-repeat metric dicts.

    Returns:
        Mean metric dict.
    """
    frame = pd.DataFrame(rows)
    return frame.mean(numeric_only=True).to_dict()
