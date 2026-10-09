"""Tests for evaluation metrics -- especially R2 = squared Pearson correlation."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import classification_metrics, gmean, regression_metrics


def test_regression_perfect_prediction():
    m = regression_metrics([2, 4, 6, 8], [2, 4, 6, 8])
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["mape"] == 0.0
    assert abs(m["r2"] - 1.0) < 1e-9


def test_regression_r2_is_squared_correlation_not_sklearn_r2():
    # A constant offset keeps correlation perfect (R2 = 1) while RMSE > 0.
    # sklearn's r2_score would be < 1 here; the R code's corr^2 is exactly 1.
    m = regression_metrics([2, 4, 6, 8], [3, 5, 7, 9])
    assert abs(m["rmse"] - 1.0) < 1e-9
    assert abs(m["mae"] - 1.0) < 1e-9
    assert abs(m["r2"] - 1.0) < 1e-9


def test_regression_mape_definition():
    m = regression_metrics([10.0, 20.0], [11.0, 18.0])
    # (|1|/10 + |-2|/20)/2 * 100 = (0.1 + 0.1)/2 * 100 = 10%
    assert abs(m["mape"] - 10.0) < 1e-9


def test_classification_metrics_basic():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_score = [0.1, 0.6, 0.7, 0.8]
    m = classification_metrics(y_true, y_pred, y_score)
    assert abs(m["accuracy"] - 0.75) < 1e-9
    # positive class = 1: tp=2, fp=1, fn=0 -> F1 = 0.8
    assert abs(m["f1"] - 0.8) < 1e-9
    # balanced acc = (TPR=1 + TNR=0.5)/2 = 0.75
    assert abs(m["balanced_accuracy"] - 0.75) < 1e-9


def test_gmean():
    assert abs(gmean(0.8, 0.5) - np.sqrt(0.4)) < 1e-9
    assert gmean(0.0, 1.0) == 0.0
