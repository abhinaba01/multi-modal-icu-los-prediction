"""Tests for the stepwise helpers (G-mean threshold + confusion counts)."""

from __future__ import annotations

import numpy as np

from src.models.stepwise import confusion_counts, gmean_threshold


def test_gmean_threshold_separable():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    threshold = gmean_threshold(y_true, y_score)
    pred_long = (np.asarray(y_score) >= threshold).astype(int)
    # Perfectly separable -> the chosen threshold classifies all correctly.
    assert list(pred_long) == [0, 0, 1, 1]


def test_gmean_threshold_single_class_falls_back():
    assert gmean_threshold([1, 1, 1], [0.3, 0.4, 0.5]) == 0.5


def test_confusion_counts_orientation():
    counts = confusion_counts([1, 1, 0, 0], [1, 0, 1, 0])
    assert counts["actual_long_pred_long"] == 1
    assert counts["actual_long_pred_short"] == 1
    assert counts["actual_short_pred_long"] == 1
    assert counts["actual_short_pred_short"] == 1
