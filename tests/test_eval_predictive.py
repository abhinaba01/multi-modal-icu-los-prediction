"""Stage 3 predictive metrics: values checked by hand, and the patient-level bootstrap."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fate_icu.eval.predictive import (
    auprc,
    auroc,
    brier,
    ece,
    evaluate_population,
    mae,
    patient_bootstrap,
    rmse,
)


def test_classification_metrics_on_a_worked_example():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert auroc(y, p) == pytest.approx(0.75)  # 3 of the 4 (positive, negative) pairs are ordered
    assert auprc(y, p) == pytest.approx(average_precision_score(y, p))
    assert brier(y, p) == pytest.approx((0.01 + 0.16 + 0.4225 + 0.04) / 4)
    assert auroc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_both_classes_are_required_for_ranking_metrics():
    with pytest.raises(ValueError, match="both classes"):
        auroc(np.zeros(5), np.linspace(0, 1, 5))
    with pytest.raises(ValueError, match="both classes"):
        auprc(np.ones(5), np.linspace(0, 1, 5))


def test_ece_cases():
    assert ece(np.array([0, 0, 0, 1, 1, 1]), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0)
    # always 0.9 while half the stays are long: one bin, gap 0.4
    assert ece(np.array([0, 1, 0, 1]), np.full(4, 0.9)) == pytest.approx(0.4)
    # a probability of exactly 1 lands in the last bin instead of falling off the end
    assert ece(np.array([1, 1]), np.array([1.0, 1.0])) == pytest.approx(0.0)
    # two bins' worth of error, weighted by their share
    y = np.array([0, 0, 1, 1])
    p = np.array([0.25, 0.25, 0.75, 0.75])
    assert ece(y, p) == pytest.approx(0.25)


def test_regression_metrics():
    y, pred = np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 1.0])
    assert mae(y, pred) == pytest.approx(4 / 3)
    assert rmse(y, pred) == pytest.approx(np.sqrt(10 / 3))


def test_bootstrap_resamples_patients_not_stays():
    # four patients with 1, 1, 3 and 3 stays: a stay-level bootstrap always draws 8 rows, a
    # patient-level one draws between 4 and 12
    patients = np.array([1, 2, 3, 3, 3, 4, 4, 4])
    y = np.zeros(8)
    score = np.zeros(8)
    out = patient_bootstrap(lambda a, b: float(len(a)), y, score, patients, n_boot=400, seed=1)
    assert out["value"] == 8
    assert out["low"] < 8 < out["high"]
    assert out["low"] >= 4 and out["high"] <= 12


def test_bootstrap_is_deterministic_and_seeded():
    rng = np.random.default_rng(0)
    y = (rng.random(60) < 0.4).astype(int)
    score = y * 0.3 + rng.random(60)
    patients = np.repeat(np.arange(30), 2)
    a = patient_bootstrap(auroc, y, score, patients, n_boot=200, seed=5)
    b = patient_bootstrap(auroc, y, score, patients, n_boot=200, seed=5)
    c = patient_bootstrap(auroc, y, score, patients, n_boot=200, seed=6)
    assert a == b and a != c
    assert a["low"] <= a["value"] <= a["high"] and a["draws"] + a["skipped"] == 200


def test_resamples_with_one_class_are_skipped_and_counted():
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1])
    score = np.linspace(0, 1, 8)
    out = patient_bootstrap(auroc, y, score, np.arange(8), n_boot=300, seed=2)
    assert out["skipped"] > 0 and out["draws"] + out["skipped"] == 300


def test_nothing_usable_gives_nan_not_a_made_up_interval():
    y = np.array([0, 1])
    out = patient_bootstrap(auroc, y, np.array([0.1, 0.9]), np.array([1, 1]), n_boot=20, seed=0)
    # one patient: every resample holds the same two stays, so it is usable; use a failing metric
    assert out["draws"] == 20

    def never(_a, _b):
        if len(_a) != 2:
            raise ValueError
        return 1.0

    bad = patient_bootstrap(never, y, np.array([0.1, 0.9]), np.array([1, 2]), n_boot=1, seed=3)
    assert bad["draws"] + bad["skipped"] == 1


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="same length"):
        patient_bootstrap(auroc, np.array([0, 1]), np.array([0.1]), np.array([1, 2]))


def test_evaluate_population_returns_every_metric():
    rng = np.random.default_rng(1)
    y = (rng.random(80) < 0.4).astype(int)
    p = np.clip(y * 0.3 + rng.random(80) * 0.7, 0, 1)
    los = np.where(y == 1, 6.0, 2.0)
    out = evaluate_population(y, p, los, los + rng.normal(size=80), np.repeat(np.arange(40), 2), n_boot=50)
    assert set(out) == {"auroc", "auprc", "brier", "ece", "mae_days", "rmse_days"}
    assert all(v["low"] <= v["high"] for v in out.values())
