"""Evaluation statistics, each checked on a small example worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fate_icu.eval.predictive import auroc
from fate_icu.eval.stats import (
    holm_bonferroni,
    paired_mean_difference,
    paired_patient_bootstrap,
    seed_summary,
)


def test_holm_matches_the_hand_computed_adjustment():
    # sorted p: 0.01, 0.03, 0.04 with m = 3 -> 0.03, 0.06, 0.04, made non-decreasing: 0.03, 0.06, 0.06
    out = holm_bonferroni({"H1": 0.01, "H2": 0.04, "H3": 0.03})
    assert out["H1"]["p_adjusted"] == pytest.approx(0.03)
    assert out["H3"]["p_adjusted"] == pytest.approx(0.06)
    assert out["H2"]["p_adjusted"] == pytest.approx(0.06)
    assert [out[n]["reject"] for n in ("H1", "H2", "H3")] == [True, False, False]


def test_holm_caps_at_one_and_keeps_a_single_p_value():
    assert holm_bonferroni({"a": 0.6, "b": 0.9})["b"]["p_adjusted"] == 1.0
    one = holm_bonferroni({"only": 0.03})["only"]
    assert one["p_adjusted"] == pytest.approx(0.03) and one["reject"] is True


def test_seed_summary_mean_and_sample_sd():
    s = seed_summary([1.0, 2.0, 3.0])
    assert s == {"mean": 2.0, "sd": 1.0, "n": 3}
    assert math.isnan(seed_summary([5.0])["sd"])
    with pytest.raises(ValueError):
        seed_summary([])


def _two_scorers(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    good = y * 0.6 + rng.random(n) * 0.5  # informative
    bad = rng.random(n)  # noise
    patients = np.arange(n)
    return y, good, bad, patients


def test_a_better_scorer_gives_a_positive_difference_that_excludes_zero():
    y, good, bad, patients = _two_scorers()
    out = paired_patient_bootstrap(auroc, y, good, bad, patients, n_boot=300, seed=1)
    assert out["difference"] > 0.2 and out["low"] > 0 and out["p_value"] < 0.05
    assert out["draws"] == 300 and out["skipped"] == 0


def test_identical_scorers_have_zero_difference_and_p_value_one():
    y, good, _, patients = _two_scorers()
    out = paired_patient_bootstrap(auroc, y, good, good, patients, n_boot=100, seed=1)
    assert out["difference"] == 0 and out["low"] == 0 == out["high"] and out["p_value"] == 1.0


def test_the_bootstrap_is_reproducible_and_depends_on_the_seed():
    y, good, bad, patients = _two_scorers()
    a = paired_patient_bootstrap(auroc, y, good, bad, patients, n_boot=100, seed=3)
    b = paired_patient_bootstrap(auroc, y, good, bad, patients, n_boot=100, seed=3)
    c = paired_patient_bootstrap(auroc, y, good, bad, patients, n_boot=100, seed=4)
    assert a == b and a != c


def test_resampling_is_by_patient_not_by_stay():
    # patient 0 differs by +1 on all 100 of its stays, patient 1 by -1 on all 100. A patient-level draw is
    # (P0,P0), (P0,P1) or (P1,P1), so its mean is +1, 0 or -1 and the interval spans [-1, 1]; a stay-level
    # draw would average out to a narrow interval around 0.
    a = np.repeat([1.0, -1.0], 100)
    b = np.zeros(200)
    by_patient = paired_mean_difference(a, b, np.repeat([0, 1], 100), n_boot=400, seed=2)
    by_stay = paired_mean_difference(a, b, np.arange(200), n_boot=400, seed=2)
    assert by_patient["low"] == -1.0 and by_patient["high"] == 1.0
    assert by_stay["high"] - by_stay["low"] < 0.5


def test_draws_with_one_class_are_skipped_and_counted_not_replaced():
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # one positive patient: many resamples have no positive at all
    a = np.linspace(0, 1, 8)
    out = paired_patient_bootstrap(auroc, y, a, a[::-1], np.arange(8), n_boot=200, seed=0)
    assert out["skipped"] > 0 and out["draws"] + out["skipped"] == 200


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="same length"):
        paired_patient_bootstrap(
            auroc, np.array([0, 1]), np.array([0.1]), np.array([0.2, 0.3]), np.array([0, 1])
        )
    with pytest.raises(ValueError, match="same length"):
        paired_mean_difference(np.array([1.0, 2.0]), np.array([1.0]), np.array([0, 1]))


def test_paired_mean_difference_matches_the_hand_computed_mean():
    a = np.array([3.0, 3.0, 3.0, 3.0])
    b = np.array([1.0, 2.0, 1.0, 2.0])
    out = paired_mean_difference(a, b, np.array([0, 1, 2, 3]), n_boot=200, seed=1)
    assert (
        out["difference"] == pytest.approx(1.5)
        and out["low"] > 0.9
        and out["high"] < 2.1
        and out["skipped"] == 0
    )


def test_mean_ci_leaves_out_undefined_values_and_counts_them():
    from fate_icu.eval.stats import mean_ci

    out = mean_ci([1.0, 2.0, 3.0, None, float("nan")], np.arange(5), n_boot=200, seed=0)
    assert out["mean"] == 2.0 and out["n"] == 3 and out["missing"] == 2 and out["low"] <= 2.0 <= out["high"]
    assert math.isnan(mean_ci([None, None], np.arange(2))["mean"])


def test_ratio_ci_is_a_ratio_of_sums_not_a_mean_of_ratios():
    from fate_icu.eval.stats import ratio_ci

    num, den = np.array([1, 0, 9]), np.array([1, 10, 10])  # 10 of 21 supported overall, not mean(1, 0, 0.9)
    out = ratio_ci(num, den, np.arange(3), n_boot=300, seed=1)
    assert out["ratio"] == pytest.approx(10 / 21) and out["low"] <= out["ratio"] <= out["high"]
    empty = ratio_ci(np.zeros(3), np.zeros(3), np.arange(3), n_boot=10)
    assert math.isnan(empty["ratio"]) and empty["skipped"] == 10
