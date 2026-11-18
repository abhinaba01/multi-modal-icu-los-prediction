"""Stage 4 faithfulness metrics, on a model whose answer is known exactly (an additive one)."""

from __future__ import annotations

import numpy as np
import pytest

from fate_icu.eval.attr_faithfulness import (
    comprehensiveness_sufficiency,
    curves,
    precision_recall_at_k,
    random_order,
    randomisation_agreement,
    ranking,
)

EFFECT = np.array([3.0, 2.0, 1.0, 0.0, 0.0, 0.0])  # unit j adds EFFECT[j] to the logit
BASE = -1.0


def additive(removed: np.ndarray) -> np.ndarray:
    """logit = BASE + sum of the effects of the units that are kept."""
    removed = np.atleast_2d(removed)
    return BASE + ((~removed) * EFFECT).sum(axis=1)  # full input: 5.0, a long-stay logit


def test_precision_and_recall_at_k():
    scores = np.array([0.9, -0.8, 0.1, 0.0, 0.7, 0.2])
    truth = np.array([True, False, False, False, True, False])
    assert precision_recall_at_k(scores, truth, 1) == (1.0, 0.5)
    assert precision_recall_at_k(scores, truth, 2) == (0.5, 0.5)  # |-0.8| outranks 0.7
    assert precision_recall_at_k(scores, truth, 3) == (pytest.approx(2 / 3), 1.0)
    with pytest.raises(ValueError, match="no recall"):
        precision_recall_at_k(scores, np.zeros(6, dtype=bool), 1)
    with pytest.raises(ValueError):
        precision_recall_at_k(scores, truth, 0)


def test_ranking_uses_the_size_of_a_score_and_is_stable_on_ties():
    assert ranking(np.array([0.5, -2.0, 0.5, 1.0])).tolist() == [1, 3, 0, 2]
    pool = np.array([True, False, True, True])
    assert ranking(np.array([0.5, -2.0, 0.5, 1.0]), pool).tolist() == [3, 0, 2]


def test_a_correct_ranking_beats_a_random_one_on_every_measure():
    n = len(EFFECT)
    good = ranking(EFFECT)
    bad = good[::-1]
    c_good, c_bad = curves(good, n, additive), curves(bad, n, additive)
    assert c_good["aopc"] > c_bad["aopc"] and c_good["deletion_auc"] < c_bad["deletion_auc"]
    assert c_good["insertion_auc"] > c_bad["insertion_auc"]
    cs_good, cs_bad = (
        comprehensiveness_sufficiency(good, n, additive),
        comprehensiveness_sufficiency(bad, n, additive),
    )
    assert cs_good[1]["comprehensiveness"] > cs_bad[1]["comprehensiveness"]
    assert cs_good[1]["sufficiency"] < cs_bad[1]["sufficiency"]


def test_comprehensiveness_and_sufficiency_match_a_hand_computation():
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    out = comprehensiveness_sufficiency(ranking(EFFECT), len(EFFECT), additive, ks=(1, 2))
    full = sig(5.0)
    # top-1 removed: logit 2.0; only top-1 kept: logit -1 + 3 = 2.0
    assert out[1]["comprehensiveness"] == pytest.approx(full - sig(2.0))
    assert out[1]["sufficiency"] == pytest.approx(full - sig(2.0))
    # top-2 removed: logit 0.0; only top-2 kept: logit -1 + 5 = 4.0
    assert out[2]["comprehensiveness"] == pytest.approx(full - sig(0.0))
    assert out[2]["sufficiency"] == pytest.approx(full - sig(4.0))


def test_the_deletion_curve_falls_and_the_insertion_curve_rises():
    c = curves(ranking(EFFECT), len(EFFECT), additive)
    assert np.all(np.diff(c["deletion"]) <= 1e-12) and np.all(np.diff(c["insertion"]) >= -1e-12)
    assert c["p_full"] == pytest.approx(1 / (1 + np.exp(-5.0))) and c["predicted_long"] is True
    assert len(c["deletion"]) == len(EFFECT) + 1
    capped = curves(ranking(EFFECT), len(EFFECT), additive, max_steps=2)
    assert len(capped["deletion"]) == 3


def test_a_stay_predicted_short_is_explained_by_the_short_class():
    def shortish(removed):
        return -4.0 + ((~np.atleast_2d(removed)) * -EFFECT).sum(axis=1)  # evidence pushes toward short

    c = curves(ranking(EFFECT), len(EFFECT), shortish)
    assert (
        c["predicted_long"] is False and c["deletion"][-1] < c["p_full"]
    )  # P(short) falls without the evidence


def test_random_order_covers_only_the_allowed_units_and_is_seeded():
    pool = np.array([True, False, True, True, False, False])
    a = random_order(6, pool, np.random.default_rng(1))
    assert sorted(a.tolist()) == [0, 2, 3] and np.array_equal(
        a, random_order(6, pool, np.random.default_rng(1))
    )
    assert len(random_order(6, None, np.random.default_rng(1))) == 6


def test_randomisation_agreement():
    a = np.array([5.0, -3.0, 1.0, 0.5, 0.0])
    assert randomisation_agreement(a, a) == pytest.approx(1.0)
    assert randomisation_agreement(a, -a * 2) == pytest.approx(1.0)  # only the size of a score matters
    assert randomisation_agreement(a, a[::-1]) < 0.5
    assert np.isnan(randomisation_agreement(a, np.zeros(5)))
