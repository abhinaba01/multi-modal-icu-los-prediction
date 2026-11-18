"""Stage 4 attribution methods, on a model trained to the planted rule (see attribution/synthetic)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fate_icu.attribution import methods as M
from fate_icu.attribution.benchmark import BenchmarkConfig, train_planted_model
from fate_icu.attribution.gates import GatedLogit, ones_gates, stay_tensors
from fate_icu.attribution.synthetic import planted_stays
from fate_icu.attribution.units import StayScorer
from fate_icu.models.train import predict

CFG = BenchmarkConfig(
    seed=7, n_stays=1500, epochs=50, lr=0.003, model={"d": 32, "heads": 4, "layers": 1, "dropout": 0.1}
)
MCFG = M.MethodConfig(ig_steps=64, mask_epochs=200, mask_lambda_1=3.0, dynamask_epochs=20, seed=7)


@pytest.fixture(scope="module")
def world():
    planted = planted_stays(n=CFG.n_stays, seed=CFG.seed)
    trained, auroc = train_planted_model(planted, CFG)
    data = planted.data
    test = data.rows("test")
    probs = predict(trained, data, test)["p_long"]
    rows = test[(data.y_long[test] > 0.5) & (probs > 0.5) & planted.truth[test].any(axis=1)][:24]
    return planted, GatedLogit(trained.model, planted.layout), rows, stay_tensors(data, rows)


def test_occlusion_is_the_exact_removal_delta_and_matches_the_per_stay_scorer(world):
    planted, model, rows, fixed = world
    batch = M.ts_occlusion(model, planted.space, fixed, MCFG)
    assert batch.shape == (len(rows), planted.space.n_ts)
    for i in (0, 5):
        scorer = StayScorer(model, planted.space, planted.data, int(rows[i]))
        assert np.allclose(batch[i], scorer.removal_delta(np.arange(planted.space.n_ts)), atol=1e-5)


def test_integrated_gradients_is_complete_for_every_modality(world):
    # the attributions of one modality sum to logit(all kept) - logit(that modality removed)
    planted, model, rows, fixed = world
    space = planted.space
    n = len(rows)
    ones = ones_gates(space.layout, n)
    full = space.gates(np.zeros((n, len(space)), dtype=bool))
    base = M.BatchScorer(model, space, fixed).base()
    cases = {
        "ts": (M.ts_ig(model, space, fixed, MCFG), {"ts": torch.zeros_like(ones["ts"])}),
        "static": (M.static_ig(model, space, fixed, MCFG), {"static": torch.zeros_like(ones["static"])}),
        "concept": (M.concept_ig(model, space, fixed, MCFG), {"concept": torch.zeros_like(ones["concept"])}),
    }
    assert full["ts"].shape == ones["ts"].shape
    for name, (attribution, closed) in cases.items():
        from fate_icu.attribution.gates import logit

        with torch.no_grad():
            without = logit(model, fixed, **closed).numpy()
        assert np.allclose(
            attribution.sum(axis=1), base - without, atol=0.08 * max(1.0, np.abs(base - without).max())
        ), name


def test_scores_have_one_value_per_unit_of_the_modality(world):
    planted, model, rows, fixed = world
    space = planted.space
    assert M.static_ablation(model, space, fixed, MCFG).shape == (len(rows), space.n_static)
    assert M.concept_ablation(model, space, fixed, MCFG).shape == (len(rows), space.n_concepts)
    assert M.whole_image_ablation(model, space, fixed, MCFG).shape == (len(rows),)
    full = M.attribute(model, space, fixed, MCFG, ts="ig", static="ablation", concept="ablation")
    assert full.shape == (len(rows), len(space)) and np.isfinite(full).all()


@pytest.mark.parametrize("method", ["occlusion", "ig", "extremal_mask"])
def test_time_series_methods_find_the_planted_unit(world, method):
    planted, model, rows, fixed = world
    ts_truth = planted.truth[rows][:, : planted.space.n_ts]
    have = ts_truth.any(axis=1)
    assert have.sum() >= 8
    scores = M.TS[method](model, planted.space, fixed, MCFG)
    top1 = np.abs(scores).argmax(axis=1)
    hit = np.array([ts_truth[i, top1[i]] for i in range(len(rows))])[have].mean()
    assert hit >= 0.6, f"{method}: top-1 hit rate {hit:.2f}"


def test_the_static_and_concept_methods_find_the_planted_unit(world):
    planted, model, rows, fixed = world
    space = planted.space
    for table, part, name in (
        (M.STATIC, space.static_slice, "static"),
        (M.CONCEPT, space.concept_slice, "concept"),
    ):
        truth = planted.truth[rows][:, part]
        have = truth.any(axis=1)
        if have.sum() < 3:
            continue
        for method, fn in table.items():
            scores = fn(model, space, fixed, MCFG)
            hit = np.array([truth[i, np.abs(scores[i]).argmax()] for i in range(len(rows))])[have].mean()
            # static ablation is the weakest recoverer in the benchmark (D-048): only require
            # it to beat chance (1 in 3 units); the others must find the unit most of the time
            floor = 0.34 if (name, method) == ("static", "ablation") else 0.5
            assert hit >= floor, f"{name} {method}: {hit:.2f}"


def test_mask_importances_are_between_zero_and_one(world):
    planted, model, rows, fixed = world
    scores = M.ts_extremal_mask(model, planted.space, fixed, MCFG)
    assert scores.min() >= -1e-6 and scores.max() <= 1 + 1e-6


def test_the_other_modalities_are_held_fixed(world):
    # explaining the time series must not depend on what the static gates would do
    planted, model, rows, fixed = world
    a = M.ts_ig(model, planted.space, fixed, MCFG)
    changed = {**fixed, "static": fixed["static"] * 0 + 7.0}
    b = M.ts_ig(model, planted.space, changed, MCFG)
    assert not np.allclose(a, b)  # the fixed static input does enter the logit ...
    again = M.ts_ig(model, planted.space, fixed, MCFG)
    assert np.allclose(a, again)  # ... and nothing else changes the result


def test_unknown_method_names_are_refused(world):
    planted, model, rows, fixed = world
    with pytest.raises(ValueError, match="unknown time series method"):
        M.attribute(model, planted.space, fixed, MCFG, ts="shap")
    with pytest.raises(ValueError, match="unknown static method"):
        M.attribute(model, planted.space, fixed, MCFG, static="lime")


def test_removing_a_planted_unit_usually_lowers_a_long_stay_logit(world):
    planted, model, rows, fixed = world
    space = planted.space
    base = M.BatchScorer(model, space, fixed).base()
    drops = []
    for i, row in enumerate(rows):
        truth = np.flatnonzero(planted.truth[row])
        if len(truth) != 1:
            continue
        removed = np.zeros((1, len(space)), dtype=bool)
        removed[0, truth] = True
        scorer = StayScorer(model, space, planted.data, int(row))
        drops.append(base[i] - scorer.logits(removed)[0])  # Delta > 0: the unit supported "long"
    drops = np.array(drops)
    assert len(drops) >= 5
    assert (drops > 0).mean() >= 0.7 and drops.mean() > 0.3  # a model may use another cue in a stay
