"""Stage 3 arms: late fusion, same test patients for every arm, populations and thresholds."""

from __future__ import annotations

import numpy as np
import pytest
from stay_data import random_stay_data

from fate_icu.models.arms import ARMS, LATE_FUSION_PARTS, evaluate_arm, late_fusion, run_arms, sized
from fate_icu.models.train import TrainConfig
from fate_icu.tabular._legacy import legacy

FAST = TrainConfig(epochs=3, batch_size=16, patience=3, augment=True)
SMALL = {name: sized(arm, d=16, heads=2, layers=1, dropout=0.0) for name, arm in ARMS.items()}


@pytest.fixture(scope="module")
def data():
    return random_stay_data(n=160, seed=21)


@pytest.fixture(scope="module")
def predictions(data):
    return run_arms(["B1", "B2", "B3", "B4", "B5", "B6", "M"], data, FAST, seed=0, arms=SMALL)


def test_the_arm_table_matches_the_stage_spec():
    assert list(ARMS) == ["B1", "B2", "B3", "B4", "B5", "B6", "M", "M_transformer"]
    assert ARMS["B4"].paired_only and ARMS["B4"].populations == ("P1",)
    assert ARMS["B5"].spec is None
    assert ARMS["M"].spec.fusion == "attention" and ARMS["B6"].spec.fusion == "concat"
    assert ARMS["M_transformer"].spec.ts_encoder == "transformer"
    assert ARMS["B1"].spec.modalities == ("static",) and ARMS["B3"].spec.modalities == ("static", "ts")


def test_every_arm_predicts_every_stay(data, predictions):
    for name, pred in predictions.items():
        for key in ("logit", "p_long", "los_days"):
            assert pred[key].shape == (len(data),), (name, key)
    for name in ("B1", "B2", "B3", "B6", "M"):
        assert np.isfinite(predictions[name]["p_long"]).all()


def test_the_image_only_arm_has_no_prediction_without_an_image(data, predictions):
    p = predictions["B4"]["p_long"]
    assert np.isnan(p[~data.has_image]).all() and np.isfinite(p[data.has_image]).all()


def test_late_fusion_is_the_mean_of_the_available_unimodal_logits(predictions, data):
    b5 = predictions["B5"]["logit"]
    parts = np.stack([predictions[k]["logit"] for k in LATE_FUSION_PARTS])
    assert np.allclose(b5, np.nanmean(parts, axis=0))
    no_image = ~data.has_image
    assert np.allclose(b5[no_image], (predictions["B1"]["logit"] + predictions["B2"]["logit"])[no_image] / 2)
    assert np.allclose(predictions["B5"]["p_long"], 1 / (1 + np.exp(-b5)))


def test_late_fusion_of_nothing_is_nan_not_zero():
    nan = np.full(3, np.nan)
    out = late_fusion({"a": {"logit": nan, "los_days": nan}, "b": {"logit": nan, "los_days": nan}})
    assert np.isnan(out["p_long"]).all()


def test_b5_needs_its_parts_and_asking_for_it_trains_them(data):
    out = run_arms(["B5"], data, FAST, seed=1, arms=SMALL)
    assert list(out) == ["B5"] and np.isfinite(out["B5"]["logit"]).all()


def test_every_arm_is_scored_on_the_same_test_patients(data, predictions):
    p1 = {
        n: evaluate_arm(predictions[n], data, "P1", n_boot=20)
        for n in ("B1", "B2", "B3", "B4", "B5", "B6", "M")
    }
    assert len({r["n_stays"] for r in p1.values()}) == 1 and len({r["n_patients"] for r in p1.values()}) == 1
    p2 = {n: evaluate_arm(predictions[n], data, "P2", n_boot=20) for n in ("B1", "B2", "B3", "B5", "B6", "M")}
    assert len({r["n_stays"] for r in p2.values()}) == 1
    test_stays = int((data.split == "test").sum())
    assert p2["M"]["n_stays"] == test_stays
    assert p1["M"]["n_stays"] == int((data.has_image & (data.split == "test")).sum()) < test_stays


def test_the_threshold_is_the_legacy_gmean_on_train_scores_only(data, predictions):
    result = evaluate_arm(predictions["M"], data, "P2", n_boot=10)
    train = data.rows("train")
    expected = legacy("models.stepwise").gmean_threshold(
        data.y_long[train], predictions["M"]["p_long"][train]
    )
    assert result["gmean_threshold"] == pytest.approx(expected)
    changed = {k: v.copy() for k, v in predictions["M"].items()}
    test = data.rows("test")
    changed["p_long"][test] = 1 - changed["p_long"][test]
    assert evaluate_arm(changed, data, "P2", n_boot=10)["gmean_threshold"] == result["gmean_threshold"]


def test_metrics_and_the_two_stage_regression_are_reported(data, predictions):
    result = evaluate_arm(predictions["M"], data, "P2", n_boot=30)
    assert {"auroc", "auprc", "brier", "ece", "mae_days", "rmse_days"} <= set(result)
    assert result["n_predicted_short"] >= 0
    if "pred_short" in result:
        assert set(result["pred_short"]) == {"mae_days", "rmse_days"}


def test_a_population_without_both_classes_is_skipped_with_a_reason(data, predictions):
    one_class = random_stay_data(n=160, seed=21)
    one_class.y_long[one_class.rows("test")] = 0.0
    result = evaluate_arm(predictions["M"], one_class, "P2", n_boot=10)
    assert "skipped" in result and "auroc" not in result
    with pytest.raises(ValueError, match="unknown population"):
        evaluate_arm(predictions["M"], data, "P3")


def test_running_twice_gives_identical_predictions(data):
    a = run_arms(["M", "B3"], data, FAST, seed=2, arms=SMALL)
    b = run_arms(["M", "B3"], data, FAST, seed=2, arms=SMALL)
    for name in a:
        assert np.array_equal(a[name]["logit"], b[name]["logit"])
