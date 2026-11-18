"""Stage 3 training: it can overfit a tiny batch, is seed-deterministic and never reads test."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score
from stay_data import random_stay_data

from fate_icu.models.fusion import ModelSpec
from fate_icu.models.train import TrainConfig, load_trained, predict, save_trained, train_model

SMALL = dict(d=16, heads=2, layers=1, dropout=0.0)
FAST = TrainConfig(epochs=4, batch_size=16, patience=4, augment=True)


def spec(modalities=("static", "ts", "image"), fusion="attention", **kw):
    return ModelSpec(modalities, fusion, **{**SMALL, **kw})


def test_a_tiny_batch_can_be_memorised():
    data = random_stay_data(n=16, seed=0)
    data.split[:] = "train"
    cfg = TrainConfig(
        epochs=300,
        batch_size=16,
        lr=3e-3,
        patience=300,
        augment=False,
        lambda_reg=1.0,
        warmup_frac=0.02,
        weight_decay=0.0,
        allow_loss_fallback=True,
    )
    trained = train_model(spec(), data, cfg, seed=0, train_rows=np.arange(16), val_rows=np.arange(16))
    out = predict(trained, data, np.arange(16))
    assert trained.history[-1]["train_loss"] < 0.15  # both heads: started near 1.4 (BCE + Huber)
    # the best validation epoch may come before the probabilities are calibrated, so check the ranking
    assert roc_auc_score(data.y_long, out["p_long"]) == 1.0


def test_the_same_seed_gives_identical_models_and_different_seeds_do_not():
    data = random_stay_data(n=80, seed=1)
    a = predict(train_model(spec(), data, FAST, seed=3), data, np.arange(80))
    b = predict(train_model(spec(), data, FAST, seed=3), data, np.arange(80))
    c = predict(train_model(spec(), data, FAST, seed=4), data, np.arange(80))
    assert np.array_equal(a["logit"], b["logit"]) and np.array_equal(a["los_days"], b["los_days"])
    assert not np.allclose(a["logit"], c["logit"])


def test_the_test_split_is_never_read():
    data = random_stay_data(n=100, seed=2)
    other = copy.deepcopy(data)
    test = other.split == "test"
    rng = np.random.default_rng(0)
    other.y_long[test] = rng.integers(0, 2, test.sum()).astype(np.float32)
    other.los_days[test] = rng.random(test.sum()).astype(np.float32) * 20
    other.static[test] += 5.0
    other.values[test] += 5.0
    other.image[test & other.has_image] += 5.0
    a = predict(train_model(spec(), data, FAST, seed=0), data, data.rows("train"))
    b = predict(train_model(spec(), other, FAST, seed=0), other, other.rows("train"))
    assert np.array_equal(a["logit"], b["logit"])


def test_learning_beats_chance_on_a_signal_and_not_on_noise():
    cfg = TrainConfig(epochs=40, batch_size=32, lr=3e-3, patience=40, augment=False, warmup_frac=0.05)
    signal = random_stay_data(n=400, seed=5, signal=True)
    noise = random_stay_data(n=400, seed=5, signal=False)
    scores = {}
    for name, data in (("signal", signal), ("noise", noise)):
        trained = train_model(spec(("static", "ts"), "concat"), data, cfg, seed=0)
        rows = data.rows("test")
        scores[name] = roc_auc_score(data.y_long[rows], predict(trained, data, rows)["p_long"])
    assert scores["signal"] > 0.8 and scores["noise"] < 0.7


def test_the_best_validation_epoch_is_restored():
    data = random_stay_data(n=120, seed=6)
    trained = train_model(
        spec(), data, TrainConfig(epochs=6, batch_size=16, patience=6, augment=False), seed=0
    )
    scores = [h["val_auprc"] for h in trained.history]
    assert trained.history[trained.best_epoch]["val_auprc"] == pytest.approx(max(scores))
    assert trained.selection == "auprc"


def test_early_stopping_stops_after_the_patience():
    data = random_stay_data(n=120, seed=6, signal=False)
    cfg = TrainConfig(epochs=60, batch_size=16, patience=2, augment=False, lr=1e-2)
    trained = train_model(spec(), data, cfg, seed=0)
    assert len(trained.history) < 60 and len(trained.history) - trained.best_epoch - 1 == 2


def test_a_validation_split_with_one_class_stops_the_run_unless_allowed():
    data = random_stay_data(n=100, seed=7)
    val = data.rows("val")
    data.y_long[val] = 0.0
    with pytest.raises(ValueError, match="one class only"):
        train_model(spec(), data, FAST, seed=0)
    trained = train_model(spec(), data, TrainConfig(**{**FAST.__dict__, "allow_loss_fallback": True}), seed=0)
    assert trained.selection == "val_mae_log1p"


def test_training_needs_both_classes_in_train():
    data = random_stay_data(n=60, seed=8)
    data.y_long[data.rows("train")] = 0.0
    with pytest.raises(ValueError, match="both short and long"):
        train_model(spec(), data, FAST, seed=0)


def test_an_image_only_model_trains_on_paired_stays_and_is_nan_without_an_image():
    data = random_stay_data(n=140, seed=9)
    trained = train_model(
        spec(("image",), "none"),
        data,
        FAST,
        seed=0,
        train_rows=data.rows("train", paired_only=True),
        val_rows=data.rows("val", paired_only=True),
    )
    out = predict(trained, data, np.arange(len(data)))
    assert np.isnan(out["p_long"][~data.has_image]).all() and np.isfinite(out["p_long"][data.has_image]).all()


def test_predictions_use_the_images_a_stay_actually_has():
    data = random_stay_data(n=100, seed=10)
    trained = train_model(spec(), data, FAST, seed=0)
    rows = np.flatnonzero(data.has_image)[:10]
    before = predict(trained, data, rows)["logit"]
    scrambled = copy.deepcopy(data)
    scrambled.image[rows] = np.random.default_rng(0).normal(size=scrambled.image[rows].shape) * 5
    after = predict(trained, scrambled, rows)["logit"]
    assert not np.allclose(before, after)
    # a stay without an image does not change when image values are scrambled elsewhere
    none = np.flatnonzero(~data.has_image)[:10]
    assert np.array_equal(predict(trained, data, none)["logit"], predict(trained, scrambled, none)["logit"])


def test_config_validation():
    with pytest.raises(ValueError):
        TrainConfig(epochs=0)
    with pytest.raises(ValueError):
        TrainConfig(unit_drop_low=0.2, unit_drop_high=0.1)
    with pytest.raises(ValueError, match="AUPRC"):
        TrainConfig(selection="auroc")


def test_threads_do_not_change_results():
    data = random_stay_data(n=80, seed=11)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        a = predict(train_model(spec(), data, FAST, seed=1), data, np.arange(80))["logit"]
        torch.set_num_threads(max(2, previous))
        b = predict(train_model(spec(), data, FAST, seed=1), data, np.arange(80))["logit"]
    finally:
        torch.set_num_threads(previous)
    assert np.allclose(a, b, atol=1e-5)


@pytest.mark.gpu
def test_bf16_autocast_trains_on_a_gpu_and_keeps_float32_losses():
    data = random_stay_data(n=80, seed=12)
    cfg = TrainConfig(epochs=2, batch_size=16, device="cuda", autocast_bf16=True)
    trained = train_model(spec(), data, cfg, seed=0)
    assert all(np.isfinite(h["train_loss"]) for h in trained.history)
    assert np.isfinite(predict(trained, data, np.arange(80))["logit"]).all()


def test_autocast_is_ignored_on_the_cpu():
    data = random_stay_data(n=60, seed=13)
    a = predict(train_model(spec(), data, FAST, seed=0), data, np.arange(60))["logit"]
    on = TrainConfig(**{**FAST.__dict__, "autocast_bf16": True, "device": "cpu"})
    b = predict(train_model(spec(), data, on, seed=0), data, np.arange(60))["logit"]
    assert np.array_equal(a, b)


def test_a_saved_model_loads_back_with_identical_predictions(tmp_path):
    data = random_stay_data(n=80, seed=14)
    trained = train_model(spec(), data, FAST, seed=0)
    path = save_trained(trained, tmp_path / "sub" / "M.pt", data.dims, extra={"arm": "M", "seed": 0})
    loaded, extra = load_trained(path)
    rows = np.arange(len(data))
    assert extra == {"arm": "M", "seed": 0}
    assert loaded.spec == trained.spec and loaded.best_epoch == trained.best_epoch
    assert loaded.history == trained.history and not loaded.model.training
    assert np.array_equal(predict(trained, data, rows)["logit"], predict(loaded, data, rows)["logit"])
    assert not list(tmp_path.rglob("*.tmp"))  # written atomically


def test_an_unknown_checkpoint_format_is_refused(tmp_path):
    torch.save({"format": 99}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="unknown checkpoint format"):
        load_trained(tmp_path / "x.pt")
