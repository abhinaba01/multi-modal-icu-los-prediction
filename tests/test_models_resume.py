"""Stage 3 resume: a reclaimed spot VM loses at most the arm in progress."""

from __future__ import annotations

import numpy as np
import pytest
from stay_data import random_stay_data

from fate_icu.models import run as run_module
from fate_icu.models.arms import ARMS, sized
from fate_icu.models.run import _load_predictions, _predictions_file, train_or_resume_arms
from fate_icu.models.train import TrainConfig

FAST = TrainConfig(epochs=2, batch_size=16, patience=2, augment=False)
SMALL = {name: sized(arm, d=16, heads=2, layers=1, dropout=0.0) for name, arm in ARMS.items()}


@pytest.fixture(scope="module")
def data():
    return random_stay_data(n=100, seed=41)


def run_once(data, tmp_path, neural, seed=0, log=lambda _m: None):
    return train_or_resume_arms(
        neural, data, FAST, seed, SMALL, out_dir=tmp_path / "out", ckpt_dir=tmp_path / "ckpt",
        run_id="r1", log=log,
    )  # fmt: skip


def test_every_arm_is_saved_as_soon_as_it_finishes(data, tmp_path):
    (tmp_path / "out").mkdir()
    run_once(data, tmp_path, ["B1", "B2"])
    for name in ("B1", "B2"):
        assert _predictions_file(tmp_path / "out", name, 0).exists()
        assert (tmp_path / "ckpt" / f"{name}_seed0.pt").exists()
    assert not list((tmp_path / "out").glob("*.tmp.npz"))  # written atomically


def test_a_finished_arm_is_read_back_not_retrained(data, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    first = run_once(data, tmp_path, ["B1", "B3"])

    def refuse(*_a, **_k):
        raise AssertionError("a finished arm must not be trained again")

    monkeypatch.setattr(run_module, "run_arm", refuse)
    messages: list[str] = []
    again = run_once(data, tmp_path, ["B1", "B3"], log=messages.append)
    assert [m for m in messages if m.startswith("resumed")] == [
        "resumed B1 seed 0 (already trained)",
        "resumed B3 seed 0 (already trained)",
    ]
    for name in ("B1", "B3"):
        assert np.array_equal(first[name]["logit"], again[name]["logit"])


def test_only_the_arm_that_was_interrupted_is_trained(data, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    first = run_once(data, tmp_path, ["B1", "B2"])
    # the VM was reclaimed while B2 was being saved: its predictions file never appeared
    _predictions_file(tmp_path / "out", "B2", 0).unlink()
    trained = []
    real = run_module.run_arm

    def watch(arm, *args, **kwargs):
        trained.append(arm.name)
        return real(arm, *args, **kwargs)

    monkeypatch.setattr(run_module, "run_arm", watch)
    again = run_once(data, tmp_path, ["B1", "B2"])
    assert trained == ["B2"]
    assert np.array_equal(first["B1"]["logit"], again["B1"]["logit"])
    assert np.array_equal(first["B2"]["logit"], again["B2"]["logit"])  # same seed: same model


def test_late_fusion_is_recomputed_from_the_saved_parts_and_never_saved_as_a_model(data, tmp_path):
    (tmp_path / "out").mkdir()
    out = run_once(data, tmp_path, ["B5"])
    assert set(out) == {"B5"}
    assert {p.name for p in (tmp_path / "ckpt").glob("*.pt")} == {"B1_seed0.pt", "B2_seed0.pt", "B4_seed0.pt"}
    parts = np.stack([
        _load_predictions(_predictions_file(tmp_path / "out", n, 0), data.stay_ids)["logit"]
        for n in ("B1", "B2", "B4")
    ])  # fmt: skip
    assert np.allclose(out["B5"]["logit"], np.nanmean(parts, axis=0))


def test_predictions_made_for_other_stays_are_refused(data, tmp_path):
    (tmp_path / "out").mkdir()
    run_once(data, tmp_path, ["B1"])
    other = random_stay_data(n=100, seed=42)
    other.stay_ids[:] = other.stay_ids + 1  # a different cohort
    with pytest.raises(ValueError, match="different stays"):
        run_once(other, tmp_path, ["B1"])


def test_seeds_are_kept_apart(data, tmp_path):
    (tmp_path / "out").mkdir()
    a = run_once(data, tmp_path, ["B1"], seed=0)
    b = run_once(data, tmp_path, ["B1"], seed=1)
    assert (tmp_path / "ckpt" / "B1_seed0.pt").exists() and (tmp_path / "ckpt" / "B1_seed1.pt").exists()
    assert not np.array_equal(a["B1"]["logit"], b["B1"]["logit"])
