"""Stage 3 run driver: config checks, smoke image assignment, summaries, and the demo end to end."""

from __future__ import annotations

import json

import pytest

from fate_icu.models.run import (
    ModelsRunConfig,
    _summarise,
    load_run_config,
    resolve_run,
    run,
    smoke_image_keys,
)
from fate_icu.utils.env import REPO_ROOT


def test_the_committed_configs_load_and_the_full_one_is_strict():
    smoke = load_run_config(REPO_ROOT / "configs/models/smoke.yaml")
    full = load_run_config(REPO_ROOT / "configs/models/full.yaml")
    assert not smoke.require_frozen_threshold and smoke.images == "fixtures" and smoke.n_seeds < 5
    assert full.require_frozen_threshold and full.require_signal and full.images == "mimic"
    assert full.n_seeds == 5 and full.model.d == 256 and full.model.layers == 2
    assert full.train["patience"] == 8 and full.train["epochs"] == 50 and full.train["batch_size"] == 64
    assert full.arms == ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "M", "M_transformer"]
    assert not full.train.get("allow_loss_fallback", False)  # the full run never falls back silently


def test_unknown_config_keys_are_errors():
    with pytest.raises(ValueError):
        ModelsRunConfig.model_validate({"run_name": "x", "typo": 1})


def test_smoke_images_depend_on_the_study_id_only():
    keys = ["a", "b", "c", "d"]
    ids = [str(i) for i in range(200)]
    first = smoke_image_keys(ids, keys)
    assert first == smoke_image_keys(ids, keys)  # deterministic
    assert set(first) == set(keys)  # every public image gets used
    assert smoke_image_keys(["x"], keys) == smoke_image_keys(["x"], keys)


def test_a_finished_run_is_found_by_id_or_by_folder_and_a_missing_one_stops(tmp_path, settings):
    folder = tmp_path / "run"
    folder.mkdir()
    assert resolve_run(settings, "timeseries", str(folder)) == folder
    with pytest.raises(FileNotFoundError, match="run that stage first"):
        resolve_run(settings, "timeseries", "no_such_run_id")


def _result(auroc, skipped=False):
    if skipped:
        return {"skipped": "only one class", "n_stays": 3}
    one = {"value": auroc, "low": auroc - 0.1, "high": auroc + 0.1}
    return {
        "n_stays": 20,
        "n_patients": 15,
        **{m: dict(one) for m in ("auroc", "auprc", "brier", "ece", "mae_days", "rmse_days")},
        "pred_short": {"mae_days": {"value": 1.0}, "rmse_days": {"value": 2.0}},
    }


def test_seed_summaries_report_mean_sd_and_the_mean_interval():
    out = _summarise([_result(0.6), _result(0.8)])
    assert out["n_seeds"] == 2 and out["auroc"]["mean"] == pytest.approx(0.7)
    assert out["auroc"]["sd_over_seeds"] == pytest.approx(0.1414, abs=1e-3)
    assert out["auroc"]["ci_low"] == pytest.approx(0.6) and out["auroc"]["ci_high"] == pytest.approx(0.8)
    assert out["pred_short"]["mae_days"]["mean"] == 1.0
    assert _summarise([_result(0.6)])["auroc"]["sd_over_seeds"] is None  # one seed has no spread


def test_skipped_seeds_are_not_averaged_in():
    assert _summarise([_result(0, skipped=True)]) == {"skipped": "only one class"}
    out = _summarise([_result(0, skipped=True), _result(0.8)])
    assert out["n_seeds"] == 1 and out["auroc"]["mean"] == pytest.approx(0.8)


@pytest.mark.demo
def test_the_smoke_config_runs_end_to_end_on_the_demo(tmp_path, monkeypatch, settings):
    monkeypatch.setenv("FATE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("FATE_CHECKPOINTS_DIR", str(tmp_path / "checkpoints"))
    config = load_run_config(REPO_ROOT / "configs/models/smoke.yaml").model_copy(
        update={"n_seeds": 1, "arms": ["B0", "B1", "B4", "B5", "M"], "n_boot": 20}
    )
    from fate_icu.utils.env import load_settings

    report = run(config, settings=load_settings())
    assert set(report["arms"]) == {"B0", "B1", "B4", "B5", "M"}
    assert set(report["arms"]["B4"]) == {"P1"}  # no image, no P2 prediction
    sizes = {(arm, pop): r["n_stays"] for arm, pops in report["arms"].items() for pop, r in pops.items()}
    assert len({n for (_, pop), n in sizes.items() if pop == "P2"}) == 1  # same test stays in every arm
    assert len({n for (_, pop), n in sizes.items() if pop == "P1"}) == 1
    out = tmp_path / "outputs" / "models"
    folder = next(out.iterdir())
    assert json.loads((folder / "run_manifest.json").read_text())["seeds"] == [666]
    assert (folder / "metrics.json").exists() and (folder / "predictions_M_seed666.npz").exists()
    checkpoints = tmp_path / "checkpoints" / "models" / folder.name
    assert (checkpoints / "M_seed666.pt").exists() and (checkpoints / "B1_seed666.pt").exists()
    assert not (checkpoints / "B5_seed666.pt").exists()  # late fusion is a mean, not a network
    inputs = json.loads((checkpoints / "inputs.json").read_text())
    assert inputs["run_id"] == folder.name and inputs["threshold_source"] == "train_q3"
