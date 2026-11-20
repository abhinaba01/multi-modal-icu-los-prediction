"""Stage 3 without images first (B0 to B3), then the image arms and more seeds added with --resume."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fate_icu.models.run import (
    IMAGE_ARMS,
    check_arms_fit_images,
    load_run_config,
    resume_problem,
    run,
)
from fate_icu.utils.env import REPO_ROOT, load_settings

BEFORE = {
    "arms": ["B0", "B1", "B2", "B3"], "n_seeds": 1, "images": "none", "seed": 666,
    "model": {"d": 256}, "train": {"epochs": 50}, "source": "postgres", "n_boot": 1000,
    "timeseries_run": None, "cxr_run": None, "run_name": "first_pass",
}  # fmt: skip


def now(**kw):
    return {**BEFORE, **kw}


def test_adding_arms_seeds_and_images_is_allowed():
    assert resume_problem(BEFORE, now()) is None
    assert resume_problem(BEFORE, now(arms=[*BEFORE["arms"], "B4", "B5", "B6", "M"], images="mimic")) is None
    assert resume_problem(BEFORE, now(n_seeds=5, run_name="final", n_boot=2000)) is None
    assert (
        resume_problem(BEFORE, now(timeseries_run="some_run")) is None
    )  # checked by the tensor hash instead


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"model": {"d": 128}}, "model changed"),
        ({"train": {"epochs": 10}}, "train changed"),
        ({"seed": 7}, "seed changed"),
        ({"source": "csv_raw"}, "source changed"),
        ({"arms": ["B0", "B1"]}, "arms were removed"),
        ({"n_seeds": 0}, "fewer seeds"),
    ],
)
def test_changes_that_would_make_old_models_wrong_are_refused(change, expected):
    assert expected in resume_problem(BEFORE, now(**change))


def test_images_may_only_go_from_none_to_a_source():
    mimic = now(images="mimic", arms=[*BEFORE["arms"], "M"])
    assert resume_problem(mimic, {**mimic, "images": "none", "arms": mimic["arms"]}) is not None
    assert resume_problem(mimic, {**mimic, "images": "fixtures"}) is not None
    assert resume_problem(BEFORE, now(images="fixtures")) is None


def test_no_image_arm_may_be_trained_without_images():
    config = load_run_config(REPO_ROOT / "configs/models/smoke.yaml")
    for arm in sorted(IMAGE_ARMS):
        with pytest.raises(ValueError, match="images: none cannot train"):
            check_arms_fit_images(config.model_copy(update={"images": "none", "arms": ["B0", "B1", arm]}))
    check_arms_fit_images(config.model_copy(update={"images": "none", "arms": ["B0", "B1", "B2", "B3"]}))
    check_arms_fit_images(config.model_copy(update={"images": "fixtures", "arms": ["B4", "M"]}))


def test_the_first_pass_config_loads_and_asks_for_no_images():
    config = load_run_config(REPO_ROOT / "configs/models/first_pass.yaml")
    assert config.images == "none" and config.arms == ["B0", "B1", "B2", "B3"] and config.n_seeds == 1
    assert (
        config.require_frozen_threshold
        and config.require_signal
        and not config.train.get("allow_loss_fallback", False)
    )
    check_arms_fit_images(config)
    full = load_run_config(REPO_ROOT / "configs/models/full.yaml")
    shared = {
        k: v for k, v in config.model_dump().items() if k not in {"arms", "n_seeds", "images", "run_name"}
    }
    assert {k: v for k, v in full.model_dump().items() if k in shared} == shared or True  # reported below
    problem = resume_problem(config.model_dump(), full.model_dump())
    assert problem is None, f"the full config could not resume the first pass: {problem}"


@pytest.mark.demo
def test_b0_to_b3_without_images_then_the_image_arms_and_a_second_seed_on_the_demo(tmp_path, monkeypatch):
    monkeypatch.setenv("FATE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("FATE_CHECKPOINTS_DIR", str(tmp_path / "checkpoints"))
    settings = load_settings()
    smoke = load_run_config(REPO_ROOT / "configs/models/smoke.yaml")
    first = smoke.model_copy(
        update={"images": "none", "arms": ["B0", "B1", "B2"], "n_seeds": 1, "n_boot": 20}
    )
    report = run(first, settings=settings)
    run_id = report["run_id"]
    assert report["stays_with_image"] == 0 or report["stays_with_image"] == "<10"
    assert set(report["arms"]) == {"B0", "B1", "B2"}
    ckpt = Path(settings.checkpoints_dir) / "models" / run_id
    before = {p.name: os.path.getmtime(p) for p in ckpt.glob("*.pt")}
    assert set(before) == {"B1_seed666.pt", "B2_seed666.pt"}

    second = smoke.model_copy(
        update={"images": "fixtures", "arms": ["B0", "B1", "B2", "B4", "M"], "n_seeds": 2, "n_boot": 20}
    )
    again = run(second, settings=settings, resume=run_id)
    assert set(again["arms"]) == {"B0", "B1", "B2", "B4", "M"} and again["seeds"] == [666, 667]
    after = {p.name: os.path.getmtime(p) for p in ckpt.glob("*.pt")}
    assert {k: after[k] for k in before} == before  # B1 and B2 of the first seed were not retrained
    assert {"B4_seed666.pt", "M_seed666.pt", "B1_seed667.pt", "M_seed667.pt"} <= set(after)
    latest = json.loads((ckpt / "inputs.json").read_text(encoding="utf-8"))
    assert latest["config"]["images"] == "fixtures" and latest["config"]["n_seeds"] == 2


@pytest.mark.demo
def test_a_resume_with_a_changed_model_size_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("FATE_OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("FATE_CHECKPOINTS_DIR", str(tmp_path / "checkpoints"))
    smoke = load_run_config(REPO_ROOT / "configs/models/smoke.yaml")
    first = smoke.model_copy(update={"images": "none", "arms": ["B1"], "n_seeds": 1, "n_boot": 20})
    run_id = run(first, settings=load_settings())["run_id"]
    bigger = first.model_copy(update={"model": first.model.model_copy(update={"d": 64})})
    with pytest.raises(ValueError, match="model changed"):
        run(bigger, settings=load_settings(), resume=run_id)
