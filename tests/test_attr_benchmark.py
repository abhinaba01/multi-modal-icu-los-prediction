"""The benchmark runner: its machinery on a small, fast configuration (the real one is a 5 minute run)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fate_icu.attribution.benchmark import BenchmarkConfig, _interval, load_config, run, summary_table
from fate_icu.utils.env import REPO_ROOT

FAST = dict(
    seed=3, n_stays=600, n_eval=12, epochs=15, lr=0.003, min_auroc=0.0,
    model={"d": 16, "heads": 2, "layers": 1, "dropout": 0.1}, ig_steps=8, mask_epochs=10, dynamask_epochs=5,
    ts_methods=["occlusion", "ig"], static_methods=["ablation"], image_methods=["ig"],
    combos=[["ig", "ig", "ig"]], n_boot=20,
)  # fmt: skip


def test_the_committed_config_is_the_tuned_one():
    cfg = load_config(REPO_ROOT / "configs/attribution/benchmark.yaml")
    assert (
        cfg.mask_lambda_1 == 3.0 and cfg.mask_epochs == 500 and cfg.n_stays >= 2000 and cfg.min_auroc >= 0.9
    )


def test_a_model_that_has_not_learned_the_rule_is_not_explained(tmp_path, monkeypatch):
    monkeypatch.setenv("FATE_OUTPUTS_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="planted rule"):
        run(BenchmarkConfig(**{**FAST, "min_auroc": 0.999, "epochs": 1}))


def test_a_small_run_writes_results_and_a_readable_table(tmp_path, monkeypatch):
    monkeypatch.setenv("FATE_OUTPUTS_DIR", str(tmp_path))
    results = run(BenchmarkConfig(**FAST))
    assert set(results["modalities"]) <= {"timeseries", "static", "image"} and results["n_explained"] >= 10
    ts = results["modalities"]["timeseries"]["methods"]
    assert set(ts) == {"occlusion", "ig"}
    for metric in (
        "precision@1",
        "recall@3",
        "comprehensiveness@3",
        "random_comprehensiveness@3",
        "deletion_aopc@5",
    ):
        assert {"mean", "low", "high", "n"} <= set(ts["occlusion"][metric])
    assert set(results["selection"]) == {"ig+ig+ig"} and set(results["randomisation_spearman"]) == {
        "ig+ig+ig"
    }
    assert results["tau"] > 0
    written = json.loads((Path(results["out_dir"]) / "benchmark.json").read_text(encoding="utf-8"))
    assert written["n_explained"] == results["n_explained"]
    assert (Path(results["out_dir"]) / "run_manifest.json").exists()
    table = summary_table(results)
    assert "occlusion" in table and "ig+ig+ig" in table and "tau" in table


def test_intervals_are_means_with_bootstrap_bounds_and_ignore_nan():
    out = _interval([0.0, 1.0, 1.0, float("nan")], 200, 0)
    assert out["n"] == 3 and out["mean"] == pytest.approx(2 / 3) and out["low"] <= out["mean"] <= out["high"]
    assert _interval([float("nan")], 10, 0)["n"] == 0
