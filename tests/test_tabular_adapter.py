"""The Stage 0 adapter API end to end on synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from fate_icu.data.targets import resolve_long_stay_threshold
from fate_icu.tabular import (
    explain_shap,
    load_cohort,
    load_splits,
    load_static_features,
    train_two_stage_xgb,
)
from fate_icu.tabular._legacy import legacy_config
from fate_icu.tabular.run import TabularRunConfig


@pytest.fixture(scope="module")
def fitted(synth, synth_dir, synth_tables, contract):
    X = load_static_features("fate_static", tables=synth_tables, contract=contract)
    split = load_splits(synth_tables.cohort, contract)
    threshold, _ = resolve_long_stay_threshold(synth_tables.los, split, contract, require_frozen=False)
    cfg = legacy_config(contract, gem_path=synth.gem_path(synth_dir), xgb_n_estimators=20, n_jobs=1)
    model = train_two_stage_xgb(
        X, synth_tables.los, split, threshold_days=threshold, contract=contract, legacy_cfg=cfg
    )
    return X, split, model


def test_cohort_and_features(synth_tables, contract):
    cohort = load_cohort(tables=synth_tables)
    assert cohort.index.name == "stay_id"
    assert ((cohort["t0"] - cohort["intime"]) == pd.Timedelta(hours=contract.window.hours)).all()
    X = load_static_features("fate_static", tables=synth_tables, contract=contract)
    assert list(X.columns) == contract.feature_names("fate_static")
    assert X.index.equals(cohort.index)
    assert not X.isna().any().any()


def test_model_fits_on_train_only(fitted, synth_tables):
    X, split, model = fitted
    train = split == "train"
    encoded = model.preprocessor._encode(X[train], fit=False)
    pd.testing.assert_series_equal(model.preprocessor.mins, encoded.min())
    assert 0.0 < model.gmean_threshold < 1.0
    assert model.regressor is not None


def test_predict_and_evaluate(fitted, synth_tables):
    X, split, model = fitted
    pred = model.predict(X)
    assert list(pred.columns) == ["p_long", "pred_long_05", "pred_long_gmean", "los_days"]
    assert pred["p_long"].between(0, 1).all()
    test = split == "test"
    report = model.evaluate(X[test], synth_tables.los[test])
    assert report["n_stays"] == int(test.sum())
    assert {"accuracy", "f1", "auc", "auprc", "brier"} <= set(report["classification"])
    assert sum(report["stepwise_confusion"].values()) == int(test.sum())


def test_shap_is_additive_and_grouped(fitted, contract):
    X, _, model = fitted
    shap = explain_shap(model, X)
    margin = model.classifier.get_booster().predict(xgb.DMatrix(model.transform(X)), output_margin=True)
    np.testing.assert_allclose(shap.values.sum(axis=1) + shap.base_value, margin, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(shap.group_values.sum(axis=1), shap.values.sum(axis=1), rtol=1e-6, atol=1e-6)
    assert set(shap.group_values.columns) <= set(contract.feature_groups)
    assert "diagnoses" not in shap.group_values.columns


def test_run_writes_manifest_and_metrics(synth, synth_dir, synth_tables, contract, tmp_path, monkeypatch):
    import fate_icu.tabular.run as run_mod

    monkeypatch.setenv("FATE_OUTPUTS_DIR", str(tmp_path))
    monkeypatch.setenv("ICD9_GEM_PATH", str(synth.gem_path(synth_dir)))
    monkeypatch.setenv("FATE_DB_URL", "postgresql+psycopg://alice:s3cret@localhost:5432/mimiciv_demo")
    monkeypatch.setattr(run_mod, "load_tables", lambda *a, **k: synth_tables)
    config = TabularRunConfig(
        run_name="unit",
        source="postgres",
        feature_sets=["fate_static"],
        require_frozen_threshold=False,
        seed=666,
        xgb_n_estimators=10,
        n_jobs=1,
    )
    results = run_mod.run(config, contract)

    run_dir = tmp_path / "tabular" / results["run_id"]
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert manifest["contract_sha256"] == contract.sha256
    assert manifest["legacy_tree_sha256"] == contract.provenance.legacy_tree_sha256
    assert manifest["seeds"] == [666]
    assert "s3cret" not in json.dumps(manifest) and "alice" not in json.dumps(manifest)
    assert metrics["threshold_source"] == "train_q3"
    assert "fate_static" in metrics["models"]
    # counts below 10 are suppressed in every printed/saved aggregate
    for split in metrics["splits"].values():
        assert isinstance(split["stays"], str) or split["stays"] >= 10
