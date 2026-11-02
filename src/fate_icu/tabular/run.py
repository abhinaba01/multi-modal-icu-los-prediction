"""B0 end to end: contract cohort -> features -> FaTE split -> two-stage XGB -> SHAP.

    python -m fate_icu.tabular.run --config configs/tabular/smoke.yaml   # laptop, demo DB
    python -m fate_icu.tabular.run --config configs/tabular/full.yaml    # lab

Writes ``<outputs>/tabular/<run_id>/`` (run_manifest.json, metrics.json) and prints
aggregates only: counts below 10 are suppressed, no identifiers or rows (CLAUDE.md A3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from fate_icu.data.contract import Contract, load_contract
from fate_icu.data.splits import SPLIT_ORDER
from fate_icu.data.targets import long_stay_labels, resolve_long_stay_threshold
from fate_icu.tabular._legacy import legacy_config, legacy_tree_sha256
from fate_icu.tabular.api import TabularTables, load_splits, load_tables
from fate_icu.tabular.model import explain_shap, train_two_stage_xgb
from fate_icu.utils.aggregates import format_aggregates, safe_share, suppress, suppress_counts
from fate_icu.utils.device import get_device
from fate_icu.utils.env import load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest
from fate_icu.utils.seed import seed_everything


class TabularRunConfig(BaseModel):
    """``configs/tabular/{smoke,full}.yaml``."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    source: Literal["postgres", "csv_demo", "csv_raw"]
    feature_sets: list[str]
    require_frozen_threshold: bool
    seed: int
    xgb_n_estimators: int | None = None  # None: the contract's B0 value
    n_jobs: int = -1
    cache: bool = True
    strict_sofa: bool = True
    shap_top_k: int = 10


def load_run_config(path: str | Path) -> TabularRunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TabularRunConfig.model_validate(yaml.safe_load(handle))


def split_summary(tables: TabularTables, split: pd.Series, y_long: pd.Series) -> dict[str, Any]:
    """Stays, patients and long-stay share per split (suppressed)."""
    out = {}
    for name in SPLIT_ORDER:
        mask = split == name
        out[name] = {
            "stays": suppress(int(mask.sum())),
            "patients": suppress(int(tables.cohort.loc[mask, "subject_id"].nunique())),
            "long_stay_share": safe_share(int(y_long[mask].sum()), int(mask.sum())),
        }
    return out


def _suppress_eval(metrics: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    out["n_stays"] = suppress(metrics["n_stays"])
    out["n_long"] = suppress(metrics["n_long"])
    out["stepwise_confusion"] = suppress_counts(metrics["stepwise_confusion"])
    regression = dict(metrics["stepwise_regression"])
    regression["n_predicted_short"] = suppress(regression["n_predicted_short"])
    out["stepwise_regression"] = regression
    return out


def run(config: TabularRunConfig, contract: Contract | None = None) -> dict[str, Any]:
    contract = contract or load_contract()
    settings = load_settings()
    run_id = new_run_id(config.run_name)
    out_dir = settings.outputs_dir / "tabular" / run_id
    seed_everything(config.seed)

    tables = load_tables(
        config.source,
        contract=contract,
        settings=settings,
        cache=config.cache,
        strict_sofa=config.strict_sofa,
    )
    split = load_splits(tables.cohort, contract)
    threshold, threshold_source = resolve_long_stay_threshold(
        tables.los, split, contract, config.require_frozen_threshold
    )
    y_long = long_stay_labels(tables.los, threshold, contract)
    cfg = legacy_config(
        contract,
        gem_path=settings.icd9_gem_path,
        random_state=config.seed,
        xgb_n_estimators=config.xgb_n_estimators,
        n_jobs=config.n_jobs,
    )

    results: dict[str, Any] = {
        "run_id": run_id,
        "cohort_flow": suppress_counts(tables.flow),
        "splits": split_summary(tables, split, y_long),
        "long_stay_threshold_days": round(threshold, 4),
        "threshold_source": threshold_source,
        "models": {},
    }
    for feature_set in config.feature_sets:
        X = tables.features[contract.feature_names(feature_set)]
        model = train_two_stage_xgb(
            X, tables.los, split, threshold_days=threshold, contract=contract, legacy_cfg=cfg
        )
        entry: dict[str, Any] = {"n_features": X.shape[1], "gmean_threshold": round(model.gmean_threshold, 4)}
        for name in ("val", "test"):
            mask = split == name
            if mask.sum() >= 2:
                entry[name] = _suppress_eval(model.evaluate(X[mask], tables.los[mask]))
        test = split == "test"
        if test.sum() >= 1:
            shap = explain_shap(model, X[test])
            mean_abs = shap.values.abs().mean().sort_values(ascending=False)
            entry["shap_mean_abs_by_group_test"] = (
                shap.group_values.abs().mean().sort_values(ascending=False).round(4).to_dict()
            )
            entry["shap_top_features_test"] = mean_abs.head(config.shap_top_k).round(4).to_dict()
        results["models"][feature_set] = entry

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, default=_json_default), encoding="utf-8"
    )
    write_run_manifest(
        out_dir,
        run_id=run_id,
        config=config.model_dump(),
        seeds=[config.seed],
        contract_sha256=contract.sha256,
        legacy_tree_sha256=legacy_tree_sha256(),
        data={
            "source": config.source,
            "db": settings.db_url_redacted,
            "mimic_version": contract.mimic.demo_version
            if settings.machine == "laptop"
            else contract.mimic.version,
        },
        model_revisions={
            "b0": f"legacy two-stage {contract.b0.prior} (legacy_los {contract.provenance.legacy_tag})"
        },
        device=get_device(settings.device),
        extra={"machine": settings.machine},
    )
    return results


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/tabular/smoke.yaml or full.yaml")
    args = parser.parse_args(argv)
    results = run(load_run_config(args.config))
    print(format_aggregates(results))


if __name__ == "__main__":
    main()
