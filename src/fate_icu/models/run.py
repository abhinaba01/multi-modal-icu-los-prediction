"""Stage 3 end to end: inputs -> every arm x seed -> metrics with patient-level CIs.

    python -m fate_icu.models.run --config configs/models/smoke.yaml   # laptop CPU, demo DB
    python -m fate_icu.models.run --config configs/models/full.yaml    # lab

Inputs (built here when not given): the contract cohort and static features (Stage 0), the
hourly tensors (Stage 2) and the image features (Stage 1 links, then the frozen DenseNet).
Writes ``<outputs>/models/<run_id>/``: ``metrics.json``, ``run_manifest.json`` and one
``predictions_<arm>_seed<k>.npz`` per arm and seed (stay-level derived data: it stays in the
git-ignored outputs folder and is never printed). The console shows aggregates only: counts
below 10 are suppressed, no identifiers or rows (CLAUDE.md A3).

Smoke runs use the 4 public chest X-rays, handed to the stays that have a (synthetic)
eligible study by a hash of the study's id: this never sees a label, but it is not a real
pairing, so smoke image results mean nothing. Full runs use the patient's own images only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from fate_icu.data.contract import Contract, load_contract
from fate_icu.data.cxr import link as cxr_link
from fate_icu.data.cxr.cache import fixture_items, mimic_items
from fate_icu.data.cxr.preprocess import build_cache
from fate_icu.data.targets import resolve_long_stay_threshold
from fate_icu.data.timeseries import build as ts_build
from fate_icu.models.arms import ARMS, POPULATIONS, evaluate_arm, run_arms, sized
from fate_icu.models.data import StayData, assemble
from fate_icu.models.image import WEIGHTS, cached_features, load_image_encoder
from fate_icu.models.train import TrainConfig, save_trained
from fate_icu.tabular._legacy import legacy_config, legacy_tree_sha256
from fate_icu.tabular.api import TabularTables, load_splits, load_tables
from fate_icu.tabular.model import train_two_stage_xgb
from fate_icu.utils.aggregates import format_aggregates, suppress
from fate_icu.utils.env import REPO_ROOT, Settings, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest
from fate_icu.utils.seed import run_seeds, seed_everything

NEURAL_ARMS = tuple(ARMS)


class ModelSize(BaseModel):
    model_config = ConfigDict(extra="forbid")
    d: int = 256
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1


class ModelsRunConfig(BaseModel):
    """``configs/models/{smoke,full}.yaml``."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    source: Literal["postgres", "csv_demo", "csv_raw"]
    require_frozen_threshold: bool
    require_signal: bool = True  # stop when a static feature is constant in train (smoke: False)
    seed: int
    n_seeds: int = Field(default=5, ge=1)
    arms: list[str] = Field(default_factory=lambda: ["B0", *NEURAL_ARMS])
    model: ModelSize = ModelSize()
    train: dict[str, Any] = Field(default_factory=dict)  # TrainConfig fields
    n_boot: int = Field(default=1000, ge=10)
    timeseries_config: str
    cxr_config: str
    # Reuse a finished Stage 2 / Stage 1 run (run id under artifacts/, or a folder) instead of
    # building it again; the lab config sets these so a model run never rebuilds its inputs.
    timeseries_run: str | None = None
    cxr_run: str | None = None
    images: Literal["fixtures", "mimic"]
    image_weights: str = WEIGHTS
    cache: bool = True
    strict_sofa: bool = True
    n_jobs: int = -1
    xgb_n_estimators: int | None = None


def load_run_config(path: str | Path) -> ModelsRunConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ModelsRunConfig.model_validate(yaml.safe_load(handle))


def resolve_run(settings: Settings, kind: str, value: str) -> Path:
    """A finished run folder: ``value`` is a folder, or a run id under ``<artifacts>/<kind>/``."""
    path = Path(value)
    folder = path if path.is_dir() else settings.artifacts_dir / kind / value
    if not folder.is_dir():
        raise FileNotFoundError(f"no {kind} run at {folder}: run that stage first (docs/lab_runbook.md)")
    return folder


def smoke_image_keys(dicom_ids: list[str], keys: list[str]) -> list[str]:
    """Public test image for each study id, by a hash of the id alone (smoke runs only)."""
    return [keys[int(hashlib.sha256(str(d).encode()).hexdigest()[:8], 16) % len(keys)] for d in dicom_ids]


def image_features_by_stay(
    config: ModelsRunConfig, settings: Settings, paired: pd.DataFrame
) -> tuple[dict[int, np.ndarray], int, dict[str, Any], dict[int, str]]:
    """Feature vector per stay with an eligible X-ray, the vector length, a short report and the
    image-cache key of each stay's image (for Stage 4's Grad-CAM)."""
    encoder = load_image_encoder(config.image_weights)
    cache_dir = settings.artifacts_dir / "cxr_cache"
    feature_dir = settings.artifacts_dir / "cxr_features"
    linked = paired[paired["has_cxr"]]
    if config.images == "fixtures":
        items = fixture_items(REPO_ROOT / "tests" / "fixtures" / "cxr")
        summary = build_cache(items, cache_dir)
        if summary.missing or summary.failed:
            raise RuntimeError("some public test images are missing or unreadable")
        keys = smoke_image_keys(linked["dicom_id"].astype(str).tolist(), [k for k, _ in items])
    else:
        if settings.mimic_cxr_dir is None:
            raise RuntimeError("MIMIC_CXR_DIR is not set (see .env.example)")
        summary = build_cache(mimic_items(paired, settings.mimic_cxr_dir), cache_dir)
        if summary.missing or summary.failed:
            raise RuntimeError("some selected images are missing or unreadable: run the Stage 1 cache step")
        keys = linked["dicom_id"].astype(str).tolist()
    features = cached_features(keys, cache_dir, feature_dir, encoder)
    by_stay = {int(s): features[k] for s, k in zip(linked["stay_id"], keys, strict=True)}
    key_of = {int(s): k for s, k in zip(linked["stay_id"], keys, strict=True)}
    report = {"weights": encoder.weights, "paired_stays": suppress(len(by_stay))}
    return by_stay, encoder.dim, report, key_of


def b0_predictions(
    tables: TabularTables,
    split: pd.Series,
    data: StayData,
    contract: Contract,
    settings: Settings,
    config: ModelsRunConfig,
    threshold: float,
) -> dict[str, np.ndarray]:
    """B0 (the internship's two-stage XGBoost on the FaTE split) for every stay in ``data``."""
    X = tables.features[contract.feature_names("fate_static")]
    cfg = legacy_config(
        contract,
        gem_path=settings.icd9_gem_path,
        random_state=config.seed,
        xgb_n_estimators=config.xgb_n_estimators,
        n_jobs=config.n_jobs,
    )
    model = train_two_stage_xgb(
        X, tables.los, split, threshold_days=threshold, contract=contract, legacy_cfg=cfg
    )
    out = model.predict(X.loc[data.stay_ids])
    p = np.clip(out["p_long"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    return {"p_long": p, "logit": np.log(p / (1 - p)), "los_days": out["los_days"].to_numpy(dtype=float)}


def _summarise(per_seed: list[dict]) -> dict[str, Any]:
    """Mean and SD across seeds of every metric value, and the mean CI bounds."""
    usable = [r for r in per_seed if "skipped" not in r]
    if not usable:
        return {"skipped": per_seed[0].get("skipped", "no usable seed") if per_seed else "no seed"}
    out: dict[str, Any] = {
        "n_stays": suppress(usable[0]["n_stays"]),
        "n_patients": suppress(usable[0]["n_patients"]),
        "n_seeds": len(usable),
    }
    for metric in ("auroc", "auprc", "brier", "ece", "mae_days", "rmse_days"):
        values = np.array([r[metric]["value"] for r in usable])
        low = np.array([r[metric]["low"] for r in usable])
        high = np.array([r[metric]["high"] for r in usable])
        out[metric] = {
            "mean": float(values.mean()),
            "sd_over_seeds": float(values.std(ddof=1)) if len(values) > 1 else None,
            "ci_low": float(np.nanmean(low)),
            "ci_high": float(np.nanmean(high)),
        }
    shorts = [r["pred_short"] for r in usable if "pred_short" in r]
    if shorts:
        out["pred_short"] = {
            m: {"mean": float(np.mean([s[m]["value"] for s in shorts]))} for m in ("mae_days", "rmse_days")
        }
    return out


@dataclass
class Prepared:
    """Everything a model run (or a Stage 4 explanation run) needs, built once."""

    tables: TabularTables
    split: pd.Series
    threshold: float
    threshold_source: str
    data: StayData
    encoding: Any
    ts_dir: Path
    ts_report: dict
    link_dir: Path
    image_report: dict
    raw: np.ndarray  # [N, T, C] hourly values in original units, NaN = not measured
    image_keys: dict[int, str]  # stay_id -> key of its image in the image cache
    image_cache_dir: Path


def prepare_inputs(
    config: ModelsRunConfig,
    contract: Contract,
    settings: Settings,
    tables: TabularTables | None = None,
) -> Prepared:
    """Cohort, split, threshold, hourly tensors, image features: the ``StayData`` of a run."""
    tables = tables or load_tables(
        config.source,
        contract=contract,
        settings=settings,
        cache=config.cache,
        strict_sofa=config.strict_sofa,
    )
    cohort = tables.cohort.sort_index()
    split = load_splits(cohort, contract)
    threshold, threshold_source = resolve_long_stay_threshold(
        tables.los, split, contract, config.require_frozen_threshold
    )

    if config.timeseries_run:
        ts_dir = resolve_run(settings, "timeseries", config.timeseries_run)
        ts_report = json.loads((ts_dir / "report.json").read_text(encoding="utf-8"))
    else:
        ts_report = ts_build.run(
            ts_build.load_config(REPO_ROOT / config.timeseries_config), contract, tables=tables
        )
        ts_dir = Path(ts_report["out_dir"])
    tensors = ts_build.load_tensors(ts_dir)
    if config.cxr_run:
        link_dir = resolve_run(settings, "cxr", config.cxr_run)
    else:
        link_report = cxr_link.run(
            cxr_link.load_config(REPO_ROOT / config.cxr_config), contract, tables=tables, settings=settings
        )
        link_dir = Path(link_report["out_dir"])
    paired = pd.read_parquet(link_dir / cxr_link.PAIRED_FILE)
    image_features, image_dim, image_report, image_keys = image_features_by_stay(config, settings, paired)
    unknown_stays = set(image_features) - set(tensors["stay_ids"].tolist())
    image_features = {s: v for s, v in image_features.items() if s not in unknown_stays}

    data, encoding = assemble(
        cohort=cohort,
        features=tables.features[contract.feature_names("fate_static")],
        split=split,
        tensors=tensors,
        image_features=image_features,
        image_dim=image_dim,
        contract=contract,
        threshold_days=threshold,
        require_signal=config.require_signal,
    )

    return Prepared(
        tables, split, threshold, threshold_source, data, encoding, ts_dir, ts_report, link_dir, image_report,
        np.asarray(tensors["raw"]), image_keys, settings.artifacts_dir / "cxr_cache",
    )  # fmt: skip


def run(
    config: ModelsRunConfig,
    contract: Contract | None = None,
    *,
    tables: TabularTables | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    contract = contract or load_contract()
    settings = settings or load_settings()
    run_id = new_run_id(config.run_name)
    out_dir = settings.outputs_dir / "models" / run_id
    seed_everything(config.seed)
    unknown = [a for a in config.arms if a != "B0" and a not in ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; known: B0, {', '.join(ARMS)}")

    prep = prepare_inputs(config, contract, settings, tables)
    tables, split, threshold, threshold_source = (
        prep.tables,
        prep.split,
        prep.threshold,
        prep.threshold_source,
    )
    data, encoding, ts_report, image_report = prep.data, prep.encoding, prep.ts_report, prep.image_report

    train_cfg = TrainConfig(**config.train)
    arms = {
        name: sized(arm, config.model.d, config.model.heads, config.model.layers, config.model.dropout)
        for name, arm in ARMS.items()
    }
    seeds = run_seeds(config.seed, config.n_seeds)
    neural = [a for a in config.arms if a != "B0"]
    out_dir.mkdir(parents=True, exist_ok=True)

    per_seed: dict[str, dict[str, list[dict]]] = {a: {p: [] for p in POPULATIONS} for a in config.arms}
    for seed in seeds:
        trained: dict = {}
        predictions = run_arms(neural, data, train_cfg, seed, arms, trained) if neural else {}
        for name, model in trained.items():
            save_trained(
                model,
                settings.checkpoints_dir / "models" / run_id / f"{name}_seed{seed}.pt",
                data.dims,
                extra={"arm": name, "seed": seed, "run_id": run_id},
            )
        if "B0" in config.arms:
            predictions["B0"] = b0_predictions(tables, split, data, contract, settings, config, threshold)
        for name, pred in predictions.items():
            np.savez_compressed(
                out_dir / f"predictions_{name}_seed{seed}.npz", stay_ids=data.stay_ids, **pred
            )
            populations = ARMS[name].populations if name in ARMS else POPULATIONS
            for population in populations:
                per_seed[name][population].append(
                    evaluate_arm(pred, data, population, n_boot=config.n_boot, seed=config.seed)
                )

    inputs_file = settings.checkpoints_dir / "models" / run_id / "inputs.json"
    inputs_file.parent.mkdir(parents=True, exist_ok=True)
    inputs_file.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": config.model_dump(),
                "timeseries_run": str(prep.ts_dir),
                "cxr_run": str(prep.link_dir),
                "threshold_days": threshold,
                "threshold_source": threshold_source,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    results: dict[str, Any] = {
        "run_id": run_id,
        "long_stay_threshold_days": round(threshold, 4),
        "threshold_source": threshold_source,
        "stays": suppress(len(data)),
        "splits": {s: suppress(int((data.split == s).sum())) for s in ("train", "val", "test")},
        "stays_with_image": suppress(int(data.has_image.sum())),
        "image": image_report,
        "static_inputs": data.dims["static"],
        "constant_static_features": suppress(len(encoding.constant_columns)),
        "seeds": seeds,
        "arms": {
            name: {pop: _summarise(runs) for pop, runs in populations.items() if runs}
            for name, populations in per_seed.items()
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    write_run_manifest(
        out_dir,
        run_id=run_id,
        config=config.model_dump(),
        seeds=seeds,
        contract_sha256=contract.sha256,
        legacy_tree_sha256=legacy_tree_sha256(),
        data={
            "source": config.source,
            "db": settings.db_url_redacted,
            "timeseries_tensor_sha256": ts_report["tensor_sha256"],
            "image_weights": config.image_weights,
        },
        extra={"machine": settings.machine, "out_dir": str(out_dir)},
    )
    results["out_dir"] = str(out_dir)
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config)
    report = run(load_run_config(path if path.is_absolute() else REPO_ROOT / path))
    print(format_aggregates(report))


if __name__ == "__main__":
    main()
