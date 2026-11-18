"""Evidence packets for a split of stays (Stage 4).

    python -m fate_icu.attribution.build_packets --config configs/attribution/smoke.yaml   # laptop, demo DB
    python -m fate_icu.attribution.build_packets --config configs/attribution/full.yaml    # lab

Takes the inputs of a Stage 3 model run (``models.run.prepare_inputs``), the trained model
(a Stage 3 checkpoint, or trained here for smoke runs), fits ``tau`` on validation stays, and
writes one ``packet_<stay_id>.json`` per explained stay under ``<outputs>/attribution/<run id>/
packets/``. Packets hold patient information derived from credentialed data in a real run, so
they stay in the git-ignored outputs folder; the console shows aggregates only, with counts
below 10 suppressed (CLAUDE.md A3).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from fate_icu.attribution import methods as M
from fate_icu.attribution.explain import Explainer, ImageContext
from fate_icu.attribution.gates import GatedLogit, build_layout
from fate_icu.attribution.image_cam import operating_thresholds
from fate_icu.attribution.packet import check_keys
from fate_icu.attribution.select import SelectionConfig
from fate_icu.attribution.units import UnitSpace
from fate_icu.data.contract import load_contract
from fate_icu.data.timeseries.variables import load_variables
from fate_icu.models.arms import ARMS, sized
from fate_icu.models.image import load_image_encoder
from fate_icu.models.run import load_run_config, prepare_inputs
from fate_icu.models.train import TrainConfig, load_trained, train_model
from fate_icu.rag.drivers import driver_keys
from fate_icu.utils.aggregates import format_aggregates, suppress, suppress_counts
from fate_icu.utils.env import REPO_ROOT, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest
from fate_icu.utils.seed import seed_everything


class PacketsConfig(BaseModel):
    """``configs/attribution/{smoke,full}.yaml``."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    models_config: str  # the Stage 3 run config whose inputs and model are explained
    checkpoint_run: str | None = None  # a Stage 3 run id; None trains the arm here (smoke)
    arm: str = "M"
    seed: int = 666
    split: str = "test"
    n_stays: int | None = None  # explain the first n stays of the split (None: all)
    n_tau_stays: int | None = None  # validation stays to fit tau on (None: all)
    # default methods (D-048): ExtremalMask for the time series, IG for static groups and concepts
    ts_method: str = "extremal_mask"
    static_method: str = "ig"
    concept_method: str = "ig"
    ig_steps: int = 32
    mask_epochs: int = 500
    mask_lr: float = 0.01
    mask_lambda_1: float = 3.0
    dynamask_epochs: int = 200
    per_modality: int = Field(default=5, ge=1)
    max_drivers: int = Field(default=5, ge=1, le=5)
    tau_quantile: float = Field(default=0.95, gt=0, lt=1)
    n_random_units: int = Field(default=20, ge=1)


def load_config(path: str | Path) -> PacketsConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PacketsConfig.model_validate(yaml.safe_load(handle))


def run(config: PacketsConfig) -> dict:
    contract = load_contract()
    settings = load_settings()
    run_id = new_run_id(config.run_name)
    out_dir = settings.outputs_dir / "attribution" / run_id
    seed_everything(config.seed)
    models_cfg = load_run_config(REPO_ROOT / config.models_config)
    prep = prepare_inputs(models_cfg, contract, settings)
    data = prep.data
    encoder = load_image_encoder(models_cfg.image_weights)

    if config.checkpoint_run:
        path = (
            settings.checkpoints_dir / "models" / config.checkpoint_run / f"{config.arm}_seed{config.seed}.pt"
        )
        trained, extra = load_trained(path)
        if trained.model.spec.modalities != ARMS[config.arm].spec.modalities:
            raise ValueError(f"checkpoint {path.name} is not the {config.arm} arm")
        source = {"checkpoint": config.checkpoint_run, **extra}
    else:
        arm = sized(ARMS[config.arm], models_cfg.model.d, models_cfg.model.heads, models_cfg.model.layers,
                    models_cfg.model.dropout)  # fmt: skip
        trained = train_model(arm.spec, data, TrainConfig(**models_cfg.train), config.seed)
        source = {"checkpoint": None, "trained_here": True}
    layout = build_layout(data, encoder.pathology_names)
    space = UnitSpace(layout)
    model = GatedLogit(trained.model, layout)
    image = ImageContext(encoder, prep.image_cache_dir, prep.image_keys, operating_thresholds(encoder))
    explainer = Explainer(
        trained=trained,
        model=model,
        space=space,
        data=data,
        raw=prep.raw,
        variables=load_variables(),
        contract=contract,
        features=prep.tables.features[contract.feature_names("fate_static")],
        methods={"ts": config.ts_method, "static": config.static_method, "concept": config.concept_method},
        mcfg=M.MethodConfig(
            ig_steps=config.ig_steps,
            mask_epochs=config.mask_epochs,
            mask_lr=config.mask_lr,
            mask_lambda_1=config.mask_lambda_1,
            dynamask_epochs=config.dynamask_epochs,
            seed=config.seed,
        ),
        sel=SelectionConfig(
            config.per_modality, config.max_drivers, config.tau_quantile, config.n_random_units
        ),
        image=image,
        window_hours=contract.window.hours,
    )
    val_rows = data.rows("val")[: config.n_tau_stays]
    tau = explainer.fit_tau(val_rows)
    rows = data.rows(config.split)[: config.n_stays]
    packets = explainer.packets(rows)

    allowed: dict[str, set[str]] = {}
    for key in driver_keys(load_variables(), contract, list(encoder.pathology_names)):
        allowed.setdefault(key.modality, set()).add(key.key)
    folder = out_dir / "packets"
    folder.mkdir(parents=True, exist_ok=True)
    for row, packet in packets:
        check_keys(packet, allowed)
        (folder / f"packet_{int(data.stay_ids[row])}.json").write_text(
            packet.model_dump_json(indent=2), encoding="utf-8"
        )

    report = summarise(packets, tau, config, source)
    report["run_id"] = run_id
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    write_run_manifest(
        out_dir,
        run_id=run_id,
        config=config.model_dump(),
        seeds=[config.seed],
        contract_sha256=contract.sha256,
        data={"models_config": config.models_config, **source},
        extra={"machine": settings.machine, "out_dir": str(out_dir)},
    )
    report["out_dir"] = str(out_dir)
    return report


def summarise(packets, tau: float, config: PacketsConfig, source: dict) -> dict:
    """Aggregates only: no ids, no free text."""
    n = len(packets)
    per_stay = Counter(len(p.drivers) for _, p in packets)
    modality = Counter(d.modality for _, p in packets for d in p.drivers)
    direction = Counter(d.direction for _, p in packets for d in p.drivers)
    image = [d for _, p in packets for d in p.drivers if d.modality == "image"]
    return {
        "split": config.split,
        "packets": suppress(n),
        "schema_valid": suppress(n),  # a packet that fails validation raises before it is written
        "tau": round(float(tau), 4),
        "methods": {
            "timeseries": config.ts_method,
            "static": config.static_method,
            "image": config.concept_method,
        },
        "drivers_per_stay_mean": round(float(np.mean([len(p.drivers) for _, p in packets])), 2)
        if n
        else None,
        "stays_by_driver_count": suppress_counts({str(k): v for k, v in sorted(per_stay.items())}),
        "drivers_by_modality": suppress_counts(dict(modality)),
        "drivers_by_direction": suppress_counts(dict(direction)),
        "image_drivers": suppress(len(image)),
        "image_findings_reported": suppress(sum(1 for d in image if d.reported_finding)),
        "image_drivers_with_region": suppress(sum(1 for d in image if d.region)),
        "model_source": {k: v for k, v in source.items() if k != "run_id"},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    path = Path(args.config)
    print(format_aggregates(run(load_config(path if path.is_absolute() else REPO_ROOT / path))))


if __name__ == "__main__":
    main()
