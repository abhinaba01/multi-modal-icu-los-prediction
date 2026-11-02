"""Build the Stage 2 hourly tensors for the contract cohort.

    python -m fate_icu.data.timeseries.build --config configs/timeseries/smoke.yaml   # laptop, demo DB
    python -m fate_icu.data.timeseries.build --config configs/timeseries/full.yaml    # lab

Writes ``<artifacts>/timeseries/<run_id>/``: ``tensors.npz`` (``values``, ``mask``,
``delta``, ``raw``, ``stay_ids``, ``split``, ``channels``), ``normaliser.json``,
``report.json`` and ``run_manifest.json``. Prints aggregates only (CLAUDE.md A3). Every
build checks that no input is dated after t0 and stops otherwise.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from fate_icu.data.contract import Contract, load_contract
from fate_icu.data.timeseries.extract import Extraction, assert_no_future, extract
from fate_icu.data.timeseries.report import legacy_comparison, missingness
from fate_icu.data.timeseries.sources import SqlEventSource
from fate_icu.data.timeseries.tensors import Normaliser, hourly_grid, tensor_sha256
from fate_icu.data.timeseries.variables import VariableSet, load_variables
from fate_icu.tabular._legacy import legacy_tree_sha256
from fate_icu.tabular.api import TabularTables, load_splits, load_tables
from fate_icu.utils.aggregates import format_aggregates, suppress, suppress_counts
from fate_icu.utils.env import REPO_ROOT, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest
from fate_icu.utils.seed import seed_everything

TENSOR_FILE = "tensors.npz"
NORMALISER_FILE = "normaliser.json"


class TimeSeriesConfig(BaseModel):
    """``configs/timeseries/{smoke,full}.yaml``."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    source: Literal["postgres"]
    variables: str = "configs/variables.yaml"
    clip_quantiles: tuple[float, float] = (0.005, 0.995)
    delta_cap_hours: int = 24
    require_signal: bool = True
    batch_size: int = 5000
    cache_static: bool = True
    strict_sofa: bool = True
    compare_legacy: bool = True
    seed: int = 666


def load_config(path: str | Path) -> TimeSeriesConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return TimeSeriesConfig.model_validate(yaml.safe_load(handle))


@dataclass
class TimeSeriesBuild:
    """One build: tensors for the cohort in ``stay_ids`` order."""

    stay_ids: np.ndarray
    split: np.ndarray
    channels: list[str]
    raw: np.ndarray
    arrays: dict[str, np.ndarray]
    normaliser: Normaliser
    extraction: Extraction
    sha256: str


def build(
    source: Any,
    cohort: pd.DataFrame,
    split: pd.Series,
    variables: VariableSet,
    hours: int,
    *,
    quantiles: tuple[float, float] = (0.005, 0.995),
    delta_cap_hours: int = 24,
    require_signal: bool = True,
    batch_size: int | None = None,
) -> TimeSeriesBuild:
    """Extract, grid, fit the normaliser on train stays, and make the tensors.

    Args:
        source: Event source (SQL or in-memory).
        cohort: Contract cohort, index ``stay_id`` with ``subject_id`` and ``intime``.
        split: FaTE split label per stay.
        variables: Channel definitions.
        hours: The window W.
    """
    cohort = cohort.sort_index()
    split = split.reindex(cohort.index)
    if split.isna().any():
        raise ValueError("every cohort stay needs a split label")
    extraction = extract(source, cohort, variables, hours, batch_size=batch_size)
    assert_no_future(extraction, cohort, hours)
    raw = hourly_grid(extraction, cohort, variables, hours)
    train = (split == "train").to_numpy()
    normaliser = Normaliser.fit(raw, train, variables, quantiles, delta_cap_hours, require_signal)
    arrays = normaliser.transform(raw, variables)
    return TimeSeriesBuild(
        stay_ids=cohort.index.to_numpy(dtype=np.int64),
        split=split.to_numpy(dtype=str),
        channels=variables.names,
        raw=raw.astype(np.float32),
        arrays=arrays,
        normaliser=normaliser,
        extraction=extraction,
        sha256=tensor_sha256(arrays),
    )


def save(result: TimeSeriesBuild, out_dir: str | Path) -> Path:
    """Write ``tensors.npz`` and ``normaliser.json``; returns the tensor path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / TENSOR_FILE
    np.savez_compressed(
        path,
        **result.arrays,
        raw=result.raw,
        stay_ids=result.stay_ids,
        split=np.array(result.split, dtype="U5"),
        channels=np.array(result.channels, dtype="U64"),
    )
    (out_dir / NORMALISER_FILE).write_text(
        json.dumps(result.normaliser.to_dict(), indent=2), encoding="utf-8"
    )
    return path


def load_tensors(run_dir: str | Path) -> dict[str, Any]:
    """Read a build back: the arrays, ``channels`` (list) and the fitted normaliser."""
    run_dir = Path(run_dir)
    with np.load(run_dir / TENSOR_FILE, allow_pickle=False) as data:
        out: dict[str, Any] = {name: data[name] for name in data.files}
    out["channels"] = [str(c) for c in out["channels"]]
    out["normaliser"] = Normaliser.from_dict(
        json.loads((run_dir / NORMALISER_FILE).read_text(encoding="utf-8"))
    )
    return out


def make_source(config: TimeSeriesConfig, contract: Contract, db_url: str | None) -> SqlEventSource:
    if not db_url:
        raise RuntimeError("FATE_DB_URL is not set (see .env.example)")
    from sqlalchemy import create_engine

    return SqlEventSource(create_engine(db_url, pool_pre_ping=True), contract, contract.window.hours)


def run(
    config: TimeSeriesConfig,
    contract: Contract | None = None,
    *,
    tables: TabularTables | None = None,
    source: Any | None = None,
) -> dict[str, Any]:
    """Build, save and report. ``tables`` and ``source`` can be injected (tests)."""
    contract = contract or load_contract()
    settings = load_settings()
    variables = load_variables(REPO_ROOT / config.variables)
    hours = contract.window.hours
    run_id = new_run_id(config.run_name)
    out_dir = settings.artifacts_dir / "timeseries" / run_id
    seed_everything(config.seed)

    tables = tables or load_tables(
        config.source,
        contract=contract,
        settings=settings,
        cache=config.cache_static,
        strict_sofa=config.strict_sofa,
    )
    cohort = tables.cohort.sort_index()
    split = load_splits(cohort, contract)
    source = source or make_source(config, contract, settings.db_url)
    result = build(
        source,
        cohort,
        split,
        variables,
        hours,
        quantiles=config.clip_quantiles,
        delta_cap_hours=config.delta_cap_hours,
        require_signal=config.require_signal,
        batch_size=config.batch_size,
    )

    n, w, c = result.arrays["values"].shape
    report: dict[str, Any] = {
        "run_id": run_id,
        "stays": suppress(n),
        "shape": {"stays": suppress(n), "hours": w, "channels": c},
        "splits": {name: suppress(int((result.split == name).sum())) for name in ("train", "val", "test")},
        "rows_by_source": {
            table: suppress_counts(counts) for table, counts in result.extraction.counts.items()
        },
        "implausible_values_dropped": suppress_counts(result.extraction.dropped_by_channel),
        "missingness": missingness(result.raw, result.split, variables),
        "normaliser_warnings": result.normaliser.warnings,
        "tensor_sha256": result.sha256,
    }
    if config.compare_legacy:
        report["internship_comparison"] = legacy_comparison(result.extraction, tables.features)

    save(result, out_dir)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
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
            "variables_sha256": variables.sha256,
        },
        extra={"machine": settings.machine, "tensor_sha256": result.sha256, "out_dir": str(out_dir)},
    )
    report["out_dir"] = str(out_dir)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/timeseries/smoke.yaml or full.yaml")
    args = parser.parse_args(argv)
    print(format_aggregates(run(load_config(args.config))))


if __name__ == "__main__":
    main()
