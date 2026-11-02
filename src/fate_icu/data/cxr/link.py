"""Link the contract cohort to MIMIC-CXR-JPG and write the download list (Stage 1).

    python -m fate_icu.data.cxr.link --config configs/cxr/smoke.yaml   # laptop: demo DB + synthetic metadata
    python -m fate_icu.data.cxr.link --config configs/cxr/full.yaml    # lab: real metadata

Writes ``<artifacts>/cxr/<run_id>/``: ``paired_cohort.parquet`` (every stay, with
``has_cxr``), ``image_list.txt`` (relative paths of the selected images only, for wget),
``report.json`` and ``run_manifest.json``. With synthetic metadata it also checks every
planted answer. Prints aggregates only (CLAUDE.md A3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from fate_icu.data.contract import Contract, load_contract
from fate_icu.data.cxr.linkage import link, write_image_list
from fate_icu.data.cxr.metadata import METADATA_FILE, read_metadata
from fate_icu.data.cxr.report import check_planted_answers, flow_report
from fate_icu.data.cxr.synthetic import generate_cxr_metadata
from fate_icu.data.targets import resolve_long_stay_threshold
from fate_icu.tabular._legacy import legacy_tree_sha256
from fate_icu.tabular.api import TabularTables, load_splits, load_tables
from fate_icu.tabular.loaders import PostgresLoader
from fate_icu.utils.aggregates import format_aggregates
from fate_icu.utils.env import REPO_ROOT, Settings, load_settings
from fate_icu.utils.manifest import new_run_id, write_run_manifest
from fate_icu.utils.seed import seed_everything

PAIRED_FILE = "paired_cohort.parquet"
IMAGE_LIST_FILE = "image_list.txt"


class CxrConfig(BaseModel):
    """``configs/cxr/{smoke,full}.yaml`` (used by ``link`` and ``cache``)."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    metadata: Literal["synthetic", "file"]
    images: Literal["fixtures", "mimic"]
    lookback_hours: float = 24.0
    synthetic_seed: int = 0
    fixtures_dir: str = "tests/fixtures/cxr"
    min_paired_train_stays: int = 2000
    cache_static: bool = True
    strict_sofa: bool = True
    seed: int = 666


def load_config(path: str | Path) -> CxrConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return CxrConfig.model_validate(yaml.safe_load(handle))


def anchor_year_groups(loader: Any) -> pd.Series:
    """``anchor_year_group`` per ``subject_id`` from ``mimiciv_hosp.patients``."""
    patients = loader.load_table("patients", usecols=["subject_id", "anchor_year_group"])
    return patients.drop_duplicates("subject_id").set_index("subject_id")["anchor_year_group"]


def run(
    config: CxrConfig,
    contract: Contract | None = None,
    *,
    tables: TabularTables | None = None,
    loader: Any | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Link, save and report. ``tables`` and ``loader`` can be injected (tests)."""
    contract = contract or load_contract()
    settings = settings or load_settings()
    hours = contract.window.hours
    run_id = new_run_id(config.run_name)
    out_dir = settings.artifacts_dir / "cxr" / run_id
    seed_everything(config.seed)

    tables = tables or load_tables(
        "postgres",
        contract=contract,
        settings=settings,
        cache=config.cache_static,
        strict_sofa=config.strict_sofa,
    )
    loader = loader or PostgresLoader(settings.db_url, contract)
    cohort = tables.cohort.sort_index()
    split = load_splits(cohort, contract)

    synthetic = None
    if config.metadata == "synthetic":
        synthetic = generate_cxr_metadata(cohort, hours, config.lookback_hours, config.synthetic_seed)
        metadata = synthetic.metadata
    else:
        if settings.mimic_cxr_dir is None:
            raise RuntimeError("MIMIC_CXR_DIR is not set (see .env.example and docs/lab_runbook.md)")
        metadata = read_metadata(settings.mimic_cxr_dir / METADATA_FILE)

    linkage = link(metadata, cohort, hours, config.lookback_hours)
    linked = linkage.paired[linkage.paired["has_cxr"]]
    t0 = cohort["t0"].reindex(linked["stay_id"]).to_numpy()
    if (linked["study_datetime"].to_numpy() > t0).any():  # fail loudly: never an image after t0
        raise RuntimeError("an image dated after t0 was selected")

    threshold, threshold_source = resolve_long_stay_threshold(
        cohort["los"], split, contract, require_frozen=False
    )
    report = flow_report(
        linkage,
        metadata,
        cohort,
        split,
        anchor_year_groups(loader),
        threshold,
        config.lookback_hours,
        config.min_paired_train_stays,
        hours,
    )
    report = {
        "run_id": run_id,
        "metadata": config.metadata,
        **report,
        "long_stay_threshold_source": threshold_source,
    }
    if synthetic is not None:
        report["planted_answers"] = check_planted_answers(linkage.paired, synthetic.expected)

    out_dir.mkdir(parents=True, exist_ok=True)
    linkage.paired.to_parquet(out_dir / PAIRED_FILE, index=False)
    report["images_to_download"] = write_image_list(linkage.paired, out_dir / IMAGE_LIST_FILE)
    if synthetic is not None:
        synthetic.write_csv(out_dir / "synthetic-metadata.csv.gz")
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_run_manifest(
        out_dir,
        run_id=run_id,
        config=config.model_dump(),
        seeds=[config.seed, config.synthetic_seed],
        contract_sha256=contract.sha256,
        legacy_tree_sha256=legacy_tree_sha256(),
        data={
            "metadata": config.metadata,
            "db": settings.db_url_redacted,
            "mimic_version": contract.mimic.demo_version
            if settings.machine == "laptop"
            else contract.mimic.version,
            "mimic_cxr_jpg_version": "2.1.0",
        },
        extra={"machine": settings.machine, "out_dir": str(out_dir)},
    )
    report["out_dir"] = str(out_dir)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/cxr/smoke.yaml or full.yaml")
    args = parser.parse_args(argv)
    report = run(load_config(REPO_ROOT / args.config if not Path(args.config).is_absolute() else args.config))
    print(format_aggregates(report))
    planted = report.get("planted_answers")
    if planted is not None and not planted["all_correct"]:
        raise SystemExit("planted answers wrong: " + json.dumps(planted["wrong_by_scenario"]))


if __name__ == "__main__":
    main()
