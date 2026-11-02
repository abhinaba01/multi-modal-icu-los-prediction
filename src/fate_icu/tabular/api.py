"""Thin adapter over the internship pipeline: contract cohort, static features, splits.

The legacy functions in ``legacy_los/`` do the work unchanged. This module adds:

* a data source of choice (PostgreSQL, the legacy CSV reader, or in-memory tables);
* the cohort-defining row filters the legacy code applies inside its preprocessing
  (duplicates, complete case on ``legacy_hempel``, vital-mean ranges), kept separate
  from its whole-dataset encoding and scaling, which FaTE-ICU fits on train only;
* provenance: the latest event time each stay's features used, per source, so tests can
  assert nothing after t0 was read (CLAUDE.md A4);
* a cache of the built tables under ``artifacts/`` (git-ignored).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fate_icu.data.contract import Contract, load_contract
from fate_icu.data.splits import fate_splits
from fate_icu.tabular._legacy import legacy, legacy_config, legacy_tree_sha256
from fate_icu.tabular.loaders import (
    PostgresLoader,
    csv_loader,
    describe_source,
    legacy_itemid_filters,
)
from fate_icu.utils.env import Settings, load_settings
from fate_icu.utils.hashing import file_sha256

TARGET = "los"
PROVENANCE_SOURCES = ("vitals", "labs", "sofa_chart", "sofa_lab", "sofa_input", "sofa_output")
CACHE_FORMAT = 1


@dataclass
class TabularTables:
    """Everything the tabular stages need, for the final contract cohort.

    Attributes:
        cohort: One row per stay (index ``stay_id``): ids, times, ``t0``, ``los`` and the
            legacy cohort columns.
        features: Every contract feature, raw (unscaled, string categoricals), in
            contract order (index ``stay_id``).
        provenance: Latest event time each stay's features used, per source.
        flow: Cohort counts after each rule.
    """

    cohort: pd.DataFrame
    features: pd.DataFrame
    provenance: pd.DataFrame
    flow: dict[str, int]

    @property
    def los(self) -> pd.Series:
        return self.cohort[TARGET].astype(float)


# --- building ----------------------------------------------------------------------


def _scan(loader: Any, cfg: dict[str, Any], cohort: pd.DataFrame):
    """One pass per event table, as legacy ``_scan_sources(want_sofa=True)``, plus provenance.

    The extra raw sinks select exactly the rows the mean sinks average (same itemids,
    same ``(intime, t0]`` window); the legacy multi-sink scanner guarantees sinks do
    not affect each other (legacy ``tests/test_aggregate.py``).
    """
    agg = legacy("features.aggregate")
    vitals = legacy("features.vitals")
    labs = legacy("features.labs")
    sofa = legacy("features.sofa_static")

    chart_sinks = [
        agg.MeanSink("vitals", vitals.VITAL_ITEMIDS),
        agg.RawSink("sofa_chart", set(sofa.CHART_ITEMIDS)),
        agg.RawSink("prov_vitals", set(vitals.VITAL_ITEMIDS), lower_inclusive=False, upper_inclusive=True),
    ]
    lab_sinks = [
        agg.MeanSink("labs", labs.LAB_ITEMIDS),
        agg.RawSink("sofa_lab", set(sofa.LAB_ITEMIDS)),
        agg.RawSink("prov_labs", set(labs.LAB_ITEMIDS), lower_inclusive=False, upper_inclusive=True),
    ]
    chart = agg.stream_multi(
        loader,
        cfg,
        "chartevents",
        "stay_id",
        "valuenum",
        "charttime",
        cohort[["stay_id", "intime"]],
        chart_sinks,
    )
    lab = agg.stream_multi(
        loader,
        cfg,
        "labevents",
        "hadm_id",
        "valuenum",
        "charttime",
        cohort[["hadm_id", "stay_id", "intime"]],
        lab_sinks,
    )
    inputs = agg.stream_multi(
        loader,
        cfg,
        "inputevents",
        "stay_id",
        "rate",
        "starttime",
        cohort[["stay_id", "intime"]],
        [agg.RawSink("input", set(sofa.VASO_ITEMIDS))],
    )
    outputs = agg.stream_multi(
        loader,
        cfg,
        "outputevents",
        "stay_id",
        "value",
        "charttime",
        cohort[["stay_id", "intime"]],
        [agg.RawSink("output", set(sofa.URINE_ITEMIDS))],
    )
    events = {
        "chart": chart["sofa_chart"],
        "lab": lab["sofa_lab"],
        "input": inputs["input"],
        "output": outputs["output"],
    }

    used = {
        "vitals": chart["prov_vitals"],
        "labs": lab["prov_labs"],
        "sofa_chart": events["chart"],
        "sofa_lab": events["lab"],
        "sofa_input": events["input"],
        "sofa_output": events["output"],
    }
    provenance = pd.DataFrame(index=pd.Index(cohort["stay_id"].astype("int64"), name="stay_id"))
    for source, rows in used.items():
        if rows.empty:
            provenance[source] = pd.NaT
            continue
        latest = (
            rows.assign(charttime=pd.to_datetime(rows["charttime"])).groupby("stay_id")["charttime"].max()
        )
        latest.index = latest.index.astype("int64")
        provenance[source] = latest.reindex(provenance.index)
    return chart["vitals"], lab["labs"], events, provenance


def cohort_flow(loader: Any, cfg: dict[str, Any]) -> dict[str, int]:
    """Counts after each legacy ``build_cohort`` rule (same order, same helpers)."""
    cohort_mod = legacy("features.cohort")
    rules = cfg["cohort"]
    icu = loader.load_table(
        "icustays",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )
    flow = {"icustays": len(icu)}
    icu = icu[icu["stay_id"].isin(cohort_mod._non_readmission_stays(icu, hours=rules["readmission_hours"]))]
    flow["after_readmission_48h"] = len(icu)
    adm = loader.load_table("admissions", usecols=["hadm_id", "deathtime"], parse_dates=["deathtime"])
    pat = loader.load_table("patients", usecols=["subject_id", "anchor_age"])
    df = icu.merge(adm, on="hadm_id", how="inner").merge(pat, on="subject_id", how="inner")
    df = df[df["deathtime"].isna() | (df["deathtime"] > df["outtime"])]
    flow["after_death_in_icu"] = len(df)
    los = pd.to_numeric(df["los"], errors="coerce")
    df = df[(los > rules["min_los_days_exclusive"]) & (los <= rules["max_los_days"])]
    flow["after_los_1_21"] = len(df)
    df = df[pd.to_numeric(df["anchor_age"], errors="coerce") >= rules["min_age"]]
    flow["after_age"] = int(df["stay_id"].nunique())
    return flow


def build_tables(
    loader: Any, cfg: dict[str, Any], contract: Contract, strict_sofa: bool = True
) -> TabularTables:
    """Run the legacy feature pipeline once and apply the contract cohort rules.

    Args:
        loader: Any object with the legacy loader interface.
        cfg: Legacy config from :func:`legacy_config`.
        contract: The frozen contract.
        strict_sofa: Fail on degenerate SOFA components (legacy quality guard).

    Returns:
        The tables for the final contract cohort.
    """
    bd = legacy("features.build_dataset")
    prep = legacy("preprocessing.prepare")
    sofa = legacy("features.sofa_static")

    cohort = bd._build_cohort(loader, cfg, None)
    flow = cohort_flow(loader, cfg)
    if flow["after_age"] != len(cohort):
        raise RuntimeError(f"cohort flow ({flow['after_age']}) disagrees with build_cohort ({len(cohort)})")

    vitals, labs, events, provenance = _scan(loader, cfg, cohort)
    base = bd._assemble_base_from_parts(loader, cfg, cohort, vitals, labs)
    with_static = bd._add_static_sofa(base, events, cohort, strict_sofa)
    with_dynamics = bd._add_sofa_dynamics(base, events, cohort, cfg, strict_sofa)
    wide = with_dynamics.join(with_static[sofa.SOFA_STATIC_COLUMNS], how="left")

    all_features = [f.name for f in contract.features]
    if set(wide.columns) - {TARGET} != set(all_features):
        raise RuntimeError(
            "legacy feature columns differ from the contract: "
            f"extra={sorted(set(wide.columns) - {TARGET} - set(all_features))}, "
            f"missing={sorted(set(all_features) - set(wide.columns))}"
        )

    # Cohort-defining rows filters = legacy prepare_dataset steps 1-3 on the base table.
    case_columns = [TARGET, *contract.feature_names(contract.cohort.complete_case_feature_set)]
    frame = base[case_columns]
    if contract.cohort.drop_duplicate_rows:
        frame = frame.drop_duplicates()
    flow["after_duplicates"] = len(frame)
    frame = frame.dropna(subset=case_columns)
    flow["after_complete_case"] = len(frame)
    frame = prep._clean_vital_means(frame, cfg)
    flow["final"] = len(frame)
    kept = frame.index

    cohort = cohort.set_index("stay_id")
    cohort.index = cohort.index.astype("int64")
    kept = pd.Index(kept.astype("int64"), name="stay_id")
    final_cohort = cohort.loc[kept].copy()
    final_cohort["t0"] = contract.t0(final_cohort["intime"])
    features = wide.loc[kept, all_features].copy()
    features.index = kept
    return TabularTables(
        cohort=final_cohort,
        features=features,
        provenance=provenance.reindex(kept),
        flow={k: int(v) for k, v in flow.items()},
    )


# --- sources and cache -------------------------------------------------------------


def make_loader(source: str, contract: Contract, settings: Settings, cfg: dict[str, Any]) -> Any:
    """Loader for a named source: ``postgres``, ``csv_demo`` or ``csv_raw``."""
    if source == "postgres":
        if not settings.db_url:
            raise RuntimeError("FATE_DB_URL is not set (see .env.example)")
        return PostgresLoader(settings.db_url, contract, itemid_filters=legacy_itemid_filters())
    if source in ("csv_demo", "csv_raw"):
        folder = settings.mimic_demo_dir if source == "csv_demo" else settings.mimic_raw_dir
        if folder is None:
            raise RuntimeError(f"{'MIMIC_DEMO_DIR' if source == 'csv_demo' else 'MIMIC_RAW_DIR'} is not set")
        cfg = {**cfg, "data": {**cfg["data"], "raw_dir": str(folder)}}
        return csv_loader(cfg)
    raise ValueError(f"unknown source {source!r}")


def _cache_key(contract: Contract, loader: Any, gem_path: Path, strict_sofa: bool) -> str:
    payload = {
        "format": CACHE_FORMAT,
        "contract": contract.sha256,
        "legacy": legacy_tree_sha256(),
        "source": describe_source(loader),
        "gem": file_sha256(gem_path),
        "strict_sofa": strict_sofa,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _read_cache(folder: Path) -> TabularTables | None:
    files = {name: folder / f"{name}.parquet" for name in ("cohort", "features", "provenance")}
    if not all(p.exists() for p in files.values()) or not (folder / "flow.json").exists():
        return None
    return TabularTables(
        cohort=pd.read_parquet(files["cohort"]),
        features=pd.read_parquet(files["features"]),
        provenance=pd.read_parquet(files["provenance"]),
        flow=json.loads((folder / "flow.json").read_text(encoding="utf-8")),
    )


def _write_cache(folder: Path, tables: TabularTables) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    tables.cohort.to_parquet(folder / "cohort.parquet")
    tables.features.to_parquet(folder / "features.parquet")
    tables.provenance.to_parquet(folder / "provenance.parquet")
    (folder / "flow.json").write_text(json.dumps(tables.flow, indent=2), encoding="utf-8")


def load_tables(
    source: str = "postgres",
    *,
    contract: Contract | None = None,
    settings: Settings | None = None,
    loader: Any | None = None,
    gem_path: str | Path | None = None,
    cache: bool = True,
    strict_sofa: bool = True,
) -> TabularTables:
    """Build (or read from cache) the contract cohort, features and provenance.

    Args:
        source: ``postgres``, ``csv_demo`` or ``csv_raw`` (ignored if ``loader`` is given).
        contract: Defaults to ``configs/contract.yaml``.
        settings: Defaults to ``.env`` / environment.
        loader: An explicit loader (e.g. ``FrameLoader`` over synthetic tables).
        gem_path: ICD-9 -> ICD-10 GEM file; defaults to ``ICD9_GEM_PATH``.
        cache: Read/write ``<artifacts>/tabular/<key>/``. Never used for in-memory loaders.
        strict_sofa: Fail on degenerate SOFA components.
    """
    contract = contract or load_contract()
    settings = settings or load_settings()
    gem = Path(gem_path) if gem_path else settings.icd9_gem_path
    if gem is None or not Path(gem).exists():
        raise RuntimeError(
            "The ICD-9 -> ICD-10 GEM file is required (the contract cohort is complete-case on "
            "the legacy diagnosis features). Set ICD9_GEM_PATH; see docs/laptop_setup.md."
        )
    cfg = legacy_config(contract, gem_path=gem)
    loader = loader or make_loader(source, contract, settings, cfg)
    if source in ("csv_demo", "csv_raw") and hasattr(loader, "config"):
        cfg = loader.config

    folder = None
    if cache and describe_source(loader) != "in-memory":
        folder = settings.artifacts_dir / "tabular" / _cache_key(contract, loader, Path(gem), strict_sofa)
        cached = _read_cache(folder)
        if cached is not None:
            return cached
    tables = build_tables(loader, cfg, contract, strict_sofa=strict_sofa)
    if folder is not None:
        _write_cache(folder, tables)
    return tables


# --- the Stage 0 adapter API ---------------------------------------------------------


def load_cohort(tables: TabularTables | None = None, **kwargs: Any) -> pd.DataFrame:
    """The final contract cohort (index ``stay_id``), including ``t0`` and ``los``."""
    return (tables or load_tables(**kwargs)).cohort


def load_static_features(
    feature_set: str = "fate_static", tables: TabularTables | None = None, **kwargs: Any
) -> pd.DataFrame:
    """Raw internship features of a contract feature set (index ``stay_id``)."""
    tables = tables or load_tables(**kwargs)
    contract = kwargs.get("contract") or load_contract()
    return tables.features[contract.feature_names(feature_set)]


def load_splits(
    cohort: pd.DataFrame | None = None, contract: Contract | None = None, **kwargs: Any
) -> pd.Series:
    """Patient-level FaTE split label per stay (index ``stay_id``)."""
    contract = contract or load_contract()
    cohort = cohort if cohort is not None else load_cohort(contract=contract, **kwargs)
    return fate_splits(cohort, contract)
