"""Bridge to the frozen internship code in ``legacy_los/`` (never edited).

The legacy package is literally named ``src`` and imports itself as ``src.<module>``,
so ``legacy_los/`` must be on ``sys.path``. This module adds it once and checks that
``src`` really resolves to the legacy package.
"""

from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from fate_icu.data.contract import Contract
from fate_icu.utils.env import REPO_ROOT
from fate_icu.utils.hashing import tree_sha256

LEGACY_ROOT = REPO_ROOT / "legacy_los"
# Top-level legacy folders that hold run outputs, not code (excluded from the hash).
LEGACY_OUTPUT_DIRS = frozenset({"data", "results"})


def ensure_legacy_importable() -> None:
    """Put ``legacy_los/`` on ``sys.path`` and verify that ``import src`` resolves there."""
    root = str(LEGACY_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    src = importlib.import_module("src")
    location = Path(getattr(src, "__file__", None) or "").resolve()
    if LEGACY_ROOT.resolve() not in location.parents:
        raise ImportError(
            f"'src' resolved to {location or 'a namespace package'}, not legacy_los/src. "
            "Another 'src' was imported first; import fate_icu.tabular before it."
        )


def legacy(module: str) -> ModuleType:
    """Import ``src.<module>`` from the legacy package, e.g. ``legacy("features.cohort")``."""
    ensure_legacy_importable()
    return importlib.import_module(f"src.{module}")


def legacy_tree_sha256() -> str:
    """Current LF-normalised hash of ``legacy_los/`` (compare with the contract)."""
    return tree_sha256(LEGACY_ROOT, LEGACY_OUTPUT_DIRS)


def legacy_config(
    contract: Contract,
    *,
    raw_dir: str | Path | None = None,
    gem_path: str | Path | None = None,
    xgb_n_estimators: int | None = None,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> dict[str, Any]:
    """The legacy ``config.yaml`` with run-specific overrides, checked against the contract.

    Args:
        contract: The frozen contract; the legacy cohort and window values must match it.
        raw_dir: MIMIC CSV folder for the legacy CSV loader (absolute).
        gem_path: Path of the CMS ICD-9 -> ICD-10 GEM file (absolute).
        xgb_n_estimators: Boosting rounds (smoke configs shrink this).
        random_state: Model seed.
        n_jobs: Threads for the tree models.

    Returns:
        A new config dictionary for the legacy functions.

    Raises:
        ValueError: The legacy config disagrees with the contract.
    """
    cfg = copy.deepcopy(legacy("config").load_config(LEGACY_ROOT / "config.yaml"))
    check_legacy_config_matches_contract(cfg, contract)
    if raw_dir is not None:
        cfg["data"]["raw_dir"] = str(Path(raw_dir).resolve())
    if gem_path is not None:
        cfg["data"]["icd9to10_gem"] = str(Path(gem_path).resolve())
    if xgb_n_estimators is not None:
        cfg["models"]["xgb_n_estimators"] = int(xgb_n_estimators)
    if random_state is not None:
        cfg["project"]["random_state"] = int(random_state)
    if n_jobs is not None:
        cfg.setdefault("compute", {})["n_jobs"] = int(n_jobs)
    return cfg


def check_legacy_config_matches_contract(cfg: dict[str, Any], contract: Contract) -> None:
    """Raise if any frozen definition in the legacy config differs from the contract."""
    cohort = contract.cohort
    pairs = {
        "observation_window_hours": (cfg["observation_window_hours"], contract.window.hours),
        "min_los_days_exclusive": (cfg["cohort"]["min_los_days_exclusive"], cohort.los_days_min_exclusive),
        "max_los_days": (cfg["cohort"]["max_los_days"], cohort.los_days_max_inclusive),
        "min_age": (cfg["cohort"]["min_age"], cohort.min_anchor_age),
        "readmission_hours": (cfg["cohort"]["readmission_hours"], cohort.readmission_exclusion_hours),
        "random_state": (cfg["project"]["random_state"], contract.b0.random_state),
        "xgb_n_estimators": (cfg["models"]["xgb_n_estimators"], contract.b0.xgb_n_estimators),
        "rf_n_estimators": (cfg["models"]["rf_n_estimators"], contract.b0.rf_n_estimators),
        "n_repeats": (cfg["evaluation"]["n_repeats"], contract.splits.legacy.n_repeats),
        "test_size": (cfg["evaluation"]["test_size"], contract.splits.legacy.test_size),
    }
    ranges = {
        name: (spec["low"], spec["high"], spec["inclusive"])
        for name, spec in cfg["vital_mean_ranges"].items()
    }
    contract_ranges = {name: (r.low, r.high, r.inclusive) for name, r in cohort.vital_mean_ranges.items()}
    pairs["vital_mean_ranges"] = (ranges, contract_ranges)
    mismatched = {k: v for k, v in pairs.items() if v[0] != v[1]}
    if mismatched:
        raise ValueError(f"legacy config.yaml disagrees with the contract: {mismatched}")
