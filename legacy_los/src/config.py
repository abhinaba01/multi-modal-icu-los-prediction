"""Configuration loading, path resolution, and data-availability guards.

Central place for reading ``config.yaml`` and turning its relative paths into
absolute ones anchored at the project root, plus the "is the data actually
here?" guards the run scripts call before doing any work (per the requirement to
stop with a clear message when required data is missing).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class MissingDataError(RuntimeError):
    """Raised when a required raw MIMIC-IV table is not present on disk.

    Run scripts catch nothing and let this propagate: a missing input must halt
    execution loudly rather than silently produce a degenerate result.
    """


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the project YAML configuration.

    Args:
        config_path: Optional path to a config file. Defaults to
            ``<project_root>/config.yaml``.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def raw_dir(config: dict[str, Any]) -> Path:
    """Resolve the raw MIMIC-IV data directory.

    Resolution order: ``MIMIC_RAW_DIR`` environment variable, then
    ``data.raw_dir`` from config. Relative paths resolve against the project root.

    Args:
        config: Parsed configuration.

    Returns:
        Absolute path to the raw data directory.
    """
    configured = os.getenv("MIMIC_RAW_DIR", config["data"]["raw_dir"])
    path = Path(configured)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def raw_table_path(config: dict[str, Any], table_name: str) -> Path:
    """Resolve the path to a raw table, accepting either ``.csv`` or ``.csv.gz``.

    Args:
        config: Parsed configuration.
        table_name: Key under ``data.files`` (e.g. ``icustays``).

    Returns:
        Existing path to the table's CSV or CSV.GZ file (``.csv.gz`` is preferred
        when both would resolve, since that is how MIMIC-IV ships).

    Raises:
        KeyError: If ``table_name`` is not configured.
        MissingDataError: If neither the plain nor gzipped file exists.
    """
    relative = config["data"]["files"][table_name]
    base = raw_dir(config) / relative
    gz = base.with_suffix(base.suffix + ".gz")
    if gz.exists():
        return gz
    if base.exists():
        return base
    raise MissingDataError(
        f"Required raw table '{table_name}' not found. Looked for:\n"
        f"  {gz}\n  {base}\n"
        f"Set data.raw_dir in config.yaml or the MIMIC_RAW_DIR environment "
        f"variable to your MIMIC-IV directory (expects the standard icu/ and "
        f"hosp/ sub-folders)."
    )


def require_tables(config: dict[str, Any], table_names: list[str]) -> dict[str, Path]:
    """Guard: ensure every named raw table exists, else stop with one message.

    Args:
        config: Parsed configuration.
        table_names: Raw table keys the caller needs.

    Returns:
        Mapping of table name to its resolved path.

    Raises:
        MissingDataError: Listing every table that is missing.
    """
    resolved: dict[str, Path] = {}
    missing: list[str] = []
    for name in table_names:
        try:
            resolved[name] = raw_table_path(config, name)
        except MissingDataError as exc:
            missing.append(str(exc))
    if missing:
        raise MissingDataError(
            "Cannot run: required MIMIC-IV table(s) are missing.\n\n"
            + "\n\n".join(missing)
        )
    return resolved


def resolved_path(config: dict[str, Any], relative: str | Path) -> Path:
    """Resolve a project-relative path to an absolute one.

    Args:
        config: Parsed configuration (unused; kept for call-site symmetry).
        relative: A path relative to the project root, or absolute.

    Returns:
        Absolute path.
    """
    path = Path(relative)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def processed_path(config: dict[str, Any], key: str) -> Path:
    """Resolve the path to a processed feature parquet by config key.

    Args:
        config: Parsed configuration.
        key: One of ``hempel``, ``hempel_sofa``, ``hempel_sofa_dynamics``.

    Returns:
        Absolute path under ``features.processed_dir``.
    """
    processed_dir = resolved_path(config, config["features"]["processed_dir"])
    return processed_dir / config["features"][key]


def apply_fast_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the config with the ``fast`` smoke-test overrides applied.

    The fast profile shrinks the two expensive model sizes and the repeat count.
    It is for wiring smoke tests ONLY and must never back reported numbers.

    Args:
        config: Parsed configuration.

    Returns:
        A shallow-copied config with fast values folded into ``models``/
        ``evaluation``.
    """
    import copy

    fast = config.get("fast", {})
    updated = copy.deepcopy(config)
    updated["models"]["rf_n_estimators"] = fast.get("rf_n_estimators", 100)
    updated["models"]["xgb_n_estimators"] = fast.get("xgb_n_estimators", 100)
    updated["evaluation"]["n_repeats"] = fast.get("n_repeats", 2)
    updated["_fast"] = True
    return updated
