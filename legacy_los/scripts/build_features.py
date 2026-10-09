"""Build the per-stay feature tables from the raw MIMIC-IV CSVs.

Usage:
    python scripts/build_features.py --which all
    python scripts/build_features.py --which hempel --limit 2000     # quick smoke
    python scripts/build_features.py --which sofa

Guards: stops with a clear message if a required raw table is missing. The SOFA
variants additionally require ``inputevents`` and ``outputevents`` and, unless
``--no-strict-sofa`` is given, fail if the resulting SOFA components carry no
signal (symptom of a missing/empty source table).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, processed_path, require_tables  # noqa: E402
from src.data.loader import MIMICLoader  # noqa: E402
from src.features.build_dataset import (assemble_all, assemble_hempel, assemble_sofa,  # noqa: E402
                                        assemble_sofa_dynamics)
from src.reporting import ensure_dir  # noqa: E402

BASE_TABLES = ["icustays", "admissions", "patients", "diagnoses_icd", "chartevents", "labevents"]
SOFA_TABLES = BASE_TABLES + ["inputevents", "outputevents"]

LOGGER = logging.getLogger("build_features")


def configure_logging() -> None:
    """Configure console + file logging."""
    log_dir = ensure_dir(PROJECT_ROOT / "results" / "logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "build_features.log"), logging.StreamHandler()],
    )


def main() -> None:
    """Build the requested feature table(s)."""
    parser = argparse.ArgumentParser(description="Build Hempel / SOFA feature tables from raw MIMIC-IV CSVs.")
    parser.add_argument("--which", choices=["hempel", "sofa", "dynamics", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Cap cohort size (smoke test only).")
    parser.add_argument("--no-strict-sofa", action="store_true", help="Warn instead of fail on degenerate SOFA.")
    args = parser.parse_args()

    configure_logging()
    config = load_config()
    loader = MIMICLoader(config)
    strict = not args.no_strict_sofa

    needs_sofa = args.which in {"sofa", "dynamics", "all"}
    require_tables(config, SOFA_TABLES if needs_sofa else BASE_TABLES)
    ensure_dir(processed_path(config, "hempel").parent)

    if args.which == "all":
        # One set of raw-table scans builds all three tables (see assemble_all).
        LOGGER.info("Building all three tables from a single set of scans ...")
        tables = assemble_all(loader, config, limit=args.limit, strict=strict)
        _save(tables["hempel"], config, "hempel")
        _save(tables["hempel_sofa"], config, "hempel_sofa")
        _save(tables["hempel_sofa_dynamics"], config, "hempel_sofa_dynamics")
    elif args.which == "hempel":
        LOGGER.info("Building Hempel baseline table ...")
        _save(assemble_hempel(loader, config, limit=args.limit), config, "hempel")
    elif args.which == "sofa":
        LOGGER.info("Building Hempel + static SOFA table ...")
        _save(assemble_sofa(loader, config, limit=args.limit, strict=strict), config, "hempel_sofa")
    elif args.which == "dynamics":
        LOGGER.info("Building Hempel + SOFA-dynamics table ...")
        _save(assemble_sofa_dynamics(loader, config, limit=args.limit, strict=strict), config, "hempel_sofa_dynamics")

    LOGGER.info("Done.")


def _save(df, config, key: str) -> None:
    """Write a feature table to its configured parquet path.

    Args:
        df: Feature table indexed by stay_id.
        config: Parsed configuration.
        key: Feature key.
    """
    path = processed_path(config, key)
    df.to_parquet(path)
    LOGGER.info("wrote %s (%d stays x %d cols)", path, df.shape[0], df.shape[1])


if __name__ == "__main__":
    main()
