"""Run the Hempel replication augmented with STATIC SOFA features.

Identical pipeline to ``run_hempel.py`` (same cohort, preprocessing, four models,
two-stage flow) but on the baseline + static (worst-in-24h) SOFA feature table:
``sofa_total``, the six component sub-scores, and the raw ``urine_24h`` /
``bilirubin_max`` signals.

Prerequisite: python scripts/build_features.py --which sofa
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import apply_fast_overrides, load_config  # noqa: E402
from src.pipeline import format_summary, run_and_report  # noqa: E402
from src.reporting import ensure_dir  # noqa: E402

NAME = "hempel_sofa"
KEY = "hempel_sofa"


def main() -> None:
    """Entry point for the Hempel + static SOFA run."""
    parser = argparse.ArgumentParser(description="Hempel replication + static SOFA features.")
    parser.add_argument("--fast", action="store_true", help="Shrink model sizes/repeats (smoke test only).")
    args = parser.parse_args()

    config = load_config()
    if args.fast:
        config = apply_fast_overrides(config)

    log_dir = ensure_dir(PROJECT_ROOT / "results" / "logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / f"run_{NAME}.log"), logging.StreamHandler()],
    )

    results = run_and_report(config, KEY, NAME)
    print("\n" + format_summary(results))
    print(f"\nResults written to: {results['out_dir']}")


if __name__ == "__main__":
    main()
