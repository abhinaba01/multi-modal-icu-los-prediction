"""Run the two ablation studies from the run scripts' saved per-repeat metrics:

    A) Hempel  vs  Hempel + static SOFA
    B) Hempel  vs  Hempel + SOFA-dynamics

This does **no model fitting** -- it reuses the per-repeat metrics that
``run_hempel.py`` / ``run_hempel_sofa.py`` / ``run_hempel_sofa_dynamics.py``
already wrote (on identical splits), and computes per-model metric deltas
(augmented - baseline) plus a paired significance test. The Hempel baseline is
loaded once and reused across both studies.

Prerequisite: run all three pipelines first (same config):
    python scripts/run_hempel.py
    python scripts/run_hempel_sofa.py
    python scripts/run_hempel_sofa_dynamics.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ablation.compare import run_ablation_study  # noqa: E402
from src.config import load_config  # noqa: E402
from src.pipeline import load_per_repeat  # noqa: E402
from src.reporting import ensure_dir, save_table  # noqa: E402

# study name -> (baseline run name, augmented run name)
STUDIES = {
    "hempel_vs_sofa": ("hempel", "hempel_sofa"),
    "hempel_vs_sofa_dynamics": ("hempel", "hempel_sofa_dynamics"),
}

LOGGER = logging.getLogger("run_ablation")


def main() -> None:
    """Entry point for the two ablation studies."""
    argparse.ArgumentParser(description="Hempel vs SOFA / SOFA-dynamics ablation (reuses saved per-repeat metrics).").parse_args()

    config = load_config()
    log_dir = ensure_dir(PROJECT_ROOT / "results" / "logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "run_ablation.log"), logging.StreamHandler()],
    )

    # Load each run's per-repeat metrics once (guards if a run hasn't been done).
    needed = {name for pair in STUDIES.values() for name in pair}
    per_repeat = {name: load_per_repeat(config, name) for name in needed}

    for study_name, (base_key, aug_key) in STUDIES.items():
        LOGGER.info("=== Ablation study: %s ===", study_name)
        tables = run_ablation_study(per_repeat[base_key], per_repeat[aug_key], base_key, aug_key)
        out_dir = ensure_dir(PROJECT_ROOT / config["output"]["results_dir"] / "ablation" / study_name)
        for name, df in tables.items():
            save_table(df, out_dir, name)
        print(f"\n### {study_name} -- classification delta (augmented - baseline) ###")
        print(tables["classification_delta"].round(4).to_string())
        print(f"\n### {study_name} -- regression (full) delta ###")
        print(tables["regression_delta"].round(4).to_string())
        print(f"\nWritten to: {out_dir}")


if __name__ == "__main__":
    main()
