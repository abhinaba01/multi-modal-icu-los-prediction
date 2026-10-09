"""Assemble the per-stay analysis tables from the raw MIMIC-IV CSVs.

Produces three feature tables that all share one base cohort, so the ablation
studies compare like with like:

* ``hempel``               -- the paper's baseline features only.
* ``hempel_sofa``          -- baseline + static (worst-in-24h) SOFA.
* ``hempel_sofa_dynamics`` -- baseline + the three SOFA-dynamics groups
  (trajectory/velocity, organ-failure onset sequence, sub-score interactions).

Tables store RAW (unscaled, string-categorical) features + ``los``. Cleaning,
encoding, scaling and target construction happen in ``preprocessing/prepare.py``.

Performance: :func:`_scan_sources` reads each big event table (``chartevents``,
``labevents``, and for SOFA ``inputevents``/``outputevents``) exactly ONCE via
:func:`aggregate.stream_multi`, pulling the baseline mean features and the SOFA
raw inputs in the same pass. :func:`assemble_all` builds all three tables from a
single set of scans (used by ``build_features.py --which all``).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.features.administrative import ADMINISTRATIVE_CATEGORICAL, build_administrative
from src.features.aggregate import MeanSink, RawSink, stream_multi
from src.features.cohort import build_cohort
from src.features.demographics import (DEMOGRAPHIC_CATEGORICAL, DEMOGRAPHIC_NUMERIC, build_demographics)
from src.features.diagnoses import DIAGNOSIS_COLUMNS, build_diagnoses
from src.features.labs import LAB_COLUMNS, LAB_ITEMIDS
from src.features.sofa_dynamics import (build_sofa_dynamics, build_sofa_interactions,
                                        dynamics_feature_columns, interaction_feature_columns,
                                        score_sofa_windows)
from src.features.sofa_static import (CHART_ITEMIDS, LAB_ITEMIDS as SOFA_LAB_ITEMIDS, SOFA_STATIC_COLUMNS,
                                      URINE_ITEMIDS, VASO_ITEMIDS, build_sofa_static)
from src.features.sofa_windows import build_sofa_windows_raw
from src.features.vitals import VITAL_COLUMNS, VITAL_ITEMIDS
from src.data.loader import MIMICLoader

LOGGER = logging.getLogger(__name__)

# Baseline feature groups (the paper's Table 1 feature set). Categorical order
# matches the reference R label-encoding column order.
BASE_CATEGORICAL = ADMINISTRATIVE_CATEGORICAL + DIAGNOSIS_COLUMNS + DEMOGRAPHIC_CATEGORICAL
BASE_NUMERIC = DEMOGRAPHIC_NUMERIC + VITAL_COLUMNS + LAB_COLUMNS
BASE_FEATURES = BASE_CATEGORICAL + BASE_NUMERIC
TARGET = "los"


def _build_cohort(loader: MIMICLoader, config: dict[str, Any], limit: int | None) -> pd.DataFrame:
    """Build the cohort, optionally truncated for a smoke test.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        limit: Optional cap on cohort stays.

    Returns:
        Cohort DataFrame.
    """
    cohort = build_cohort(loader, config)
    if limit is not None:
        cohort = cohort.head(limit).copy()
        LOGGER.warning("LIMIT active: cohort truncated to %d stays (smoke test only)", len(cohort))
    return cohort


def _scan_sources(loader: MIMICLoader, config: dict[str, Any], cohort: pd.DataFrame,
                  want_sofa: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame] | None]:
    """Read each big event table once, returning baseline means (+ SOFA raw).

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        cohort: Base cohort with ``stay_id``, ``hadm_id``, ``intime``.
        want_sofa: If True, also collect the raw SOFA inputs in the same passes.

    Returns:
        ``(vitals_means, labs_means, events)`` where ``events`` is ``None`` when
        ``want_sofa`` is False, else a dict with ``chart``/``lab``/``input``/
        ``output`` raw-row frames.
    """
    chart_sinks: list[MeanSink | RawSink] = [MeanSink("vitals", VITAL_ITEMIDS)]
    lab_sinks: list[MeanSink | RawSink] = [MeanSink("labs", LAB_ITEMIDS)]
    if want_sofa:
        chart_sinks.append(RawSink("sofa_chart", set(CHART_ITEMIDS)))
        lab_sinks.append(RawSink("sofa_lab", set(SOFA_LAB_ITEMIDS)))

    chart = stream_multi(loader, config, "chartevents", "stay_id", "valuenum", "charttime",
                         cohort[["stay_id", "intime"]], chart_sinks)
    labs = stream_multi(loader, config, "labevents", "hadm_id", "valuenum", "charttime",
                        cohort[["hadm_id", "stay_id", "intime"]], lab_sinks)

    events: dict[str, pd.DataFrame] | None = None
    if want_sofa:
        inputs = stream_multi(loader, config, "inputevents", "stay_id", "rate", "starttime",
                              cohort[["stay_id", "intime"]], [RawSink("input", set(VASO_ITEMIDS))])["input"]
        outputs = stream_multi(loader, config, "outputevents", "stay_id", "value", "charttime",
                               cohort[["stay_id", "intime"]], [RawSink("output", set(URINE_ITEMIDS))])["output"]
        events = {"chart": chart["sofa_chart"], "lab": labs["sofa_lab"], "input": inputs, "output": outputs}

    return chart["vitals"], labs["labs"], events


def _assemble_base_from_parts(loader: MIMICLoader, config: dict[str, Any], cohort: pd.DataFrame,
                              vitals: pd.DataFrame, labs: pd.DataFrame) -> pd.DataFrame:
    """Combine cohort + scanned vitals/labs + the cheap feature blocks into base.

    Args:
        loader: Configured MIMIC loader (for the diagnoses table).
        config: Parsed configuration.
        cohort: Base cohort.
        vitals: Per-stay vital means (from :func:`_scan_sources`).
        labs: Per-stay lab means (from :func:`_scan_sources`).

    Returns:
        Baseline feature table indexed by ``stay_id`` (``los`` + all baseline features).
    """
    demographics = build_demographics(cohort)
    administrative = build_administrative(cohort)
    diagnoses = build_diagnoses(loader, config, cohort)
    los = cohort.set_index("stay_id")["los"].astype(float)

    base = pd.concat([los.rename(TARGET), administrative, diagnoses, demographics, vitals, labs], axis=1)
    base = base.reindex(columns=[TARGET, *BASE_FEATURES])
    base.index.name = "stay_id"
    LOGGER.info("assembled base table: %d stays x %d columns", base.shape[0], base.shape[1])
    return base


def _add_static_sofa(base: pd.DataFrame, events: dict[str, pd.DataFrame], cohort: pd.DataFrame,
                     strict: bool) -> pd.DataFrame:
    """Attach static SOFA columns to the base table (0-filled where absent).

    Args:
        base: Baseline feature table.
        events: SOFA raw-input frames.
        cohort: Base cohort (for the stay-id universe).
        strict: Raise on degenerate SOFA components.

    Returns:
        Baseline + static SOFA table.
    """
    sofa = build_sofa_static(events, cohort["stay_id"], strict=strict)
    combined = base.join(sofa, how="left")
    combined[SOFA_STATIC_COLUMNS] = combined[SOFA_STATIC_COLUMNS].fillna(0.0)
    return combined


def _add_sofa_dynamics(base: pd.DataFrame, events: dict[str, pd.DataFrame], cohort: pd.DataFrame,
                       config: dict[str, Any], strict: bool) -> pd.DataFrame:
    """Attach the three SOFA-dynamics feature groups to the base table.

    Args:
        base: Baseline feature table.
        events: SOFA raw-input frames.
        cohort: Base cohort.
        config: Parsed configuration (SOFA windowing).
        strict: Raise on degenerate SOFA components.

    Returns:
        Baseline + dynamics + interactions table.
    """
    win = _window_config(config)
    static = build_sofa_static(events, cohort["stay_id"], strict=strict)
    windows_raw = build_sofa_windows_raw(events, n_windows=win["n_windows"], window_hours=win["window_hours"])
    scored = score_sofa_windows(windows_raw) if not windows_raw.empty else pd.DataFrame(columns=["stay_id", "win"])
    dynamics = build_sofa_dynamics(scored, cohort_stay_ids=cohort["stay_id"],
                                   n_windows=win["n_windows"], window_hours=win["window_hours"],
                                   onset_threshold=win["onset_threshold"])
    interactions = build_sofa_interactions(static)

    combined = base.join(dynamics, how="left").join(interactions, how="left")
    added = dynamics_feature_columns(win["n_windows"])
    added_columns = added["continuous"] + added["indicator"] + interaction_feature_columns()
    combined[added_columns] = combined[added_columns].fillna(0.0)
    return combined


def assemble_base(loader: MIMICLoader, config: dict[str, Any], limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the base cohort and the Hempel baseline feature table.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        limit: Optional cohort cap (smoke tests).

    Returns:
        ``(cohort_df, base_df)``.
    """
    cohort = _build_cohort(loader, config, limit)
    vitals, labs, _ = _scan_sources(loader, config, cohort, want_sofa=False)
    return cohort, _assemble_base_from_parts(loader, config, cohort, vitals, labs)


def assemble_hempel(loader: MIMICLoader, config: dict[str, Any], limit: int | None = None) -> pd.DataFrame:
    """Assemble the Hempel baseline table.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        limit: Optional cohort cap.

    Returns:
        Baseline feature table indexed by ``stay_id``.
    """
    _, base = assemble_base(loader, config, limit=limit)
    return base


def assemble_sofa(loader: MIMICLoader, config: dict[str, Any], limit: int | None = None,
                  strict: bool = True) -> pd.DataFrame:
    """Assemble baseline + static SOFA (single pass over each event table).

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        limit: Optional cohort cap.
        strict: Raise if SOFA components are degenerate.

    Returns:
        Feature table indexed by ``stay_id``.
    """
    cohort = _build_cohort(loader, config, limit)
    vitals, labs, events = _scan_sources(loader, config, cohort, want_sofa=True)
    base = _assemble_base_from_parts(loader, config, cohort, vitals, labs)
    return _add_static_sofa(base, events, cohort, strict)


def assemble_sofa_dynamics(loader: MIMICLoader, config: dict[str, Any], limit: int | None = None,
                           strict: bool = True) -> pd.DataFrame:
    """Assemble baseline + the three SOFA-dynamics feature groups (single pass).

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        limit: Optional cohort cap.
        strict: Raise if SOFA components are degenerate.

    Returns:
        Feature table indexed by ``stay_id``.
    """
    cohort = _build_cohort(loader, config, limit)
    vitals, labs, events = _scan_sources(loader, config, cohort, want_sofa=True)
    base = _assemble_base_from_parts(loader, config, cohort, vitals, labs)
    return _add_sofa_dynamics(base, events, cohort, config, strict)


def assemble_all(loader: MIMICLoader, config: dict[str, Any], limit: int | None = None,
                 strict: bool = True) -> dict[str, pd.DataFrame]:
    """Build all three feature tables from ONE set of raw-table scans.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        limit: Optional cohort cap.
        strict: Raise if SOFA components are degenerate.

    Returns:
        Dict with keys ``hempel``, ``hempel_sofa``, ``hempel_sofa_dynamics``.
    """
    cohort = _build_cohort(loader, config, limit)
    vitals, labs, events = _scan_sources(loader, config, cohort, want_sofa=True)
    base = _assemble_base_from_parts(loader, config, cohort, vitals, labs)
    return {
        "hempel": base,
        "hempel_sofa": _add_static_sofa(base, events, cohort, strict),
        "hempel_sofa_dynamics": _add_sofa_dynamics(base, events, cohort, config, strict),
    }


def _window_config(config: dict[str, Any]) -> dict[str, int]:
    """Return the SOFA windowing parameters (with sensible defaults).

    Args:
        config: Parsed configuration.

    Returns:
        Dict with ``n_windows``, ``window_hours``, ``onset_threshold``.
    """
    win = config.get("sofa_windows", {})
    return {
        "n_windows": int(win.get("n_windows", 4)),
        "window_hours": int(win.get("window_hours", 6)),
        "onset_threshold": int(win.get("onset_threshold", 1)),
    }
