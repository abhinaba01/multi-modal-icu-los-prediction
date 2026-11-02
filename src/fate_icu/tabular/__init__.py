"""Thin adapter over the internship pipeline in ``legacy_los/`` (never edited).

Public API::

    load_cohort()            # final contract cohort with t0
    load_static_features()   # internship features of a contract feature set
    load_splits()            # patient-level FaTE split
    train_two_stage_xgb()    # B0 on the FaTE split
    explain_shap()           # TreeSHAP for B0
"""

from fate_icu.tabular.api import (
    TabularTables,
    build_tables,
    load_cohort,
    load_splits,
    load_static_features,
    load_tables,
)
from fate_icu.tabular.model import ShapExplanation, TwoStageXGB, explain_shap, train_two_stage_xgb

__all__ = [
    "ShapExplanation",
    "TabularTables",
    "TwoStageXGB",
    "build_tables",
    "explain_shap",
    "load_cohort",
    "load_splits",
    "load_static_features",
    "load_tables",
    "train_two_stage_xgb",
]
