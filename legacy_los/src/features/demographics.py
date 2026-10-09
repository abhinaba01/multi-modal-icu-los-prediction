"""Demographic features: age and gender (paper Table 1, 'Demographic')."""

from __future__ import annotations

import pandas as pd

# gender is categorical (label-encoded later); anchor_age is numeric.
DEMOGRAPHIC_CATEGORICAL = ["gender"]
DEMOGRAPHIC_NUMERIC = ["anchor_age"]
DEMOGRAPHIC_COLUMNS = DEMOGRAPHIC_NUMERIC + DEMOGRAPHIC_CATEGORICAL


def build_demographics(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Select demographic features from the cohort.

    Args:
        cohort_df: Cohort with ``stay_id``, ``anchor_age``, ``gender``.

    Returns:
        DataFrame indexed by ``stay_id`` with ``anchor_age`` and ``gender``.
    """
    out = cohort_df[["stay_id", *DEMOGRAPHIC_COLUMNS]].drop_duplicates("stay_id").set_index("stay_id")
    return out
