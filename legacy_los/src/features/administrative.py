"""Administrative features (paper Table 1, 'Administrative').

First care unit, admission type, and admission location. Diagnoses and LOS are
also 'Administrative' in the paper but are handled by ``diagnoses.py`` and the
target builder respectively.
"""

from __future__ import annotations

import pandas as pd

# All three are categorical (label-encoded later).
ADMINISTRATIVE_CATEGORICAL = ["first_careunit", "admission_type", "admission_location"]
ADMINISTRATIVE_COLUMNS = ADMINISTRATIVE_CATEGORICAL


def build_administrative(cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Select administrative features from the cohort.

    Args:
        cohort_df: Cohort with ``stay_id`` and the administrative columns.

    Returns:
        DataFrame indexed by ``stay_id`` with the administrative columns.
    """
    out = cohort_df[["stay_id", *ADMINISTRATIVE_COLUMNS]].drop_duplicates("stay_id").set_index("stay_id")
    return out
