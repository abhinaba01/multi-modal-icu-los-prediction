"""Cohort construction: the inclusion / exclusion rules from the paper + R code.

Mirrors the reference ``Extraction.sql`` (see ``sql/Extraction.sql``) and
``data_preparation.R``:

* exclude ICU readmissions within 48h of a previous ICU discharge in the SAME
  hospital admission (``no_readmissions`` materialized view);
* exclude patients who died during the ICU stay (``deathtime`` at or before
  ``outtime``);
* keep ICU stays with ``los > 1`` and ``los <= 21`` days;
* keep adults (``anchor_age >= 18``); the paper states ">18" and the R code
  applies no age filter -- MIMIC-IV is adults-only, so ``>=18`` reconciles both.

The complete-case rule (drop any stay missing a feature) is applied later, after
all feature blocks are merged (see ``preprocessing/prepare.py``), matching R's
``complete.cases`` on the fully assembled table.

Output: one row per qualifying ``stay_id`` with the identifiers, times, LOS, and
the administrative / demographic columns needed downstream.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.data.loader import MIMICLoader

LOGGER = logging.getLogger(__name__)


def build_cohort(loader: MIMICLoader, config: dict[str, Any]) -> pd.DataFrame:
    """Build the base ICU-stay cohort.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration (``cohort`` block drives the thresholds).

    Returns:
        DataFrame with columns ``stay_id, subject_id, hadm_id, intime, outtime,
        los, first_careunit, admission_type, admission_location, gender,
        anchor_age`` (one row per included stay).
    """
    cohort_cfg = config["cohort"]

    icustays = loader.load_table(
        "icustays",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )
    admissions = loader.load_table(
        "admissions",
        usecols=["hadm_id", "deathtime", "admission_type", "admission_location"],
        parse_dates=["deathtime"],
    )
    patients = loader.load_table("patients", usecols=["subject_id", "gender", "anchor_age"])

    n0 = len(icustays)
    LOGGER.info("icustays: %d rows", n0)

    # --- Exclude ICU readmissions within 48h (within the same hadm_id) ---------
    keep_stays = _non_readmission_stays(icustays, hours=cohort_cfg["readmission_hours"])
    icustays = icustays[icustays["stay_id"].isin(keep_stays)]
    LOGGER.info("after 48h-readmission exclusion: %d rows (-%d)", len(icustays), n0 - len(icustays))

    # --- Join administrative + demographic sources -----------------------------
    df = icustays.merge(admissions, on="hadm_id", how="inner")
    df = df.merge(patients, on="subject_id", how="inner")

    # --- Exclude death during the ICU stay -------------------------------------
    # Keep when there is no death record, or death occurred strictly after ICU
    # discharge (deathtime > outtime).
    survived_icu = df["deathtime"].isna() | (df["deathtime"] > df["outtime"])
    n_before = len(df)
    df = df[survived_icu]
    LOGGER.info("after death-in-ICU exclusion: %d rows (-%d)", len(df), n_before - len(df))

    # --- LOS window: strictly > 1 day and <= 21 days ---------------------------
    los = pd.to_numeric(df["los"], errors="coerce")
    n_before = len(df)
    df = df[(los > cohort_cfg["min_los_days_exclusive"]) & (los <= cohort_cfg["max_los_days"])]
    LOGGER.info("after LOS in (%s, %s] days: %d rows (-%d)",
                cohort_cfg["min_los_days_exclusive"], cohort_cfg["max_los_days"], len(df), n_before - len(df))

    # --- Adults only -----------------------------------------------------------
    age = pd.to_numeric(df["anchor_age"], errors="coerce")
    n_before = len(df)
    df = df[age >= cohort_cfg["min_age"]]
    LOGGER.info("after age >= %s: %d rows (-%d)", cohort_cfg["min_age"], len(df), n_before - len(df))

    columns = [
        "stay_id", "subject_id", "hadm_id", "intime", "outtime", "los",
        "first_careunit", "admission_type", "admission_location", "gender", "anchor_age",
    ]
    cohort = df[columns].drop_duplicates(subset="stay_id").reset_index(drop=True)
    LOGGER.info("final base cohort: %d stays", len(cohort))
    return cohort


def _non_readmission_stays(icustays: pd.DataFrame, hours: int) -> set[int]:
    """Return stay_ids that are NOT a readmission within ``hours`` of a prior stay.

    Reproduces the ``no_readmissions`` view: within each hospital admission
    (``hadm_id``), order stays by ``intime`` and compare each stay's admission to
    the previous stay's discharge (``outtime``). A stay is a readmission when the
    gap to the previous discharge is under ``hours``; the first stay of an
    admission (no predecessor) is always kept.

    Args:
        icustays: Raw icustays with ``hadm_id, stay_id, intime, outtime``.
        hours: Readmission window in hours (paper: 48).

    Returns:
        Set of ``stay_id`` values to keep.
    """
    df = icustays[["hadm_id", "stay_id", "intime", "outtime"]].copy()
    df["intime"] = pd.to_datetime(df["intime"], errors="coerce")
    df["outtime"] = pd.to_datetime(df["outtime"], errors="coerce")
    df = df.sort_values(["hadm_id", "intime"])
    prev_out = df.groupby("hadm_id")["outtime"].shift(1)
    # Readmission when previous discharge is later than (intime - window): i.e.
    # the gap between prior discharge and this admission is < window.
    threshold = df["intime"] - pd.Timedelta(hours=hours)
    is_readmission = prev_out.notna() & (prev_out > threshold)
    keep = df.loc[~is_readmission, "stay_id"]
    return set(keep.astype("int64").tolist())
