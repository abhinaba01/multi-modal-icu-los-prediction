"""Aggregate flow report for the X-ray linkage (safe to paste back from the lab).

Counts after each filter step; the share of stays linked per contract split and per
``anchor_year_group`` (MIMIC-CXR covers 2011-2016, so coverage differs by group); the
paired train count against the ~2,000 threshold of Stage 1; the spread of
``hours_before_t0``; LOS for paired vs unpaired stays; the view mix; images serving two
stays; and a date-alignment check. The PhysioNet page does not say that MIMIC-CXR dates
use MIMIC-IV's per-patient shift, so the check reports how many cohort patients with any
X-ray have one inside an ICU stay: near zero would mean the timelines do not line up.

Counts below 10 are suppressed (CLAUDE.md A3).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from fate_icu.data.cxr.linkage import Linkage
from fate_icu.data.cxr.metadata import parse_study_datetimes
from fate_icu.utils.aggregates import MIN_CELL, safe_share, suppress, suppress_counts

FLOW_ORDER = (
    "metadata_images",
    "unique_images",
    "with_valid_study_datetime",
    "frontal",
    "frontal_of_cohort_patients",
)
HOUR_EDGES = (0, 6, 12, 24, 36)  # plus a last edge at W + lookback


def _summary(values: pd.Series) -> dict[str, Any]:
    values = values.dropna().astype(float)
    if len(values) < MIN_CELL:
        return {"n": suppress(len(values))}
    q = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "n": suppress(len(values)),
        "mean": round(float(values.mean()), 2),
        "p5": round(float(q[0.05]), 2),
        "p25": round(float(q[0.25]), 2),
        "median": round(float(q[0.5]), 2),
        "p75": round(float(q[0.75]), 2),
        "p95": round(float(q[0.95]), 2),
    }


def date_alignment(metadata: pd.DataFrame, cohort: pd.DataFrame, lookback_hours: float) -> dict[str, Any]:
    """Share of cohort patients with any dated X-ray who have one inside an ICU stay."""
    meta = metadata[["subject_id", "StudyDate", "StudyTime"]].copy()
    meta["subject_id"] = pd.to_numeric(meta["subject_id"]).astype("int64")
    meta = meta[meta["subject_id"].isin(cohort["subject_id"].astype("int64"))]
    meta["when"] = parse_study_datetimes(meta["StudyDate"], meta["StudyTime"]).to_numpy()
    meta = meta[meta["when"].notna()]
    stays = cohort.reset_index()[["subject_id", "intime", "outtime"]].copy()
    stays["subject_id"] = stays["subject_id"].astype("int64")
    merged = meta.merge(stays, on="subject_id")
    inside = (merged["when"] >= merged["intime"] - pd.Timedelta(hours=lookback_hours)) & (
        merged["when"] <= merged["outtime"]
    )
    n_with_any = int(meta["subject_id"].nunique())
    n_inside = int(merged.loc[inside, "subject_id"].nunique())
    return {
        "cohort_patients_with_any_xray": suppress(n_with_any),
        "share_with_an_xray_during_an_icu_stay": safe_share(n_inside, n_with_any),
    }


def flow_report(
    linkage: Linkage,
    metadata: pd.DataFrame,
    cohort: pd.DataFrame,
    split: pd.Series,
    anchor_year_group: pd.Series,
    long_stay_threshold_days: float,
    lookback_hours: float,
    min_paired_train_stays: int = 2000,
    hours: int = 24,
) -> dict[str, Any]:
    """The aggregate report (see module docstring).

    Args:
        linkage: Output of :func:`~fate_icu.data.cxr.linkage.link`.
        metadata: The metadata the linkage used.
        cohort: Index ``stay_id``; ``subject_id``, ``intime``, ``outtime``, ``los``.
        split: FaTE split label per stay.
        anchor_year_group: ``anchor_year_group`` per ``subject_id``.
        long_stay_threshold_days: For the long-stay share of paired and unpaired stays.
        lookback_hours: The lookback used.
        min_paired_train_stays: Stage 1's threshold for paired train stays.
        hours: The window W (the histogram runs to W + lookback).
    """
    paired = linkage.paired.set_index("stay_id")
    cohort = cohort.sort_index()
    has = paired["has_cxr"].reindex(cohort.index).fillna(False).astype(bool)
    split = split.reindex(cohort.index)
    group = cohort["subject_id"].map(anchor_year_group).fillna("unknown")

    flow = {k: linkage.flow[k] for k in FLOW_ORDER if k in linkage.flow}
    flow.update(
        {
            "stay_image_pairs_in_window": linkage.flow["stay_image_pairs_in_window"],
            "stays": linkage.flow["stays"],
            "stays_with_cxr": linkage.flow["stays_with_cxr"],
        }
    )
    paired_train = int((has & (split == "train")).sum())

    hours_before = paired.loc[paired["has_cxr"], "hours_before_t0"]
    edges = sorted({*(e for e in HOUR_EDGES if e < hours + lookback_hours), hours + lookback_hours})
    labels = [f"{a:g}-{b:g} h" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    bins = pd.cut(hours_before, bins=edges, labels=labels, right=True, include_lowest=True)
    histogram = {str(k): suppress(int(v)) for k, v in bins.value_counts(sort=False).items()}

    los = cohort["los"].astype(float)
    los_by = {}
    for name, mask in (("paired", has), ("unpaired", ~has)):
        stats = _summary(los[mask])
        stats["long_stay_share"] = safe_share(
            int((los[mask].round(2) > long_stay_threshold_days).sum()), int(mask.sum())
        )
        los_by[name] = stats

    uses = linkage.paired.loc[linkage.paired["has_cxr"], "dicom_id"].value_counts()
    return {
        "flow": suppress_counts(flow),
        "linked_share_by_split": {
            name: safe_share(int((has & (split == name)).sum()), int((split == name).sum()))
            for name in ("train", "val", "test")
        },
        "paired_stays_by_split": {
            name: suppress(int((has & (split == name)).sum())) for name in ("train", "val", "test")
        },
        "paired_train_stays": suppress(paired_train),
        "below_min_paired_train_stays": paired_train < min_paired_train_stays,
        "min_paired_train_stays": min_paired_train_stays,
        "linked_share_by_anchor_year_group": {
            str(g): safe_share(int((has & (group == g)).sum()), int((group == g).sum()))
            for g in sorted(group.unique())
        },
        "hours_before_t0": _summary(hours_before),
        "hours_before_t0_histogram": histogram,
        "los_days": los_by,
        "long_stay_threshold_days": round(float(long_stay_threshold_days), 4),
        "views_of_paired_images": suppress_counts(
            paired.loc[paired["has_cxr"], "view"].value_counts().to_dict()
        ),
        "images_serving_more_than_one_stay": suppress(int((uses > 1).sum())),
        "date_alignment": date_alignment(metadata, cohort, lookback_hours),
        "lookback_hours": lookback_hours,
    }


def check_planted_answers(paired: pd.DataFrame, expected: pd.DataFrame) -> dict[str, Any]:
    """Compare a linkage on synthetic metadata with its planted answers (no IDs reported)."""
    got = paired.set_index("stay_id").reindex(expected.index)
    same_flag = got["has_cxr"].astype(bool) == expected["has_cxr"].astype(bool)
    linked = expected["has_cxr"].astype(bool)
    same_image = (got["dicom_id"].astype(object) == expected["dicom_id"].astype(object)) | ~linked
    same_view = (got["view"].astype(object) == expected["view"].astype(object)) | ~linked
    same_time = (got["study_datetime"] == expected["study_datetime"]) | ~linked
    correct = same_flag & same_image & same_view & same_time
    wrong = expected.loc[~correct, "scenario"].value_counts().to_dict()
    return {
        "stays": int(len(expected)),
        "all_correct": bool(correct.all()),
        "wrong_by_scenario": {k: int(v) for k, v in wrong.items()},
        "scenarios_planted": int(expected["scenario"].nunique()),
    }
