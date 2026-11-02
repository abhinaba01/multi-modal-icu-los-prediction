"""Patient-level train/val/test split shared by every FaTE-ICU model arm.

Each ``subject_id`` is assigned by a seeded hash (``contract.splits.fate``), so all of a
patient's stays land in the same split, the assignment is identical on every machine,
and a patient's split never changes when the cohort around it changes.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from fate_icu.data.contract import Contract, PatientHashSplit

SPLIT_ORDER = ("train", "val", "test")


def patient_hash_unit(subject_id: int, seed: int) -> float:
    """Map a patient to [0, 1): first 8 bytes of ``sha256("<seed>:<subject_id>")`` / 2^64."""
    digest = hashlib.sha256(f"{seed}:{int(subject_id)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_patient_splits(subject_ids: pd.Series, spec: PatientHashSplit) -> pd.Series:
    """Split label (``train``/``val``/``test``) for each row of ``subject_ids``."""
    units = {s: patient_hash_unit(s, spec.seed) for s in pd.unique(subject_ids)}
    u = subject_ids.map(units).to_numpy(dtype=float)
    cut_train = spec.fractions["train"]
    cut_val = cut_train + spec.fractions["val"]
    labels = np.where(u < cut_train, "train", np.where(u < cut_val, "val", "test"))
    return pd.Series(labels, index=subject_ids.index, name="split")


def fate_splits(cohort: pd.DataFrame, contract: Contract) -> pd.Series:
    """Split label per stay for a cohort indexed by ``stay_id`` with a ``subject_id`` column."""
    spec = contract.splits.fate
    if spec.group_column not in cohort.columns:
        raise KeyError(f"cohort needs a {spec.group_column!r} column")
    return assign_patient_splits(cohort[spec.group_column], spec)


def legacy_splits(n: int, contract: Contract) -> list[tuple[np.ndarray, np.ndarray]]:
    """The internship's repeated stay-level 80/20 splits (B0-faithful reproduction only)."""
    from fate_icu.tabular._legacy import legacy

    spec = contract.splits.legacy
    return legacy("evaluation.experiment").make_splits(n, spec.n_repeats, spec.test_size, spec.seed)
