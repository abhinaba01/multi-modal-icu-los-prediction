"""Diagnosis features: top-2 ICD codes -> ICD-10 -> 22 WHO chapters.

Faithful port of the reference ``data_preparation.R`` diagnosis handling:

1. Take the top-2 diagnoses per stay (``seq_num`` 1 and 2) from ``diagnoses_icd``.
2. Convert ICD-9 codes to ICD-10 using the CMS General Equivalence Mapping (GEM)
   -- this is exactly what the R package ``touch::icd_map`` wraps. When an ICD-9
   code maps to several ICD-10 codes, collapse them to their longest common
   prefix (the R "reduce to 4/3/2/1 leading characters until unique" loop).
3. Group each resulting ICD-10 code into one of 22 WHO ICD-10 chapters
   (exact port of ``convert_ICD10_to_chapter``).

Codes that do not map (unknown ICD-9, non-ICD-10 GEM target, or a diagnosis that
is simply absent) become NA and are dropped later by the complete-case rule.

The GEM crosswalk is an external data asset (``data.icd9to10_gem`` in config).
If it is missing, :func:`load_gem` raises with download instructions rather than
silently degrading the diagnosis feature.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import MissingDataError, resolved_path
from src.data.loader import MIMICLoader

LOGGER = logging.getLogger(__name__)

DIAGNOSIS_COLUMNS = ["Diagnosis_1", "Diagnosis_2"]

# The 22 WHO ICD-10 chapters, keyed by first letter with numeric sub-ranges where
# a letter spans two chapters (D and H). Mirrors convert_ICD10_to_chapter in R.
_SIMPLE_LETTER_CHAPTER = {
    "A": "A00-B99", "B": "A00-B99",
    "C": "C00-D48",
    "E": "E00-E90",
    "F": "F00-F99",
    "G": "G00-G99",
    "I": "I00-I99",
    "J": "J00-J99",
    "K": "K00-K93",
    "L": "L00-L99",
    "M": "M00-M99",
    "N": "N00-N99",
    "O": "O00-O99",
    "P": "P00-P96",
    "Q": "Q00-Q99",
    "R": "R00-R99",
    "S": "S00-T98", "T": "S00-T98",
    "V": "V01-Y84", "W": "V01-Y84", "X": "V01-Y84", "Y": "V01-Y84",
    "Z": "Z00-Z99",
    "U": "U00-U99",
}


def load_gem(config: dict[str, Any]) -> dict[str, set[str]]:
    """Load the CMS ICD-9 -> ICD-10 GEM crosswalk into a lookup dict.

    Args:
        config: Parsed configuration (``data.icd9to10_gem`` gives the path).

    Returns:
        Mapping ``icd9_code -> {icd10_code, ...}`` (codes upper-cased, no dots).

    Raises:
        MissingDataError: If the GEM file is not present.
    """
    path = resolved_path(config, config["data"]["icd9to10_gem"])
    if not Path(path).exists():
        raise MissingDataError(
            f"ICD-9->ICD-10 GEM crosswalk not found at {path}.\n"
            "This is the CMS General Equivalence Mapping that the R package "
            "`touch` wraps. Obtain '2018_I9gem.txt' (three whitespace-separated "
            "columns: icd9, icd10, flags) from the CMS 2018 ICD-10 GEMs and place "
            "it there, or update data.icd9to10_gem in config.yaml."
        )
    gem: dict[str, set[str]] = {}
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            icd9 = parts[0].strip().upper()
            icd10 = parts[1].strip().upper()
            gem.setdefault(icd9, set()).add(icd10)
    LOGGER.info("Loaded GEM crosswalk: %d ICD-9 codes", len(gem))
    return gem


def _reduce_common_prefix(codes: set[str]) -> str | None:
    """Collapse several ICD-10 codes to their longest common leading substring.

    Ports the R loop that truncates to 4, then 3, 2, 1 characters until a single
    distinct value remains (else NA).

    Args:
        codes: Candidate ICD-10 codes.

    Returns:
        The common-prefix code, or ``None`` if not even the first character agrees.
    """
    cleaned = {c for c in codes if c and c != "NODX"}
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return next(iter(cleaned))
    for n in (4, 3, 2, 1):
        prefixes = {c[:n] for c in cleaned}
        if len(prefixes) == 1:
            return next(iter(prefixes))
    return None


def map_icd9_to_icd10(code9: str, gem: dict[str, set[str]]) -> str | None:
    """Map a single ICD-9 code to an ICD-10 code via the GEM.

    Args:
        code9: ICD-9 code (no dots), as stored in MIMIC.
        gem: GEM lookup from :func:`load_gem`.

    Returns:
        A single ICD-10 code (possibly a common-prefix truncation), or ``None``.
    """
    targets = gem.get(str(code9).strip().upper())
    if not targets:
        return None
    return _reduce_common_prefix(targets)


def icd10_to_chapter(code: str | None) -> str | None:
    """Map an ICD-10 code to one of the 22 WHO chapters.

    Exact port of ``convert_ICD10_to_chapter``: the D and H letters split into two
    chapters on the two-digit block number; all other letters map directly.

    Args:
        code: ICD-10 code (no dots), or ``None``.

    Returns:
        The chapter label (e.g. ``"I00-I99"``), or ``None`` if unmappable.
    """
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    code = str(code).strip().upper()
    if not code:
        return None
    letter = code[0]
    block = code[1:3]
    number = _leading_int(block)

    if letter == "D":
        if number is None:
            return None
        return "C00-D48" if number < 49 else "D50-D89"
    if letter == "H":
        if number is None:
            return None
        return "H00-H59" if number < 60 else "H60-H95"
    return _SIMPLE_LETTER_CHAPTER.get(letter)


def _leading_int(text: str) -> int | None:
    """Parse the leading integer of a short numeric string.

    Args:
        text: Two-character block (may contain trailing letters).

    Returns:
        The integer value, or ``None`` if no leading digits.
    """
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        else:
            break
    return int(digits) if digits else None


def build_diagnoses(loader: MIMICLoader, config: dict[str, Any], cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Build the two diagnosis-chapter features for the cohort.

    Args:
        loader: Configured MIMIC loader.
        config: Parsed configuration.
        cohort_df: Cohort with ``stay_id`` and ``hadm_id``.

    Returns:
        DataFrame indexed by ``stay_id`` with ``Diagnosis_1`` and ``Diagnosis_2``
        as chapter labels (NA where the diagnosis is absent or unmappable).
    """
    gem = load_gem(config)

    diag = loader.load_table(
        "diagnoses_icd",
        usecols=["hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    diag = diag[diag["seq_num"].isin([1, 2])].copy()
    cohort_hadm = cohort_df[["stay_id", "hadm_id"]].drop_duplicates()
    diag = diag.merge(cohort_hadm, on="hadm_id", how="inner")
    diag = diag.drop_duplicates(subset=["stay_id", "seq_num"])

    # Map each unique (code, version) once, then broadcast back.
    unique_codes = diag[["icd_code", "icd_version"]].drop_duplicates()
    chapter_lookup: dict[tuple[str, int], str | None] = {}
    for code, version in unique_codes.itertuples(index=False):
        chapter_lookup[(code, int(version))] = _code_to_chapter(code, int(version), gem)

    diag["chapter"] = [
        chapter_lookup.get((code, int(version)))
        for code, version in zip(diag["icd_code"], diag["icd_version"])
    ]

    first = diag[diag["seq_num"] == 1].set_index("stay_id")["chapter"].rename("Diagnosis_1")
    second = diag[diag["seq_num"] == 2].set_index("stay_id")["chapter"].rename("Diagnosis_2")
    out = pd.concat([first, second], axis=1)
    out = out.reindex(columns=DIAGNOSIS_COLUMNS)
    out.index.name = "stay_id"
    return out


def _code_to_chapter(code: str, version: int, gem: dict[str, set[str]]) -> str | None:
    """Convert a raw (code, icd_version) pair to a WHO chapter label.

    Args:
        code: ICD code as stored in MIMIC (no dots).
        version: 9 or 10.
        gem: GEM lookup.

    Returns:
        Chapter label or ``None``.
    """
    if code is None or (isinstance(code, float) and pd.isna(code)):
        return None
    if version == 9:
        code10 = map_icd9_to_icd10(code, gem)
    else:
        code10 = str(code).strip().upper()
    return icd10_to_chapter(code10)
