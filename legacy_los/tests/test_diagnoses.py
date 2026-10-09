"""Tests for the ICD-9->10 mapping and 22-chapter grouping (diagnoses.py)."""

from __future__ import annotations

import pytest

from src.features.diagnoses import _reduce_common_prefix, icd10_to_chapter, map_icd9_to_icd10


@pytest.mark.parametrize("code,expected", [
    ("I10", "I00-I99"),
    ("K7460", "K00-K93"),
    ("C34", "C00-D48"),
    ("D45", "C00-D48"),    # D block < 49 -> neoplasm range
    ("D50", "D50-D89"),    # D block > 48 -> blood range
    ("H10", "H00-H59"),    # H block < 60 -> eye
    ("H65", "H60-H95"),    # H block > 59 -> ear
    ("S72", "S00-T98"),
    ("T81", "S00-T98"),
    ("V50", "V01-Y84"),
    ("Z51", "Z00-Z99"),
    ("U071", "U00-U99"),   # COVID-19
    ("A00", "A00-B99"),
    ("E11", "E00-E90"),
])
def test_icd10_to_chapter(code, expected):
    assert icd10_to_chapter(code) == expected


def test_icd10_to_chapter_unmappable():
    assert icd10_to_chapter(None) is None
    assert icd10_to_chapter("") is None
    assert icd10_to_chapter("999") is None  # no leading letter


def test_reduce_common_prefix_collapses_to_shared_stem():
    assert _reduce_common_prefix({"K740", "K7460", "K7469"}) == "K74"


def test_reduce_common_prefix_single_code_kept_whole():
    assert _reduce_common_prefix({"K740"}) == "K740"


def test_reduce_common_prefix_no_agreement_returns_none():
    assert _reduce_common_prefix({"A01", "B02"}) is None


def test_reduce_common_prefix_drops_nodx():
    assert _reduce_common_prefix({"K740", "NODX"}) == "K740"


def test_map_icd9_to_icd10_uses_gem_and_reduces():
    gem = {"5715": {"K740", "K7460", "K7469"}}
    assert map_icd9_to_icd10("5715", gem) == "K74"


def test_map_icd9_to_icd10_unknown_code():
    assert map_icd9_to_icd10("99999", {}) is None
