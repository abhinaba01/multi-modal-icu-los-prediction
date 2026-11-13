"""Stage 5 guidelines: PDF text cleaning, headings and sections, page locators, JATS XML."""

from __future__ import annotations

import hashlib

import pytest
from rag_pdf import write_fixture_guideline, write_pdf

from fate_icu.rag.sources.guidelines import (
    GuidelineSpec,
    _dehyphenate,
    _drop_headers,
    _is_heading,
    parse_guideline,
)
from fate_icu.utils.env import REPO_ROOT

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "rag" / "guidelines"


def spec(**kw):
    base = dict(
        id="fixture_guideline",
        title="Synthetic guideline",
        citation="Test. 2026.",
        url="https://e.org",
        file="fixture_guideline.pdf",
        licence="synthetic test fixture",
        year=2026,
    )
    return GuidelineSpec(**{**base, **kw})


EXPECTED = [
    (
        "Abstract",
        ("101",),
        "This synthetic guideline was written for software tests. It describes how lactate, "
        "creatinine and urine output are monitored in critically ill adults.",
    ),
    (
        "1. Lactate monitoring",
        ("101",),
        "Measure lactate within the first hour of admission in adults with suspected "
        "shock. A lactate above 2 mmol/L signals tissue hypoperfusion and should prompt reassessment "
        "of fluid and "
        "vasopressor therapy.",
    ),
    (
        "2. Kidney function",
        ("102",),
        "Serum creatinine and urine output define acute kidney injury stages. A rise of "
        "0.3 mg/dL within 48 hours is stage 1.",
    ),
    (
        "2.1 Urine output",
        ("102",),
        "Urine output below 0.5 mL/kg/h for 6 hours also defines stage 1. Oliguria is "
        "associated with a longer stay.",
    ),
    (
        "Recommendation 3",
        ("103",),
        "We suggest light rather than deep sedation in ventilated adults, because deep "
        "sedation prolongs mechanical ventilation and the intensive care stay.",
    ),
    (
        "3. Appendix: Definitions",
        ("104",),
        "Shock is defined in this synthetic text as hypotension that needs vasopressors.",
    ),
]


def _summary(doc):
    return [(s.title, s.locators, " ".join(s.paragraphs)) for s in doc.sections]


def test_the_committed_fixture_parses_to_the_expected_sections():
    doc = parse_guideline(spec(), FIXTURE_DIR)
    assert _summary(doc) == EXPECTED
    assert doc.source == "guideline" and doc.doc_id == "fixture_guideline" and doc.meta["pages"] == 4
    raw = (FIXTURE_DIR / "fixture_guideline.pdf").read_bytes()
    assert doc.meta["file_sha256"] == hashlib.sha256(raw).hexdigest()
    text = " ".join(p for s in doc.sections for p in s.paragraphs)
    for absent in ("Synth Crit Care Guidel", "Rao A", "Lindqvist", "synthetic reviewers"):
        assert absent not in text


def test_the_fixture_can_be_regenerated(tmp_path):
    write_fixture_guideline(tmp_path / "fixture_guideline.pdf")
    assert (tmp_path / "fixture_guideline.pdf").read_bytes() == (
        FIXTURE_DIR / "fixture_guideline.pdf"
    ).read_bytes()
    assert _summary(parse_guideline(spec(), tmp_path)) == EXPECTED


def test_page_ranges():
    later = parse_guideline(spec(first_page=3), FIXTURE_DIR)
    assert [s.title for s in later.sections] == ["Recommendation 3", "3. Appendix: Definitions"]
    earlier = parse_guideline(spec(last_page=2), FIXTURE_DIR)
    assert [s.title for s in earlier.sections] == [
        "Abstract",
        "1. Lactate monitoring",
        "2. Kidney function",
        "2.1 Urine output",
    ]


def test_extra_excluded_titles():
    doc = parse_guideline(spec(exclude_titles=["2.1 Urine output", "Abstract"]), FIXTURE_DIR)
    assert "2.1 Urine output" not in [s.title for s in doc.sections] and "Abstract" not in [
        s.title for s in doc.sections
    ]


@pytest.mark.parametrize(
    "line, expected",
    [
        ("2.1 Urine output", True),
        ("1. Lactate monitoring", True),
        ("Recommendation 12", True),
        ("Chapter 3: Definitions", True),
        ("ABSTRACT", True),
        ("Haemodynamic Monitoring and Support", True),
        ("Initial resuscitation", False),  # sentence case without a number: not detected
        ("1. Measure lactate within the first hour of admission in adults with", False),
        ("1. Rao A, Mensah K. A synthetic reference about lactate. Synth J. 2020;1:1-10.", False),
        ("This is a sentence.", False),
        ("123", False),
        ("", False),
    ],
)
def test_heading_rules(line, expected):
    assert _is_heading(line) is expected


def test_running_headers_need_three_pages_and_a_third_of_them():
    def pages(n_with_header, n_pages):
        return [
            (str(i), (["Running Title 2026"] if i < n_with_header else []) + [f"Body {i}."])
            for i in range(n_pages)
        ]

    assert all("Running" not in line for _, line in _drop_headers(pages(3, 10)))
    assert any("Running" in line for _, line in _drop_headers(pages(2, 3)))  # only 2 pages
    assert any("Running" in line for _, line in _drop_headers(pages(3, 20)))  # 15% of pages
    assert [line for _, line in _drop_headers([("1", ["12", "Page 3 of 9", "Text."])] * 1)] == ["Text."]


def test_dehyphenation_rules():
    assert _dehyphenate([("1", "acute kidney in-"), ("2", "jury stages.")]) == [
        ("1", "acute kidney injury stages.")
    ]
    assert _dehyphenate([("1", "the anti-"), ("1", "Inflammatory")]) == [
        ("1", "the anti-"),
        ("1", "Inflammatory"),
    ]
    assert _dehyphenate([("1", "COVID-"), ("1", "19 patients")]) == [("1", "COVID-"), ("1", "19 patients")]


def test_missing_wrong_and_empty_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="fixture_guideline"):
        parse_guideline(spec(), tmp_path)
    (tmp_path / "g.docx").write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        parse_guideline(spec(file="g.docx"), tmp_path)
    write_pdf(tmp_path / "blank.pdf", [[], []])
    with pytest.raises(ValueError, match="scanned"):
        parse_guideline(spec(file="blank.pdf"), tmp_path)


def test_jats_guidelines(tmp_path):
    (tmp_path / "g.xml").write_text(
        "<article><front><article-meta><title-group><article-title>G</article-title></title-group>"
        "</article-meta></front><body><sec><title>Methods</title><p>Panel process.</p></sec>"
        "<sec><title>Recommendations</title><p>Use light sedation.<xref>12</xref></p></sec>"
        "<sec><title>Acknowledgements</title><p>Thanks.</p></sec></body>"
        "<back><ref-list><ref>R</ref></ref-list></back></article>",
        encoding="utf-8",
    )
    doc = parse_guideline(spec(file="g.xml", exclude_titles=["Methods"]), tmp_path)
    assert _summary(doc) == [("Recommendations", (), "Use light sedation.")]


def test_spec_validation():
    with pytest.raises(ValueError):
        spec(licence="")
    with pytest.raises(ValueError):
        spec(first_page=0)
    with pytest.raises(ValueError):
        spec(first_page=3, last_page=2)
    with pytest.raises(ValueError):
        spec(id="Bad Id")
