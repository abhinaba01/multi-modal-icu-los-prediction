"""Stage 5 guidelines: PDF text cleaning, headings and sections, page locators, JATS XML."""

from __future__ import annotations

import hashlib

import pytest
from rag_pdf import write_fixture_guideline, write_pdf

from fate_icu.rag.sources.guidelines import (
    Block,
    GuidelineSpec,
    _cut,
    _dehyphenate,
    _drop_blocks,
    _drop_headers,
    _heading_in_context,
    _is_heading,
    _sections,
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
        # found on the real guidelines (2026-11-15)
        ("(Not Graded)", False),  # KDIGO's grade on its own line
        ("Recommendation: We suggest using low-dose ketamine", False),  # PADIS label
        ("Recommendation 2.1.2. ( Not Graded )", True),
        ("Chapter 3.4: The use of diuretics in AKI", True),
        ("Dementia (PAINAD) (47), and the Behavior Pain Assessment", False),  # cites
        ("Immunosuppression [173, 180, 181]", False),
        ("Table 1 (continued)", True),
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
    with pytest.raises(ValueError, match="bad pattern"):
        spec(start_at="^Intro(")
    with pytest.raises(ValueError, match="bad pattern"):
        spec(drop_blocks=[{"start": "^Box$", "end": "["}])
    with pytest.raises(ValueError):
        spec(drop_blocks=[{"start": "^Box$", "end": "^End$", "max_lines": 1}])
    with pytest.raises(ValueError):
        spec(drop_blocks=[{"start": "^Box$", "end": "^End$", "colour": "red"}])


def _lines(*texts, page="1"):
    return [(page, t) for t in texts]


ARTICLE = _lines(
    "Intensive Care Med",
    "A. Author, B. Author",
    "Introduction",
    "Sepsis is common.",
    "See the Introduction of the companion paper.",
    "Acknowledgements",
    "We thank the panel.",
    "Supplementary Information",
    "The online version contains supplementary material.",
)


def test_start_at_and_end_at_cut_the_whole_lines_they_match():
    cut = _cut(ARTICLE, spec(start_at=r"^Introduction$", end_at=r"^Supplementary Information$"))
    assert [t for _, t in cut] == [
        "Introduction",
        "Sepsis is common.",
        "See the Introduction of the companion paper.",
        "Acknowledgements",
        "We thank the panel.",
    ]
    # case-insensitive
    assert _cut(ARTICLE, spec(start_at=r"^introduction$"))[0][1] == "Introduction"
    # end_at is searched from the line after the start, so the start line never ends the cut
    assert [t for _, t in _cut(ARTICLE, spec(start_at=r"^Sepsis", end_at=r"^S"))] == ["Sepsis is common."]


def test_cut_markers_that_match_nothing_are_errors():
    with pytest.raises(ValueError, match="start_at"):
        _cut(ARTICLE, spec(start_at=r"^Background$"))
    with pytest.raises(ValueError, match="end_at"):
        _cut(ARTICLE, spec(end_at=r"^References$"))
    with pytest.raises(ValueError, match="end_at"):  # only found before the start
        _cut(ARTICLE, spec(start_at=r"^Sepsis", end_at=r"^Introduction$"))
    with pytest.raises(ValueError, match="start_at"):  # matched at the start of a line only
        _cut(ARTICLE, spec(start_at=r"companion"))


def test_no_markers_keep_every_line():
    assert _cut(ARTICLE, spec()) == ARTICLE


def test_cut_points_on_a_parsed_pdf():
    doc = parse_guideline(
        spec(start_at=r"^1\. Lactate monitoring$", end_at=r"^Recommendation 3$"), FIXTURE_DIR
    )
    assert [s.title for s in doc.sections] == [
        "1. Lactate monitoring",
        "2. Kidney function",
        "2.1 Urine output",
    ]
    with pytest.raises(ValueError, match="start_at"):
        parse_guideline(spec(start_at=r"^Introduction$"), FIXTURE_DIR)


BOX = Block(start=r"^SUPPLEMENTARY MATERIAL$", end=r"^Supplementary material is linked to", max_lines=5)


def test_drop_blocks_removes_every_box_inclusive():
    lines = _lines(
        "Creatinine rises.",
        "SUPPLEMENTARY MATERIAL",
        "Table S1 lists the studies.",
        "Supplementary material is linked to the online version.",
        "Urine output falls.",
        "Supplementary Material",  # matched case-insensitively
        "Supplementary material is linked to the paper.",
        "Dialysis may follow.",
    )
    assert [t for _, t in _drop_blocks(lines, [BOX], "g")] == [
        "Creatinine rises.",
        "Urine output falls.",
        "Dialysis may follow.",
    ]
    assert _drop_blocks(lines, [], "g") == lines


def test_a_box_that_does_not_close_within_its_limit_is_an_error():
    lines = _lines("SUPPLEMENTARY MATERIAL", *[f"Line {i}." for i in range(6)], page="12")
    with pytest.raises(ValueError, match=r"page 12.*5 lines"):
        _drop_blocks(lines, [BOX], "kdigo")
    closed_late = [*lines, ("12", "Supplementary material is linked to it.")]
    with pytest.raises(ValueError):
        _drop_blocks(closed_late, [BOX], "kdigo")
    # max_lines counts both the first and the last line
    just_in = _lines("SUPPLEMENTARY MATERIAL", "a.", "b.", "c.", "Supplementary material is linked to x.")
    assert _drop_blocks(just_in, [BOX], "kdigo") == []


@pytest.mark.parametrize(
    "texts, i, expected",
    [
        (("Sepsis is common.", "Initial Resuscitation", "Give fluids."), 1, True),
        # a Title-Case phrase wrapped inside a sentence: the line before does not end one
        (("in patients with", "Septic Shock Who Need", "Vasopressors we suggest."), 1, False),
        # the next line continues in lower case
        (("Sepsis is common.", "Acute Kidney Injury", "is defined by creatinine."), 1, False),
        # cut at a small word or a hyphen
        (("Sepsis is common.", "Fluids and", "Vasopressors."), 1, False),
        (("Sepsis is common.", "Anti-", "Inflammatory drugs."), 1, False),
        # a heading after a heading
        (("Kidney Function", "Urine Output", "Oliguria matters."), 1, True),
        # keyword headings need no sentence end before them
        (("in adults with", "Recommendation 12", "We suggest light sedation."), 1, True),
        # the last line is never a heading
        (("Sepsis is common.", "Initial Resuscitation"), 1, False),
        # the first line is a heading when it looks like one
        (("Initial Resuscitation", "Give fluids."), 0, True),
    ],
)
def test_headings_are_judged_in_context(texts, i, expected):
    assert _heading_in_context(_lines(*texts), i) is expected


def test_a_recommendation_keeps_its_first_line_and_its_grade():
    lines = _lines(
        "Question: Should ketamine be used as an adjunct to an opioid?",
        "Recommendation: We suggest using low-dose ketamine",
        "(0.5 mg/kg IVP x 1 followed by 1-2 μg/kg/min infusion) as an",
        "adjunct to opioid therapy (conditional recommendation).",
        "Rationale: Two trials were found.",
        "",
        "2.3.1: Evaluate patients with AKI promptly.",
        "(Not Graded)",
        "2.3.2: Monitor patients with AKI.",
    )
    [section] = _sections(lines, spec().excluded())
    text = " ".join(section.paragraphs)
    assert "Recommendation: We suggest using low-dose ketamine (0.5 mg/kg" in text
    assert "promptly. (Not Graded) 2.3.2" in text


def test_sponsorship_and_disclaimer_boilerplate_is_excluded():
    lines = _lines(
        "Urine output falls.",
        "SPONSORSHIP",
        "KDIGO thanks its sponsors.",
        "Disclaimer",
        "Use with care.",
        "Chapter 2.2: Risk assessment",
        "Risk is high.",
    )
    sections = _sections(lines, spec().excluded())
    assert [s.title for s in sections] == ["", "Chapter 2.2: Risk assessment"]


def test_one_line_blocks_drop_single_lines():
    journal = [
        Block(start=r"^http://www\.kidney-international\.org", end=r"^http://www\.kidney-international\.org"),
        Block(start=r"^& 2012 KDIGO$", end=r"^& 2012 KDIGO$"),
    ]
    lines = _lines(
        "renal perfusion, but not",
        "http://www.kidney-international.org chapter 1.1",
        "& 2012 KDIGO",
        "to sustain filtration.",
    )
    assert [t for _, t in _drop_blocks(lines, journal, "kdigo")] == [
        "renal perfusion, but not",
        "to sustain filtration.",
    ]


def test_a_wrapped_reference_entry_does_not_end_the_reference_list():
    lines = _lines(
        "Sepsis is common.",
        "References",
        "1. Singer M, Deutschman CS",
        "Seymour CW. Sepsis-3. JAMA. 2016;315:801.",
        "Appendix 1",
        "Definitions are listed here.",
    )
    sections = _sections(lines, spec().excluded())
    assert [s.title for s in sections] == ["", "Appendix 1"]
    assert "Singer" not in " ".join(p for s in sections for p in s.paragraphs)
