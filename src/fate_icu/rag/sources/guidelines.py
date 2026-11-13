"""Clinical guidelines from PDF (pypdf) or JATS XML, as Documents (Stage 5).

The user downloads each guideline and records its licence in the config. PDF text is read
page by page with pypdf's "plain" mode, which follows the content stream and so keeps a
two-column journal page in reading order. Then:

* running headers and footers are dropped: a line that, with its digits replaced, occurs
  on at least 30% of the pages (and at least 3), and page-number-only lines;
* a word hyphenated at a line end is rejoined when the next line starts in lower case;
* headings are short lines (at most 12 words, no final full stop): numbered with a short
  or Title Case remainder ("2.3 Urine output"), introduced by a keyword ("Recommendation
  12", "Chapter 3"), or in Title Case or capitals. Sentence-case headings without a number
  are not detected; their text stays in the previous section, so nothing is lost;
* a heading starts a section; its lines form paragraphs, and every paragraph carries the
  page label where it starts (the passage locator, e.g. "p. 1183");
* excluded sections (references, acknowledgements, funding, conflicts of interest, ...)
  are skipped. Inside a reference list only a Chapter, Section, Appendix, Part or Annex
  heading ends the skip, because wrapped reference entries ("1. Singer M, Deutschman CS")
  look like numbered headings.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fate_icu.rag.corpus import Document, Section
from fate_icu.rag.sources.jats import parse_jats_sections, parse_xml
from fate_icu.rag.text import normalise_text

DEFAULT_EXCLUDED_HEADINGS = (
    "References",
    "Bibliography",
    "Acknowledgements",
    "Acknowledgments",
    "Funding",
    "Conflicts of interest",
    "Conflict of interest",
    "Disclosures",
    "Author information",
    "Supplementary information",
    "Supplementary Material",
)
REFERENCE_HEADINGS = frozenset({"references", "bibliography"})
HEADER_SHARE, HEADER_MIN_PAGES = 0.3, 3

_NUMBER_PREFIX = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(\S.*)$")
_KEYWORD = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:Recommendation|Chapter|Section|Appendix|Part|Annex)\b")
_CHAPTER = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:Chapter|Section|Appendix|Part|Annex)\b")
_PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "vs"}
)


class GuidelineSpec(BaseModel):
    """One guideline in ``sources.guidelines.items`` of a Stage 5 config."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    title: str
    citation: str
    url: str
    file: str
    licence: str = Field(min_length=1)
    year: int
    first_page: int | None = None
    last_page: int | None = None
    exclude_titles: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pages(self) -> GuidelineSpec:
        if self.first_page is not None and self.first_page < 1:
            raise ValueError("first_page is 1-based")
        if self.first_page and self.last_page and self.last_page < self.first_page:
            raise ValueError("last_page is before first_page")
        return self

    def excluded(self) -> set[str]:
        return {t.strip().lower() for t in (*DEFAULT_EXCLUDED_HEADINGS, *self.exclude_titles)}


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _header_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _title_like(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 3:
        return False
    if all(c.isupper() for c in letters):
        return True
    significant = [w for w in text.split() if w.lower().strip(":") not in _SMALL_WORDS and w[0].isalpha()]
    return bool(significant) and all(w[0].isupper() for w in significant)


def _is_heading(line: str) -> bool:
    text = line.strip()
    words = text.split()
    if not 1 <= len(words) <= 12 or len(text) > 80 or text.endswith((".", ",", ";")):
        return False
    if not any(c.isalpha() for c in text):
        return False
    if _KEYWORD.match(text):
        return True
    numbered = _NUMBER_PREFIX.match(text)
    if numbered:
        rest = numbered.group(2)
        return len(rest.split()) <= 6 or _title_like(rest)
    return _title_like(text)


def _pages(reader, spec: GuidelineSpec) -> list[tuple[str, list[str]]]:
    n = len(reader.pages)
    try:
        labels = list(reader.page_labels)
    except Exception:  # noqa: BLE001 - malformed label trees fall back to page numbers
        labels = []
    if len(labels) != n:
        labels = [str(i + 1) for i in range(n)]
    first = (spec.first_page or 1) - 1
    last = min(spec.last_page or n, n)
    pages = []
    for i in range(first, last):
        text = reader.pages[i].extract_text(extraction_mode="plain") or ""
        pages.append((labels[i], [line.strip() for line in text.splitlines()]))
    return pages


def _drop_headers(pages: list[tuple[str, list[str]]]) -> list[tuple[str, str]]:
    on_pages = Counter(key for _, lines in pages for key in {_header_key(x) for x in lines if x})
    threshold = max(HEADER_MIN_PAGES, HEADER_SHARE * len(pages))
    repeated = {key for key, n in on_pages.items() if n >= threshold}
    out = []
    for label, lines in pages:
        for line in lines:
            if line and (_header_key(line) in repeated or _PAGE_NUMBER.match(line)):
                continue
            out.append((label, line))
    return out


def _dehyphenate(lines: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for label, line in lines:
        if out and line and line[0].islower() and re.search(r"[A-Za-z]-$", out[-1][1]):
            previous_label, previous = out[-1]
            out[-1] = (previous_label, previous[:-1] + line)
        else:
            out.append((label, line))
    return out


def _sections(lines: list[tuple[str, str]], excluded: set[str]) -> list[Section]:
    sections: list[Section] = []
    title, paragraphs, locators, current, current_label = "", [], [], [], ""
    skipping = in_references = False

    def end_paragraph():
        nonlocal current
        text = normalise_text(" ".join(current))
        if text and not skipping:
            paragraphs.append(text)
            locators.append(current_label)
        current = []

    def end_section():
        end_paragraph()
        if paragraphs and not skipping:
            sections.append(Section(title, tuple(paragraphs), tuple(locators)))
        paragraphs.clear()
        locators.clear()

    for i, (label, line) in enumerate(lines):
        if not line:
            end_paragraph()
            continue
        heading = _is_heading(line) and i + 1 < len(lines)
        if heading and in_references and not _CHAPTER.match(line):
            heading = False  # a wrapped reference entry, not the next chapter
        if heading:
            end_section()
            name = normalise_text(line)
            bare = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", name).lower()
            skipping = bare in excluded or name.lower() in excluded
            in_references = bare in REFERENCE_HEADINGS
            title = name
            continue
        if not current:
            current_label = label
        current.append(line)
        if re.search(r"[.!?][\"')\]]*$", line) and i + 1 < len(lines) and lines[i + 1][1][:1].isupper():
            end_paragraph()
    end_section()
    return sections


def parse_guideline_pdf(path: str | Path, spec: GuidelineSpec) -> Document:
    """A guideline PDF as a Document (see the module rules).

    Raises:
        ValueError: No text could be extracted (e.g. a scanned PDF).
    """
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    lines = _dehyphenate(_drop_headers(_pages(reader, spec)))
    if not any(line for _, line in lines):
        raise ValueError(f"no text could be extracted from {path.name} ({spec.id}); is it a scanned PDF?")
    sections = _sections(lines, spec.excluded())
    if not sections:
        raise ValueError(f"{path.name} ({spec.id}) gave no sections of text")
    return Document(
        source="guideline",
        doc_id=spec.id,
        title=spec.title,
        sections=tuple(sections),
        url=spec.url,
        licence=spec.licence,
        citation=spec.citation,
        year=spec.year,
        meta={"file_sha256": _raw_sha256(path), "pages": len(reader.pages)},
    )


def parse_guideline_jats(path: str | Path, spec: GuidelineSpec) -> Document:
    """A guideline downloaded as JATS XML (e.g. from PubMed Central) as a Document."""
    path = Path(path)
    sections = parse_jats_sections(parse_xml(path.read_bytes()), exclude_titles=spec.excluded())
    if not sections:
        raise ValueError(f"{path.name} ({spec.id}) has no body text")
    return Document(
        source="guideline",
        doc_id=spec.id,
        title=spec.title,
        sections=tuple(sections),
        url=spec.url,
        licence=spec.licence,
        citation=spec.citation,
        year=spec.year,
        meta={"file_sha256": _raw_sha256(path)},
    )


def parse_guideline(spec: GuidelineSpec, base_dir: str | Path) -> Document:
    """Parse ``spec.file`` (relative to ``base_dir``) by its suffix: .pdf, .xml or .nxml."""
    path = Path(spec.file)
    path = path if path.is_absolute() else Path(base_dir) / path
    if not path.exists():
        raise FileNotFoundError(f"guideline {spec.id}: expected the file at {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_guideline_pdf(path, spec)
    if suffix in (".xml", ".nxml"):
        return parse_guideline_jats(path, spec)
    raise ValueError(f"guideline {spec.id}: unsupported file type {suffix!r} (use .pdf, .xml or .nxml)")
