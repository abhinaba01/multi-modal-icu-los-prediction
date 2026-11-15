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
  are not detected; their text stays in the previous section, so nothing is lost. Never a
  heading: a line wholly in parentheses (a grade, "(Not Graded)"), a line with a numeric
  citation ("(47)", "[173, 180]"), or a keyword used as a label ("Recommendation: We
  suggest ..."), so a recommendation's first line stays in its text. A line
  only counts as a heading in context: the line before it must end a sentence (or be a
  heading itself, unless the heading starts with a keyword) and the line after it must not
  continue in lower case, so a Title-Case phrase wrapped inside a sentence stays text;
* a heading starts a section; its lines form paragraphs, and every paragraph carries the
  page label where it starts (the passage locator, e.g. "p. 1183");
* excluded sections (references, acknowledgements, funding, conflicts of interest, ...)
  are skipped. Inside a reference list only a Chapter, Section, Appendix, Part or Annex
  heading ends the skip, because wrapped reference entries ("1. Singer M, Deutschman CS")
  look like numbered headings;
* the spec can name where the content starts and ends (``start_at``, ``end_at``: regular
  expressions matched at the start of a whole line, case-insensitive), which drops the author
  affiliations, tables of contents, acknowledgements, references and biographies, and the
  boxes to remove (``drop_blocks``: a first line, a last line, and a limit on the length).
  A marker that is not found is an error, never a silent fallback;
* pypdf's warnings (it complains once per font per page) are counted, not printed, and the
  count and the number of suspicious characters (replacement or private-use code points)
  go into the document's ``meta``.
"""

from __future__ import annotations

import hashlib
import logging
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
    "Sponsorship",
    "Disclaimer",
)
REFERENCE_HEADINGS = frozenset({"references", "bibliography"})
HEADER_SHARE, HEADER_MIN_PAGES = 0.3, 3

_NUMBER_PREFIX = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(\S.*)$")
_KEYWORD = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:Recommendation|Chapter|Section|Appendix|Part|Annex)\b")
_CHAPTER = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?(?:Chapter|Section|Appendix|Part|Annex)\b")
# "Recommendation: We suggest ..." is a label opening a sentence, not a heading
_LABEL = re.compile(r"^(?:Recommendation|Chapter|Section|Appendix|Part|Annex)\s*:")
# a numeric citation, "(47)" or "[173, 180-181]": headings do not cite
_CITATION = re.compile(r"[(\[]\d+(?:\s*[,–-]\s*\d+)*[)\]]")
_PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)
_SMALL_WORDS = frozenset(
    {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with", "vs"}
)


class Block(BaseModel):
    """A boxed note to drop: from the line matching ``start`` to the line matching ``end``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    start: str
    end: str
    max_lines: int = Field(default=40, ge=2)


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
    start_at: str | None = None  # the content begins at the first line matching this regex
    end_at: str | None = None  # the content ends before the first line (after the start) matching it
    drop_blocks: list[Block] = Field(default_factory=list)

    @model_validator(mode="after")
    def _patterns(self) -> GuidelineSpec:
        patterns = [self.start_at, self.end_at, *(p for b in self.drop_blocks for p in (b.start, b.end))]
        for pattern in filter(None, patterns):
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"guideline {self.id}: bad pattern {pattern!r}: {error}") from error
        return self

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


def _keyword_heading(text: str) -> bool:
    return bool(_KEYWORD.match(text)) and not _LABEL.match(text)


def _is_heading(line: str) -> bool:
    text = line.strip()
    words = text.split()
    if not 1 <= len(words) <= 12 or len(text) > 80 or text.endswith((".", ",", ";")):
        return False
    if not any(c.isalpha() for c in text):
        return False
    if (text.startswith("(") and text.endswith(")")) or _CITATION.search(text):
        return False  # a grade such as "(Not Graded)", or body text that cites
    if _keyword_heading(text):
        return True
    numbered = _NUMBER_PREFIX.match(text)
    if numbered:
        rest = numbered.group(2)
        return len(rest.split()) <= 6 or _title_like(rest)
    return _title_like(text)


_SENTENCE_END = re.compile(r"[.!?][\"')\]\u201d\u2019]*$")


def _neighbour(lines: list[tuple[str, str]], start: int, step: int) -> str | None:
    """The nearest non-empty line in a direction (looking at most four lines away)."""
    for i in range(start, start + 4 * step, step):
        if not 0 <= i < len(lines):
            return None
        if lines[i][1].strip():
            return lines[i][1].strip()
    return None


def _heading_in_context(lines: list[tuple[str, str]], i: int) -> bool:
    """Is line ``i`` a heading, given the lines around it? (See the module rules.)"""
    line = lines[i][1].strip()
    if not _is_heading(line):
        return False
    following = _neighbour(lines, i + 1, 1)
    if following is None or following[:1].islower():
        return False  # the last line, or a wrapped line that continues in lower case
    last_word = line.split()[-1].lower().strip(",;:")
    if line.endswith("-") or last_word in _SMALL_WORDS:
        return False  # a phrase cut at the end of a line
    if _keyword_heading(line):
        return True
    previous = _neighbour(lines, i - 1, -1)
    return previous is None or bool(_SENTENCE_END.search(previous)) or _is_heading(previous)


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


def _find(lines: list[tuple[str, str]], pattern: str, begin: int = 0) -> int | None:
    expression = re.compile(pattern, re.IGNORECASE)
    for i in range(begin, len(lines)):
        if expression.match(lines[i][1].strip()):
            return i
    return None


def _cut(lines: list[tuple[str, str]], spec: GuidelineSpec) -> list[tuple[str, str]]:
    """Keep the lines from ``start_at`` up to (not including) ``end_at``."""
    if spec.start_at:
        start = _find(lines, spec.start_at)
        if start is None:
            raise ValueError(
                f"guideline {spec.id}: start_at {spec.start_at!r} matches no line of {spec.file}"
            )
        lines = lines[start:]
    if spec.end_at:
        end = _find(lines, spec.end_at, begin=1)
        if end is None:
            raise ValueError(f"guideline {spec.id}: end_at {spec.end_at!r} matches no line of {spec.file}")
        lines = lines[:end]
    return lines


def _drop_blocks(lines: list[tuple[str, str]], blocks: list[Block], spec_id: str) -> list[tuple[str, str]]:
    """Remove each boxed note (first through last line); a box that never closes is an error."""
    out: list[tuple[str, str]] = []
    i = 0
    starts = [re.compile(b.start, re.IGNORECASE) for b in blocks]
    ends = [re.compile(b.end, re.IGNORECASE) for b in blocks]
    while i < len(lines):
        matched = next(
            (k for k, expression in enumerate(starts) if expression.match(lines[i][1].strip())), None
        )
        if matched is None:
            out.append(lines[i])
            i += 1
            continue
        stop = next(
            (
                j
                for j in range(i, min(len(lines), i + blocks[matched].max_lines))
                if ends[matched].match(lines[j][1].strip())
            ),
            None,
        )
        if stop is None:
            raise ValueError(
                f"guideline {spec_id}: the block starting at {lines[i][1][:40]!r} (page {lines[i][0]}) "
                f"does not end within {blocks[matched].max_lines} lines"
            )
        i = stop + 1
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
        heading = _heading_in_context(lines, i)
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


class _WarningCounter(logging.Handler):
    """Counts pypdf's warnings instead of printing them (it repeats the same one per page)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.total = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.total += 1


def parse_guideline_pdf(path: str | Path, spec: GuidelineSpec) -> Document:
    """A guideline PDF as a Document (see the module rules).

    Raises:
        ValueError: No text could be extracted (e.g. a scanned PDF).
    """
    from pypdf import PdfReader

    path = Path(path)
    counter = _WarningCounter()
    pypdf_logger = logging.getLogger("pypdf")
    saved = (pypdf_logger.handlers[:], pypdf_logger.propagate, pypdf_logger.level)
    pypdf_logger.handlers, pypdf_logger.propagate, pypdf_logger.level = [counter], False, logging.WARNING
    try:
        reader = PdfReader(str(path))
        lines = _drop_headers(_pages(reader, spec))
    finally:
        pypdf_logger.handlers, pypdf_logger.propagate, pypdf_logger.level = saved
    lines = _dehyphenate(_drop_blocks(_cut(lines, spec), spec.drop_blocks, spec.id))
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
        meta={
            "file_sha256": _raw_sha256(path),
            "pages": len(reader.pages),
            "pdf_warnings": counter.total,
            "suspicious_characters": sum(
                1 for _, line in lines for c in line if c == "\ufffd" or "\ue000" <= c <= "\uf8ff"
            ),
        },
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
