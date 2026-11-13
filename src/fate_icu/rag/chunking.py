"""Passages from documents: sentence splitting and balanced chunks within a section (Stage 5).

Rules (docs/decisions.md):

* a passage never crosses a section boundary;
* a section's sentences are packed into the fewest chunks of at most ``max_tokens`` tokens,
  and among those the most even, so a long section gives chunks of about 150-300 tokens;
* tokens are counted with the MedCPT article encoder's tokenizer in the real build
  (:class:`HFTokenCounter`) and with a simple word counter in tests (:class:`WordCounter`);
* a sentence longer than ``max_tokens`` is split at word boundaries;
* sentences of one paragraph are joined by a space, paragraphs by a newline;
* chunk numbers run from 0 over the whole document, so a ``passage_id`` is stable.

The sentence splitter is a deterministic set of regular expressions for biomedical
English. A missed or extra split only moves a possible chunk boundary; it never loses text.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from fate_icu.rag.corpus import Document, Passage, Section, passage_id
from fate_icu.rag.text import normalise_text

MAX_CHUNKS_PER_DOCUMENT = 10_000  # the passage_id chunk field has four digits


class TokenCounter(Protocol):
    """Counts tokens of many texts at once."""

    name: str

    def count(self, texts: Sequence[str]) -> list[int]: ...


class WordCounter:
    """Words and punctuation marks as tokens (tests and quick checks)."""

    name = "words"
    _TOKEN = re.compile(r"\w+|[^\w\s]")

    def count(self, texts: Sequence[str]) -> list[int]:
        return [len(self._TOKEN.findall(text)) for text in texts]


class HFTokenCounter:
    """Tokens of a Hugging Face tokenizer, without special tokens.

    Args:
        tokenizer: e.g. the MedCPT article encoder's tokenizer.
        name: Recorded in the corpus report, e.g. ``hf:ncbi/MedCPT-Article-Encoder@d05a736d``.
        batch_size: Texts per tokenizer call (bounds memory).
    """

    def __init__(self, tokenizer: Any, name: str, batch_size: int = 1000):
        self.tokenizer = tokenizer
        self.name = name
        self.batch_size = batch_size

    def count(self, texts: Sequence[str]) -> list[int]:
        counts: list[int] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(batch, add_special_tokens=False, truncation=False, verbose=False)
            counts.extend(len(ids) for ids in encoded["input_ids"])
        return counts


# --- sentences -----------------------------------------------------------------------

# Lower-cased words that end with a period but do not end a sentence.
_ABBREVIATIONS = frozenset(
    {
        "e.g",
        "i.e",
        "al",
        "vs",
        "fig",
        "figs",
        "no",
        "nos",
        "approx",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "st",
        "cf",
        "ca",
        "inc",
        "ltd",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "eq",
        "eqs",
        "ref",
        "refs",
        "vol",
        "p",
        "pp",
    }
)
# A sentence end: . ! or ? (plus closing quotes or brackets), one space, then an uppercase
# letter, a digit, or an opening quote or bracket.
_BOUNDARY = re.compile(r"([.!?])([\"'”’)\]]*) (?=[\"'“‘(\[]?[A-Z0-9])")
_WORD_BEFORE = re.compile(r"([A-Za-z][A-Za-z.]*)$")


def _is_abbreviation(text: str, period_at: int) -> bool:
    match = _WORD_BEFORE.search(text, 0, period_at)
    if match is None:
        return False
    word = match.group(1)
    if len(word) == 1 and word.isupper():  # an initial: "J. Smith"
        return True
    return word.lower() in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Split normalised text into sentences; ``" ".join(result) == text``.

    Args:
        text: One paragraph, already normalised (single spaces).
    """
    sentences, start = [], 0
    for match in _BOUNDARY.finditer(text):
        if match.group(1) == "." and _is_abbreviation(text, match.start(1)):
            continue
        sentences.append(text[start : match.end(2)])
        start = match.end()
    if start < len(text):
        sentences.append(text[start:])
    return [s for s in sentences if s]


# --- balanced partition ---------------------------------------------------------------


def _greedy(lengths: Sequence[int], capacity: int) -> list[tuple[int, int]]:
    groups, start, total = [], 0, 0
    for i, n in enumerate(lengths):
        if i > start and total + n > capacity:
            groups.append((start, i))
            start, total = i, 0
        total += n
    groups.append((start, len(lengths)))
    return groups


def partition_balanced(lengths: Sequence[int], max_tokens: int) -> list[tuple[int, int]]:
    """Contiguous groups ``[start, end)`` with the fewest groups, then the smallest largest group.

    Greedy left-to-right packing is optimal for the number of groups at any capacity, so a
    binary search for the smallest capacity that still needs the minimal number of groups
    gives the most even partition; packing at that capacity is the deterministic tie-break.

    Raises:
        ValueError: An item is longer than ``max_tokens``.
    """
    if not lengths:
        return []
    if max(lengths) > max_tokens:
        raise ValueError(f"an item has {max(lengths)} tokens, more than max_tokens={max_tokens}")
    fewest = len(_greedy(lengths, max_tokens))
    low, high = max(lengths), max_tokens
    while low < high:
        middle = (low + high) // 2
        if len(_greedy(lengths, middle)) <= fewest:
            high = middle
        else:
            low = middle + 1
    return _greedy(lengths, low)


def split_long_unit(text: str, counter: TokenCounter, max_tokens: int) -> list[str]:
    """Split one over-long sentence at word boundaries into the fewest pieces that fit.

    A single word longer than ``max_tokens`` is cut into character pieces as a last resort.
    """
    words = text.split(" ")
    total = counter.count([text])[0]
    pieces_needed = max(1, math.ceil(total / max_tokens))
    while pieces_needed <= len(words):
        bounds = [round(i * len(words) / pieces_needed) for i in range(pieces_needed + 1)]
        pieces = [" ".join(words[a:b]) for a, b in zip(bounds, bounds[1:], strict=False) if b > a]
        if all(n <= max_tokens for n in counter.count(pieces)):
            return pieces
        pieces_needed += 1
    out: list[str] = []
    for word in words:  # every word alone: cut the ones that still do not fit
        if counter.count([word])[0] <= max_tokens:
            out.append(word)
            continue
        size = len(word)
        while (
            size > 1
            and max(counter.count([word[i : i + size] for i in range(0, len(word), size)])) > max_tokens
        ):
            size = max(1, size // 2)
        out.extend(word[i : i + size] for i in range(0, len(word), size))
    return out


# --- chunks ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkText:
    text: str
    n_tokens: int
    locator: str = ""


@dataclass(frozen=True)
class _Unit:
    text: str
    paragraph: int


def _section_units(section: Section) -> list[_Unit]:
    units = []
    for index, paragraph in enumerate(section.paragraphs):
        paragraph = normalise_text(paragraph)
        if paragraph:
            units.extend(_Unit(sentence, index) for sentence in split_sentences(paragraph))
    return units


def _join(units: Sequence[_Unit]) -> str:
    parts = [units[0].text]
    for previous, unit in zip(units, units[1:], strict=False):
        parts.append(("\n" if unit.paragraph != previous.paragraph else " ") + unit.text)
    return "".join(parts)


def _locator(section: Section, units: Sequence[_Unit]) -> str:
    if not section.locators:
        return ""
    first, last = section.locators[units[0].paragraph], section.locators[units[-1].paragraph]
    return f"p. {first}" if first == last else f"pp. {first}-{last}"


def _expand_long(units: list[_Unit], counts: list[int], counter: TokenCounter, max_tokens: int):
    """Units with every over-long sentence split, and their token counts."""
    if all(n <= max_tokens for n in counts):
        return units, counts
    out_units, out_counts = [], []
    for unit, n in zip(units, counts, strict=True):
        if n <= max_tokens:
            out_units.append(unit)
            out_counts.append(n)
            continue
        pieces = split_long_unit(unit.text, counter, max_tokens)
        out_units.extend(_Unit(piece, unit.paragraph) for piece in pieces)
        out_counts.extend(counter.count(pieces))
    return out_units, out_counts


def _fit(units: Sequence[_Unit], counter: TokenCounter, max_tokens: int) -> list[tuple[list[_Unit], int]]:
    """Split a group whose joined text is over the limit (counts need not add up)."""
    text = _join(units)
    n = counter.count([text])[0]
    if n <= max_tokens or len(units) == 1:
        return [(list(units), n)]
    middle = len(units) // 2
    return _fit(units[:middle], counter, max_tokens) + _fit(units[middle:], counter, max_tokens)


def _pack(units: list[_Unit], counts: list[int], counter: TokenCounter, max_tokens: int) -> list[list[_Unit]]:
    units, counts = _expand_long(units, counts, counter, max_tokens)
    return [units[a:b] for a, b in partition_balanced(counts, max_tokens)]


def _finish(
    section: Section,
    groups: list[list[_Unit]],
    joined_counts: list[int],
    counter: TokenCounter,
    max_tokens: int,
) -> list[ChunkText]:
    chunks = []
    for group, n in zip(groups, joined_counts, strict=True):
        parts = [(group, n)] if n <= max_tokens else _fit(group, counter, max_tokens)
        chunks.extend(ChunkText(_join(g), count, _locator(section, g)) for g, count in parts)
    return chunks


def chunk_section(section: Section, counter: TokenCounter, max_tokens: int) -> list[ChunkText]:
    """The chunks of one section, in order."""
    units = _section_units(section)
    if not units:
        return []
    groups = _pack(units, counter.count([u.text for u in units]), counter, max_tokens)
    return _finish(section, groups, counter.count([_join(g) for g in groups]), counter, max_tokens)


def _passages(doc: Document, sections_chunks: Iterable[tuple[Section, list[ChunkText]]], min_tokens: int):
    passages = []
    for section, chunks in sections_chunks:
        for chunk in chunks:
            if chunk.n_tokens < min_tokens:
                continue
            number = len(passages)
            if number >= MAX_CHUNKS_PER_DOCUMENT:
                raise ValueError(
                    f"{doc.source}:{doc.doc_id} yields more than {MAX_CHUNKS_PER_DOCUMENT} chunks"
                )
            passages.append(
                Passage(
                    passage_id=passage_id(doc.source, doc.doc_id, number),
                    source=doc.source,
                    doc_id=doc.doc_id,
                    chunk=number,
                    title=doc.title,
                    section=section.title,
                    locator=chunk.locator,
                    text=chunk.text,
                    n_tokens=chunk.n_tokens,
                    year=doc.year,
                    url=doc.url,
                    licence=doc.licence,
                    citation=doc.citation,
                )
            )
    return passages


def chunk_document(
    doc: Document, counter: TokenCounter, *, max_tokens: int = 300, min_tokens: int = 1
) -> list[Passage]:
    """Every passage of one document: chunks numbered from 0 across its sections.

    Chunks with fewer than ``min_tokens`` tokens are dropped before numbering.
    """
    return _passages(doc, ((s, chunk_section(s, counter, max_tokens)) for s in doc.sections), min_tokens)


def chunk_documents(
    docs: Iterable[Document],
    counter: TokenCounter,
    *,
    max_tokens: int = 300,
    min_tokens: int = 1,
    batch_size: int = 256,
) -> list[Passage]:
    """The same passages as :func:`chunk_document` on each document, with batched token counts.

    The tokenizer is the bottleneck on a corpus of ~200,000 documents, so sentences and joined
    chunks of ``batch_size`` documents are counted in one call each.
    """
    docs = list(docs)
    passages: list[Passage] = []
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        sections = [(i, s, _section_units(s)) for i, d in enumerate(batch) for s in d.sections]
        counts = counter.count([u.text for _, _, units in sections for u in units])
        packed, offset = [], 0
        for _, _, units in sections:
            own = counts[offset : offset + len(units)]
            offset += len(units)
            packed.append(_pack(units, own, counter, max_tokens) if units else [])
        joined = counter.count([_join(g) for groups in packed for g in groups])
        chunks_by_doc: list[list[tuple[Section, list[ChunkText]]]] = [[] for _ in batch]
        offset = 0
        for (i, section, _), groups in zip(sections, packed, strict=True):
            own = joined[offset : offset + len(groups)]
            offset += len(groups)
            chunks_by_doc[i].append((section, _finish(section, groups, own, counter, max_tokens)))
        for doc, chunks in zip(batch, chunks_by_doc, strict=True):
            passages.extend(_passages(doc, chunks, min_tokens))
    return passages
