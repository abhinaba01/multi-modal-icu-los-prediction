"""Documents and passages: what the retrieval index holds (Stage 5).

Source parsers (``fate_icu.rag.sources``) turn each downloaded record into a
:class:`Document`: a title, sections of paragraphs, a URL, a citation and a licence. The
chunker (``fate_icu.rag.chunking``) cuts every section into :class:`Passage` objects; a
passage never crosses a section boundary.

The corpus is one Parquet file of passages sorted by ``passage_id``. A passage's row number
is its index in BM25 and FAISS, so a tie broken by row number is a tie broken by
``passage_id``. The corpus hash covers the passages' content (canonical JSON, one line per
passage, in order), not the Parquet bytes, so it is the same on every machine and with any
pyarrow version.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, model_validator

CORPUS_SCHEMA_VERSION = 1
PASSAGES_FILE = "passages.parquet"
CORPUS_INFO_FILE = "corpus.json"

Source = Literal["pubmed", "statpearls", "guideline"]
SOURCES: tuple[str, ...] = ("pubmed", "statpearls", "guideline")
_DOC_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def passage_id(source: str, doc_id: str, chunk: int) -> str:
    """``<source>:<doc_id>:<chunk>`` with a 4-digit chunk number, e.g. ``pubmed:31234567:0001``."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    if not _DOC_ID.match(doc_id):
        raise ValueError(f"doc_id must match {_DOC_ID.pattern}: {doc_id!r}")
    if not 0 <= chunk <= 9999:
        raise ValueError(f"chunk number out of range: {chunk}")
    return f"{source}:{doc_id}:{chunk:04d}"


@dataclass(frozen=True)
class Section:
    """A titled run of paragraphs.

    ``locators`` is either empty or one page label per paragraph (guideline PDFs), so a
    passage can be cited as "p. 12" or "pp. 12-13".
    """

    title: str
    paragraphs: tuple[str, ...]
    locators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.locators and len(self.locators) != len(self.paragraphs):
            raise ValueError("locators must be empty or give one label per paragraph")


@dataclass(frozen=True)
class Document:
    """One source record: a PubMed abstract, a StatPearls article or a guideline."""

    source: Source
    doc_id: str
    title: str
    sections: tuple[Section, ...]
    url: str
    licence: str
    citation: str
    year: int | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)  # reported, never indexed

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown source {self.source!r}")
        if not _DOC_ID.match(self.doc_id):
            raise ValueError(f"doc_id must match {_DOC_ID.pattern}: {self.doc_id!r}")
        if not self.licence:
            raise ValueError(f"{self.source}:{self.doc_id} has no licence")


class Passage(BaseModel):
    """One retrievable chunk of a document section."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passage_id: str
    source: Source
    doc_id: str
    chunk: int
    title: str
    section: str = ""
    locator: str = ""
    text: str
    n_tokens: int
    year: int | None = None
    url: str
    licence: str
    citation: str

    @model_validator(mode="after")
    def _consistent(self) -> Passage:
        expected = passage_id(self.source, self.doc_id, self.chunk)
        if self.passage_id != expected:
            raise ValueError(f"passage_id {self.passage_id!r} should be {expected!r}")
        if not self.text.strip():
            raise ValueError(f"{self.passage_id} has no text")
        if self.n_tokens < 1:
            raise ValueError(f"{self.passage_id} has n_tokens < 1")
        return self

    def search_text(self) -> str:
        """What BM25 indexes: the title, the section title and the text."""
        return "\n".join(part for part in (self.title, self.section, self.text) if part)

    def encoder_pair(self) -> tuple[str, str]:
        """The MedCPT article-encoder input ``(title, text)``; a section title prefixes the text."""
        return self.title, f"{self.section}. {self.text}" if self.section else self.text

    def cross_text(self) -> str:
        """The MedCPT cross-encoder's article text, "Title. Text" as in its model card."""
        title, text = self.encoder_pair()
        title = title.rstrip()
        return f"{title} {text}" if title.endswith((".", "?", "!")) else f"{title}. {text}"


_ARROW_SCHEMA = pa.schema(
    [
        ("passage_id", pa.string()),
        ("source", pa.string()),
        ("doc_id", pa.string()),
        ("chunk", pa.int32()),
        ("title", pa.string()),
        ("section", pa.string()),
        ("locator", pa.string()),
        ("text", pa.string()),
        ("n_tokens", pa.int32()),
        ("year", pa.int32()),
        ("url", pa.string()),
        ("licence", pa.string()),
        ("citation", pa.string()),
    ]
)
assert _ARROW_SCHEMA.names == list(Passage.model_fields), "the Parquet schema must follow Passage"


def canonical_line(passage: Passage) -> str:
    """The passage as canonical JSON (sorted keys, no spaces): the unit of the corpus hash."""
    return json.dumps(passage.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def corpus_sha256(passages: Iterable[Passage]) -> str:
    """SHA-256 over the canonical lines of ``passages`` in the given order."""
    digest = hashlib.sha256(f"fate-icu-corpus-v{CORPUS_SCHEMA_VERSION}\n".encode())
    for passage in passages:
        digest.update(canonical_line(passage).encode("utf-8") + b"\n")
    return digest.hexdigest()


class CorpusInfo(BaseModel):
    """``corpus.json``: the content hash and aggregate counts of a written corpus."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int
    sha256: str
    n_passages: int
    n_documents: int
    passages_by_source: dict[str, int]
    documents_by_source: dict[str, int]
    tokens: dict[str, float]  # min, median, p95, max of n_tokens


def _token_summary(counts: Sequence[int]) -> dict[str, float]:
    if not counts:
        return {}
    ordered = sorted(counts)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "min": float(ordered[0]),
        "median": float(statistics.median(ordered)),
        "p95": float(p95),
        "max": float(ordered[-1]),
    }


def corpus_info(passages: Sequence[Passage]) -> CorpusInfo:
    """Hash and counts for passages already sorted by ``passage_id``."""
    documents = {(p.source, p.doc_id) for p in passages}
    return CorpusInfo(
        schema_version=CORPUS_SCHEMA_VERSION,
        sha256=corpus_sha256(passages),
        n_passages=len(passages),
        n_documents=len(documents),
        passages_by_source=dict(sorted(Counter(p.source for p in passages).items())),
        documents_by_source=dict(sorted(Counter(source for source, _ in documents).items())),
        tokens=_token_summary([p.n_tokens for p in passages]),
    )


def sort_passages(passages: Iterable[Passage]) -> list[Passage]:
    """Passages sorted by ``passage_id``; duplicate ids are an error."""
    ordered = sorted(passages, key=lambda p: p.passage_id)
    duplicates = [
        a.passage_id for a, b in zip(ordered, ordered[1:], strict=False) if a.passage_id == b.passage_id
    ]
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate passage ids, e.g. {duplicates[0]}")
    return ordered


def write_corpus(passages: Iterable[Passage], out_dir: str | Path) -> CorpusInfo:
    """Sort, hash and write ``passages.parquet`` and ``corpus.json`` to ``out_dir``.

    Returns:
        The written :class:`CorpusInfo`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sort_passages(passages)
    table = pa.Table.from_pylist([p.model_dump() for p in ordered], schema=_ARROW_SCHEMA)
    tmp = out_dir / (PASSAGES_FILE + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out_dir / PASSAGES_FILE)
    info = corpus_info(ordered)
    (out_dir / CORPUS_INFO_FILE).write_text(info.model_dump_json(indent=2), encoding="utf-8")
    return info


def _passages_path(path: str | Path) -> Path:
    path = Path(path)
    return path / PASSAGES_FILE if path.is_dir() else path


def read_corpus_table(path: str | Path, columns: Sequence[str] | None = None) -> pa.Table:
    """The corpus as an Arrow table (``path`` is the corpus folder or its Parquet file)."""
    return pq.read_table(_passages_path(path), columns=list(columns) if columns else None)


def passages_from_table(table: pa.Table) -> list[Passage]:
    """Validated passages from a full corpus table, in row order."""
    return [Passage(**row) for row in table.to_pylist()]


def load_corpus(path: str | Path) -> list[Passage]:
    """Every passage of a corpus, in row order (= ``passage_id`` order)."""
    return passages_from_table(read_corpus_table(path))


def load_corpus_info(corpus_dir: str | Path) -> CorpusInfo:
    """``corpus.json`` of a written corpus."""
    return CorpusInfo.model_validate_json((Path(corpus_dir) / CORPUS_INFO_FILE).read_text(encoding="utf-8"))


def verify_corpus(corpus_dir: str | Path) -> CorpusInfo:
    """Recompute the content hash from the Parquet file and compare it with ``corpus.json``.

    Raises:
        RuntimeError: The passages are out of order, or their hash differs from the record.
    """
    info = load_corpus_info(corpus_dir)
    passages = load_corpus(corpus_dir)
    if [p.passage_id for p in passages] != sorted(p.passage_id for p in passages):
        raise RuntimeError(f"{corpus_dir}: passages are not sorted by passage_id")
    actual = corpus_sha256(passages)
    if actual != info.sha256:
        raise RuntimeError(
            f"{corpus_dir}: corpus hash {actual[:12]} differs from corpus.json {info.sha256[:12]}"
        )
    return info
