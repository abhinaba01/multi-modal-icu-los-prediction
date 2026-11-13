"""PubMed abstracts: ICU-focused searches, cached records, and parsing to Documents (Stage 5).

1. ``configs/rag/pubmed_queries.yaml`` defines the searches: topic searches (ICU length of
   stay, AKI, sepsis, ARDS, delirium, vasopressors, mechanical ventilation) and one or more
   searches per driver key, all ANDed with the same filters (English, with an abstract,
   not animal-only, not paediatric-only, no letters, editorials, errata or retractions).
2. :func:`collect_pmids` runs each search through E-utilities, sliced by publication date
   so no slice reaches the 10,000-id cap, checks that the slices together return the
   search's full count, and saves the PMIDs (``esearch/<query>.json`` and ``pmids.json``).
   The saved PMID list pins the corpus.
3. :func:`fetch_records` downloads the records in batches of 200 (``efetch/batch-*.xml``,
   resumable, atomic writes) and lists the current batches in ``efetch/index.json``.
4. :func:`iter_documents` parses those batches into Documents, excluding records without a
   title or abstract, non-English records, retracted articles and non-research publication
   types. A PubMed passage's section title is empty, so the MedCPT article encoder sees
   (title, abstract), the format it was trained on.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import os
import re
import string
import threading
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Callable, Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fate_icu.rag.corpus import Document, Section
from fate_icu.rag.sources.eutils import ESEARCH_CAP, EutilsClient, EutilsError
from fate_icu.rag.text import normalise_text
from fate_icu.utils.env import REPO_ROOT

DEFAULT_QUERIES_PATH = REPO_ROOT / "configs" / "rag" / "pubmed_queries.yaml"
SLICE_CAP = ESEARCH_CAP - 1
EXCLUDED_PUBLICATION_TYPES = frozenset(
    {
        "Comment",
        "Editorial",
        "Letter",
        "News",
        "Newspaper Article",
        "Erratum",
        "Published Erratum",
        "Retraction of Publication",
        "Retracted Publication",
        "Expression of Concern",
    }
)
MAX_MISSING_SHARE = 0.01  # deleted or withdrawn records may be absent from efetch


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(payload, indent=1, sort_keys=True).encode("utf-8"))


# --- the searches ---------------------------------------------------------------------


class PubmedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    term: str
    purpose: Literal["topic", "driver"]
    keys: list[str] = Field(default_factory=list)  # driver keys "modality:key" this search serves
    note: str = ""


class PubmedQuerySet(BaseModel):
    """``configs/rag/pubmed_queries.yaml`` with every ``{block}`` placeholder expanded."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    filters: str
    min_year: int
    max_year: int
    blocks: dict[str, str] = Field(default_factory=dict)
    queries: list[PubmedQuery]

    @model_validator(mode="before")
    @classmethod
    def _expand_blocks(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        blocks = data.get("blocks") or {}
        expanded = []
        for query in data.get("queries", []):
            query = dict(query)
            fields = {name for _, name, _, _ in string.Formatter().parse(query["term"]) if name}
            unknown = fields - set(blocks)
            if unknown:
                raise ValueError(f"query {query.get('id')}: unknown blocks {sorted(unknown)}")
            query["term"] = query["term"].format_map(blocks)
            expanded.append(query)
        return {**data, "queries": expanded}

    @model_validator(mode="after")
    def _consistent(self) -> PubmedQuerySet:
        ids = [q.id for q in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("search ids must be unique")
        if self.min_year > self.max_year:
            raise ValueError("min_year is after max_year")
        return self

    @property
    def sha256(self) -> str:
        text = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def full_term(self, query: PubmedQuery) -> str:
        return f"({query.term}) AND ({self.filters})"


def load_pubmed_queries(path: str | Path = DEFAULT_QUERIES_PATH) -> PubmedQuerySet:
    with Path(path).open("r", encoding="utf-8") as handle:
        return PubmedQuerySet.model_validate(yaml.safe_load(handle))


# --- PMIDs ----------------------------------------------------------------------------


@dataclass(frozen=True)
class DateSlice:
    mindate: str
    maxdate: str
    count: int


def _fmt(day: date) -> str:
    return day.strftime("%Y/%m/%d")


def _parts(start: date, end: date, level: str) -> list[tuple[date, date, str]]:
    if level == "range":
        return [(date(y, 1, 1), date(y, 12, 31), "year") for y in range(start.year, end.year + 1)]
    if level == "year":
        return [
            (date(start.year, m, 1), date(start.year, m, calendar.monthrange(start.year, m)[1]), "month")
            for m in range(1, 13)
        ]
    if level == "month":
        last = calendar.monthrange(start.year, start.month)[1]
        return [
            (date(start.year, start.month, d), date(start.year, start.month, d), "day")
            for d in range(1, last + 1)
        ]
    raise AssertionError(level)


def _slice(client: EutilsClient, term: str, start: date, end: date, level: str, cap: int, count: int | None):
    count = client.esearch_count(term, mindate=_fmt(start), maxdate=_fmt(end)) if count is None else count
    if count == 0:
        return []
    if count <= cap:
        return [DateSlice(_fmt(start), _fmt(end), count)]
    if level == "day":
        raise EutilsError(f"more than {cap} records on {_fmt(start)} for one search: narrow the search")
    return [s for a, b, sub in _parts(start, end, level) for s in _slice(client, term, a, b, sub, cap, None)]


def slice_dates(
    client: EutilsClient, term: str, min_year: int, max_year: int, cap: int = SLICE_CAP
) -> tuple[list[DateSlice], int]:
    """Date slices of at most ``cap`` records each, and the search's total count.

    The whole range is tried first, then years, months and days, chronologically.
    """
    start, end = date(min_year, 1, 1), date(max_year, 12, 31)
    total = client.esearch_count(term, mindate=_fmt(start), maxdate=_fmt(end))
    return _slice(client, term, start, end, "range", cap, total), total


class PmidCollection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pmids: list[int]
    per_query: dict[str, int]
    retrieved_on: str
    query_set_sha256: str
    # PubMed's errors and warnings for a search (a phrase or field it did not find), by search id
    warnings: dict[str, dict[str, Any]] = Field(default_factory=dict)


def collect_pmids(
    client: EutilsClient,
    query_set: PubmedQuerySet,
    cache_dir: Path,
    *,
    max_records: int | None = None,
    log: Callable[[str], None] | None = None,
) -> PmidCollection:
    """Run every search (reusing a cached result only for an identical term) and save the PMIDs.

    Raises:
        EutilsError: The date slices of a search return fewer ids than its full count, or a
            search finds no records at all (a malformed search).
    """
    cache_dir = Path(cache_dir)
    per_query, union, dates, problems = {}, set(), [], {}
    for number, query in enumerate(query_set.queries, start=1):
        term = query_set.full_term(query)
        path = cache_dir / "esearch" / f"{query.id}.json"
        cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        reused = cached is not None and cached.get("term") == term
        if not reused:
            slices, total = slice_dates(client, term, query_set.min_year, query_set.max_year)
            ids = sorted(
                {i for s in slices for i in client.esearch_ids(term, mindate=s.mindate, maxdate=s.maxdate)}
            )
            if len(ids) != total:
                raise EutilsError(
                    f"search {query.id}: the date slices returned {len(ids)} of {total} records"
                )
            cached = {
                "term": term,
                "slices": [vars(s) for s in slices],
                "count": total,
                "ids": ids,
                "notes": client.esearch_notes(term),
                "retrieved_on": _today(),
            }
            _write_json(path, cached)
        notes = cached.get("notes", {})
        flagged = {key: notes[key] for key in ("errors", "warnings") if notes.get(key)}
        if not cached["ids"]:  # a malformed search matches nothing, and PubMed does not call it an error
            translation = notes.get("translation", "")[:400]
            raise EutilsError(
                f"search {query.id} found no records. PubMed read it as: {translation!r}. "
                f"PubMed's notes: {json.dumps(flagged)}"
            )
        per_query[query.id] = len(cached["ids"])
        union.update(cached["ids"])
        dates.append(cached["retrieved_on"])
        if flagged:
            problems[query.id] = flagged
        if log:
            log(
                f"[{number}/{len(query_set.queries)}] {query.id}: {len(cached['ids']):,} records "
                f"({'cached' if reused else 'searched'}); {len(union):,} unique so far"
                + (f"; PubMed notes: {json.dumps(flagged)}" if flagged else "")
            )
    pmids = sorted(union)
    if max_records is not None:
        pmids = pmids[-max_records:]  # the most recent PMIDs, deterministically
    collection = PmidCollection(
        pmids=pmids,
        per_query=per_query,
        retrieved_on=max(dates) if dates else _today(),
        query_set_sha256=query_set.sha256,
        warnings=problems,
    )
    _write_json(cache_dir / "pmids.json", collection.model_dump())
    return collection


# --- records --------------------------------------------------------------------------


class FetchReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batches: int  # batch files holding requested records (fetched now + reused)
    fetched: int  # batches downloaded by this call
    reused: int  # batch files from earlier calls that hold requested records
    pmids_requested: int
    pmids_found: int
    pmids_missing: int


def batch_name(pmids: Sequence[int]) -> str:
    """``batch-<first pmid>-<count>-<hash>.xml``: named by content, never by position in a list."""
    digest = hashlib.sha256(",".join(str(p) for p in pmids).encode("ascii")).hexdigest()[:8]
    return f"batch-{pmids[0]:09d}-{len(pmids):04d}-{digest}.xml"


def _batch_pmids(path: Path) -> set[int] | None:
    """PMIDs of the articles and book chapters in a batch file, or None if the file is unreadable."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None
    if root.tag != "PubmedArticleSet":
        return None
    found = root.findall("PubmedArticle/MedlineCitation/PMID") + root.findall(
        "PubmedBookArticle/BookDocument/PMID"
    )
    return {int(e.text) for e in found if e.text}


def fetch_records(
    client: EutilsClient,
    pmids: Sequence[int],
    cache_dir: Path,
    *,
    batch_size: int = 200,
    max_workers: int = 1,
    log: Callable[[str], None] | None = None,
) -> FetchReport:
    """Download the requested records into ``cache_dir/efetch/``, only those not already there.

    Every readable batch file in the folder counts, whichever version of the searches fetched
    it, so adding or removing a search downloads only the records that are new. A PMID that
    PubMed returned nothing for (a withdrawn or deleted record) is remembered in
    ``efetch/index.json`` and not requested again. The index lists the batch files that hold
    requested records, which is what :func:`iter_documents` reads.

    Raises:
        EutilsError: More than 1% of the requested PMIDs are missing from the records.
        RuntimeError: A requested PMID is in two batch files.
    """
    efetch = Path(cache_dir) / "efetch"
    efetch.mkdir(parents=True, exist_ok=True)
    ordered = sorted({int(p) for p in pmids})
    wanted = set(ordered)
    index_path = efetch / "index.json"
    previous = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
    unavailable = set(previous.get("unavailable", []))
    on_disk: dict[str, set[int]] = {}
    for batch_file in sorted(efetch.glob("batch-*.xml")):
        found = _batch_pmids(batch_file)
        if found is not None:
            on_disk[batch_file.name] = found
    covered = set().union(*on_disk.values()) if on_disk else set()
    to_fetch = [p for p in ordered if p not in covered and p not in unavailable]
    batches = [to_fetch[i : i + batch_size] for i in range(0, len(to_fetch), batch_size)]
    names = [batch_name(batch) for batch in batches]

    progress = {"done": 0}
    progress_lock = threading.Lock()

    def fetch(i: int) -> None:
        xml = client.efetch_xml(batches[i])
        _atomic_write(efetch / names[i], xml)
        with progress_lock:
            progress["done"] += 1
            if log and (progress["done"] % 25 == 0 or progress["done"] == len(batches)):
                log(f"fetched {progress['done']:,} / {len(batches):,} batches")

    if log:
        log(
            f"{len(to_fetch):,} of {len(ordered):,} records to fetch in {len(batches):,} batches "
            f"({len(ordered) - len(to_fetch):,} already on disk or unavailable)"
        )
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(fetch, range(len(batches))))  # re-raises the first failure
    else:
        for i in range(len(batches)):
            fetch(i)
    for name, batch in zip(names, batches, strict=True):
        got = _batch_pmids(efetch / name) or set()
        on_disk[name] = got
        unavailable |= set(batch) - got

    used = {name: ids & wanted for name, ids in on_disk.items() if ids & wanted}
    seen: set[int] = set()
    for name, ids in used.items():
        if seen & ids:
            raise RuntimeError(
                f"PMID {min(seen & ids)} is in two batch files of {efetch} (one is {name}): "
                "remove the older file"
            )
        seen |= ids
    earlier = previous.get("files", {})
    files = {}
    for name, ids in used.items():
        when = earlier.get(name, {}).get("retrieved_on") if name not in names else None
        if when is None:
            stamp = (efetch / name).stat().st_mtime
            when = _today() if name in names else datetime.fromtimestamp(stamp, UTC).date().isoformat()
        files[name] = {"pmids": len(ids), "retrieved_on": when}
    _write_json(index_path, {"files": files, "unavailable": sorted(unavailable)})
    missing = len(ordered) - len(seen)
    if ordered and missing > MAX_MISSING_SHARE * len(ordered):
        raise EutilsError(f"{missing} of {len(ordered)} requested PMIDs are missing from the fetched records")
    reused = sum(1 for name in used if name not in names)
    return FetchReport(
        batches=len(names) + reused,
        fetched=len(names),
        reused=reused,
        pmids_requested=len(ordered),
        pmids_found=len(seen),
        pmids_missing=missing,
    )


@dataclass(frozen=True)
class PubmedRecord:
    pmid: int
    title: str
    abstract: tuple[str, ...]
    languages: tuple[str, ...]
    publication_types: tuple[str, ...]
    year: int | None
    journal: str
    first_author: str
    n_authors: int
    doi: str
    mesh: tuple[str, ...]
    retracted: bool
    kind: str = "article"  # "book" for a PubmedBookArticle: a book chapter, e.g. a StatPearls chapter


def _text(element: ET.Element | None) -> str:
    return normalise_text("".join(element.itertext())) if element is not None else ""


_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


def _year(article: ET.Element, pubmed_data: ET.Element | None) -> int | None:
    pub_date = article.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year = pub_date.findtext("Year")
        if year and year.strip().isdigit():
            return int(year)
        match = _YEAR.search(pub_date.findtext("MedlineDate") or "")
        if match:
            return int(match.group(1))
    year = article.findtext("ArticleDate/Year")
    if year and year.strip().isdigit():
        return int(year)
    if pubmed_data is not None:
        for entry in pubmed_data.findall("History/PubMedPubDate"):
            if entry.get("PubStatus") == "pubmed" and (entry.findtext("Year") or "").strip().isdigit():
                return int(entry.findtext("Year"))
    return None


def _first_author(article: ET.Element) -> tuple[str, int]:
    authors = article.findall("AuthorList/Author")
    for author in authors[:1]:
        collective = _text(author.find("CollectiveName"))
        if collective:
            return collective, len(authors)
        last, initials = _text(author.find("LastName")), _text(author.find("Initials"))
        return " ".join(part for part in (last, initials) if part), len(authors)
    return "", 0


def _parse_article(element: ET.Element) -> PubmedRecord | None:
    citation = element.find("MedlineCitation")
    article = citation.find("Article") if citation is not None else None
    pmid_text = citation.findtext("PMID") if citation is not None else None
    if article is None or not pmid_text or not pmid_text.strip().isdigit():
        return None
    pubmed_data = element.find("PubmedData")
    abstract = []
    for item in article.findall("Abstract/AbstractText"):
        text = _text(item)
        if text:
            label = (item.get("Label") or "").strip()
            abstract.append(f"{label}: {text}" if label else text)
    publication_types = tuple(_text(p) for p in article.findall("PublicationTypeList/PublicationType"))
    retraction_notice = any(
        c.get("RefType") == "RetractionIn"
        for c in citation.findall("CommentsCorrectionsList/CommentsCorrections")
    )
    doi = ""
    if pubmed_data is not None:
        doi = next(
            (_text(a) for a in pubmed_data.findall("ArticleIdList/ArticleId") if a.get("IdType") == "doi"), ""
        )
    if not doi:
        doi = next((_text(e) for e in article.findall("ELocationID") if e.get("EIdType") == "doi"), "")
    journal = _text(article.find("Journal/ISOAbbreviation")) or _text(article.find("Journal/Title"))
    first_author, n_authors = _first_author(article)
    return PubmedRecord(
        pmid=int(pmid_text),
        title=_text(article.find("ArticleTitle")),
        abstract=tuple(abstract),
        languages=tuple(_text(lang) for lang in article.findall("Language")),
        publication_types=publication_types,
        year=_year(article, pubmed_data),
        journal=journal,
        first_author=first_author,
        n_authors=n_authors,
        doi=doi,
        mesh=tuple(_text(d) for d in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")),
        retracted="Retracted Publication" in publication_types or retraction_notice,
    )


def _parse_book_article(element: ET.Element) -> PubmedRecord | None:
    """A book chapter, kept only so it is counted and excluded (StatPearls comes from its own archive)."""
    document = element.find("BookDocument")
    pmid_text = document.findtext("PMID") if document is not None else None
    if not pmid_text or not pmid_text.strip().isdigit():
        return None
    return PubmedRecord(
        pmid=int(pmid_text),
        title=_text(document.find("ArticleTitle")),
        abstract=(),
        languages=(),
        publication_types=(),
        year=None,
        journal="",
        first_author="",
        n_authors=0,
        doi="",
        mesh=(),
        retracted=False,
        kind="book",
    )


def parse_pubmed_xml(xml: bytes) -> list[PubmedRecord]:
    """Every journal article and book chapter of a ``PubmedArticleSet`` (deletions are skipped)."""
    root = ET.fromstring(xml)
    if root.tag != "PubmedArticleSet":
        raise ValueError(f"not a PubmedArticleSet: {root.tag!r}")
    records = [_parse_article(e) for e in root.findall("PubmedArticle")]
    records += [_parse_book_article(e) for e in root.findall("PubmedBookArticle")]
    return [r for r in records if r is not None]


def exclusion_reason(record: PubmedRecord) -> str | None:
    """Why a record is left out of the corpus, or None to keep it."""
    if record.kind == "book":
        return "book_article"
    if not record.title:
        return "no_title"
    if not record.abstract:
        return "no_abstract"
    if "eng" not in record.languages:
        return "not_english"
    if record.retracted:
        return "retracted"
    for publication_type in record.publication_types:
        if publication_type in EXCLUDED_PUBLICATION_TYPES:
            return f"publication_type:{publication_type}"
    return None


def citation_text(record: PubmedRecord) -> str:
    """e.g. "Smith J, et al. Crit Care Med. 2019." (missing parts left out)."""
    parts = []
    if record.first_author:
        parts.append(record.first_author + (", et al." if record.n_authors > 1 else "."))
    if record.journal:
        parts.append(record.journal.rstrip(".") + ".")
    if record.year:
        parts.append(f"{record.year}.")
    return " ".join(parts)


def record_to_document(
    record: PubmedRecord, *, licence: str, retrieved_on: str | None = None
) -> Document | None:
    """The record as a Document, or None when :func:`exclusion_reason` excludes it."""
    if exclusion_reason(record) is not None:
        return None
    meta = {
        "journal": record.journal,
        "doi": record.doi,
        "mesh": list(record.mesh),
        "publication_types": list(record.publication_types),
    }
    if retrieved_on:
        meta["retrieved_on"] = retrieved_on
    return Document(
        source="pubmed",
        doc_id=str(record.pmid),
        title=record.title,
        sections=(Section("", record.abstract),),
        url=f"https://pubmed.ncbi.nlm.nih.gov/{record.pmid}/",
        licence=licence,
        citation=citation_text(record),
        year=record.year,
        meta=meta,
    )


def iter_documents(
    cache_dir: Path, pmids: Collection[int], *, licence: str
) -> tuple[list[Document], Counter[str]]:
    """Documents for ``pmids`` from the batches listed in ``efetch/index.json``.

    Returns:
        Documents sorted by PMID, and counts of exclusion reasons plus "kept" and "missing".

    Raises:
        ValueError: A PMID appears in two batch files.
    """
    efetch = Path(cache_dir) / "efetch"
    index_path = efetch / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} is missing: run fate_icu.rag.fetch --source pubmed first")
    wanted = {int(p) for p in pmids}
    files = json.loads(index_path.read_text(encoding="utf-8"))["files"]
    seen: set[int] = set()
    documents, counts = [], Counter()
    for name in sorted(files):
        for record in parse_pubmed_xml((efetch / name).read_bytes()):
            if record.pmid not in wanted:
                continue
            if record.pmid in seen:
                raise ValueError(f"PMID {record.pmid} appears in two batch files")
            seen.add(record.pmid)
            reason = exclusion_reason(record)
            counts[reason or "kept"] += 1
            if reason is None:
                documents.append(
                    record_to_document(record, licence=licence, retrieved_on=files[name]["retrieved_on"])
                )
    counts["missing"] = len(wanted - seen)
    documents.sort(key=lambda d: int(d.doc_id))
    return documents, counts
