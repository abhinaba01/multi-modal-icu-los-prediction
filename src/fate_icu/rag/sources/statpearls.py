"""StatPearls from the NCBI Bookshelf archive: resumable download and article parsing (Stage 5).

StatPearls (NCBI Bookshelf NBK430685) is distributed as one ``.tar.gz`` of JATS/BITS
``.nxml`` articles plus their images. The whole book is used (docs/decisions.md):

* :func:`download` fetches the archive into ``raw/statpearls/`` with resume (HTTP Range),
  checks the size and records the SHA-256 in a sidecar JSON;
* :func:`iter_archive_articles` streams the archive and yields the ``.nxml`` members
  without extracting anything to disk (unsafe or oversized members are an error);
* :func:`parse_statpearls_article` turns one article into a Document, dropping the
  "Continuing Education Activity", "Review Questions", "References" and "Disclosure"
  sections (boilerplate, quiz links and citations).

Licence: CC BY-NC-ND 4.0. Passages are quoted with attribution; the parsed copy stays in
``FATE_RAG_DIR`` and is never published.
"""

from __future__ import annotations

import hashlib
import os
import re
import tarfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from fate_icu.rag.corpus import Document
from fate_icu.rag.sources.jats import (
    jats_authors,
    jats_license,
    jats_title,
    jats_year,
    parse_jats_sections,
    parse_xml,
)

STATPEARLS_URL = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/3d/12/statpearls_NBK430685.tar.gz"
BOOK_ID = "NBK430685"
BOOK_URL = "https://www.ncbi.nlm.nih.gov/books/NBK430685/"
DEFAULT_LICENCE = "CC BY-NC-ND 4.0 (StatPearls Publishing)"
DEFAULT_EXCLUDED_SECTIONS = (
    "Continuing Education Activity",
    "Review Questions",
    "References",
    "Disclosure",
    "Disclosures",
)
MAX_PARSE_ERROR_SHARE = 0.01
_NBK = re.compile(r"NBK\d+")
_UNSAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")


# --- HTTP -----------------------------------------------------------------------------


@dataclass
class HttpResponse:
    status: int
    headers: Mapping[str, str]  # lower-case names
    stream: BinaryIO


class HttpTransport(Protocol):
    def head(self, url: str, timeout: float) -> tuple[int, Mapping[str, str]]: ...

    def get(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse: ...


class UrllibHttp:
    def head(self, url: str, timeout: float) -> tuple[int, Mapping[str, str]]:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "fate-icu"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status, {k.lower(): v for k, v in response.headers.items()}
        except urllib.error.HTTPError as error:
            return error.code, {k.lower(): v for k, v in (error.headers or {}).items()}

    def get(self, url: str, headers: Mapping[str, str], timeout: float) -> HttpResponse:
        request = urllib.request.Request(url, headers={"User-Agent": "fate-icu", **headers})
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            return HttpResponse(error.code, {k.lower(): v for k, v in (error.headers or {}).items()}, error)
        return HttpResponse(response.status, {k.lower(): v for k, v in response.headers.items()}, response)


class DownloadInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    url: str
    size: int
    sha256: str
    last_modified: str
    retrieved_on: str


def remote_size(url: str, transport: HttpTransport | None = None, timeout: float = 60.0) -> int:
    """The archive's size in bytes (HEAD Content-Length)."""
    status, headers = (transport or UrllibHttp()).head(url, timeout)
    if status != 200 or "content-length" not in headers:
        raise RuntimeError(f"HEAD {url}: HTTP {status}, no Content-Length")
    return int(headers["content-length"])


def download(
    url: str,
    dest_dir: str | Path,
    *,
    transport: HttpTransport | None = None,
    chunk_size: int = 1 << 20,
    expected_size: int | None = None,
    timeout: float = 120.0,
) -> DownloadInfo:
    """Download ``url`` into ``dest_dir`` with resume, a size check and a SHA-256 sidecar.

    A partial ``<name>.part`` is resumed with a Range request (restarted if the server
    ignores the range). An existing file with a matching sidecar is returned as is.

    Raises:
        RuntimeError: The remote size differs from ``expected_size``, the download is
            incomplete, or a file of that name exists without a matching sidecar.
    """
    transport = transport or UrllibHttp()
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = PurePosixPath(urlsplit(url).path).name
    final, part, sidecar = dest_dir / name, dest_dir / f"{name}.part", dest_dir / f"{name}.json"
    status, headers = transport.head(url, timeout)
    if status != 200 or "content-length" not in headers:
        raise RuntimeError(f"HEAD {url}: HTTP {status}, no Content-Length")
    size = int(headers["content-length"])
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"{name} is {size} bytes, not the expected {expected_size}")
    if final.exists():
        if sidecar.exists():
            info = DownloadInfo.model_validate_json(sidecar.read_text(encoding="utf-8"))
            if info.url == url and info.size == size == final.stat().st_size:
                return info
        raise RuntimeError(
            f"{final} exists but does not match the remote file; move it away to download again"
        )

    digest = hashlib.sha256()
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        offset = 0  # a different file: start again
    if offset:
        with part.open("rb") as existing:
            while block := existing.read(chunk_size):
                digest.update(block)
    if offset < size:
        response = transport.get(url, {"Range": f"bytes={offset}-"} if offset else {}, timeout)
        if response.status == 206 and offset:
            mode = "ab"
        elif response.status == 200:
            mode, digest = "wb", hashlib.sha256()  # the server ignored the range
        else:
            raise RuntimeError(f"GET {url}: HTTP {response.status}")
        try:
            with part.open(mode) as out:
                while block := response.stream.read(chunk_size):
                    out.write(block)
                    digest.update(block)
        finally:
            response.stream.close()
    got = part.stat().st_size
    if got != size:
        raise RuntimeError(f"{name}: downloaded {got:,} of {size:,} bytes; run the command again to resume")
    os.replace(part, final)
    info = DownloadInfo(
        path=str(final),
        url=url,
        size=size,
        sha256=digest.hexdigest(),
        last_modified=headers.get("last-modified", ""),
        retrieved_on=datetime.now(UTC).date().isoformat(),
    )
    sidecar.write_text(info.model_dump_json(indent=2), encoding="utf-8")
    return info


# --- the archive ----------------------------------------------------------------------


def _unsafe(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return path.is_absolute() or ".." in path.parts or bool(re.match(r"^[A-Za-z]:", name))


def iter_archive_articles(
    archive_path: str | Path, *, max_articles: int | None = None, max_member_bytes: int = 50_000_000
) -> Iterator[tuple[str, bytes]]:
    """``(member name, bytes)`` of every ``.nxml`` member, streamed in archive order.

    Raises:
        ValueError: An unsafe member name, or an ``.nxml`` member over ``max_member_bytes``.
    """
    yielded = 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            if _unsafe(member.name):
                raise ValueError(f"unsafe member name in {archive_path}: {member.name!r}")
            if not member.isfile() or not member.name.endswith(".nxml"):
                continue
            if member.size > max_member_bytes:
                raise ValueError(
                    f"{member.name} is {member.size:,} bytes, over the {max_member_bytes:,} limit"
                )
            handle = archive.extractfile(member)
            yield member.name, handle.read() if handle is not None else b""
            yielded += 1
            if max_articles is not None and yielded >= max_articles:
                return


def _article_nbk(root: ET.Element) -> str | None:
    meta = root.find(".//book-part-meta")
    scope = meta if meta is not None else root
    for element in scope.iter():
        for value in (element.text or "", *element.attrib.values()):
            for match in _NBK.findall(value):
                if match != BOOK_ID:
                    return match
    return None


def statpearls_citation(authors: list[str], title: str, year: int | None) -> str:
    """e.g. "Smith J, Lee K. Lactic Acidosis. In: StatPearls [Internet]. Treasure Island (FL):
    StatPearls Publishing; 2025." (three authors at most, then "et al")."""
    names = ", ".join(authors[:3]) + (", et al" if len(authors) > 3 else "")
    lead = f"{names}. " if names else ""
    when = f"; {year}" if year else ""
    source = "In: StatPearls [Internet]. Treasure Island (FL): StatPearls Publishing"
    return f"{lead}{title.rstrip('.')}. {source}{when}."


def parse_statpearls_article(
    name: str,
    data: bytes,
    *,
    licence_default: str = DEFAULT_LICENCE,
    exclude_titles: Collection[str] = DEFAULT_EXCLUDED_SECTIONS,
) -> tuple[Document | None, str | None]:
    """One article as a Document, or ``(None, reason)`` for "no_title" and "no_sections".

    Raises:
        xml.etree.ElementTree.ParseError: The member is not well-formed XML.
    """
    root = parse_xml(data)
    title = jats_title(root)
    if not title:
        return None, "no_title"
    sections = parse_jats_sections(root, exclude_titles=exclude_titles)
    if not sections:
        return None, "no_sections"
    nbk = _article_nbk(root)
    doc_id = nbk or _UNSAFE_ID.sub("-", PurePosixPath(name).stem).strip("-") or "article"
    year = jats_year(root)
    return (
        Document(
            source="statpearls",
            doc_id=doc_id,
            title=title,
            sections=tuple(sections),
            url=f"https://www.ncbi.nlm.nih.gov/books/{nbk}/" if nbk else BOOK_URL,
            licence=jats_license(root) or licence_default,
            citation=statpearls_citation(jats_authors(root), title, year),
            year=year,
            meta={"member": name, "nbk": nbk},
        ),
        None,
    )


def iter_statpearls_documents(
    archive_path: str | Path,
    *,
    max_articles: int | None = None,
    licence_default: str = DEFAULT_LICENCE,
    exclude_titles: Collection[str] = DEFAULT_EXCLUDED_SECTIONS,
) -> tuple[list[Document], Counter[str]]:
    """Every article of the archive, sorted by doc_id, and counts of skip reasons plus "kept".

    Raises:
        ValueError: Two articles share a doc_id, or more than 1% of members are not valid XML.
    """
    documents, counts, seen = [], Counter(), set()
    for name, data in iter_archive_articles(archive_path, max_articles=max_articles):
        try:
            document, reason = parse_statpearls_article(
                name, data, licence_default=licence_default, exclude_titles=exclude_titles
            )
        except ET.ParseError:
            counts["parse_error"] += 1
            continue
        counts[reason or "kept"] += 1
        if document is None:
            continue
        if document.doc_id in seen:
            raise ValueError(f"two StatPearls articles have doc_id {document.doc_id}")
        seen.add(document.doc_id)
        documents.append(document)
    total = sum(counts.values())
    if total and counts["parse_error"] > MAX_PARSE_ERROR_SHARE * total:
        raise ValueError(f"{counts['parse_error']} of {total} StatPearls members are not valid XML")
    return sorted(documents, key=lambda d: d.doc_id), counts
