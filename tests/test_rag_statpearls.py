"""Stage 5 StatPearls: JATS/BITS parsing, the archive stream and the resumable download (offline)."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET

import pytest

from fate_icu.rag.sources.jats import (
    jats_authors,
    jats_license,
    jats_title,
    jats_year,
    parse_jats_sections,
    parse_xml,
)
from fate_icu.rag.sources.statpearls import (
    BOOK_URL,
    DEFAULT_EXCLUDED_SECTIONS,
    DEFAULT_LICENCE,
    PART_URL,
    HttpResponse,
    article_url,
    download,
    iter_archive_articles,
    iter_statpearls_documents,
    parse_statpearls_article,
    remote_size,
)
from fate_icu.utils.env import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "rag" / "statpearls"
NAMES = ["article-90001.nxml", "article-90002.nxml", "article-90003.nxml"]


def _root(name):
    return parse_xml((FIXTURES / name).read_bytes())


def test_sections_paragraphs_and_exclusions():
    sections = parse_jats_sections(_root("article-90001.nxml"), exclude_titles=DEFAULT_EXCLUDED_SECTIONS)
    assert [s.title for s in sections] == [
        "Introduction",
        "Etiology",
        "Treatment / Management",
        "Treatment / Management: Fluids and Vasopressors",
    ]
    assert sections[0].paragraphs == (
        "Lactic acidosis is a raised blood lactate with a low blood pH. "
        "In critically ill adults it most often "
        "reflects inadequate oxygen delivery to tissues.",
        "Its severity is usually graded by the lactate concentration in mmol/L.",
    )
    assert sections[1].paragraphs == (
        "Causes are grouped as follows:",
        "Type A: tissue hypoperfusion, as in septic or cardiogenic shock.",
        "Type B: impaired lactate clearance, as in liver failure or with some drugs.",
    )
    assert sections[3].paragraphs == (
        "Fluids are given first; vasopressors are added when hypotension persists.",
    )
    text = " ".join(p for s in sections for p in s.paragraphs)
    for absent in (
        "Boilerplate",
        "Objective one",
        "Quiz",
        "caption",
        "cell text",
        "Synthetic reference",
        "[1]",
    ):
        assert absent not in text


def test_lists_inside_paragraphs_boxes_and_namespaces():
    sections = parse_jats_sections(_root("article-90002.nxml"), exclude_titles=["references"])
    assert [s.title for s in sections] == ["", "Evaluation", "Prognosis"]
    assert sections[0].paragraphs == (
        "Acute kidney injury is a sudden fall in kidney function over hours to days.",
    )
    assert sections[1].paragraphs == (
        "Staging uses serum creatinine and urine output:",
        "Stage 1: creatinine 1.5 to 1.9 times baseline.",
        "Stage 3: creatinine three times baseline, or renal replacement therapy.",
        "Oliguria lasting more than 12 hours suggests a severe injury.",
    )
    assert all("Key point" not in p for s in sections for p in s.paragraphs)


def test_list_items_without_paragraphs_and_no_body():
    root = parse_xml(
        b"<article><body><sec><title>T</title><list><list-item>Bare item text.</list-item>"
        b"</list></sec></body></article>"
    )
    assert parse_jats_sections(root)[0].paragraphs == ("Bare item text.",)
    assert parse_jats_sections(parse_xml(b"<article><front/></article>")) == []


def test_metadata():
    first, second, third = (_root(n) for n in NAMES)
    assert [jats_title(r) for r in (first, second, third)] == [
        "Lactic Acidosis",
        "Acute Kidney Injury",
        "Septic Shock",
    ]
    assert [jats_year(r) for r in (first, second, third)] == [2025, 2024, 2023]
    assert [jats_license(r) for r in (first, second, third)] == ["CC BY-NC-ND 4.0", None, "CC BY-NC-ND 4.0"]
    assert jats_authors(first) == ["Varga EA", "Osei K"]  # the book editor is not an article author
    assert jats_authors(second) == ["Achebe N", "Berg LE", "Costa R", "Dubois C"]


def test_articles_become_documents():
    docs = {}
    for name in NAMES:
        doc, reason = parse_statpearls_article(name, (FIXTURES / name).read_bytes())
        assert reason is None
        docs[doc.doc_id] = doc
    assert sorted(docs) == ["NBK900001", "NBK900003", "article-90002"]
    lactic = docs["NBK900001"]
    assert (
        lactic.url == "https://www.ncbi.nlm.nih.gov/books/NBK900001/" and lactic.licence == "CC BY-NC-ND 4.0"
    )
    assert lactic.citation == (
        "Varga EA, Osei K. Lactic Acidosis. In: StatPearls [Internet]. "
        "Treasure Island (FL): StatPearls Publishing; 2025."
    )
    aki = docs["article-90002"]  # no article NBK id: the book's own id never counts
    assert aki.url == f"{PART_URL}article-90002/" and aki.licence == DEFAULT_LICENCE
    assert article_url(None, "some-other-name") == BOOK_URL  # not the real archive's naming
    assert article_url("NBK1", "article-1") == "https://www.ncbi.nlm.nih.gov/books/NBK1/"
    assert aki.citation.startswith("Achebe N, Berg LE, Costa R, et al. Acute Kidney Injury. In: StatPearls")
    septic = docs["NBK900003"]  # its team-outcomes and disclosure sections are boilerplate
    assert [s.title for s in septic.sections] == ["Introduction", "Treatment / Management"]
    assert "Enhancing Healthcare Team Outcomes" in DEFAULT_EXCLUDED_SECTIONS
    assert all("boilerplate" not in p for s in septic.sections for p in s.paragraphs)


def test_skip_reasons_and_parse_errors():
    no_title = (
        b"<book-part-wrapper><book-part><body><sec><title>A</title><p>Text.</p></sec></body>"
        b"</book-part></book-part-wrapper>"
    )
    assert parse_statpearls_article("x.nxml", no_title) == (None, "no_title")
    no_body = (
        b"<book-part-wrapper><book-part><book-part-meta><title-group><title>T</title></title-group>"
        b"</book-part-meta></book-part></book-part-wrapper>"
    )
    assert parse_statpearls_article("x.nxml", no_body) == (None, "no_sections")
    with pytest.raises(ET.ParseError):
        parse_statpearls_article("x.nxml", b"<not closed")


def _archive(path, members):
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def _fixture_members():
    return [(f"statpearls_NBK430685/{n}", (FIXTURES / n).read_bytes()) for n in NAMES]


def test_archive_stream_yields_nxml_members_only(tmp_path):
    archive = _archive(
        tmp_path / "sp.tar.gz", [*_fixture_members(), ("statpearls_NBK430685/fig1.jpg", b"\xff\xd8")]
    )
    names = [name for name, _ in iter_archive_articles(archive)]
    assert names == [f"statpearls_NBK430685/{n}" for n in NAMES]
    assert len(list(iter_archive_articles(archive, max_articles=2))) == 2
    unsafe = _archive(tmp_path / "unsafe.tar.gz", [("../evil.nxml", b"<a/>")])
    with pytest.raises(ValueError, match="unsafe"):
        list(iter_archive_articles(unsafe))
    with pytest.raises(ValueError, match="limit"):
        list(iter_archive_articles(archive, max_member_bytes=10))


def test_iter_statpearls_documents(tmp_path):
    archive = _archive(tmp_path / "sp.tar.gz", _fixture_members())
    docs, counts = iter_statpearls_documents(archive)
    assert [d.doc_id for d in docs] == ["NBK900001", "NBK900003", "article-90002"] and counts == {"kept": 3}
    twice = _archive(
        tmp_path / "twice.tar.gz",
        [*_fixture_members(), ("copy/article-90001.nxml", _fixture_members()[0][1])],
    )
    with pytest.raises(ValueError, match="doc_id NBK900001"):
        iter_statpearls_documents(twice)
    broken = _archive(tmp_path / "broken.tar.gz", [*_fixture_members(), ("bad.nxml", b"<unclosed")])
    with pytest.raises(ValueError, match="not valid XML"):
        iter_statpearls_documents(broken)


class FakeHttp:
    def __init__(self, content, *, honour_range=True, truncate_to=None):
        self.content, self.honour_range, self.truncate_to = content, honour_range, truncate_to
        self.gets = []

    def head(self, url, timeout):
        return 200, {
            "content-length": str(len(self.content)),
            "last-modified": "Sun, 08 Nov 2026 08:14:18 GMT",
        }

    def get(self, url, headers, timeout):
        self.gets.append(dict(headers))
        start, status = 0, 200
        if "Range" in headers and self.honour_range:
            start, status = int(headers["Range"].split("=")[1].rstrip("-")), 206
        body = self.content[start:]
        if self.truncate_to is not None:
            body = body[: self.truncate_to]
        return HttpResponse(status, {}, io.BytesIO(body))


URL = "https://example.org/pub/statpearls_test.tar.gz"
CONTENT = bytes(range(256)) * 40


def test_download_fresh_then_reused(tmp_path):
    http = FakeHttp(CONTENT)
    info = download(URL, tmp_path, transport=http, chunk_size=1000)
    path = tmp_path / "statpearls_test.tar.gz"
    assert path.read_bytes() == CONTENT and not (tmp_path / "statpearls_test.tar.gz.part").exists()
    assert info.sha256 == hashlib.sha256(CONTENT).hexdigest() and info.size == len(CONTENT)
    assert json.loads((tmp_path / "statpearls_test.tar.gz.json").read_text())["sha256"] == info.sha256
    assert download(URL, tmp_path, transport=http) == info and len(http.gets) == 1  # reused, no second GET
    assert remote_size(URL, transport=http) == len(CONTENT)


def test_download_resumes_or_restarts(tmp_path):
    for honour_range in (True, False):
        folder = tmp_path / str(honour_range)
        folder.mkdir()
        (folder / "statpearls_test.tar.gz.part").write_bytes(CONTENT[:1000])
        http = FakeHttp(CONTENT, honour_range=honour_range)
        info = download(URL, folder, transport=http, chunk_size=333)
        assert http.gets[0] == {"Range": "bytes=1000-"}
        assert (folder / "statpearls_test.tar.gz").read_bytes() == CONTENT
        assert info.sha256 == hashlib.sha256(CONTENT).hexdigest()


def test_download_failures_are_loud(tmp_path):
    with pytest.raises(RuntimeError, match="not the expected"):
        download(URL, tmp_path / "a", transport=FakeHttp(CONTENT), expected_size=5)
    with pytest.raises(RuntimeError, match="run the command again"):
        download(URL, tmp_path / "b", transport=FakeHttp(CONTENT, truncate_to=500))
    assert (tmp_path / "b" / "statpearls_test.tar.gz.part").stat().st_size == 500  # kept for resuming
    assert download(URL, tmp_path / "b", transport=FakeHttp(CONTENT)).size == len(CONTENT)
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "statpearls_test.tar.gz").write_bytes(b"something else")
    with pytest.raises(RuntimeError, match="move it away"):
        download(URL, tmp_path / "c", transport=FakeHttp(CONTENT))


def test_fixtures_are_synthetic():
    for name in NAMES:
        assert "Synthetic StatPearls-style article written for the FaTE-ICU tests" in (
            FIXTURES / name
        ).read_text("utf-8")
