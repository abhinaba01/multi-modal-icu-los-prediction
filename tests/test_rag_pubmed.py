"""Stage 5 PubMed: the E-utilities client, date slicing, cached records and parsing (offline)."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from datetime import date
from pathlib import Path

import pytest

from fate_icu.rag.drivers import driver_keys
from fate_icu.rag.sources.eutils import EutilsClient, EutilsError, RateLimiter, TransportResponse
from fate_icu.rag.sources.pubmed import (
    PubmedQuerySet,
    batch_name,
    citation_text,
    collect_pmids,
    exclusion_reason,
    fetch_records,
    iter_documents,
    load_pubmed_queries,
    parse_pubmed_xml,
    record_to_document,
    slice_dates,
)
from fate_icu.utils.env import REPO_ROOT

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "rag" / "pubmed"
EMAIL, KEY = "someone@example.org", "key-123"


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _ok_json(count, ids=()):
    return TransportResponse(
        200,
        {},
        json.dumps({"esearchresult": {"count": str(count), "idlist": [str(i) for i in ids]}}).encode(),
    )


def _articles(ids):
    body = "".join(
        f"<PubmedArticle><MedlineCitation><PMID Version='1'>{i}</PMID><Article><ArticleTitle>Title {i}."
        f"</ArticleTitle><Abstract><AbstractText>Abstract {i}.</AbstractText></Abstract>"
        "<Language>eng</Language>"
        "</Article></MedlineCitation></PubmedArticle>"
        for i in ids
    )
    return f"<?xml version='1.0'?><PubmedArticleSet>{body}</PubmedArticleSet>".encode()


class FakeEutils:
    """esearch over {term: [(pmid, date)]} and efetch of minimal records."""

    def __init__(self, records=None, missing=(), count_bonus=0):
        self.records = records or {}
        self.missing = set(missing)
        self.count_bonus = count_bonus
        self.requests = []
        self.lock = threading.Lock()

    def _items(self, data):
        term = next((t for t in self.records if data["term"].startswith(f"({t})")), None)
        items = self.records.get(term, [])
        if "mindate" in data:
            low = date(*map(int, data["mindate"].split("/")))
            high = date(*map(int, data["maxdate"].split("/")))
            items = [(p, d) for p, d in items if low <= d <= high]
        return items

    def post(self, url, data, timeout):
        endpoint = url.rsplit("/", 1)[1]
        with self.lock:
            self.requests.append((endpoint, dict(data)))
        if endpoint == "esearch.fcgi":
            items = self._items(data)
            retmax = int(data["retmax"])
            bonus = self.count_bonus if retmax == 0 and data.get("mindate", "").startswith("1990") else 0
            return _ok_json(len(items) + bonus, sorted(p for p, _ in items)[:retmax])
        ids = [int(i) for i in data["id"].split(",") if int(i) not in self.missing]
        return TransportResponse(200, {}, _articles(ids))


def client_for(transport, *, api_key=None, **kw):
    fake = FakeTime()
    return EutilsClient(
        email=EMAIL, api_key=api_key, transport=transport, clock=fake.clock, sleep=fake.sleep, **kw
    ), fake


# --- client ---------------------------------------------------------------------------


def test_requests_carry_contact_details_and_secrets_stay_hidden():
    transport = FakeEutils({"a": [(1, date(2020, 1, 1))]})
    client, _ = client_for(transport, api_key=KEY)
    assert client.esearch_count("(a) AND (x)") == 1
    _, data = transport.requests[0]
    assert data["tool"] == "fate-icu" and data["email"] == EMAIL and data["api_key"] == KEY
    assert EMAIL not in repr(client) and KEY not in repr(client)

    class Rejecting:
        def post(self, url, data, timeout):
            return TransportResponse(400, {}, f"bad request from {data['email']}".encode())

    client, _ = client_for(Rejecting(), api_key=KEY)
    with pytest.raises(EutilsError) as error:
        client.esearch_count("(a)")
    assert EMAIL not in str(error.value) and KEY not in str(error.value)
    with pytest.raises(ValueError, match="email"):
        EutilsClient(email="  ")


def test_rate_limit_spacing_with_and_without_a_key():
    for api_key, interval in ((None, 1 / 3), (KEY, 1 / 10)):
        transport = FakeEutils({"a": []})
        client, fake = client_for(transport, api_key=api_key)
        for _ in range(4):
            client.esearch_count("(a)")
        assert fake.sleeps == pytest.approx([interval] * 3)


def test_rate_limiter_is_thread_safe():
    limiter = RateLimiter(50.0, time.perf_counter, time.sleep)
    starts, lock = [], threading.Lock()

    def worker():
        limiter.wait()
        with lock:
            starts.append(time.perf_counter())

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    gaps = [b - a for a, b in zip(sorted(starts), sorted(starts)[1:], strict=False)]
    assert min(gaps) >= 1 / 50 - 0.004


def test_retries_with_backoff_then_success():
    responses = [
        TransportResponse(429, {}, b""),
        TransportResponse(503, {}, b""),
        OSError("connection reset"),
        TransportResponse(200, {}, b"not json"),
        _ok_json(7),
    ]

    class Flaky:
        def post(self, url, data, timeout):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    client, fake = client_for(Flaky(), backoff_seconds=2.0)
    assert client.esearch_count("(a)") == 7
    assert fake.sleeps == [2.0, 4.0, 8.0, 16.0]


def test_gives_up_and_does_not_retry_client_errors():
    class Always:
        def __init__(self, status, body=b""):
            self.status, self.body, self.calls = status, body, 0

        def post(self, url, data, timeout):
            self.calls += 1
            return TransportResponse(self.status, {}, self.body)

    busy = Always(503)
    client, _ = client_for(busy, max_retries=2)
    with pytest.raises(EutilsError, match="gave up after 3 attempts"):
        client.esearch_count("(a)")
    assert busy.calls == 3
    bad = Always(404)
    client, _ = client_for(bad)
    with pytest.raises(EutilsError, match="not retried"):
        client.esearch_count("(a)")
    assert bad.calls == 1
    rejected = Always(200, json.dumps({"esearchresult": {"count": "0", "ERROR": "bad term"}}).encode())
    client, _ = client_for(rejected)
    with pytest.raises(EutilsError, match="rejected"):
        client.esearch_count("(a)")
    assert rejected.calls == 1
    broken_xml = Always(200, b"<html>busy</html>")
    client, _ = client_for(broken_xml, max_retries=1)
    with pytest.raises(EutilsError, match="unexpected XML"):
        client.efetch_xml([1])


def test_esearch_ids_cap_and_mismatch():
    class Fixed:
        def __init__(self, count, ids):
            self.count, self.ids = count, ids

        def post(self, url, data, timeout):
            return _ok_json(self.count, self.ids)

    client, _ = client_for(Fixed(12_000, []))
    with pytest.raises(EutilsError, match="slice the date range"):
        client.esearch_ids("(a)")
    client, _ = client_for(Fixed(3, [1, 2]))
    with pytest.raises(EutilsError, match="returned 2 ids"):
        client.esearch_ids("(a)")
    with pytest.raises(ValueError):
        client.esearch_ids("(a)", retmax=10_000)


# --- slicing and PMIDs -----------------------------------------------------------------


def _dated(start_pmid, days):
    return [(start_pmid + i, d) for i, d in enumerate(days)]


def test_slice_dates_splits_years_then_months_then_days():
    days = (
        [date(2019, 6, 1)] * 4
        + [date(2020, 3, d) for d in range(1, 13) for _ in range(2)]
        + [date(2021, 1, 5)]
    )
    records = {"a": _dated(1000, days)}
    client, _ = client_for(FakeEutils(records))
    slices, total = slice_dates(client, "(a) AND (f)", 2019, 2021, cap=10)
    assert total == len(days)
    assert all(0 < s.count <= 10 for s in slices)
    assert [s.mindate for s in slices] == sorted(s.mindate for s in slices)  # chronological
    assert sum(s.count for s in slices) == len(days)
    assert any(s.mindate == s.maxdate for s in slices)  # March 2020 needed day slices
    crowded = {"a": _dated(1, [date(2020, 3, 3)] * 11)}
    client, _ = client_for(FakeEutils(crowded))
    with pytest.raises(EutilsError, match="single day|on 2020/03/03"):
        slice_dates(client, "(a)", 2020, 2020, cap=10)


def _query_set(terms, filters="english[lang]"):
    return PubmedQuerySet.model_validate(
        {
            "version": 1,
            "filters": filters,
            "min_year": 1990,
            "max_year": 2026,
            "queries": [{"id": f"q_{t}", "term": t, "purpose": "topic"} for t in terms],
        }
    )


def test_collect_pmids_caches_by_term_and_keeps_the_latest_records(tmp_path):
    records = {"a": _dated(10, [date(2020, 1, 1)] * 5), "b": _dated(13, [date(2021, 1, 1)] * 4)}
    transport = FakeEutils(records)
    client, _ = client_for(transport)
    collection = collect_pmids(client, _query_set(["a", "b"]), tmp_path)
    assert collection.pmids == list(range(10, 17)) and collection.per_query == {"q_a": 5, "q_b": 4}
    assert json.loads((tmp_path / "pmids.json").read_text())["pmids"] == collection.pmids
    n_requests = len(transport.requests)
    again = collect_pmids(client, _query_set(["a", "b"]), tmp_path)
    assert again.pmids == collection.pmids and len(transport.requests) == n_requests  # cache reused
    changed = collect_pmids(client, _query_set(["a", "b"], filters="english[lang] AND x"), tmp_path)
    assert changed.query_set_sha256 != collection.query_set_sha256 and len(transport.requests) > n_requests
    limited = collect_pmids(client, _query_set(["a", "b"]), tmp_path, max_records=3)
    assert limited.pmids == [14, 15, 16]


def test_collect_pmids_fails_when_slices_lose_records(tmp_path):
    client, _ = client_for(FakeEutils({"a": _dated(1, [date(2020, 1, 1)] * 3)}, count_bonus=1))
    with pytest.raises(EutilsError, match="returned 3 of 4"):
        collect_pmids(client, _query_set(["a"]), tmp_path)


class NotingEutils(FakeEutils):
    """A fake that also returns PubMed's translation and warnings for searches without dates."""

    def post(self, url, data, timeout):
        response = super().post(url, data, timeout)
        if url.endswith("esearch.fcgi") and "mindate" not in data and data["retmax"] == "0":
            payload = json.loads(response.body)
            payload["esearchresult"].update(
                querytranslation="translated " + data["term"][:12],
                errorlist={"phrasesnotfound": ["Imaginary Heading"], "fieldsnotfound": []},
                warninglist={"quotedphrasesnotfound": [], "outputmessages": ["Quoted phrase not found"]},
            )
            return TransportResponse(200, {}, json.dumps(payload).encode())
        return response


def test_collect_pmids_records_pubmeds_notes_and_logs_progress(tmp_path):
    records = {"a": _dated(10, [date(2020, 1, 1)] * 3), "b": _dated(20, [date(2021, 1, 1)] * 2)}
    client, _ = client_for(NotingEutils(records))
    lines = []
    collection = collect_pmids(client, _query_set(["a", "b"]), tmp_path, log=lines.append)
    expected = {
        "errors": {"phrasesnotfound": ["Imaginary Heading"]},  # empty lists are left out
        "warnings": {"outputmessages": ["Quoted phrase not found"]},
    }
    assert collection.warnings == {"q_a": expected, "q_b": expected}
    saved = json.loads((tmp_path / "esearch" / "q_a.json").read_text(encoding="utf-8"))
    assert saved["notes"]["translation"].startswith("translated (a) AND") and saved["notes"]["errors"]
    assert json.loads((tmp_path / "pmids.json").read_text())["warnings"]["q_b"] == expected
    assert lines[0].startswith("[1/2] q_a: 3 records (searched); 3 unique so far; PubMed notes:")
    assert lines[1].startswith("[2/2] q_b: 2 records (searched); 5 unique so far")
    again = []
    collect_pmids(client, _query_set(["a", "b"]), tmp_path, log=again.append)
    assert "(cached)" in again[0] and "(cached)" in again[1]  # the notes come back from the cache
    plain, _ = client_for(FakeEutils(records))
    assert collect_pmids(plain, _query_set(["a"]), tmp_path / "plain").warnings == {}  # nothing to report


def test_fetch_command_ids_only_first_then_the_records(tmp_path, monkeypatch, capsys):
    from fate_icu.rag import fetch
    from fate_icu.rag.config import load_config
    from fate_icu.utils.env import load_settings

    records = {"a": _dated(10, [date(2020, 1, 1)] * 3), "b": _dated(20, [date(2021, 1, 1)] * 2)}
    transport = NotingEutils(records)
    monkeypatch.setattr(
        fetch,
        "eutils_client",
        lambda settings: EutilsClient(email=EMAIL, transport=transport, sleep=lambda s: None),
    )
    monkeypatch.setattr(fetch, "load_pubmed_queries", lambda path: _query_set(["a", "b"]))
    monkeypatch.setenv("FATE_RAG_DIR", str(tmp_path / "rag"))
    for name in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings(env_file=tmp_path / "none.env")
    config = load_config("configs/rag/full.yaml")
    efetch = tmp_path / "rag" / "raw" / "pubmed" / "efetch"

    lines = []
    ids = fetch.run(config, "pubmed", ids_only=True, settings=settings, log=lines.append)["pubmed"]
    assert (ids["searches"], ids["unique_pmids"], ids["batches_to_fetch"]) == (2, 5, 1)
    assert ids["ids_per_search"] == {"q_a": 3, "q_b": 2} and set(ids["searches_with_pubmed_notes"]) == {
        "q_a",
        "q_b",
    }
    assert "fetch" not in ids and not efetch.exists() and "without --ids-only" in ids["next_step"]

    searches_so_far = len([r for r in transport.requests if r[0] == "esearch.fcgi"])
    full = fetch.run(config, "pubmed", settings=settings, log=lines.append)["pubmed"]
    assert (
        full["fetch"]["pmids_found"] == 5
        and full["fetch"]["fetched"] == 1
        and len(list(efetch.glob("batch-*.xml"))) == 1
    )
    assert len([r for r in transport.requests if r[0] == "esearch.fcgi"]) == searches_so_far  # not repeated
    assert "(cached)" in lines[2] and not any("EMAIL" in line or EMAIL in line for line in lines)

    with pytest.raises(ValueError, match="--ids-only"):
        fetch.run(config, "statpearls", ids_only=True, settings=settings)
    fetch.timestamped("hello")
    assert capsys.readouterr().out.strip().endswith(" hello")


# --- records --------------------------------------------------------------------------


def test_fetch_records_batches_resume_and_parallel_equals_serial(tmp_path):
    pmids = list(range(100, 110))
    serial_dir, parallel_dir = tmp_path / "serial", tmp_path / "parallel"
    client, _ = client_for(FakeEutils())
    report = fetch_records(client, list(reversed(pmids)), serial_dir, batch_size=3)
    assert (report.batches, report.fetched, report.reused, report.pmids_found) == (4, 4, 0, 10)
    files = sorted(p.name for p in (serial_dir / "efetch").glob("batch-*.xml"))
    assert files == [batch_name(pmids[3 * i : 3 * i + 3]) for i in range(4)]
    assert not list((serial_dir / "efetch").glob("*.tmp*"))
    assert fetch_records(client, pmids, serial_dir, batch_size=3).fetched == 0  # resumed
    (serial_dir / "efetch" / files[1]).unlink()
    assert fetch_records(client, pmids, serial_dir, batch_size=3).fetched == 1
    real_time_client = EutilsClient(email=EMAIL, transport=FakeEutils(), sleep=lambda s: None)
    parallel = fetch_records(real_time_client, pmids, parallel_dir, batch_size=3, max_workers=3)
    assert parallel == fetch_records(client, pmids, tmp_path / "serial2", batch_size=3)
    for name in files:
        assert (parallel_dir / "efetch" / name).read_bytes() == (
            tmp_path / "serial2" / "efetch" / name
        ).read_bytes()


def test_fetch_records_logs_progress(tmp_path):
    client = EutilsClient(email=EMAIL, transport=FakeEutils(), sleep=lambda s: None)
    lines = []
    fetch_records(client, list(range(1, 61)), tmp_path, batch_size=2, max_workers=3, log=lines.append)
    assert lines[0] == "60 of 60 records to fetch in 30 batches (0 already on disk or unavailable)"
    assert lines[1:] == ["fetched 25 / 30 batches", "fetched 30 / 30 batches"]
    again = []
    fetch_records(client, list(range(1, 61)), tmp_path, batch_size=2, log=again.append)
    assert again == ["0 of 60 records to fetch in 0 batches (60 already on disk or unavailable)"]


def _requested(transport):
    return {
        int(i)
        for endpoint, data in transport.requests
        if endpoint == "efetch.fcgi"
        for i in data["id"].split(",")
    }


def test_fetch_records_downloads_only_what_is_new(tmp_path):
    transport = FakeEutils(missing={7})
    client, _ = client_for(transport)
    first = fetch_records(client, list(range(1, 201)), tmp_path, batch_size=50)
    assert (first.batches, first.fetched, first.reused, first.pmids_found, first.pmids_missing) == (
        4,
        4,
        0,
        199,
        1,
    )

    transport.requests.clear()  # more searches added: 60 more PMIDs, and PMID 7 is not asked for again
    grown = fetch_records(client, list(range(1, 261)), tmp_path, batch_size=50)
    assert (grown.batches, grown.fetched, grown.reused, grown.pmids_found) == (6, 2, 4, 259)
    assert _requested(transport) == set(range(201, 261))

    transport.requests.clear()  # a search removed, and a different set of PMIDs: nothing to download
    smaller = fetch_records(client, list(range(2, 152)), tmp_path, batch_size=7)
    assert (smaller.batches, smaller.fetched, smaller.reused) == (4, 0, 4) and not transport.requests
    assert smaller.pmids_found == 149 and smaller.pmids_missing == 1  # PMID 7 stays unavailable
    index = json.loads((tmp_path / "efetch" / "index.json").read_text(encoding="utf-8"))
    assert index["unavailable"] == [7] and len(index["files"]) == 4
    assert all(entry["retrieved_on"] for entry in index["files"].values())


def test_fetch_records_counts_book_chapters_as_found(tmp_path):
    book = "<PubmedBookArticle><BookDocument><PMID Version='1'>3</PMID></BookDocument></PubmedBookArticle>"

    class WithBooks(FakeEutils):
        def post(self, url, data, timeout):  # PMID 3 comes back as a book chapter, as StatPearls does
            if not url.endswith("efetch.fcgi"):
                return super().post(url, data, timeout)
            ids = [int(i) for i in data["id"].split(",")]
            articles = _articles([i for i in ids if i != 3]).decode().replace("</PubmedArticleSet>", "")
            return TransportResponse(
                200, {}, (articles + (book if 3 in ids else "") + "</PubmedArticleSet>").encode()
            )

    client, _ = client_for(WithBooks())
    report = fetch_records(client, list(range(1, 201)), tmp_path, batch_size=100)
    assert (report.pmids_found, report.pmids_missing) == (200, 0)  # the chapter counts as found
    records = [r for f in (tmp_path / "efetch").glob("batch-*.xml") for r in parse_pubmed_xml(f.read_bytes())]
    chapter = next(r for r in records if r.pmid == 3)
    assert chapter.kind == "book" and exclusion_reason(chapter) == "book_article"


def test_fetch_records_refuses_a_pmid_in_two_batch_files(tmp_path):
    client = EutilsClient(email=EMAIL, transport=FakeEutils(), sleep=lambda s: None)
    efetch = tmp_path / "efetch"
    efetch.mkdir()
    (efetch / "batch-000000001-0003-aaaaaaaa.xml").write_bytes(_articles([1, 2, 3]))
    (efetch / "batch-000000003-0003-bbbbbbbb.xml").write_bytes(_articles([3, 4, 5]))
    with pytest.raises(RuntimeError, match="PMID 3 is in two batch files"):
        fetch_records(client, [1, 2, 3, 4, 5], tmp_path)


def test_fetch_records_missing_threshold(tmp_path):
    client, _ = client_for(FakeEutils(missing={105}))
    report = fetch_records(client, list(range(1, 201)), tmp_path / "ok", batch_size=50)
    assert (report.pmids_missing, report.pmids_found) == (1, 199)  # 0.5%: counted, not fatal
    client, _ = client_for(FakeEutils(missing={3, 7}))
    with pytest.raises(EutilsError, match="2 of 10"):
        fetch_records(client, list(range(1, 11)), tmp_path / "bad", batch_size=5)


def _fixture_records():
    records = parse_pubmed_xml((FIXTURES / "batch_a.xml").read_bytes())
    records += parse_pubmed_xml((FIXTURES / "batch_b.xml").read_bytes())
    return {r.pmid: r for r in records}


def test_parse_fixture_records_every_field():
    records = _fixture_records()
    # the book chapter is read (and excluded); the deletion notice is skipped
    assert sorted(records) == [*range(90000001, 90000013), 90000098]
    assert records[90000098].kind == "book" and records[90000001].kind == "article"
    first = records[90000001]
    assert first.title == "Early lactate clearance and length of stay in a synthetic intensive care cohort."
    assert first.abstract[0].startswith("BACKGROUND: Raised blood lactate") and len(first.abstract) == 4
    assert all("Copyright" not in p for p in first.abstract)
    assert "(p < 0.05)" in first.abstract[2]
    assert (first.journal, first.first_author, first.n_authors, first.year) == (
        "Crit Care Med",
        "Rao A",
        3,
        2019,
    )
    assert (
        first.doi == "10.0000/synthetic.0001"
        and first.mesh[1] == "Lactic Acid"
        and first.languages == ("eng",)
    )
    second = records[90000002]
    assert second.title.endswith("first day of intensive care and the risk of acute kidney injury.")
    assert "per 1.73 m2 where" in second.abstract[0]
    assert (second.year, second.journal, second.first_author, second.n_authors) == (
        2018,
        "Journal of synthetic nephrology",
        "Synthetic ICU Outcomes Collaborative",
        1,
    )
    assert second.doi == "10.0000/synthetic.0002"
    assert records[90000008].year == 2021  # ArticleDate when PubDate has no year
    assert records[90000012].year == 2024  # PubMed history date as a last resort
    assert records[90000009].retracted and records[90000005].retracted


@pytest.mark.parametrize(
    "pmid, reason",
    [
        (90000001, None),
        (90000003, "no_abstract"),
        (90000004, "not_english"),
        (90000005, "retracted"),
        (90000006, "publication_type:Letter"),
        (90000007, "publication_type:Editorial"),
        (90000009, "retracted"),
        (90000012, None),
        (90000098, "book_article"),
    ],
)
def test_exclusion_reasons(pmid, reason):
    assert exclusion_reason(_fixture_records()[pmid]) == reason


def test_documents_and_citations():
    records = _fixture_records()
    doc = record_to_document(records[90000001], licence="L", retrieved_on="2026-11-13")
    assert (
        doc.source == "pubmed"
        and doc.doc_id == "90000001"
        and doc.url == "https://pubmed.ncbi.nlm.nih.gov/90000001/"
    )
    assert len(doc.sections) == 1 and doc.sections[0].title == "" and doc.licence == "L"
    assert doc.meta["retrieved_on"] == "2026-11-13" and doc.meta["journal"] == "Crit Care Med"
    assert citation_text(records[90000001]) == "Rao A, et al. Crit Care Med. 2019."
    assert (
        citation_text(records[90000002])
        == "Synthetic ICU Outcomes Collaborative. Journal of synthetic nephrology. 2018."
    )
    assert citation_text(records[90000012]) == "Moreau J. Synthetic journal of neurocritical care. 2024."
    assert record_to_document(records[90000003], licence="L") is None


def _cache(tmp_path, files):
    efetch = tmp_path / "efetch"
    efetch.mkdir(parents=True)
    index = {}
    for i, source in enumerate(files):
        name = f"batch-{i:05d}-test.xml"
        shutil.copy(source, efetch / name)
        index[name] = {"pmids": 0, "retrieved_on": f"2026-11-1{i}"}
    (efetch / "index.json").write_text(json.dumps({"files": index}))
    return tmp_path


def test_iter_documents_filters_counts_and_detects_duplicates(tmp_path):
    cache = _cache(tmp_path / "ok", [FIXTURES / "batch_a.xml", FIXTURES / "batch_b.xml"])
    wanted = set(range(90000001, 90000013)) | {90000050, 90000098}
    documents, counts = iter_documents(cache, wanted, licence="L")
    assert [d.doc_id for d in documents] == [
        "90000001",
        "90000002",
        "90000008",
        "90000010",
        "90000011",
        "90000012",
    ]
    assert counts == {
        "kept": 6,
        "no_abstract": 1,
        "not_english": 1,
        "retracted": 2,
        "publication_type:Letter": 1,
        "publication_type:Editorial": 1,
        "book_article": 1,
        "missing": 1,
    }
    assert (
        documents[0].meta["retrieved_on"] == "2026-11-10"
        and documents[2].meta["retrieved_on"] == "2026-11-11"
    )
    subset, _ = iter_documents(cache, {90000002}, licence="L")
    assert [d.doc_id for d in subset] == ["90000002"]
    twice = _cache(tmp_path / "dup", [FIXTURES / "batch_a.xml", FIXTURES / "batch_a.xml"])
    with pytest.raises(ValueError, match="two batch files"):
        iter_documents(twice, wanted, licence="L")
    with pytest.raises(FileNotFoundError, match="fetch"):
        iter_documents(tmp_path / "none", wanted, licence="L")


# --- the searches ---------------------------------------------------------------------


def _balanced(term: str) -> bool:
    depth = 0
    for ch in term:
        depth += {"(": 1, ")": -1}.get(ch, 0)
        if depth < 0:
            return False
    return depth == 0 and term.count('"') % 2 == 0


def test_a_search_that_finds_nothing_is_an_error(tmp_path):
    client, _ = client_for(NotingEutils({"a": _dated(10, [date(2020, 1, 1)] * 3)}))
    with pytest.raises(EutilsError, match=r"search q_empty found no records.*translated.*Imaginary Heading"):
        collect_pmids(client, _query_set(["a", "empty"]), tmp_path)
    assert (tmp_path / "esearch" / "q_a.json").exists()  # the good search is kept for the next run


def test_the_search_file_covers_every_driver_key_and_topic(variables, contract):
    query_set = load_pubmed_queries()
    served = {k for q in query_set.queries if q.purpose == "driver" for k in q.keys}
    expected = {f"{k.modality}:{k.key}" for k in driver_keys(variables, contract)}
    assert served == expected, (sorted(expected - served), sorted(served - expected))
    topics = {q.id for q in query_set.queries if q.purpose == "topic"}
    assert {
        "topic_length_of_stay",
        "topic_acute_kidney_injury",
        "topic_sepsis",
        "topic_ards",
        "topic_delirium",
        "topic_vasopressors",
        "topic_mechanical_ventilation",
    } <= topics
    for query in query_set.queries:
        full = query_set.full_term(query)
        assert _balanced(full) and "{" not in full and "}" not in full, query.id
        # PubMed silently turns "A AND NOT B" into "A AND B": exclusions must be written "NOT"
        assert not re.search(r"\b(AND|OR)\s+NOT\b", full, re.IGNORECASE), query.id
        assert "retraction of publication" not in full and "type of admission" not in full, query.id
        if query.purpose == "driver":
            assert "Intensive Care Units" in full and "Length of Stay" in full, query.id
    assert len(query_set.sha256) == 64


def test_unknown_blocks_fail_loudly():
    with pytest.raises(ValueError, match="unknown blocks"):
        PubmedQuerySet.model_validate(
            {
                "version": 1,
                "filters": "x",
                "min_year": 2000,
                "max_year": 2001,
                "blocks": {"icu": "ICU"},
                "queries": [{"id": "a", "term": "{icu} AND {missing}", "purpose": "topic"}],
            }
        )


def test_fixture_folder_is_synthetic():
    for path in Path(FIXTURES).glob("*.xml"):
        assert "Synthetic records written for the FaTE-ICU tests" in path.read_text(encoding="utf-8")
