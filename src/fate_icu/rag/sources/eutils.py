"""NCBI E-utilities client for PubMed (Stage 5): rate limit, retries and contact details.

NCBI's usage policy allows at most 3 requests per second without an API key and 10 with
one, and asks every request to name the tool and a contact email. Here the email and the
key are secrets: they are posted to NCBI and appear nowhere else (not in error messages,
logs or ``repr``). ``esearch`` returns only the first 10,000 ids of a search, so callers
slice searches by date (:func:`fate_icu.rag.sources.pubmed.slice_dates`).
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ESEARCH_CAP = 10_000  # esearch returns at most this many ids per search
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class EutilsError(RuntimeError):
    """An E-utilities request failed for good."""


@dataclass(frozen=True)
class TransportResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


class Transport(Protocol):
    def post(self, url: str, data: Mapping[str, str], timeout: float) -> TransportResponse: ...


class UrllibTransport:
    """HTTP POST with the standard library; HTTP errors are returned, not raised."""

    def post(self, url: str, data: Mapping[str, str], timeout: float) -> TransportResponse:
        request = urllib.request.Request(
            url,
            data=urllib.parse.urlencode(data).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "fate-icu"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return TransportResponse(response.status, dict(response.headers), response.read())
        except urllib.error.HTTPError as error:
            return TransportResponse(error.code, dict(error.headers or {}), error.read() or b"")


class RateLimiter:
    """Spaces request starts at least ``1 / per_second`` apart, across threads."""

    def __init__(self, per_second: float, clock: Callable[[], float], sleep: Callable[[float], None]):
        self.interval = 1.0 / per_second
        self._clock, self._sleep = clock, sleep
        self._next = float("-inf")
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:  # reserve a slot, then sleep outside the lock
            now = self._clock()
            start = max(now, self._next)
            self._next = start + self.interval
        if start > now:
            self._sleep(start - now)


# A validator returns None when the body is good, else (problem, retryable).
Validator = Callable[[bytes], tuple[str, bool] | None]


def _esearch_problem(body: bytes) -> tuple[str, bool] | None:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return "malformed JSON", True
    if "error" in payload:
        return "error payload", True  # e.g. rate limit exceeded
    result = payload.get("esearchresult")
    if not isinstance(result, dict) or "count" not in result:
        return "no esearchresult count", True
    if "ERROR" in result:
        return "search rejected by NCBI", False
    return None


def _efetch_problem(body: bytes) -> tuple[str, bool] | None:
    try:
        root = ET.fromstring(body)
    except ET.ParseError:
        return "malformed XML", True
    if root.tag != "PubmedArticleSet" or root.find(".//ERROR") is not None:
        return f"unexpected XML root {root.tag!r}", True
    return None


class EutilsClient:
    """PubMed esearch and efetch with NCBI's rate limit and contact parameters.

    Args:
        email: Contact email sent to NCBI with every request (required).
        api_key: Optional NCBI API key (raises the limit from 3 to 10 requests per second).
        tool: Tool name sent to NCBI.
        transport: HTTP transport (tests inject a fake one).
        clock, sleep: Time functions (tests inject fakes).
        max_retries: Retries after the first attempt.
        backoff_seconds: First retry delay; doubles each attempt.
        timeout: Seconds per request.
    """

    def __init__(
        self,
        *,
        email: str,
        api_key: str | None = None,
        tool: str = "fate-icu",
        transport: Transport | None = None,
        clock: Callable[
            [], float
        ] = time.perf_counter,  # monotonic; time.monotonic ticks every ~15 ms on Windows
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        timeout: float = 60.0,
    ):
        if not email or not email.strip():
            raise ValueError("an NCBI contact email is required (NCBI_EMAIL in .env)")
        self._email = email.strip()
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self.tool = tool
        self.transport = transport or UrllibTransport()
        self.sleep = sleep
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.timeout = timeout
        self.limiter = RateLimiter(10.0 if self._api_key else 3.0, clock, sleep)

    def __repr__(self) -> str:
        return f"EutilsClient(tool={self.tool!r}, api_key={'set' if self._api_key else 'not set'})"

    def _redact(self, text: str) -> str:
        for secret in filter(None, (self._email, self._api_key)):
            text = text.replace(secret, "<redacted>").replace(urllib.parse.quote(secret), "<redacted>")
        return text

    def _post(self, endpoint: str, params: Mapping[str, str], validate: Validator) -> bytes:
        data = {**params, "tool": self.tool, "email": self._email}
        if self._api_key:
            data["api_key"] = self._api_key
        last = "no attempt"
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                response = self.transport.post(EUTILS_BASE + endpoint, data, self.timeout)
            except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
                last = type(error).__name__  # the message could echo the request
            else:
                if response.status == 200:
                    problem = validate(response.body)
                    if problem is None:
                        return response.body
                    last, retryable = problem
                    if not retryable:
                        raise EutilsError(self._redact(f"{endpoint}: {last}"))
                elif response.status in RETRYABLE_STATUS:
                    last = f"HTTP {response.status}"
                else:
                    raise EutilsError(f"{endpoint}: HTTP {response.status} (not retried)")
            if attempt < self.max_retries:
                self.sleep(self.backoff_seconds * 2**attempt)
        raise EutilsError(self._redact(f"{endpoint}: gave up after {self.max_retries + 1} attempts ({last})"))

    def _esearch(
        self, term: str, retmax: int, mindate: str | None, maxdate: str | None, datetype: str
    ) -> dict:
        params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(retmax)}
        if mindate or maxdate:
            if not (mindate and maxdate):
                raise ValueError("give both mindate and maxdate")
            params.update(datetype=datetype, mindate=mindate, maxdate=maxdate)
        return json.loads(self._post("esearch.fcgi", params, _esearch_problem))["esearchresult"]

    def esearch_count(
        self, term: str, *, mindate: str | None = None, maxdate: str | None = None, datetype: str = "pdat"
    ) -> int:
        """Number of PubMed records matching ``term`` (dates as ``YYYY/MM/DD``)."""
        return int(self._esearch(term, 0, mindate, maxdate, datetype)["count"])

    def esearch_ids(
        self,
        term: str,
        *,
        mindate: str | None = None,
        maxdate: str | None = None,
        datetype: str = "pdat",
        retmax: int = ESEARCH_CAP - 1,
    ) -> list[int]:
        """Every PMID matching ``term`` in the date range.

        Raises:
            EutilsError: More than ``retmax`` matches (slice the dates), or the id list is
                shorter than the reported count.
        """
        if retmax >= ESEARCH_CAP:
            raise ValueError(f"retmax must be below {ESEARCH_CAP}")
        result = self._esearch(term, retmax, mindate, maxdate, datetype)
        count = int(result["count"])
        if count > retmax:
            raise EutilsError(f"{count} records exceed retmax={retmax}: slice the date range")
        ids = [int(x) for x in result.get("idlist", [])]
        if len(ids) != count:
            raise EutilsError(f"esearch reported {count} records but returned {len(ids)} ids")
        return ids

    def efetch_xml(self, pmids: Sequence[int]) -> bytes:
        """PubMed XML (``PubmedArticleSet``) for the given PMIDs."""
        if not pmids:
            raise ValueError("no PMIDs to fetch")
        params = {
            "db": "pubmed",
            "id": ",".join(str(int(p)) for p in pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        return self._post("efetch.fcgi", params, _efetch_problem)
