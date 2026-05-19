"""EDGAR HTTP client.

Handles the SEC-mandated User-Agent header, polite throttling, retry with backoff, and
walks the daily index plus full-text search to discover 8-K filings.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date, timedelta
from types import TracebackType
from typing import Any

import httpx

_PLACEHOLDER_USER_AGENT = "Your Name your.email@example.com"

_DAILY_INDEX_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{q}/master.{ymd}.idx"
)
_FULL_SUBMISSION_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/"
    "{acc_with_dashes}.txt"
)
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"

_ACCESSION_DASHED_RE = re.compile(r"^\d{10}-\d{2}-\d{6}$")
_ACCESSION_DASHED_FIND_RE = re.compile(r"\d{10}-\d{2}-\d{6}")
_ACCESSION_NODASH_RE = re.compile(r"^\d{18}$")
_CIK_RE = re.compile(r"^\d{1,10}$")
_EIGHT_K_FORMS = ("8-K", "8-K/A")


class EdgarError(Exception):
    """Base class for EDGAR-related errors."""


class EdgarForbiddenError(EdgarError):
    """SEC returned 403 — typically a missing or unacceptable User-Agent."""


class EdgarNotFoundError(EdgarError):
    """SEC returned 404 — resource does not exist (e.g., weekend daily index)."""


class EdgarRateLimitedError(EdgarError):
    """SEC returned 429 after retry budget was exhausted."""


@dataclass(frozen=True)
class IndexRow:
    """One row from the daily master.idx file."""

    cik: str
    company: str
    form_type: str
    filing_date: date
    filename_path: str
    accession_with_dashes: str


@dataclass(frozen=True)
class SubmissionsLookup:
    """Subset of fields from data.sec.gov submissions JSON."""

    name: str
    tickers: list[str] = field(default_factory=list)
    exchanges: list[str] = field(default_factory=list)


class RateLimiter:
    """Sleep-to-next-slot scheduler. Process-wide rate limiter at `rps` requests/sec."""

    def __init__(self, rps: float) -> None:
        if rps <= 0:
            raise ValueError("rps must be positive")
        self._min_interval = 1.0 / rps
        self._last = 0.0

    def acquire(self) -> None:
        now = time.monotonic()
        wait = self._last + self._min_interval - now
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()


def normalize_cik(raw: str | int) -> str:
    """Pad to 10 digits. Accept int or string with/without leading zeros."""
    if isinstance(raw, int):
        s = str(raw)
    else:
        s = raw.strip()
    if not _CIK_RE.fullmatch(s):
        raise ValueError(f"invalid CIK: {raw!r}")
    return s.zfill(10)


def normalize_accession(raw: str) -> tuple[str, str]:
    """Return (with_dashes, no_dashes). Accept either input form."""
    s = raw.strip()
    if _ACCESSION_DASHED_RE.fullmatch(s):
        return s, s.replace("-", "")
    if _ACCESSION_NODASH_RE.fullmatch(s):
        return f"{s[0:10]}-{s[10:12]}-{s[12:18]}", s
    raise ValueError(f"invalid accession number: {raw!r}")


def daily_index_url(d: date) -> str:
    q = (d.month - 1) // 3 + 1
    return _DAILY_INDEX_URL.format(year=d.year, q=q, ymd=d.strftime("%Y%m%d"))


def full_submission_url(cik_int: str, acc_nodash: str, acc_with_dashes: str) -> str:
    """URL to the combined-submission .txt file containing the SGML envelope."""
    return _FULL_SUBMISSION_URL.format(
        cik_int=cik_int, acc_nodash=acc_nodash, acc_with_dashes=acc_with_dashes
    )


def submissions_url(cik_padded: str) -> str:
    return _SUBMISSIONS_URL.format(cik_padded=cik_padded)


def parse_daily_index(text: str) -> Iterator[IndexRow]:
    """Yield 8-K and 8-K/A rows from a master.<DATE>.idx body."""
    started = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not started:
            if line.startswith("---"):
                started = True
            continue
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik_raw, company, form_type, date_str, filename = parts
        if form_type not in _EIGHT_K_FORMS:
            continue
        try:
            cik = normalize_cik(cik_raw)
            filing_date = date.fromisoformat(date_str)
        except ValueError:
            continue
        accession_with_dashes = _accession_from_filename(filename)
        if accession_with_dashes is None:
            continue
        yield IndexRow(
            cik=cik,
            company=company.strip(),
            form_type=form_type,
            filing_date=filing_date,
            filename_path=filename.strip(),
            accession_with_dashes=accession_with_dashes,
        )


def _accession_from_filename(filename: str) -> str | None:
    """Pull the dashed accession number out of a master.idx Filename column.

    The Filename column varies historically: `.../<acc>.txt` (modern combined
    submission), `.../<acc>-index.htm` (filing index page), or `.../<acc>/...`.
    Robust against all three via substring search.
    """
    base = filename.rsplit("/", 1)[-1]
    m = _ACCESSION_DASHED_FIND_RE.search(base)
    return m.group(0) if m else None


class EdgarClient:
    """Thin httpx wrapper enforcing User-Agent, rate limit, retry, and typed errors."""

    def __init__(
        self,
        user_agent: str,
        rps: float = 8.0,
        timeout: float = 30.0,
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        ua = (user_agent or "").strip()
        if not ua or ua == _PLACEHOLDER_USER_AGENT:
            raise ValueError(
                "EDGAR_USER_AGENT is empty or matches the placeholder. "
                "Set it to '<Your Name> <your.email@example.com>' per SEC fair-use policy."
            )
        self._user_agent = ua
        self._limiter = limiter if limiter is not None else RateLimiter(rps)
        self._client = httpx.Client(
            headers={
                "User-Agent": ua,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=httpx.Timeout(connect=5.0, read=timeout, write=10.0, pool=10.0),
            transport=transport,
            follow_redirects=True,
        )

    def get_text(self, url: str) -> str:
        return self._request(url).text

    def get_bytes(self, url: str) -> bytes:
        return self._request(url).content

    def get_json(self, url: str) -> dict[str, Any]:
        data = self._request(url).json()
        if not isinstance(data, dict):
            raise EdgarError(f"expected JSON object from {url}, got {type(data).__name__}")
        return data

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _request(self, url: str) -> httpx.Response:
        """Issue a GET with rate-limit, retry on 429/5xx/transient, raise typed errors otherwise."""
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential_jitter,
        )

        @retry(
            retry=retry_if_exception_type(
                (EdgarRateLimitedError, httpx.TransportError, httpx.ReadTimeout)
            ),
            wait=wait_exponential_jitter(initial=1.0, max=30.0),
            stop=stop_after_attempt(5),
            reraise=True,
        )
        def _do() -> httpx.Response:
            self._limiter.acquire()
            response = self._client.get(url)
            status = response.status_code
            if status == 200:
                return response
            if status == 403:
                raise EdgarForbiddenError(
                    f"EDGAR returned 403 for {url}. Confirm User-Agent contains a real name "
                    "and a working email (see .env.example)."
                )
            if status == 404:
                raise EdgarNotFoundError(f"EDGAR returned 404 for {url}")
            if status == 429:
                raise EdgarRateLimitedError(f"EDGAR returned 429 for {url}")
            if 500 <= status < 600:
                raise EdgarRateLimitedError(f"EDGAR returned {status} for {url}")
            raise EdgarError(f"unexpected status {status} for {url}")

        return _do()


def iter_8k_accessions(
    client: EdgarClient, start: date, end: date
) -> Iterator[IndexRow]:
    """Yield 8-K/8-K/A index rows in [start, end], most recent date first.

    Weekends are skipped preemptively (no request fired). SEC returns 403 (not 404)
    for absent daily indexes on weekends/holidays, so we treat 403 on this endpoint
    the same as 404 — "no data, continue". A real rate-limit 403 will still surface
    on subsequent filing-data fetches (which propagate the error).
    """
    if start > end:
        raise ValueError("start must be on or before end")
    cur = end
    while cur >= start:
        if cur.weekday() >= 5:
            cur -= timedelta(days=1)
            continue
        url = daily_index_url(cur)
        try:
            text = client.get_text(url)
        except (EdgarNotFoundError, EdgarForbiddenError):
            cur -= timedelta(days=1)
            continue
        yield from parse_daily_index(text)
        cur -= timedelta(days=1)


def fetch_submissions(client: EdgarClient, cik_padded: str) -> SubmissionsLookup:
    """Look up filer name and ticker list from data.sec.gov submissions JSON."""
    data = client.get_json(submissions_url(cik_padded))
    tickers_raw = data.get("tickers") or []
    exchanges_raw = data.get("exchanges") or []
    return SubmissionsLookup(
        name=str(data.get("name", "")).strip(),
        tickers=[str(t) for t in tickers_raw if t],
        exchanges=[str(e) for e in exchanges_raw if e],
    )
