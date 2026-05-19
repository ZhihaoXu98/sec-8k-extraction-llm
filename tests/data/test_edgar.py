"""Tests for sec8k.data.edgar — HTTP client, rate limiter, URL builders, normalizers."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest

from sec8k.data.edgar import (
    EdgarClient,
    EdgarForbiddenError,
    EdgarNotFoundError,
    RateLimiter,
    daily_index_url,
    full_submission_url,
    iter_8k_accessions,
    normalize_accession,
    normalize_cik,
    parse_daily_index,
    submissions_url,
)


def test_normalize_cik_from_int() -> None:
    assert normalize_cik(320193) == "0000320193"


def test_normalize_cik_from_unpadded_str() -> None:
    assert normalize_cik("320193") == "0000320193"


def test_normalize_cik_already_padded() -> None:
    assert normalize_cik("0000320193") == "0000320193"


@pytest.mark.parametrize("bad", ["", "abc", "12345678901", "12-34"])
def test_normalize_cik_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_cik(bad)


def test_normalize_accession_dashed() -> None:
    assert normalize_accession("0000320193-24-000123") == (
        "0000320193-24-000123",
        "000032019324000123",
    )


def test_normalize_accession_nodash() -> None:
    assert normalize_accession("000032019324000123") == (
        "0000320193-24-000123",
        "000032019324000123",
    )


@pytest.mark.parametrize("bad", ["", "abc", "1234567890", "0000320193-24-0001234"])
def test_normalize_accession_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        normalize_accession(bad)


def test_daily_index_url_q2() -> None:
    assert daily_index_url(date(2024, 5, 2)) == (
        "https://www.sec.gov/Archives/edgar/daily-index/2024/QTR2/master.20240502.idx"
    )


def test_daily_index_url_q1_boundary() -> None:
    assert "QTR1" in daily_index_url(date(2024, 3, 31))


def test_daily_index_url_q3_boundary() -> None:
    assert "QTR3" in daily_index_url(date(2024, 7, 1))


def test_full_submission_url() -> None:
    assert full_submission_url(
        "320193", "000032019324000123", "0000320193-24-000123"
    ) == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/"
        "0000320193-24-000123.txt"
    )


def test_submissions_url() -> None:
    assert (
        submissions_url("0000320193")
        == "https://data.sec.gov/submissions/CIK0000320193.json"
    )


def test_parse_daily_index_yields_8k_only(edgar_fixtures_dir: Path) -> None:
    text = (edgar_fixtures_dir / "daily_index_sample.idx").read_text()
    rows = list(parse_daily_index(text))
    assert len(rows) == 4
    assert all(r.form_type in ("8-K", "8-K/A") for r in rows)
    assert "8-K/A" in [r.form_type for r in rows]


def test_parse_daily_index_extracts_canonical_fields(edgar_fixtures_dir: Path) -> None:
    text = (edgar_fixtures_dir / "daily_index_sample.idx").read_text()
    rows = list(parse_daily_index(text))
    by_acc = {r.accession_with_dashes: r for r in rows}
    apple = by_acc["0000320193-24-000123"]
    assert apple.cik == "0000320193"
    assert apple.company == "Apple Inc."
    assert apple.filing_date == date(2024, 5, 2)


def test_rate_limiter_paces_calls() -> None:
    limiter = RateLimiter(rps=20.0)
    start = time.monotonic()
    for _ in range(4):
        limiter.acquire()
    elapsed = time.monotonic() - start
    assert 0.10 < elapsed < 0.5


def test_rate_limiter_rejects_zero_rps() -> None:
    with pytest.raises(ValueError):
        RateLimiter(rps=0)


def test_edgar_client_rejects_empty_user_agent() -> None:
    with pytest.raises(ValueError):
        EdgarClient(user_agent="")


def test_edgar_client_rejects_placeholder_user_agent() -> None:
    with pytest.raises(ValueError):
        EdgarClient(user_agent="Your Name your.email@example.com")


def test_edgar_client_sets_user_agent_header(
    make_edgar_client: Callable[[Callable[[httpx.Request], httpx.Response]], EdgarClient],
    test_user_agent: str,
) -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("User-Agent", "")
        return httpx.Response(200, text="ok")

    with make_edgar_client(handler) as client:
        assert client.get_text("https://example.com/") == "ok"
    assert captured["ua"] == test_user_agent


def test_edgar_client_403_raises_forbidden(
    no_sleep: None,
    make_edgar_client: Callable[[Callable[[httpx.Request], httpx.Response]], EdgarClient],
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with make_edgar_client(handler) as client:
        with pytest.raises(EdgarForbiddenError):
            client.get_text("https://example.com/")


def test_edgar_client_404_raises_not_found(
    no_sleep: None,
    make_edgar_client: Callable[[Callable[[httpx.Request], httpx.Response]], EdgarClient],
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with make_edgar_client(handler) as client:
        with pytest.raises(EdgarNotFoundError):
            client.get_text("https://example.com/")


def test_edgar_client_429_retries_then_succeeds(
    no_sleep: None,
    make_edgar_client: Callable[[Callable[[httpx.Request], httpx.Response]], EdgarClient],
) -> None:
    counter = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] < 2:
            return httpx.Response(429)
        return httpx.Response(200, text="ok")

    with make_edgar_client(handler) as client:
        assert client.get_text("https://example.com/") == "ok"
    assert counter["n"] == 2


def test_edgar_client_5xx_retries_then_succeeds(
    no_sleep: None,
    make_edgar_client: Callable[[Callable[[httpx.Request], httpx.Response]], EdgarClient],
) -> None:
    counter = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] < 2:
            return httpx.Response(503)
        return httpx.Response(200, text="ok")

    with make_edgar_client(handler) as client:
        assert client.get_text("https://example.com/") == "ok"


def test_iter_8k_accessions_handles_weekend_404(
    no_sleep: None,
    edgar_fixtures_dir: Path,
    make_edgar_client: Callable[[Callable[[httpx.Request], httpx.Response]], EdgarClient],
) -> None:
    idx_text = (edgar_fixtures_dir / "daily_index_sample.idx").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "master.20240503.idx" in url:
            # Friday's index exists
            return httpx.Response(200, text=idx_text)
        # Weekend dates return 404 from EDGAR
        return httpx.Response(404)

    with make_edgar_client(handler) as client:
        rows = list(
            iter_8k_accessions(client, start=date(2024, 5, 3), end=date(2024, 5, 5))
        )
    # Only the Friday index yields rows; Sat/Sun 404s are absorbed.
    assert len(rows) == 4
