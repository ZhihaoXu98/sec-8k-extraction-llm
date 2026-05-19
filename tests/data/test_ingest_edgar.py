"""Tests for sec8k.data.ingest_edgar — category mapping, orchestrator, CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from sec8k.data.edgar import EdgarClient, RateLimiter
from sec8k.data.ingest_edgar import (
    CATEGORY_PRIORITY,
    ITEM_TO_CATEGORY,
    app,
    assign_primary_category,
    assign_primary_item,
    ingest,
)


# ---- category mapping -----------------------------------------------------------


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (["2.02"], "financial_results"),
        (["5.02"], "executive_change"),
        (["1.01"], "material_agreement"),  # 1.01 defaults to material_agreement at ingest
        (["2.01"], "m_and_a"),
        (["7.01"], "other"),
        (["8.01"], "other"),
        (["4.02"], "regulatory"),
        (["2.02", "9.01"], "financial_results"),  # 9.01 ignored
        (["5.02", "9.01"], "executive_change"),
    ],
)
def test_assign_primary_category_single(items: list[str], expected: str) -> None:
    assert assign_primary_category(items) == expected


@pytest.mark.parametrize(
    ("items", "expected"),
    [
        (["1.01", "2.01"], "m_and_a"),  # m_and_a > material_agreement
        (["4.02", "2.02"], "regulatory"),  # regulatory > financial_results
        (["5.02", "1.01"], "executive_change"),  # executive_change > material_agreement
        (["2.02", "5.02"], "financial_results"),  # financial_results > executive_change
        (["7.01", "5.02"], "executive_change"),  # executive_change > other
        (["2.01", "4.02", "5.02"], "m_and_a"),  # m_and_a tops the table
    ],
)
def test_assign_primary_category_priority(items: list[str], expected: str) -> None:
    assert assign_primary_category(items) == expected


def test_assign_primary_category_only_9_01_raises() -> None:
    with pytest.raises(ValueError):
        assign_primary_category(["9.01"])


def test_assign_primary_category_empty_raises() -> None:
    with pytest.raises(ValueError):
        assign_primary_category([])


def test_assign_primary_item_lowest_in_category() -> None:
    assert assign_primary_item(["3.03", "3.01"], "regulatory") == "3.01"


def test_assign_primary_item_skips_other_categories() -> None:
    # 5.02 is executive_change; lowest within financial_results is 2.02.
    assert assign_primary_item(["5.02", "2.02"], "financial_results") == "2.02"


def test_assign_primary_item_ignores_9_01() -> None:
    assert assign_primary_item(["2.02", "9.01"], "financial_results") == "2.02"


def test_assign_primary_item_raises_when_no_match() -> None:
    with pytest.raises(ValueError):
        assign_primary_item(["7.01"], "m_and_a")


def test_category_priority_covers_all_categories() -> None:
    assert set(CATEGORY_PRIORITY) == set(ITEM_TO_CATEGORY.values()) | {"material_agreement"}


# ---- orchestrator end-to-end ----------------------------------------------------


_PRIMARY_DOCS = {
    "0000320193-24-000123": ("8-K", "sample_earnings.html", "0000320193"),
    "0000789019-24-000456": ("8-K", "sample_ceo.html", "0000789019"),
    "0000123456-24-000789": ("8-K", "sample_merger.html", "0000123456"),
    "0000654321-24-000111": ("8-K/A", "sample_amendment.html", "0000654321"),
}
_SUBMISSIONS = {
    "0000320193": {"name": "Apple Inc.", "tickers": ["AAPL"], "exchanges": ["Nasdaq"]},
    "0000789019": {"name": "Microsoft Corporation", "tickers": ["MSFT"], "exchanges": ["Nasdaq"]},
    "0000123456": {"name": "Acme Holdings Inc", "tickers": [], "exchanges": []},
    "0000654321": {"name": "Beta Industries Corp", "tickers": ["BETA"], "exchanges": ["NYSE"]},
}


def _wrap_sgml(form_type: str, html: str) -> str:
    return (
        "<SEC-DOCUMENT>fake.txt : 20240502\n"
        "<SEC-HEADER>\n"
        "<ACCEPTANCE-DATETIME>20240502163000\n"
        "</SEC-HEADER>\n"
        "<DOCUMENT>\n"
        f"<TYPE>{form_type}\n"
        "<SEQUENCE>1\n"
        "<FILENAME>doc.htm\n"
        "<TEXT>\n"
        f"{html}\n"
        "</TEXT>\n"
        "</DOCUMENT>\n"
        "</SEC-DOCUMENT>\n"
    )


def _build_handler(
    edgar_fixtures_dir: Path,
) -> Callable[[httpx.Request], httpx.Response]:
    daily_text = (edgar_fixtures_dir / "daily_index_sample.idx").read_text()

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/daily-index/" in url:
            if "master.20240502.idx" in url:
                return httpx.Response(200, text=daily_text)
            return httpx.Response(404)

        if "/submissions/CIK" in url:
            cik = url.split("CIK")[1].split(".json")[0]
            data = _SUBMISSIONS.get(
                cik, {"name": "Unknown", "tickers": [], "exchanges": []}
            )
            return httpx.Response(200, json=data)

        for acc, (form_type, html_fixture, _cik_padded) in _PRIMARY_DOCS.items():
            if f"{acc}.txt" in url:
                html = (edgar_fixtures_dir / html_fixture).read_text()
                return httpx.Response(200, text=_wrap_sgml(form_type, html))

        return httpx.Response(404)

    return handler


def _client_for(handler: Callable[[httpx.Request], httpx.Response]) -> EdgarClient:
    return EdgarClient(
        user_agent="Test User test@example.com",
        rps=100000.0,
        limiter=RateLimiter(rps=100000.0),
        transport=httpx.MockTransport(handler),
    )


def test_ingest_writes_one_file_per_filing(
    edgar_fixtures_dir: Path, tmp_path: Path, no_sleep: None
) -> None:
    with _client_for(_build_handler(edgar_fixtures_dir)) as client:
        summary = ingest(
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 2),
            out_dir=tmp_path,
            client=client,
            limit=10,
        )
    json_files = sorted(tmp_path.glob("*.json"))
    assert len(json_files) == 4
    manifest_lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
    assert len(manifest_lines) == 4
    assert sum(summary.written.values()) == 4
    assert not summary.errors


def test_ingest_records_expected_categories(
    edgar_fixtures_dir: Path, tmp_path: Path, no_sleep: None
) -> None:
    with _client_for(_build_handler(edgar_fixtures_dir)) as client:
        ingest(
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 2),
            out_dir=tmp_path,
            client=client,
        )
    cats: set[str] = set()
    for line in (tmp_path / "manifest.jsonl").read_text().splitlines():
        cats.add(json.loads(line)["primary_category"])
    assert cats == {"financial_results", "executive_change", "m_and_a", "regulatory"}


def test_ingest_metadata_complete(
    edgar_fixtures_dir: Path, tmp_path: Path, no_sleep: None
) -> None:
    with _client_for(_build_handler(edgar_fixtures_dir)) as client:
        ingest(
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 2),
            out_dir=tmp_path,
            client=client,
        )
    aapl = json.loads((tmp_path / "000032019324000123.json").read_text())
    md = aapl["metadata"]
    assert md["cik"] == "0000320193"
    assert md["ticker"] == "AAPL"
    assert md["form_type"] == "8-K"
    assert md["items"] == ["2.02", "9.01"]
    assert md["primary_category"] == "financial_results"
    assert md["primary_item"] == "2.02"
    assert md["accession_number"] == "0000320193-24-000123"
    assert "raw_html" in aapl
    assert "parsed_text" in aapl
    assert aapl["schema_version"] == "1"


def test_ingest_skips_existing_files(
    edgar_fixtures_dir: Path, tmp_path: Path, no_sleep: None
) -> None:
    # Pre-create one file with valid metadata so reconcile picks it up.
    preexisting = tmp_path / "000032019324000123.json"
    preexisting.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fetched_at": "2024-05-02T00:00:00Z",
                "metadata": {
                    "accession_number": "0000320193-24-000123",
                    "accession_number_nodash": "000032019324000123",
                    "cik": "0000320193",
                    "filer_name": "Apple Inc.",
                    "ticker": "AAPL",
                    "form_type": "8-K",
                    "filing_date": "2024-05-02",
                    "items": ["2.02", "9.01"],
                    "primary_item": "2.02",
                    "primary_category": "financial_results",
                },
                "raw_html": "",
                "parsed_text": "stub",
            }
        )
    )

    with _client_for(_build_handler(edgar_fixtures_dir)) as client:
        summary = ingest(
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 2),
            out_dir=tmp_path,
            client=client,
        )
    # Existing file untouched; 3 new ones written.
    assert "stub" in (tmp_path / "000032019324000123.json").read_text()
    assert sum(summary.written.values()) == 3
    assert summary.skipped_existing >= 1


def test_ingest_respects_quota(
    edgar_fixtures_dir: Path, tmp_path: Path, no_sleep: None
) -> None:
    # Cap all categories at 0 except executive_change -> only the CEO filing is written.
    quotas = {
        "m_and_a": 0,
        "executive_change": 1,
        "financial_results": 0,
        "material_agreement": 0,
        "regulatory": 0,
        "other": 0,
    }
    with _client_for(_build_handler(edgar_fixtures_dir)) as client:
        summary = ingest(
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 2),
            out_dir=tmp_path,
            client=client,
            quotas=quotas,
        )
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    md = json.loads(json_files[0].read_text())["metadata"]
    assert md["primary_category"] == "executive_change"
    assert summary.skipped_quota >= 1


def test_ingest_short_circuits_when_all_quotas_met(
    edgar_fixtures_dir: Path, tmp_path: Path, no_sleep: None
) -> None:
    quotas = {c: 0 for c in (
        "m_and_a", "executive_change", "financial_results",
        "material_agreement", "regulatory", "other",
    )}
    with _client_for(_build_handler(edgar_fixtures_dir)) as client:
        summary = ingest(
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 2),
            out_dir=tmp_path,
            client=client,
            quotas=quotas,
        )
    assert sum(summary.written.values()) == 0


def test_ingest_dry_run_writes_nothing(
    edgar_fixtures_dir: Path, tmp_path: Path, no_sleep: None, capsys: pytest.CaptureFixture[str]
) -> None:
    with _client_for(_build_handler(edgar_fixtures_dir)) as client:
        summary = ingest(
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 2),
            out_dir=tmp_path,
            client=client,
            dry_run=True,
        )
    assert sum(summary.written.values()) == 0
    assert list(tmp_path.glob("*.json")) == []
    out = capsys.readouterr().out
    assert "metadata" in out


def test_ingest_reconciles_orphan_file(
    edgar_fixtures_dir: Path, tmp_path: Path, no_sleep: None
) -> None:
    # File on disk but no manifest row -> reconcile must append a manifest row.
    orphan = tmp_path / "000032019324000123.json"
    orphan.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fetched_at": "2024-05-02T00:00:00Z",
                "metadata": {
                    "accession_number": "0000320193-24-000123",
                    "cik": "0000320193",
                    "filer_name": "Apple Inc.",
                    "ticker": "AAPL",
                    "filing_date": "2024-05-02",
                    "items": ["2.02", "9.01"],
                    "primary_category": "financial_results",
                },
                "parsed_text": "stub",
            }
        )
    )
    with _client_for(_build_handler(edgar_fixtures_dir)) as client:
        ingest(
            start_date=date(2024, 5, 2),
            end_date=date(2024, 5, 2),
            out_dir=tmp_path,
            client=client,
        )
    rows = [
        json.loads(line)
        for line in (tmp_path / "manifest.jsonl").read_text().splitlines()
    ]
    accs = [r["accession"] for r in rows]
    assert accs.count("0000320193-24-000123") == 1


# ---- CLI ------------------------------------------------------------------------


def test_cli_rejects_missing_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 2
    combined = result.output + getattr(result, "stderr", "")
    assert "EDGAR_USER_AGENT" in combined


def test_cli_rejects_start_after_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_USER_AGENT", "Test test@example.com")
    result = CliRunner().invoke(
        app, ["--start-date", "2024-05-10", "--end-date", "2024-05-01"]
    )
    assert result.exit_code == 2


def test_cli_rejects_window_over_365_days(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDGAR_USER_AGENT", "Test test@example.com")
    result = CliRunner().invoke(
        app, ["--start-date", "2020-01-01", "--end-date", "2024-01-01"]
    )
    assert result.exit_code == 2
