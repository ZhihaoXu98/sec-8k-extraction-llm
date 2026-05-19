"""Tests for sec8k.data.llm_label.

Uses a FakeAnthropic client with scripted responses constructed from
``anthropic.types.Message`` — no real network. Cheap, fully deterministic.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import typer
from anthropic import RateLimitError
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from sec8k.data import llm_label as llm_label_mod

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


VALID_LABEL: dict[str, Any] = {
    "form_type": "8-K",
    "filer_company": "Test Corp",
    "filer_ticker": "TST",
    "filer_cik": "0000123456",
    "filing_date": "2025-04-30",
    "event_date": "2025-04-29",
    "items": ["2.02", "9.01"],
    "primary_category": "financial_results",
    "counterparties": [],
    "monetary_amount": None,
    "currency": None,
    "amount_type": None,
    "summary": "Test Corp announced Q1 results.",
    "expected_impact_period": "current_quarter",
}


def _make_response(
    *,
    tool_input: dict[str, Any] | None = None,
    text: str | None = None,
    cache_hit: bool = False,
    input_tokens: int = 500,
    output_tokens: int = 700,
    tool_use_id: str = "toolu_test",
) -> Message:
    content: list[Any] = []
    if tool_input is not None:
        content.append(
            ToolUseBlock(
                type="tool_use",
                id=tool_use_id,
                name=llm_label_mod.TOOL_NAME,
                input=tool_input,
            )
        )
    if text is not None:
        content.append(TextBlock(type="text", text=text, citations=None))
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model=llm_label_mod.MODEL_DEFAULT,
        content=content,
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0 if cache_hit else 10_300,
            cache_read_input_tokens=10_300 if cache_hit else 0,
            cache_creation=None,
            inference_geo=None,
            server_tool_use=None,
            service_tier=None,
        ),
    )


class _FakeMessages:
    def __init__(self, queue: list[Message | Exception]) -> None:
        self._queue = list(queue)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if not self._queue:
            raise RuntimeError("FakeAnthropic queue exhausted")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeAnthropic:
    def __init__(self, scripted: list[Message | Exception]) -> None:
        self.messages = _FakeMessages(scripted)


def _make_rate_limit_error() -> RateLimitError:
    import httpx

    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        429, request=request, content=b'{"error":{"type":"rate_limit_error","message":"slow down"}}'
    )
    return RateLimitError(
        "rate limit", response=response, body={"error": {"type": "rate_limit_error"}}
    )


def _make_filing(
    in_dir: Path,
    accession: str,
    category: str = "financial_results",
    parsed_text: str = (
        "Item 2.02. Results of Operations.\n"
        "On 2025-04-30 the company announced earnings."
    ),
) -> Path:
    no_dash = accession.replace("-", "")
    path = in_dir / f"{no_dash}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fetched_at": "2025-04-30T00:00:00Z",
                "metadata": {
                    "accession_number": accession,
                    "accession_number_nodash": no_dash,
                    "cik": "0000123456",
                    "filer_name": "Test Corp",
                    "ticker": "TST",
                    "primary_category": category,
                    "filing_date": "2025-04-30",
                    "items": ["2.02", "9.01"],
                },
                "raw_html": "<html></html>",
                "parsed_text": parsed_text,
            }
        )
    )
    return path


def _make_manifest(in_dir: Path, by_cat: dict[str, list[str]]) -> None:
    in_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for cat, accs in by_cat.items():
        for acc in accs:
            rows.append(
                json.dumps(
                    {
                        "accession": acc,
                        "primary_category": cat,
                        "filing_date": "2025-04-30",
                    }
                )
            )
    (in_dir / "manifest.jsonl").write_text("\n".join(rows) + "\n")


@pytest.fixture
def guidelines_file(tmp_path: Path) -> Path:
    path = tmp_path / "labeling_guidelines.md"
    path.write_text("# Labeling guidelines\n\nFollow these rules.\n")
    return path


# ---------------------------------------------------------------------------
# Pure-helper tests
# ---------------------------------------------------------------------------


def test_compute_guidelines_sha_matches_known_format(tmp_path: Path) -> None:
    path = tmp_path / "g.md"
    path.write_bytes(b"hello\n")
    expected = hashlib.sha1(b"blob 6\0hello\n").hexdigest()
    assert llm_label_mod.compute_guidelines_sha(path) == expected


def test_compute_guidelines_sha_matches_git_hash_object(tmp_path: Path) -> None:
    path = tmp_path / "g.md"
    path.write_text("test content\n")
    try:
        proc = subprocess.run(
            ["git", "hash-object", str(path)], capture_output=True, text=True, check=True
        )
    except FileNotFoundError:
        pytest.skip("git not installed")
    assert llm_label_mod.compute_guidelines_sha(path) == proc.stdout.strip()


def test_filing8k_input_schema_inlines_defs() -> None:
    schema = llm_label_mod._filing8k_input_schema()
    serialized = json.dumps(schema)
    assert "$ref" not in serialized
    assert "$defs" not in serialized
    assert schema["type"] == "object"
    expected_required = {
        "form_type",
        "filer_company",
        "filer_cik",
        "filing_date",
        "items",
        "primary_category",
        "summary",
    }
    assert expected_required <= set(schema["required"])


def test_compute_call_cost_known_usage() -> None:
    cost = llm_label_mod.compute_call_cost(
        input_tokens=1_000,
        output_tokens=500,
        cache_creation_input_tokens=10_000,
        cache_read_input_tokens=0,
        model="claude-sonnet-4-6",
    )
    # 1000 * 3/M + 500 * 15/M + 10000 * 3.75/M + 0
    expected = (1000 * 3.0 + 500 * 15.0 + 10000 * 3.75) / 1_000_000
    assert cost == pytest.approx(expected, rel=1e-9)


def test_compute_call_cost_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        llm_label_mod.compute_call_cost(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            model="nope",
        )


def test_estimate_cost_shape(tmp_path: Path, guidelines_file: Path) -> None:
    f1 = _make_filing(tmp_path, "0000111111-25-000001", parsed_text="x" * 4000)
    f2 = _make_filing(tmp_path, "0000111111-25-000002", parsed_text="y" * 4000)
    est = llm_label_mod.estimate_cost(
        [f1, f2], model="claude-sonnet-4-6", guidelines_path=guidelines_file
    )
    assert est.n_filings == 2
    assert est.est_cache_write_tokens > 0
    assert est.est_cache_read_tokens > 0
    assert est.est_total_usd > 0


def test_estimate_cost_empty() -> None:
    est = llm_label_mod.estimate_cost([], model="claude-sonnet-4-6")
    assert est.n_filings == 0
    assert est.est_total_usd == 0.0


def test_load_latest_sha_per_accession_latest_wins(tmp_path: Path) -> None:
    gold = tmp_path / "v1.jsonl"
    gold.write_text(
        "\n".join(
            [
                json.dumps({"accession": "A", "provenance": {"guidelines_sha": "old"}}),
                json.dumps({"accession": "B", "provenance": {"guidelines_sha": "x"}}),
                json.dumps({"accession": "A", "provenance": {"guidelines_sha": "new"}}),
            ]
        )
    )
    m = llm_label_mod.load_latest_sha_per_accession(gold)
    assert m == {"A": "new", "B": "x"}


def test_load_latest_sha_per_accession_missing_file_empty(tmp_path: Path) -> None:
    assert llm_label_mod.load_latest_sha_per_accession(tmp_path / "missing.jsonl") == {}


def test_load_latest_sha_per_accession_skips_corrupt_lines(tmp_path: Path) -> None:
    gold = tmp_path / "v1.jsonl"
    gold.write_text(
        '{"accession":"A","provenance":{"guidelines_sha":"good"}}\n'
        "not valid json\n"
        '{"accession":"B","provenance":{"guidelines_sha":"x"}}\n'
    )
    assert llm_label_mod.load_latest_sha_per_accession(gold) == {"A": "good", "B": "x"}


def test_select_filings_stratified_deterministic(tmp_path: Path) -> None:
    _make_manifest(
        tmp_path,
        {
            "m_and_a": [f"0000000001-25-{i:06d}" for i in range(100)],
            "executive_change": [f"0000000002-25-{i:06d}" for i in range(100)],
            "financial_results": [f"0000000003-25-{i:06d}" for i in range(100)],
            "material_agreement": [f"0000000004-25-{i:06d}" for i in range(100)],
            "regulatory": [f"0000000005-25-{i:06d}" for i in range(100)],
            "other": [f"0000000006-25-{i:06d}" for i in range(100)],
        },
    )
    picked1 = llm_label_mod.select_filings(tmp_path, limit=60, explicit_accessions=None, seed=42)
    picked2 = llm_label_mod.select_filings(tmp_path, limit=60, explicit_accessions=None, seed=42)
    assert picked1 == picked2  # deterministic
    assert len(picked1) == 60
    picked_seed_other = llm_label_mod.select_filings(
        tmp_path, limit=60, explicit_accessions=None, seed=99
    )
    assert picked_seed_other != picked1


def test_select_filings_explicit_accessions_takes_precedence(tmp_path: Path) -> None:
    paths = llm_label_mod.select_filings(
        tmp_path,
        limit=50,  # ignored
        explicit_accessions=["0000111-25-000001", "0000111-25-000002"],
    )
    assert [p.name for p in paths] == ["000011125000001.json", "000011125000002.json"]


def test_select_filings_short_category_warns_and_truncates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _make_manifest(
        tmp_path,
        {
            "m_and_a": [],
            "executive_change": [],
            "financial_results": [f"0000000003-25-{i:06d}" for i in range(2)],
            "material_agreement": [],
            "regulatory": [],
            "other": [],
        },
    )
    caplog.set_level("WARNING")
    picked = llm_label_mod.select_filings(tmp_path, limit=60, explicit_accessions=None, seed=42)
    # Only financial_results has 2; we take all 2, the rest produce 0.
    assert len(picked) == 2
    assert "quota is" in caplog.text


def test_select_filings_neither_arg_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        llm_label_mod.select_filings(tmp_path, limit=None, explicit_accessions=None)


# ---------------------------------------------------------------------------
# label_filing tests
# ---------------------------------------------------------------------------


def test_label_filing_success(tmp_path: Path, guidelines_file: Path) -> None:
    filing = _make_filing(tmp_path, "0000111-25-000001")
    client = FakeAnthropic([_make_response(tool_input=VALID_LABEL)])
    outcome = llm_label_mod.label_filing(
        client=client,
        filing_path=filing,
        guidelines=guidelines_file.read_text(),
        guidelines_sha="abc123",
        model="claude-sonnet-4-6",
    )
    assert isinstance(outcome, llm_label_mod.LabelResult)
    assert outcome.label.filer_company == "Test Corp"
    assert outcome.cost_usd > 0
    assert outcome.retried is False
    assert len(client.messages.calls) == 1


def test_label_filing_validation_retry_succeeds(tmp_path: Path, guidelines_file: Path) -> None:
    filing = _make_filing(tmp_path, "0000111-25-000001")
    bad = {**VALID_LABEL, "form_type": "8-K/B"}  # invalid form_type
    client = FakeAnthropic(
        [
            _make_response(tool_input=bad),
            _make_response(tool_input=VALID_LABEL, cache_hit=True),
        ]
    )
    outcome = llm_label_mod.label_filing(
        client=client,
        filing_path=filing,
        guidelines=guidelines_file.read_text(),
        guidelines_sha="abc",
        model="claude-sonnet-4-6",
    )
    assert isinstance(outcome, llm_label_mod.LabelResult)
    assert outcome.retried is True
    assert len(client.messages.calls) == 2
    # Second call must include a tool_result block with is_error=True.
    second = client.messages.calls[1]
    assert any(
        any(
            b.get("type") == "tool_result" and b.get("is_error") is True
            for b in (turn.get("content") or [])
            if isinstance(b, dict)
        )
        for turn in second["messages"]
        if turn.get("role") == "user"
    )


def test_label_filing_validation_retry_exhausted(tmp_path: Path, guidelines_file: Path) -> None:
    filing = _make_filing(tmp_path, "0000111-25-000001")
    bad = {**VALID_LABEL, "form_type": "8-K/B"}
    client = FakeAnthropic([_make_response(tool_input=bad), _make_response(tool_input=bad)])
    outcome = llm_label_mod.label_filing(
        client=client,
        filing_path=filing,
        guidelines=guidelines_file.read_text(),
        guidelines_sha="abc",
        model="claude-sonnet-4-6",
    )
    assert isinstance(outcome, llm_label_mod.LabelFailure)
    assert "ValidationError" in outcome.error
    assert outcome.cost_usd > 0


def test_label_filing_rate_limit_retries(
    tmp_path: Path, guidelines_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda _s: None)
    filing = _make_filing(tmp_path, "0000111-25-000001")
    client = FakeAnthropic(
        [
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_response(tool_input=VALID_LABEL),
        ]
    )
    outcome = llm_label_mod.label_filing(
        client=client,
        filing_path=filing,
        guidelines=guidelines_file.read_text(),
        guidelines_sha="abc",
        model="claude-sonnet-4-6",
    )
    assert isinstance(outcome, llm_label_mod.LabelResult)
    assert len(client.messages.calls) == 3


def test_label_filing_no_tool_use_block_fails(tmp_path: Path, guidelines_file: Path) -> None:
    filing = _make_filing(tmp_path, "0000111-25-000001")
    client = FakeAnthropic([_make_response(text="I cannot label this.")])
    outcome = llm_label_mod.label_filing(
        client=client,
        filing_path=filing,
        guidelines=guidelines_file.read_text(),
        guidelines_sha="abc",
        model="claude-sonnet-4-6",
    )
    assert isinstance(outcome, llm_label_mod.LabelFailure)
    assert "no tool_use" in outcome.error


def test_label_filing_truncates_oversize(
    tmp_path: Path, guidelines_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    huge = "X" * (llm_label_mod.MAX_INPUT_CHARS + 10_000)
    filing = _make_filing(tmp_path, "0000111-25-000001", parsed_text=huge)
    client = FakeAnthropic([_make_response(tool_input=VALID_LABEL)])
    caplog.set_level("WARNING")
    outcome = llm_label_mod.label_filing(
        client=client,
        filing_path=filing,
        guidelines=guidelines_file.read_text(),
        guidelines_sha="abc",
        model="claude-sonnet-4-6",
    )
    assert isinstance(outcome, llm_label_mod.LabelResult)
    assert "TRUNCATED" in caplog.text or "truncating" in caplog.text
    sent = client.messages.calls[0]["messages"][0]["content"][0]["text"]
    assert "[...TRUNCATED]" in sent


# ---------------------------------------------------------------------------
# label_batch orchestrator tests
# ---------------------------------------------------------------------------


def _setup_batch(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    in_dir = tmp_path / "raw"
    gold = tmp_path / "gold" / "v1.jsonl"
    failures = tmp_path / "gold" / "failures.jsonl"
    guidelines = tmp_path / "guidelines.md"
    guidelines.write_text("# Rules\n")
    return in_dir, gold, failures, guidelines


def test_label_batch_writes_label_and_provenance(tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_batch(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _make_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})

    client = FakeAnthropic([_make_response(tool_input=VALID_LABEL)])
    summary = llm_label_mod.label_batch(
        in_dir=in_dir,
        gold_path=gold,
        failures_path=failures,
        guidelines_path=guidelines,
        model="claude-sonnet-4-6",
        limit=None,
        explicit_accessions=["0000111-25-000001"],
        seed=42,
        max_cost_usd=100.0,
        dry_run=False,
        rerun=False,
        api_key="fake",
        client=client,
    )
    assert summary.n_succeeded == 1
    assert summary.n_failed == 0
    row = json.loads(gold.read_text().splitlines()[0])
    assert row["accession"] == "0000111-25-000001"
    assert row["provenance"]["model"] == "claude-sonnet-4-6"
    assert row["provenance"]["guidelines_sha"] == llm_label_mod.compute_guidelines_sha(guidelines)
    assert row["provenance"]["cost_usd"] > 0


def test_label_batch_skip_if_current_sha(tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_batch(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _make_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})
    current_sha = llm_label_mod.compute_guidelines_sha(guidelines)
    gold.parent.mkdir(parents=True, exist_ok=True)
    gold.write_text(
        json.dumps(
            {
                "accession": "0000111-25-000001",
                "label": {},
                "provenance": {"guidelines_sha": current_sha},
            }
        )
        + "\n"
    )
    client = FakeAnthropic([])  # no calls expected
    summary = llm_label_mod.label_batch(
        in_dir=in_dir,
        gold_path=gold,
        failures_path=failures,
        guidelines_path=guidelines,
        model="claude-sonnet-4-6",
        limit=None,
        explicit_accessions=["0000111-25-000001"],
        seed=42,
        max_cost_usd=100.0,
        dry_run=False,
        rerun=False,
        api_key="fake",
        client=client,
    )
    assert summary.n_skipped == 1
    assert summary.n_attempted == 0
    assert len(client.messages.calls) == 0


def test_label_batch_rerun_forces_relabel(tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_batch(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _make_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})
    current_sha = llm_label_mod.compute_guidelines_sha(guidelines)
    gold.parent.mkdir(parents=True, exist_ok=True)
    gold.write_text(
        json.dumps(
            {
                "accession": "0000111-25-000001",
                "label": {},
                "provenance": {"guidelines_sha": current_sha},
            }
        )
        + "\n"
    )
    client = FakeAnthropic([_make_response(tool_input=VALID_LABEL)])
    summary = llm_label_mod.label_batch(
        in_dir=in_dir,
        gold_path=gold,
        failures_path=failures,
        guidelines_path=guidelines,
        model="claude-sonnet-4-6",
        limit=None,
        explicit_accessions=["0000111-25-000001"],
        seed=42,
        max_cost_usd=100.0,
        dry_run=False,
        rerun=True,
        api_key="fake",
        client=client,
    )
    assert summary.n_succeeded == 1
    lines = gold.read_text().splitlines()
    assert len(lines) == 2  # old + new


def test_label_batch_relabels_on_sha_change(tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_batch(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _make_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})
    gold.parent.mkdir(parents=True, exist_ok=True)
    gold.write_text(
        json.dumps(
            {
                "accession": "0000111-25-000001",
                "label": {},
                "provenance": {"guidelines_sha": "different-sha"},
            }
        )
        + "\n"
    )
    client = FakeAnthropic([_make_response(tool_input=VALID_LABEL)])
    summary = llm_label_mod.label_batch(
        in_dir=in_dir,
        gold_path=gold,
        failures_path=failures,
        guidelines_path=guidelines,
        model="claude-sonnet-4-6",
        limit=None,
        explicit_accessions=["0000111-25-000001"],
        seed=42,
        max_cost_usd=100.0,
        dry_run=False,
        rerun=False,
        api_key="fake",
        client=client,
    )
    assert summary.n_succeeded == 1


def test_label_batch_validation_failure_writes_failures(tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_batch(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _make_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})
    bad = {**VALID_LABEL, "form_type": "8-K/B"}
    client = FakeAnthropic([_make_response(tool_input=bad), _make_response(tool_input=bad)])
    summary = llm_label_mod.label_batch(
        in_dir=in_dir,
        gold_path=gold,
        failures_path=failures,
        guidelines_path=guidelines,
        model="claude-sonnet-4-6",
        limit=None,
        explicit_accessions=["0000111-25-000001"],
        seed=42,
        max_cost_usd=100.0,
        dry_run=False,
        rerun=False,
        api_key="fake",
        client=client,
    )
    assert summary.n_failed == 1
    assert summary.n_succeeded == 0
    assert not gold.exists()  # no successful row written
    failure_rows = failures.read_text().splitlines()
    assert len(failure_rows) == 1
    assert "ValidationError" in json.loads(failure_rows[0])["error"]


def test_label_batch_dry_run_no_writes(tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_batch(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _make_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})
    client = FakeAnthropic([_make_response(tool_input=VALID_LABEL)])
    summary = llm_label_mod.label_batch(
        in_dir=in_dir,
        gold_path=gold,
        failures_path=failures,
        guidelines_path=guidelines,
        model="claude-sonnet-4-6",
        limit=None,
        explicit_accessions=["0000111-25-000001"],
        seed=42,
        max_cost_usd=100.0,
        dry_run=True,
        rerun=False,
        api_key="fake",
        client=client,
    )
    assert summary.n_succeeded == 1
    assert not gold.exists()
    assert not failures.exists()
    # Real call was still made.
    assert len(client.messages.calls) == 1


def test_label_batch_cost_cap_pre_run_refuses(tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_batch(tmp_path)
    _make_filing(in_dir, "0000111-25-000001", parsed_text="x" * 4000)
    _make_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})
    client = FakeAnthropic([])
    with pytest.raises(typer.Exit) as exc_info:
        llm_label_mod.label_batch(
            in_dir=in_dir,
            gold_path=gold,
            failures_path=failures,
            guidelines_path=guidelines,
            model="claude-sonnet-4-6",
            limit=None,
            explicit_accessions=["0000111-25-000001"],
            seed=42,
            max_cost_usd=0.000001,
            dry_run=False,
            rerun=False,
            api_key="fake",
            client=client,
        )
    assert exc_info.value.exit_code == 2
    assert len(client.messages.calls) == 0


def test_label_batch_per_category_counts(tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_batch(tmp_path)
    cats = [
        "m_and_a",
        "executive_change",
        "financial_results",
        "material_agreement",
        "regulatory",
        "other",
    ]
    for i, c in enumerate(cats):
        acc = f"0000111-25-{i:06d}"
        _make_filing(in_dir, acc, category=c)
    _make_manifest(in_dir, {c: [f"0000111-25-{i:06d}"] for i, c in enumerate(cats)})

    client = FakeAnthropic([_make_response(tool_input=VALID_LABEL) for _ in cats])
    summary = llm_label_mod.label_batch(
        in_dir=in_dir,
        gold_path=gold,
        failures_path=failures,
        guidelines_path=guidelines,
        model="claude-sonnet-4-6",
        limit=6,
        explicit_accessions=None,
        seed=42,
        max_cost_usd=100.0,
        dry_run=False,
        rerun=False,
        api_key="fake",
        client=client,
    )
    assert summary.n_succeeded == 6
    assert all(v == 1 for v in summary.per_category_succeeded.values())
