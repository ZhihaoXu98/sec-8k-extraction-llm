"""CLI end-to-end tests for scripts/llm_label_gold_set.py.

Mocks the Anthropic SDK via monkeypatch so no real API calls are made; exercises
the typer CLI with a tmp_path filesystem. Covers prompt construction, response
parsing, validation retry, provenance recording, skip-if-already-labeled logic,
and the --dry-run path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from typer.testing import CliRunner

from sec8k.data import llm_label as llm_label_mod
from sec8k.data.llm_label import app

# ---------------------------------------------------------------------------
# Helpers + fake Anthropic
# ---------------------------------------------------------------------------


_VALID_LABEL: dict[str, Any] = {
    "form_type": "8-K",
    "filer_company": "Test Corp",
    "filer_ticker": None,
    "filer_cik": "0000123456",
    "filing_date": "2025-04-30",
    "event_date": None,
    "items": ["2.02"],
    "primary_category": "financial_results",
    "counterparties": [],
    "monetary_amount": None,
    "currency": None,
    "amount_type": None,
    "summary": "Smoke-test label.",
    "expected_impact_period": None,
}


def _make_response(
    *,
    tool_input: dict[str, Any] | None = None,
    text: str | None = None,
    cache_hit: bool = False,
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
            input_tokens=500,
            output_tokens=600,
            cache_creation_input_tokens=0 if cache_hit else 10_000,
            cache_read_input_tokens=10_000 if cache_hit else 0,
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


def _patch_anthropic(monkeypatch: pytest.MonkeyPatch, fake: FakeAnthropic) -> None:
    """Make `from anthropic import Anthropic` inside label_batch return our fake."""
    import anthropic

    def factory(*_args: Any, **_kwargs: Any) -> FakeAnthropic:
        return fake

    monkeypatch.setattr(anthropic, "Anthropic", factory)


def _make_filing(
    in_dir: Path,
    accession: str,
    category: str = "financial_results",
    parsed_text: str = "Item 2.02. On 2025-04-30 Test Corp announced earnings.",
) -> Path:
    no_dash = accession.replace("-", "")
    in_dir.mkdir(parents=True, exist_ok=True)
    path = in_dir / f"{no_dash}.json"
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
                    "acceptance_datetime": "2025-04-30T16:30:00",
                    "items": ["2.02"],
                },
                "raw_html": "<html></html>",
                "parsed_text": parsed_text,
            }
        )
    )
    return path


def _write_manifest(in_dir: Path, by_cat: dict[str, list[str]]) -> None:
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


def _setup_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    in_dir = tmp_path / "raw"
    gold = tmp_path / "gold" / "v1.jsonl"
    failures = tmp_path / "gold" / "failures.jsonl"
    guidelines = tmp_path / "guidelines.md"
    guidelines.write_text("# Labeling guidelines\n\nReturn a Filing8K JSON.\n")
    return in_dir, gold, failures, guidelines


def _cli_args(
    *,
    in_dir: Path,
    out: Path,
    failures: Path,
    guidelines: Path,
    extra: list[str] | None = None,
) -> list[str]:
    args = [
        "--in-dir",
        str(in_dir),
        "--out",
        str(out),
        "--failures",
        str(failures),
        "--guidelines",
        str(guidelines),
    ]
    if extra:
        args.extend(extra)
    return args


# ---------------------------------------------------------------------------
# Original argument-validation tests (unchanged)
# ---------------------------------------------------------------------------


def test_cli_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["--limit", "1"])
    assert result.exit_code == 2
    combined = result.output + getattr(result, "stderr", "")
    assert "ANTHROPIC_API_KEY" in combined


def test_cli_rejects_accessions_and_limit_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    result = CliRunner().invoke(app, ["--limit", "1", "--accessions", "0000111-25-000001"])
    assert result.exit_code == 2


def test_cli_rejects_neither_limit_nor_accessions_nor_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    result = CliRunner().invoke(app, [])
    assert result.exit_code == 2


def test_cli_help_lists_all_flags() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for flag in [
        "--in-dir",
        "--out",
        "--failures",
        "--guidelines",
        "--model",
        "--limit",
        "--accessions",
        "--seed",
        "--max-cost-usd",
        "--dry-run",
        "--rerun",
        "--api-key",
        "--log-level",
    ]:
        assert flag in result.output, f"missing {flag} in --help"


# ---------------------------------------------------------------------------
# CLI end-to-end with mocked Anthropic
# ---------------------------------------------------------------------------


def test_cli_prompt_construction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_paths(tmp_path)
    _make_filing(in_dir, "0000111-25-000001", parsed_text="MY_UNIQUE_FILING_TOKEN inside the body")
    _write_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})

    fake = FakeAnthropic([_make_response(tool_input=_VALID_LABEL)])
    _patch_anthropic(monkeypatch, fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    result = CliRunner().invoke(
        app,
        _cli_args(
            in_dir=in_dir,
            out=gold,
            failures=failures,
            guidelines=guidelines,
            extra=["--accessions", "0000111-25-000001"],
        ),
    )
    assert result.exit_code == 0, result.output

    assert len(fake.messages.calls) == 1
    kwargs = fake.messages.calls[0]

    # System prompt is a single text block carrying the guidelines doc with cache_control.
    assert isinstance(kwargs["system"], list) and len(kwargs["system"]) == 1
    sys_block = kwargs["system"][0]
    assert sys_block["type"] == "text"
    assert sys_block["text"] == guidelines.read_text()
    assert sys_block["cache_control"] == {"type": "ephemeral"}

    # Tool definition uses Filing8K schema and is forced via tool_choice.
    assert kwargs["tools"][0]["name"] == llm_label_mod.TOOL_NAME
    assert kwargs["tool_choice"] == {"type": "tool", "name": llm_label_mod.TOOL_NAME}

    # User message references the accession and contains the filing's parsed text.
    user_turn = kwargs["messages"][0]
    assert user_turn["role"] == "user"
    user_text = user_turn["content"][0]["text"]
    assert "0000111-25-000001" in user_text
    assert "MY_UNIQUE_FILING_TOKEN" in user_text
    # v2 prompt: expose EDGAR ACCEPTANCE_DATETIME so the LLM can use it as filing_date.
    assert "ACCEPTANCE_DATETIME:" in user_text
    assert "2025-04-30T16:30:00" in user_text


def test_cli_response_parsing_writes_filing8k_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    in_dir, gold, failures, guidelines = _setup_paths(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _write_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})

    fake = FakeAnthropic([_make_response(tool_input=_VALID_LABEL)])
    _patch_anthropic(monkeypatch, fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    result = CliRunner().invoke(
        app,
        _cli_args(
            in_dir=in_dir,
            out=gold,
            failures=failures,
            guidelines=guidelines,
            extra=["--accessions", "0000111-25-000001"],
        ),
    )
    assert result.exit_code == 0, result.output

    rows = [json.loads(line) for line in gold.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    row = rows[0]
    assert row["accession"] == "0000111-25-000001"
    # label round-trips through Filing8K:
    from sec8k.schema import Filing8K

    Filing8K.model_validate(row["label"])


def test_cli_validation_retry_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_paths(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _write_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})

    bad = {**_VALID_LABEL, "form_type": "8-K/B"}  # unknown form_type
    fake = FakeAnthropic(
        [
            _make_response(tool_input=bad),
            _make_response(tool_input=_VALID_LABEL, cache_hit=True),
        ]
    )
    _patch_anthropic(monkeypatch, fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    result = CliRunner().invoke(
        app,
        _cli_args(
            in_dir=in_dir,
            out=gold,
            failures=failures,
            guidelines=guidelines,
            extra=["--accessions", "0000111-25-000001"],
        ),
    )
    assert result.exit_code == 0, result.output
    assert len(fake.messages.calls) == 2

    # Second call's last user turn must be a tool_result with is_error=True.
    second_call = fake.messages.calls[1]
    last_user = next(m for m in reversed(second_call["messages"]) if m["role"] == "user")
    tool_result_block = next(
        b for b in last_user["content"] if isinstance(b, dict) and b.get("type") == "tool_result"
    )
    assert tool_result_block["is_error"] is True
    assert "validation" in tool_result_block["content"].lower()

    # Final gold row records retried=True in provenance.
    row = json.loads(gold.read_text().splitlines()[0])
    assert row["provenance"]["retried"] is True


def test_cli_provenance_recording(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_paths(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _write_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})

    fake = FakeAnthropic([_make_response(tool_input=_VALID_LABEL)])
    _patch_anthropic(monkeypatch, fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    result = CliRunner().invoke(
        app,
        _cli_args(
            in_dir=in_dir,
            out=gold,
            failures=failures,
            guidelines=guidelines,
            extra=["--accessions", "0000111-25-000001"],
        ),
    )
    assert result.exit_code == 0, result.output

    row = json.loads(gold.read_text().splitlines()[0])
    prov = row["provenance"]
    assert prov["model"] == llm_label_mod.MODEL_DEFAULT
    assert prov["guidelines_sha"] == llm_label_mod.compute_guidelines_sha(guidelines)
    assert prov["prompt_version"] == llm_label_mod.PROMPT_VERSION
    assert prov["labeled_at"].endswith("Z")
    assert prov["input_tokens"] > 0
    assert prov["output_tokens"] > 0
    assert prov["cost_usd"] > 0
    assert "retried" in prov


def test_cli_skip_if_already_labeled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    in_dir, gold, failures, guidelines = _setup_paths(tmp_path)
    _make_filing(in_dir, "0000111-25-000001")
    _write_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})

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

    fake = FakeAnthropic([])  # no API calls expected
    _patch_anthropic(monkeypatch, fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    result = CliRunner().invoke(
        app,
        _cli_args(
            in_dir=in_dir,
            out=gold,
            failures=failures,
            guidelines=guidelines,
            extra=["--accessions", "0000111-25-000001"],
        ),
    )
    assert result.exit_code == 0, result.output
    assert len(fake.messages.calls) == 0
    assert "Skipped (current SHA): 1" in result.output


def test_cli_dry_run_single_filing_no_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    in_dir, gold, failures, guidelines = _setup_paths(tmp_path)
    _make_filing(in_dir, "0000111-25-000001", parsed_text="UNIQUE_DRYRUN_TOKEN_XYZ")
    _write_manifest(in_dir, {"financial_results": ["0000111-25-000001"]})

    fake = FakeAnthropic([_make_response(tool_input=_VALID_LABEL)])
    _patch_anthropic(monkeypatch, fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")

    result = CliRunner().invoke(
        app,
        _cli_args(
            in_dir=in_dir,
            out=gold,
            failures=failures,
            guidelines=guidelines,
            extra=["--dry-run", "--accessions", "0000111-25-000001"],
        ),
    )
    assert result.exit_code == 0, result.output

    # API was called once; nothing written to disk.
    assert len(fake.messages.calls) == 1
    assert not gold.exists()
    assert not failures.exists()

    # Dry-run prints the prompt + response sections to stdout.
    out = result.output
    assert "--- system prompt" in out
    assert "--- user prompt" in out
    assert "--- response content" in out
    assert "UNIQUE_DRYRUN_TOKEN_XYZ" in out  # user prompt contains the filing body
    assert "emit_filing8k" in out  # response shows the tool call
