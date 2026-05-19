"""CLI end-to-end tests for scripts/verify_calibration.py.

Argument-validation smoke tests + driven-stdin tests that exercise the real
TerminalPrompter's per-field keep/edit/ambig loop and confirm the provenance
preservation guarantees (llm_label_snapshot, fields_changed, fields_ambiguous,
verified_by/at/session_id, duration_seconds).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from sec8k.data.verify import app
from sec8k.schema import Filing8K

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_VALID_LABEL: dict[str, Any] = {
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

# Filing8K declaration order — the per-field prompts hit these in order.
_FIELD_ORDER = tuple(Filing8K.model_fields.keys())
_N_FIELDS = len(_FIELD_ORDER)
_PRIMARY_CATEGORY_IDX = _FIELD_ORDER.index("primary_category")  # 0-based
_EXPECTED_IMPACT_IDX = _FIELD_ORDER.index("expected_impact_period")


def _seed_one_filing(tmp_path: Path) -> tuple[Path, Path]:
    """Write one unverified gold row + corresponding filing JSON. Return (gold, raw_dir)."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accession = "0000111-25-000001"
    no_dash = accession.replace("-", "")
    (raw_dir / f"{no_dash}.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "metadata": {"accession_number": accession},
                "raw_html": "<html></html>",
                "parsed_text": "Item 2.02. Earnings.",
            }
        )
    )

    gold = tmp_path / "gold" / "v1.jsonl"
    gold.parent.mkdir(parents=True, exist_ok=True)
    gold.write_text(
        json.dumps(
            {
                "accession": accession,
                "label": _VALID_LABEL,
                "provenance": {
                    "model": "claude-sonnet-4-6",
                    "guidelines_sha": "sha-test",
                    "labeled_at": "2025-04-30T00:00:00Z",
                    "cost_usd": 0.02,
                    "retried": False,
                },
            }
        )
        + "\n"
    )
    return gold, raw_dir


def _cli_args(gold: Path, raw_dir: Path, *extra: str) -> list[str]:
    return [
        "--gold-path",
        str(gold),
        "--raw-dir",
        str(raw_dir),
        "--no-pager",
        "--session-id",
        "test-session",
        "--verifier",
        "tester",
        *extra,
    ]


def _last_row(gold: Path) -> dict[str, Any]:
    lines = [line for line in gold.read_text().splitlines() if line.strip()]
    row: dict[str, Any] = json.loads(lines[-1])
    return row


# ---------------------------------------------------------------------------
# Argument-validation smoke tests
# ---------------------------------------------------------------------------


def test_cli_help_lists_flags() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for flag in [
        "--gold-path",
        "--raw-dir",
        "--category",
        "--limit",
        "--session-id",
        "--verifier",
        "--seed",
        "--metrics-out",
        "--no-pager",
        "--log-level",
    ]:
        assert flag in result.output, f"missing {flag} in --help"


def test_cli_rejects_unknown_category(tmp_path: Path) -> None:
    gold = tmp_path / "v1.jsonl"
    gold.touch()
    result = CliRunner().invoke(app, ["--gold-path", str(gold), "--category", "not-real"])
    assert result.exit_code == 2
    combined = result.output + getattr(result, "stderr", "")
    assert "category" in combined.lower()


def test_cli_no_unverified_rows_exits_clean(tmp_path: Path) -> None:
    gold = tmp_path / "v1.jsonl"
    gold.write_text(
        json.dumps(
            {
                "accession": "A",
                "label": {
                    "form_type": "8-K",
                    "filer_company": "X",
                    "filer_cik": "0000000001",
                    "filing_date": "2025-01-01",
                    "items": ["2.02"],
                    "primary_category": "financial_results",
                    "summary": "x",
                },
                "provenance": {"verified_by": "alice", "verification_type": "human_review"},
            }
        )
        + "\n"
    )
    result = CliRunner().invoke(app, ["--gold-path", str(gold), "--raw-dir", str(tmp_path / "raw")])
    assert result.exit_code == 0
    assert "No unverified rows" in result.output


# ---------------------------------------------------------------------------
# Per-field keep/edit/ambig loop + provenance preservation
# ---------------------------------------------------------------------------


def test_cli_keep_all_preserves_label_and_records_snapshot(tmp_path: Path) -> None:
    """All 14 fields kept → label unchanged, full snapshot in provenance."""
    gold, raw_dir = _seed_one_filing(tmp_path)

    # Default for each prompt_field is "k"; empty input picks default.
    # 14 field prompts + 1 notes + 1 end-of-example save = 16 newlines.
    stdin = "\n" * (_N_FIELDS + 2)
    result = CliRunner().invoke(app, _cli_args(gold, raw_dir), input=stdin)
    assert result.exit_code == 0, result.output

    new_row = _last_row(gold)
    assert new_row["accession"] == "0000111-25-000001"
    # Label is unchanged.
    assert new_row["label"] == _VALID_LABEL
    prov = new_row["provenance"]
    # Provenance verification block populated.
    assert prov["verified_by"] == "tester"
    assert prov["verification_type"] == "human_review"
    assert "verifier" not in prov  # collapsed into verified_by
    assert prov["verification_session_id"] == "test-session"
    assert prov["verified_at"].endswith("Z")
    assert prov["fields_changed"] == []
    assert prov["fields_ambiguous"] == []
    assert prov["notes"] == ""
    assert prov["duration_seconds"] >= 0
    # Original LLM label preserved verbatim.
    assert prov["llm_label_snapshot"] == _VALID_LABEL


def test_cli_edit_primary_category_records_change_and_snapshot(tmp_path: Path) -> None:
    """Edit primary_category → label updated, original preserved as snapshot."""
    gold, raw_dir = _seed_one_filing(tmp_path)

    # Build stdin: keep all fields except primary_category (index = _PRIMARY_CATEGORY_IDX).
    lines: list[str] = []
    for i in range(_N_FIELDS):
        if i == _PRIMARY_CATEGORY_IDX:
            lines.append("e")  # edit action
            lines.append("1")  # 1-based index into Literal choices → "m_and_a"
        else:
            lines.append("")  # default "k"
    lines.append("reclassified per body read")  # notes
    lines.append("")  # default "s" (save)
    stdin = "\n".join(lines) + "\n"

    result = CliRunner().invoke(app, _cli_args(gold, raw_dir), input=stdin)
    assert result.exit_code == 0, result.output

    new_row = _last_row(gold)
    # Label updated.
    assert new_row["label"]["primary_category"] == "m_and_a"
    # All other fields unchanged.
    for k, v in _VALID_LABEL.items():
        if k != "primary_category":
            assert new_row["label"][k] == v, f"field {k} unexpectedly changed"
    # Provenance records the change and preserves the original.
    prov = new_row["provenance"]
    assert prov["fields_changed"] == ["primary_category"]
    assert prov["fields_ambiguous"] == []
    assert prov["notes"] == "reclassified per body read"
    assert prov["llm_label_snapshot"]["primary_category"] == "financial_results"
    # Snapshot must be the *complete* original label, not just one field.
    assert prov["llm_label_snapshot"] == _VALID_LABEL


def test_cli_ambig_marks_field_without_changing_value(tmp_path: Path) -> None:
    """Marking expected_impact_period as ambiguous → value unchanged, field listed."""
    gold, raw_dir = _seed_one_filing(tmp_path)

    lines: list[str] = []
    for i in range(_N_FIELDS):
        if i == _EXPECTED_IMPACT_IDX:
            lines.append("a")  # ambig action
        else:
            lines.append("")
    lines.append("")  # notes
    lines.append("")  # save
    stdin = "\n".join(lines) + "\n"

    result = CliRunner().invoke(app, _cli_args(gold, raw_dir), input=stdin)
    assert result.exit_code == 0, result.output

    new_row = _last_row(gold)
    assert new_row["label"]["expected_impact_period"] == "current_quarter"
    prov = new_row["provenance"]
    assert prov["fields_changed"] == []
    assert prov["fields_ambiguous"] == ["expected_impact_period"]


def test_cli_edit_invalid_reprompts_then_accepts(tmp_path: Path) -> None:
    """Invalid edit value → re-prompt; second attempt valid → save with change."""
    gold, raw_dir = _seed_one_filing(tmp_path)

    lines: list[str] = []
    for i in range(_N_FIELDS):
        if i == _PRIMARY_CATEGORY_IDX:
            lines.append("e")
            lines.append("not-a-real-category")  # parse error → re-prompt
            lines.append("2")  # valid: "executive_change"
        else:
            lines.append("")
    lines.append("")  # notes
    lines.append("")  # save
    stdin = "\n".join(lines) + "\n"

    result = CliRunner().invoke(app, _cli_args(gold, raw_dir), input=stdin)
    assert result.exit_code == 0, result.output
    # Error message printed before the successful retry.
    assert "expected one of" in result.output or "parse error" in result.output

    new_row = _last_row(gold)
    assert new_row["label"]["primary_category"] == "executive_change"
    assert new_row["provenance"]["fields_changed"] == ["primary_category"]
    assert new_row["provenance"]["llm_label_snapshot"]["primary_category"] == "financial_results"


def test_cli_verification_type_recorded_in_provenance(tmp_path: Path) -> None:
    """--verification-type controls both the provenance string and the skip filter."""
    gold, raw_dir = _seed_one_filing(tmp_path)
    lines = [""] * _N_FIELDS + ["", ""]
    stdin = "\n".join(lines) + "\n"

    result = CliRunner().invoke(
        app,
        _cli_args(gold, raw_dir, "--verification-type", "llm_critical_review"),
        input=stdin,
    )
    assert result.exit_code == 0, result.output

    prov = _last_row(gold)["provenance"]
    assert prov["verification_type"] == "llm_critical_review"
    assert prov["verified_by"] == "tester"


def test_cli_discard_does_not_append_row(tmp_path: Path) -> None:
    """End-of-example discard → no row written, original file unchanged."""
    gold, raw_dir = _seed_one_filing(tmp_path)
    before = gold.read_text()

    lines = [""] * _N_FIELDS + ["", "d"]  # all keep, no notes, discard
    stdin = "\n".join(lines) + "\n"

    result = CliRunner().invoke(app, _cli_args(gold, raw_dir), input=stdin)
    assert result.exit_code == 0, result.output
    assert gold.read_text() == before  # untouched
