"""Tests for sec8k.data.verify.

All interactive flow exercised via ScriptedPrompter — no real TTY, no pager.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sec8k.data import verify as V
from sec8k.data.verify import (
    EndChoice,
    FieldDecision,
    FieldInfo,
    Prompter,
    load_gold_latest,
    select_unverified,
    verify_batch,
    verify_one,
)
from sec8k.schema import Filing8K


# ---------------------------------------------------------------------------
# Fixtures + scripted prompter
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


def _make_filing(raw_dir: Path, accession: str, parsed_text: str = "Filing body.") -> Path:
    no_dash = accession.replace("-", "")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{no_dash}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "fetched_at": "2025-04-30T00:00:00Z",
                "metadata": {"accession_number": accession},
                "raw_html": "<html></html>",
                "parsed_text": parsed_text,
            }
        )
    )
    return path


def _make_gold_row(
    accession: str,
    category: str = "financial_results",
    verified: bool = False,
    label_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label = {**_VALID_LABEL, "primary_category": category}
    if label_overrides:
        label.update(label_overrides)
    prov: dict[str, Any] = {
        "model": "claude-sonnet-4-6",
        "guidelines_sha": "sha-test",
        "labeled_at": "2025-04-30T00:00:00Z",
        "cost_usd": 0.02,
        "retried": False,
    }
    if verified:
        prov["verified_by"] = "test-user"
        prov["verification_type"] = "human_review"
    return {"accession": accession, "label": label, "provenance": prov}


@dataclass
class ScriptedPrompter:
    """Prompter that returns canned answers; records calls."""

    field_decisions: list[FieldDecision]
    notes: str = ""
    end_choices: list[EndChoice] = field(default_factory=lambda: ["save"])
    redo_field_decisions: list[FieldDecision] = field(default_factory=list)
    show_filing_calls: list[tuple[str, str]] = field(default_factory=list)
    show_label_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_field_calls: list[tuple[int, str]] = field(default_factory=list)
    _in_redo: bool = False

    def show_filing(self, accession: str, parsed_text: str) -> None:
        self.show_filing_calls.append((accession, parsed_text))

    def show_label(self, label: dict[str, Any], fields: list[FieldInfo]) -> None:
        self.show_label_calls.append(label)

    def prompt_field(
        self,
        idx: int,
        total: int,
        field_info: FieldInfo,
        current_value: Any,
        validate: Callable[[Any], None],
    ) -> FieldDecision:
        self.prompt_field_calls.append((idx, field_info.name))
        queue = self.redo_field_decisions if self._in_redo else self.field_decisions
        if not queue:
            return FieldDecision("keep")
        decision = queue.pop(0)
        if decision.action == "edit":
            validate(decision.new_value)  # surface ValidationError to test if invalid
        return decision

    def prompt_notes(self) -> str:
        return self.notes

    def prompt_end_of_example(self, changes_summary: str) -> EndChoice:
        if not self.end_choices:
            return "save"
        choice = self.end_choices.pop(0)
        if choice == "redo":
            self._in_redo = True
        return choice


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


def test_load_gold_latest_dedup(tmp_path: Path) -> None:
    gold = tmp_path / "v1.jsonl"
    gold.write_text(
        "\n".join(
            [
                json.dumps(_make_gold_row("A", label_overrides={"summary": "first"})),
                json.dumps(_make_gold_row("B")),
                json.dumps(_make_gold_row("A", label_overrides={"summary": "second"})),
            ]
        )
    )
    rows = load_gold_latest(gold)
    assert set(rows.keys()) == {"A", "B"}
    assert rows["A"]["label"]["summary"] == "second"


def test_load_gold_latest_missing_file_empty(tmp_path: Path) -> None:
    assert load_gold_latest(tmp_path / "missing.jsonl") == {}


def test_select_unverified_skips_verified() -> None:
    rows = {
        "A": _make_gold_row("A", verified=True),
        "B": _make_gold_row("B"),
    }
    picked = list(select_unverified(rows, category=None, limit=None))
    assert [r["accession"] for r in picked] == ["B"]


def test_select_unverified_round_robin() -> None:
    rows: dict[str, dict[str, Any]] = {}
    for cat in V.CATEGORIES:
        for i in range(3):
            acc = f"acc-{cat}-{i}"
            rows[acc] = _make_gold_row(acc, category=cat)
    picked = list(select_unverified(rows, category=None, limit=12))
    # First 6 should cover all 6 categories once before any category repeats.
    seen_first_pass: set[str] = set()
    for r in picked[:6]:
        seen_first_pass.add(r["label"]["primary_category"])
    assert seen_first_pass == set(V.CATEGORIES)


def test_select_unverified_category_filter() -> None:
    rows = {
        "A": _make_gold_row("A", category="m_and_a"),
        "B": _make_gold_row("B", category="financial_results"),
        "C": _make_gold_row("C", category="m_and_a"),
    }
    picked = list(select_unverified(rows, category="m_and_a", limit=None))
    assert {r["accession"] for r in picked} == {"A", "C"}


# ---------------------------------------------------------------------------
# verify_one tests
# ---------------------------------------------------------------------------


def test_verify_one_all_keep_writes_snapshot_no_changes(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_filing(raw_dir, "0000111-25-000001")
    row = _make_gold_row("0000111-25-000001")
    n_fields = len(Filing8K.model_fields)
    prompter = ScriptedPrompter(
        field_decisions=[FieldDecision("keep")] * n_fields,
        notes="",
        end_choices=["save"],
    )
    outcome = verify_one(
        row, raw_dir=raw_dir, prompter=prompter, session_id="s1", verifier="zhihao"
    )
    assert outcome.saved is True
    assert outcome.fields_changed == []
    assert outcome.fields_ambiguous == []
    assert outcome.new_row is not None
    prov = outcome.new_row["provenance"]
    assert prov["verified_by"] == "zhihao"
    assert prov["verification_type"] == "human_review"
    assert prov["verification_session_id"] == "s1"
    assert "verifier" not in prov  # collapsed into verified_by
    assert prov["llm_label_snapshot"] == _VALID_LABEL
    assert outcome.new_row["label"] == _VALID_LABEL


def test_verify_one_edit_field_writes_new_value_and_snapshot(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_filing(raw_dir, "0000111-25-000001")
    row = _make_gold_row("0000111-25-000001")
    n_fields = len(Filing8K.model_fields)
    decisions: list[FieldDecision] = []
    for name in Filing8K.model_fields:
        if name == "primary_category":
            decisions.append(FieldDecision("edit", new_value="m_and_a"))
        else:
            decisions.append(FieldDecision("keep"))
    prompter = ScriptedPrompter(field_decisions=decisions, notes="reclassified")
    outcome = verify_one(
        row, raw_dir=raw_dir, prompter=prompter, session_id="s1", verifier="zhihao"
    )
    assert outcome.saved is True
    assert outcome.fields_changed == ["primary_category"]
    assert outcome.new_row is not None
    assert outcome.new_row["label"]["primary_category"] == "m_and_a"
    assert outcome.new_row["provenance"]["llm_label_snapshot"]["primary_category"] == "financial_results"
    assert outcome.new_row["provenance"]["notes"] == "reclassified"


def test_verify_one_edit_invalid_raises_validation_error(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_filing(raw_dir, "0000111-25-000001")
    row = _make_gold_row("0000111-25-000001")
    # primary_category accepts only 6 enum values; "junk" must fail validation.
    decisions = [FieldDecision("keep")] * len(Filing8K.model_fields)
    for i, name in enumerate(Filing8K.model_fields):
        if name == "primary_category":
            decisions[i] = FieldDecision("edit", new_value="junk-category")
            break
    prompter = ScriptedPrompter(field_decisions=decisions)
    # The scripted prompter invokes the validate callback before returning
    # the edit decision — that raises ValidationError out of verify_one.
    with pytest.raises(ValidationError):
        verify_one(row, raw_dir=raw_dir, prompter=prompter, session_id="s1", verifier="u")


def test_verify_one_ambig(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_filing(raw_dir, "0000111-25-000001")
    row = _make_gold_row("0000111-25-000001")
    decisions = [FieldDecision("keep")] * len(Filing8K.model_fields)
    for i, name in enumerate(Filing8K.model_fields):
        if name == "expected_impact_period":
            decisions[i] = FieldDecision("ambig")
            break
    prompter = ScriptedPrompter(field_decisions=decisions)
    outcome = verify_one(
        row, raw_dir=raw_dir, prompter=prompter, session_id="s1", verifier="u"
    )
    assert outcome.fields_ambiguous == ["expected_impact_period"]
    assert outcome.new_row is not None
    assert outcome.new_row["label"]["expected_impact_period"] == "current_quarter"


def test_verify_one_skip_rest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_filing(raw_dir, "0000111-25-000001")
    row = _make_gold_row("0000111-25-000001")
    # First decision is skip_rest; should jump straight to end-of-example.
    prompter = ScriptedPrompter(field_decisions=[FieldDecision("skip_rest")])
    outcome = verify_one(
        row, raw_dir=raw_dir, prompter=prompter, session_id="s1", verifier="u"
    )
    assert outcome.saved is True
    assert outcome.fields_changed == []
    assert outcome.fields_ambiguous == []
    # Only one prompt_field call happened.
    assert len(prompter.prompt_field_calls) == 1


def test_verify_one_quit(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_filing(raw_dir, "0000111-25-000001")
    row = _make_gold_row("0000111-25-000001")
    prompter = ScriptedPrompter(field_decisions=[FieldDecision("quit")])
    outcome = verify_one(
        row, raw_dir=raw_dir, prompter=prompter, session_id="s1", verifier="u"
    )
    assert outcome.saved is False
    assert outcome.quit_requested is True
    assert outcome.new_row is None


def test_verify_one_discard_no_write(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_filing(raw_dir, "0000111-25-000001")
    row = _make_gold_row("0000111-25-000001")
    n_fields = len(Filing8K.model_fields)
    prompter = ScriptedPrompter(
        field_decisions=[FieldDecision("keep")] * n_fields,
        end_choices=["discard"],
    )
    outcome = verify_one(
        row, raw_dir=raw_dir, prompter=prompter, session_id="s1", verifier="u"
    )
    assert outcome.saved is False
    assert outcome.new_row is None


def test_verify_one_missing_filing_skipped(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    row = _make_gold_row("0000999-25-000999")  # no filing file made
    prompter = ScriptedPrompter(field_decisions=[])
    outcome = verify_one(
        row, raw_dir=raw_dir, prompter=prompter, session_id="s1", verifier="u"
    )
    assert outcome.saved is False
    assert prompter.show_filing_calls == []


# ---------------------------------------------------------------------------
# verify_batch tests
# ---------------------------------------------------------------------------


def _seed_gold(tmp_path: Path, n: int) -> Path:
    """Write n unverified rows, one per category (round-robin)."""
    gold = tmp_path / "gold" / "v1.jsonl"
    gold.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_path / "raw"
    rows: list[str] = []
    for i in range(n):
        cat = V.CATEGORIES[i % len(V.CATEGORIES)]
        acc = f"0000111-25-{i:06d}"
        _make_filing(raw_dir, acc)
        rows.append(json.dumps(_make_gold_row(acc, category=cat)))
    gold.write_text("\n".join(rows) + "\n")
    return gold


def test_verify_batch_limit_terminates(tmp_path: Path) -> None:
    gold = _seed_gold(tmp_path, n=12)
    raw_dir = tmp_path / "raw"

    n_fields = len(Filing8K.model_fields)
    prompter = ScriptedPrompter(
        field_decisions=[FieldDecision("keep")] * (n_fields * 3),
        end_choices=["save", "save", "save"],
    )
    metrics = verify_batch(
        gold_path=gold,
        raw_dir=raw_dir,
        prompter=prompter,
        category=None,
        limit=3,
        seed=42,
        session_id="s1",
        verifier="u",
    )
    assert metrics.examples_verified == 3
    new_rows = [
        json.loads(line) for line in gold.read_text().splitlines() if line.strip()
    ]
    # 12 original + 3 verified appended = 15
    assert len(new_rows) == 15


def test_verify_batch_quit_mid_session(tmp_path: Path) -> None:
    gold = _seed_gold(tmp_path, n=6)
    raw_dir = tmp_path / "raw"

    n_fields = len(Filing8K.model_fields)
    # 1st example: all keep, save. 2nd example: quit on field 1.
    decisions = (
        [FieldDecision("keep")] * n_fields  # example 1
        + [FieldDecision("quit")]            # example 2
    )
    prompter = ScriptedPrompter(field_decisions=decisions, end_choices=["save"])
    metrics = verify_batch(
        gold_path=gold,
        raw_dir=raw_dir,
        prompter=prompter,
        category=None,
        limit=None,
        seed=42,
        session_id="s1",
        verifier="u",
    )
    assert metrics.examples_verified == 1
    assert metrics.examples_discarded == 1
    new_rows = [
        json.loads(line) for line in gold.read_text().splitlines() if line.strip()
    ]
    assert len(new_rows) == 7  # 6 original + 1 verified


def test_verify_batch_writes_metrics_jsonl(tmp_path: Path) -> None:
    gold = _seed_gold(tmp_path, n=2)
    raw_dir = tmp_path / "raw"
    metrics_path = tmp_path / "metrics.jsonl"

    n_fields = len(Filing8K.model_fields)
    prompter = ScriptedPrompter(
        field_decisions=[FieldDecision("keep")] * (n_fields * 2),
        end_choices=["save", "save"],
    )
    verify_batch(
        gold_path=gold,
        raw_dir=raw_dir,
        prompter=prompter,
        category=None,
        limit=2,
        seed=42,
        session_id="s-metrics",
        verifier="u",
        metrics_out=metrics_path,
    )
    lines = [
        json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()
    ]
    assert len(lines) == 1
    assert lines[0]["session_id"] == "s-metrics"
    assert lines[0]["examples_verified"] == 2


def test_verify_batch_rejects_unknown_category(tmp_path: Path) -> None:
    gold = _seed_gold(tmp_path, n=1)
    raw_dir = tmp_path / "raw"
    prompter = ScriptedPrompter(field_decisions=[])
    with pytest.raises(ValueError):
        verify_batch(
            gold_path=gold,
            raw_dir=raw_dir,
            prompter=prompter,
            category="not-a-real-category",
            limit=None,
            seed=42,
            session_id="s",
            verifier="u",
        )


def test_session_metrics_summary_format() -> None:
    from datetime import UTC, datetime

    m = V.SessionMetrics(
        session_id="abc", verifier="u", started_at=datetime.now(UTC)
    )
    m.record(
        V.ExampleOutcome(
            saved=True, new_row={}, fields_changed=["a", "b"], fields_ambiguous=[],
            notes="", duration_seconds=12.5,
        )
    )
    m.record(
        V.ExampleOutcome(
            saved=True, new_row={}, fields_changed=[], fields_ambiguous=[],
            notes="", duration_seconds=4.0,
        )
    )
    text = m.summary()
    assert "verified=2" in text
    assert "mean fields changed/example: 1.00" in text
    assert "mean seconds/example:        8.2" in text


# ---------------------------------------------------------------------------
# Field type introspection sanity
# ---------------------------------------------------------------------------


def test_field_info_form_type_is_literal() -> None:
    info = V._field_info("form_type")
    assert info.base_kind == "literal"
    assert info.literal_choices == ("8-K", "8-K/A")
    assert info.is_nullable is False


def test_field_info_filer_ticker_is_nullable_str() -> None:
    info = V._field_info("filer_ticker")
    assert info.base_kind == "str"
    assert info.is_nullable is True


def test_field_info_items_is_list_str() -> None:
    info = V._field_info("items")
    assert info.base_kind == "list_str"
    assert info.is_list_of_str is True


def test_field_info_event_date_is_nullable_date() -> None:
    info = V._field_info("event_date")
    assert info.base_kind == "date"
    assert info.is_nullable is True


def test_field_info_monetary_amount_is_nullable_float() -> None:
    info = V._field_info("monetary_amount")
    assert info.base_kind == "float"
    assert info.is_nullable is True


def test_parse_input_value_literal_by_index() -> None:
    info = V._field_info("primary_category")
    assert info.literal_choices is not None
    assert V._parse_input_value("1", info) == info.literal_choices[0]
    assert V._parse_input_value(info.literal_choices[2], info) == info.literal_choices[2]


def test_parse_input_value_list_str_semicolons() -> None:
    info = V._field_info("counterparties")
    assert V._parse_input_value("Acme; Beta LLC; Gamma", info) == ["Acme", "Beta LLC", "Gamma"]
    assert V._parse_input_value("", info) == []


def test_parse_input_value_date() -> None:
    info = V._field_info("filing_date")
    assert V._parse_input_value("2025-04-30", info) == date(2025, 4, 30)


def test_parse_input_value_nullable_null_keyword() -> None:
    info = V._field_info("filer_ticker")
    assert V._parse_input_value("null", info) is None
    assert V._parse_input_value("AAPL", info) == "AAPL"
