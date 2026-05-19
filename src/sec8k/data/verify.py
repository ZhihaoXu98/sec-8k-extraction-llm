"""Human-in-the-loop verification of LLM-generated gold-set labels.

Walks a stratified sample of unverified rows from ``data/gold/v1.jsonl``,
displays each filing's parsed text alongside the LLM label, prompts per-field
keep/edit/ambig decisions, and appends a verification block to the row's
provenance (preserving the original LLM label as ``llm_label_snapshot``).

I/O is fronted by a ``Prompter`` Protocol so tests can drive the per-example
loop without TTYs.
"""

from __future__ import annotations

import json
import logging
import os
import pydoc
import random
import sys
import time
import types
import typing
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import typer
from dotenv import load_dotenv
from pydantic import ValidationError

from sec8k.schema import Filing8K

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants & module-level config
# ---------------------------------------------------------------------------


GOLD_PATH_DEFAULT = Path("data/gold/v1.jsonl")
RAW_DIR_DEFAULT = Path("data/raw/edgar")
METRICS_PATH_DEFAULT: Path | None = None
SEED = 42

CATEGORIES: tuple[str, ...] = (
    "m_and_a",
    "executive_change",
    "financial_results",
    "material_agreement",
    "regulatory",
    "other",
)

FIELD_ORDER: tuple[str, ...] = tuple(Filing8K.model_fields.keys())

ActionStr = Literal["keep", "edit", "ambig", "skip_rest", "quit"]
EndChoice = Literal["save", "redo", "discard"]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldInfo:
    """Display + parsing context for one Filing8K field."""

    name: str
    annotation: Any
    description: str
    is_nullable: bool
    literal_choices: tuple[str, ...] | None  # set when underlying type is Literal
    is_list_of_str: bool
    base_kind: Literal["str", "date", "float", "literal", "list_str", "other"]


@dataclass(frozen=True)
class FieldDecision:
    """One field's user-supplied verdict during the per-field loop."""

    action: ActionStr
    new_value: Any = None  # populated when action == "edit" with a *validated* value


@dataclass
class ExampleOutcome:
    """Per-example result returned from :func:`verify_one`."""

    saved: bool
    new_row: dict[str, Any] | None
    fields_changed: list[str]
    fields_ambiguous: list[str]
    notes: str
    duration_seconds: float
    quit_requested: bool = False


@dataclass
class SessionMetrics:
    """In-memory session accounting; printed at end of run."""

    session_id: str
    verifier: str
    started_at: datetime
    examples_verified: int = 0
    examples_discarded: int = 0
    total_fields_changed: int = 0
    fields_changed_per_example: list[int] = field(default_factory=list)
    seconds_per_example: list[float] = field(default_factory=list)

    def record(self, outcome: ExampleOutcome) -> None:
        if outcome.saved:
            self.examples_verified += 1
            self.total_fields_changed += len(outcome.fields_changed)
            self.fields_changed_per_example.append(len(outcome.fields_changed))
            self.seconds_per_example.append(outcome.duration_seconds)
        else:
            self.examples_discarded += 1

    def summary(self) -> str:
        n = self.examples_verified
        elapsed = (datetime.now(UTC) - self.started_at).total_seconds()
        mean_fields = (
            self.total_fields_changed / n if n else 0.0
        )
        mean_sec = (
            sum(self.seconds_per_example) / n if n else 0.0
        )
        return (
            f"session={self.session_id} verifier={self.verifier}\n"
            f"verified={n}  discarded={self.examples_discarded}  "
            f"elapsed={elapsed:.1f}s\n"
            f"mean fields changed/example: {mean_fields:.2f}\n"
            f"mean seconds/example:        {mean_sec:.1f}"
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["started_at"] = self.started_at.isoformat()
        return d


# ---------------------------------------------------------------------------
# Prompter protocol
# ---------------------------------------------------------------------------


class Prompter(Protocol):
    """Dependency-injection seam between verify_one and the terminal.

    The TerminalPrompter implementation uses pydoc.pager + typer.prompt.
    Tests inject a ScriptedPrompter that returns canned decisions.
    """

    def show_filing(self, accession: str, parsed_text: str) -> None: ...
    def show_label(self, label: dict[str, Any], fields: list[FieldInfo]) -> None: ...
    def prompt_field(
        self,
        idx: int,
        total: int,
        field_info: FieldInfo,
        current_value: Any,
        validate: Callable[[Any], None],
    ) -> FieldDecision: ...
    def prompt_notes(self) -> str: ...
    def prompt_end_of_example(self, changes_summary: str) -> EndChoice: ...


# ---------------------------------------------------------------------------
# Field introspection
# ---------------------------------------------------------------------------


def _field_info(name: str) -> FieldInfo:
    f = Filing8K.model_fields[name]
    ann = f.annotation
    description = (f.description or "").strip()

    args = typing.get_args(ann)
    nullable = type(None) in args
    inner = ann
    origin = typing.get_origin(ann)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            inner = non_none[0]
            origin = typing.get_origin(inner)
            args = typing.get_args(inner)

    literal_choices: tuple[str, ...] | None = None
    kind: Literal["str", "date", "float", "literal", "list_str", "other"]
    is_list = False

    if origin is typing.Literal:
        literal_choices = tuple(str(a) for a in args)
        kind = "literal"
    elif origin is list:
        is_list = True
        kind = "list_str"
    elif inner is date:
        kind = "date"
    elif inner is float:
        kind = "float"
    elif inner is str:
        kind = "str"
    else:
        kind = "other"

    return FieldInfo(
        name=name,
        annotation=ann,
        description=description,
        is_nullable=nullable,
        literal_choices=literal_choices,
        is_list_of_str=is_list,
        base_kind=kind,
    )


def field_infos() -> list[FieldInfo]:
    """All 14 Filing8K fields with type metadata, in declaration order."""
    return [_field_info(name) for name in FIELD_ORDER]


# ---------------------------------------------------------------------------
# Pure helpers: load + select
# ---------------------------------------------------------------------------


def load_gold_latest(gold_path: Path) -> dict[str, dict[str, Any]]:
    """Build accession → latest row from ``gold_path``. Append-only, latest-wins."""
    out: dict[str, dict[str, Any]] = {}
    if not gold_path.exists():
        return out
    for raw_line in gold_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("corrupt line in %s; skipping", gold_path)
            continue
        acc = row.get("accession")
        if isinstance(acc, str):
            out[acc] = row
    return out


def select_unverified(
    rows: dict[str, dict[str, Any]],
    category: str | None,
    limit: int | None,
    seed: int = SEED,
    skip_verification_type: str = "human_review",
) -> Iterator[dict[str, Any]]:
    """Yield unverified rows stratified across primary_category, round-robin.

    A row is considered "verified" (and therefore excluded) if its latest
    ``provenance.verification_type`` matches ``skip_verification_type``. Default
    skips human-reviewed rows only; LLM-critic-reviewed rows still surface so a
    human can override them. Pass a different value to skip a different protocol.
    """
    candidates = [
        r
        for r in rows.values()
        if (r.get("provenance") or {}).get("verification_type") != skip_verification_type
        and (category is None or (r.get("label") or {}).get("primary_category") == category)
    ]
    by_cat: dict[str, list[dict[str, Any]]] = {c: [] for c in CATEGORIES}
    for r in candidates:
        cat = (r.get("label") or {}).get("primary_category")
        if isinstance(cat, str) and cat in by_cat:
            by_cat[cat].append(r)

    rng = random.Random(seed)
    for cat in CATEGORIES:
        rng.shuffle(by_cat[cat])

    yielded = 0
    while True:
        any_yielded = False
        for cat in CATEGORIES:
            if not by_cat[cat]:
                continue
            yield by_cat[cat].pop()
            any_yielded = True
            yielded += 1
            if limit is not None and yielded >= limit:
                return
        if not any_yielded:
            return


# ---------------------------------------------------------------------------
# verify_one — the per-example flow
# ---------------------------------------------------------------------------


def _now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _filing_text_for(row: dict[str, Any], raw_dir: Path) -> str | None:
    accession = row.get("accession", "")
    if not isinstance(accession, str):
        return None
    path = raw_dir / f"{accession.replace('-', '')}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    text = data.get("parsed_text")
    return text if isinstance(text, str) else None


def _format_validation_errors(exc: ValidationError) -> str:
    errs = exc.errors()
    parts = [f"{len(errs)} validation error(s):"]
    for e in errs[:5]:
        loc = ".".join(str(p) for p in e.get("loc", ()))
        parts.append(f"  - {loc}: {e.get('msg', '')}")
    if len(errs) > 5:
        parts.append(f"  ...and {len(errs) - 5} more")
    return "\n".join(parts)


def verify_one(
    row: dict[str, Any],
    raw_dir: Path,
    prompter: Prompter,
    session_id: str,
    verifier: str,
    verification_type: str = "human_review",
) -> ExampleOutcome:
    """Run the per-example verification flow. Returns outcome and an optional new row.

    ``verifier`` is the identity string recorded as ``provenance.verified_by``
    (e.g., ``"zhihao"``, ``"claude-opus-4-7"``). ``verification_type`` is the
    protocol — ``"human_review"`` (default), ``"llm_critical_review"``, etc.
    """
    started_at = time.monotonic()
    accession = str(row.get("accession", ""))
    original_label = dict(row.get("label") or {})
    working_label = dict(original_label)

    parsed_text = _filing_text_for(row, raw_dir)
    if parsed_text is None:
        logger.warning(
            "filing text missing for %s under %s; skipping", accession, raw_dir
        )
        return ExampleOutcome(
            saved=False,
            new_row=None,
            fields_changed=[],
            fields_ambiguous=[],
            notes="",
            duration_seconds=time.monotonic() - started_at,
        )

    fields = field_infos()
    prompter.show_filing(accession, parsed_text)
    prompter.show_label(working_label, fields)

    fields_changed: list[str] = []
    fields_ambiguous: list[str] = []
    skip_rest = False
    quit_requested = False

    for idx, field_info in enumerate(fields, start=1):
        if skip_rest:
            break
        current_value = working_label.get(field_info.name)

        def _validate(candidate_value: Any, _name: str = field_info.name) -> None:
            candidate = {**working_label, _name: candidate_value}
            Filing8K.model_validate(candidate)

        decision = prompter.prompt_field(
            idx=idx,
            total=len(fields),
            field_info=field_info,
            current_value=current_value,
            validate=_validate,
        )
        if decision.action == "keep":
            continue
        if decision.action == "ambig":
            fields_ambiguous.append(field_info.name)
            continue
        if decision.action == "skip_rest":
            skip_rest = True
            break
        if decision.action == "quit":
            quit_requested = True
            return ExampleOutcome(
                saved=False,
                new_row=None,
                fields_changed=fields_changed,
                fields_ambiguous=fields_ambiguous,
                notes="",
                duration_seconds=time.monotonic() - started_at,
                quit_requested=True,
            )
        if decision.action == "edit":
            working_label[field_info.name] = decision.new_value
            fields_changed.append(field_info.name)
            continue

    notes = prompter.prompt_notes()

    changes_summary = (
        f"changed={fields_changed}\nambiguous={fields_ambiguous}\nnotes={notes!r}"
    )
    end_choice = prompter.prompt_end_of_example(changes_summary)
    if end_choice == "discard":
        return ExampleOutcome(
            saved=False,
            new_row=None,
            fields_changed=fields_changed,
            fields_ambiguous=fields_ambiguous,
            notes=notes,
            duration_seconds=time.monotonic() - started_at,
        )
    if end_choice == "redo":
        # Recurse once with a fresh working copy. The prompter's queue should
        # supply a second round of decisions; tests can exercise this path.
        return verify_one(row, raw_dir, prompter, session_id, verifier, verification_type)

    duration = time.monotonic() - started_at
    # Final validation guard — refuses to write a row that doesn't satisfy
    # Filing8K. In practice prompt_field's validate callback prevents this, but
    # defense in depth is cheap.
    Filing8K.model_validate(working_label)

    new_provenance = dict(row.get("provenance") or {})
    new_provenance.update(
        {
            "verified_by": verifier,
            "verification_type": verification_type,
            "verified_at": _now_utc_iso(),
            "verification_session_id": session_id,
            "fields_changed": fields_changed,
            "fields_ambiguous": fields_ambiguous,
            "notes": notes,
            "duration_seconds": round(duration, 3),
            "llm_label_snapshot": original_label,
        }
    )
    new_row = {
        "accession": accession,
        "label": working_label,
        "provenance": new_provenance,
    }

    return ExampleOutcome(
        saved=True,
        new_row=new_row,
        fields_changed=fields_changed,
        fields_ambiguous=fields_ambiguous,
        notes=notes,
        duration_seconds=duration,
        quit_requested=quit_requested,
    )


# ---------------------------------------------------------------------------
# verify_batch — orchestrator
# ---------------------------------------------------------------------------


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def verify_batch(
    *,
    gold_path: Path,
    raw_dir: Path,
    prompter: Prompter,
    category: str | None,
    limit: int | None,
    seed: int,
    session_id: str,
    verifier: str,
    verification_type: str = "human_review",
    metrics_out: Path | None = None,
) -> SessionMetrics:
    """Stream selected unverified rows through ``verify_one``."""
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}; valid: {CATEGORIES}")

    rows = load_gold_latest(gold_path)
    selected = select_unverified(
        rows,
        category=category,
        limit=limit,
        seed=seed,
        skip_verification_type=verification_type,
    )

    metrics = SessionMetrics(
        session_id=session_id,
        verifier=verifier,
        started_at=datetime.now(UTC),
    )

    try:
        for row in selected:
            outcome = verify_one(
                row=row,
                raw_dir=raw_dir,
                prompter=prompter,
                session_id=session_id,
                verifier=verifier,
                verification_type=verification_type,
            )
            metrics.record(outcome)
            if outcome.saved and outcome.new_row is not None:
                _append_jsonl(gold_path, outcome.new_row)
            if outcome.quit_requested:
                break
    except KeyboardInterrupt:
        logger.warning("interrupted; %d verified", metrics.examples_verified)

    if metrics_out is not None:
        _append_jsonl(metrics_out, metrics.to_dict())

    return metrics


# ---------------------------------------------------------------------------
# TerminalPrompter — the real TUI (exercised manually, not unit-tested)
# ---------------------------------------------------------------------------


def _format_value_for_display(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _parse_input_value(raw: str, field_info: FieldInfo) -> Any:
    """Parse a user-typed value into the Python representation for the field."""
    text = raw.strip()
    if field_info.is_nullable and text.lower() in {"null", "none", ""}:
        return None
    if field_info.base_kind == "literal":
        choices = field_info.literal_choices or ()
        # Accept the literal value verbatim or a 1-based index.
        if text.isdigit():
            i = int(text)
            if 1 <= i <= len(choices):
                return choices[i - 1]
        if text in choices:
            return text
        raise ValueError(
            f"expected one of {list(choices)} (or 1..{len(choices)}); got {text!r}"
        )
    if field_info.base_kind == "list_str":
        # Semicolon-separated; trim each element. Empty list → [].
        if not text:
            return []
        return [p.strip() for p in text.split(";") if p.strip()]
    if field_info.base_kind == "date":
        return date.fromisoformat(text)
    if field_info.base_kind == "float":
        return float(text)
    if field_info.base_kind == "str":
        return text
    # Fallback: try JSON.
    return json.loads(text)


class TerminalPrompter:
    """pydoc.pager + typer-based TTY prompter. Manual smoke only."""

    def __init__(self, use_pager: bool = True) -> None:
        self.use_pager = use_pager

    def show_filing(self, accession: str, parsed_text: str) -> None:
        header = (
            f"\n========================================================\n"
            f"ACCESSION: {accession}\n"
            f"--------------------------------------------------------\n"
        )
        body = header + parsed_text
        if self.use_pager and sys.stdout.isatty():
            pydoc.pager(body)
        else:
            typer.echo(body)

    def show_label(self, label: dict[str, Any], fields: list[FieldInfo]) -> None:
        typer.echo("\n========== Current label (LLM) ==========")
        for f in fields:
            value = _format_value_for_display(label.get(f.name))
            typer.echo(f"  {f.name:<24} {value}")
            if f.description:
                typer.echo(f"    spec: {f.description}")
        typer.echo("==========================================\n")

    def prompt_field(
        self,
        idx: int,
        total: int,
        field_info: FieldInfo,
        current_value: Any,
        validate: Callable[[Any], None],
    ) -> FieldDecision:
        typer.echo(f"\n[{idx}/{total}] {field_info.name}")
        typer.echo(f"  value: {_format_value_for_display(current_value)}")
        if field_info.description:
            typer.echo(f"  spec:  {field_info.description}")
        if field_info.literal_choices:
            typer.echo(f"  choices: {list(field_info.literal_choices)}")

        while True:
            raw = typer.prompt(
                "  [k]eep / [e]dit / [a]mbig / [s]kip-rest / [q]uit",
                default="k",
                show_default=False,
            ).strip().lower()
            if raw in {"k", "keep", ""}:
                return FieldDecision("keep")
            if raw in {"a", "ambig"}:
                return FieldDecision("ambig")
            if raw in {"s", "skip", "skip-rest"}:
                return FieldDecision("skip_rest")
            if raw in {"q", "quit"}:
                return FieldDecision("quit")
            if raw in {"e", "edit"}:
                return self._prompt_edit(field_info, current_value, validate)
            typer.echo("  unknown choice; type k, e, a, s, or q")

    def _prompt_edit(
        self,
        field_info: FieldInfo,
        current_value: Any,
        validate: Callable[[Any], None],
    ) -> FieldDecision:
        hint = self._format_hint(field_info)
        typer.echo(f"  edit {field_info.name}  ({hint}) — type :cancel to abort edit")
        while True:
            raw = typer.prompt("  new value", default="", show_default=False)
            if raw.strip() == ":cancel":
                return FieldDecision("keep")
            try:
                value = _parse_input_value(raw, field_info)
            except (ValueError, json.JSONDecodeError) as exc:
                typer.echo(f"  parse error: {exc}")
                continue
            try:
                validate(value)
            except ValidationError as exc:
                typer.echo(_format_validation_errors(exc))
                continue
            return FieldDecision("edit", value)

    @staticmethod
    def _format_hint(field_info: FieldInfo) -> str:
        if field_info.base_kind == "literal":
            return f"one of {list(field_info.literal_choices or ())}"
        if field_info.base_kind == "list_str":
            return "semicolon-separated list, e.g. Acme; Beta LLC"
        if field_info.base_kind == "date":
            return "YYYY-MM-DD"
        if field_info.base_kind == "float":
            return "number (or 'null' if nullable)"
        if field_info.base_kind == "str":
            return "plain text (or 'null' if nullable)"
        return "JSON literal"

    def prompt_notes(self) -> str:
        result = typer.prompt(
            "  notes (optional, press Enter to skip)", default="", show_default=False
        )
        return str(result)

    def prompt_end_of_example(self, changes_summary: str) -> EndChoice:
        typer.echo(f"\n{changes_summary}")
        while True:
            raw = typer.prompt(
                "  [s]ave / [r]edo / [d]iscard", default="s", show_default=False
            ).strip().lower()
            if raw in {"s", "save", ""}:
                return "save"
            if raw in {"r", "redo"}:
                return "redo"
            if raw in {"d", "discard"}:
                return "discard"
            typer.echo("  unknown choice; type s, r, or d")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def cli(
    gold_path: Annotated[Path, typer.Option("--gold-path")] = GOLD_PATH_DEFAULT,
    raw_dir: Annotated[Path, typer.Option("--raw-dir")] = RAW_DIR_DEFAULT,
    category: Annotated[
        str | None, typer.Option("--category", help="Restrict to one primary_category.")
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Stop after N examples verified.")
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session-id", help="Group rows for downstream analysis."
        ),
    ] = None,
    verifier: Annotated[
        str | None,
        typer.Option(
            "--verifier",
            help="Identity recorded as provenance.verified_by; defaults to $USER.",
        ),
    ] = None,
    verification_type: Annotated[
        str,
        typer.Option(
            "--verification-type",
            help=(
                "Protocol recorded as provenance.verification_type. Default "
                "'human_review'. Pass 'llm_critical_review' if running a Claude "
                "or other LLM critic via a programmatic Prompter. The same value "
                "controls which rows are considered already-verified (skipped)."
            ),
        ),
    ] = "human_review",
    seed: Annotated[int, typer.Option("--seed")] = SEED,
    metrics_out: Annotated[
        Path | None,
        typer.Option("--metrics-out", help="Append session metrics as one JSON line."),
    ] = None,
    no_pager: Annotated[
        bool, typer.Option("--no-pager", help="Dump filing text instead of paging.")
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", envvar="SEC8K_LOG_LEVEL"),
    ] = None,
) -> None:
    """Walk unverified rows in v1.jsonl, prompting per-field keep/edit/ambig."""
    logging.basicConfig(
        level=(log_level or "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if category is not None and category not in CATEGORIES:
        typer.echo(
            f"--category must be one of {list(CATEGORIES)}; got {category!r}", err=True
        )
        raise typer.Exit(code=2)

    if session_id is None:
        session_id = f"cal-{datetime.now(UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    if not verifier:
        verifier = os.environ.get("USER", "unknown")

    prompter = TerminalPrompter(use_pager=not no_pager)

    rows = load_gold_latest(gold_path)
    candidates = list(
        select_unverified(
            rows,
            category=category,
            limit=limit,
            seed=seed,
            skip_verification_type=verification_type,
        )
    )
    if not candidates:
        typer.echo("No unverified rows to process (matching filters).")
        return

    typer.echo(
        f"session={session_id} verifier={verifier} type={verification_type}  "
        f"candidates={len(candidates)}"
    )

    metrics = verify_batch(
        gold_path=gold_path,
        raw_dir=raw_dir,
        prompter=prompter,
        category=category,
        limit=limit,
        seed=seed,
        session_id=session_id,
        verifier=verifier,
        verification_type=verification_type,
        metrics_out=metrics_out,
    )
    typer.echo("\n========== session summary ==========")
    typer.echo(metrics.summary())


def main() -> None:
    """Script entry point: load .env and run the typer app."""
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
