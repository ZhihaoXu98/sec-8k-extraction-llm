"""Claude-driven gold-set labeler for SEC 8-K filings.

Reads ingested filings from ``data/raw/edgar/<acc_nodash>.json`` and uses Claude
Sonnet 4.6 (via the Anthropic SDK, with prompt caching on the labeling
guidelines doc) to produce Filing8K-valid labels in ``data/gold/v1.jsonl``.

The labeling guidelines doc (``docs/labeling_guidelines.md``) is sent as the
system prompt; the Filing8K JSON schema is enforced via tool use. Validation
failures get one in-conversation retry; second-pass failures are recorded to
``data/gold/failures.jsonl`` so a bad filing never kills the batch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Protocol

import typer
from dotenv import load_dotenv
from pydantic import ValidationError

from sec8k.schema import Filing8K

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DEFAULT = "claude-sonnet-4-6"
TOOL_NAME = "emit_filing8k"
PROMPT_VERSION = "v2"  # v2: user prompt exposes EDGAR ACCEPTANCE_DATETIME for filing_date
SEED = 42

GUIDELINES_PATH = Path("docs/labeling_guidelines.md")
RAW_DIR_DEFAULT = Path("data/raw/edgar")
GOLD_PATH_DEFAULT = Path("data/gold/v1.jsonl")
FAILURES_PATH_DEFAULT = Path("data/gold/failures.jsonl")

# Sonnet 4.6 has a 200K input window; budget input at ~70K tokens to leave room
# for the cached prefix (~10K) and output (~4K). 280K chars ÷ 4 chars/token = 70K.
MAX_INPUT_CHARS = 280_000

CATEGORIES: tuple[str, ...] = (
    "m_and_a",
    "executive_change",
    "financial_results",
    "material_agreement",
    "regulatory",
    "other",
)

# USD per token. Cache write is +25% vs base input; cache read is -90%.
PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {
        "input": 3.00 / 1_000_000,
        "output": 15.00 / 1_000_000,
        "cache_write": 3.75 / 1_000_000,
        "cache_read": 0.30 / 1_000_000,
    },
    "claude-opus-4-6": {
        "input": 15.00 / 1_000_000,
        "output": 75.00 / 1_000_000,
        "cache_write": 18.75 / 1_000_000,
        "cache_read": 1.50 / 1_000_000,
    },
}

USER_PROMPT_TEMPLATE = (
    "You are labeling a single SEC 8-K filing for the gold dataset.\n\n"
    "Apply the labeling guidelines (in the system prompt) to the filing text below "
    "and call the `emit_filing8k` tool with the extracted fields. Output only the "
    "tool call; do not produce explanatory text.\n\n"
    "ACCESSION: {accession}\n"
    "ACCEPTANCE_DATETIME: {acceptance_datetime}  "
    "(EDGAR's official acceptance stamp — this IS the canonical `filing_date` per spec. "
    "Use the date portion. Do not use the body's 'Date of Report' line, which is `event_date`.)\n\n"
    "FILING TEXT:\n{parsed_text}"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostEstimate:
    """Pre-run cost prediction returned by :func:`estimate_cost`."""

    n_filings: int
    est_input_tokens: int
    est_output_tokens: int
    est_cache_write_tokens: int
    est_cache_read_tokens: int
    est_total_usd: float
    model: str

    def render(self) -> str:
        return (
            f"Cost estimate ({self.model}, {self.n_filings} filings):\n"
            f"  input tokens (per-filing user content): ~{self.est_input_tokens:,}\n"
            f"  output tokens (Filing8K JSON):          ~{self.est_output_tokens:,}\n"
            f"  cache write tokens (1 system prefix):   ~{self.est_cache_write_tokens:,}\n"
            f"  cache read tokens (cached reads):       ~{self.est_cache_read_tokens:,}\n"
            f"  estimated total cost:                   ~${self.est_total_usd:.4f}"
        )


@dataclass(frozen=True)
class LabelResult:
    """Successful label, including provenance for the gold-set row."""

    accession: str
    label: Filing8K
    model: str
    guidelines_sha: str
    labeled_at: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float
    retried: bool


@dataclass(frozen=True)
class LabelFailure:
    """Failure record persisted to failures.jsonl."""

    accession: str
    error: str
    raw_response: dict[str, Any] | None
    model: str
    guidelines_sha: str
    failed_at: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class LabelingSummary:
    """Per-run accounting returned by :func:`label_batch`."""

    started_at: datetime
    finished_at: datetime
    model: str
    guidelines_sha: str
    n_attempted: int = 0
    n_succeeded: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    per_category_succeeded: dict[str, int] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_input_tokens: int = 0
    total_cache_read_input_tokens: int = 0
    total_cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compute_guidelines_sha(path: Path = GUIDELINES_PATH) -> str:
    """Git-blob SHA-1 of the labeling guidelines file.

    Matches ``git hash-object`` output so provenance cross-references with git history.
    Pure stdlib — no subprocess, no .git dependency.
    """
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def _filing8k_input_schema() -> dict[str, Any]:
    """Filing8K JSON schema in a shape Anthropic's `input_schema` field accepts.

    Inlines ``$defs`` references and strips Pydantic-only ``title`` keys.
    """
    raw = Filing8K.model_json_schema()
    defs: dict[str, Any] = raw.pop("$defs", {})

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str) and node["$ref"].startswith(
                "#/$defs/"
            ):
                key = node["$ref"].split("/")[-1]
                return inline(defs[key])
            return {k: inline(v) for k, v in node.items() if k != "title"}
        if isinstance(node, list):
            return [inline(x) for x in node]
        return node

    schema = inline(raw)
    if not isinstance(schema, dict):
        raise RuntimeError("Filing8K.model_json_schema() did not return an object")
    return schema


def compute_call_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int,
    cache_read_input_tokens: int,
    model: str,
) -> float:
    """USD cost of one Anthropic API call given its usage counters."""
    if model not in PRICING:
        raise ValueError(f"no pricing entry for model {model!r}; add to PRICING")
    p = PRICING[model]
    return (
        input_tokens * p["input"]
        + output_tokens * p["output"]
        + cache_creation_input_tokens * p["cache_write"]
        + cache_read_input_tokens * p["cache_read"]
    )


def _approx_tokens_from_chars(n_chars: int) -> int:
    """Claude's rule of thumb is ~4 chars/token for English."""
    return (n_chars + 3) // 4


def estimate_cost(
    filings: list[Path],
    model: str,
    guidelines_path: Path = GUIDELINES_PATH,
) -> CostEstimate:
    """Predict total USD cost before launching the run.

    Assumes one cache-write turn (first call) plus one cache-read per subsequent call.
    Output tokens are estimated at ~700 per call (Filing8K JSON has ~14 fields).
    """
    n = len(filings)
    if n == 0:
        return CostEstimate(0, 0, 0, 0, 0, 0.0, model)

    guidelines_tokens = _approx_tokens_from_chars(len(guidelines_path.read_text(encoding="utf-8"))) + 200

    input_tokens_total = 0
    for path in filings:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text_len = len(data.get("parsed_text", ""))
        if text_len > MAX_INPUT_CHARS:
            text_len = MAX_INPUT_CHARS
        input_tokens_total += _approx_tokens_from_chars(text_len) + 100

    output_tokens_total = n * 700
    cache_write_tokens = guidelines_tokens
    cache_read_tokens = guidelines_tokens * (n - 1) if n > 1 else 0

    cost = compute_call_cost(
        input_tokens=input_tokens_total,
        output_tokens=output_tokens_total,
        cache_creation_input_tokens=cache_write_tokens,
        cache_read_input_tokens=cache_read_tokens,
        model=model,
    )

    return CostEstimate(
        n_filings=n,
        est_input_tokens=input_tokens_total,
        est_output_tokens=output_tokens_total,
        est_cache_write_tokens=cache_write_tokens,
        est_cache_read_tokens=cache_read_tokens,
        est_total_usd=cost,
        model=model,
    )


def load_latest_sha_per_accession(gold_path: Path) -> dict[str, str]:
    """Map accession -> latest provenance.guidelines_sha in ``gold_path``.

    The file is append-only; later rows override earlier ones in this mapping.
    Missing file → empty dict.
    """
    out: dict[str, str] = {}
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
        accession = row.get("accession")
        provenance = row.get("provenance") or {}
        sha = provenance.get("guidelines_sha")
        if isinstance(accession, str) and isinstance(sha, str):
            out[accession] = sha
    return out


def _resolve_accession_to_path(in_dir: Path, accession: str) -> Path:
    """Map an accession (with or without dashes) to its on-disk JSON path."""
    no_dashes = accession.replace("-", "")
    return in_dir / f"{no_dashes}.json"


def select_filings(
    in_dir: Path,
    limit: int | None,
    explicit_accessions: list[str] | None,
    seed: int = SEED,
) -> list[Path]:
    """Choose which filing JSONs to label.

    Either pass ``explicit_accessions`` (takes precedence) or ``limit`` for a
    stratified random sample across the 6 primary_category groups in
    ``manifest.jsonl``. Deterministic given seed.
    """
    if explicit_accessions:
        return [_resolve_accession_to_path(in_dir, acc) for acc in explicit_accessions]

    if limit is None:
        raise ValueError("either limit or explicit_accessions must be supplied")

    manifest_path = in_dir / "manifest.jsonl"
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    if manifest_path.exists():
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            cat = row.get("primary_category")
            acc = row.get("accession")
            if isinstance(cat, str) and cat in by_cat and isinstance(acc, str):
                by_cat[cat].append(acc)

    rng = random.Random(seed)
    for cat in CATEGORIES:
        rng.shuffle(by_cat[cat])

    base = limit // len(CATEGORIES)
    remainder = limit - base * len(CATEGORIES)
    quotas: dict[str, int] = {c: base for c in CATEGORIES}
    for c in sorted(CATEGORIES)[:remainder]:
        quotas[c] += 1

    picked: list[str] = []
    for cat in CATEGORIES:
        available = by_cat[cat]
        want = quotas[cat]
        if len(available) < want:
            logger.warning(
                "category %s has %d filings available but quota is %d; taking all",
                cat,
                len(available),
                want,
            )
        picked.extend(available[: min(want, len(available))])

    return [_resolve_accession_to_path(in_dir, acc) for acc in picked]


# ---------------------------------------------------------------------------
# Anthropic call (single filing)
# ---------------------------------------------------------------------------


class _AnthropicLike(Protocol):
    """Minimal shape the orchestrator needs from an Anthropic client (for tests)."""

    @property
    def messages(self) -> Any: ...


def _build_tool_definition() -> dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": (
            "Emit the extracted Filing8K fields for the provided 8-K filing. "
            "Every field must conform to the input_schema exactly. Use null for "
            "fields the filing does not disclose (where the field is nullable)."
        ),
        "input_schema": _filing8k_input_schema(),
    }


def _build_call_kwargs(
    *,
    model: str,
    guidelines: str,
    user_block: str,
    extra_turns: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": user_block}]},
    ]
    if extra_turns:
        messages.extend(extra_turns)
    return {
        "model": model,
        "max_tokens": 4096,
        "system": [
            {
                "type": "text",
                "text": guidelines,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [_build_tool_definition()],
        "tool_choice": {"type": "tool", "name": TOOL_NAME},
        "messages": messages,
    }


def _extract_tool_use(content: Iterable[Any]) -> tuple[str, dict[str, Any]] | None:
    """Return (tool_use_id, tool_input_dict) if a tool_use block is present."""
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "tool_use":
            tool_input = getattr(block, "input", None)
            tool_id = getattr(block, "id", None)
            if isinstance(tool_input, dict) and isinstance(tool_id, str):
                return tool_id, tool_input
    return None


def _format_validation_errors(exc: ValidationError, payload: dict[str, Any]) -> str:
    """Produce a short multi-line summary of Pydantic errors for the retry prompt."""
    errors = exc.errors()
    parts = [f"{len(errors)} validation error(s):"]
    for err in errors[:5]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "")
        parts.append(f"  - {loc}: {msg}")
    if len(errors) > 5:
        parts.append(f"  ...and {len(errors) - 5} more")
    parts.append("Re-emit the tool with corrected fields. Same schema, same accession.")
    return "\n".join(parts)


def _content_to_dicts(content: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert anthropic SDK content blocks back to JSON-serializable dicts."""
    out: list[dict[str, Any]] = []
    for block in content:
        if hasattr(block, "model_dump"):
            out.append(block.model_dump(mode="json", exclude_none=True))
        elif isinstance(block, dict):
            out.append(block)
    return out


def _now_utc_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def label_filing(
    *,
    client: _AnthropicLike,
    filing_path: Path,
    guidelines: str,
    guidelines_sha: str,
    model: str,
    dry_run: bool = False,
) -> LabelResult | LabelFailure:
    """Label one filing. Returns ``LabelResult`` on success or ``LabelFailure``."""
    filing_data = json.loads(filing_path.read_text(encoding="utf-8"))
    metadata = filing_data.get("metadata") or {}
    accession = str(metadata.get("accession_number", filing_path.stem))
    parsed_text = filing_data.get("parsed_text", "")
    acceptance_dt = metadata.get("acceptance_datetime") or "<unavailable>"

    if len(parsed_text) > MAX_INPUT_CHARS:
        logger.warning(
            "filing %s parsed_text length %d > MAX_INPUT_CHARS; truncating",
            accession,
            len(parsed_text),
        )
        parsed_text = parsed_text[:MAX_INPUT_CHARS] + "\n\n[...TRUNCATED]"

    user_block = USER_PROMPT_TEMPLATE.format(
        accession=accession,
        parsed_text=parsed_text,
        acceptance_datetime=acceptance_dt,
    )

    if dry_run:
        typer.echo("--- system prompt (first 400 chars) ---")
        typer.echo(guidelines[:400])
        typer.echo("--- user prompt (first 600 chars) ---")
        typer.echo(user_block[:600])

    cumulative_cost = 0.0
    in_tok_sum = 0
    out_tok_sum = 0
    cc_in_sum = 0
    cr_in_sum = 0

    def _bump_usage(usage: Any) -> None:
        nonlocal cumulative_cost, in_tok_sum, out_tok_sum, cc_in_sum, cr_in_sum
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cc_in = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cr_in = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        in_tok_sum += in_tok
        out_tok_sum += out_tok
        cc_in_sum += cc_in
        cr_in_sum += cr_in
        cumulative_cost += compute_call_cost(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cc_in,
            cache_read_input_tokens=cr_in,
            model=model,
        )

    # Turn 1.
    call_kwargs = _build_call_kwargs(model=model, guidelines=guidelines, user_block=user_block)
    resp = _call_with_retry(client, call_kwargs)
    _bump_usage(resp.usage)

    if dry_run:
        typer.echo("--- response content ---")
        typer.echo(json.dumps(_content_to_dicts(resp.content), indent=2)[:2000])

    extracted = _extract_tool_use(resp.content)
    if extracted is None:
        return LabelFailure(
            accession=accession,
            error="no tool_use block in response",
            raw_response={"content": _content_to_dicts(resp.content)},
            model=model,
            guidelines_sha=guidelines_sha,
            failed_at=_now_utc_iso(),
            input_tokens=in_tok_sum,
            output_tokens=out_tok_sum,
            cache_creation_input_tokens=cc_in_sum,
            cache_read_input_tokens=cr_in_sum,
            cost_usd=cumulative_cost,
        )

    tool_use_id, tool_input = extracted
    try:
        label = Filing8K.model_validate(tool_input)
        return LabelResult(
            accession=accession,
            label=label,
            model=model,
            guidelines_sha=guidelines_sha,
            labeled_at=_now_utc_iso(),
            input_tokens=in_tok_sum,
            output_tokens=out_tok_sum,
            cache_creation_input_tokens=cc_in_sum,
            cache_read_input_tokens=cr_in_sum,
            cost_usd=cumulative_cost,
            retried=False,
        )
    except ValidationError as exc:
        retry_message = _format_validation_errors(exc, tool_input)

    # Turn 2: feed the validation error back as a tool_result and try again.
    assistant_turn = {"role": "assistant", "content": _content_to_dicts(resp.content)}
    user_correction = {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "is_error": True,
                "content": retry_message,
            }
        ],
    }
    retry_kwargs = _build_call_kwargs(
        model=model,
        guidelines=guidelines,
        user_block=user_block,
        extra_turns=[assistant_turn, user_correction],
    )
    resp2 = _call_with_retry(client, retry_kwargs)
    _bump_usage(resp2.usage)

    extracted2 = _extract_tool_use(resp2.content)
    if extracted2 is None:
        return LabelFailure(
            accession=accession,
            error="retry returned no tool_use block",
            raw_response={"content": _content_to_dicts(resp2.content)},
            model=model,
            guidelines_sha=guidelines_sha,
            failed_at=_now_utc_iso(),
            input_tokens=in_tok_sum,
            output_tokens=out_tok_sum,
            cache_creation_input_tokens=cc_in_sum,
            cache_read_input_tokens=cr_in_sum,
            cost_usd=cumulative_cost,
        )
    _, tool_input2 = extracted2
    try:
        label = Filing8K.model_validate(tool_input2)
        return LabelResult(
            accession=accession,
            label=label,
            model=model,
            guidelines_sha=guidelines_sha,
            labeled_at=_now_utc_iso(),
            input_tokens=in_tok_sum,
            output_tokens=out_tok_sum,
            cache_creation_input_tokens=cc_in_sum,
            cache_read_input_tokens=cr_in_sum,
            cost_usd=cumulative_cost,
            retried=True,
        )
    except ValidationError as exc2:
        return LabelFailure(
            accession=accession,
            error=f"ValidationError after retry: {exc2}",
            raw_response={"content": _content_to_dicts(resp2.content)},
            model=model,
            guidelines_sha=guidelines_sha,
            failed_at=_now_utc_iso(),
            input_tokens=in_tok_sum,
            output_tokens=out_tok_sum,
            cache_creation_input_tokens=cc_in_sum,
            cache_read_input_tokens=cr_in_sum,
            cost_usd=cumulative_cost,
        )


def _call_with_retry(client: _AnthropicLike, kwargs: dict[str, Any]) -> Any:
    """Call ``client.messages.create(**kwargs)`` with tenacity-backed retry on 429/5xx."""
    from anthropic import APIConnectionError, InternalServerError, RateLimitError
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=2.0, max=60.0),
        retry=retry_if_exception_type(
            (RateLimitError, APIConnectionError, InternalServerError)
        ),
    )
    def _do() -> Any:
        return client.messages.create(**kwargs)

    return _do()


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------


def _serialize_label_row(result: LabelResult) -> dict[str, Any]:
    return {
        "accession": result.accession,
        "label": json.loads(result.label.model_dump_json()),
        "provenance": {
            "model": result.model,
            "guidelines_sha": result.guidelines_sha,
            "prompt_version": PROMPT_VERSION,
            "labeled_at": result.labeled_at,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cache_creation_input_tokens": result.cache_creation_input_tokens,
            "cache_read_input_tokens": result.cache_read_input_tokens,
            "cost_usd": round(result.cost_usd, 6),
            "retried": result.retried,
        },
    }


def _serialize_failure_row(failure: LabelFailure) -> dict[str, Any]:
    return {
        "accession": failure.accession,
        "error": failure.error,
        "raw_response": failure.raw_response,
        "model": failure.model,
        "guidelines_sha": failure.guidelines_sha,
        "failed_at": failure.failed_at,
        "cost_usd": round(failure.cost_usd, 6),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def label_batch(
    *,
    in_dir: Path,
    gold_path: Path,
    failures_path: Path,
    guidelines_path: Path,
    model: str,
    limit: int | None,
    explicit_accessions: list[str] | None,
    seed: int,
    max_cost_usd: float,
    dry_run: bool,
    rerun: bool,
    api_key: str,
    client: _AnthropicLike | None = None,
) -> LabelingSummary:
    """Run the labeler over a selected subset of filings.

    ``client`` is injected by tests; production callers leave it None and an
    ``Anthropic`` client is constructed from ``api_key``.
    """
    from anthropic import Anthropic

    guidelines = guidelines_path.read_text(encoding="utf-8")
    guidelines_sha = compute_guidelines_sha(guidelines_path)

    selected = select_filings(in_dir, limit, explicit_accessions, seed=seed)
    if not selected:
        logger.warning("no filings selected; nothing to do")
        now = datetime.now(UTC)
        return LabelingSummary(
            started_at=now, finished_at=now, model=model, guidelines_sha=guidelines_sha
        )

    # Pre-flight cost gate.
    estimate = estimate_cost(selected, model=model, guidelines_path=guidelines_path)
    typer.echo(estimate.render(), err=True)
    if estimate.est_total_usd > max_cost_usd:
        typer.echo(
            f"Estimated ${estimate.est_total_usd:.4f} exceeds --max-cost-usd "
            f"${max_cost_usd:.2f}. Refusing to run. Pass --max-cost-usd <higher> to override.",
            err=True,
        )
        raise typer.Exit(code=2)

    api_client: _AnthropicLike
    if client is None:
        timeout = float(os.environ.get("SEC8K_HTTP_TIMEOUT", "60.0"))
        headers: dict[str, str] = {}
        beta = os.environ.get("ANTHROPIC_BETA")
        if beta:
            headers["anthropic-beta"] = beta
        api_client = Anthropic(
            api_key=api_key, timeout=timeout, default_headers=headers or None
        )
    else:
        api_client = client

    latest_sha = load_latest_sha_per_accession(gold_path)

    started_at = datetime.now(UTC)
    summary = LabelingSummary(
        started_at=started_at,
        finished_at=started_at,
        model=model,
        guidelines_sha=guidelines_sha,
        per_category_succeeded={c: 0 for c in CATEGORIES},
    )

    abort_cap = max_cost_usd * 1.5

    try:
        for idx, filing_path in enumerate(selected, start=1):
            if not filing_path.exists():
                logger.warning("missing filing %s; skipping", filing_path)
                continue

            # Read accession + category once for skip / accounting.
            try:
                filing_data = json.loads(filing_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("could not read %s: %s; skipping", filing_path, exc)
                continue
            metadata = filing_data.get("metadata") or {}
            accession = str(metadata.get("accession_number", filing_path.stem))
            category = str(metadata.get("primary_category", ""))

            if not rerun and latest_sha.get(accession) == guidelines_sha:
                summary.n_skipped += 1
                continue

            summary.n_attempted += 1
            try:
                outcome = label_filing(
                    client=api_client,
                    filing_path=filing_path,
                    guidelines=guidelines,
                    guidelines_sha=guidelines_sha,
                    model=model,
                    dry_run=dry_run,
                )
            except Exception as exc:
                logger.exception("unexpected error labeling %s", accession)
                failure = LabelFailure(
                    accession=accession,
                    error=f"{type(exc).__name__}: {exc}",
                    raw_response=None,
                    model=model,
                    guidelines_sha=guidelines_sha,
                    failed_at=_now_utc_iso(),
                    cost_usd=0.0,
                )
                if not dry_run:
                    _append_jsonl(failures_path, _serialize_failure_row(failure))
                summary.n_failed += 1
                continue

            summary.total_input_tokens += outcome.input_tokens
            summary.total_output_tokens += outcome.output_tokens
            summary.total_cache_creation_input_tokens += outcome.cache_creation_input_tokens
            summary.total_cache_read_input_tokens += outcome.cache_read_input_tokens
            summary.total_cost_usd += outcome.cost_usd

            if isinstance(outcome, LabelResult):
                if not dry_run:
                    _append_jsonl(gold_path, _serialize_label_row(outcome))
                    latest_sha[accession] = guidelines_sha
                summary.n_succeeded += 1
                if category in summary.per_category_succeeded:
                    summary.per_category_succeeded[category] += 1
            else:
                if not dry_run:
                    _append_jsonl(failures_path, _serialize_failure_row(outcome))
                summary.n_failed += 1

            if idx % 10 == 0:
                typer.echo(
                    f"[{idx}/{len(selected)}] ok={summary.n_succeeded} "
                    f"failed={summary.n_failed} skipped={summary.n_skipped} "
                    f"cost=${summary.total_cost_usd:.4f}",
                    err=True,
                )

            if summary.total_cost_usd > abort_cap:
                typer.echo(
                    f"Cumulative cost ${summary.total_cost_usd:.4f} exceeded "
                    f"--max-cost-usd ${max_cost_usd:.2f} * 1.5; aborting cleanly.",
                    err=True,
                )
                break

            if dry_run:
                # Dry-run handles exactly one filing.
                break
    except KeyboardInterrupt:
        logger.warning(
            "interrupted; %d succeeded, %d failed, %d remaining",
            summary.n_succeeded,
            summary.n_failed,
            len(selected) - (summary.n_succeeded + summary.n_failed + summary.n_skipped),
        )

    summary.finished_at = datetime.now(UTC)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def cli(
    in_dir: Annotated[Path, typer.Option("--in-dir")] = RAW_DIR_DEFAULT,
    out: Annotated[Path, typer.Option("--out")] = GOLD_PATH_DEFAULT,
    failures: Annotated[Path, typer.Option("--failures")] = FAILURES_PATH_DEFAULT,
    guidelines: Annotated[Path, typer.Option("--guidelines")] = GUIDELINES_PATH,
    model: Annotated[str, typer.Option("--model")] = MODEL_DEFAULT,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Stratified random sample size; mutex with --accessions."),
    ] = None,
    accessions: Annotated[
        str | None,
        typer.Option("--accessions", help="Comma-separated accession numbers (with dashes)."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = SEED,
    max_cost_usd: Annotated[
        float,
        typer.Option("--max-cost-usd", help="Pre-flight cap; estimate > cap → refuse."),
    ] = 100.0,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="One filing, print prompt+response, no writes."),
    ] = False,
    rerun: Annotated[
        bool,
        typer.Option("--rerun", help="Force re-label even if guidelines SHA matches."),
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", envvar="ANTHROPIC_API_KEY"),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", envvar="SEC8K_LOG_LEVEL"),
    ] = None,
) -> None:
    """Label SEC 8-K filings under data/raw/edgar/ with Claude into data/gold/v1.jsonl."""
    logging.basicConfig(
        level=(log_level or "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if accessions and limit is not None:
        typer.echo("--accessions and --limit are mutually exclusive", err=True)
        raise typer.Exit(code=2)
    if not accessions and limit is None and not dry_run:
        typer.echo("Specify --limit N or --accessions a,b,c (or --dry-run)", err=True)
        raise typer.Exit(code=2)
    if not api_key or not api_key.strip():
        typer.echo("ANTHROPIC_API_KEY is not set (use --api-key or env).", err=True)
        raise typer.Exit(code=2)

    explicit = [a.strip() for a in accessions.split(",")] if accessions else None
    effective_limit = limit
    if dry_run and effective_limit is None and not explicit:
        effective_limit = 1

    summary = label_batch(
        in_dir=in_dir,
        gold_path=out,
        failures_path=failures,
        guidelines_path=guidelines,
        model=model,
        limit=effective_limit,
        explicit_accessions=explicit,
        seed=seed,
        max_cost_usd=max_cost_usd,
        dry_run=dry_run,
        rerun=rerun,
        api_key=api_key,
    )

    typer.echo(f"Succeeded: {summary.n_succeeded}/{summary.n_attempted}")
    typer.echo(f"Skipped (current SHA): {summary.n_skipped}")
    typer.echo(f"Failed: {summary.n_failed} (see {failures})")
    typer.echo(f"Per category: {summary.per_category_succeeded}")
    typer.echo(
        f"Tokens: input={summary.total_input_tokens} output={summary.total_output_tokens} "
        f"cache_write={summary.total_cache_creation_input_tokens} "
        f"cache_read={summary.total_cache_read_input_tokens}"
    )
    typer.echo(f"Total cost: ${summary.total_cost_usd:.4f}")


def main() -> None:
    """Script entry point: load .env and run the typer app."""
    load_dotenv()
    app()


if __name__ == "__main__":
    main()
