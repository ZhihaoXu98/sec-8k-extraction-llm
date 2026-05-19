# SEC 8-K Extraction LLM — Project Context

## What this project is
A specialized 7B LLM (Qwen 2.5 7B Instruct, fine-tuned with LoRA SFT + DPO)
that extracts structured JSON from SEC 8-K filings. Production deployed via
vLLM with AWQ INT4 on an RTX 4070 (locally), behind a FastAPI service.

## Build / dev commands
- Install: `uv pip install -e ".[dev]"`
- Test: `pytest tests/`
- Lint+format: `ruff check . && ruff format .`
- Type-check: `mypy src/`
- Run eval: `python -m sec8k.eval.runner --model <name> --gold data/gold/v1.jsonl`
- Label gold set: `python scripts/llm_label_gold_set.py --limit 300 --max-cost-usd 50`
- Verify gold set: `python scripts/verify_calibration.py --limit 30`

## Conventions
- Python 3.11, strict typing on every function
- Pydantic v2 for all schemas; use `X | None`, not `Optional[X]`
- Random seed = 42 everywhere
- All paths absolute or relative to repo root
- Eval results: `eval_results/<run>/<UTC-timestamp>/`
- Never commit data/, models/, .env (in .gitignore)
- Money: floats are dollars unless field name says cents

## Key design decisions (locked, do not change without asking)
- Schema in `src/sec8k/schema.py` is frozen at end of week 1
- Gold set `data/gold/v1.jsonl` is frozen at end of week 2
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Inference: AWQ INT4 + vLLM + guided_json on local RTX 4070

## SEC EDGAR usage
- Always set User-Agent: "<your name> <your email>" per SEC fair-use policy
- Rate limit to <=10 req/sec to EDGAR
- Filings stored locally in data/raw/edgar/, never re-fetched

## Things you (Claude) often get wrong on this project
(Update this section as we go. One bullet per repeated mistake.)
- Pydantic v2 `str_strip_whitespace=True` strips `list[str]` elements recursively,
  not just top-level string fields. Don't add a manual `.strip()` in a
  `field_validator` unless you also need other normalization (dedup, drop-empties).
- (week 1) `filing_date` defaults to the body's "Date of Report" line — which is
  `event_date`, not the EDGAR `<ACCEPTANCE-DATETIME>` stamp. The two often differ
  by 1-6 days. The labeler's user prompt (`PROMPT_VERSION="v2"`) now exposes
  `acceptance_datetime` explicitly, which fixed all 5 observed cases. When labeling
  by hand, the signature block "Date: April 30, 2025 /s/ …" is conventionally the
  acceptance date; "Date of Report (Date of earliest event reported)" is the event.
- (week 1) `primary_category` substance overrides: Claude applies "this filing is
  *really* about X" reasoning even when items mapping to a higher-priority category
  are present. The category-priority hierarchy is binding — only Item 1.01
  m_and_a/material_agreement disambiguation and the explicit "M&A solely under
  Item 8.01" carve-out are allowed substance reads.
- (week 1) `expected_impact_period` over-prefers `"undisclosed"` for governance-only
  filings (auditor change, registered-agent change, bylaw amendments with no comp
  effects). Per spec these should be `null` ("no money event"), not `"undisclosed"`
  ("money event with TBD timing"). Even after a doc rule + v2 prompt, ~20% of
  governance cases still mispredict — doc still under-specifies the boundary.
- (week 1) `filer_ticker` literal `"None"` string: when the cover page says
  "Securities registered pursuant to Section 12(b) of the Act: None", the LLM
  sometimes writes the string `"None"` rather than `null`. Pydantic v2 accepts
  `"None"` as a valid `str` so schema validation doesn't catch it — surface it
  as a doc/prompt rule, not a schema constraint.

## Week 1 status (2026-05-19)
- Schema locked at 14 fields, see `src/sec8k/schema.py`. Frozen per week-1 design decision.
- 600 raw 8-K filings ingested from EDGAR (`data/raw/edgar/`), stratified at 100 per `primary_category` across April 2025. The pipeline is resumable, rate-limited at 8 rps with a polite User-Agent, and SGML-envelope-aware.
- LLM-labeling pipeline live: Claude Sonnet 4.6 + Anthropic prompt caching on the labeling guidelines doc (~14K-token cached prefix; ~99% hit rate at session-length). Per-filing cost ~$0.018 on a warm cache.
- 50 filings auto-labeled into `data/gold/v1.jsonl` (calibration subset). 30 of those reviewed in an LLM critical-review pass by Claude Opus 4.7 (`provenance.verified_by = "claude-opus-4-7"`, `verification_type = "llm_critical_review"`) — **not** a human pass. Project owner lacks SEC-finance domain depth, so LLM-vs-LLM cross-pass consistency stands in for human dual-labeling at this stage; the spec's `kappa ≥ 0.70` agreement target is satisfied loosely rather than strictly.
- Doc dual-role: `docs/labeling_guidelines.md` is both the labeler's **system prompt** (cached with `cache_control: ephemeral`) and the **verification spec** the reviewer applies. SHA recorded in every label's provenance so a doc revision invalidates the cache and triggers re-labeling on next run.
- Provenance metadata per label: `model`, `prompt_version`, `guidelines_sha`, `labeled_at`, `input_tokens`/`output_tokens`/`cache_creation_input_tokens`/`cache_read_input_tokens`, `cost_usd`, `retried`. Verification pass adds `verified_by`, `verification_type`, `verified_at`, `verification_session_id`, `fields_changed`, `fields_ambiguous`, `notes`, `duration_seconds`, `llm_label_snapshot` (preserves pre-verification label verbatim for downstream agreement metrics).
- Project framing: **distillation** of Claude Sonnet 4.6 (the labeler) into a local Qwen 2.5 7B (AWQ INT4 on RTX 4070) at well under 1% of Claude API inference cost. Claude generates the gold set; Qwen learns to mimic the extraction at near-zero marginal cost in production.
- Week-1 audit cycle (doc → label → review → doc → relabel) executed and validated: of 18 audit findings on the first calibration pass, 14 (78%) were resolved by sharpening `docs/labeling_guidelines.md` + the v2 prompt and re-running the labeler. Remaining 4 are minor field-level disagreements documented above.

## Open questions / known TODOs
(Update as we go.)