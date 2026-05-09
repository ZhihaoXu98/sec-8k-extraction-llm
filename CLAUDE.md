# CLAUDE.md

Instructions for AI agents (and humans) working in this repository.

## Project Overview

`sec8k` ingests SEC 8-K filings from EDGAR, fine-tunes Qwen 2.5 7B Instruct with LoRA SFT then DPO, serves the resulting model via vLLM with AWQ INT4 quantization on a single RTX 4070 (12 GB), and exposes structured extraction through a FastAPI service. A Streamlit demo UI lives under `frontend/`.

## Repository State

The repo is currently at the **docstring-only skeleton** stage: every Python file under `src/sec8k/`, `tests/`, and `scripts/` is a stub with only a module docstring and no implementation logic. The dependency manifest, tooling (ruff, mypy strict, pytest), CI workflow, and pre-commit hooks are wired up and green; module bodies are empty.

`tests/test_smoke.py` is the one real test — it imports every subpackage to guard against accidental top-level GPU/heavy-dep imports as modules get filled in.

When you start implementing a module, replace its docstring-only body with code, add real tests under `tests/` mirroring the source path, and refresh this section if the state of the repo changes meaningfully (for example once SFT is functional or the FastAPI server boots end-to-end).

## Build & Setup

First-time setup:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Common Make targets:

| Target | Action |
|---|---|
| `make install` | Main + `[dev]` (lint, type, test tools) |
| `make install-data` | Adds `[data]` (beautifulsoup4, lxml, tenacity, ...) for ingestion work |
| `make install-train` | Adds `[train]` (torch, transformers, peft, trl, bitsandbytes, datasets) |
| `make install-serve` | Adds `[serve]` (vllm, xgrammar, fastapi, ...) and `[quant]` (autoawq) |
| `make install-ui` | Adds `[ui]` (streamlit) |
| `make check` | `lint` + `format-check` + `typecheck` + `test` |
| `make ingest` / `build-sft` / `build-dpo` / `train-sft` / `train-dpo` / `quantize` / `serve` / `eval` / `ui` | Pipeline stages |

## Project Conventions

- **Layout**: `src/` layout, single package `sec8k`. Import as `from sec8k.<module> import …`.
- **Python**: 3.11 only. `requires-python = ">=3.11,<3.12"`.
- **Typing**: type hints required everywhere; `mypy --strict` is the gate in CI.
- **Validation**: all I/O — function arguments at module boundaries, FastAPI bodies, eval records — uses Pydantic v2 models declared in `src/sec8k/schema.py`.
- **Prompts**: prompt templates live in `src/sec8k/prompts.py` with explicit version strings; prompt version is included in every Langfuse trace.
- **Configs**: runtime YAML under `config/`; training YAML under `src/sec8k/train/configs/`. Layering rule: `default.yaml` → env vars → CLI overrides.
- **Tests**: pytest under `tests/`; layout mirrors `src/sec8k/` (e.g., `tests/data/test_parse.py` covers `src/sec8k/data/parse.py`). GPU tests are marked `@pytest.mark.gpu` and skipped in CI.
- **No GPU code on import path**: importing any `sec8k.*` module must not require torch/vllm. Heavy imports happen inside functions.

## Locked Decisions

| Area | Choice |
|---|---|
| Python | 3.11 |
| Package manager | `uv` |
| Layout | `src/` layout, package `sec8k` |
| Lint + format | Ruff |
| Type check | `mypy --strict` |
| Test runner | pytest (+ asyncio, cov) |
| Validation | Pydantic v2 |
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Post-training | LoRA SFT then DPO via TRL + PEFT (QLoRA on a 4070) |
| Inference quant | AWQ INT4 via `autoawq` |
| Serving | vLLM (OpenAI-compatible) behind a FastAPI gateway |
| Constrained decoding | vLLM structured outputs (xgrammar backend) |
| Frontend | Streamlit (`frontend/app.py`) |
| Observability | Langfuse |
| CI | GitHub Actions (lint + typecheck + CPU tests) |
| License | MIT |

Any change to this table lands in its own PR with rationale.

## Common Workflows

- **Ingest**: `make ingest` — discover, download, parse 8-Ks; persists under `data/` (gitignored).
- **Build SFT**: `make build-sft` — JSONL of (prompt, completion) pairs.
- **Train SFT**: `make train-sft` — QLoRA SFT on Qwen 2.5 7B.
- **Build DPO**: `make build-dpo` — preference pairs from judge or rule perturbations.
- **Train DPO**: `make train-dpo` — DPO over the SFT adapter.
- **Quantize**: `make quantize` — merge LoRA, AWQ INT4 quantize for vLLM.
- **Serve**: `make serve` — vLLM + FastAPI together (`scripts/serve_local.sh`).
- **Eval**: `make eval` — drive harness against a configured endpoint, write timestamped report under `eval_results/`.
- **UI**: `make ui` — Streamlit demo.

## Hardware Notes

- Inference target: single RTX 4070 (12 GB).
- **Training is QLoRA only** on a 4070 (4-bit base via `bitsandbytes` + LoRA adapters). Full-precision 7B SFT requires an A100/H100 — run remotely.
- AWQ INT4 of Qwen 2.5 7B fits comfortably in 12 GB for low-concurrency serving.

## File-Path Map

- Schemas → `src/sec8k/schema.py`
- Prompts → `src/sec8k/prompts.py`
- Ingestion → `src/sec8k/data/`
- Eval harness → `src/sec8k/eval/`
- Training → `src/sec8k/train/` (+ `configs/{sft,dpo}.yaml`)
- Serving → `src/sec8k/serve/`
- Observability → `src/sec8k/observability/`
- Runtime config → `config/default.yaml`
- CLI entry points → `scripts/`
- Demo UI → `frontend/app.py`

## Things to Avoid

- Committing anything under `data/`, `models/`, or `eval_results/` (all gitignored).
- Committing `.env` or any secret material — `.env.example` is the only doc.
- Changing `src/sec8k/schema.py` without a migration note in the PR body.
- Hardcoding paths or URLs that belong in `config/default.yaml` or env vars.
- Loading torch/vllm at module import time — these belong inside function bodies.
- Bypassing `mypy --strict` with broad `# type: ignore`.
- Removing the `exit 1` line from a `scripts/*.sh` stub without replacing it with the real launch command — these stubs fail loudly on purpose so `make train-sft`/`make serve` doesn't silently no-op while the underlying module is still empty.
