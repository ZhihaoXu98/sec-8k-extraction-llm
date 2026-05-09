# Architecture

End-to-end pipeline:

1. **Ingestion** (`src/sec8k/data/`) — pull 8-K filings from EDGAR, parse Item sections, filter, split with no filer/time leakage.
2. **Training** (`src/sec8k/train/`) — QLoRA SFT, then DPO, on `Qwen/Qwen2.5-7B-Instruct`.
3. **Quantization** (`scripts/quantize_awq.py`) — merge LoRA into the base, then AWQ INT4 quantize for vLLM.
4. **Serving** (`src/sec8k/serve/`) — vLLM (OpenAI-compatible) + FastAPI gateway with JSON-schema-guided decoding via xgrammar.
5. **Eval** (`src/sec8k/eval/`) — field-level metrics, JSON-schema validity, sliced reports, optional LLM-as-judge for free-text fields.
6. **Observability** (`src/sec8k/observability/`) — Langfuse tracing across ingestion, training data prep, eval, and serving.

Detailed methodology, training recipes, and the eval rubric will be filled in as the corresponding modules land.
