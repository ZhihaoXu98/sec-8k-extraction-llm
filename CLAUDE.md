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

## Open questions / known TODOs
(Update as we go.)