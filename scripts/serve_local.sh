#!/usr/bin/env bash
# Wrapper: start vLLM (AWQ artefact) and the FastAPI gateway side by side.
set -euo pipefail

# vllm serve "$MODEL_AWQ_DIR" --quantization awq --port 8000 &
# uvicorn sec8k.serve.app:app --host 0.0.0.0 --port 8001
echo "[serve_local.sh] stub — implement once src/sec8k/serve/app.py is filled in." >&2
exit 1
