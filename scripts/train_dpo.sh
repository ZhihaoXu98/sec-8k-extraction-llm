#!/usr/bin/env bash
# Wrapper: launch DPO via accelerate using src/sec8k/train/configs/dpo.yaml.
set -euo pipefail

# accelerate launch -m sec8k.train.dpo --config src/sec8k/train/configs/dpo.yaml "$@"
echo "[train_dpo.sh] stub — implement once src/sec8k/train/dpo.py is filled in." >&2
exit 1
