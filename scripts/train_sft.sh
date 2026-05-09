#!/usr/bin/env bash
# Wrapper: launch QLoRA SFT via accelerate using src/sec8k/train/configs/sft.yaml.
set -euo pipefail

# accelerate launch -m sec8k.train.sft --config src/sec8k/train/configs/sft.yaml "$@"
echo "[train_sft.sh] stub — implement once src/sec8k/train/sft.py is filled in." >&2
exit 1
