"""CLI: turn parsed 8-Ks into SFT (prompt, completion) JSONL.

Optionally calls :mod:`sec8k.data.synthesize` to bootstrap labels with a teacher model.
Writes ``train.jsonl`` / ``val.jsonl`` / ``test.jsonl`` under ``data/sft/`` (gitignored).
"""
