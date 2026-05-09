"""CLI: produce DPO ``(chosen, rejected)`` preference pairs.

Pairs are derived from judge-score deltas or rule-based perturbations of a chosen
output. Writes ``pairs.jsonl`` under ``data/dpo/`` (gitignored).
"""
