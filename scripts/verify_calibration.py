"""CLI: human-in-the-loop verification of LLM gold-set labels.

Wraps :mod:`sec8k.data.verify` behind a Typer entry point. Walks unverified
rows from ``data/gold/v1.jsonl``, prompts per-field keep/edit/ambig, and
appends a verified row with provenance back to the same file (latest-wins).
"""

from sec8k.data.verify import main

if __name__ == "__main__":
    main()
