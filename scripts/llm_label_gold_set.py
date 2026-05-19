"""CLI: label SEC 8-K filings with Claude into data/gold/v1.jsonl.

Wraps :mod:`sec8k.data.llm_label` behind a Typer entry point. Reads ingested
filings from ``data/raw/edgar/`` and appends provenanced labels to
``data/gold/v1.jsonl``; validation failures land in ``data/gold/failures.jsonl``.
"""

from sec8k.data.llm_label import main

if __name__ == "__main__":
    main()
