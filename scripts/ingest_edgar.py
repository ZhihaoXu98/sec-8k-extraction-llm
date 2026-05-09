"""CLI: discover, download, and parse 8-K filings for a date range and form-type filter.

Wraps :mod:`sec8k.data.ingest` behind a Typer entry point. Output is written under
``data/`` (gitignored) as parquet plus a manifest JSON.
"""
