"""CLI: discover, download, and parse 8-K filings for a date range and form-type filter.

Wraps :mod:`sec8k.data.ingest_edgar` behind a Typer entry point. Output is written under
``data/raw/edgar/`` (gitignored) as one JSON per filing plus an append-only manifest.
"""

from sec8k.data.ingest_edgar import main

if __name__ == "__main__":
    main()
