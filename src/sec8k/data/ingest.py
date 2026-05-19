"""End-to-end ingestion orchestrator.

Wires :mod:`edgar` (discovery + download), :mod:`parse`, :mod:`filter`, and
:mod:`splits` into a single pipeline and persists artefacts (parquet/jsonl) under
``data/`` (gitignored).
"""
