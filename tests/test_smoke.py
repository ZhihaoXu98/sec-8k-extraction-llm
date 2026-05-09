"""Smoke test: the package and every subpackage import cleanly without optional extras."""

import importlib


def test_imports() -> None:
    """Every module importable without [data], [train], [serve], [quant], [ui] installed."""
    for name in (
        "sec8k",
        "sec8k.schema",
        "sec8k.prompts",
        "sec8k.data",
        "sec8k.eval",
        "sec8k.train",
        "sec8k.serve",
        "sec8k.observability",
    ):
        importlib.import_module(name)
