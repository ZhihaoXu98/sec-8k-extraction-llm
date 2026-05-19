"""Shared pytest fixtures.

Provides a tiny EDGAR sample, a stubbed vLLM client, and an isolated temporary config
so unit tests stay hermetic and never hit the network or the GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def edgar_fixtures_dir() -> Path:
    """Filesystem path to recorded EDGAR fixture files."""
    return Path(__file__).parent / "data" / "fixtures" / "edgar"
