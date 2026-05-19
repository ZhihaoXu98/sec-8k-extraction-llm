"""Test fixtures local to :mod:`sec8k.data`."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from sec8k.data.edgar import EdgarClient, RateLimiter

TEST_USER_AGENT = "Test User test-runner@example.com"


@pytest.fixture
def test_user_agent() -> str:
    return TEST_USER_AGENT


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable time.sleep so rate-limiter and tenacity backoffs don't slow tests."""
    monkeypatch.setattr("time.sleep", lambda _seconds: None)


@pytest.fixture
def make_edgar_client(
    test_user_agent: str,
) -> Callable[[Callable[[httpx.Request], httpx.Response]], EdgarClient]:
    """Factory that builds an EdgarClient against an in-process mock transport."""

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> EdgarClient:
        return EdgarClient(
            user_agent=test_user_agent,
            rps=100000.0,
            limiter=RateLimiter(rps=100000.0),
            transport=httpx.MockTransport(handler),
        )

    return factory
