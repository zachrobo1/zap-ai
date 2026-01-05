"""Tests for tracing global registry."""

import pytest

from zap_ai.tracing import (
    NoOpTracingProvider,
    get_tracing_provider,
    reset_tracing_provider,
    set_tracing_provider,
)
from zap_ai.tracing.noop_provider import NoOpTracingProvider as NoOpClass


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the global registry before and after each test."""
    reset_tracing_provider()
    yield
    reset_tracing_provider()


class TestTracingRegistry:
    """Tests for the global tracing registry."""

    def test_get_returns_noop_by_default(self):
        """get_tracing_provider should return NoOpTracingProvider by default."""
        provider = get_tracing_provider()
        assert isinstance(provider, NoOpClass)

    def test_set_and_get(self):
        """set_tracing_provider should set the global provider."""
        custom_provider = NoOpTracingProvider()
        set_tracing_provider(custom_provider)

        result = get_tracing_provider()
        assert result is custom_provider

    def test_reset_clears_provider(self):
        """reset_tracing_provider should clear the global provider."""
        custom_provider = NoOpTracingProvider()
        set_tracing_provider(custom_provider)

        reset_tracing_provider()

        # Should return a new NoOp instance, not the custom one
        result = get_tracing_provider()
        assert result is not custom_provider
        assert isinstance(result, NoOpClass)

    def test_subsequent_gets_return_same_instance(self):
        """Multiple get calls should return the same default instance."""
        first = get_tracing_provider()
        second = get_tracing_provider()
        assert first is second

    def test_set_overwrites_previous(self):
        """Setting a new provider should overwrite the previous one."""
        first_provider = NoOpTracingProvider()
        second_provider = NoOpTracingProvider()

        set_tracing_provider(first_provider)
        set_tracing_provider(second_provider)

        result = get_tracing_provider()
        assert result is second_provider
        assert result is not first_provider
