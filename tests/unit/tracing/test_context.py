"""Tests for tracing context propagation."""

import pytest

from zap_ai.tracing import (
    TraceContext,
    clear_current_trace_context,
    get_current_trace_context,
    set_current_trace_context,
)


@pytest.fixture(autouse=True)
def reset_context():
    """Reset the context before and after each test."""
    clear_current_trace_context()
    yield
    clear_current_trace_context()


class TestTraceContextPropagation:
    """Tests for trace context propagation functions."""

    def test_get_returns_none_by_default(self):
        """get_current_trace_context should return None when not set."""
        result = get_current_trace_context()
        assert result is None

    def test_set_and_get(self):
        """set_current_trace_context should set the context."""
        ctx = TraceContext(trace_id="trace-123", span_id="span-456")
        set_current_trace_context(ctx)

        result = get_current_trace_context()
        assert result is ctx
        assert result.trace_id == "trace-123"
        assert result.span_id == "span-456"

    def test_clear_resets_context(self):
        """clear_current_trace_context should reset to None."""
        ctx = TraceContext(trace_id="trace-123", span_id="span-456")
        set_current_trace_context(ctx)

        clear_current_trace_context()

        result = get_current_trace_context()
        assert result is None

    def test_set_overwrites_previous(self):
        """Setting a new context should overwrite the previous one."""
        first_ctx = TraceContext(trace_id="trace-1", span_id="span-1")
        second_ctx = TraceContext(trace_id="trace-2", span_id="span-2")

        set_current_trace_context(first_ctx)
        set_current_trace_context(second_ctx)

        result = get_current_trace_context()
        assert result is second_ctx
        assert result is not first_ctx

    def test_multiple_gets_return_same_instance(self):
        """Multiple get calls should return the same context instance."""
        ctx = TraceContext(trace_id="trace-123", span_id="span-456")
        set_current_trace_context(ctx)

        first = get_current_trace_context()
        second = get_current_trace_context()
        assert first is second
