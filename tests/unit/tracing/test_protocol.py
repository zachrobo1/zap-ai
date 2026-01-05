"""Tests for tracing protocol and types."""

import pytest

from zap_ai.tracing.protocol import ObservationType, TraceContext


class TestObservationType:
    """Tests for ObservationType enum."""

    def test_observation_types_exist(self):
        """All expected observation types should exist."""
        assert ObservationType.SPAN == "span"
        assert ObservationType.GENERATION == "generation"
        assert ObservationType.TOOL == "tool"
        assert ObservationType.AGENT == "agent"

    def test_observation_type_is_string(self):
        """ObservationType should be usable as a string."""
        # The value should be the string
        assert ObservationType.SPAN.value == "span"
        assert ObservationType.GENERATION.value == "generation"
        # Should be comparable to strings
        assert ObservationType.SPAN == "span"
        assert ObservationType.TOOL == "tool"

    def test_observation_type_values(self):
        """All values should match Langfuse types."""
        values = [t.value for t in ObservationType]
        assert "span" in values
        assert "generation" in values
        assert "tool" in values
        assert "agent" in values


class TestTraceContext:
    """Tests for TraceContext dataclass."""

    def test_create_minimal(self):
        """Create context with required fields only."""
        ctx = TraceContext(trace_id="trace-123", span_id="span-456")
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.provider_data is None

    def test_create_with_provider_data(self):
        """Create context with provider data."""
        ctx = TraceContext(
            trace_id="trace-123",
            span_id="span-456",
            provider_data={"langfuse_id": "abc123"},
        )
        assert ctx.provider_data == {"langfuse_id": "abc123"}

    def test_to_dict(self):
        """Serialize context to dict."""
        ctx = TraceContext(
            trace_id="trace-123",
            span_id="span-456",
            provider_data={"key": "value"},
        )
        result = ctx.to_dict()
        assert result == {
            "trace_id": "trace-123",
            "span_id": "span-456",
            "provider_data": {"key": "value"},
        }

    def test_to_dict_without_provider_data(self):
        """Serialize context without provider data."""
        ctx = TraceContext(trace_id="trace-123", span_id="span-456")
        result = ctx.to_dict()
        assert result == {
            "trace_id": "trace-123",
            "span_id": "span-456",
            "provider_data": None,
        }

    def test_from_dict(self):
        """Deserialize context from dict."""
        data = {
            "trace_id": "trace-123",
            "span_id": "span-456",
            "provider_data": {"key": "value"},
        }
        ctx = TraceContext.from_dict(data)
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.provider_data == {"key": "value"}

    def test_from_dict_without_provider_data(self):
        """Deserialize context without provider data."""
        data = {
            "trace_id": "trace-123",
            "span_id": "span-456",
        }
        ctx = TraceContext.from_dict(data)
        assert ctx.trace_id == "trace-123"
        assert ctx.span_id == "span-456"
        assert ctx.provider_data is None

    def test_roundtrip(self):
        """Context should survive serialization roundtrip."""
        original = TraceContext(
            trace_id="trace-abc",
            span_id="span-def",
            provider_data={"nested": {"key": "value"}},
        )
        serialized = original.to_dict()
        restored = TraceContext.from_dict(serialized)
        assert restored.trace_id == original.trace_id
        assert restored.span_id == original.span_id
        assert restored.provider_data == original.provider_data

    def test_from_dict_missing_required_fields(self):
        """Deserialize with missing required fields should raise."""
        with pytest.raises(KeyError):
            TraceContext.from_dict({"trace_id": "trace-123"})

        with pytest.raises(KeyError):
            TraceContext.from_dict({"span_id": "span-456"})
