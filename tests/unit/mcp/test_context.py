"""Unit tests for zap_ai.mcp.context module."""

from unittest.mock import MagicMock

from fastmcp import Context

from zap_ai.mcp.context import get_zap_context, get_zap_context_value


def _create_mock_context(zap_context: dict | None = None) -> MagicMock:
    """Create a mock Context with optional zap_context in meta."""
    ctx = MagicMock(spec=Context)

    if zap_context is not None:
        # Meta is a Pydantic model with zap_context as an attribute
        meta = MagicMock()
        meta.zap_context = zap_context
        ctx.request_context.meta = meta
    else:
        ctx.request_context.meta = None

    return ctx


class TestGetZapContext:
    """Tests for get_zap_context function."""

    def test_returns_context_from_meta(self) -> None:
        """Test that context is extracted from request_context.meta."""
        ctx = _create_mock_context({"user_id": "user_123", "tenant": "acme"})

        result = get_zap_context(ctx)

        assert result == {"user_id": "user_123", "tenant": "acme"}

    def test_returns_empty_dict_when_no_meta(self) -> None:
        """Test that empty dict is returned when meta is None."""
        ctx = _create_mock_context(None)

        result = get_zap_context(ctx)

        assert result == {}

    def test_returns_empty_dict_when_no_request_context(self) -> None:
        """Test that empty dict is returned when request_context is None."""
        ctx = MagicMock(spec=Context)
        ctx.request_context = None

        result = get_zap_context(ctx)

        assert result == {}

    def test_returns_empty_dict_when_no_zap_context_attribute(self) -> None:
        """Test that empty dict is returned when zap_context attribute is missing."""
        ctx = MagicMock(spec=Context)
        # Create meta without zap_context attribute
        meta = MagicMock(spec=[])  # Empty spec means no attributes
        del meta.zap_context  # Ensure attribute doesn't exist
        ctx.request_context.meta = meta

        result = get_zap_context(ctx)

        assert result == {}

    def test_handles_complex_context(self) -> None:
        """Test that complex nested context structures are handled."""
        ctx = _create_mock_context(
            {
                "user_id": "user_123",
                "permissions": ["read", "write"],
                "metadata": {"source": "api", "version": "2.0"},
            }
        )

        result = get_zap_context(ctx)

        assert result["user_id"] == "user_123"
        assert result["permissions"] == ["read", "write"]
        assert result["metadata"] == {"source": "api", "version": "2.0"}


class TestGetZapContextValue:
    """Tests for get_zap_context_value function."""

    def test_returns_value_for_existing_key(self) -> None:
        """Test that the correct value is returned for an existing key."""
        ctx = _create_mock_context({"user_id": "user_123", "tenant": "acme"})

        result = get_zap_context_value(ctx, "user_id")

        assert result == "user_123"

    def test_returns_default_for_missing_key(self) -> None:
        """Test that default is returned when key doesn't exist."""
        ctx = _create_mock_context({"user_id": "user_123"})

        result = get_zap_context_value(ctx, "missing_key", "default_value")

        assert result == "default_value"

    def test_returns_none_as_default(self) -> None:
        """Test that None is returned as default when not specified."""
        ctx = _create_mock_context({"user_id": "user_123"})

        result = get_zap_context_value(ctx, "missing_key")

        assert result is None

    def test_returns_default_when_no_context(self) -> None:
        """Test that default is returned when no context exists."""
        ctx = MagicMock(spec=Context)
        ctx.request_context = None

        result = get_zap_context_value(ctx, "user_id", "default_user")

        assert result == "default_user"

    def test_returns_typed_default(self) -> None:
        """Test that typed defaults work correctly."""
        ctx = _create_mock_context({})

        # Integer default
        int_result = get_zap_context_value(ctx, "count", 0)
        assert int_result == 0
        assert isinstance(int_result, int)

        # List default
        list_result = get_zap_context_value(ctx, "items", [])
        assert list_result == []
        assert isinstance(list_result, list)

        # Dict default
        dict_result = get_zap_context_value(ctx, "config", {})
        assert dict_result == {}
        assert isinstance(dict_result, dict)

    def test_returns_actual_value_over_default(self) -> None:
        """Test that actual value is returned even if default is provided."""
        ctx = _create_mock_context({"count": 42, "items": ["a", "b"]})

        assert get_zap_context_value(ctx, "count", 0) == 42
        assert get_zap_context_value(ctx, "items", []) == ["a", "b"]
