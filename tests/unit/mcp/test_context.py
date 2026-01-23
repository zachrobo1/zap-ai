"""Unit tests for zap_ai.mcp.context module."""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from fastmcp import Context
from pydantic import BaseModel

from zap_ai.mcp.context import (
    TypedZapContext,
    ZapContext,
    ZapContextValue,
)


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


# Test fixtures
@dataclass
class DataclassContextFixture:
    user_id: str
    tenant: str
    permissions: list[str]


class PydanticContextFixture(BaseModel):
    session_id: str
    user_name: str
    authenticated: bool


class TestZapContextDependency:
    """Tests for ZapContext dependency injection helper."""

    def test_returns_deserialized_dict_context(self) -> None:
        """Should return deserialized dict when used as dependency."""
        ctx = _create_mock_context({"user_id": "user_123", "tenant": "acme"})

        result = ZapContext(ctx)

        assert isinstance(result, dict)
        assert result == {"user_id": "user_123", "tenant": "acme"}

    def test_returns_empty_dict_when_no_context(self) -> None:
        """Should return empty dict when no context exists."""
        ctx = _create_mock_context(None)

        result = ZapContext(ctx)

        assert result == {}

    def test_returns_empty_dict_when_no_request_context(self) -> None:
        """Should return empty dict when request_context is None."""
        ctx = MagicMock(spec=Context)
        ctx.request_context = None

        result = ZapContext(ctx)

        assert result == {}

    def test_returns_empty_dict_when_no_zap_context_attribute(self) -> None:
        """Should return empty dict when zap_context attribute is missing."""
        ctx = MagicMock(spec=Context)
        # Create meta without zap_context attribute
        meta = MagicMock(spec=[])  # Empty spec means no attributes
        del meta.zap_context  # Ensure attribute doesn't exist
        ctx.request_context.meta = meta

        result = ZapContext(ctx)

        assert result == {}

    def test_handles_complex_context(self) -> None:
        """Should handle complex nested context structures."""
        ctx = _create_mock_context(
            {
                "user_id": "user_123",
                "permissions": ["read", "write"],
                "metadata": {"source": "api", "version": "2.0"},
            }
        )

        result = ZapContext(ctx)

        assert result["user_id"] == "user_123"
        assert result["permissions"] == ["read", "write"]
        assert result["metadata"] == {"source": "api", "version": "2.0"}

    def test_deserializes_typed_context_with_metadata(self) -> None:
        """Should deserialize typed context when metadata is present."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": "tests.unit.mcp.test_context.PydanticContextFixture",
            "__zap_context_version__": "1",
            "session_id": "session_123",
            "user_name": "Alice",
            "authenticated": True,
        }

        result = ZapContext(ctx)

        assert isinstance(result, PydanticContextFixture)
        assert result.session_id == "session_123"
        assert result.user_name == "Alice"
        assert result.authenticated is True

    def test_deserializes_dataclass_context_with_metadata(self) -> None:
        """Should deserialize dataclass context when metadata is present."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": "tests.unit.mcp.test_context.DataclassContextFixture",
            "__zap_context_version__": "1",
            "user_id": "user_456",
            "tenant": "acme_corp",
            "permissions": ["read", "write"],
        }

        result = ZapContext(ctx)

        assert isinstance(result, DataclassContextFixture)
        assert result.user_id == "user_456"
        assert result.tenant == "acme_corp"
        assert result.permissions == ["read", "write"]

    def test_returns_dict_when_type_cannot_be_imported(self) -> None:
        """Should return dict with warning when type can't be imported."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": "nonexistent.module.FakeContext",
            "__zap_context_version__": "1",
            "data": "value",
        }

        with pytest.warns(UserWarning, match="Could not import context type"):
            result = ZapContext(ctx)

        assert isinstance(result, dict)
        assert result == {"data": "value"}

    def test_works_without_explicit_ctx_parameter(self) -> None:
        """Should work with default CurrentContext() when no ctx provided."""
        # This tests that the function signature is correct for dependency injection
        import inspect

        sig = inspect.signature(ZapContext)
        params = list(sig.parameters.values())

        assert len(params) == 1
        assert params[0].name == "ctx"
        assert params[0].default is not inspect.Parameter.empty


class TestZapContextValueDependency:
    """Tests for ZapContextValue dependency factory."""

    def test_extracts_specific_value(self) -> None:
        """Should extract specific value from context."""
        ctx = _create_mock_context({"user_id": "user_123", "tenant": "acme"})

        get_user_id = ZapContextValue("user_id")
        result = get_user_id(ctx)

        assert result == "user_123"

    def test_returns_default_for_missing_key(self) -> None:
        """Should return default when key doesn't exist."""
        ctx = _create_mock_context({"user_id": "user_123"})

        get_tenant = ZapContextValue("tenant", "default_tenant")
        result = get_tenant(ctx)

        assert result == "default_tenant"

    def test_returns_none_as_default(self) -> None:
        """Should return None when no default specified."""
        ctx = _create_mock_context({"user_id": "user_123"})

        get_missing = ZapContextValue("missing_key")
        result = get_missing(ctx)

        assert result is None

    def test_creates_reusable_dependency(self) -> None:
        """Should create reusable dependency that can be called multiple times."""
        UserId = ZapContextValue("user_id", "anonymous")

        ctx1 = _create_mock_context({"user_id": "user_123"})
        ctx2 = _create_mock_context({"user_id": "user_456"})
        ctx3 = _create_mock_context({})

        assert UserId(ctx1) == "user_123"
        assert UserId(ctx2) == "user_456"
        assert UserId(ctx3) == "anonymous"

    def test_works_with_different_types(self) -> None:
        """Should work with different value types."""
        ctx = _create_mock_context(
            {
                "count": 42,
                "items": ["a", "b", "c"],
                "config": {"key": "value"},
                "active": True,
            }
        )

        get_count = ZapContextValue("count", 0)
        get_items = ZapContextValue("items", [])
        get_config = ZapContextValue("config", {})
        get_active = ZapContextValue("active", False)

        assert get_count(ctx) == 42
        assert get_items(ctx) == ["a", "b", "c"]
        assert get_config(ctx) == {"key": "value"}
        assert get_active(ctx) is True

    def test_returns_default_when_no_context(self) -> None:
        """Should return default when no context exists."""
        ctx = MagicMock(spec=Context)
        ctx.request_context = None

        get_user_id = ZapContextValue("user_id", "default_user")
        result = get_user_id(ctx)

        assert result == "default_user"

    def test_returns_typed_default(self) -> None:
        """Should work with typed defaults."""
        ctx = _create_mock_context({})

        # Integer default
        int_result = ZapContextValue("count", 0)(ctx)
        assert int_result == 0
        assert isinstance(int_result, int)

        # List default
        list_result = ZapContextValue("items", [])(ctx)
        assert list_result == []
        assert isinstance(list_result, list)

        # Dict default
        dict_result = ZapContextValue("config", {})(ctx)
        assert dict_result == {}
        assert isinstance(dict_result, dict)

    def test_returns_actual_value_over_default(self) -> None:
        """Should return actual value even if default is provided."""
        ctx = _create_mock_context({"count": 42, "items": ["a", "b"]})

        assert ZapContextValue("count", 0)(ctx) == 42
        assert ZapContextValue("items", [])(ctx) == ["a", "b"]

    def test_dependency_has_correct_signature(self) -> None:
        """Should have correct signature for dependency injection."""
        import inspect

        UserId = ZapContextValue("user_id")
        sig = inspect.signature(UserId)
        params = list(sig.parameters.values())

        assert len(params) == 1
        assert params[0].name == "ctx"
        assert params[0].default is not inspect.Parameter.empty


class TestTypedZapContextDependency:
    """Tests for TypedZapContext dependency factory."""

    def test_deserializes_pydantic_model(self) -> None:
        """Should deserialize to Pydantic model."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": "tests.unit.mcp.test_context.PydanticContextFixture",
            "__zap_context_version__": "1",
            "session_id": "session_999",
            "user_name": "Bob",
            "authenticated": False,
        }

        GetSession = TypedZapContext(PydanticContextFixture)
        result = GetSession(ctx)

        assert isinstance(result, PydanticContextFixture)
        assert result.session_id == "session_999"
        assert result.user_name == "Bob"
        assert result.authenticated is False

    def test_deserializes_dataclass(self) -> None:
        """Should deserialize to dataclass."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": "tests.unit.mcp.test_context.DataclassContextFixture",
            "__zap_context_version__": "1",
            "user_id": "user_789",
            "tenant": "acme",
            "permissions": ["read", "write", "delete"],
        }

        GetContext = TypedZapContext(DataclassContextFixture)
        result = GetContext(ctx)

        assert isinstance(result, DataclassContextFixture)
        assert result.user_id == "user_789"
        assert result.tenant == "acme"
        assert result.permissions == ["read", "write", "delete"]

    def test_raises_on_type_mismatch(self) -> None:
        """Should raise TypeError when context type doesn't match expected."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": "tests.unit.mcp.test_context.DataclassContextFixture",
            "__zap_context_version__": "1",
            "user_id": "user_123",
            "tenant": "acme",
            "permissions": [],
        }

        GetSession = TypedZapContext(PydanticContextFixture)

        with pytest.raises(TypeError, match="type mismatch"):
            GetSession(ctx)

    def test_raises_on_missing_type(self) -> None:
        """Should raise TypeError when type can't be imported."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": "nonexistent.module.FakeContext",
            "__zap_context_version__": "1",
            "data": "value",
        }

        GetContext = TypedZapContext(PydanticContextFixture)

        with pytest.raises(TypeError, match="Could not import context type"):
            GetContext(ctx)

    def test_raises_when_no_context_provided(self) -> None:
        """Should raise TypeError when no context is provided."""
        ctx = _create_mock_context(None)

        GetSession = TypedZapContext(PydanticContextFixture)

        with pytest.raises(TypeError, match="no context was provided"):
            GetSession(ctx)

    def test_raises_when_context_has_no_type_metadata(self) -> None:
        """Should raise TypeError when context has no type metadata."""
        ctx = _create_mock_context({"user_id": "user_123", "tenant": "acme"})

        GetSession = TypedZapContext(PydanticContextFixture)

        with pytest.raises(TypeError, match="no type metadata"):
            GetSession(ctx)

    def test_creates_reusable_typed_dependency(self) -> None:
        """Should create reusable typed dependency."""
        CurrentSession = TypedZapContext(PydanticContextFixture)

        ctx1 = MagicMock(spec=Context)
        ctx1.request_context.meta.zap_context = {
            "__zap_context_type__": "tests.unit.mcp.test_context.PydanticContextFixture",
            "__zap_context_version__": "1",
            "session_id": "session_1",
            "user_name": "Alice",
            "authenticated": True,
        }

        ctx2 = MagicMock(spec=Context)
        ctx2.request_context.meta.zap_context = {
            "__zap_context_type__": "tests.unit.mcp.test_context.PydanticContextFixture",
            "__zap_context_version__": "1",
            "session_id": "session_2",
            "user_name": "Bob",
            "authenticated": False,
        }

        result1 = CurrentSession(ctx1)
        result2 = CurrentSession(ctx2)

        assert result1.session_id == "session_1"
        assert result1.user_name == "Alice"
        assert result2.session_id == "session_2"
        assert result2.user_name == "Bob"

    def test_dependency_has_correct_signature(self) -> None:
        """Should have correct signature for dependency injection."""
        import inspect

        GetSession = TypedZapContext(PydanticContextFixture)
        sig = inspect.signature(GetSession)
        params = list(sig.parameters.values())

        assert len(params) == 1
        assert params[0].name == "ctx"
        assert params[0].default is not inspect.Parameter.empty
