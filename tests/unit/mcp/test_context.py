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

    @pytest.mark.parametrize(
        "ctx_setup,description",
        [
            (lambda: _create_mock_context(None), "no context"),
            (
                lambda: (
                    ctx := MagicMock(spec=Context),
                    setattr(ctx, "request_context", None),
                    ctx,
                )[2],
                "no request_context",
            ),
            (
                lambda: (
                    ctx := MagicMock(spec=Context),
                    setattr(ctx.request_context, "meta", (meta := MagicMock(spec=[]))),
                    delattr(meta, "zap_context"),
                    ctx,
                )[3],
                "no zap_context attribute",
            ),
        ],
        ids=["no_context", "no_request_context", "no_zap_context_attribute"],
    )
    def test_returns_empty_dict_for_missing_context(self, ctx_setup, description) -> None:
        """Should return empty dict when context is missing or incomplete."""
        ctx = ctx_setup()
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

    @pytest.mark.parametrize(
        "fixture_class,type_path,context_data",
        [
            (
                PydanticContextFixture,
                "tests.unit.mcp.test_context.PydanticContextFixture",
                {
                    "session_id": "session_123",
                    "user_name": "Alice",
                    "authenticated": True,
                },
            ),
            (
                DataclassContextFixture,
                "tests.unit.mcp.test_context.DataclassContextFixture",
                {
                    "user_id": "user_456",
                    "tenant": "acme_corp",
                    "permissions": ["read", "write"],
                },
            ),
        ],
        ids=["pydantic", "dataclass"],
    )
    def test_deserializes_typed_context_with_metadata(
        self, fixture_class, type_path, context_data
    ) -> None:
        """Should deserialize typed context when metadata is present."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": type_path,
            "__zap_context_version__": "1",
            **context_data,
        }

        result = ZapContext(ctx)

        assert isinstance(result, fixture_class)
        for key, value in context_data.items():
            assert getattr(result, key) == value

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

    def test_has_correct_dependency_signature(self) -> None:
        """Should have correct signature for dependency injection."""
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

    def test_creates_reusable_dependency(self) -> None:
        """Should create reusable dependency that can be called multiple times."""
        UserId = ZapContextValue("user_id", "anonymous")

        ctx1 = _create_mock_context({"user_id": "user_123"})
        ctx2 = _create_mock_context({"user_id": "user_456"})
        ctx3 = _create_mock_context({})

        assert UserId(ctx1) == "user_123"
        assert UserId(ctx2) == "user_456"
        assert UserId(ctx3) == "anonymous"

    @pytest.mark.parametrize(
        "context_data,key,default,expected",
        [
            ({"user_id": "user_123"}, "tenant", "default_tenant", "default_tenant"),
            ({"user_id": "user_123"}, "missing_key", None, None),
            ({}, "user_id", "default_user", "default_user"),
            ({"count": 42}, "count", 0, 42),
            ({"items": ["a", "b"]}, "items", [], ["a", "b"]),
        ],
        ids=[
            "missing_key_with_default",
            "missing_key_none_default",
            "no_context_with_default",
            "actual_value_over_default_int",
            "actual_value_over_default_list",
        ],
    )
    def test_returns_default_for_missing_keys(self, context_data, key, default, expected) -> None:
        """Should return default when key doesn't exist or return actual value when present."""
        ctx = _create_mock_context(context_data)
        get_value = ZapContextValue(key, default)
        result = get_value(ctx)
        assert result == expected

    @pytest.mark.parametrize(
        "context_data,key,default,expected_value,expected_type",
        [
            ({"count": 42}, "count", 0, 42, int),
            ({}, "count", 0, 0, int),
            ({"items": ["a", "b", "c"]}, "items", [], ["a", "b", "c"], list),
            ({}, "items", [], [], list),
            ({"config": {"key": "value"}}, "config", {}, {"key": "value"}, dict),
            ({}, "config", {}, {}, dict),
            ({"active": True}, "active", False, True, bool),
            ({}, "active", False, False, bool),
        ],
        ids=[
            "int_value",
            "int_default",
            "list_value",
            "list_default",
            "dict_value",
            "dict_default",
            "bool_value",
            "bool_default",
        ],
    )
    def test_works_with_different_types(
        self, context_data, key, default, expected_value, expected_type
    ) -> None:
        """Should work with different value types and typed defaults."""
        ctx = _create_mock_context(context_data)
        get_value = ZapContextValue(key, default)
        result = get_value(ctx)
        assert result == expected_value
        assert isinstance(result, expected_type)


class TestTypedZapContextDependency:
    """Tests for TypedZapContext dependency factory."""

    @pytest.mark.parametrize(
        "fixture_class,type_path,context_data",
        [
            (
                PydanticContextFixture,
                "tests.unit.mcp.test_context.PydanticContextFixture",
                {
                    "session_id": "session_999",
                    "user_name": "Bob",
                    "authenticated": False,
                },
            ),
            (
                DataclassContextFixture,
                "tests.unit.mcp.test_context.DataclassContextFixture",
                {
                    "user_id": "user_789",
                    "tenant": "acme",
                    "permissions": ["read", "write", "delete"],
                },
            ),
        ],
        ids=["pydantic", "dataclass"],
    )
    def test_deserializes_typed_models(self, fixture_class, type_path, context_data) -> None:
        """Should deserialize to Pydantic model or dataclass."""
        ctx = MagicMock(spec=Context)
        ctx.request_context.meta.zap_context = {
            "__zap_context_type__": type_path,
            "__zap_context_version__": "1",
            **context_data,
        }

        GetContext = TypedZapContext(fixture_class)
        result = GetContext(ctx)

        assert isinstance(result, fixture_class)
        for key, value in context_data.items():
            assert getattr(result, key) == value

    @pytest.mark.parametrize(
        "context_setup,error_match,description",
        [
            (
                lambda: {
                    "__zap_context_type__": "tests.unit.mcp.test_context.DataclassContextFixture",
                    "__zap_context_version__": "1",
                    "user_id": "user_123",
                    "tenant": "acme",
                    "permissions": [],
                },
                "type mismatch",
                "type_mismatch",
            ),
            (
                lambda: {
                    "__zap_context_type__": "nonexistent.module.FakeContext",
                    "__zap_context_version__": "1",
                    "data": "value",
                },
                "Could not import context type",
                "missing_type",
            ),
            (
                lambda: None,
                "no context was provided",
                "no_context",
            ),
            (
                lambda: {"user_id": "user_123", "tenant": "acme"},
                "no type metadata",
                "no_type_metadata",
            ),
        ],
        ids=["type_mismatch", "missing_type", "no_context", "no_type_metadata"],
    )
    def test_raises_on_invalid_context(self, context_setup, error_match, description) -> None:
        """Should raise TypeError for invalid context scenarios."""
        zap_context = context_setup()
        ctx = _create_mock_context(zap_context)

        GetSession = TypedZapContext(PydanticContextFixture)

        with pytest.raises(TypeError, match=error_match):
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
