"""
FastMCP context injection utilities for MCP tools.

This module provides dependency injection helpers for accessing ZapContext within MCP tools.
Context is passed from the Zap workflow via FastMCP's `meta` parameter and automatically
injected into your tool functions using FastMCP's Depends() system.

Basic Usage (Dict Context):
    ```python
    from fastmcp import FastMCP
    from fastmcp.dependencies import Depends
    from zap_ai.mcp.context import ZapContext

    mcp = FastMCP("MyService")

    @mcp.tool()
    async def my_tool(
        query: str,
        zap_ctx: dict = Depends(ZapContext)
    ) -> str:
        user_id = zap_ctx.get("user_id")
        # zap_ctx is automatically injected and hidden from LLM schema
        ...
    ```

Extracting Specific Values:
    ```python
    from fastmcp import FastMCP
    from fastmcp.dependencies import Depends
    from zap_ai.mcp.context import ZapContextValue

    mcp = FastMCP("MyService")

    # Create reusable dependencies
    UserId = ZapContextValue("user_id")
    Tenant = ZapContextValue("tenant", "default")

    @mcp.tool()
    async def my_tool(
        query: str,
        user_id: str | None = Depends(UserId),
        tenant: str = Depends(Tenant)
    ) -> str:
        # Individual values are automatically extracted
        ...
    ```

Typed Context (Pydantic or Dataclasses):
    ```python
    from dataclasses import dataclass
    from fastmcp import FastMCP
    from fastmcp.dependencies import Depends
    from zap_ai.mcp.context import TypedZapContext

    @dataclass
    class UserSession:
        user_id: str
        tenant: str

    mcp = FastMCP("MyService")
    CurrentSession = TypedZapContext(UserSession)

    @mcp.tool()
    async def my_tool(
        query: str,
        session: UserSession = Depends(CurrentSession)
    ) -> str:
        # session is fully typed with IDE autocomplete
        orders = await db.query(user_id=session.user_id)
        ...
    ```
"""

import importlib
import warnings
from dataclasses import is_dataclass
from typing import Any, Callable, TypeVar

from fastmcp import Context
from fastmcp.server.dependencies import CurrentContext

T = TypeVar("T")


# Dependency Injection Helpers


def ZapContext(ctx: Context = CurrentContext()) -> dict[str, Any]:
    """
    FastMCP dependency that deserializes ZapContext.

    Use this with `Depends()` to automatically inject the deserialized
    ZapContext into your tool functions without manual extraction.

    Returns:
        Deserialized ZapContext dict (empty dict if no context).

    Example:
        ```python
        from fastmcp import FastMCP
        from fastmcp.dependencies import Depends
        from zap_ai.mcp.context import ZapContext

        mcp = FastMCP("MyService")

        @mcp.tool()
        async def get_orders(
            limit: int = 10,
            zap_ctx: dict = Depends(ZapContext)
        ) -> str:
            '''Get orders for the current user.'''
            user_id = zap_ctx.get("user_id")
            if not user_id:
                return "Error: No user context provided"
            orders = await db.query_orders(user_id=user_id, limit=limit)
            return format_orders(orders)
        ```
    """
    # Extract raw context from FastMCP meta
    if not ctx.request_context or not ctx.request_context.meta:
        return {}

    raw_context = getattr(ctx.request_context.meta, "zap_context", {}) or {}

    if not raw_context:
        return {}

    # Check for type metadata
    type_id = raw_context.get("__zap_context_type__")

    # No metadata - return as plain dict (backward compatible)
    if not type_id:
        return raw_context

    # Extract data (remove metadata keys)
    data = {k: v for k, v in raw_context.items() if not k.startswith("__zap_context_")}

    # Import and reconstruct type
    try:
        # Parse module and class name
        module_name, class_name = type_id.rsplit(".", 1)
        module = importlib.import_module(module_name)
        context_class = getattr(module, class_name)

        # Reconstruct based on type
        if hasattr(context_class, "model_validate"):  # Pydantic v2
            return context_class.model_validate(data)
        elif is_dataclass(context_class):
            return context_class(**data)
        else:
            warnings.warn(
                f"Context type '{type_id}' does not support deserialization. Returning dict."
            )
            return data

    except (ImportError, AttributeError, ValueError) as e:
        # Type not found - log warning and return dict
        warnings.warn(f"Could not import context type '{type_id}': {e}. Returning dict.")
        return data


def ZapContextValue(key: str, default: Any = None) -> Callable[[Context], Any]:
    """
    Factory for FastMCP dependencies that extract specific ZapContext values.

    Use this to create dependencies that extract individual values from
    the context, reducing boilerplate in tool functions.

    Args:
        key: The key to extract from ZapContext.
        default: Default value if key is not present.

    Returns:
        A dependency function that extracts the specified value.

    Example:
        ```python
        from fastmcp import FastMCP
        from fastmcp.dependencies import Depends
        from zap_ai.mcp.context import ZapContextValue

        mcp = FastMCP("MyService")

        # Create reusable dependencies
        UserId = ZapContextValue("user_id")
        Tenant = ZapContextValue("tenant", "default")

        @mcp.tool()
        async def tenant_search(
            query: str,
            user_id: str | None = Depends(UserId),
            tenant: str = Depends(Tenant)
        ) -> str:
            '''Search within tenant's data scope.'''
            if not user_id:
                return "Error: User not authenticated"
            results = await search_engine.search(
                query,
                user_id=user_id,
                tenant_filter=tenant
            )
            return format_results(results)
        ```
    """

    def _get_value(ctx: Context = CurrentContext()) -> Any:
        zap_ctx = ZapContext(ctx)
        return zap_ctx.get(key, default)

    return _get_value


def TypedZapContext(context_type: type[T]) -> Callable[[Context], T]:
    """
    Factory for FastMCP dependencies that deserialize ZapContext to typed objects.

    Use this to create dependencies that automatically deserialize and validate
    the ZapContext to Pydantic models or dataclasses, providing full type safety.

    Args:
        context_type: The expected context class (Pydantic model or dataclass).

    Returns:
        A dependency function that deserializes to the specified type.

    Example:
        ```python
        from dataclasses import dataclass
        from fastmcp import FastMCP
        from fastmcp.dependencies import Depends
        from zap_ai.mcp.context import TypedZapContext

        @dataclass
        class UserSession:
            user_id: str
            tenant: str
            role: str

        mcp = FastMCP("MyService")

        # Create reusable typed dependency
        CurrentSession = TypedZapContext(UserSession)

        @mcp.tool()
        async def get_orders(
            limit: int = 10,
            session: UserSession = Depends(CurrentSession)
        ) -> str:
            '''Get orders for the current user.'''
            # session is fully typed - IDE autocomplete works!
            orders = await db.query_orders(
                user_id=session.user_id,
                tenant=session.tenant,
                limit=limit
            )
            return format_orders(orders)
        ```
    """

    def _get_typed_context(ctx: Context = CurrentContext()) -> T:
        # Extract raw context from FastMCP meta
        if not ctx.request_context or not ctx.request_context.meta:
            raise TypeError(
                f"Expected context of type {context_type.__name__}, but no context was provided"
            )

        raw_context = getattr(ctx.request_context.meta, "zap_context", {}) or {}

        if not raw_context:
            raise TypeError(
                f"Expected context of type {context_type.__name__}, but no context was provided"
            )

        # Check for type metadata
        type_id = raw_context.get("__zap_context_type__")

        # No metadata - can't deserialize to typed form
        if not type_id:
            raise TypeError(
                f"Expected context of type {context_type.__name__}, "
                "but context has no type metadata. "
                "Ensure context was created with a typed class (Pydantic/dataclass)."
            )

        # Extract data (remove metadata keys)
        data = {k: v for k, v in raw_context.items() if not k.startswith("__zap_context_")}

        # Import and reconstruct type
        try:
            # Parse module and class name
            module_name, class_name = type_id.rsplit(".", 1)
            module = importlib.import_module(module_name)
            context_class = getattr(module, class_name)

            # Validate expected type if provided
            if not issubclass(context_class, context_type):
                raise TypeError(
                    f"Context type mismatch: expected {context_type.__name__}, "
                    f"got {context_class.__name__}"
                )

            # Reconstruct based on type
            if hasattr(context_class, "model_validate"):  # Pydantic v2
                return context_class.model_validate(data)
            elif is_dataclass(context_class):
                return context_class(**data)
            else:
                raise TypeError(
                    f"Context type '{type_id}' does not support deserialization. "
                    "Only Pydantic models and dataclasses are supported."
                )

        except (ImportError, AttributeError, ValueError) as e:
            # Type not found - raise error
            raise TypeError(
                f"Could not import context type '{type_id}': {e}. "
                f"Expected context of type {context_type.__name__}."
            ) from e

    return _get_typed_context
