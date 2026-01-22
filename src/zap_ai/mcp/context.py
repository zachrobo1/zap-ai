"""
FastMCP context injection utilities for MCP tools.

This module provides helpers for accessing ZapContext within MCP tools.
Context is passed from the Zap workflow via FastMCP's `meta` parameter
and can be accessed using the utilities defined here.

Example:
    ```python
    from fastmcp import FastMCP, Context
    from fastmcp.server.dependencies import CurrentContext
    from zap_ai.mcp.context import get_zap_context

    mcp = FastMCP("MyService")

    @mcp.tool()
    async def my_tool(query: str, ctx: Context = CurrentContext()) -> str:
        zap_ctx = get_zap_context(ctx)
        user_id = zap_ctx.get("user_id")
        # ctx is injected by FastMCP; zap_ctx is hidden from LLM schema
        ...
    ```
"""

from typing import Any, TypeVar

from fastmcp import Context

T = TypeVar("T")


def get_zap_context(ctx: Context) -> dict[str, Any]:
    """
    Extract ZapContext from FastMCP request metadata.

    This function retrieves the context that was passed to Zap.run()
    or execute_task() and injected via FastMCP's meta parameter.

    Args:
        ctx: FastMCP Context object (injected via CurrentContext()).

    Returns:
        The ZapContext dict, or empty dict if no context was provided.

    Example:
        ```python
        from fastmcp import Context
        from fastmcp.server.dependencies import CurrentContext
        from zap_ai.mcp.context import get_zap_context

        @mcp.tool()
        async def get_orders(
            limit: int = 10,
            ctx: Context = CurrentContext()
        ) -> str:
            '''Get recent orders for the current user.'''
            zap_ctx = get_zap_context(ctx)
            user_id = zap_ctx.get("user_id")
            if not user_id:
                return "Error: No user context provided"
            orders = await db.query_orders(user_id=user_id, limit=limit)
            return format_orders(orders)
        ```
    """
    if ctx.request_context and ctx.request_context.meta:
        # Meta is a Pydantic model; zap_context is stored as an attribute
        return getattr(ctx.request_context.meta, "zap_context", {}) or {}
    return {}


def get_zap_context_value(ctx: Context, key: str, default: T = None) -> T:  # type: ignore[assignment]
    """
    Get a specific value from ZapContext.

    This is a convenience function for extracting individual values from
    the context without needing to access the full dict.

    Args:
        ctx: FastMCP Context object (injected via CurrentContext()).
        key: The key to extract from the ZapContext.
        default: Default value if the key is not present.

    Returns:
        The value for the key, or the default if not present.

    Example:
        ```python
        from fastmcp import Context
        from fastmcp.server.dependencies import CurrentContext
        from zap_ai.mcp.context import get_zap_context_value

        @mcp.tool()
        async def tenant_search(
            query: str,
            ctx: Context = CurrentContext()
        ) -> str:
            '''Search within tenant's data scope.'''
            tenant = get_zap_context_value(ctx, "tenant", "default")
            results = await search_engine.search(query, tenant_filter=tenant)
            return format_results(results)
        ```
    """
    zap_ctx = get_zap_context(ctx)
    return zap_ctx.get(key, default)
