# FastMCP Client Context Injection

This document describes how to implement automatic `ZapContext` injection into MCP tools, allowing tools to access workflow context (user data, session info, etc.) without the LLM seeing these parameters.

## Overview

**Problem**: MCP tools run on the server side, but `TContext` (user-defined application context) lives client-side in the Zap workflow. Tools currently only receive `tool_name` and `arguments`.

**Solution**: Use FastMCP 3.x's `meta` parameter to pass serialized context from client to server, then create a custom dependency for easy access in tools.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Zap Workflow                             │
│                                                                 │
│  TContext ──► serialize ──► ToolExecutionInput.context          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   tool_execution_activity                       │
│                                                                 │
│  client.call_tool(name, args, meta={"zap_context": context})    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastMCP Server                             │
│                                                                 │
│  @mcp.tool()                                                    │
│  async def my_tool(query: str, ctx = CurrentZapContext()):      │
│      # ctx contains the injected ZapContext                     │
│      # Hidden from LLM schema via FastMCP's DI filtering        │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- FastMCP >= 2.13.1 (for `meta` parameter support)
- Recommended: FastMCP 3.x for full dependency injection features

## Implementation Steps

### Step 1: Extend ToolExecutionInput

Add a `context` field to carry serialized context to the activity.

```python
# src/zap_ai/activities/tool_execution.py

@dataclass
class ToolExecutionInput:
    agent_name: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    trace_context: dict[str, Any] | None = None
    context: dict[str, Any] | None = None  # NEW: ZapContext data
```

### Step 2: Extend AgentWorkflowInput

Add context to the workflow input so it's available during tool execution.

```python
# src/zap_ai/workflows/models.py

@dataclass
class AgentWorkflowInput:
    agent_name: str
    initial_task: str | list[dict[str, Any]]
    system_prompt: str = ""
    model: str = "gpt-4o"
    # ... existing fields ...
    context: dict[str, Any] | None = None  # NEW: Serialized TContext
```

### Step 3: Pass Context Through the Workflow

Update `AgentWorkflow` to store and pass context to tool execution.

```python
# src/zap_ai/workflows/agent_workflow.py

class AgentWorkflow:
    def __init__(self) -> None:
        # ... existing fields ...
        self._context: dict[str, Any] | None = None  # NEW

    @workflow.run
    async def run(self, input: AgentWorkflowInput) -> str:
        # ... existing initialization ...
        self._context = input.context  # NEW: Store context
        # ...

    async def _execute_mcp_tool(self, tool_call: dict[str, Any]) -> str:
        # ... existing code ...
        return await workflow.execute_activity(
            tool_execution_activity,
            ToolExecutionInput(
                agent_name=self._agent_name,
                tool_name=tool_name,
                arguments=arguments,
                trace_context=self._trace_context.to_dict() if self._trace_context else None,
                context=self._context,  # NEW: Pass context
            ),
            # ... existing options ...
        )
```

### Step 4: Update tool_execution_activity

Pass context via FastMCP's `meta` parameter.

```python
# src/zap_ai/activities/tool_execution.py

@activity.defn
async def tool_execution_activity(input: ToolExecutionInput) -> str:
    # ... existing setup ...

    async def _execute_tool() -> str:
        # Build meta dict with zap_context
        meta = None
        if input.context:
            meta = {"zap_context": input.context}

        result = await client.call_tool(
            input.tool_name,
            input.arguments,
            meta=meta,  # NEW: Pass context via meta
        )

        # ... existing result handling ...

    # ... rest of function ...
```

### Step 5: Update Zap.run() to Accept and Serialize Context

```python
# src/zap_ai/zap.py

async def run(
    self,
    agent: ZapAgent[TContext] | str,
    task: str | list[dict[str, Any]],
    context: TContext | None = None,  # NEW parameter
    # ... existing params ...
) -> str:
    # Serialize context if provided
    serialized_context = None
    if context is not None:
        if isinstance(context, dict):
            serialized_context = context
        elif hasattr(context, "model_dump"):  # Pydantic model
            serialized_context = context.model_dump()
        elif hasattr(context, "__dict__"):
            serialized_context = context.__dict__
        else:
            raise ValueError(f"Cannot serialize context of type {type(context)}")

    # Include in workflow input
    workflow_input = AgentWorkflowInput(
        # ... existing fields ...
        context=serialized_context,
    )
```

### Step 6: Create Helper Utilities for Tool Authors

Provide a clean API for tool authors to access ZapContext.

```python
# New file: src/zap_ai/mcp/context.py

from typing import Any, TypeVar
from fastmcp import Context
from fastmcp.server.dependencies import CurrentContext, Depends

T = TypeVar("T")


def get_zap_context(ctx: Context = CurrentContext()) -> dict[str, Any]:
    """
    Extract ZapContext from request metadata.

    Returns an empty dict if no context was provided.
    """
    if ctx.request_context and ctx.request_context.meta:
        return ctx.request_context.meta.get("zap_context", {})
    return {}


def CurrentZapContext() -> Any:
    """
    Dependency that injects the ZapContext into a tool.

    Usage:
        @mcp.tool()
        async def my_tool(query: str, zap_ctx: dict = CurrentZapContext()) -> str:
            user_id = zap_ctx.get("user_id")
            ...

    The zap_ctx parameter is:
    - Automatically injected by FastMCP's DI system
    - Hidden from the MCP schema (LLM never sees it)
    - Contains whatever context was passed to Zap.run()
    """
    return Depends(get_zap_context)


def get_zap_context_value(key: str, default: T = None) -> T:
    """
    Get a specific value from ZapContext.

    Usage:
        @mcp.tool()
        async def my_tool(
            query: str,
            user_id: str = ZapContextValue("user_id", "anonymous")
        ) -> str:
            ...
    """
    def _get_value(ctx: Context = CurrentContext()) -> T:
        zap_ctx = get_zap_context(ctx)
        return zap_ctx.get(key, default)

    return Depends(_get_value)


# Convenience alias
ZapContextValue = get_zap_context_value
```

## Usage Examples

### Example 1: Basic Context Injection

**Zap client code:**
```python
from zap_ai import Zap, ZapAgent

zap = Zap(temporal_client, mcp_clients)
agent = ZapAgent(name="MyAgent", prompt="You are helpful.")

# Run with context
result = await zap.run(
    agent,
    task="Find user's recent orders",
    context={"user_id": "user_123", "tenant": "acme_corp"}
)
```

**MCP tool code:**
```python
from fastmcp import FastMCP
from zap_ai.mcp.context import CurrentZapContext

mcp = FastMCP("Order Service")

@mcp.tool()
async def get_orders(
    limit: int = 10,
    zap_ctx: dict = CurrentZapContext()
) -> str:
    """Get recent orders for the current user."""
    user_id = zap_ctx.get("user_id")
    if not user_id:
        return "Error: No user context provided"

    # Query orders for this user
    orders = await db.query_orders(user_id=user_id, limit=limit)
    return format_orders(orders)
```

### Example 2: Typed Context with Pydantic

**Define a typed context:**
```python
from pydantic import BaseModel

class UserContext(BaseModel):
    user_id: str
    tenant: str
    permissions: list[str] = []
    preferences: dict[str, Any] = {}
```

**Zap client code:**
```python
from zap_ai import Zap, ZapAgent

zap: Zap[UserContext] = Zap(temporal_client, mcp_clients)

context = UserContext(
    user_id="user_123",
    tenant="acme_corp",
    permissions=["read", "write"],
    preferences={"theme": "dark"}
)

result = await zap.run(agent, task="...", context=context)
```

**MCP tool with typed access:**
```python
from zap_ai.mcp.context import CurrentZapContext

@mcp.tool()
async def update_preferences(
    theme: str,
    zap_ctx: dict = CurrentZapContext()
) -> str:
    """Update user preferences."""
    user_id = zap_ctx.get("user_id")
    permissions = zap_ctx.get("permissions", [])

    if "write" not in permissions:
        return "Error: Insufficient permissions"

    await db.update_preferences(user_id, {"theme": theme})
    return f"Updated theme to {theme}"
```

### Example 3: Extracting Specific Values

```python
from zap_ai.mcp.context import ZapContextValue

@mcp.tool()
async def tenant_specific_search(
    query: str,
    tenant: str = ZapContextValue("tenant", "default")
) -> str:
    """Search within tenant's data scope."""
    results = await search_engine.search(query, tenant_filter=tenant)
    return format_results(results)
```

## Security Considerations

1. **Context is not visible to the LLM**: FastMCP's dependency injection system filters dependency parameters from the MCP schema. The LLM cannot see or manipulate `zap_ctx`.

2. **Context must be JSON-serializable**: The context passes through MCP's JSON protocol. Ensure all context values are serializable.

3. **Validate context in tools**: Tools should validate that expected context values exist and have correct types:
   ```python
   @mcp.tool()
   async def secure_tool(data: str, zap_ctx: dict = CurrentZapContext()) -> str:
       user_id = zap_ctx.get("user_id")
       if not user_id:
           raise ValueError("user_id required in context")
       # ...
   ```

4. **Don't put secrets in context**: While context is hidden from the LLM, it travels over the network. Use context for identifiers and metadata, not credentials.

## Testing

### Unit Testing Tools with Context

```python
import pytest
from fastmcp import Context
from unittest.mock import MagicMock

@pytest.fixture
def mock_context():
    ctx = MagicMock(spec=Context)
    ctx.request_context.meta = {
        "zap_context": {"user_id": "test_user", "tenant": "test_tenant"}
    }
    return ctx

async def test_tool_with_context(mock_context):
    # Test your tool logic with mocked context
    result = await get_orders(limit=5, zap_ctx={"user_id": "test_user"})
    assert "test_user" in result
```

### Integration Testing

```python
async def test_context_flows_to_tool():
    """Verify context reaches MCP tools."""
    zap = Zap(temporal_client, mcp_clients)

    # Use a tool that echoes back context
    result = await zap.run(
        agent,
        task="Echo my user ID",
        context={"user_id": "integration_test_user"}
    )

    assert "integration_test_user" in result
```

## Migration Notes

- This feature requires FastMCP >= 2.13.1
- Existing tools without `CurrentZapContext()` continue to work unchanged
- Context injection is opt-in per tool
- No changes required to existing `Zap.run()` calls without context

## References

- [FastMCP call_tool with meta](https://gofastmcp.com/clients/tools)
- [FastMCP Server Dependencies](https://gofastmcp.com/python-sdk/fastmcp-server-dependencies)
- [FastMCP v2.13.1 Release Notes](https://github.com/jlowin/fastmcp/releases/tag/v2.13.1)