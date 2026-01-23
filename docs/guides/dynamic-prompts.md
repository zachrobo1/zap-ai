# Dynamic Prompts and Context

Zap supports passing context at execution time for two purposes:

1. **Dynamic prompts** - Customize agent system prompts with runtime values
2. **MCP tool injection** - Provide context data to MCP tools without the LLM seeing it

This is useful for personalizing agent behavior based on user information, session data, tenant isolation, or other runtime values.

## Basic Usage

Instead of a static string, pass a callable to the `prompt` parameter:

```python
from zap_ai import Zap, ZapAgent

# Agent with dynamic prompt
agent = ZapAgent(
    name="Assistant",
    prompt=lambda ctx: f"You are {ctx['user_name']}'s personal assistant.",
    model="gpt-4o",
)

zap = Zap(agents=[agent])
await zap.start()

# Pass context when executing
task = await zap.execute_task(
    agent_name="Assistant",
    task="Help me draft an email.",
    context={"user_name": "Alice"},
)
```

## Typed Context

For type safety, use a dataclass or Pydantic model and specify the generic type:

```python
from dataclasses import dataclass
from zap_ai import Zap, ZapAgent

@dataclass
class UserContext:
    user_name: str
    company: str
    role: str

# Specify the context type for better IDE support
agent = ZapAgent[UserContext](
    name="WorkAssistant",
    prompt=lambda ctx: f"""You are an assistant for {ctx.user_name},
a {ctx.role} at {ctx.company}. Be professional and helpful.""",
    model="gpt-4o",
)

zap = Zap(agents=[agent])
await zap.start()

task = await zap.execute_task(
    agent_name="WorkAssistant",
    task="Summarize our Q4 goals.",
    context=UserContext(
        user_name="Alice",
        company="Acme Corp",
        role="Product Manager",
    ),
)
```

## Multiple Agents with Shared Context

When using multiple agents, they can all receive the same context type:

```python
@dataclass
class SessionContext:
    user_id: str
    language: str
    timezone: str

researcher = ZapAgent[SessionContext](
    name="Researcher",
    prompt=lambda ctx: f"Research assistant. Respond in {ctx.language}.",
    model="gpt-4o",
)

writer = ZapAgent[SessionContext](
    name="Writer",
    prompt=lambda ctx: f"Technical writer. Use {ctx.timezone} for timestamps.",
    model="gpt-4o",
    discovery_prompt="Use for writing and formatting content",
)

zap = Zap(agents=[researcher, writer])
```

## Default Context

If no context is provided, an empty dict `{}` is used. Agents with dynamic prompts will receive a warning if called without context:

```python
# This will work but emit a warning
task = await zap.execute_task(
    agent_name="Assistant",
    task="Hello",
    # context not provided - warning emitted
)
```

To avoid warnings, always provide context or use a static prompt for agents that don't need runtime customization.

## Checking for Dynamic Prompts

You can check if an agent uses a dynamic prompt:

```python
agent = ZapAgent(
    name="Test",
    prompt=lambda ctx: f"Hello {ctx['name']}",
)

if agent.is_dynamic_prompt():
    print("This agent requires context")
```

## Context for MCP Tools

Context passed to `execute_task()` is automatically available to your MCP tools via FastMCP's dependency injection system. This enables secure multi-tenancy, user-scoped operations, and personalized tool behavior - all while keeping sensitive identifiers hidden from the LLM.

**Quick Example:**

```python
from dataclasses import dataclass
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai import Zap, ZapAgent
from zap_ai.mcp.context import TypedZapContext

# Define your context type
@dataclass
class UserContext:
    user_id: str
    tenant: str

# Agent with dynamic prompt
agent = ZapAgent[UserContext](
    name="Assistant",
    prompt=lambda ctx: f"You are an assistant for tenant {ctx.tenant}.",
    mcp_clients=[Client("./tools.py")],
)

zap = Zap(agents=[agent])
await zap.start()

# Context is used for BOTH the dynamic prompt AND tool access
task = await zap.execute_task(
    agent_name="Assistant",
    task="Search for invoices",
    context=UserContext(user_id="user_123", tenant="acme_corp"),
)
```

**In your MCP tool:**

```python
mcp = FastMCP("Invoice Service")
CurrentUser = TypedZapContext(UserContext)

@mcp.tool()
async def search_invoices(
    query: str,
    user_ctx: UserContext = Depends(CurrentUser)
) -> str:
    """Search invoices - automatically scoped to user's tenant."""
    invoices = await db.search_invoices(
        query=query,
        tenant_filter=user_ctx.tenant,  # Secure tenant isolation
    )
    return format_invoices(invoices)
```

The `user_ctx` parameter is injected by FastMCP and hidden from the LLM - the AI model only sees the `query` parameter.

!!! info "Comprehensive MCP Context Guide"
    For complete details on context injection including:

    - Three access patterns (`ZapContext`, `ZapContextValue`, `TypedZapContext`)
    - Type safety and automatic deserialization
    - Security best practices
    - Troubleshooting and advanced patterns

    See the [**FastMCP Context Injection Guide**](fastmcp-client-context.md)

## API Reference

For dynamic prompts:

- [`ZapAgent`](../api/core.md#zap_ai.ZapAgent) - Agent configuration with `prompt` parameter
- [`Zap.execute_task`](../api/core.md#zap_ai.Zap.execute_task) - Execute tasks with context

For MCP context injection:

- [FastMCP Context Injection Guide](fastmcp-client-context.md) - Complete reference for accessing context in tools
