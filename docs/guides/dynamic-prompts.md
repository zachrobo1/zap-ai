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

## Context Injection to MCP Tools

Context passed to `execute_task()` is also automatically available to MCP tools. This allows tools to access user data, tenant information, or session state without exposing it to the LLM.

### How It Works

When you pass context to `execute_task()`, Zap:

1. Serializes the context (dict, dataclass, or Pydantic model)
2. Passes it via FastMCP's `meta` parameter during tool execution
3. Makes it accessible within tools via helper functions

The LLM never sees this context - it's hidden from the tool schema.

### Accessing Context in Tools

Use the helpers from `zap_ai.mcp.context`:

```python
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import CurrentContext
from zap_ai.mcp.context import get_zap_context, get_zap_context_value

mcp = FastMCP("Order Service")

@mcp.tool()
async def get_orders(
    limit: int = 10,
    ctx: Context = CurrentContext()
) -> str:
    """Get recent orders for the current user."""
    # Get the full context dict
    zap_ctx = get_zap_context(ctx)
    user_id = zap_ctx.get("user_id")

    if not user_id:
        return "Error: No user context provided"

    orders = await db.query_orders(user_id=user_id, limit=limit)
    return format_orders(orders)

@mcp.tool()
async def tenant_search(
    query: str,
    ctx: Context = CurrentContext()
) -> str:
    """Search within tenant's data scope."""
    # Get a specific value with default
    tenant = get_zap_context_value(ctx, "tenant", "default")
    results = await search_engine.search(query, tenant_filter=tenant)
    return format_results(results)
```

### Full Example

**Zap client code:**
```python
from dataclasses import dataclass
from zap_ai import Zap, ZapAgent

@dataclass
class UserContext:
    user_id: str
    tenant: str
    permissions: list[str]

def make_prompt(ctx: UserContext) -> str:
    return f"You are an assistant for tenant {ctx.tenant}."

agent = ZapAgent[UserContext](
    name="TenantAgent",
    prompt=make_prompt,
    mcp_clients=[Client("./tools.py")],
)

zap: Zap[UserContext] = Zap(agents=[agent])
await zap.start()

# Context is used for BOTH the dynamic prompt AND tool calls
task = await zap.execute_task(
    agent_name="TenantAgent",
    task="Search for recent invoices",
    context=UserContext(
        user_id="user_123",
        tenant="acme_corp",
        permissions=["read", "write"],
    ),
)
```

**MCP tool code:**
```python
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import CurrentContext
from zap_ai.mcp.context import get_zap_context

mcp = FastMCP("Invoice Service")

@mcp.tool()
async def search_invoices(
    query: str,
    ctx: Context = CurrentContext()
) -> str:
    """Search invoices."""
    zap_ctx = get_zap_context(ctx)

    # Tenant isolation - tool automatically scoped to user's tenant
    tenant = zap_ctx.get("tenant")
    permissions = zap_ctx.get("permissions", [])

    if "read" not in permissions:
        return "Error: Insufficient permissions"

    invoices = await db.search_invoices(
        query=query,
        tenant_filter=tenant,
    )
    return format_invoices(invoices)
```

### Security Considerations

1. **Context is hidden from the LLM** - FastMCP's dependency injection filters it from the schema
2. **Context travels over the network** - Use it for identifiers and metadata, not secrets
3. **Validate in tools** - Always validate that expected context values exist

## API Reference

See the full API documentation:

- [`ZapAgent`](../api/core.md#zap_ai.ZapAgent) - Agent configuration
- [`Zap.execute_task`](../api/core.md#zap_ai.Zap.execute_task) - Execute with context
