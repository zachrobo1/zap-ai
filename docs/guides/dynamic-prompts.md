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

Use dependency injection with helpers from `zap_ai.mcp.context`:

```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import ZapContext, ZapContextValue

mcp = FastMCP("Order Service")

@mcp.tool()
async def get_orders(
    limit: int = 10,
    zap_ctx: dict = Depends(ZapContext)
) -> str:
    """Get recent orders for the current user."""
    # Context is automatically injected via Depends()
    user_id = zap_ctx.get("user_id")

    if not user_id:
        return "Error: No user context provided"

    orders = await db.query_orders(user_id=user_id, limit=limit)
    return format_orders(orders)

# Or extract specific values using ZapContextValue
Tenant = ZapContextValue("tenant", "default")

@mcp.tool()
async def tenant_search(
    query: str,
    tenant: str = Depends(Tenant)
) -> str:
    """Search within tenant's data scope."""
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
from dataclasses import dataclass
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import TypedZapContext

@dataclass
class UserContext:
    user_id: str
    tenant: str
    permissions: list[str]

mcp = FastMCP("Invoice Service")

# Create typed dependency
CurrentUser = TypedZapContext(UserContext)

@mcp.tool()
async def search_invoices(
    query: str,
    user_ctx: UserContext = Depends(CurrentUser)
) -> str:
    """Search invoices."""
    # Context is automatically injected and typed!
    # Tenant isolation - tool automatically scoped to user's tenant
    if "read" not in user_ctx.permissions:
        return "Error: Insufficient permissions"

    invoices = await db.search_invoices(
        query=query,
        tenant_filter=user_ctx.tenant,
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
