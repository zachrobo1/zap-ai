# Context Injection for MCP Tools

Zap automatically passes context (user data, session info, tenant IDs) to your MCP tools without exposing it to the LLM. This enables secure multi-tenancy, user-scoped operations, and personalized tool behavior while keeping sensitive identifiers hidden from the AI model.

## Basic Usage

Pass context when executing a task, then access it in your tools using FastMCP's dependency injection system:

**Client code:**
```python
from zap_ai import Zap, ZapAgent
from fastmcp import Client

# Define your agent
agent = ZapAgent(
    name="DocumentAgent",
    prompt="You help users find their documents.",
    mcp_clients=[Client("./document_tools.py")],
)

zap = Zap(agents=[agent])
await zap.start()

# Pass context with user/tenant information
result = await zap.execute_task(
    agent_name="DocumentAgent",
    task="Show me my recent documents",
    context={"user_id": "user_123", "tenant": "acme_corp"}
)
```

**Tool code (MCP server):**
```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import ZapContext

mcp = FastMCP("Document Tools")

@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 10,
    zap_ctx: dict = Depends(ZapContext)
) -> str:
    """Search for documents in the user's tenant."""
    # Extract context values
    user_id = zap_ctx.get("user_id")
    tenant = zap_ctx.get("tenant")

    if not user_id or not tenant:
        return "Error: No user context provided"

    # Query with tenant/user scoping
    results = await db.search_documents(
        query=query,
        user_id=user_id,
        tenant=tenant,
        limit=limit
    )

    return format_results(results)
```

The `zap_ctx: dict = Depends(ZapContext)` parameter is hidden from the LLM via FastMCP's dependency injection system. The AI model only sees `query` and `limit` in the tool schema.

## Typed Context with Automatic Deserialization

For better type safety and IDE support, use Pydantic models or dataclasses for your context. Zap automatically preserves type information, enabling type-safe access in tools.

**Define a typed context:**
```python
from pydantic import BaseModel

class UserContext(BaseModel):
    user_id: str
    tenant: str
    permissions: list[str] = []
    preferences: dict[str, str] = {}
```

**Client code with typed context:**
```python
from zap_ai import Zap, ZapAgent

agent = ZapAgent[UserContext](
    name="DocumentAgent",
    prompt="You help users manage their documents.",
    mcp_clients=[Client("./document_tools.py")],
)

zap = Zap(agents=[agent])
await zap.start()

# Pass a Pydantic model - type metadata is automatically added
context = UserContext(
    user_id="user_123",
    tenant="acme_corp",
    permissions=["read", "write", "delete"],
    preferences={"view": "grid", "sort": "date"}
)

result = await zap.execute_task(
    agent_name="DocumentAgent",
    task="Delete the file named 'draft.txt'",
    context=context
)
```

**Tool with typed deserialization:**
```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import TypedZapContext

mcp = FastMCP("Document Tools")

# Create a reusable typed dependency
CurrentUser = TypedZapContext(UserContext)

@mcp.tool()
async def delete_document(
    filename: str,
    user_ctx: UserContext = Depends(CurrentUser)
) -> str:
    """Delete a document."""
    # user_ctx is fully typed with IDE autocomplete
    if "delete" not in user_ctx.permissions:
        return "Error: You don't have permission to delete documents"

    await db.delete_document(
        filename=filename,
        user_id=user_ctx.user_id,
        tenant=user_ctx.tenant
    )

    return f"Deleted '{filename}'"
```

**Benefits of typed context:**
- ✅ **IDE Support**: Full autocomplete and type checking
- ✅ **Refactoring Safety**: Rename fields with confidence
- ✅ **Runtime Validation**: Pydantic validates types automatically
- ✅ **Zero Configuration**: Works automatically for Pydantic models and dataclasses
- ✅ **Backward Compatible**: Falls back to dict if type unavailable

## Context Access Patterns

Zap provides three dependency injection helpers for accessing context. Choose based on your needs:

### `ZapContext` - Full Context Dict

Extract the complete context as a dictionary.

**Use when**: You need the complete context or working with plain dict contexts.

```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import ZapContext

mcp = FastMCP("My Service")

@mcp.tool()
async def my_tool(
    query: str,
    zap_ctx: dict = Depends(ZapContext)
) -> str:
    """A tool that needs full context."""
    user_id = zap_ctx.get("user_id")
    tenant = zap_ctx.get("tenant")
    # ... use values
```

**Note**: If the context was created with a Pydantic model or dataclass, `ZapContext` will automatically deserialize it back to the typed object (not a dict). To always get a dict, access individual fields using `ZapContextValue` instead.

---

### `ZapContextValue(key, default)` - Extract Specific Values

Create a dependency that extracts a specific value from context with a default.

**Use when**: You only need one or two specific values.

```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import ZapContextValue

mcp = FastMCP("Search Service")

# Create reusable dependencies
Tenant = ZapContextValue("tenant", "default")
UserId = ZapContextValue("user_id")

@mcp.tool()
async def tenant_search(
    query: str,
    tenant: str = Depends(Tenant),
    user_id: str | None = Depends(UserId)
) -> str:
    """Search within tenant's data scope."""
    if not user_id:
        return "Error: User not authenticated"
    return await search_engine.search(query, tenant_filter=tenant)
```

**Benefits**:
- Cleaner function signatures
- Reusable dependencies across tools
- Explicit defaults in one place

---

### `TypedZapContext(context_type)` - Type-Safe Extraction

Create a dependency that deserializes context to a specific typed object with validation.

**Use when**: You want guaranteed type safety for critical operations.

```python
from dataclasses import dataclass
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import TypedZapContext

@dataclass
class SessionContext:
    user_id: str
    authenticated: bool
    role: str

mcp = FastMCP("Payment Service")

# Create typed dependency
CurrentSession = TypedZapContext(SessionContext)

@mcp.tool()
async def charge_payment(
    amount: float,
    session: SessionContext = Depends(CurrentSession)
) -> str:
    """Process a payment."""
    # session is guaranteed to be SessionContext or raises TypeError
    if not session.authenticated:
        return "Error: Not authenticated"

    return await payment_service.charge(session.user_id, amount)
```

**Error handling**: Raises `TypeError` if context cannot be deserialized to the expected type.

## Security Best Practices

### 1. Context is hidden from the LLM

The LLM cannot see or manipulate context parameters. FastMCP's dependency injection system automatically filters these parameters from the tool schema shown to the model.

### 2. Use for identifiers, not secrets

Context travels through the Temporal workflow and MCP protocol. Pass user IDs, tenant identifiers, and session metadata—not API keys, passwords, or credentials. Use environment variables or tool-specific configuration for secrets.

```python
# Good
context = {"user_id": "user_123", "tenant": "acme_corp", "role": "admin"}

# Bad - don't put secrets in context
context = {"user_id": "user_123", "api_key": "sk_live_123..."}
```

### 3. Always validate context in tools

Check that required context values exist and have valid types:

```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import TypedZapContext

mcp = FastMCP("Protected Service")
CurrentUser = TypedZapContext(UserContext)

@mcp.tool()
async def protected_action(
    action: str,
    user_ctx: UserContext = Depends(CurrentUser)
) -> str:
    """Perform a protected action."""
    # TypedZapContext raises TypeError if context is invalid
    # At this point, user_ctx is guaranteed to be UserContext

    if not user_ctx.user_id:
        return "Error: User ID required"

    # Safe to proceed
    return await perform_action(user_ctx, action)
```

### 4. Type safety reduces bugs

Using Pydantic models or dataclasses provides compile-time checking and reduces runtime errors:

```python
# With types: IDE catches errors immediately
user_ctx.usr_id  # TypeError: 'UserContext' has no attribute 'usr_id'

# Without types: Runtime error only when code executes
zap_ctx.get("usr_id")  # Returns None, silent bug
```

### 5. Keep context minimal

Only include what tools actually need. Smaller contexts are faster to serialize and easier to debug.

## How It Works

Context flows from your Zap workflow to MCP tools through FastMCP's `meta` parameter:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Zap Workflow                             │
│                                                                 │
│  context (Pydantic/dataclass/dict) → serialize with metadata    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Temporal Workflow Activity                    │
│                                                                 │
│  client.call_tool(name, args, meta={"zap_context": context})    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastMCP Server                             │
│                                                                 │
│  @mcp.tool()                                                    │
│  async def my_tool(query: str, ctx = Depends(ZapContext)):      │
│      # ZapContext extracts from meta, deserializes if typed     │
│      # Dependency parameter hidden from LLM schema              │
└─────────────────────────────────────────────────────────────────┘
```

### Context Serialization

When you pass a Pydantic model or dataclass, Zap automatically adds type metadata:

```python
# Your context
UserContext(user_id="123", tenant="acme")

# Serialized with metadata
{
    "user_id": "123",
    "tenant": "acme",
    "__zap_context_type__": "myapp.models.UserContext",
    "__zap_context_version__": "1"
}
```

The `ZapContext` and `TypedZapContext` dependencies use this metadata to reconstruct the original type:
1. Dynamically imports the type using `importlib`
2. Validates the type if using `TypedZapContext`
3. Reconstructs using `model_validate()` (Pydantic) or constructor (dataclass)
4. Falls back to dict with warning if type cannot be imported

Plain dicts pass through unchanged without metadata.

### Why LLM Can't See Context

FastMCP's dependency injection system uses `Depends()` to inject dependencies. When FastMCP generates the tool schema for the LLM, it filters out all dependency parameters. The LLM only sees the "regular" parameters (like `query` and `limit`), never the context.

## Advanced: Dependency Injection Pattern

The Zap context injection system follows FastMCP's standard dependency injection pattern:

```python
# These are dependency factories - they return functions that extract context
ZapContext          # Returns: (ctx: Context) -> dict | typed_object
ZapContextValue(k)  # Returns: (ctx: Context) -> Any
TypedZapContext(T)  # Returns: (ctx: Context) -> T

# Use with Depends() in your tools
@mcp.tool()
async def tool(
    param: str,
    # Depends() tells FastMCP to call the dependency function
    ctx_value: dict = Depends(ZapContext)
):
    ...
```

**Important**: The context class must be importable from the MCP tool's Python environment. If you define `UserContext` in your client code but the MCP server can't import it, deserialization will fall back to dict.

### Custom Context Types

Any class with these methods works:
- **Pydantic v2**: `model_validate(data)` for reconstruction
- **Dataclass**: Constructor accepts `**kwargs`

Custom classes without these methods will return as dict.

## Troubleshooting

### "Could not import context type" warning

**Cause**: The context class isn't available in the MCP tool's Python environment.

**Solution**:
- Define shared context types in a common module both client and server can import
- Or use plain dicts for simple cases
- The tool will still work with dict access via `ZapContext`

### Context is empty ({}) in tools

**Cause**: Either no context was passed, or context didn't flow through the workflow.

**Solution**:
- Verify you're passing `context=` parameter to `execute_task()`
- Check that FastMCP version is >= 2.13.1 (meta parameter support)
- Ensure you're using `Depends(ZapContext)` or similar in your tool signature

### TypeError: Expected context of type X

**Cause**: Using `TypedZapContext` but the actual context doesn't match the expected type.

**Solution**:
- Verify you're passing the correct context type in `execute_task()`
- Check that type metadata is present (context was created with Pydantic/dataclass)
- Use plain `ZapContext` if you want automatic fallback to dict

### Context returns {} but I passed context

**Cause**: The dependency returns `{}` instead of `None` when no context exists.

**Solution**: Check for specific required fields instead:
```python
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import ZapContext

@mcp.tool()
async def tool(zap_ctx: dict = Depends(ZapContext)) -> str:
    if not zap_ctx.get("user_id"):
        return "Error: user_id required"
    ...
```

## API Reference

### `ZapContext`

A FastMCP dependency that deserializes the full ZapContext. Use with `Depends()`.

**Returns**: Deserialized context dict, or the typed object if context was created with Pydantic/dataclass. Returns `{}` if no context provided.

**Example**:
```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import ZapContext

@mcp.tool()
async def my_tool(zap_ctx: dict = Depends(ZapContext)) -> str:
    user_id = zap_ctx.get("user_id")
    ...
```

---

### `ZapContextValue(key: str, default: Any = None)`

Factory that creates a FastMCP dependency for extracting a specific value from context.

**Parameters**:
- `key`: The key to extract from context
- `default`: Default value if key not present (default: `None`)

**Returns**: A dependency function suitable for use with `Depends()`

**Example**:
```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import ZapContextValue

# Create reusable dependencies
Tenant = ZapContextValue("tenant", "default")
UserId = ZapContextValue("user_id")

@mcp.tool()
async def tool(
    tenant: str = Depends(Tenant),
    user_id: str | None = Depends(UserId)
) -> str:
    ...
```

---

### `TypedZapContext(context_type: type[T])`

Factory that creates a FastMCP dependency for type-safe context deserialization with validation.

**Parameters**:
- `context_type`: The expected context class (Pydantic model or dataclass)

**Returns**: A dependency function that deserializes to the specified type

**Raises**: `TypeError` if context cannot be deserialized to the expected type

**Example**:
```python
from dataclasses import dataclass
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import TypedZapContext

@dataclass
class SessionContext:
    user_id: str
    role: str

# Create typed dependency
CurrentSession = TypedZapContext(SessionContext)

@mcp.tool()
async def tool(session: SessionContext = Depends(CurrentSession)) -> str:
    # session is guaranteed to be SessionContext
    ...
```

## Requirements

- **FastMCP >= 2.13.1** for `meta` parameter support
- **Pydantic v2** if using Pydantic models for context
- Context classes must be importable in the MCP tool's Python environment

## References

- [FastMCP call_tool with meta](https://gofastmcp.com/clients/tools)
- [FastMCP Server Dependencies](https://gofastmcp.com/python-sdk/fastmcp-server-dependencies)
