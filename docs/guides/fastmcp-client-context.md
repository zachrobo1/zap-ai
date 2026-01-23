# Context Injection for MCP Tools

Zap automatically passes context (user data, session info, tenant IDs) to your MCP tools without exposing it to the LLM. This enables secure multi-tenancy, user-scoped operations, and personalized tool behavior while keeping sensitive identifiers hidden from the AI model.

## Basic Usage

Pass context when executing a task, then access it in your tools using helper functions:

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
from fastmcp import FastMCP, Context
from fastmcp.server.dependencies import CurrentContext
from zap_ai.mcp.context import get_zap_context

mcp = FastMCP("Document Tools")

@mcp.tool()
async def search_documents(
    query: str,
    limit: int = 10,
    ctx: Context = CurrentContext()
) -> str:
    """Search for documents in the user's tenant."""
    # Extract context passed from Zap
    zap_ctx = get_zap_context(ctx)
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

The `ctx: Context = CurrentContext()` parameter is hidden from the LLM via FastMCP's dependency injection system. The AI model only sees `query` and `limit` in the tool schema.

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
from fastmcp import Context
from fastmcp.server.dependencies import CurrentContext
from zap_ai.mcp.context import deserialize_zap_context

@mcp.tool()
async def delete_document(
    filename: str,
    ctx: Context = CurrentContext()
) -> str:
    """Delete a document."""
    # Automatically deserialize to UserContext
    user_ctx = deserialize_zap_context(ctx, UserContext)

    # Type-safe access with IDE autocomplete
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

Zap provides four helper functions for accessing context. Choose based on your needs:

### `get_zap_context(ctx) -> dict`

Extract the full context as a dictionary.

**Use when**: You need the complete context or working with plain dict contexts.

```python
from zap_ai.mcp.context import get_zap_context

@mcp.tool()
async def my_tool(ctx: Context = CurrentContext()) -> str:
    zap_ctx = get_zap_context(ctx)
    user_id = zap_ctx.get("user_id")
    tenant = zap_ctx.get("tenant")
    # ... use values
```

### `get_zap_context_value(ctx, key, default) -> Any`

Extract a specific value from context with a default.

**Use when**: You only need one specific value.

```python
from zap_ai.mcp.context import get_zap_context_value

@mcp.tool()
async def tenant_search(query: str, ctx: Context = CurrentContext()) -> str:
    tenant = get_zap_context_value(ctx, "tenant", "default")
    return await search_engine.search(query, tenant_filter=tenant)
```

### `deserialize_zap_context(ctx, expected_type=None) -> T | dict`

Automatically deserialize context to its original typed form.

**Use when**: You want type-safe access with optional validation and graceful fallback.

```python
from zap_ai.mcp.context import deserialize_zap_context

@mcp.tool()
async def update_settings(theme: str, ctx: Context = CurrentContext()) -> str:
    # Automatically reconstructs UserContext if type metadata present
    user_ctx = deserialize_zap_context(ctx, UserContext)

    # Type-safe access (or dict if type unavailable)
    if isinstance(user_ctx, UserContext):
        await db.update_user(user_ctx.user_id, {"theme": theme})
```

### `get_typed_zap_context(ctx, context_type) -> T`

Strict type-safe extraction with required validation.

**Use when**: You require guaranteed type safety for critical operations.

```python
from zap_ai.mcp.context import get_typed_zap_context

@mcp.tool()
async def charge_payment(amount: float, ctx: Context = CurrentContext()) -> str:
    # Raises TypeError if context can't be deserialized to SessionContext
    session = get_typed_zap_context(ctx, SessionContext)

    # session is guaranteed to be SessionContext
    if not session.authenticated:
        return "Error: Not authenticated"

    return await payment_service.charge(session.user_id, amount)
```

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
@mcp.tool()
async def protected_action(ctx: Context = CurrentContext()) -> str:
    user_ctx = deserialize_zap_context(ctx, UserContext)

    # Validate required fields
    if not isinstance(user_ctx, UserContext):
        return "Error: Invalid context type"

    if not user_ctx.user_id:
        return "Error: User ID required"

    # Safe to proceed
    return await perform_action(user_ctx)
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
│  async def my_tool(query: str, ctx = CurrentContext()):         │
│      zap_ctx = get_zap_context(ctx)  # Extracts from meta       │
│      # ctx parameter hidden from LLM schema                     │
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

The `deserialize_zap_context()` function uses this metadata to reconstruct the original type:
1. Dynamically imports the type using `importlib`
2. Validates the type if `expected_type` is provided
3. Reconstructs using `model_validate()` (Pydantic) or constructor (dataclass)
4. Falls back to dict with warning if type cannot be imported

Plain dicts pass through unchanged without metadata.

### Why LLM Can't See Context

FastMCP's dependency injection system uses `CurrentContext()` to inject the Context object. When FastMCP generates the tool schema for the LLM, it filters out all dependency parameters. The LLM only sees the "regular" parameters (like `query` and `limit`), never the context.

## Advanced: Type Metadata

For advanced use cases where you need to understand the type reconstruction mechanism.

### Type Metadata Format

```python
{
    "user_id": "123",
    "tenant": "acme",
    "__zap_context_type__": "module.path.ClassName",  # Fully qualified name
    "__zap_context_version__": "1"  # Version for future compatibility
}
```

### Import Resolution

The `deserialize_zap_context()` function:
1. Extracts `__zap_context_type__` from context
2. Parses it as `module_name.ClassName`
3. Uses `importlib.import_module()` to load the module
4. Retrieves the class with `getattr()`
5. Reconstructs the instance

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
- The tool will still work with dict access: `get_zap_context(ctx)`

### Context is empty ({}) in tools

**Cause**: Either no context was passed, or context didn't flow through the workflow.

**Solution**:
- Verify you're passing `context=` parameter to `execute_task()`
- Check that FastMCP version is >= 2.13.1 (meta parameter support)
- Ensure `ctx: Context = CurrentContext()` is in your tool signature

### Type mismatch error after renaming class

**Cause**: Type metadata uses fully qualified name (`module.ClassName`). Old serialized contexts still reference the old name.

**Solution**:
- Version your context types to handle migrations
- Or keep backward compatibility by maintaining old class names
- Or clear any cached/persistent workflow data

### get_zap_context() returns {} but I passed context

**Cause**: The helper functions return `{}` instead of `None` when no context exists.

**Solution**: Check for specific required fields instead:
```python
zap_ctx = get_zap_context(ctx)
if not zap_ctx.get("user_id"):
    return "Error: user_id required"
```

## Function Reference

### `get_zap_context(ctx: Context) -> dict[str, Any]`

Extract raw context as a dictionary. Returns `{}` if no context provided.

```python
zap_ctx = get_zap_context(ctx)
user_id = zap_ctx.get("user_id")
```

---

### `get_zap_context_value(ctx: Context, key: str, default: T = None) -> T`

Extract a specific value from context with a default.

**Parameters**:
- `ctx`: FastMCP Context object
- `key`: Key to extract
- `default`: Default value if key not present

```python
tenant = get_zap_context_value(ctx, "tenant", "default")
```

---

### `deserialize_zap_context(ctx: Context, expected_type: type[T] | None = None) -> T | dict[str, Any]`

Deserialize context to its original typed form (Pydantic/dataclass) with optional validation.

**Parameters**:
- `ctx`: FastMCP Context object
- `expected_type`: Optional type for validation

**Returns**: Typed instance if metadata present and importable, otherwise dict.

**Raises**: `TypeError` if expected_type provided and doesn't match actual type.

```python
# Without validation
user_ctx = deserialize_zap_context(ctx)

# With type validation
user_ctx = deserialize_zap_context(ctx, UserContext)
```

---

### `get_typed_zap_context(ctx: Context, context_type: type[T]) -> T`

Type-safe context extraction with strict validation.

**Parameters**:
- `ctx`: FastMCP Context object
- `context_type`: Required type class

**Returns**: Typed context instance (guaranteed to be of context_type).

**Raises**: `TypeError` if context cannot be deserialized to expected type.

```python
session = get_typed_zap_context(ctx, SessionContext)  # Guaranteed type or error
```

## Requirements

- **FastMCP >= 2.13.1** for `meta` parameter support
- **Pydantic v2** if using Pydantic models for context
- Context classes must be importable in the MCP tool's Python environment

## References

- [FastMCP call_tool with meta](https://gofastmcp.com/clients/tools)
- [FastMCP Server Dependencies](https://gofastmcp.com/python-sdk/fastmcp-server-dependencies)
