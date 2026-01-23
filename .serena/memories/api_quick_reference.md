# API Quick Reference

Quick reference for Zap's main classes, methods, and code patterns.

## Main Classes

### Core Classes
- `Zap` - Main orchestrator that manages agents and Temporal connections
- `ZapAgent[TContext]` - Agent configuration (name, prompt, model, mcp_clients, sub_agents, temperature, max_tokens, approval_rules)
- `Task` - Represents an executing or completed task with rich inspection methods
- `TaskStatus` - Enum: PENDING, THINKING, AWAITING_TOOL, AWAITING_APPROVAL, COMPLETED, FAILED

### Task Inspection
- `ToolCallInfo` - Information about a tool call and its result
- `ConversationTurn` - A single turn in the conversation

### Multimodal Content
- `TextContent` - Text content part: `TextContent(type="text", text="...")`
- `ImageContent` - Image content part with factory methods:
  - `ImageContent.from_url(url)` 
  - `ImageContent.from_base64(data, media_type)`
- `ContentPart` - Type alias: `TextContent | ImageContent`
- `MessageContent` - Type alias: `str | list[ContentPart]`

### Streaming Events
- `StreamEvent` - Base class (type, timestamp, task_id, seq)
- `ThinkingEvent` - LLM starts reasoning (iteration number)
- `ToolCallEvent` - Tool called (name, arguments, phrase)
- `ToolResultEvent` - Tool completed (name, result, success)
- `TokenEvent` - Token streaming (Phase 2, not yet implemented)
- `CompletedEvent` - Task succeeded (result)
- `ErrorEvent` - Task failed (error message)

### Approval Workflows
- `ApprovalRules(patterns, timeout_seconds)` - Glob patterns for tools requiring approval
- `ApprovalRequest` - Pending request (id, tool_name, arguments, requested_at, timeout_at)
- `ApprovalDecision` - Response (approved, reason)

### Tracing
- `TracingProvider` - Protocol for tracing implementations
- `BaseTracingProvider` - ABC for custom providers
- `LangfuseTracingProvider` - Langfuse integration
- `NoOpTracingProvider` - Fallback

## Zap Methods

```python
# Lifecycle
await zap.start()                    # Connect to Temporal, initialize MCP clients
await zap.stop()                     # Graceful shutdown

# Task execution
task = await zap.execute_task(
    agent_name="MyAgent",
    task="Do something",              # str or MessageContent (multimodal)
    context={"user_id": "123"}        # Optional context for dynamic prompts + MCP tools
)

# Streaming
async for event in zap.stream_task(agent_name, task, context):
    if isinstance(event, ThinkingEvent):
        print(f"Thinking (iteration {event.iteration})...")
    elif isinstance(event, ToolCallEvent):
        print(f"Calling tool: {event.tool_name}")
    elif isinstance(event, CompletedEvent):
        print(f"Result: {event.result}")

# Task/agent retrieval
task = await zap.get_task(task_id)   # Get task by ID
agent = zap.get_agent("AgentName")   # Get agent config
tools = await zap.get_agent_tools("AgentName")  # Get available tools
```

## Task Methods

```python
# Content access
text = task.get_text_content()                 # All text from conversation
tool_calls = task.get_tool_calls()             # List[ToolCallInfo]

# Conversation turns
turns = task.get_turns()                       # All turns
turn = task.get_turn(0)                        # Specific turn (0-indexed)
count = task.turn_count()                      # Number of turns

# Sub-tasks
sub_tasks = await task.get_sub_tasks(zap)     # Fetch child task objects

# Approvals
approvals = task.get_pending_approvals()      # List[ApprovalRequest]
await task.approve(approval_id)               # Approve request
await task.reject(approval_id, "reason")      # Reject with reason

# Status
if task.status.is_terminal():                 # COMPLETED or FAILED
    print(task.result or task.error)
```

## Context Injection Code Patterns

Import from `zap_ai.mcp.context` in MCP tools:

```python
from fastmcp import FastMCP
from fastmcp.dependencies import Depends
from zap_ai.mcp.context import ZapContext, ZapContextValue, TypedZapContext

mcp = FastMCP("MyService")

# Pattern 1: Full dict context
@mcp.tool()
async def my_tool(query: str, zap_ctx: dict = Depends(ZapContext)) -> str:
    user_id = zap_ctx.get("user_id")
    tenant = zap_ctx.get("tenant")
    ...

# Pattern 2: Extract specific values
UserId = ZapContextValue("user_id")
Tenant = ZapContextValue("tenant", "default")

@mcp.tool()
async def tenant_search(
    query: str,
    user_id: str | None = Depends(UserId),
    tenant: str = Depends(Tenant)
) -> str:
    if not user_id:
        return "Error: Not authenticated"
    ...

# Pattern 3: Typed context (Pydantic/dataclass)
from dataclasses import dataclass

@dataclass
class UserSession:
    user_id: str
    tenant: str
    permissions: list[str]

CurrentSession = TypedZapContext(UserSession)

@mcp.tool()
async def get_orders(
    limit: int = 10,
    session: UserSession = Depends(CurrentSession)
) -> str:
    # Full type safety and IDE autocomplete!
    orders = await db.query(
        user_id=session.user_id,
        tenant=session.tenant,
        limit=limit
    )
    ...
```

## Common Imports

```python
# Core
from zap_ai import Zap, ZapAgent, Task, TaskStatus

# Multimodal
from zap_ai import TextContent, ImageContent, ContentPart, MessageContent

# Streaming
from zap_ai import (
    ThinkingEvent, ToolCallEvent, ToolResultEvent,
    TokenEvent, CompletedEvent, ErrorEvent
)

# Approvals
from zap_ai import ApprovalRules, ApprovalRequest

# Conversation
from zap_ai import ConversationTurn, ToolCallInfo

# Exceptions
from zap_ai import (
    ZapConfigurationError, ZapNotStartedError,
    AgentNotFoundError, TaskNotFoundError,
    VisionNotSupportedError
)

# MCP Context (use in MCP server code, NOT client)
from zap_ai.mcp.context import ZapContext, ZapContextValue, TypedZapContext

# MCP Sampling (for MCP servers that need LLM access)
from zap_ai.mcp.sampling import create_mcp_client, LiteLLMSamplingHandler

# Tracing
from zap_ai.tracing import set_tracing_provider, LangfuseTracingProvider

# Worker
from zap_ai.worker import create_worker, run_worker, run_worker_with_zap
```

## Type Hints

```python
from zap_ai import TContext, DefaultContext, DynamicPrompt

# Agent with typed context
agent = ZapAgent[UserContext](
    name="MyAgent",
    prompt=lambda ctx: f"User: {ctx.user_name}",  # DynamicPrompt[UserContext]
    model="gpt-4o",
)

# Zap with typed context
zap: Zap[UserContext] = Zap(agents=[agent])

# Execute with context
task = await zap.execute_task(
    agent_name="MyAgent",
    task="Help me",
    context=UserContext(user_name="Alice", ...)
)
```
