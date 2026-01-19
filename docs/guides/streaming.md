# Streaming Events

Zap provides real-time streaming of events during task execution via an async generator API. This allows you to display progress to users, log events, or build reactive UIs.

## Basic Usage

Use `stream_task()` instead of `execute_task()` to receive events as they occur:

```python
from zap_ai import Zap, ZapAgent
from zap_ai.streaming import ThinkingEvent, ToolCallEvent, ToolResultEvent, CompletedEvent, ErrorEvent

agent = ZapAgent(
    name="Assistant",
    prompt="You are a helpful assistant.",
    model="gpt-4o",
    mcp_clients=[...],
)

zap = Zap(agents=[agent])

async def main():
    await zap.start()

    async for event in zap.stream_task(agent_name="Assistant", task="What's the weather?"):
        match event:
            case ThinkingEvent(iteration=i):
                print(f"Thinking (iteration {i})...")
            case ToolCallEvent(name=name, phrase=phrase):
                print(phrase)  # e.g., "Getting weather for London..."
            case ToolResultEvent(name=name, success=success):
                status = "✓" if success else "✗"
                print(f"{status} {name} completed")
            case CompletedEvent(result=result):
                print(f"\nDone: {result}")
            case ErrorEvent(error=error):
                print(f"\nError: {error}")
```

## Event Types

All events inherit from `StreamEvent` and include:

- `type` - Event type string (e.g., "thinking", "tool_call")
- `timestamp` - ISO 8601 timestamp
- `task_id` - ID of the task
- `seq` - Sequence number for ordering

### ThinkingEvent

Emitted when the LLM starts processing an iteration.

```python
@dataclass
class ThinkingEvent(StreamEvent):
    iteration: int  # Loop iteration number (0-indexed)
```

### ToolCallEvent

Emitted when the agent calls a tool.

```python
@dataclass
class ToolCallEvent(StreamEvent):
    name: str              # Tool name
    arguments: dict        # Tool arguments
    phrase: str            # Human-readable description
    tool_call_id: str      # Unique ID for this call
```

The `phrase` field provides a user-friendly description generated from the tool's MCP description. For example:

- Tool description: "Get weather for a city"
- Arguments: `{"city": "London"}`
- Phrase: "Getting weather for London..."

### ToolResultEvent

Emitted when a tool execution completes.

```python
@dataclass
class ToolResultEvent(StreamEvent):
    name: str           # Tool name
    result: str         # Tool result (stringified)
    tool_call_id: str   # Matches the ToolCallEvent
    success: bool       # Whether execution succeeded
```

### CompletedEvent

Emitted when the task completes successfully.

```python
@dataclass
class CompletedEvent(StreamEvent):
    result: str  # Final result from the agent
```

### ErrorEvent

Emitted when the task fails.

```python
@dataclass
class ErrorEvent(StreamEvent):
    error: str  # Error message
```

### TokenEvent

Reserved for future token-level streaming (Phase 2).

```python
@dataclass
class TokenEvent(StreamEvent):
    token: str   # Individual token
    index: int   # Token index in response
```

## Configuration Options

`stream_task()` accepts the same parameters as `execute_task()`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `poll_interval` | `float` | `0.1` | Seconds between polling for events |
| `include_thinking` | `bool` | `True` | Include ThinkingEvent in stream |
| `include_tool_events` | `bool` | `True` | Include ToolCallEvent/ToolResultEvent |

Example with filtering:

```python
# Only receive completion events (skip thinking and tool events)
async for event in zap.stream_task(
    agent_name="Assistant",
    task="Calculate something",
    include_thinking=False,
    include_tool_events=False,
):
    if isinstance(event, CompletedEvent):
        print(f"Result: {event.result}")
```

## Architecture

Streaming uses Temporal query polling:

1. The workflow emits events to an internal buffer as it runs
2. `stream_task()` polls the workflow via `get_events` query
3. Events are yielded to the async generator as they're discovered
4. The buffer is bounded to prevent memory issues (last 500 events kept)

This approach provides ~100ms latency (configurable via `poll_interval`) and works reliably with Temporal's deterministic workflow model.

## Example: Progress Display

```python
import sys
from zap_ai.streaming import ThinkingEvent, ToolCallEvent, ToolResultEvent, CompletedEvent

async def run_with_progress(zap, task: str):
    tool_count = 0

    async for event in zap.stream_task(agent_name="Assistant", task=task):
        match event:
            case ThinkingEvent():
                sys.stdout.write(".")
                sys.stdout.flush()
            case ToolCallEvent(phrase=phrase):
                tool_count += 1
                print(f"\n[{tool_count}] {phrase}")
            case ToolResultEvent(success=success):
                print("  └─ " + ("Done" if success else "Failed"))
            case CompletedEvent(result=result):
                print(f"\n\nCompleted: {result}")
                return result

# Usage
result = await run_with_progress(zap, "What's the weather in Paris?")
```
