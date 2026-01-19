# Streaming API

The streaming module provides event types for real-time task progress monitoring.

## Event Types

All events inherit from `StreamEvent` and can be used with `Zap.stream_task()`.

### StreamEvent

Base class for all streaming events.

::: zap_ai.streaming.StreamEvent
    options:
      show_source: true

### ThinkingEvent

Emitted when the LLM starts processing an iteration.

::: zap_ai.streaming.ThinkingEvent
    options:
      show_source: true

### ToolCallEvent

Emitted when the agent calls a tool.

::: zap_ai.streaming.ToolCallEvent
    options:
      show_source: true

### ToolResultEvent

Emitted when a tool execution completes.

::: zap_ai.streaming.ToolResultEvent
    options:
      show_source: true

### TokenEvent

Reserved for future token-level streaming.

::: zap_ai.streaming.TokenEvent
    options:
      show_source: true

### CompletedEvent

Emitted when the task completes successfully.

::: zap_ai.streaming.CompletedEvent
    options:
      show_source: true

### ErrorEvent

Emitted when the task fails.

::: zap_ai.streaming.ErrorEvent
    options:
      show_source: true

## Utility Functions

### generate_tool_phrase

::: zap_ai.streaming.generate_tool_phrase
    options:
      show_source: true

### parse_event

::: zap_ai.streaming.parse_event
    options:
      show_source: true
