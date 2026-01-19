"""
Streaming support for Zap task execution.

This module provides real-time event streaming for task execution via
Temporal workflow queries.

Example:
    ```python
    async for event in zap.stream_task(agent_name="Agent", task="..."):
        match event:
            case ThinkingEvent():
                print("Thinking...")
            case ToolCallEvent(phrase=phrase):
                print(phrase)
            case CompletedEvent(result=result):
                print(f"Done: {result}")
    ```
"""

from zap_ai.streaming.events import (
    CompletedEvent,
    ErrorEvent,
    Event,
    StreamEvent,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    generate_tool_phrase,
    parse_event,
)

__all__ = [
    # Base type
    "StreamEvent",
    # Event types
    "ThinkingEvent",
    "TokenEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "CompletedEvent",
    "ErrorEvent",
    # Type alias
    "Event",
    # Utilities
    "generate_tool_phrase",
    "parse_event",
]
