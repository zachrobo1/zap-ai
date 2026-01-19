"""
Streaming event types for real-time task progress updates.

This module defines event dataclasses for streaming task execution events
via Temporal workflow queries.

Example:
    ```python
    async for event in zap.stream_task(agent_name="Agent", task="..."):
        match event:
            case ThinkingEvent(content=content):
                print(f"Thinking: {content}")
            case ToolCallEvent(phrase=phrase):
                print(phrase)  # "Getting weather for London..."
            case CompletedEvent(result=result):
                print(f"Done: {result}")
    ```
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class StreamEvent:
    """
    Base class for all streaming events.

    Attributes:
        type: Event type identifier.
        timestamp: ISO format timestamp when event occurred.
        task_id: ID of the task this event belongs to.
        seq: Sequence number for ordering events.
    """

    type: str
    timestamp: str
    task_id: str
    seq: int


@dataclass
class ThinkingEvent(StreamEvent):
    """
    Emitted when the LLM starts reasoning/generating.

    Attributes:
        iteration: Current agentic loop iteration number.
    """

    iteration: int

    def __init__(
        self,
        iteration: int,
        task_id: str = "",
        seq: int = 0,
        timestamp: str | None = None,
    ) -> None:
        self.type = "thinking"
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.task_id = task_id
        self.seq = seq
        self.iteration = iteration


@dataclass
class ToolCallEvent(StreamEvent):
    """
    Emitted when a tool call is about to be executed.

    Attributes:
        name: Tool name being called.
        arguments: Tool call arguments.
        phrase: Human-readable description of the tool call.
        tool_call_id: Unique identifier for this tool call.
    """

    name: str
    arguments: dict[str, Any]
    phrase: str
    tool_call_id: str

    def __init__(
        self,
        name: str,
        arguments: dict[str, Any],
        phrase: str,
        tool_call_id: str,
        task_id: str = "",
        seq: int = 0,
        timestamp: str | None = None,
    ) -> None:
        self.type = "tool_call"
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.task_id = task_id
        self.seq = seq
        self.name = name
        self.arguments = arguments
        self.phrase = phrase
        self.tool_call_id = tool_call_id


@dataclass
class ToolResultEvent(StreamEvent):
    """
    Emitted when a tool call completes.

    Attributes:
        name: Tool name that was called.
        result: Tool execution result (may be truncated).
        tool_call_id: Unique identifier matching the ToolCallEvent.
        success: Whether the tool executed successfully.
    """

    name: str
    result: str
    tool_call_id: str
    success: bool

    def __init__(
        self,
        name: str,
        result: str,
        tool_call_id: str,
        success: bool,
        task_id: str = "",
        seq: int = 0,
        timestamp: str | None = None,
    ) -> None:
        self.type = "tool_result"
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.task_id = task_id
        self.seq = seq
        self.name = name
        self.result = result
        self.tool_call_id = tool_call_id
        self.success = success


@dataclass
class TokenEvent(StreamEvent):
    """
    Emitted for individual tokens during streaming (Phase 2 only).

    Attributes:
        token: The token text.
        index: Token index in the current generation.
    """

    token: str
    index: int

    def __init__(
        self,
        token: str,
        index: int,
        task_id: str = "",
        seq: int = 0,
        timestamp: str | None = None,
    ) -> None:
        self.type = "token"
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.task_id = task_id
        self.seq = seq
        self.token = token
        self.index = index


@dataclass
class CompletedEvent(StreamEvent):
    """
    Emitted when the task completes successfully.

    Attributes:
        result: Final result string from the agent.
    """

    result: str

    def __init__(
        self,
        result: str,
        task_id: str = "",
        seq: int = 0,
        timestamp: str | None = None,
    ) -> None:
        self.type = "completed"
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.task_id = task_id
        self.seq = seq
        self.result = result


@dataclass
class ErrorEvent(StreamEvent):
    """
    Emitted when the task fails.

    Attributes:
        error: Error message describing the failure.
    """

    error: str

    def __init__(
        self,
        error: str,
        task_id: str = "",
        seq: int = 0,
        timestamp: str | None = None,
    ) -> None:
        self.type = "error"
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()
        self.task_id = task_id
        self.seq = seq
        self.error = error


# Type alias for all event types
Event = ThinkingEvent | ToolCallEvent | ToolResultEvent | TokenEvent | CompletedEvent | ErrorEvent


def generate_tool_phrase(tool_name: str, description: str | None, arguments: dict[str, Any]) -> str:
    """
    Generate a human-readable phrase for a tool call.

    Converts tool descriptions to present participle form and substitutes
    argument values for a user-friendly status message.

    Args:
        tool_name: Name of the tool being called.
        description: Tool description from MCP registry.
        arguments: Arguments being passed to the tool.

    Returns:
        Human-readable phrase like "Getting weather for London..."

    Example:
        >>> generate_tool_phrase("get_weather", "Get weather for a city", {"city": "London"})
        "Getting weather for London..."
    """
    if not description:
        return f"Calling {tool_name}..."

    phrase = description

    # Convert to -ing form for common verbs
    verb_mappings = {
        "Get ": "Getting ",
        "Search ": "Searching ",
        "Create ": "Creating ",
        "Update ": "Updating ",
        "Delete ": "Deleting ",
        "Send ": "Sending ",
        "Fetch ": "Fetching ",
        "Find ": "Finding ",
        "Check ": "Checking ",
        "Validate ": "Validating ",
        "Process ": "Processing ",
        "Generate ": "Generating ",
        "Calculate ": "Calculating ",
        "Analyze ": "Analyzing ",
        "Read ": "Reading ",
        "Write ": "Writing ",
        "List ": "Listing ",
        "Query ": "Querying ",
        "Execute ": "Executing ",
        "Run ": "Running ",
    }

    for verb, replacement in verb_mappings.items():
        if phrase.startswith(verb):
            phrase = replacement + phrase[len(verb) :]
            break

    # Substitute argument values into the phrase
    for key, value in arguments.items():
        str_value = str(value)
        # Try various placeholder patterns
        phrase = phrase.replace(f"{{{key}}}", str_value)
        phrase = phrase.replace(f"a {key}", str_value)
        phrase = phrase.replace(f"the {key}", str_value)
        # Also try with underscores converted to spaces
        key_spaced = key.replace("_", " ")
        phrase = phrase.replace(f"a {key_spaced}", str_value)
        phrase = phrase.replace(f"the {key_spaced}", str_value)

    # Ensure phrase ends with ellipsis
    phrase = phrase.rstrip(".")
    if not phrase.endswith("..."):
        phrase += "..."

    return phrase


def parse_event(event_data: dict[str, Any], task_id: str) -> Event:
    """
    Parse a raw event dictionary into a typed Event object.

    Args:
        event_data: Raw event data from workflow query.
        task_id: ID of the task this event belongs to.

    Returns:
        Typed event object.

    Raises:
        ValueError: If event type is unknown.
    """
    event_type = event_data.get("type")
    seq = event_data.get("seq", 0)
    timestamp = event_data.get("timestamp", "")

    if event_type == "thinking":
        return ThinkingEvent(
            iteration=event_data.get("iteration", 0),
            task_id=task_id,
            seq=seq,
            timestamp=timestamp,
        )
    elif event_type == "tool_call":
        return ToolCallEvent(
            name=event_data.get("name", ""),
            arguments=event_data.get("arguments", {}),
            phrase=event_data.get("phrase", ""),
            tool_call_id=event_data.get("tool_call_id", ""),
            task_id=task_id,
            seq=seq,
            timestamp=timestamp,
        )
    elif event_type == "tool_result":
        return ToolResultEvent(
            name=event_data.get("name", ""),
            result=event_data.get("result", ""),
            tool_call_id=event_data.get("tool_call_id", ""),
            success=event_data.get("success", True),
            task_id=task_id,
            seq=seq,
            timestamp=timestamp,
        )
    elif event_type == "token":
        return TokenEvent(
            token=event_data.get("token", ""),
            index=event_data.get("index", 0),
            task_id=task_id,
            seq=seq,
            timestamp=timestamp,
        )
    elif event_type == "completed":
        return CompletedEvent(
            result=event_data.get("result", ""),
            task_id=task_id,
            seq=seq,
            timestamp=timestamp,
        )
    elif event_type == "error":
        return ErrorEvent(
            error=event_data.get("error", ""),
            task_id=task_id,
            seq=seq,
            timestamp=timestamp,
        )
    else:
        raise ValueError(f"Unknown event type: {event_type}")
