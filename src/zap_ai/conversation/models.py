"""Data models for conversation history."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallInfo:
    """
    Information about a tool call in the conversation.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool that was called.
        arguments: Parsed arguments dict passed to the tool.
        result: The tool's result, if available.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    result: str | None = None


@dataclass
class ConversationTurn:
    """
    A single turn in the conversation.

    A turn consists of a user message (or system prompt for turn 0),
    followed by all assistant responses and tool interactions until
    the next user message.

    Attributes:
        turn_number: Zero-indexed turn number.
        user_message: The user (or system) message that started this turn.
        assistant_messages: All assistant responses in this turn.
        tool_messages: All tool result messages in this turn.
    """

    turn_number: int
    user_message: dict[str, Any] | None = None
    assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
