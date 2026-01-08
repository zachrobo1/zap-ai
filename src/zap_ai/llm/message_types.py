"""Message types for LLM inference."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from zap_ai.utils import parse_tool_arguments


@dataclass
class ToolCall:
    """
    Represents a tool call requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call (from LLM response).
        name: Name of the tool to execute.
        arguments: Parsed arguments dict for the tool.
        arguments_raw: Raw JSON string of arguments (for serialization).
    """

    id: str
    name: str
    arguments: dict[str, Any]
    arguments_raw: str = ""

    @classmethod
    def from_litellm(cls, tool_call: dict[str, Any]) -> ToolCall:
        """
        Parse a tool call from LiteLLM response format.

        Args:
            tool_call: Tool call dict from LiteLLM response.

        Returns:
            ToolCall instance.
        """
        func = tool_call.get("function", {})
        args_raw = func.get("arguments", "{}")
        args = parse_tool_arguments(args_raw)

        return cls(
            id=tool_call.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
            arguments_raw=args_raw if isinstance(args_raw, str) else json.dumps(args_raw),
        )

    def to_litellm(self) -> dict[str, Any]:
        """Convert to LiteLLM format for message history."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments_raw,
            },
        }


@dataclass
class Message:
    """
    Represents a message in the conversation history.

    Compatible with LiteLLM message format.

    Attributes:
        role: Message role ("system", "user", "assistant", "tool").
        content: Message content (text).
        tool_calls: List of tool calls if this is an assistant message.
        tool_call_id: ID of the tool call this responds to (for tool role).
        name: Name of the tool (for tool role).
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_litellm(self) -> dict[str, Any]:
        """Convert to LiteLLM message format."""
        msg: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            msg["content"] = self.content

        if self.tool_calls:
            msg["tool_calls"] = [tc.to_litellm() for tc in self.tool_calls]

        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id

        if self.name is not None:
            msg["name"] = self.name

        return msg

    @classmethod
    def from_litellm(cls, msg: dict[str, Any]) -> Message:
        """Parse a message from LiteLLM format."""
        tool_calls = []
        if "tool_calls" in msg and msg["tool_calls"]:
            tool_calls = [ToolCall.from_litellm(tc) for tc in msg["tool_calls"]]

        return cls(
            role=msg.get("role", ""),
            content=msg.get("content"),
            tool_calls=tool_calls,
            tool_call_id=msg.get("tool_call_id"),
            name=msg.get("name"),
        )

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        """Create an assistant message."""
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, tool_call_id: str, name: str, content: str) -> Message:
        """Create a tool result message."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


@dataclass
class InferenceResult:
    """
    Result of an LLM inference call.

    Attributes:
        content: Text content of the response (may be None if tool calls).
        tool_calls: List of tool calls requested by the LLM.
        finish_reason: Why the LLM stopped ("stop", "tool_calls", "length").
        usage: Token usage dict if available.
    """

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def is_complete(self) -> bool:
        """Check if this is a final response (no tool calls)."""
        return not self.has_tool_calls

    def to_message(self) -> Message:
        """Convert to a Message for conversation history."""
        return Message.assistant(content=self.content, tool_calls=self.tool_calls)
