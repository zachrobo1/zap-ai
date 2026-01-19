"""Message types for LLM inference."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from zap_ai.utils import parse_tool_arguments

logger = logging.getLogger(__name__)

# =============================================================================
# Multimodal Content Types
# =============================================================================

# Valid detail levels for vision models
ImageDetailLevel = Literal["auto", "low", "high"]

# Valid image MIME types
VALID_IMAGE_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }
)


@dataclass
class TextContent:
    """
    Text content part in a multimodal message.

    Attributes:
        text: The text content.
        type: Content type identifier (always "text").
    """

    text: str
    type: Literal["text"] = "text"

    def to_litellm(self) -> dict[str, Any]:
        """Convert to LiteLLM format."""
        return {"type": "text", "text": self.text}

    @classmethod
    def from_litellm(cls, data: dict[str, Any]) -> TextContent:
        """Parse from LiteLLM format."""
        return cls(text=data.get("text", ""))


@dataclass
class ImageContent:
    """
    Image content part in a multimodal message.

    Supports URLs (including data: URIs for base64) per LiteLLM spec.

    Attributes:
        url: Image URL or data: URI with base64 content.
        type: Content type identifier (always "image_url").
        detail: Optional detail level ("auto", "low", "high") for vision models.
        format: Optional explicit mime type (e.g., "image/png").
    """

    url: str
    type: Literal["image_url"] = "image_url"
    detail: ImageDetailLevel | None = None
    format: str | None = None

    def to_litellm(self) -> dict[str, Any]:
        """Convert to LiteLLM format."""
        image_url: dict[str, Any] = {"url": self.url}
        if self.detail:
            image_url["detail"] = self.detail
        if self.format:
            image_url["format"] = self.format
        return {"type": "image_url", "image_url": image_url}

    @classmethod
    def from_litellm(cls, data: dict[str, Any]) -> ImageContent:
        """Parse from LiteLLM format."""
        img_data = data.get("image_url", {})
        if isinstance(img_data, str):
            # Handle case where image_url is just a string URL
            return cls(url=img_data)
        return cls(
            url=img_data.get("url", ""),
            detail=img_data.get("detail"),
            format=img_data.get("format"),
        )

    @classmethod
    def from_url(cls, url: str, detail: ImageDetailLevel | None = None) -> ImageContent:
        """
        Create from an image URL.

        Args:
            url: Image URL (http/https).
            detail: Optional detail level ("auto", "low", "high") for vision models.

        Returns:
            ImageContent instance.
        """
        return cls(url=url, detail=detail)

    @classmethod
    def from_base64(
        cls,
        data: str,
        mime_type: str = "image/png",
        detail: ImageDetailLevel | None = None,
    ) -> ImageContent:
        """
        Create from base64-encoded image data.

        Args:
            data: Base64-encoded image data (without data: prefix).
            mime_type: Image MIME type (e.g., "image/png", "image/jpeg").
                Must be a valid image/* type.
            detail: Optional detail level ("auto", "low", "high") for vision models.

        Returns:
            ImageContent instance with data: URI.

        Raises:
            ValueError: If mime_type is not a valid image type.
        """
        if mime_type not in VALID_IMAGE_MIME_TYPES:
            raise ValueError(
                f"Invalid image MIME type: '{mime_type}'. "
                f"Must be one of: {sorted(VALID_IMAGE_MIME_TYPES)}"
            )
        url = f"data:{mime_type};base64,{data}"
        return cls(url=url, detail=detail, format=mime_type)


# Type alias for content parts
ContentPart = TextContent | ImageContent

# Type alias for message content (backwards compatible - str still works)
MessageContent = str | list[ContentPart]


def _parse_content_parts(raw: list[dict[str, Any]]) -> list[ContentPart]:
    """
    Parse content parts from LiteLLM format.

    Args:
        raw: List of content part dicts from LiteLLM.

    Returns:
        List of ContentPart objects (unknown types are skipped with warning).
    """
    parts: list[ContentPart] = []
    for item in raw:
        content_type = item.get("type")
        if content_type == "text":
            parts.append(TextContent.from_litellm(item))
        elif content_type == "image_url":
            parts.append(ImageContent.from_litellm(item))
        else:
            # Log warning for unknown content types to aid debugging
            logger.warning(
                "Skipping unknown content type '%s' in message content",
                content_type,
            )
    return parts


def content_has_images(content: MessageContent | None) -> bool:
    """
    Check if content contains any images.

    Args:
        content: Message content (string or list of parts).

    Returns:
        True if content contains ImageContent, False otherwise.
    """
    if content is None:
        return False
    if isinstance(content, str):
        return False
    return any(isinstance(part, ImageContent) for part in content)


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

    Compatible with LiteLLM message format. Supports both text-only
    and multimodal (text + images) content.

    Attributes:
        role: Message role ("system", "user", "assistant", "tool").
        content: Message content - either a string (text-only) or
            list of ContentPart objects (multimodal).
        tool_calls: List of tool calls if this is an assistant message.
        tool_call_id: ID of the tool call this responds to (for tool role).
        name: Name of the tool (for tool role).
    """

    role: str
    content: MessageContent | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_litellm(self) -> dict[str, Any]:
        """Convert to LiteLLM message format."""
        msg: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            if isinstance(self.content, str):
                msg["content"] = self.content
            else:
                # Multimodal: list of content parts
                msg["content"] = [part.to_litellm() for part in self.content]

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

        # Parse content (may be string or list of content parts)
        raw_content = msg.get("content")
        content: MessageContent | None = None

        if raw_content is not None:
            if isinstance(raw_content, str):
                content = raw_content
            elif isinstance(raw_content, list):
                content = _parse_content_parts(raw_content)
            else:
                # Fallback: convert to string
                content = str(raw_content)

        return cls(
            role=msg.get("role", ""),
            content=content,
            tool_calls=tool_calls,
            tool_call_id=msg.get("tool_call_id"),
            name=msg.get("name"),
        )

    @classmethod
    def system(cls, content: str) -> Message:
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: MessageContent) -> Message:
        """
        Create a user message.

        Args:
            content: Text string or list of content parts (for multimodal).

        Returns:
            User message.
        """
        return cls(role="user", content=content)

    @classmethod
    def user_with_images(
        cls,
        text: str,
        images: list[ImageContent],
    ) -> Message:
        """
        Create a user message with text and images.

        Args:
            text: Text content to include.
            images: List of ImageContent objects.

        Returns:
            User message with multimodal content.
        """
        parts: list[ContentPart] = [TextContent(text=text)]
        parts.extend(images)
        return cls(role="user", content=parts)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        """Create an assistant message."""
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, tool_call_id: str, name: str, content: MessageContent) -> Message:
        """
        Create a tool result message.

        Args:
            tool_call_id: ID of the tool call this responds to.
            name: Name of the tool.
            content: Result content (text or multimodal).

        Returns:
            Tool result message.
        """
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)

    def has_images(self) -> bool:
        """
        Check if this message contains image content.

        Returns:
            True if message contains ImageContent, False otherwise.
        """
        return content_has_images(self.content)

    def get_text_content(self) -> str | None:
        """
        Extract text content from message, ignoring images.

        For multimodal messages, concatenates all text parts.

        Returns:
            Text content or None if no text.
        """
        if self.content is None:
            return None
        if isinstance(self.content, str):
            return self.content
        # Extract text from multimodal content
        text_parts = [part.text for part in self.content if isinstance(part, TextContent)]
        return " ".join(text_parts) if text_parts else None


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
