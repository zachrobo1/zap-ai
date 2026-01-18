"""LiteLLM provider wrapper for LLM inference."""

from __future__ import annotations

from typing import Any

import litellm

from zap_ai.exceptions import LLMProviderError, VisionNotSupportedError
from zap_ai.llm.message_types import InferenceResult, ToolCall


def supports_vision(model: str) -> bool:
    """
    Check if a model supports vision/image inputs.

    Uses LiteLLM's model capability checking.

    Args:
        model: LiteLLM model identifier (e.g., "gpt-4o", "anthropic/claude-sonnet-4-20250514").

    Returns:
        True if the model supports vision, False otherwise.
    """
    try:
        return litellm.supports_vision(model)
    except Exception:
        # If capability check fails, assume no vision support for safety
        return False


def _messages_contain_images(messages: list[dict[str, Any]]) -> bool:
    """
    Check if any message contains image content.

    Args:
        messages: List of messages in LiteLLM format.

    Returns:
        True if any message contains image_url content.
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def validate_vision_support(model: str, messages: list[dict[str, Any]]) -> None:
    """
    Validate that the model supports vision if messages contain images.

    Args:
        model: LiteLLM model identifier.
        messages: Messages to check for image content.

    Raises:
        VisionNotSupportedError: If messages contain images but model
            doesn't support vision.
    """
    if not _messages_contain_images(messages):
        return

    if not supports_vision(model):
        raise VisionNotSupportedError(
            f"Model '{model}' does not support vision. Cannot process messages containing images."
        )


async def complete(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> InferenceResult:
    """
    Call LLM for completion via LiteLLM.

    Args:
        model: LiteLLM model identifier (e.g., "gpt-4o", "claude-3-opus").
        messages: Conversation history in LiteLLM format.
        tools: Optional list of tool definitions in LiteLLM format.
        temperature: Sampling temperature (0.0 - 2.0).
        max_tokens: Maximum tokens to generate.

    Returns:
        InferenceResult with content and/or tool calls.

    Raises:
        LLMProviderError: If the LLM call fails.
        VisionNotSupportedError: If messages contain images but model
            doesn't support vision.
    """
    # Validate vision support as a safety net (defense in depth)
    validate_vision_support(model, messages)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    try:
        response = await litellm.acompletion(**kwargs)
    except Exception as e:
        raise LLMProviderError(f"LLM call failed: {e}") from e

    # Parse response
    choice = response.choices[0]
    message = choice.message

    # Parse tool calls if present
    tool_calls: list[ToolCall] = []
    if hasattr(message, "tool_calls") and message.tool_calls:
        for tc in message.tool_calls:
            tool_calls.append(
                ToolCall.from_litellm(
                    {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
            )

    # Parse usage
    usage = {}
    if hasattr(response, "usage") and response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return InferenceResult(
        content=message.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason or "stop",
        usage=usage,
    )


def convert_messages_to_litellm(messages: list[Any]) -> list[dict[str, Any]]:
    """
    Convert Message objects to LiteLLM format dicts.

    Handles both Message objects and raw dicts.

    Args:
        messages: List of Message objects or dicts.

    Returns:
        List of dicts in LiteLLM format.
    """
    result = []
    for msg in messages:
        if hasattr(msg, "to_litellm"):
            result.append(msg.to_litellm())
        elif isinstance(msg, dict):
            result.append(msg)
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")
    return result
