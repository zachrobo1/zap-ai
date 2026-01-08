"""LiteLLM provider wrapper for LLM inference."""

from __future__ import annotations

from typing import Any

import litellm

from zap_ai.exceptions import LLMProviderError
from zap_ai.llm.message_types import InferenceResult, ToolCall


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
    """
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
