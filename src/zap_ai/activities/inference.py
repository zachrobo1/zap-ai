"""LLM inference activity for Temporal workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from temporalio import activity

from zap_ai.llm.message_types import InferenceResult
from zap_ai.llm.provider import complete


@dataclass
class InferenceInput:
    """
    Input for the inference activity.

    Attributes:
        agent_name: Name of the agent making the inference.
        model: LiteLLM model identifier.
        messages: Conversation history in LiteLLM format.
        tools: Tool definitions in LiteLLM format.
    """

    agent_name: str
    model: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InferenceOutput:
    """
    Output from the inference activity.

    Serializable version of InferenceResult for Temporal.

    Attributes:
        content: Text content of the response.
        tool_calls: List of tool call dicts.
        finish_reason: Why the LLM stopped.
    """

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"

    @classmethod
    def from_result(cls, result: InferenceResult) -> InferenceOutput:
        """Convert from InferenceResult."""
        return cls(
            content=result.content,
            tool_calls=[tc.to_litellm() for tc in result.tool_calls],
            finish_reason=result.finish_reason,
        )


@activity.defn
async def inference_activity(input: InferenceInput) -> InferenceOutput:
    """
    Execute LLM inference as a Temporal activity.

    This activity wraps the LLM provider call and is configured
    with appropriate retry policies for handling transient failures.

    Args:
        input: InferenceInput with agent name, model, messages, and tools.

    Returns:
        InferenceOutput with content and tool calls.

    Raises:
        LLMProviderError: If inference fails after retries.
    """
    activity.logger.info(
        f"Running inference for agent '{input.agent_name}' "
        f"with model '{input.model}' "
        f"({len(input.messages)} messages, {len(input.tools)} tools)"
    )
    result = await complete(
        model=input.model,
        messages=input.messages,
        tools=input.tools if input.tools else None,
    )

    activity.logger.info(
        f"Inference complete: "
        f"content={'yes' if result.content else 'no'}, "
        f"tool_calls={len(result.tool_calls)}"
    )

    return InferenceOutput.from_result(result)
