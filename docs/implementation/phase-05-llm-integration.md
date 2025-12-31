# Phase 5: LLM Integration

**Goal:** Wrap LiteLLM for inference with proper tool handling.

**Dependencies:** Phase 4 (Worker).

---

## Task 3.1: Implement Message Types

**File:** `src/zap_ai/llm/message_types.py`

Create dataclasses compatible with LiteLLM format:

- `Message` dataclass compatible with LiteLLM format
- `ToolCall` dataclass: `id`, `name`, `arguments`
- `InferenceResult` dataclass: `content`, `tool_calls`

```python
"""Message types for LLM inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json


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
    def from_litellm(cls, tool_call: dict[str, Any]) -> "ToolCall":
        """
        Parse a tool call from LiteLLM response format.

        Args:
            tool_call: Tool call dict from LiteLLM response.

        Returns:
            ToolCall instance.
        """
        func = tool_call.get("function", {})
        args_raw = func.get("arguments", "{}")

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}

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
            }
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
    def from_litellm(cls, msg: dict[str, Any]) -> "Message":
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
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool_result(cls, tool_call_id: str, name: str, content: str) -> "Message":
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
```

**Checklist:**
- [ ] Create `src/zap_ai/llm/message_types.py`
- [ ] Implement `ToolCall` dataclass with `from_litellm()` and `to_litellm()`
- [ ] Implement `Message` dataclass with factory methods and serialization
- [ ] Implement `InferenceResult` dataclass
- [ ] Add comprehensive docstrings

---

## Task 3.2: Implement LLM Provider Wrapper

**File:** `src/zap_ai/llm/provider.py`

Wrap LiteLLM for inference with proper tool handling:

- `async def complete(model, messages, tools) -> InferenceResult`
- Parse tool_calls from LiteLLM response
- Handle errors gracefully (let Temporal retry)

```python
"""LiteLLM provider wrapper for LLM inference."""

from __future__ import annotations

from typing import Any

import litellm

from zap_ai.llm.message_types import InferenceResult, ToolCall


class LLMProviderError(Exception):
    """Raised when LLM provider call fails."""
    pass


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
            tool_calls.append(ToolCall.from_litellm({
                "id": tc.id,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }))

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
```

**Checklist:**
- [ ] Create `src/zap_ai/llm/provider.py`
- [ ] Implement `LLMProviderError` exception
- [ ] Implement `complete()` async function
- [ ] Parse tool calls from LiteLLM response
- [ ] Parse usage statistics
- [ ] Implement `convert_messages_to_litellm()` helper
- [ ] Handle errors gracefully for Temporal retry

---

## Task 3.3: Implement Inference Activity

**File:** `src/zap_ai/activities/inference.py`

Create the Temporal activity for LLM inference:

- `@activity.defn async def inference_activity(input: InferenceInput) -> InferenceResult`
- Input: `agent_name`, `messages`
- Fetch tools from registry, call LLM, return result

```python
"""LLM inference activity for Temporal workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from temporalio import activity

from zap_ai.llm.provider import complete
from zap_ai.llm.message_types import InferenceResult


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
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


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
    tool_calls: list[dict[str, Any]]
    finish_reason: str

    @classmethod
    def from_result(cls, result: InferenceResult) -> "InferenceOutput":
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
```

**Checklist:**
- [ ] Create `src/zap_ai/activities/inference.py`
- [ ] Implement `InferenceInput` dataclass
- [ ] Implement `InferenceOutput` dataclass (serializable for Temporal)
- [ ] Implement `inference_activity` with `@activity.defn` decorator
- [ ] Add activity logging
- [ ] Handle error propagation for Temporal retry

---

## Task 3.4: Update LLM Module Init

**File:** `src/zap_ai/llm/__init__.py`

```python
"""LLM provider integration via LiteLLM."""

from zap_ai.llm.message_types import (
    Message,
    ToolCall,
    InferenceResult,
)
from zap_ai.llm.provider import (
    complete,
    convert_messages_to_litellm,
    LLMProviderError,
)

__all__ = [
    # Types
    "Message",
    "ToolCall",
    "InferenceResult",
    # Functions
    "complete",
    "convert_messages_to_litellm",
    # Exceptions
    "LLMProviderError",
]
```

**Checklist:**
- [ ] Update `src/zap_ai/llm/__init__.py`
- [ ] Import and export all public types and functions
- [ ] Define `__all__` list

---

## Phase 3 Verification

After completing all tasks, verify:

1. **Message types work:**
   ```python
   from zap_ai.llm import Message, ToolCall, InferenceResult

   # Create messages
   msg = Message.user("Hello")
   assert msg.to_litellm() == {"role": "user", "content": "Hello"}

   # Parse tool calls
   tc = ToolCall.from_litellm({
       "id": "call_123",
       "function": {"name": "search", "arguments": '{"query": "test"}'}
   })
   assert tc.name == "search"
   assert tc.arguments == {"query": "test"}
   ```

2. **Provider wrapper works:**
   ```python
   from zap_ai.llm import complete

   result = await complete(
       model="gpt-4o-mini",
       messages=[{"role": "user", "content": "Say hello"}],
   )
   assert result.content is not None
   assert result.is_complete
   ```

3. **Activity runs in Temporal:**
   ```python
   from zap_ai.activities.inference import inference_activity, InferenceInput

   # In a workflow test
   result = await workflow.execute_activity(
       inference_activity,
       InferenceInput(
           agent_name="Test",
           model="gpt-4o-mini",
           messages=[{"role": "user", "content": "Hello"}],
           tools=[],
       ),
       start_to_close_timeout=timedelta(seconds=60),
   )
   assert result.content is not None
   ```
