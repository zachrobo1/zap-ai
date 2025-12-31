# Phase 3: Activities (Stubs)

**Goal:** Activity interfaces with stub implementations that return mock data.

**Dependencies:** Phase 2 (Temporal Workflows)

---

## Overview

This phase creates stub activity implementations that the workflow can call. The stubs return mock data, allowing the full workflow loop to be tested without real LLM or MCP integrations.

**Why stubs first?**
- Validates Temporal activity patterns early
- No API costs during development
- Faster test execution
- Clear interfaces for Phase 5 (LLM) and Phase 6 (MCP) to implement

---

## Task 3.1: Inference Activity Stub

**File:** `src/zap_ai/activities/inference.py`

Create stub implementation for LLM inference:

```python
"""LLM inference activity for Temporal workflows.

This module provides stub implementations for Phase 2/3.
Real LLM integration will be added in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from temporalio import activity


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

    Serializable version for Temporal.

    Attributes:
        content: Text content of the response.
        tool_calls: List of tool call dicts.
        finish_reason: Why the LLM stopped.
    """

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"


@activity.defn
async def inference_activity(input: InferenceInput) -> InferenceOutput:
    """
    Execute LLM inference as a Temporal activity.

    STUB IMPLEMENTATION: Returns mock responses for testing.
    Real LLM integration will be added in Phase 5.

    Args:
        input: InferenceInput with agent name, model, messages, and tools.

    Returns:
        InferenceOutput with content and tool calls.
    """
    activity.logger.info(
        f"[STUB] Running inference for agent '{input.agent_name}' "
        f"with model '{input.model}' "
        f"({len(input.messages)} messages, {len(input.tools)} tools)"
    )

    # Get the last user message if any
    last_user_msg = ""
    for msg in reversed(input.messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    # Simple stub: echo back a response
    return InferenceOutput(
        content=f"[STUB] Processed: {last_user_msg[:100] if last_user_msg else 'no message'}",
        tool_calls=[],
        finish_reason="stop",
    )
```

**Checklist:**
- [x] Create `src/zap_ai/activities/inference.py`
- [x] Implement `InferenceInput` dataclass with default factories
- [x] Implement `InferenceOutput` dataclass (serializable for Temporal)
- [x] Implement `inference_activity` with `@activity.defn` decorator
- [x] Return stub response with `[STUB]` prefix
- [x] Add activity logging

---

## Task 3.2: Tool Execution Activity Stub

**File:** `src/zap_ai/activities/tool_execution.py`

Create stub implementation for MCP tool execution:

```python
"""Tool execution activity for Temporal workflows.

This module provides stub implementations for Phase 2/3.
Real MCP integration will be added in Phase 6.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from temporalio import activity


@dataclass
class ToolExecutionInput:
    """
    Input for the tool execution activity.

    Attributes:
        agent_name: Name of the agent executing the tool.
        tool_name: Name of the MCP tool to execute.
        arguments: Arguments to pass to the tool.
    """

    agent_name: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""

    pass


class ToolNotFoundError(Exception):
    """Raised when a tool is not found for the agent."""

    pass


# Global reference to tool registry (set during worker initialization)
_tool_registry: Any = None


def set_tool_registry(registry: Any) -> None:
    """
    Set the global tool registry for activities.

    Called during worker initialization to provide activities
    access to MCP clients.

    Args:
        registry: ToolRegistry instance.
    """
    global _tool_registry
    _tool_registry = registry


def get_tool_registry() -> Any:
    """Get the global tool registry."""
    if _tool_registry is None:
        raise RuntimeError(
            "Tool registry not initialized. "
            "Call set_tool_registry() during worker setup."
        )
    return _tool_registry


@activity.defn
async def tool_execution_activity(input: ToolExecutionInput) -> str:
    """
    Execute an MCP tool as a Temporal activity.

    STUB IMPLEMENTATION: Returns mock results for testing.
    Real MCP integration will be added in Phase 6.

    Args:
        input: ToolExecutionInput with agent name, tool name, and arguments.

    Returns:
        Tool execution result as a JSON string.
    """
    activity.logger.info(
        f"[STUB] Executing tool '{input.tool_name}' for agent '{input.agent_name}' "
        f"with args: {json.dumps(input.arguments)[:200]}"
    )

    # Simple stub: return a mock result
    return json.dumps(
        {
            "status": "success",
            "tool": input.tool_name,
            "result": f"[STUB] Mock result for {input.tool_name}",
            "args_received": input.arguments,
        }
    )
```

**Checklist:**
- [x] Create `src/zap_ai/activities/tool_execution.py`
- [x] Implement `ToolExecutionInput` dataclass
- [x] Implement `ToolExecutionError` and `ToolNotFoundError` exceptions
- [x] Implement global registry getter/setter for worker initialization
- [x] Implement `tool_execution_activity` with `@activity.defn` decorator
- [x] Return stub JSON response with status and mock result
- [x] Add activity logging

---

## Task 3.3: Update Activities Module Init

**File:** `src/zap_ai/activities/__init__.py`

Export all activity functions and types:

```python
"""Temporal activity definitions for Zap workflows."""

from zap_ai.activities.inference import (
    InferenceInput,
    InferenceOutput,
    inference_activity,
)
from zap_ai.activities.tool_execution import (
    ToolExecutionError,
    ToolExecutionInput,
    ToolNotFoundError,
    get_tool_registry,
    set_tool_registry,
    tool_execution_activity,
)

__all__ = [
    # Inference
    "inference_activity",
    "InferenceInput",
    "InferenceOutput",
    # Tool execution
    "tool_execution_activity",
    "ToolExecutionInput",
    "ToolExecutionError",
    "ToolNotFoundError",
    "set_tool_registry",
    "get_tool_registry",
]
```

**Checklist:**
- [x] Update `src/zap_ai/activities/__init__.py`
- [x] Import and export all activity functions and types
- [x] Define `__all__` list

---

## Phase 3 Verification

After completing all tasks, verify:

1. **Inference stub returns mock response:**
   ```python
   from zap_ai.activities import inference_activity, InferenceInput

   result = await inference_activity(
       InferenceInput(
           agent_name="Test",
           model="gpt-4o",
           messages=[{"role": "user", "content": "Hello"}],
           tools=[],
       )
   )
   assert "[STUB]" in result.content
   assert result.tool_calls == []
   ```

2. **Tool execution stub returns mock result:**
   ```python
   from zap_ai.activities import tool_execution_activity, ToolExecutionInput
   import json

   result = await tool_execution_activity(
       ToolExecutionInput(
           agent_name="Test",
           tool_name="search",
           arguments={"query": "test"},
       )
   )
   parsed = json.loads(result)
   assert parsed["status"] == "success"
   assert "[STUB]" in parsed["result"]
   ```

3. **Workflow can call activities:**
   ```python
   # The AgentWorkflow from Phase 2 should be able to:
   # - Call inference_activity and get stub responses
   # - Call tool_execution_activity and get stub results
   # - Complete the full agentic loop with mock data
   ```

4. **All tests pass:**
   ```bash
   uv run pytest tests/unit/activities/ -v
   ```

---

## Notes

- Stubs use `[STUB]` prefix in responses for easy identification in logs
- The `_tool_registry` global will be set by the worker in Phase 4
- Real implementations will be added in Phase 5 (LLM) and Phase 6 (MCP)
- Activity retry policies are configured in the workflow, not the activity itself
