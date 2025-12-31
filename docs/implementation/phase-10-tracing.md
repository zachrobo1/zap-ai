# Phase 10: Tracing

**Goal:** Add comprehensive observability via an abstract `TracingProvider` protocol with Langfuse as the first implementation.

**Dependencies:** Phase 7 (Orchestration) - requires Zap class and activity infrastructure to exist.

**Note:** This phase can be implemented independently after the core functionality works. Tracing is opt-in.

---

## Overview

### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Abstract protocol + Langfuse impl | Easy to swap backends (OTEL, etc.) later |
| Granularity | Full tracing | Workflow lifecycle, iterations, inference, tools, sub-agents |
| Activation | Opt-in via config | Pass `tracing_provider` to Zap constructor |
| Context propagation | Serialize in activity inputs | Temporal activities run in separate processes |

### Module Structure

```
src/zap_ai/tracing/
    __init__.py                 # Public exports + global registry
    protocol.py                 # TracingProvider protocol, TraceContext, SpanKind
    noop_provider.py            # No-op provider (default when tracing disabled)
    langfuse_provider.py        # Langfuse implementation
```

### Trace Hierarchy

```
Trace: task-MyAgent-abc123
  +-- Span: workflow-lifecycle (WORKFLOW)
  +-- Span: iteration-0 (ITERATION)
  |     +-- Generation: inference-MyAgent (INFERENCE)
  |     +-- Span: tool-search (TOOL_CALL)
  |     +-- Span: tool-calculate (TOOL_CALL)
  +-- Span: iteration-1 (ITERATION)
  |     +-- Generation: inference-MyAgent (INFERENCE)
  |     +-- Span: sub-agent-Helper (SUB_AGENT)
  |           +-- [Linked child trace]
```

---

## Task 10.1: Core Protocol Definition

**File:** `src/zap_ai/tracing/protocol.py`

Define the `TracingProvider` protocol and supporting types:

```python
"""Tracing protocol and context for observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator


class SpanKind(str, Enum):
    """Type of span for semantic categorization."""
    WORKFLOW = "workflow"
    ITERATION = "iteration"
    INFERENCE = "inference"
    TOOL_CALL = "tool_call"
    SUB_AGENT = "sub_agent"


@dataclass
class TraceContext:
    """
    Serializable trace context for propagation across Temporal boundaries.

    This is passed through activity inputs to maintain trace continuity.
    Must be JSON-serializable for Temporal.

    Attributes:
        trace_id: Unique identifier for the trace.
        span_id: Unique identifier for the current span.
        provider_data: Provider-specific data (e.g., Langfuse observation ID).
    """
    trace_id: str
    span_id: str
    provider_data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Temporal activity input."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "provider_data": self.provider_data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceContext:
        """Deserialize from Temporal activity input."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            provider_data=data.get("provider_data"),
        )


@runtime_checkable
class TracingProvider(Protocol):
    """
    Protocol for tracing backends.

    Implementations must be async-safe and handle context propagation
    across Temporal boundaries.

    The provider is responsible for:
    - Creating traces (root spans) for workflows
    - Creating child spans for iterations, tool calls, etc.
    - Creating generation spans for LLM calls (with token usage)
    - Tracking errors and events
    - Flushing and cleanup
    """

    @asynccontextmanager
    async def start_trace(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterator[TraceContext]:
        """
        Start a new trace (root span).

        Called at workflow start. Returns context for propagation.

        Args:
            name: Name of the trace (e.g., "task-AgentName-taskId").
            session_id: Optional session for grouping traces.
            user_id: Optional user identifier.
            metadata: Additional metadata to attach.
            tags: Optional tags for filtering.

        Yields:
            TraceContext for propagation to activities and child workflows.
        """
        ...

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_context: TraceContext,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[TraceContext]:
        """
        Start a child span within an existing trace.

        Used for iterations, tool calls, sub-agent delegation, etc.

        Args:
            name: Name of the span (e.g., "iteration-0", "tool-search").
            kind: Type of span for categorization.
            parent_context: Context from parent span.
            metadata: Additional metadata to attach.

        Yields:
            TraceContext for nested spans.
        """
        ...

    async def start_generation(
        self,
        name: str,
        parent_context: TraceContext,
        model: str,
        input_messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        """
        Start an LLM generation span.

        For Langfuse-style generation tracking with model info and usage.
        Must be explicitly ended with end_generation().

        Args:
            name: Name of the generation (e.g., "inference-AgentName").
            parent_context: Context from parent span.
            model: LLM model identifier.
            input_messages: Input messages sent to the LLM.
            metadata: Additional metadata.

        Returns:
            TraceContext that must be passed to end_generation().
        """
        ...

    async def end_generation(
        self,
        context: TraceContext,
        output: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> None:
        """
        End an LLM generation span with output and usage.

        Args:
            context: Context from start_generation().
            output: LLM output (content, tool_calls, etc.).
            usage: Token usage dict (prompt_tokens, completion_tokens, total_tokens).
        """
        ...

    async def add_event(
        self,
        context: TraceContext,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an event to the current span.

        Used for logging significant occurrences (status changes, etc.).

        Args:
            context: Current trace context.
            name: Event name.
            attributes: Event attributes.
        """
        ...

    async def set_error(
        self,
        context: TraceContext,
        error: Exception,
    ) -> None:
        """
        Mark the span as errored.

        Args:
            context: Current trace context.
            error: The exception that occurred.
        """
        ...

    async def flush(self) -> None:
        """Flush any pending trace data."""
        ...

    async def shutdown(self) -> None:
        """Cleanup tracing resources."""
        ...
```

**Checklist:**
- [ ] Create `src/zap_ai/tracing/protocol.py`
- [ ] Implement `SpanKind` enum with all span types
- [ ] Implement `TraceContext` dataclass with serialization methods
- [ ] Define `TracingProvider` protocol with all methods
- [ ] Add comprehensive docstrings

---

## Task 10.2: NoOp Provider Implementation

**File:** `src/zap_ai/tracing/noop_provider.py`

Default provider when tracing is not configured:

```python
"""No-operation tracing provider."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from collections.abc import AsyncIterator
from uuid import uuid4

from zap_ai.tracing.protocol import TraceContext, SpanKind


class NoOpTracingProvider:
    """
    No-operation tracing provider.

    Used when tracing is not configured. All operations are no-ops
    but return valid contexts for code compatibility.
    """

    @asynccontextmanager
    async def start_trace(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterator[TraceContext]:
        """Return a dummy context."""
        yield TraceContext(trace_id=uuid4().hex, span_id=uuid4().hex)

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_context: TraceContext,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[TraceContext]:
        """Return a dummy context with same trace_id."""
        yield TraceContext(
            trace_id=parent_context.trace_id,
            span_id=uuid4().hex,
        )

    async def start_generation(
        self,
        name: str,
        parent_context: TraceContext,
        model: str,
        input_messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        """Return a dummy context."""
        return TraceContext(
            trace_id=parent_context.trace_id,
            span_id=uuid4().hex,
        )

    async def end_generation(
        self,
        context: TraceContext,
        output: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> None:
        """No-op."""
        pass

    async def add_event(
        self,
        context: TraceContext,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """No-op."""
        pass

    async def set_error(
        self,
        context: TraceContext,
        error: Exception,
    ) -> None:
        """No-op."""
        pass

    async def flush(self) -> None:
        """No-op."""
        pass

    async def shutdown(self) -> None:
        """No-op."""
        pass
```

**Checklist:**
- [ ] Create `src/zap_ai/tracing/noop_provider.py`
- [ ] Implement all methods as no-ops
- [ ] Return valid TraceContext objects for code compatibility

---

## Task 10.3: Tracing Module Init with Global Registry

**File:** `src/zap_ai/tracing/__init__.py`

Module initialization with global provider registry for activity access:

```python
"""
Tracing module for Zap observability.

Provides an abstract TracingProvider protocol with a Langfuse implementation.
Tracing is opt-in via the Zap constructor.

Usage:
    from zap_ai import Zap
    from zap_ai.tracing import LangfuseTracingProvider

    tracing = LangfuseTracingProvider(
        public_key="pk-...",
        secret_key="sk-...",
    )

    zap = Zap(
        agents=[agent],
        tracing_provider=tracing,
    )
"""

from __future__ import annotations

from zap_ai.tracing.protocol import (
    TracingProvider,
    TraceContext,
    SpanKind,
)
from zap_ai.tracing.noop_provider import NoOpTracingProvider

# Global provider for activities (set during worker initialization)
_tracing_provider: TracingProvider | None = None


def set_tracing_provider(provider: TracingProvider) -> None:
    """
    Set the global tracing provider for activities.

    Called during worker initialization to provide activities
    access to the tracing provider.

    Args:
        provider: TracingProvider instance.
    """
    global _tracing_provider
    _tracing_provider = provider


def get_tracing_provider() -> TracingProvider:
    """
    Get the global tracing provider.

    Returns NoOpTracingProvider if not configured.

    Returns:
        TracingProvider instance.
    """
    global _tracing_provider
    if _tracing_provider is None:
        _tracing_provider = NoOpTracingProvider()
    return _tracing_provider


__all__ = [
    # Protocol and types
    "TracingProvider",
    "TraceContext",
    "SpanKind",
    # Providers
    "NoOpTracingProvider",
    # Global registry
    "set_tracing_provider",
    "get_tracing_provider",
]

# Lazy import to avoid circular dependencies and allow optional langfuse
def __getattr__(name: str):
    if name == "LangfuseTracingProvider":
        from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider
        return LangfuseTracingProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

**Checklist:**
- [ ] Create `src/zap_ai/tracing/__init__.py`
- [ ] Implement global provider registry (`set_tracing_provider`, `get_tracing_provider`)
- [ ] Export all public types
- [ ] Use lazy import for LangfuseTracingProvider (optional dependency)

---

## Task 10.4: Langfuse Provider Implementation

**File:** `src/zap_ai/tracing/langfuse_provider.py`

Langfuse implementation of TracingProvider:

```python
"""Langfuse tracing provider implementation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from collections.abc import AsyncIterator
from uuid import uuid4

from zap_ai.tracing.protocol import TracingProvider, TraceContext, SpanKind

try:
    from langfuse import Langfuse
except ImportError:
    raise ImportError(
        "langfuse is required for LangfuseTracingProvider. "
        "Install with: pip install zap-ai[langfuse]"
    )


class LangfuseTracingProvider:
    """
    Langfuse implementation of TracingProvider.

    Uses Langfuse's low-level SDK for async-compatible tracing.

    Attributes:
        public_key: Langfuse public key (or set LANGFUSE_PUBLIC_KEY env var).
        secret_key: Langfuse secret key (or set LANGFUSE_SECRET_KEY env var).
        host: Langfuse host URL (defaults to cloud).
    """

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
    ):
        """
        Initialize Langfuse tracing provider.

        Args:
            public_key: Langfuse public key (or use LANGFUSE_PUBLIC_KEY env var).
            secret_key: Langfuse secret key (or use LANGFUSE_SECRET_KEY env var).
            host: Langfuse host URL (defaults to https://cloud.langfuse.com).
        """
        self._langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        # Track active traces and spans for context reconstruction
        self._active_traces: dict[str, Any] = {}
        self._active_spans: dict[str, Any] = {}

    def _get_or_create_trace(self, trace_id: str) -> Any:
        """Get existing trace or create reference to it."""
        if trace_id not in self._active_traces:
            self._active_traces[trace_id] = self._langfuse.trace(id=trace_id)
        return self._active_traces[trace_id]

    @asynccontextmanager
    async def start_trace(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> AsyncIterator[TraceContext]:
        """Start a new Langfuse trace."""
        trace_id = uuid4().hex
        span_id = uuid4().hex

        trace = self._langfuse.trace(
            id=trace_id,
            name=name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        )

        self._active_traces[trace_id] = trace

        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            provider_data={"langfuse_trace_id": trace_id},
        )

        try:
            yield context
        finally:
            self._active_traces.pop(trace_id, None)

    @asynccontextmanager
    async def start_span(
        self,
        name: str,
        kind: SpanKind,
        parent_context: TraceContext,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator[TraceContext]:
        """Start a child span in Langfuse."""
        span_id = uuid4().hex
        trace = self._get_or_create_trace(parent_context.trace_id)

        span = trace.span(
            id=span_id,
            name=name,
            metadata={
                "kind": kind.value,
                **(metadata or {}),
            },
        )

        self._active_spans[span_id] = span

        context = TraceContext(
            trace_id=parent_context.trace_id,
            span_id=span_id,
            provider_data={
                "langfuse_trace_id": parent_context.trace_id,
                "langfuse_span_id": span_id,
            },
        )

        try:
            yield context
        finally:
            span.end()
            self._active_spans.pop(span_id, None)

    async def start_generation(
        self,
        name: str,
        parent_context: TraceContext,
        model: str,
        input_messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        """Start a Langfuse generation for LLM calls."""
        span_id = uuid4().hex
        trace = self._get_or_create_trace(parent_context.trace_id)

        generation = trace.generation(
            id=span_id,
            name=name,
            model=model,
            input=input_messages,
            metadata=metadata,
        )

        self._active_spans[span_id] = generation

        return TraceContext(
            trace_id=parent_context.trace_id,
            span_id=span_id,
            provider_data={
                "langfuse_trace_id": parent_context.trace_id,
                "langfuse_generation_id": span_id,
            },
        )

    async def end_generation(
        self,
        context: TraceContext,
        output: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> None:
        """End a Langfuse generation with output and usage."""
        generation = self._active_spans.get(context.span_id)
        if generation:
            generation.end(
                output=output,
                usage=usage,
            )
            self._active_spans.pop(context.span_id, None)

    async def add_event(
        self,
        context: TraceContext,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the Langfuse trace."""
        trace = self._get_or_create_trace(context.trace_id)
        trace.event(name=name, metadata=attributes)

    async def set_error(
        self,
        context: TraceContext,
        error: Exception,
    ) -> None:
        """Mark the Langfuse span as errored."""
        span = self._active_spans.get(context.span_id)
        if span:
            span.update(
                level="ERROR",
                status_message=str(error),
            )

    async def flush(self) -> None:
        """Flush pending Langfuse data."""
        self._langfuse.flush()

    async def shutdown(self) -> None:
        """Shutdown Langfuse client."""
        self._langfuse.shutdown()
```

**Checklist:**
- [ ] Create `src/zap_ai/tracing/langfuse_provider.py`
- [ ] Implement all TracingProvider methods using Langfuse SDK
- [ ] Handle trace/span tracking for context reconstruction
- [ ] Add proper error handling for missing langfuse dependency
- [ ] Support both explicit keys and environment variables

---

## Task 10.5: Add Langfuse Optional Dependency

**File:** `pyproject.toml`

Add langfuse as an optional dependency:

```toml
[project.optional-dependencies]
langfuse = ["langfuse>=2.0.0"]

# Or if there are existing optional deps, add to them:
# dev = [...]
# langfuse = ["langfuse>=2.0.0"]
```

**Checklist:**
- [ ] Add `langfuse` optional dependency group in `pyproject.toml`
- [ ] Verify installation works with `pip install zap-ai[langfuse]`

---

## Task 10.6: Add Trace Context to Activity Inputs

**Files:**
- `src/zap_ai/activities/inference.py`
- `src/zap_ai/activities/tool_execution.py`

Add `trace_context` field to activity input dataclasses:

```python
# In InferenceInput (inference.py)
@dataclass
class InferenceInput:
    agent_name: str
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    # NEW: Optional trace context for observability
    trace_context: dict[str, Any] | None = None


# In ToolExecutionInput (tool_execution.py)
@dataclass
class ToolExecutionInput:
    agent_name: str
    tool_name: str
    arguments: dict[str, Any]
    # NEW: Optional trace context for observability
    trace_context: dict[str, Any] | None = None
```

**Checklist:**
- [ ] Add `trace_context: dict[str, Any] | None = None` to `InferenceInput`
- [ ] Add `trace_context: dict[str, Any] | None = None` to `ToolExecutionInput`
- [ ] Ensure backward compatibility (default None)

---

## Task 10.7: Add Trace Context to Workflow State

**File:** `src/zap_ai/workflows/models.py`

Add trace context fields to workflow models:

```python
# In ConversationState
@dataclass
class ConversationState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    pending_messages: list[str] = field(default_factory=list)
    sub_agent_conversations: dict[str, SubAgentConversation] = field(default_factory=dict)
    # NEW: Trace context for continue-as-new
    trace_context: dict[str, Any] | None = None


# In AgentWorkflowInput
@dataclass
class AgentWorkflowInput:
    agent_name: str
    initial_task: str
    state: dict[str, Any] | None = None
    parent_workflow_id: str | None = None
    # NEW: Parent trace context for sub-agent linking
    parent_trace_context: dict[str, Any] | None = None
```

**Checklist:**
- [ ] Add `trace_context` to `ConversationState`
- [ ] Add `parent_trace_context` to `AgentWorkflowInput`
- [ ] Ensure serialization works for Temporal

---

## Task 10.8: Integrate Tracing into Zap Class

**File:** `src/zap_ai/core/zap.py`

Add tracing provider parameter:

```python
from zap_ai.tracing import TracingProvider, NoOpTracingProvider

@dataclass
class Zap:
    agents: list[ZapAgent]
    temporal_client: Client | None = None
    task_queue: str = "zap-agents"
    # NEW: Optional tracing provider
    tracing_provider: TracingProvider | None = None

    # Internal
    _tracing: TracingProvider = field(
        default_factory=NoOpTracingProvider,
        init=False,
        repr=False
    )

    def __post_init__(self) -> None:
        # ... existing validation ...

        # Initialize tracing
        self._tracing = self.tracing_provider or NoOpTracingProvider()
```

**Checklist:**
- [ ] Add `tracing_provider` parameter to `Zap` class
- [ ] Initialize `_tracing` in `__post_init__`
- [ ] Pass tracing provider to worker creation

---

## Task 10.9: Integrate Tracing into Worker

**File:** `src/zap_ai/worker/worker.py`

Set global tracing provider during worker initialization:

```python
from zap_ai.tracing import TracingProvider, set_tracing_provider

async def create_worker(
    client: Client,
    task_queue: str,
    tool_registry: ToolRegistry,
    tracing_provider: TracingProvider | None = None,  # NEW
) -> Worker:
    set_tool_registry(tool_registry)

    # NEW: Set global tracing provider for activities
    if tracing_provider:
        set_tracing_provider(tracing_provider)

    return Worker(
        client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        activities=[inference_activity, tool_execution_activity],
    )
```

**Checklist:**
- [ ] Add `tracing_provider` parameter to `create_worker`
- [ ] Call `set_tracing_provider` during worker init
- [ ] Update worker documentation

---

## Task 10.10: Integrate Tracing into Inference Activity

**File:** `src/zap_ai/activities/inference.py`

Add generation tracing to inference activity:

```python
from zap_ai.tracing import get_tracing_provider, TraceContext

@activity.defn
async def inference_activity(input: InferenceInput) -> InferenceOutput:
    tracer = get_tracing_provider()
    parent_context = None
    gen_context = None

    # Reconstruct trace context if provided
    if input.trace_context:
        parent_context = TraceContext.from_dict(input.trace_context)

        # Start generation span
        gen_context = await tracer.start_generation(
            name=f"inference-{input.agent_name}",
            parent_context=parent_context,
            model=input.model,
            input_messages=input.messages,
            metadata={"tools_count": len(input.tools)},
        )

    activity.logger.info(
        f"Running inference for agent '{input.agent_name}' "
        f"with model '{input.model}' "
        f"({len(input.messages)} messages, {len(input.tools)} tools)"
    )

    try:
        result = await complete(
            model=input.model,
            messages=input.messages,
            tools=input.tools if input.tools else None,
        )

        activity.logger.info(
            f"Inference complete: "
            f"content={'yes' if result.content else 'no'}, "
            f"tool_calls={len(result.tool_calls)}, "
            f"finish_reason={result.finish_reason}"
        )

        # End generation span with output
        if gen_context:
            await tracer.end_generation(
                context=gen_context,
                output={
                    "content": result.content,
                    "tool_calls": [tc.to_litellm() for tc in result.tool_calls],
                    "finish_reason": result.finish_reason,
                },
                usage=result.usage,  # Assuming InferenceResult has usage
            )

        return InferenceOutput.from_result(result)

    except Exception as e:
        if gen_context:
            await tracer.set_error(gen_context, e)
        raise
```

**Checklist:**
- [ ] Import tracing utilities
- [ ] Reconstruct parent context from input
- [ ] Start generation span before inference
- [ ] End generation span with output and usage
- [ ] Handle errors with `set_error`

---

## Task 10.11: Integrate Tracing into Tool Execution Activity

**File:** `src/zap_ai/activities/tool_execution.py`

Add span tracing to tool execution activity:

```python
from zap_ai.tracing import get_tracing_provider, TraceContext, SpanKind

@activity.defn
async def tool_execution_activity(input: ToolExecutionInput) -> str:
    tracer = get_tracing_provider()
    parent_context = None
    span_context = None

    # Reconstruct trace context if provided
    if input.trace_context:
        parent_context = TraceContext.from_dict(input.trace_context)

    activity.logger.info(
        f"Executing tool '{input.tool_name}' for agent '{input.agent_name}' "
        f"with args: {json.dumps(input.arguments)[:200]}..."
    )

    try:
        # Start tool span if we have context
        if parent_context:
            async with tracer.start_span(
                name=f"tool-{input.tool_name}",
                kind=SpanKind.TOOL_CALL,
                parent_context=parent_context,
                metadata={
                    "tool_name": input.tool_name,
                    "agent_name": input.agent_name,
                },
            ) as span_context:
                result_str = await _execute_tool(input)
                return result_str
        else:
            return await _execute_tool(input)

    except Exception as e:
        activity.logger.error(f"Tool '{input.tool_name}' failed: {e}")
        raise ToolExecutionError(f"Tool execution failed: {e}") from e


async def _execute_tool(input: ToolExecutionInput) -> str:
    """Execute the tool and return result string."""
    registry = get_tool_registry()
    client = registry.get_client_for_tool(input.agent_name, input.tool_name)

    result = await client.call_tool(input.tool_name, input.arguments)

    # Convert result to string
    if isinstance(result, str):
        result_str = result
    elif hasattr(result, 'content'):
        result_str = str(result.content)
    else:
        result_str = json.dumps(result, default=str)

    activity.logger.info(
        f"Tool '{input.tool_name}' completed successfully "
        f"(result length: {len(result_str)})"
    )

    return result_str
```

**Checklist:**
- [ ] Import tracing utilities
- [ ] Reconstruct parent context from input
- [ ] Wrap tool execution in span context manager
- [ ] Add tool name and agent name to span metadata

---

## Task 10.12: Integrate Tracing into AgentWorkflow

**File:** `src/zap_ai/workflows/agent_workflow.py`

Add workflow lifecycle and iteration tracing:

```python
from zap_ai.tracing import get_tracing_provider, TraceContext, SpanKind

@workflow.defn
class AgentWorkflow:

    @workflow.run
    async def run(self, input: AgentWorkflowInput) -> str:
        tracer = get_tracing_provider()
        trace_context = None

        # Initialize or restore trace context
        if input.parent_trace_context:
            # Sub-agent: use parent context
            trace_context = TraceContext.from_dict(input.parent_trace_context)
        elif input.state and input.state.get("trace_context"):
            # Continue-as-new: restore context
            trace_context = TraceContext.from_dict(input.state["trace_context"])

        # Initialize state
        self._initialize_state(input)

        # Store trace context for continue-as-new
        if trace_context:
            self._state.trace_context = trace_context.to_dict()

        # Create root trace if this is a new task (not continue-as-new, not sub-agent)
        if not trace_context:
            async with tracer.start_trace(
                name=f"task-{input.agent_name}-{workflow.info().workflow_id}",
                metadata={
                    "agent_name": input.agent_name,
                    "task": input.initial_task[:200],
                },
            ) as trace_context:
                self._state.trace_context = trace_context.to_dict()
                return await self._run_agentic_loop(trace_context)
        else:
            return await self._run_agentic_loop(trace_context)

    async def _run_agentic_loop(self, trace_context: TraceContext | None) -> str:
        tracer = get_tracing_provider()

        while self._state.iteration_count < self._max_iterations:
            # Check continue-as-new
            if self._should_continue_as_new():
                workflow.continue_as_new(self._create_continue_input())

            # Create iteration span
            if trace_context:
                async with tracer.start_span(
                    name=f"iteration-{self._state.iteration_count}",
                    kind=SpanKind.ITERATION,
                    parent_context=trace_context,
                    metadata={"iteration": self._state.iteration_count},
                ) as iter_context:
                    await self._process_iteration(iter_context)
            else:
                await self._process_iteration(None)

            self._state.iteration_count += 1

        # Max iterations reached
        self._status = TaskStatus.FAILED
        return "Max iterations reached"

    async def _process_iteration(self, trace_context: TraceContext | None) -> None:
        # Process pending messages
        await self._process_pending_messages()

        # Run inference with trace context
        inference_result = await self._run_inference(trace_context)

        # ... rest of iteration logic ...

    async def _run_inference(self, trace_context: TraceContext | None):
        """Execute inference activity with tracing."""
        return await workflow.execute_activity(
            inference_activity,
            InferenceInput(
                agent_name=self._agent_name,
                model=self._model,
                messages=self._state.messages,
                tools=self._tools,
                trace_context=trace_context.to_dict() if trace_context else None,
            ),
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=self._inference_retry_policy,
        )
```

**Checklist:**
- [ ] Import tracing utilities
- [ ] Initialize/restore trace context from input or state
- [ ] Create root trace for new tasks
- [ ] Create iteration spans for each loop iteration
- [ ] Pass trace context to activity calls
- [ ] Store trace context in state for continue-as-new

---

## Task 10.13: Unit Tests for Protocol and Context

**File:** `tests/tracing/test_protocol.py`

```python
"""Tests for tracing protocol and context."""

import pytest
from zap_ai.tracing import TraceContext, SpanKind


class TestTraceContext:
    def test_serialization(self):
        ctx = TraceContext(
            trace_id="trace123",
            span_id="span456",
            provider_data={"key": "value"},
        )

        data = ctx.to_dict()
        assert data["trace_id"] == "trace123"
        assert data["span_id"] == "span456"
        assert data["provider_data"] == {"key": "value"}

    def test_deserialization(self):
        data = {
            "trace_id": "trace123",
            "span_id": "span456",
            "provider_data": {"key": "value"},
        }

        ctx = TraceContext.from_dict(data)
        assert ctx.trace_id == "trace123"
        assert ctx.span_id == "span456"
        assert ctx.provider_data == {"key": "value"}

    def test_deserialization_without_provider_data(self):
        data = {"trace_id": "trace123", "span_id": "span456"}
        ctx = TraceContext.from_dict(data)
        assert ctx.provider_data is None


class TestSpanKind:
    def test_values(self):
        assert SpanKind.WORKFLOW == "workflow"
        assert SpanKind.ITERATION == "iteration"
        assert SpanKind.INFERENCE == "inference"
        assert SpanKind.TOOL_CALL == "tool_call"
        assert SpanKind.SUB_AGENT == "sub_agent"
```

**Checklist:**
- [ ] Create `tests/tracing/test_protocol.py`
- [ ] Test TraceContext serialization/deserialization
- [ ] Test SpanKind enum values

---

## Task 10.14: Unit Tests for NoOp Provider

**File:** `tests/tracing/test_noop_provider.py`

```python
"""Tests for NoOp tracing provider."""

import pytest
from zap_ai.tracing import NoOpTracingProvider, TraceContext, SpanKind


@pytest.mark.asyncio
class TestNoOpTracingProvider:
    async def test_start_trace(self):
        provider = NoOpTracingProvider()

        async with provider.start_trace("test-trace") as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id
            assert ctx.span_id

    async def test_start_span(self):
        provider = NoOpTracingProvider()

        async with provider.start_trace("test") as parent:
            async with provider.start_span(
                "child", SpanKind.ITERATION, parent
            ) as ctx:
                assert ctx.trace_id == parent.trace_id
                assert ctx.span_id != parent.span_id

    async def test_generation_lifecycle(self):
        provider = NoOpTracingProvider()

        async with provider.start_trace("test") as parent:
            gen_ctx = await provider.start_generation(
                "gen", parent, "gpt-4o", []
            )
            assert gen_ctx.trace_id == parent.trace_id

            # Should not raise
            await provider.end_generation(gen_ctx, {"content": "test"})

    async def test_events_and_errors(self):
        provider = NoOpTracingProvider()

        async with provider.start_trace("test") as ctx:
            # Should not raise
            await provider.add_event(ctx, "event", {"key": "value"})
            await provider.set_error(ctx, ValueError("test error"))

    async def test_flush_and_shutdown(self):
        provider = NoOpTracingProvider()

        # Should not raise
        await provider.flush()
        await provider.shutdown()
```

**Checklist:**
- [ ] Create `tests/tracing/test_noop_provider.py`
- [ ] Test all provider methods work without errors
- [ ] Test context propagation through spans

---

## Task 10.15: Unit Tests for Langfuse Provider (Mocked)

**File:** `tests/tracing/test_langfuse_provider.py`

```python
"""Tests for Langfuse tracing provider with mocked Langfuse client."""

import pytest
from unittest.mock import MagicMock, patch

from zap_ai.tracing import SpanKind, TraceContext


@pytest.fixture
def mock_langfuse():
    """Mock the Langfuse client."""
    with patch("zap_ai.tracing.langfuse_provider.Langfuse") as mock:
        instance = MagicMock()
        mock.return_value = instance

        # Mock trace method
        mock_trace = MagicMock()
        instance.trace.return_value = mock_trace

        # Mock span method
        mock_span = MagicMock()
        mock_trace.span.return_value = mock_span

        # Mock generation method
        mock_generation = MagicMock()
        mock_trace.generation.return_value = mock_generation

        yield {
            "langfuse": instance,
            "trace": mock_trace,
            "span": mock_span,
            "generation": mock_generation,
        }


@pytest.mark.asyncio
class TestLangfuseTracingProvider:
    async def test_start_trace(self, mock_langfuse):
        from zap_ai.tracing import LangfuseTracingProvider

        provider = LangfuseTracingProvider(
            public_key="pk-test",
            secret_key="sk-test",
        )

        async with provider.start_trace(
            "test-trace",
            session_id="session123",
            user_id="user456",
        ) as ctx:
            assert ctx.trace_id
            mock_langfuse["langfuse"].trace.assert_called_once()

    async def test_start_span(self, mock_langfuse):
        from zap_ai.tracing import LangfuseTracingProvider

        provider = LangfuseTracingProvider()

        async with provider.start_trace("test") as parent:
            async with provider.start_span(
                "child",
                SpanKind.ITERATION,
                parent,
            ) as ctx:
                assert ctx.trace_id == parent.trace_id
                mock_langfuse["trace"].span.assert_called()

    async def test_generation_lifecycle(self, mock_langfuse):
        from zap_ai.tracing import LangfuseTracingProvider

        provider = LangfuseTracingProvider()

        async with provider.start_trace("test") as parent:
            gen_ctx = await provider.start_generation(
                "inference",
                parent,
                "gpt-4o",
                [{"role": "user", "content": "Hello"}],
            )

            mock_langfuse["trace"].generation.assert_called_once()

            await provider.end_generation(
                gen_ctx,
                {"content": "Hi there!"},
                {"prompt_tokens": 10, "completion_tokens": 5},
            )

            mock_langfuse["generation"].end.assert_called_once()
```

**Checklist:**
- [ ] Create `tests/tracing/test_langfuse_provider.py`
- [ ] Mock Langfuse client to avoid real API calls
- [ ] Test trace, span, and generation creation
- [ ] Test generation end with usage data

---

## Task 10.16: Integration Test with NoOp Provider

**File:** `tests/integration/test_tracing_integration.py`

Test that tracing integrates correctly without affecting core functionality:

```python
"""Integration tests for tracing."""

import pytest
from zap_ai.tracing import (
    NoOpTracingProvider,
    set_tracing_provider,
    get_tracing_provider,
)


@pytest.mark.asyncio
class TestTracingIntegration:
    def test_global_registry_default(self):
        # Reset global state
        import zap_ai.tracing as tracing_module
        tracing_module._tracing_provider = None

        provider = get_tracing_provider()
        assert isinstance(provider, NoOpTracingProvider)

    def test_global_registry_set(self):
        custom_provider = NoOpTracingProvider()
        set_tracing_provider(custom_provider)

        assert get_tracing_provider() is custom_provider

    async def test_activity_inputs_backward_compatible(self):
        """Ensure activities work without trace context."""
        from zap_ai.activities import InferenceInput, ToolExecutionInput

        # Should work without trace_context (backward compat)
        inf_input = InferenceInput(
            agent_name="Test",
            model="gpt-4o",
            messages=[],
            tools=[],
        )
        assert inf_input.trace_context is None

        tool_input = ToolExecutionInput(
            agent_name="Test",
            tool_name="search",
            arguments={},
        )
        assert tool_input.trace_context is None
```

**Checklist:**
- [ ] Create `tests/integration/test_tracing_integration.py`
- [ ] Test global registry default behavior
- [ ] Test activity inputs remain backward compatible
- [ ] Verify tracing doesn't break core functionality

---

## Phase 10 Verification

After completing all tasks, verify:

1. **Protocol and types work:**
   ```python
   from zap_ai.tracing import TracingProvider, TraceContext, SpanKind

   ctx = TraceContext(trace_id="123", span_id="456")
   assert ctx.to_dict()["trace_id"] == "123"
   ```

2. **NoOp provider works:**
   ```python
   from zap_ai.tracing import NoOpTracingProvider

   provider = NoOpTracingProvider()
   async with provider.start_trace("test") as ctx:
       assert ctx.trace_id
   ```

3. **Langfuse provider initializes:**
   ```python
   from zap_ai.tracing import LangfuseTracingProvider

   provider = LangfuseTracingProvider(
       public_key="pk-...",
       secret_key="sk-...",
   )
   ```

4. **Zap accepts tracing provider:**
   ```python
   from zap_ai import Zap, ZapAgent
   from zap_ai.tracing import LangfuseTracingProvider

   zap = Zap(
       agents=[...],
       tracing_provider=LangfuseTracingProvider(...),
   )
   ```

5. **All tests pass:**
   ```bash
   pytest tests/tracing/
   ```
