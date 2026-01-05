"""Tracing protocol and context for observability.

This module defines the abstract TracingProvider protocol that tracing backends
must implement, along with supporting types for trace context propagation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ObservationType(str, Enum):
    """
    Observation types aligned with Langfuse's native types.

    These types provide semantic context for different kinds of operations
    being traced in the agent workflow.
    """

    SPAN = "span"  # Generic span for workflow steps, iterations
    GENERATION = "generation"  # LLM inference calls
    TOOL = "tool"  # Tool/function calls
    AGENT = "agent"  # Sub-agent delegations


@dataclass
class TraceContext:
    """
    Serializable trace context for propagation across Temporal boundaries.

    This context is passed through activity inputs to maintain trace continuity
    across process boundaries. Must be JSON-serializable for Temporal.

    Attributes:
        trace_id: Unique identifier for the trace.
        span_id: Unique identifier for the current span/observation.
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
    - Creating traces (root spans) for tasks
    - Creating child observations for iterations, tool calls, etc.
    - Creating generation observations for LLM calls (with token usage)
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
        Start a new trace (root observation).

        Called at task start. Returns context for propagation.

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
    async def start_observation(
        self,
        name: str,
        observation_type: ObservationType,
        parent_context: TraceContext,
        metadata: dict[str, Any] | None = None,
        input_data: Any | None = None,
    ) -> AsyncIterator[TraceContext]:
        """
        Start a child observation within an existing trace.

        Used for iterations, tool calls, sub-agent delegation, etc.

        Args:
            name: Name of the observation (e.g., "iteration-0", "tool-search").
            observation_type: Type of observation for categorization.
            parent_context: Context from parent observation.
            metadata: Additional metadata to attach.
            input_data: Input data for the observation.

        Yields:
            TraceContext for nested observations.
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
        Start an LLM generation observation.

        For tracking LLM calls with model info and usage.
        Must be explicitly ended with end_generation().

        Args:
            name: Name of the generation (e.g., "inference-AgentName").
            parent_context: Context from parent observation.
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
        End an LLM generation observation with output and usage.

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
        Add an event to the current observation.

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
        Mark the observation as errored.

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
