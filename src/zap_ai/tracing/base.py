"""Abstract base class for tracing providers.

This module provides BaseTracingProvider, an ABC that users can extend
to implement custom tracing backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from zap_ai.tracing.protocol import ObservationType, TraceContext


class BaseTracingProvider(ABC):
    """
    Abstract base class for tracing providers.

    Subclass this to implement custom tracing backends. You must implement:
    - _start_trace_impl(): Core trace creation logic
    - _start_observation_impl(): Core observation creation logic
    - start_generation(): LLM generation tracking
    - end_generation(): Complete LLM generation

    Optional overrides (default to no-op):
    - add_event(): Add events to observations
    - set_error(): Mark observations as errored
    - flush(): Flush pending data
    - shutdown(): Cleanup resources

    Example:
        class MyTracingProvider(BaseTracingProvider):
            async def _start_trace_impl(self, name, **kwargs):
                ctx = self._create_context()
                return ctx, None  # No cleanup needed

            async def _start_observation_impl(
                self, name, observation_type, parent_context, **kwargs
            ):
                ctx = self._create_child_context(parent_context)
                return ctx, None

            async def start_generation(
                self, name, parent_context, model, input_messages, **kwargs
            ):
                return self._create_child_context(parent_context)

            async def end_generation(self, context, output, usage=None):
                pass  # No-op for simple implementation
    """

    # --- Utility Methods ---

    def _generate_trace_id(self) -> str:
        """Generate a unique trace ID."""
        return uuid4().hex

    def _generate_span_id(self, w3c_format: bool = False) -> str:
        """
        Generate a unique span ID.

        Args:
            w3c_format: If True, return 16 hex chars (W3C trace context format).
                       If False, return full 32 hex chars.
        """
        span_id = uuid4().hex
        return span_id[:16] if w3c_format else span_id

    def _create_context(
        self,
        trace_id: str | None = None,
        span_id: str | None = None,
        provider_data: dict[str, Any] | None = None,
    ) -> TraceContext:
        """
        Create a new TraceContext with generated IDs if not provided.

        Args:
            trace_id: Optional trace ID (generated if None).
            span_id: Optional span ID (generated if None).
            provider_data: Optional provider-specific data.

        Returns:
            New TraceContext instance.
        """
        return TraceContext(
            trace_id=trace_id or self._generate_trace_id(),
            span_id=span_id or self._generate_span_id(),
            provider_data=provider_data,
        )

    def _create_child_context(
        self,
        parent: TraceContext,
        span_id: str | None = None,
        provider_data: dict[str, Any] | None = None,
    ) -> TraceContext:
        """
        Create a child context preserving the parent's trace_id.

        Args:
            parent: Parent trace context.
            span_id: Optional span ID (generated if None).
            provider_data: Optional provider-specific data.

        Returns:
            New TraceContext with same trace_id as parent.
        """
        return TraceContext(
            trace_id=parent.trace_id,
            span_id=span_id or self._generate_span_id(),
            provider_data=provider_data,
        )

    # --- Abstract Methods (Template Pattern) ---

    @abstractmethod
    async def _start_trace_impl(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> tuple[TraceContext, Any | None]:
        """
        Implementation of trace creation.

        Args:
            name: Name of the trace.
            session_id: Optional session for grouping traces.
            user_id: Optional user identifier.
            metadata: Additional metadata to attach.
            tags: Optional tags for filtering.

        Returns:
            Tuple of (TraceContext, cleanup_data).
            cleanup_data is passed to _end_trace_cleanup if provided.
        """
        ...

    async def _end_trace_cleanup(self, context: TraceContext, cleanup_data: Any) -> None:
        """
        Optional cleanup when trace context manager exits.

        Override this to perform cleanup operations when a trace ends.
        Default implementation does nothing.

        Args:
            context: The trace context that is ending.
            cleanup_data: Data returned from _start_trace_impl.
        """
        pass

    @abstractmethod
    async def _start_observation_impl(
        self,
        name: str,
        observation_type: ObservationType,
        parent_context: TraceContext,
        metadata: dict[str, Any] | None = None,
        input_data: Any | None = None,
    ) -> tuple[TraceContext, Any | None]:
        """
        Implementation of observation creation.

        Args:
            name: Name of the observation.
            observation_type: Type of observation for categorization.
            parent_context: Context from parent observation.
            metadata: Additional metadata to attach.
            input_data: Input data for the observation.

        Returns:
            Tuple of (TraceContext, cleanup_data).
            cleanup_data is passed to _end_observation_cleanup if provided.
        """
        ...

    async def _end_observation_cleanup(self, context: TraceContext, cleanup_data: Any) -> None:
        """
        Optional cleanup when observation context manager exits.

        Override this to perform cleanup operations when an observation ends.
        Default implementation does nothing.

        Args:
            context: The observation context that is ending.
            cleanup_data: Data returned from _start_observation_impl.
        """
        pass

    # --- Concrete Context Manager Wrappers ---

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
            name: Name of the trace.
            session_id: Optional session for grouping traces.
            user_id: Optional user identifier.
            metadata: Additional metadata to attach.
            tags: Optional tags for filtering.

        Yields:
            TraceContext for propagation to activities and child workflows.
        """
        context, cleanup_data = await self._start_trace_impl(
            name=name,
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
        )
        try:
            yield context
        finally:
            await self._end_trace_cleanup(context, cleanup_data)

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

        Args:
            name: Name of the observation.
            observation_type: Type of observation for categorization.
            parent_context: Context from parent observation.
            metadata: Additional metadata to attach.
            input_data: Input data for the observation.

        Yields:
            TraceContext for nested observations.
        """
        context, cleanup_data = await self._start_observation_impl(
            name=name,
            observation_type=observation_type,
            parent_context=parent_context,
            metadata=metadata,
            input_data=input_data,
        )
        try:
            yield context
        finally:
            await self._end_observation_cleanup(context, cleanup_data)

    # --- Abstract Methods (Must Implement) ---

    @abstractmethod
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

        Args:
            name: Name of the generation.
            parent_context: Context from parent observation.
            model: LLM model identifier.
            input_messages: Input messages sent to the LLM.
            metadata: Additional metadata.

        Returns:
            TraceContext that must be passed to end_generation().
        """
        ...

    @abstractmethod
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

    # --- Optional Methods (Default No-Op) ---

    async def add_event(
        self,
        context: TraceContext,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """
        Add an event to the current observation.

        Default implementation does nothing. Override to implement.

        Args:
            context: Current trace context.
            name: Event name.
            attributes: Event attributes.
        """
        pass

    async def set_error(
        self,
        context: TraceContext,
        error: Exception,
    ) -> None:
        """
        Mark the observation as errored.

        Default implementation does nothing. Override to implement.

        Args:
            context: Current trace context.
            error: The exception that occurred.
        """
        pass

    async def flush(self) -> None:
        """
        Flush any pending trace data.

        Default implementation does nothing. Override if buffering is used.
        """
        pass

    async def shutdown(self) -> None:
        """
        Cleanup tracing resources.

        Default implementation does nothing. Override if cleanup is needed.
        """
        pass
