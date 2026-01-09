"""Langfuse tracing provider implementation.

This module provides a Langfuse implementation of the BaseTracingProvider,
using Langfuse's v3 SDK with native observation types.
"""

from __future__ import annotations

from typing import Any

from zap_ai.tracing.base import BaseTracingProvider
from zap_ai.tracing.protocol import ObservationType, TraceContext

try:
    from langfuse import Langfuse
    from langfuse.types import TraceContext as LangfuseTraceContext
except ImportError:
    raise ImportError(
        "langfuse is required for LangfuseTracingProvider. "
        "Install with: pip install zap-ai[langfuse]"
    )


class LangfuseTracingProvider(BaseTracingProvider):
    """
    Langfuse implementation of BaseTracingProvider.

    Uses Langfuse's v3 SDK for async-compatible tracing with
    native observation types (generation, tool, agent, span).

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
        # Track active observations for ending them later
        self._active_observations: dict[str, Any] = {}

    def _make_langfuse_trace_context(
        self, trace_id: str, parent_span_id: str | None = None
    ) -> LangfuseTraceContext:
        """Create a Langfuse TraceContext dict."""
        ctx: LangfuseTraceContext = {"trace_id": trace_id}
        if parent_span_id:
            ctx["parent_span_id"] = parent_span_id
        return ctx

    def _observation_type_to_langfuse(self, obs_type: ObservationType) -> str:
        """Map our observation types to Langfuse's type strings."""
        mapping = {
            ObservationType.SPAN: "span",
            ObservationType.GENERATION: "generation",
            ObservationType.TOOL: "tool",
            ObservationType.AGENT: "agent",
        }
        return mapping.get(obs_type, "span")

    async def _start_trace_impl(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> tuple[TraceContext, Any]:
        """Start a new Langfuse trace.

        In v3, the trace is implicitly created by the first span. We create a
        root span to represent the trace and update it with trace-level metadata.
        """
        trace_id = self._langfuse.create_trace_id()
        span_id = self._generate_span_id(w3c_format=True)

        trace_context = self._make_langfuse_trace_context(trace_id)

        # Create a root span that represents the trace
        root_span = self._langfuse.start_span(
            name=name,
            trace_context=trace_context,
            metadata=metadata,
        )

        # Update the trace with additional metadata
        root_span.update_trace(
            session_id=session_id,
            user_id=user_id,
            tags=tags,
        )

        self._active_observations[trace_id] = root_span

        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            provider_data={
                "langfuse_trace_id": trace_id,
                "langfuse_root_span_id": root_span.id,
            },
        )

        return context, root_span

    async def _end_trace_cleanup(self, context: TraceContext, cleanup_data: Any) -> None:
        """End the root span when trace exits."""
        root_span = cleanup_data
        if root_span:
            root_span.end()
            self._active_observations.pop(context.trace_id, None)

    async def _start_observation_impl(
        self,
        name: str,
        observation_type: ObservationType,
        parent_context: TraceContext,
        metadata: dict[str, Any] | None = None,
        input_data: Any | None = None,
    ) -> tuple[TraceContext, Any]:
        """Start a child observation in Langfuse with the appropriate type."""
        span_id = self._generate_span_id(w3c_format=True)

        # Get parent span ID
        parent_span_id = None
        if parent_context.provider_data:
            parent_span_id = parent_context.provider_data.get(
                "langfuse_observation_id"
            ) or parent_context.provider_data.get("langfuse_root_span_id")

        trace_context = self._make_langfuse_trace_context(parent_context.trace_id, parent_span_id)

        # Get the Langfuse type string and add to metadata
        langfuse_type = self._observation_type_to_langfuse(observation_type)
        obs_metadata = {
            "observation_type": langfuse_type,
            **(metadata or {}),
        }

        # Create span with appropriate type
        span = self._langfuse.start_span(
            name=name,
            trace_context=trace_context,
            input=input_data,
            metadata=obs_metadata,
        )

        self._active_observations[span_id] = span

        context = TraceContext(
            trace_id=parent_context.trace_id,
            span_id=span_id,
            provider_data={
                "langfuse_trace_id": parent_context.trace_id,
                "langfuse_observation_id": span.id,
            },
        )

        return context, span

    async def _end_observation_cleanup(self, context: TraceContext, cleanup_data: Any) -> None:
        """End the span when observation exits."""
        span = cleanup_data
        if span:
            span.end()
            self._active_observations.pop(context.span_id, None)

    async def start_generation(
        self,
        name: str,
        parent_context: TraceContext,
        model: str,
        input_messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        """Start a Langfuse generation for LLM calls."""
        span_id = self._generate_span_id(w3c_format=True)

        # Get parent span ID
        parent_span_id = None
        if parent_context.provider_data:
            parent_span_id = parent_context.provider_data.get(
                "langfuse_observation_id"
            ) or parent_context.provider_data.get("langfuse_root_span_id")

        trace_context = self._make_langfuse_trace_context(parent_context.trace_id, parent_span_id)

        # Use start_observation with as_type='generation' (v3 preferred API)
        generation = self._langfuse.start_observation(
            name=name,
            trace_context=trace_context,
            as_type="generation",
            model=model,
            input=input_messages,
            metadata=metadata,
        )

        self._active_observations[span_id] = generation

        return TraceContext(
            trace_id=parent_context.trace_id,
            span_id=span_id,
            provider_data={
                "langfuse_trace_id": parent_context.trace_id,
                "langfuse_observation_id": generation.id,
                "langfuse_generation_id": generation.id,
            },
        )

    async def end_generation(
        self,
        context: TraceContext,
        output: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> None:
        """End a Langfuse generation with output and usage."""
        generation = self._active_observations.get(context.span_id)
        if not generation:
            return

        # Convert usage to usage_details format expected by v3
        usage_details = None
        if usage:
            usage_details = {
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            }

        generation.update(output=output, usage_details=usage_details)
        generation.end()
        self._active_observations.pop(context.span_id, None)

    async def add_event(
        self,
        context: TraceContext,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        """Add an event to the Langfuse observation."""
        # Get the parent observation or use the root span
        parent_span_id = None
        if context.provider_data:
            parent_span_id = context.provider_data.get(
                "langfuse_observation_id"
            ) or context.provider_data.get("langfuse_root_span_id")

        # Get the observation to add the event to
        obs = None
        for key, observation in self._active_observations.items():
            if hasattr(observation, "id") and observation.id == parent_span_id:
                obs = observation
                break

        if obs:
            obs.create_event(name=name, metadata=attributes)

    async def set_error(
        self,
        context: TraceContext,
        error: Exception,
    ) -> None:
        """Mark the Langfuse observation as errored."""
        observation = self._active_observations.get(context.span_id)
        if observation:
            observation.update(
                level="ERROR",
                status_message=str(error),
            )

    async def flush(self) -> None:
        """Flush pending Langfuse data."""
        self._langfuse.flush()

    async def shutdown(self) -> None:
        """Shutdown Langfuse client."""
        self._langfuse.shutdown()
