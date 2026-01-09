"""No-operation tracing provider.

This provider is used when tracing is not configured. All operations are no-ops
but return valid contexts for code compatibility.
"""

from __future__ import annotations

from typing import Any

from zap_ai.tracing.base import BaseTracingProvider
from zap_ai.tracing.protocol import ObservationType, TraceContext


class NoOpTracingProvider(BaseTracingProvider):
    """
    No-operation tracing provider.

    Used when tracing is not configured. All operations are no-ops
    but return valid contexts for code compatibility.
    """

    async def _start_trace_impl(
        self,
        name: str,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> tuple[TraceContext, None]:
        """Return a dummy context, no cleanup needed."""
        return self._create_context(), None

    async def _start_observation_impl(
        self,
        name: str,
        observation_type: ObservationType,
        parent_context: TraceContext,
        metadata: dict[str, Any] | None = None,
        input_data: Any | None = None,
    ) -> tuple[TraceContext, None]:
        """Return a dummy context with same trace_id, no cleanup needed."""
        return self._create_child_context(parent_context), None

    async def start_generation(
        self,
        name: str,
        parent_context: TraceContext,
        model: str,
        input_messages: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> TraceContext:
        """Return a dummy context."""
        return self._create_child_context(parent_context)

    async def end_generation(
        self,
        context: TraceContext,
        output: dict[str, Any],
        usage: dict[str, int] | None = None,
    ) -> None:
        """No-op."""
        pass
