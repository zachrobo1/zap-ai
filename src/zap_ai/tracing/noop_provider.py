"""No-operation tracing provider.

This provider is used when tracing is not configured. All operations are no-ops
but return valid contexts for code compatibility.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from zap_ai.tracing.protocol import ObservationType, TraceContext


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
    async def start_observation(
        self,
        name: str,
        observation_type: ObservationType,
        parent_context: TraceContext,
        metadata: dict[str, Any] | None = None,
        input_data: Any | None = None,
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
