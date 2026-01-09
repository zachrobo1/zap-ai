"""
Tracing module for Zap observability.

Provides an abstract TracingProvider protocol with a Langfuse implementation.
Tracing is opt-in via the Zap constructor.

Usage:
    from zap_ai import Zap, ZapAgent
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

from typing import TYPE_CHECKING

from zap_ai.tracing.base import BaseTracingProvider
from zap_ai.tracing.noop_provider import NoOpTracingProvider
from zap_ai.tracing.protocol import (
    ObservationType,
    TraceContext,
    TracingProvider,
)

if TYPE_CHECKING:
    from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider

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


def reset_tracing_provider() -> None:
    """
    Reset the global tracing provider to None.

    Primarily used for testing to ensure clean state between tests.
    """
    global _tracing_provider
    _tracing_provider = None


__all__ = [
    # Protocol and ABC
    "TracingProvider",
    "BaseTracingProvider",
    "TraceContext",
    "ObservationType",
    # Providers
    "NoOpTracingProvider",
    "LangfuseTracingProvider",
    # Global registry
    "set_tracing_provider",
    "get_tracing_provider",
    "reset_tracing_provider",
]


# Lazy import to avoid circular dependencies and allow optional langfuse
def __getattr__(name: str):
    if name == "LangfuseTracingProvider":
        from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider

        return LangfuseTracingProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
