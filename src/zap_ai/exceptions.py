"""Custom exceptions for the Zap AI platform.

All Zap exceptions inherit from ZapError, providing a unified exception
hierarchy for error handling.
"""


class ZapError(Exception):
    """Base exception for all Zap errors."""

    pass


# =============================================================================
# Core Exceptions
# =============================================================================


class ZapConfigurationError(ZapError):
    """Raised when Zap configuration is invalid."""

    pass


class ZapNotStartedError(ZapError):
    """Raised when operations are attempted before calling start()."""

    pass


class AgentNotFoundError(ZapError):
    """Raised when referencing an agent that doesn't exist."""

    pass


class TaskNotFoundError(ZapError):
    """Raised when referencing a task that doesn't exist."""

    pass


# =============================================================================
# MCP/Tool Exceptions
# =============================================================================


class ToolNotFoundError(ZapError):
    """Raised when a tool cannot be found."""

    pass


class ToolExecutionError(ZapError):
    """Raised when tool execution fails."""

    pass


class ClientConnectionError(ZapError):
    """Raised when MCP client connection fails."""

    pass


class SchemaConversionError(ZapError):
    """Raised when schema conversion fails."""

    pass


# =============================================================================
# LLM Exceptions
# =============================================================================


class LLMProviderError(ZapError):
    """Raised when LLM provider call fails."""

    pass
