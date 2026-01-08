"""Type definitions for Zap context support."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from typing_extensions import TypeVar

# Default context type for untyped Zap instances
DefaultContext = dict[str, Any]

# TypeVar with default for context - allows Zap[MyContext] or just Zap
TContext = TypeVar("TContext", default=DefaultContext)

# Type alias for prompts that can be static or dynamic
DynamicPrompt = str | Callable[[TContext], str]
