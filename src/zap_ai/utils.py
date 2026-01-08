"""Shared utility functions for Zap AI."""

import json
from typing import Any


def parse_tool_arguments(arguments: str | dict[str, Any]) -> dict[str, Any]:
    """Parse tool call arguments from string or dict format.

    LLM providers may return tool arguments as either a JSON string or
    an already-parsed dict. This function normalizes both formats.

    Args:
        arguments: Either a JSON string or already-parsed dict.

    Returns:
        Parsed arguments as a dictionary. Returns empty dict if
        arguments is empty or parsing fails.
    """
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}
