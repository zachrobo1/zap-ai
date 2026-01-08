"""Conversation history parsing utilities.

This module provides functions for extracting structured information
from conversation history in LiteLLM message format.
"""

from typing import Any

from zap_ai.conversation.models import ConversationTurn, ToolCallInfo
from zap_ai.utils import parse_tool_arguments


def get_text_content(history: list[dict[str, Any]]) -> str:
    """
    Extract all text content from conversation history.

    Returns concatenated text from user and assistant messages,
    excluding tool calls and tool results.

    Args:
        history: List of LiteLLM message dicts.

    Returns:
        Combined text content as a single string, with messages
        separated by double newlines.
    """
    text_parts: list[str] = []
    for msg in history:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if content and isinstance(content, str):
            text_parts.append(content)
    return "\n\n".join(text_parts)


def get_tool_calls(history: list[dict[str, Any]]) -> list[ToolCallInfo]:
    """
    Get all tool calls with their results from conversation history.

    Args:
        history: List of LiteLLM message dicts.

    Returns:
        List of ToolCallInfo objects containing tool name, arguments,
        and results (if available).
    """
    tool_results: dict[str, str] = {}

    # First pass: collect tool results
    for msg in history:
        if msg.get("role") != "tool":
            continue
        tool_call_id = msg.get("tool_call_id")
        content = msg.get("content", "")
        if tool_call_id:
            tool_results[tool_call_id] = content

    # Second pass: collect tool calls and match results
    tool_calls: list[ToolCallInfo] = []
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            tc_id = tc.get("id", "")
            args_raw = func.get("arguments", "{}")
            args = parse_tool_arguments(args_raw)

            tool_calls.append(
                ToolCallInfo(
                    id=tc_id,
                    name=func.get("name", ""),
                    arguments=args,
                    result=tool_results.get(tc_id),
                )
            )

    return tool_calls


def get_turns(history: list[dict[str, Any]]) -> list[ConversationTurn]:
    """
    Get all conversation turns from history.

    A turn is defined as a user message (or system prompt for turn 0),
    followed by all assistant responses and tool interactions until
    the next user message.

    Args:
        history: List of LiteLLM message dicts.

    Returns:
        List of ConversationTurn objects, one per turn.
    """
    turns: list[ConversationTurn] = []
    current_turn = ConversationTurn(turn_number=0)

    for msg in history:
        role = msg.get("role", "")

        if role == "system":
            # System messages go in turn 0
            if current_turn.turn_number == 0 and current_turn.user_message is None:
                current_turn.user_message = msg
            continue

        if role == "user":
            # Start a new turn (save previous if it has content)
            if current_turn.user_message or current_turn.assistant_messages:
                turns.append(current_turn)
                current_turn = ConversationTurn(turn_number=len(turns))
            current_turn.user_message = msg

        elif role == "assistant":
            current_turn.assistant_messages.append(msg)

        elif role == "tool":
            current_turn.tool_messages.append(msg)

    # Don't forget the last turn
    if current_turn.user_message or current_turn.assistant_messages:
        turns.append(current_turn)

    return turns


def get_turn(history: list[dict[str, Any]], turn_num: int) -> ConversationTurn | None:
    """
    Get messages for a specific conversation turn.

    Args:
        history: List of LiteLLM message dicts.
        turn_num: Turn number (0-indexed). Turn 0 may contain system prompt.

    Returns:
        ConversationTurn with the turn's messages, or None if turn doesn't exist.
    """
    turns = get_turns(history)
    if turn_num < 0 or turn_num >= len(turns):
        return None
    return turns[turn_num]


def turn_count(history: list[dict[str, Any]]) -> int:
    """Return the number of conversation turns."""
    return len(get_turns(history))
