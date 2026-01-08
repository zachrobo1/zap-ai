"""Task models for tracking agent execution state."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """
    Status of a task execution.

    The lifecycle of a task typically follows:
    PENDING -> THINKING -> (AWAITING_TOOL <-> THINKING)* -> COMPLETED

    At any point, a task can transition to FAILED if an unrecoverable
    error occurs.

    Attributes:
        PENDING: Task has been created but workflow hasn't started yet.
        THINKING: Agent is thinking (LLM inference in progress).
        AWAITING_TOOL: Waiting for one or more tool executions to complete.
            Includes sub-agent delegation via message_agent tool.
        COMPLETED: Task finished successfully. Result is available.
        FAILED: Task failed with an error. Error details available in
            Task.error field.
    """

    PENDING = "pending"
    THINKING = "thinking"
    AWAITING_TOOL = "awaiting_tool"
    COMPLETED = "completed"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """Return True if this is a terminal (final) status."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    def is_active(self) -> bool:
        """Return True if the task is actively being processed."""
        return self in (TaskStatus.THINKING, TaskStatus.AWAITING_TOOL)


@dataclass
class ToolCallInfo:
    """
    Information about a tool call in the conversation.

    Attributes:
        id: Unique identifier for this tool call.
        name: Name of the tool that was called.
        arguments: Parsed arguments dict passed to the tool.
        result: The tool's result, if available.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    result: str | None = None


@dataclass
class ConversationTurn:
    """
    A single turn in the conversation.

    A turn consists of a user message (or system prompt for turn 0),
    followed by all assistant responses and tool interactions until
    the next user message.

    Attributes:
        turn_number: Zero-indexed turn number.
        user_message: The user (or system) message that started this turn.
        assistant_messages: All assistant responses in this turn.
        tool_messages: All tool result messages in this turn.
    """

    turn_number: int
    user_message: dict[str, Any] | None
    assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Task:
    """
    Represents a task execution within the Zap platform.

    A Task is created when you call `zap.execute_task()` and tracks the
    full lifecycle of that execution. Use `zap.get_task(task_id)` to
    retrieve updated task state.

    Example:
        ```python
        task = await zap.execute_task(agent_name="MyAgent", task="Do something")
        print(f"Task ID: {task.id}")

        # Poll for completion
        while not task.status.is_terminal():
            await asyncio.sleep(1)
            task = await zap.get_task(task.id)

        if task.status == TaskStatus.COMPLETED:
            print(f"Result: {task.result}")
        else:
            print(f"Failed: {task.error}")
        ```

    Attributes:
        id: Unique identifier for this task. Format: "{agent_name}-{uuid}".
            Used as the Temporal workflow ID.
        agent_name: Name of the agent executing this task.
        status: Current execution status. See TaskStatus for details.
        result: Final result string if completed, None otherwise.
        history: List of conversation messages in LiteLLM format.
            Each message is a dict with "role" and "content" keys.
            May include tool calls and tool results.
        sub_tasks: List of child task IDs spawned for sub-agent delegation.
        error: Error message if failed, None otherwise.
        created_at: Timestamp when task was created.
        updated_at: Timestamp of last status update.
    """

    # Required fields (set at creation)
    id: str
    agent_name: str

    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None

    # Conversation history (list of LiteLLM message dicts)
    history: list[dict[str, Any]] = field(default_factory=list)

    # Sub-task tracking (child workflow IDs)
    sub_tasks: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Private: callback to fetch sub-tasks (injected by Zap)
    _task_fetcher: Callable[[str], Awaitable["Task"]] | None = field(
        default=None, repr=False, compare=False
    )

    def is_complete(self) -> bool:
        """Return True if task has reached a terminal state."""
        return self.status.is_terminal()

    def is_successful(self) -> bool:
        """Return True if task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    def get_last_message(self) -> dict[str, Any] | None:
        """Return the most recent message in history, or None if empty."""
        if not self.history:
            return None
        return self.history[-1]

    def get_assistant_messages(self) -> list[dict[str, Any]]:
        """Return all assistant messages from history."""
        return [msg for msg in self.history if msg.get("role") == "assistant"]

    def get_tool_calls_count(self) -> int:
        """Return total number of tool calls made during this task."""
        count = 0
        for msg in self.history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                count += len(msg["tool_calls"])
        return count

    def get_text_content(self) -> str:
        """
        Extract all text content from conversation history.

        Returns concatenated text from user and assistant messages,
        excluding tool calls and tool results.

        Returns:
            Combined text content as a single string, with messages
            separated by double newlines.
        """
        text_parts: list[str] = []
        for msg in self.history:
            role = msg.get("role", "")
            if role not in ("user", "assistant"):
                continue
            content = msg.get("content")
            if content and isinstance(content, str):
                text_parts.append(content)
        return "\n\n".join(text_parts)

    def get_tool_calls(self) -> list[ToolCallInfo]:
        """
        Get all tool calls with their results.

        Returns:
            List of ToolCallInfo objects containing tool name, arguments,
            and results (if available).
        """
        tool_results: dict[str, str] = {}

        # First pass: collect tool results
        for msg in self.history:
            if msg.get("role") != "tool":
                continue
            tool_call_id = msg.get("tool_call_id")
            content = msg.get("content", "")
            if tool_call_id:
                tool_results[tool_call_id] = content

        # Second pass: collect tool calls and match results
        tool_calls: list[ToolCallInfo] = []
        for msg in self.history:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                tc_id = tc.get("id", "")
                args_raw = func.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append(
                    ToolCallInfo(
                        id=tc_id,
                        name=func.get("name", ""),
                        arguments=args,
                        result=tool_results.get(tc_id),
                    )
                )

        return tool_calls

    def get_turns(self) -> list[ConversationTurn]:
        """
        Get all conversation turns.

        A turn is defined as a user message (or system prompt for turn 0),
        followed by all assistant responses and tool interactions until
        the next user message.

        Returns:
            List of ConversationTurn objects, one per turn.
        """
        turns: list[ConversationTurn] = []
        current_turn = ConversationTurn(
            turn_number=0,
            user_message=None,
        )

        for msg in self.history:
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
                    current_turn = ConversationTurn(
                        turn_number=len(turns),
                        user_message=None,
                    )
                current_turn.user_message = msg

            elif role == "assistant":
                current_turn.assistant_messages.append(msg)

            elif role == "tool":
                current_turn.tool_messages.append(msg)

        # Don't forget the last turn
        if current_turn.user_message or current_turn.assistant_messages:
            turns.append(current_turn)

        return turns

    def get_turn(self, turn_num: int) -> ConversationTurn | None:
        """
        Get messages for a specific conversation turn.

        Args:
            turn_num: Turn number (0-indexed). Turn 0 may contain system prompt.

        Returns:
            ConversationTurn with the turn's messages, or None if turn doesn't exist.
        """
        turns = self.get_turns()
        if turn_num < 0 or turn_num >= len(turns):
            return None
        return turns[turn_num]

    def turn_count(self) -> int:
        """Return the number of conversation turns."""
        return len(self.get_turns())

    async def get_sub_tasks(self) -> list["Task"]:
        """
        Fetch full Task objects for all sub-tasks.

        This method requires the Task to have been created via `zap.get_task()`,
        which injects the necessary callback for fetching sub-task data.

        Returns:
            List of Task objects for each sub-task spawned by this task.

        Raises:
            RuntimeError: If Task was not created via Zap.get_task().

        Example:
            ```python
            task = await zap.get_task(task_id)
            sub_tasks = await task.get_sub_tasks()
            for sub in sub_tasks:
                print(f"Sub-task {sub.id}: {sub.status}")
            ```
        """
        if not self._task_fetcher:
            raise RuntimeError(
                "Cannot fetch sub-tasks: Task was not created via Zap.get_task(). "
                "Use zap.get_task(task_id) to get a Task with sub-task access."
            )

        if not self.sub_tasks:
            return []

        # Fetch all sub-tasks concurrently
        tasks = [self._task_fetcher(sub_id) for sub_id in self.sub_tasks]
        return list(await asyncio.gather(*tasks))
