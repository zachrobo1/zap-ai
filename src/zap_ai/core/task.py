"""Task models for tracking agent execution state."""

from __future__ import annotations

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
