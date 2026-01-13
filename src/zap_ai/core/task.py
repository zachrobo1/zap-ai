"""Task models for tracking agent execution state."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from zap_ai.conversation import ConversationTurn, ToolCallInfo
    from zap_ai.workflows.models import ApprovalRequest


class TaskStatus(str, Enum):
    """
    Status of a task execution.

    The lifecycle of a task typically follows:
    PENDING -> THINKING -> (AWAITING_TOOL <-> THINKING)* -> COMPLETED

    With approval rules enabled, the lifecycle may include:
    THINKING -> AWAITING_APPROVAL -> (approved) -> AWAITING_TOOL

    At any point, a task can transition to FAILED if an unrecoverable
    error occurs.

    Attributes:
        PENDING: Task has been created but workflow hasn't started yet.
        THINKING: Agent is thinking (LLM inference in progress).
        AWAITING_TOOL: Waiting for one or more tool executions to complete.
            Includes sub-agent delegation via message_agent tool.
        AWAITING_APPROVAL: Tool call requires human approval before execution.
            Use Task.get_pending_approvals() to see pending requests, and
            Task.approve() or Task.reject() to respond.
        COMPLETED: Task finished successfully. Result is available.
        FAILED: Task failed with an error. Error details available in
            Task.error field.
    """

    PENDING = "pending"
    THINKING = "thinking"
    AWAITING_TOOL = "awaiting_tool"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """Return True if this is a terminal (final) status."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    def is_active(self) -> bool:
        """Return True if the task is actively being processed."""
        return self in (TaskStatus.THINKING, TaskStatus.AWAITING_TOOL, TaskStatus.AWAITING_APPROVAL)


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

    # Private: callbacks for approval operations (injected by Zap)
    _approval_fetcher: Callable[[], Awaitable[list["ApprovalRequest"]]] | None = field(
        default=None, repr=False, compare=False
    )
    _approval_sender: Callable[[str, bool, str | None], Awaitable[None]] | None = field(
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
        from zap_ai.conversation import get_text_content

        return get_text_content(self.history)

    def get_tool_calls(self) -> list["ToolCallInfo"]:
        """
        Get all tool calls with their results.

        Returns:
            List of ToolCallInfo objects containing tool name, arguments,
            and results (if available).
        """
        from zap_ai.conversation import get_tool_calls

        return get_tool_calls(self.history)

    def get_turns(self) -> list["ConversationTurn"]:
        """
        Get all conversation turns.

        A turn is defined as a user message (or system prompt for turn 0),
        followed by all assistant responses and tool interactions until
        the next user message.

        Returns:
            List of ConversationTurn objects, one per turn.
        """
        from zap_ai.conversation import get_turns

        return get_turns(self.history)

    def get_turn(self, turn_num: int) -> "ConversationTurn | None":
        """
        Get messages for a specific conversation turn.

        Args:
            turn_num: Turn number (0-indexed). Turn 0 may contain system prompt.

        Returns:
            ConversationTurn with the turn's messages, or None if turn doesn't exist.
        """
        from zap_ai.conversation import get_turn

        return get_turn(self.history, turn_num)

    def turn_count(self) -> int:
        """Return the number of conversation turns."""
        from zap_ai.conversation import turn_count

        return turn_count(self.history)

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

    async def get_pending_approvals(self) -> list[dict[str, Any]]:
        """
        Get all pending approval requests for this task.

        This method requires the Task to have been created via `zap.get_task()`,
        which injects the necessary callback for fetching approval data.

        Returns:
            List of dicts with approval request data:
            - id: Unique approval ID
            - tool_name: Name of the tool requiring approval
            - tool_args: Arguments passed to the tool
            - requested_at: ISO timestamp of when approval was requested
            - timeout_at: ISO timestamp of when approval will timeout
            - context: Additional context (agent_name, workflow_id)

        Raises:
            RuntimeError: If Task was not created via Zap.get_task().

        Example:
            ```python
            task = await zap.get_task(task_id)
            pending = await task.get_pending_approvals()
            for req in pending:
                print(f"Tool: {req['tool_name']}, Args: {req['tool_args']}")
                await task.approve(req['id'])
            ```
        """
        if not self._approval_fetcher:
            raise RuntimeError(
                "Cannot fetch approvals: Task was not created via Zap.get_task(). "
                "Use zap.get_task(task_id) to get a Task with approval access."
            )

        requests = await self._approval_fetcher()
        return [r.to_dict() for r in requests]

    async def approve(self, approval_id: str) -> None:
        """
        Approve a pending tool execution.

        Args:
            approval_id: ID of the approval request to approve.

        Raises:
            RuntimeError: If Task was not created via Zap.get_task().

        Example:
            ```python
            task = await zap.get_task(task_id)
            pending = await task.get_pending_approvals()
            await task.approve(pending[0]['id'])
            ```
        """
        if not self._approval_sender:
            raise RuntimeError(
                "Cannot send approval: Task was not created via Zap.get_task(). "
                "Use zap.get_task(task_id) to get a Task with approval access."
            )

        await self._approval_sender(approval_id, True, None)

    async def reject(self, approval_id: str, reason: str | None = None) -> None:
        """
        Reject a pending tool execution.

        Args:
            approval_id: ID of the approval request to reject.
            reason: Optional reason for rejection.

        Raises:
            RuntimeError: If Task was not created via Zap.get_task().

        Example:
            ```python
            task = await zap.get_task(task_id)
            pending = await task.get_pending_approvals()
            await task.reject(pending[0]['id'], reason="Amount exceeds limit")
            ```
        """
        if not self._approval_sender:
            raise RuntimeError(
                "Cannot send rejection: Task was not created via Zap.get_task(). "
                "Use zap.get_task(task_id) to get a Task with approval access."
            )

        await self._approval_sender(approval_id, False, reason)
