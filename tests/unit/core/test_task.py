"""Tests for Task and TaskStatus models."""

from datetime import datetime, timezone

from zap_ai import Task, TaskStatus


class TestTaskStatus:
    """Test TaskStatus enum behavior."""

    def test_status_values(self) -> None:
        """Test all status string values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.THINKING.value == "thinking"
        assert TaskStatus.AWAITING_TOOL.value == "awaiting_tool"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"

    def test_is_terminal_completed(self) -> None:
        """Test that COMPLETED is terminal."""
        assert TaskStatus.COMPLETED.is_terminal() is True

    def test_is_terminal_failed(self) -> None:
        """Test that FAILED is terminal."""
        assert TaskStatus.FAILED.is_terminal() is True

    def test_is_terminal_pending(self) -> None:
        """Test that PENDING is not terminal."""
        assert TaskStatus.PENDING.is_terminal() is False

    def test_is_terminal_thinking(self) -> None:
        """Test that THINKING is not terminal."""
        assert TaskStatus.THINKING.is_terminal() is False

    def test_is_terminal_awaiting_tool(self) -> None:
        """Test that AWAITING_TOOL is not terminal."""
        assert TaskStatus.AWAITING_TOOL.is_terminal() is False

    def test_is_active_thinking(self) -> None:
        """Test that THINKING is active."""
        assert TaskStatus.THINKING.is_active() is True

    def test_is_active_awaiting_tool(self) -> None:
        """Test that AWAITING_TOOL is active."""
        assert TaskStatus.AWAITING_TOOL.is_active() is True

    def test_is_active_pending(self) -> None:
        """Test that PENDING is not active."""
        assert TaskStatus.PENDING.is_active() is False

    def test_is_active_completed(self) -> None:
        """Test that COMPLETED is not active."""
        assert TaskStatus.COMPLETED.is_active() is False

    def test_is_active_failed(self) -> None:
        """Test that FAILED is not active."""
        assert TaskStatus.FAILED.is_active() is False


class TestTaskCreation:
    """Test Task instantiation."""

    def test_minimal_task(self) -> None:
        """Test creating a task with only required fields."""
        task = Task(id="Agent-123", agent_name="Agent")
        assert task.id == "Agent-123"
        assert task.agent_name == "Agent"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.history == []
        assert task.sub_tasks == []

    def test_task_with_all_fields(self) -> None:
        """Test creating a task with all fields specified."""
        now = datetime.now(timezone.utc)
        task = Task(
            id="Agent-abc",
            agent_name="Agent",
            status=TaskStatus.COMPLETED,
            result="Done!",
            error=None,
            history=[{"role": "user", "content": "Hello"}],
            sub_tasks=["Child-123"],
            created_at=now,
            updated_at=now,
        )
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Done!"
        assert len(task.history) == 1
        assert len(task.sub_tasks) == 1

    def test_timestamps_auto_created(self) -> None:
        """Test that timestamps are auto-created."""
        before = datetime.now(timezone.utc)
        task = Task(id="Agent-123", agent_name="Agent")
        after = datetime.now(timezone.utc)

        assert before <= task.created_at <= after
        assert before <= task.updated_at <= after


class TestTaskHelperMethods:
    """Test Task helper methods."""

    def test_is_complete_pending(self) -> None:
        """Test is_complete returns False for PENDING."""
        task = Task(id="Agent-123", agent_name="Agent", status=TaskStatus.PENDING)
        assert task.is_complete() is False

    def test_is_complete_thinking(self) -> None:
        """Test is_complete returns False for THINKING."""
        task = Task(id="Agent-123", agent_name="Agent", status=TaskStatus.THINKING)
        assert task.is_complete() is False

    def test_is_complete_completed(self) -> None:
        """Test is_complete returns True for COMPLETED."""
        task = Task(id="Agent-123", agent_name="Agent", status=TaskStatus.COMPLETED)
        assert task.is_complete() is True

    def test_is_complete_failed(self) -> None:
        """Test is_complete returns True for FAILED."""
        task = Task(id="Agent-123", agent_name="Agent", status=TaskStatus.FAILED)
        assert task.is_complete() is True

    def test_is_successful_completed(self) -> None:
        """Test is_successful returns True for COMPLETED."""
        task = Task(id="Agent-123", agent_name="Agent", status=TaskStatus.COMPLETED)
        assert task.is_successful() is True

    def test_is_successful_failed(self) -> None:
        """Test is_successful returns False for FAILED."""
        task = Task(id="Agent-123", agent_name="Agent", status=TaskStatus.FAILED)
        assert task.is_successful() is False

    def test_is_successful_thinking(self) -> None:
        """Test is_successful returns False for non-terminal status."""
        task = Task(id="Agent-123", agent_name="Agent", status=TaskStatus.THINKING)
        assert task.is_successful() is False


class TestTaskMessageMethods:
    """Test Task message-related methods."""

    def test_get_last_message_empty(self) -> None:
        """Test get_last_message returns None for empty history."""
        task = Task(id="Agent-123", agent_name="Agent")
        assert task.get_last_message() is None

    def test_get_last_message_single(self) -> None:
        """Test get_last_message with single message."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[{"role": "user", "content": "Hello"}],
        )
        assert task.get_last_message() == {"role": "user", "content": "Hello"}

    def test_get_last_message_multiple(self) -> None:
        """Test get_last_message returns the last message."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "Bye"},
            ],
        )
        assert task.get_last_message() == {"role": "user", "content": "Bye"}

    def test_get_assistant_messages_empty(self) -> None:
        """Test get_assistant_messages with no assistant messages."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[{"role": "user", "content": "Hello"}],
        )
        assert task.get_assistant_messages() == []

    def test_get_assistant_messages_mixed(self) -> None:
        """Test get_assistant_messages filters correctly."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm fine!"},
            ],
        )
        messages = task.get_assistant_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "Hi!"
        assert messages[1]["content"] == "I'm fine!"


class TestTaskToolCallsCounting:
    """Test tool calls counting method."""

    def test_get_tool_calls_count_empty(self) -> None:
        """Test tool calls count with no tool calls."""
        task = Task(id="Agent-123", agent_name="Agent")
        assert task.get_tool_calls_count() == 0

    def test_get_tool_calls_count_no_tools(self) -> None:
        """Test tool calls count with messages but no tool calls."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        )
        assert task.get_tool_calls_count() == 0

    def test_get_tool_calls_count_single_call(self) -> None:
        """Test tool calls count with single tool call."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "1", "function": {"name": "search"}}],
                },
            ],
        )
        assert task.get_tool_calls_count() == 1

    def test_get_tool_calls_count_multiple_calls_one_message(self) -> None:
        """Test tool calls count with multiple calls in one message."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "function": {"name": "search"}},
                        {"id": "2", "function": {"name": "calculate"}},
                        {"id": "3", "function": {"name": "fetch"}},
                    ],
                },
            ],
        )
        assert task.get_tool_calls_count() == 3

    def test_get_tool_calls_count_across_messages(self) -> None:
        """Test tool calls count across multiple messages."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "1", "function": {"name": "search"}}],
                },
                {"role": "tool", "content": "result", "tool_call_id": "1"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "2", "function": {"name": "calculate"}},
                        {"id": "3", "function": {"name": "fetch"}},
                    ],
                },
            ],
        )
        assert task.get_tool_calls_count() == 3
