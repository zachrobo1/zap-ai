"""Tests for Task and TaskStatus models."""

from datetime import datetime, timezone

import pytest

from zap_ai import ConversationTurn, Task, TaskStatus, ToolCallInfo


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


class TestToolCallInfo:
    """Test ToolCallInfo dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating ToolCallInfo with minimal fields."""
        info = ToolCallInfo(id="call-1", name="search", arguments={"query": "test"})
        assert info.id == "call-1"
        assert info.name == "search"
        assert info.arguments == {"query": "test"}
        assert info.result is None

    def test_create_with_result(self) -> None:
        """Test creating ToolCallInfo with result."""
        info = ToolCallInfo(
            id="call-1", name="search", arguments={"query": "test"}, result="Found it!"
        )
        assert info.result == "Found it!"


class TestConversationTurn:
    """Test ConversationTurn dataclass."""

    def test_create_minimal(self) -> None:
        """Test creating ConversationTurn with minimal fields."""
        turn = ConversationTurn(turn_number=0, user_message=None)
        assert turn.turn_number == 0
        assert turn.user_message is None
        assert turn.assistant_messages == []
        assert turn.tool_messages == []

    def test_create_full(self) -> None:
        """Test creating ConversationTurn with all fields."""
        turn = ConversationTurn(
            turn_number=1,
            user_message={"role": "user", "content": "Hello"},
            assistant_messages=[{"role": "assistant", "content": "Hi!"}],
            tool_messages=[{"role": "tool", "content": "result", "tool_call_id": "1"}],
        )
        assert turn.turn_number == 1
        assert turn.user_message == {"role": "user", "content": "Hello"}
        assert len(turn.assistant_messages) == 1
        assert len(turn.tool_messages) == 1


class TestTaskTextContent:
    """Test Task.get_text_content() method."""

    def test_get_text_content_empty(self) -> None:
        """Test get_text_content with empty history."""
        task = Task(id="Agent-123", agent_name="Agent")
        assert task.get_text_content() == ""

    def test_get_text_content_user_only(self) -> None:
        """Test get_text_content with only user messages."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "user", "content": "World"},
            ],
        )
        assert task.get_text_content() == "Hello\n\nWorld"

    def test_get_text_content_mixed(self) -> None:
        """Test get_text_content with user and assistant messages."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ],
        )
        assert task.get_text_content() == "Hello\n\nHi there!\n\nHow are you?\n\nI'm doing well!"

    def test_get_text_content_excludes_tools(self) -> None:
        """Test get_text_content excludes tool messages."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Search for something"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "1", "function": {"name": "search"}}],
                },
                {"role": "tool", "content": "Search result", "tool_call_id": "1"},
                {"role": "assistant", "content": "Found it!"},
            ],
        )
        # Only user and assistant text content, no tools
        assert task.get_text_content() == "Search for something\n\nFound it!"


class TestTaskGetToolCalls:
    """Test Task.get_tool_calls() method."""

    def test_get_tool_calls_empty(self) -> None:
        """Test get_tool_calls with no tool calls."""
        task = Task(id="Agent-123", agent_name="Agent")
        assert task.get_tool_calls() == []

    def test_get_tool_calls_single(self) -> None:
        """Test get_tool_calls with single tool call."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {"name": "search", "arguments": '{"query": "test"}'},
                        }
                    ],
                },
                {"role": "tool", "content": "Found it!", "tool_call_id": "call-1"},
            ],
        )
        calls = task.get_tool_calls()
        assert len(calls) == 1
        assert calls[0].id == "call-1"
        assert calls[0].name == "search"
        assert calls[0].arguments == {"query": "test"}
        assert calls[0].result == "Found it!"

    def test_get_tool_calls_multiple(self) -> None:
        """Test get_tool_calls with multiple calls."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "function": {"name": "search", "arguments": "{}"}},
                        {"id": "2", "function": {"name": "fetch", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "content": "result1", "tool_call_id": "1"},
                {"role": "tool", "content": "result2", "tool_call_id": "2"},
            ],
        )
        calls = task.get_tool_calls()
        assert len(calls) == 2
        assert calls[0].name == "search"
        assert calls[0].result == "result1"
        assert calls[1].name == "fetch"
        assert calls[1].result == "result2"

    def test_get_tool_calls_without_results(self) -> None:
        """Test get_tool_calls when tool result is missing."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "1", "function": {"name": "search", "arguments": "{}"}},
                    ],
                },
                # No tool result message
            ],
        )
        calls = task.get_tool_calls()
        assert len(calls) == 1
        assert calls[0].result is None


class TestTaskTurns:
    """Test Task turn-related methods."""

    def test_get_turns_empty(self) -> None:
        """Test get_turns with empty history."""
        task = Task(id="Agent-123", agent_name="Agent")
        assert task.get_turns() == []

    def test_get_turns_single_turn(self) -> None:
        """Test get_turns with single turn."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        )
        turns = task.get_turns()
        assert len(turns) == 1
        assert turns[0].turn_number == 0
        assert turns[0].user_message == {"role": "user", "content": "Hello"}
        assert len(turns[0].assistant_messages) == 1

    def test_get_turns_multiple_turns(self) -> None:
        """Test get_turns with multiple turns."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "Fine!"},
            ],
        )
        turns = task.get_turns()
        assert len(turns) == 2
        assert turns[0].turn_number == 0
        assert turns[1].turn_number == 1

    def test_get_turns_with_tools(self) -> None:
        """Test get_turns includes tool messages."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "Search"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "1", "function": {"name": "search"}}],
                },
                {"role": "tool", "content": "result", "tool_call_id": "1"},
                {"role": "assistant", "content": "Found it!"},
            ],
        )
        turns = task.get_turns()
        assert len(turns) == 1
        assert len(turns[0].assistant_messages) == 2
        assert len(turns[0].tool_messages) == 1

    def test_get_turns_with_system_prompt(self) -> None:
        """Test get_turns includes system prompt in turn 0."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
        )
        turns = task.get_turns()
        # System prompt should be the user_message for turn 0 if it comes first
        assert len(turns) == 2
        assert turns[0].user_message == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert turns[1].user_message == {"role": "user", "content": "Hello"}

    def test_get_turn_valid(self) -> None:
        """Test get_turn returns correct turn."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "Second"},
                {"role": "assistant", "content": "Response 2"},
            ],
        )
        turn = task.get_turn(1)
        assert turn is not None
        assert turn.turn_number == 1
        assert turn.user_message == {"role": "user", "content": "Second"}

    def test_get_turn_invalid(self) -> None:
        """Test get_turn returns None for invalid turn number."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[{"role": "user", "content": "Hello"}],
        )
        assert task.get_turn(5) is None
        assert task.get_turn(-1) is None

    def test_turn_count(self) -> None:
        """Test turn_count returns correct count."""
        task = Task(
            id="Agent-123",
            agent_name="Agent",
            history=[
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "Second"},
                {"role": "assistant", "content": "Response 2"},
            ],
        )
        assert task.turn_count() == 2


class TestTaskSubTasks:
    """Test Task.get_sub_tasks() method."""

    @pytest.mark.asyncio
    async def test_get_sub_tasks_no_fetcher_raises(self) -> None:
        """Test get_sub_tasks raises error without fetcher."""
        task = Task(id="Agent-123", agent_name="Agent", sub_tasks=["Child-1"])
        with pytest.raises(RuntimeError, match="not created via Zap.get_task"):
            await task.get_sub_tasks()

    @pytest.mark.asyncio
    async def test_get_sub_tasks_empty(self) -> None:
        """Test get_sub_tasks with no sub-tasks."""

        async def mock_fetcher(task_id: str) -> Task:
            return Task(id=task_id, agent_name="Child")

        task = Task(id="Agent-123", agent_name="Agent", sub_tasks=[], _task_fetcher=mock_fetcher)
        result = await task.get_sub_tasks()
        assert result == []

    @pytest.mark.asyncio
    async def test_get_sub_tasks_fetches_tasks(self) -> None:
        """Test get_sub_tasks fetches Task objects."""
        fetched_ids: list[str] = []

        async def mock_fetcher(task_id: str) -> Task:
            fetched_ids.append(task_id)
            return Task(id=task_id, agent_name="Child")

        task = Task(
            id="Agent-123",
            agent_name="Agent",
            sub_tasks=["Child-1", "Child-2"],
            _task_fetcher=mock_fetcher,
        )
        result = await task.get_sub_tasks()

        assert len(result) == 2
        assert result[0].id == "Child-1"
        assert result[1].id == "Child-2"
        assert fetched_ids == ["Child-1", "Child-2"]


class TestTaskApprovalMethods:
    """Tests for Task approval-related methods."""

    @pytest.mark.asyncio
    async def test_get_pending_approvals_no_fetcher_raises(self) -> None:
        """Test get_pending_approvals raises error without fetcher."""
        task = Task(id="Agent-123", agent_name="Agent")
        with pytest.raises(RuntimeError, match="not created via Zap.get_task"):
            await task.get_pending_approvals()

    @pytest.mark.asyncio
    async def test_get_pending_approvals_with_fetcher(self) -> None:
        """Test get_pending_approvals returns formatted data."""
        from datetime import timedelta, timezone

        from zap_ai.workflows.models import ApprovalRequest

        now = datetime.now(timezone.utc)
        request = ApprovalRequest(
            id="approval-123",
            tool_name="transfer_funds",
            tool_args={"amount": 1000},
            requested_at=now,
            timeout_at=now + timedelta(days=7),
            context={"agent_name": "TestAgent"},
        )

        async def mock_fetcher() -> list[ApprovalRequest]:
            return [request]

        task = Task(id="Agent-123", agent_name="Agent", _approval_fetcher=mock_fetcher)
        result = await task.get_pending_approvals()

        assert len(result) == 1
        assert result[0]["id"] == "approval-123"
        assert result[0]["tool_name"] == "transfer_funds"
        assert result[0]["tool_args"] == {"amount": 1000}
        assert result[0]["context"] == {"agent_name": "TestAgent"}

    @pytest.mark.asyncio
    async def test_get_pending_approvals_empty(self) -> None:
        """Test get_pending_approvals returns empty list when no pending."""

        async def mock_fetcher() -> list:
            return []

        task = Task(id="Agent-123", agent_name="Agent", _approval_fetcher=mock_fetcher)
        result = await task.get_pending_approvals()
        assert result == []

    @pytest.mark.asyncio
    async def test_approve_no_sender_raises(self) -> None:
        """Test approve raises error without sender."""
        task = Task(id="Agent-123", agent_name="Agent")
        with pytest.raises(RuntimeError, match="not created via Zap.get_task"):
            await task.approve("some-id")

    @pytest.mark.asyncio
    async def test_approve_calls_sender(self) -> None:
        """Test approve calls sender with (id, True, None)."""
        calls: list[tuple] = []

        async def mock_sender(approval_id: str, approved: bool, reason: str | None) -> None:
            calls.append((approval_id, approved, reason))

        task = Task(id="Agent-123", agent_name="Agent", _approval_sender=mock_sender)
        await task.approve("approval-456")

        assert len(calls) == 1
        assert calls[0] == ("approval-456", True, None)

    @pytest.mark.asyncio
    async def test_reject_no_sender_raises(self) -> None:
        """Test reject raises error without sender."""
        task = Task(id="Agent-123", agent_name="Agent")
        with pytest.raises(RuntimeError, match="not created via Zap.get_task"):
            await task.reject("some-id")

    @pytest.mark.asyncio
    async def test_reject_calls_sender_with_reason(self) -> None:
        """Test reject calls sender with (id, False, reason)."""
        calls: list[tuple] = []

        async def mock_sender(approval_id: str, approved: bool, reason: str | None) -> None:
            calls.append((approval_id, approved, reason))

        task = Task(id="Agent-123", agent_name="Agent", _approval_sender=mock_sender)
        await task.reject("approval-789", reason="Amount too high")

        assert len(calls) == 1
        assert calls[0] == ("approval-789", False, "Amount too high")

    @pytest.mark.asyncio
    async def test_reject_without_reason(self) -> None:
        """Test reject calls sender with None reason when not provided."""
        calls: list[tuple] = []

        async def mock_sender(approval_id: str, approved: bool, reason: str | None) -> None:
            calls.append((approval_id, approved, reason))

        task = Task(id="Agent-123", agent_name="Agent", _approval_sender=mock_sender)
        await task.reject("approval-000")

        assert len(calls) == 1
        assert calls[0] == ("approval-000", False, None)
