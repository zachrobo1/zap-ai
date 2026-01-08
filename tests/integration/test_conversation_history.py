"""Integration tests for conversation history access features.

These tests verify the Task conversation history methods work correctly
when running against a real Temporal server with mock LLM activities.
"""

import asyncio

import pytest
from temporalio.client import Client

from zap_ai import ConversationTurn, ToolCallInfo, Zap, ZapAgent


class TestConversationHistoryAccess:
    """Tests for Task conversation history methods."""

    @pytest.mark.asyncio
    async def test_get_text_content(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test get_text_content() extracts user and assistant text."""
        agent = ZapAgent(name="TextAgent", prompt="You are helpful.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue,
        )
        await zap.start()

        try:
            task = await zap.execute_task(agent_name="TextAgent", task="Hello there")
            while not task.status.is_terminal():
                await asyncio.sleep(0.1)
                task = await zap.get_task(task.id)

            text = task.get_text_content()
            # Should contain user message
            assert "Hello there" in text
            # Should contain assistant response (from mock)
            assert "Integration test response" in text
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_get_tool_calls(
        self,
        temporal_client: Client,
        integration_worker_with_tools,
        task_queue_tools: str,
    ) -> None:
        """Test get_tool_calls() returns ToolCallInfo with results."""
        agent = ZapAgent(name="ToolAgent", prompt="You have tools.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue_tools,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="ToolAgent",
                task="What time is it?",
            )
            while not task.status.is_terminal():
                await asyncio.sleep(0.1)
                task = await zap.get_task(task.id)

            tool_calls = task.get_tool_calls()
            assert len(tool_calls) >= 1
            assert isinstance(tool_calls[0], ToolCallInfo)
            assert tool_calls[0].name == "get_time"
            assert tool_calls[0].id == "call_test_123"
            assert tool_calls[0].arguments == {"timezone": "UTC"}
            assert tool_calls[0].result is not None
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_get_turns(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test get_turns() returns ConversationTurn objects."""
        agent = ZapAgent(name="TurnAgent", prompt="You are helpful.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue,
        )
        await zap.start()

        try:
            task = await zap.execute_task(agent_name="TurnAgent", task="Hi there")
            while not task.status.is_terminal():
                await asyncio.sleep(0.1)
                task = await zap.get_task(task.id)

            turns = task.get_turns()
            assert len(turns) >= 1
            assert isinstance(turns[0], ConversationTurn)

            # turn_count should match
            assert task.turn_count() == len(turns)

            # Turn 0 has system prompt, turn 1 has user message
            # (per implementation: system prompt is turn 0's "user_message")
            assert turns[0].user_message is not None
            # Find the turn with the actual user message
            user_turn = next(
                (t for t in turns if t.user_message and t.user_message.get("role") == "user"),
                None,
            )
            assert user_turn is not None
            assert user_turn.user_message.get("content") == "Hi there"
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_get_turn_specific(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test get_turn() retrieves a specific turn."""
        agent = ZapAgent(name="SpecificTurnAgent", prompt="You are helpful.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="SpecificTurnAgent",
                task="Test message",
            )
            while not task.status.is_terminal():
                await asyncio.sleep(0.1)
                task = await zap.get_task(task.id)

            turn_0 = task.get_turn(0)
            assert turn_0 is not None
            assert turn_0.turn_number == 0

            # Out of bounds returns None
            turn_99 = task.get_turn(99)
            assert turn_99 is None
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_task_fetcher_injected(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test that get_task() injects _task_fetcher for get_sub_tasks()."""
        agent = ZapAgent(name="FetcherAgent", prompt="You are helpful.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue,
        )
        await zap.start()

        try:
            task = await zap.execute_task(agent_name="FetcherAgent", task="Test")
            while not task.status.is_terminal():
                await asyncio.sleep(0.1)
                task = await zap.get_task(task.id)

            # _task_fetcher should be injected
            assert task._task_fetcher is not None

            # No sub-tasks in single-agent case
            sub_tasks = await task.get_sub_tasks()
            assert sub_tasks == []
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_conversation_turns_with_tool_calls(
        self,
        temporal_client: Client,
        integration_worker_with_tools,
        task_queue_tools: str,
    ) -> None:
        """Test that conversation turns correctly include tool interactions."""
        agent = ZapAgent(name="ToolTurnAgent", prompt="You have tools.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue_tools,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="ToolTurnAgent",
                task="Get the time please",
            )
            while not task.status.is_terminal():
                await asyncio.sleep(0.1)
                task = await zap.get_task(task.id)

            turns = task.get_turns()
            assert len(turns) >= 1

            # Turn 0 has system prompt, actual conversation starts at turn 1
            # Find the turn with actual user message
            user_turn = next(
                (t for t in turns if t.user_message and t.user_message.get("role") == "user"),
                None,
            )
            assert user_turn is not None
            # The mock calls a tool, so we should see tool messages or assistant messages
            assert len(user_turn.tool_messages) >= 1 or len(user_turn.assistant_messages) >= 1
        finally:
            await zap.stop()
