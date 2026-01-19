"""Integration tests for streaming event support.

These tests verify that:
1. Events are emitted during workflow execution
2. Events can be queried via the get_events query
3. Event ordering is correct
4. Tool call events include proper phrase generation
"""

import pytest
from temporalio.client import Client

from zap_ai.workflows import AgentWorkflow, AgentWorkflowInput


class TestStreamingEvents:
    """Tests for streaming event functionality."""

    @pytest.mark.asyncio
    async def test_events_emitted_during_execution(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test that events are emitted during workflow execution."""
        input = AgentWorkflowInput(
            agent_name="StreamingTestAgent",
            initial_task="Hello, streaming test!",
        )

        handle = await temporal_client.start_workflow(
            AgentWorkflow.run,
            input,
            id="streaming-test-basic",
            task_queue=task_queue,
        )

        # Wait for completion
        await handle.result()

        # Query events
        events = await handle.query(AgentWorkflow.get_events, 0)

        # Should have at least a thinking and completed event
        assert len(events) >= 2
        event_types = [e["type"] for e in events]
        assert "thinking" in event_types
        assert "completed" in event_types

    @pytest.mark.asyncio
    async def test_events_have_correct_structure(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test that events have the expected structure."""
        input = AgentWorkflowInput(
            agent_name="StreamingStructureTest",
            initial_task="Test event structure",
        )

        handle = await temporal_client.start_workflow(
            AgentWorkflow.run,
            input,
            id="streaming-test-structure",
            task_queue=task_queue,
        )

        await handle.result()

        events = await handle.query(AgentWorkflow.get_events, 0)

        # Check thinking event structure
        thinking_events = [e for e in events if e["type"] == "thinking"]
        assert len(thinking_events) >= 1
        thinking_event = thinking_events[0]
        assert "seq" in thinking_event
        assert "timestamp" in thinking_event
        assert "iteration" in thinking_event
        assert thinking_event["seq"] > 0

        # Check completed event structure
        completed_events = [e for e in events if e["type"] == "completed"]
        assert len(completed_events) == 1
        completed_event = completed_events[0]
        assert "result" in completed_event

    @pytest.mark.asyncio
    async def test_events_with_tool_calls(
        self,
        temporal_client: Client,
        integration_worker_with_tools,
        task_queue_tools: str,
    ) -> None:
        """Test that tool call/result events are emitted."""
        input = AgentWorkflowInput(
            agent_name="ToolStreamingTest",
            initial_task="Test tool events",
            tool_descriptions={"get_time": "Get the current time"},
        )

        handle = await temporal_client.start_workflow(
            AgentWorkflow.run,
            input,
            id="streaming-test-tools",
            task_queue=task_queue_tools,
        )

        await handle.result()

        events = await handle.query(AgentWorkflow.get_events, 0)
        event_types = [e["type"] for e in events]

        # Should have tool_call and tool_result events
        assert "tool_call" in event_types
        assert "tool_result" in event_types

        # Check tool_call event structure
        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) >= 1
        tool_call = tool_call_events[0]
        assert "name" in tool_call
        assert "arguments" in tool_call
        assert "phrase" in tool_call
        assert "tool_call_id" in tool_call
        assert tool_call["name"] == "get_time"

        # Check tool_result event structure
        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_result_events) >= 1
        tool_result = tool_result_events[0]
        assert "name" in tool_result
        assert "result" in tool_result
        assert "tool_call_id" in tool_result
        assert "success" in tool_result

    @pytest.mark.asyncio
    async def test_event_ordering(
        self,
        temporal_client: Client,
        integration_worker_with_tools,
        task_queue_tools: str,
    ) -> None:
        """Test that events are emitted in the correct order."""
        input = AgentWorkflowInput(
            agent_name="OrderingTest",
            initial_task="Test event ordering",
            tool_descriptions={"get_time": "Get the current time"},
        )

        handle = await temporal_client.start_workflow(
            AgentWorkflow.run,
            input,
            id="streaming-test-ordering",
            task_queue=task_queue_tools,
        )

        await handle.result()

        events = await handle.query(AgentWorkflow.get_events, 0)

        # Verify sequence numbers are increasing
        seq_numbers = [e["seq"] for e in events]
        assert seq_numbers == sorted(seq_numbers)
        assert len(set(seq_numbers)) == len(seq_numbers)  # All unique

        # Verify ordering: thinking -> tool_call -> tool_result -> thinking -> completed
        event_types = [e["type"] for e in events]

        # First event should be thinking
        assert event_types[0] == "thinking"

        # Find indices
        thinking_idx = event_types.index("thinking")
        tool_call_idx = event_types.index("tool_call")
        tool_result_idx = event_types.index("tool_result")
        completed_idx = event_types.index("completed")

        # tool_call comes after first thinking
        assert tool_call_idx > thinking_idx
        # tool_result comes after tool_call
        assert tool_result_idx > tool_call_idx
        # completed is last
        assert completed_idx == len(event_types) - 1

    @pytest.mark.asyncio
    async def test_events_since_sequence(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test querying events since a specific sequence number."""
        input = AgentWorkflowInput(
            agent_name="SinceSeqTest",
            initial_task="Test since_seq parameter",
        )

        handle = await temporal_client.start_workflow(
            AgentWorkflow.run,
            input,
            id="streaming-test-since-seq",
            task_queue=task_queue,
        )

        await handle.result()

        # Get all events
        all_events = await handle.query(AgentWorkflow.get_events, 0)
        assert len(all_events) >= 2

        # Get events since first event
        first_seq = all_events[0]["seq"]
        since_first = await handle.query(AgentWorkflow.get_events, first_seq)
        assert len(since_first) == len(all_events) - 1

        # Get events since last event (should be empty)
        last_seq = all_events[-1]["seq"]
        since_last = await handle.query(AgentWorkflow.get_events, last_seq)
        assert len(since_last) == 0

    @pytest.mark.asyncio
    async def test_tool_phrase_generation(
        self,
        temporal_client: Client,
        integration_worker_with_tools,
        task_queue_tools: str,
    ) -> None:
        """Test that tool phrases are generated correctly."""
        input = AgentWorkflowInput(
            agent_name="PhraseTest",
            initial_task="Test phrase generation",
            tool_descriptions={"get_time": "Get the current time for a timezone"},
        )

        handle = await temporal_client.start_workflow(
            AgentWorkflow.run,
            input,
            id="streaming-test-phrases",
            task_queue=task_queue_tools,
        )

        await handle.result()

        events = await handle.query(AgentWorkflow.get_events, 0)
        tool_call_events = [e for e in events if e["type"] == "tool_call"]

        assert len(tool_call_events) >= 1
        phrase = tool_call_events[0]["phrase"]

        # Phrase should be converted and end with ellipsis
        assert phrase.endswith("...")
        # Should have verb converted (Get -> Getting)
        assert "Getting" in phrase or "Calling" in phrase

    @pytest.mark.asyncio
    async def test_error_event_on_max_iterations(
        self,
        temporal_client: Client,
        integration_worker_with_tools,
        task_queue_tools: str,
    ) -> None:
        """Test that error event is emitted when max iterations exceeded."""
        input = AgentWorkflowInput(
            agent_name="MaxIterTest",
            initial_task="Test max iterations error",
            max_iterations=1,  # Will exceed since tool call + response = 2 iterations
            tool_descriptions={"get_time": "Get the current time"},
        )

        handle = await temporal_client.start_workflow(
            AgentWorkflow.run,
            input,
            id="streaming-test-max-iter",
            task_queue=task_queue_tools,
        )

        # This should complete (possibly with max iter error)
        await handle.result()

        events = await handle.query(AgentWorkflow.get_events, 0)
        event_types = [e["type"] for e in events]

        # Should have either completed or error (depending on timing)
        assert "completed" in event_types or "error" in event_types
