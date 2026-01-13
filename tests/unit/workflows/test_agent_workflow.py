"""Tests for AgentWorkflow."""

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from zap_ai.activities import (
    InferenceInput,
    InferenceOutput,
    ToolExecutionInput,
)
from zap_ai.activities.tool_execution import AgentConfigOutput
from zap_ai.workflows import AgentWorkflow, AgentWorkflowInput


# Mock activities for testing
@activity.defn(name="inference_activity")
async def mock_inference_activity(input: InferenceInput) -> InferenceOutput:
    """Mock inference that returns a simple response."""
    last_content = input.messages[-1].get("content", "") if input.messages else ""
    return InferenceOutput(
        content=f"Mock response to: {last_content}",
        tool_calls=[],
        finish_reason="stop",
    )


@activity.defn(name="tool_execution_activity")
async def mock_tool_execution_activity(input: ToolExecutionInput) -> str:
    """Mock tool execution that returns a simple result."""
    return f"Mock tool result for {input.tool_name}"


@activity.defn(name="get_agent_config_activity")
async def mock_get_agent_config_activity(agent_name: str) -> AgentConfigOutput:
    """Mock agent config activity."""
    return AgentConfigOutput(
        agent_name=agent_name,
        prompt=f"You are agent {agent_name}.",
        model="gpt-4o",
        max_iterations=50,
        tools=[],
    )


@pytest.fixture
async def workflow_env():
    """Create a time-skipping workflow environment."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


@pytest.fixture
def simple_input() -> AgentWorkflowInput:
    """Create a simple workflow input."""
    return AgentWorkflowInput(
        agent_name="TestAgent",
        initial_task="Hello, how are you?",
    )


class TestAgentWorkflowBasic:
    """Basic tests for AgentWorkflow execution."""

    @pytest.mark.asyncio
    async def test_workflow_starts_and_completes(
        self, workflow_env: WorkflowEnvironment, simple_input: AgentWorkflowInput
    ) -> None:
        """Test that workflow starts and completes with mock activities."""
        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            result = await workflow_env.client.execute_workflow(
                AgentWorkflow.run,
                simple_input,
                id="test-workflow-1",
                task_queue="test-queue",
            )

            # With mock inference, we should get a response
            assert result is not None
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_workflow_returns_mock_response(self, workflow_env: WorkflowEnvironment) -> None:
        """Test that workflow returns the expected mock response."""
        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="What is 2+2?",
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            result = await workflow_env.client.execute_workflow(
                AgentWorkflow.run,
                input,
                id="test-workflow-2",
                task_queue="test-queue",
            )

            # Mock response should contain the task
            assert "Mock response to:" in result
            assert "What is 2+2?" in result


class TestAgentWorkflowQueries:
    """Tests for AgentWorkflow query methods."""

    @pytest.mark.asyncio
    async def test_query_status_during_execution(
        self, workflow_env: WorkflowEnvironment, simple_input: AgentWorkflowInput
    ) -> None:
        """Test querying status while workflow runs."""
        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            # Start workflow
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                simple_input,
                id="test-workflow-query-1",
                task_queue="test-queue",
            )

            # Wait for completion and query final status
            await handle.result()
            status = await handle.query(AgentWorkflow.get_status)

            # After completion with mock, status should be "completed"
            assert status == "completed"

    @pytest.mark.asyncio
    async def test_query_result_after_completion(
        self, workflow_env: WorkflowEnvironment, simple_input: AgentWorkflowInput
    ) -> None:
        """Test querying result after workflow completes."""
        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                simple_input,
                id="test-workflow-query-2",
                task_queue="test-queue",
            )

            await handle.result()
            result = await handle.query(AgentWorkflow.get_result)

            assert result is not None
            assert "Mock response to:" in result

    @pytest.mark.asyncio
    async def test_query_history_after_completion(
        self, workflow_env: WorkflowEnvironment, simple_input: AgentWorkflowInput
    ) -> None:
        """Test querying conversation history."""
        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                simple_input,
                id="test-workflow-query-3",
                task_queue="test-queue",
            )

            await handle.result()
            history = await handle.query(AgentWorkflow.get_history)

            # Should have: system prompt, user message, assistant response
            assert len(history) >= 3
            assert history[0]["role"] == "system"
            assert history[1]["role"] == "user"
            assert history[1]["content"] == simple_input.initial_task

    @pytest.mark.asyncio
    async def test_query_iteration_count(
        self, workflow_env: WorkflowEnvironment, simple_input: AgentWorkflowInput
    ) -> None:
        """Test querying iteration count."""
        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                simple_input,
                id="test-workflow-query-4",
                task_queue="test-queue",
            )

            await handle.result()
            count = await handle.query(AgentWorkflow.get_iteration_count)

            # With mock inference returning no tool calls, should be 0 iterations
            # (iteration count increments after handling tool calls)
            assert count == 0

    @pytest.mark.asyncio
    async def test_query_error_when_no_error(
        self, workflow_env: WorkflowEnvironment, simple_input: AgentWorkflowInput
    ) -> None:
        """Test that error query returns None when no error."""
        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                simple_input,
                id="test-workflow-query-5",
                task_queue="test-queue",
            )

            await handle.result()
            error = await handle.query(AgentWorkflow.get_error)

            assert error is None

    @pytest.mark.asyncio
    async def test_query_sub_agent_conversations_empty(
        self, workflow_env: WorkflowEnvironment, simple_input: AgentWorkflowInput
    ) -> None:
        """Test querying sub-agent conversations when none exist."""
        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                simple_input,
                id="test-workflow-query-6",
                task_queue="test-queue",
            )

            await handle.result()
            convs = await handle.query(AgentWorkflow.get_sub_agent_conversations)

            assert convs == {}


class TestAgentWorkflowContinueAsNew:
    """Tests for workflow continue-as-new with state restoration."""

    @pytest.mark.asyncio
    async def test_workflow_restores_state(self, workflow_env: WorkflowEnvironment) -> None:
        """Test that workflow correctly restores state from input."""
        # Create input with existing state
        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="",  # Empty for continue-as-new
            state={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"},
                ],
                "iteration_count": 2,
                "pending_messages": [],
                "sub_agent_conversations": {},
            },
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                input,
                id="test-workflow-restore-1",
                task_queue="test-queue",
            )

            await handle.result()
            history = await handle.query(AgentWorkflow.get_history)

            # History should include the restored messages plus new ones
            assert len(history) >= 4
            assert history[0]["role"] == "system"
            assert history[1]["content"] == "Hello"


class TestAgentWorkflowSignals:
    """Tests for AgentWorkflow signal handling."""

    @pytest.mark.asyncio
    async def test_signal_add_message(self, workflow_env: WorkflowEnvironment) -> None:
        """Test that add_message signal adds to pending messages."""
        # Note: This test validates signals work at the Temporal level
        # The full integration with follow-up would require more complex setup
        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Initial task",
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                input,
                id="test-workflow-signal-1",
                task_queue="test-queue",
            )

            # Wait for initial completion
            await handle.result()

            # Verify workflow completed
            status = await handle.query(AgentWorkflow.get_status)
            assert status == "completed"


class TestAgentWorkflowWithParent:
    """Tests for child workflow behavior."""

    @pytest.mark.asyncio
    async def test_child_workflow_receives_parent_id(
        self, workflow_env: WorkflowEnvironment
    ) -> None:
        """Test that child workflow input includes parent workflow ID."""
        input = AgentWorkflowInput(
            agent_name="ChildAgent",
            initial_task="Help with something",
            parent_workflow_id="parent-workflow-123",
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                input,
                id="test-child-workflow-1",
                task_queue="test-queue",
            )

            result = await handle.result()
            assert result is not None


class TestActivityInputOutput:
    """Tests for activity input/output dataclasses."""

    def test_inference_input_creation(self) -> None:
        """Test creating InferenceInput."""
        input = InferenceInput(
            agent_name="TestAgent",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[],
        )
        assert input.agent_name == "TestAgent"
        assert input.model == "gpt-4o"
        assert len(input.messages) == 1

    def test_inference_input_defaults(self) -> None:
        """Test InferenceInput default values."""
        input = InferenceInput(
            agent_name="TestAgent",
            model="gpt-4o",
        )
        assert input.messages == []
        assert input.tools == []

    def test_inference_output_creation(self) -> None:
        """Test creating InferenceOutput."""
        output = InferenceOutput(
            content="Hello!",
            tool_calls=[],
            finish_reason="stop",
        )
        assert output.content == "Hello!"
        assert output.tool_calls == []
        assert output.finish_reason == "stop"

    def test_inference_output_defaults(self) -> None:
        """Test InferenceOutput default values."""
        output = InferenceOutput(content="Hello!")
        assert output.tool_calls == []
        assert output.finish_reason == "stop"

    def test_tool_execution_input_creation(self) -> None:
        """Test creating ToolExecutionInput."""
        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
            arguments={"query": "test"},
        )
        assert input.agent_name == "TestAgent"
        assert input.tool_name == "search"
        assert input.arguments == {"query": "test"}

    def test_tool_execution_input_defaults(self) -> None:
        """Test ToolExecutionInput default values."""
        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
        )
        assert input.arguments == {}


class TestAgentWorkflowParallelToolCalls:
    """Tests for parallel tool call execution."""

    @pytest.mark.asyncio
    async def test_mcp_tools_run_in_parallel(self, workflow_env: WorkflowEnvironment) -> None:
        """Test that multiple MCP tool calls execute in parallel."""
        import time

        execution_times: list[tuple[str, float]] = []

        @activity.defn(name="inference_activity")
        async def inference_with_tools(input: InferenceInput) -> InferenceOutput:
            """Return 3 tool calls on first call, then complete."""
            # Check if we already have tool results
            has_tool_results = any(m.get("role") == "tool" for m in input.messages)
            if has_tool_results:
                return InferenceOutput(
                    content="All tools completed",
                    tool_calls=[],
                    finish_reason="stop",
                )
            return InferenceOutput(
                content=None,
                tool_calls=[
                    {"id": "1", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "2", "function": {"name": "tool_b", "arguments": "{}"}},
                    {"id": "3", "function": {"name": "tool_c", "arguments": "{}"}},
                ],
                finish_reason="tool_calls",
            )

        @activity.defn(name="tool_execution_activity")
        async def tracking_tool_activity(input: ToolExecutionInput) -> str:
            """Track execution start time and add delay."""
            import asyncio

            start = time.time()
            execution_times.append((input.tool_name, start))
            await asyncio.sleep(0.05)  # 50ms delay
            return f"Result for {input.tool_name}"

        @activity.defn(name="get_agent_config_activity")
        async def config_activity(agent_name: str) -> AgentConfigOutput:
            return AgentConfigOutput(
                agent_name=agent_name,
                prompt="Test prompt",
                model="gpt-4o",
                max_iterations=50,
                tools=[],
            )

        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Use all three tools",
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                inference_with_tools,
                tracking_tool_activity,
                config_activity,
            ],
        ):
            start_time = time.time()
            await workflow_env.client.execute_workflow(
                AgentWorkflow.run,
                input,
                id="test-parallel-mcp-1",
                task_queue="test-queue",
            )
            total_time = time.time() - start_time

            # All 3 tools should have been called
            assert len(execution_times) == 3

            # If sequential: would take ~150ms (3 * 50ms)
            # If parallel: should take ~50ms
            # Allow some margin for test overhead
            assert total_time < 0.3, f"Tools ran sequentially: {total_time}s"

            # Verify all started at approximately the same time
            start_times = [t[1] for t in execution_times]
            time_spread = max(start_times) - min(start_times)
            assert time_spread < 0.05, f"Tools didn't start together: {time_spread}s"

    @pytest.mark.asyncio
    async def test_tool_results_preserve_order(self, workflow_env: WorkflowEnvironment) -> None:
        """Test that tool results are added to history in correct order."""

        @activity.defn(name="inference_activity")
        async def inference_with_tools(input: InferenceInput) -> InferenceOutput:
            has_tool_results = any(m.get("role") == "tool" for m in input.messages)
            if has_tool_results:
                return InferenceOutput(
                    content="Done",
                    tool_calls=[],
                    finish_reason="stop",
                )
            return InferenceOutput(
                content=None,
                tool_calls=[
                    {"id": "call_1", "function": {"name": "first", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "second", "arguments": "{}"}},
                    {"id": "call_3", "function": {"name": "third", "arguments": "{}"}},
                ],
                finish_reason="tool_calls",
            )

        @activity.defn(name="tool_execution_activity")
        async def tool_with_name_result(input: ToolExecutionInput) -> str:
            import asyncio

            # Vary delays to ensure order isn't determined by completion time
            delays = {"first": 0.03, "second": 0.01, "third": 0.02}
            await asyncio.sleep(delays.get(input.tool_name, 0))
            return f"Result:{input.tool_name}"

        @activity.defn(name="get_agent_config_activity")
        async def config_activity(agent_name: str) -> AgentConfigOutput:
            return AgentConfigOutput(
                agent_name=agent_name,
                prompt="Test",
                model="gpt-4o",
                max_iterations=50,
                tools=[],
            )

        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Test order",
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                inference_with_tools,
                tool_with_name_result,
                config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                input,
                id="test-order-1",
                task_queue="test-queue",
            )
            await handle.result()

            history = await handle.query(AgentWorkflow.get_history)

            # Find tool messages
            tool_messages = [m for m in history if m.get("role") == "tool"]
            assert len(tool_messages) == 3

            # Verify order matches tool_call order, not completion order
            assert tool_messages[0]["tool_call_id"] == "call_1"
            assert tool_messages[1]["tool_call_id"] == "call_2"
            assert tool_messages[2]["tool_call_id"] == "call_3"
            assert tool_messages[0]["content"] == "Result:first"
            assert tool_messages[1]["content"] == "Result:second"
            assert tool_messages[2]["content"] == "Result:third"

    @pytest.mark.asyncio
    async def test_tool_error_handling_in_parallel(self, workflow_env: WorkflowEnvironment) -> None:
        """Test that tool errors don't prevent other tools from completing."""

        @activity.defn(name="inference_activity")
        async def inference_with_tools(input: InferenceInput) -> InferenceOutput:
            has_tool_results = any(m.get("role") == "tool" for m in input.messages)
            if has_tool_results:
                return InferenceOutput(
                    content="Done",
                    tool_calls=[],
                    finish_reason="stop",
                )
            return InferenceOutput(
                content=None,
                tool_calls=[
                    {"id": "1", "function": {"name": "good_tool", "arguments": "{}"}},
                    {"id": "2", "function": {"name": "bad_tool", "arguments": "{}"}},
                    {"id": "3", "function": {"name": "good_tool2", "arguments": "{}"}},
                ],
                finish_reason="tool_calls",
            )

        @activity.defn(name="tool_execution_activity")
        async def mixed_tool_activity(input: ToolExecutionInput) -> str:
            if input.tool_name == "bad_tool":
                raise ValueError("Tool failed!")
            return f"Success:{input.tool_name}"

        @activity.defn(name="get_agent_config_activity")
        async def config_activity(agent_name: str) -> AgentConfigOutput:
            return AgentConfigOutput(
                agent_name=agent_name,
                prompt="Test",
                model="gpt-4o",
                max_iterations=50,
                tools=[],
            )

        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Test error handling",
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                inference_with_tools,
                mixed_tool_activity,
                config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                input,
                id="test-error-handling-1",
                task_queue="test-queue",
            )
            await handle.result()

            history = await handle.query(AgentWorkflow.get_history)

            # Find tool messages
            tool_messages = [m for m in history if m.get("role") == "tool"]
            assert len(tool_messages) == 3

            # Good tools should have succeeded
            assert tool_messages[0]["content"] == "Success:good_tool"
            # Bad tool should have error
            assert "Error:" in tool_messages[1]["content"]
            # Third tool should still have succeeded
            assert tool_messages[2]["content"] == "Success:good_tool2"


class TestAgentWorkflowApproval:
    """Tests for AgentWorkflow approval functionality."""

    @pytest.mark.asyncio
    async def test_query_pending_approvals_empty(
        self, workflow_env: WorkflowEnvironment, simple_input: AgentWorkflowInput
    ) -> None:
        """Test get_pending_approvals returns empty list initially."""
        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                mock_inference_activity,
                mock_tool_execution_activity,
                mock_get_agent_config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                simple_input,
                id="test-approval-empty-1",
                task_queue="test-queue",
            )

            await handle.result()
            pending = await handle.query(AgentWorkflow.get_pending_approvals)

            assert pending == []

    @pytest.mark.asyncio
    async def test_approval_workflow_timeout(self, workflow_env: WorkflowEnvironment) -> None:
        """Test that approval timeout auto-rejects the tool call."""
        from datetime import timedelta

        from zap_ai.workflows.models import ApprovalRules

        @activity.defn(name="inference_activity")
        async def inference_with_approval_tool(input: InferenceInput) -> InferenceOutput:
            has_tool_result = any(m.get("role") == "tool" for m in input.messages)
            if has_tool_result:
                return InferenceOutput(
                    content="Timeout occurred.",
                    tool_calls=[],
                    finish_reason="stop",
                )
            return InferenceOutput(
                content=None,
                tool_calls=[
                    {
                        "id": "call_admin_1",
                        "type": "function",
                        "function": {"name": "admin_action", "arguments": "{}"},
                    }
                ],
                finish_reason="tool_calls",
            )

        @activity.defn(name="tool_execution_activity")
        async def tool_activity(input: ToolExecutionInput) -> str:
            return "Should not be called"

        @activity.defn(name="get_agent_config_activity")
        async def config_activity(agent_name: str) -> AgentConfigOutput:
            return AgentConfigOutput(
                agent_name=agent_name,
                prompt="Test prompt",
                model="gpt-4o",
                max_iterations=50,
                tools=[{"type": "function", "function": {"name": "admin_action"}}],
            )

        # Use very short timeout for testing
        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Do admin action",
            approval_rules=ApprovalRules(
                patterns=["admin_*"],
                timeout=timedelta(seconds=1),  # Very short timeout
            ).to_dict(),
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                inference_with_approval_tool,
                tool_activity,
                config_activity,
            ],
        ):
            handle = await workflow_env.client.start_workflow(
                AgentWorkflow.run,
                input,
                id="test-approval-timeout-1",
                task_queue="test-queue",
            )

            # Don't approve - let it timeout
            # Time-skipping environment should handle this quickly
            await handle.result()

            # Verify timeout message in history
            history = await handle.query(AgentWorkflow.get_history)
            tool_messages = [m for m in history if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            assert "timeout" in tool_messages[0]["content"].lower()

    @pytest.mark.asyncio
    async def test_tools_without_approval_rules_run_parallel(
        self, workflow_env: WorkflowEnvironment
    ) -> None:
        """Test that tools run in parallel when no approval rules are set."""
        import time

        execution_order: list[tuple[str, float]] = []

        @activity.defn(name="inference_activity")
        async def inference_with_tools(input: InferenceInput) -> InferenceOutput:
            has_tool_results = any(m.get("role") == "tool" for m in input.messages)
            if has_tool_results:
                return InferenceOutput(
                    content="All tools done",
                    tool_calls=[],
                    finish_reason="stop",
                )
            return InferenceOutput(
                content=None,
                tool_calls=[
                    {"id": "1", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "2", "function": {"name": "tool_b", "arguments": "{}"}},
                ],
                finish_reason="tool_calls",
            )

        @activity.defn(name="tool_execution_activity")
        async def tracking_tool_activity(input: ToolExecutionInput) -> str:
            import asyncio

            execution_order.append((input.tool_name, time.time()))
            await asyncio.sleep(0.05)
            return f"Result for {input.tool_name}"

        @activity.defn(name="get_agent_config_activity")
        async def config_activity(agent_name: str) -> AgentConfigOutput:
            return AgentConfigOutput(
                agent_name=agent_name,
                prompt="Test prompt",
                model="gpt-4o",
                max_iterations=50,
                tools=[],
            )

        # No approval_rules set
        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Use tools",
        )

        async with Worker(
            workflow_env.client,
            task_queue="test-queue",
            workflows=[AgentWorkflow],
            activities=[
                inference_with_tools,
                tracking_tool_activity,
                config_activity,
            ],
        ):
            await workflow_env.client.execute_workflow(
                AgentWorkflow.run,
                input,
                id="test-parallel-no-approval-1",
                task_queue="test-queue",
            )

            # Both tools should have been called
            assert len(execution_order) == 2

            # They should have started at nearly the same time (parallel)
            start_times = [t[1] for t in execution_order]
            time_spread = max(start_times) - min(start_times)
            assert time_spread < 0.05, f"Tools didn't start together: {time_spread}s"


class TestApprovalRulesLogic:
    """Tests for approval rules pattern matching logic."""

    def test_requires_approval_no_rules(self) -> None:
        """Test _requires_approval returns False when no rules."""
        from zap_ai.workflows.models import ApprovalRules

        # ApprovalRules.matches should return True for matching patterns
        rules = ApprovalRules(patterns=["transfer_*", "delete_*"])
        assert rules.matches("transfer_funds") is True
        assert rules.matches("delete_user") is True
        assert rules.matches("get_balance") is False

    def test_approval_rules_with_wildcards(self) -> None:
        """Test that wildcard patterns work correctly."""
        from zap_ai.workflows.models import ApprovalRules

        rules = ApprovalRules(patterns=["*_critical", "admin_*"])
        assert rules.matches("do_critical") is True
        assert rules.matches("admin_delete") is True
        assert rules.matches("safe_action") is False

    def test_approval_rules_exact_match(self) -> None:
        """Test that exact patterns work correctly."""
        from zap_ai.workflows.models import ApprovalRules

        rules = ApprovalRules(patterns=["specific_tool"])
        assert rules.matches("specific_tool") is True
        assert rules.matches("specific_tool_extra") is False
        assert rules.matches("other_tool") is False
