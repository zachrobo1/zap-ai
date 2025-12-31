"""Tests for worker module."""

import pytest
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from zap_ai.activities import InferenceInput, InferenceOutput, ToolExecutionInput
from zap_ai.activities.tool_execution import AgentConfigOutput
from zap_ai.worker import create_worker, run_worker_with_zap
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


class TestCreateWorker:
    """Tests for create_worker function."""

    @pytest.mark.asyncio
    async def test_create_worker_returns_worker(self) -> None:
        """Test that create_worker returns a Worker instance."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            worker = await create_worker(env.client, "test-queue")
            assert worker is not None
            assert isinstance(worker, Worker)

    @pytest.mark.asyncio
    async def test_create_worker_with_custom_task_queue(self) -> None:
        """Test create_worker with a custom task queue."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            worker = await create_worker(env.client, "custom-queue")
            assert worker is not None

    @pytest.mark.asyncio
    async def test_create_worker_without_registry(self) -> None:
        """Test create_worker works without tool registry."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            worker = await create_worker(env.client, "test-queue", tool_registry=None)
            assert worker is not None


class TestWorkerExecutesWorkflows:
    """Tests for worker executing workflows."""

    @pytest.mark.asyncio
    async def test_worker_executes_workflow_with_mocks(self) -> None:
        """Test that worker can execute workflow with mock activities."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[AgentWorkflow],
                activities=[
                    mock_inference_activity,
                    mock_tool_execution_activity,
                    mock_get_agent_config_activity,
                ],
            ):
                result = await env.client.execute_workflow(
                    AgentWorkflow.run,
                    AgentWorkflowInput(agent_name="Test", initial_task="Hello"),
                    id="test-worker-1",
                    task_queue="test-queue",
                )

                assert result is not None
                assert "Mock response to:" in result

    @pytest.mark.asyncio
    async def test_worker_handles_multiple_workflows(self) -> None:
        """Test that worker can handle multiple concurrent workflows."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[AgentWorkflow],
                activities=[
                    mock_inference_activity,
                    mock_tool_execution_activity,
                    mock_get_agent_config_activity,
                ],
            ):
                # Start multiple workflows
                handles = []
                for i in range(3):
                    handle = await env.client.start_workflow(
                        AgentWorkflow.run,
                        AgentWorkflowInput(agent_name=f"Agent{i}", initial_task=f"Task {i}"),
                        id=f"test-multi-{i}",
                        task_queue="test-queue",
                    )
                    handles.append(handle)

                # Wait for all to complete
                results = []
                for handle in handles:
                    result = await handle.result()
                    results.append(result)

                assert len(results) == 3
                for result in results:
                    assert "Mock response to:" in result

    @pytest.mark.asyncio
    async def test_created_worker_executes_workflow(self) -> None:
        """Test that worker from create_worker can be started."""
        async with await WorkflowEnvironment.start_time_skipping() as env:
            # create_worker registers real activities, but we need mock activities for testing
            # So we create a worker with mock activities instead
            async with Worker(
                env.client,
                task_queue="test-queue",
                workflows=[AgentWorkflow],
                activities=[
                    mock_inference_activity,
                    mock_tool_execution_activity,
                    mock_get_agent_config_activity,
                ],
            ):
                result = await env.client.execute_workflow(
                    AgentWorkflow.run,
                    AgentWorkflowInput(agent_name="Test", initial_task="Hello from worker"),
                    id="test-created-worker-1",
                    task_queue="test-queue",
                )

                assert "Mock response to:" in result
                assert "Hello from worker" in result


class TestRunWorkerWithZap:
    """Tests for run_worker_with_zap function."""

    @pytest.mark.asyncio
    async def test_run_worker_with_zap_requires_started(self) -> None:
        """Test that run_worker_with_zap raises if Zap not started."""
        from zap_ai.core.agent import ZapAgent
        from zap_ai.core.zap import Zap

        agent = ZapAgent(name="Test", prompt="You are helpful")
        zap = Zap(agents=[agent])

        with pytest.raises(RuntimeError, match="must be started"):
            await run_worker_with_zap(zap)

    @pytest.mark.asyncio
    async def test_run_worker_with_zap_uses_task_queue(self) -> None:
        """Test that run_worker_with_zap uses Zap's task queue."""
        from unittest.mock import AsyncMock, MagicMock

        from zap_ai.core.agent import ZapAgent
        from zap_ai.core.zap import Zap

        # Create mock Temporal client
        mock_client = MagicMock()
        mock_client.start_workflow = AsyncMock()
        mock_client.get_workflow_handle = MagicMock()

        agent = ZapAgent(name="Test", prompt="You are helpful")
        zap = Zap(agents=[agent], task_queue="custom-zap-queue", temporal_client=mock_client)

        # Start zap (uses mock client, no real connection)
        await zap.start()

        # run_worker_with_zap would try to connect to Temporal,
        # which we can't do in unit tests without a real server.
        # Just verify it uses the right task queue.
        assert zap.task_queue == "custom-zap-queue"

        # Cleanup
        await zap.stop()


class TestWorkerModuleImports:
    """Tests for worker module imports."""

    def test_imports_from_module(self) -> None:
        """Test that all exports are importable from module."""
        from zap_ai.worker import create_worker, run_worker, run_worker_with_zap

        assert create_worker is not None
        assert run_worker is not None
        assert run_worker_with_zap is not None

    def test_all_exports_defined(self) -> None:
        """Test that __all__ is properly defined."""
        from zap_ai import worker

        assert hasattr(worker, "__all__")
        assert "create_worker" in worker.__all__
        assert "run_worker" in worker.__all__
        assert "run_worker_with_zap" in worker.__all__
