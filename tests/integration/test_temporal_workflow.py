"""Integration tests for AgentWorkflow with real Temporal server.

These tests verify that:
1. Worker can connect to a real Temporal server
2. Workflows execute correctly through actual Temporal infrastructure
3. Queries and signals work against real workflows
4. Multiple workflows can run concurrently

Note: LLM and MCP calls are mocked to avoid external dependencies.
"""

import pytest
from temporalio.client import Client, WorkflowHandle

from zap_ai.workflows import AgentWorkflow, AgentWorkflowInput


class TestWorkerConnectivity:
    """Tests verifying worker-server connectivity."""

    @pytest.mark.asyncio
    async def test_client_connects_to_server(self, temporal_client: Client) -> None:
        """Verify we can connect to the Temporal server."""
        assert temporal_client is not None

        # List workflows to verify connectivity (query succeeds = connected)
        async for _ in temporal_client.list_workflows(query=""):
            break


class TestTemporalIntegration:
    """Integration tests verifying workflow execution against real Temporal."""

    @pytest.mark.asyncio
    async def test_workflow_executes_on_real_temporal(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test basic workflow execution on real Temporal server."""
        input = AgentWorkflowInput(
            agent_name="IntegrationTestAgent",
            initial_task="Hello from integration test!",
        )

        result = await temporal_client.execute_workflow(
            AgentWorkflow.run,
            input,
            id="integration-test-basic",
            task_queue=task_queue,
        )

        assert result is not None
        assert "Integration test response to:" in result
        assert "Hello from integration test!" in result

    @pytest.mark.asyncio
    async def test_workflow_queries_work(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test that workflow queries work against real Temporal."""
        input = AgentWorkflowInput(
            agent_name="QueryTestAgent",
            initial_task="Test queries",
        )

        handle: WorkflowHandle = await temporal_client.start_workflow(
            AgentWorkflow.run,
            input,
            id="integration-test-queries",
            task_queue=task_queue,
        )

        await handle.result()

        status = await handle.query(AgentWorkflow.get_status)
        assert status == "completed"

        history = await handle.query(AgentWorkflow.get_history)
        assert len(history) >= 3  # system, user, assistant
        assert history[0]["role"] == "system"
        assert history[1]["role"] == "user"
        assert history[1]["content"] == "Test queries"

    @pytest.mark.asyncio
    async def test_multiple_concurrent_workflows(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test multiple workflows running concurrently on real Temporal."""
        handles = []

        for i in range(3):
            handle = await temporal_client.start_workflow(
                AgentWorkflow.run,
                AgentWorkflowInput(
                    agent_name=f"ConcurrentAgent{i}",
                    initial_task=f"Task number {i}",
                ),
                id=f"integration-test-concurrent-{i}",
                task_queue=task_queue,
            )
            handles.append(handle)

        results = []
        for handle in handles:
            result = await handle.result()
            results.append(result)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert f"Task number {i}" in result

    @pytest.mark.asyncio
    async def test_workflow_with_state_restoration(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test workflow correctly handles state from input (continue-as-new scenario)."""
        input = AgentWorkflowInput(
            agent_name="StateTestAgent",
            initial_task="",
            state={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Previous message"},
                    {"role": "assistant", "content": "Previous response"},
                    {"role": "user", "content": "Follow up question"},
                ],
                "iteration_count": 1,
                "pending_messages": [],
                "sub_agent_conversations": {},
            },
        )

        result = await temporal_client.execute_workflow(
            AgentWorkflow.run,
            input,
            id="integration-test-state",
            task_queue=task_queue,
        )

        assert result is not None
