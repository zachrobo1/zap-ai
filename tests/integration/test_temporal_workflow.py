"""Integration tests for AgentWorkflow with real Temporal server.

These tests verify that:
1. Worker can connect to a real Temporal server
2. Workflows execute correctly through actual Temporal infrastructure
3. Queries and signals work against real workflows
4. Multiple workflows can run concurrently
5. Context-based dynamic prompts are resolved correctly

Note: LLM and MCP calls are mocked to avoid external dependencies.
"""

from dataclasses import dataclass

import pytest
from temporalio.client import Client, WorkflowHandle

from zap_ai import Zap, ZapAgent
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


class TestContextIntegration:
    """Integration tests for context-based dynamic prompts."""

    @pytest.mark.asyncio
    async def test_dynamic_prompt_resolves_with_context(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test that dynamic prompts are resolved with context and passed to workflow."""

        @dataclass
        class UserContext:
            user_name: str
            role: str

        def make_prompt(ctx: UserContext) -> str:
            return f"You are an assistant for {ctx.user_name}, who is a {ctx.role}."

        agent: ZapAgent[UserContext] = ZapAgent(
            name="ContextAgent",
            prompt=make_prompt,
        )

        zap: Zap[UserContext] = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="ContextAgent",
                task="Hello, help me with something",
                context=UserContext(user_name="Alice", role="developer"),
            )

            # Wait for task to complete
            result = await zap.get_task(task.id)
            while not result.status.is_terminal():
                import asyncio

                await asyncio.sleep(0.1)
                result = await zap.get_task(task.id)

            # Verify the resolved prompt appears in the conversation history
            assert result.history is not None
            system_message = result.history[0]
            assert system_message["role"] == "system"
            assert "Alice" in system_message["content"]
            assert "developer" in system_message["content"]
            assert (
                system_message["content"] == "You are an assistant for Alice, who is a developer."
            )
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_static_prompt_works_without_context(
        self,
        temporal_client: Client,
        integration_worker,
        task_queue: str,
    ) -> None:
        """Test that static prompts work correctly without context."""
        agent = ZapAgent(
            name="StaticAgent",
            prompt="You are a helpful static assistant.",
        )

        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="StaticAgent",
                task="Hello!",
            )

            # Wait for task to complete
            result = await zap.get_task(task.id)
            while not result.status.is_terminal():
                import asyncio

                await asyncio.sleep(0.1)
                result = await zap.get_task(task.id)

            # Verify the static prompt appears in the conversation history
            assert result.history is not None
            system_message = result.history[0]
            assert system_message["role"] == "system"
            assert system_message["content"] == "You are a helpful static assistant."
        finally:
            await zap.stop()
