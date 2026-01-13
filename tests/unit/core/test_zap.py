"""Tests for Zap orchestrator class."""

import warnings
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from zap_ai import (
    AgentNotFoundError,
    Task,
    TaskNotFoundError,
    TaskStatus,
    Zap,
    ZapAgent,
    ZapConfigurationError,
    ZapNotStartedError,
)


@pytest.fixture
def mock_temporal_client() -> MagicMock:
    """Create a mock Temporal client."""
    client = MagicMock()
    client.start_workflow = AsyncMock()
    client.get_workflow_handle = MagicMock()
    return client


@pytest.fixture
def zap_with_mock_client(sample_agent: ZapAgent, mock_temporal_client: MagicMock) -> Zap:
    """A Zap instance with a mock Temporal client."""
    return Zap(agents=[sample_agent], temporal_client=mock_temporal_client)


class TestZapCreation:
    """Test Zap instantiation with valid inputs."""

    def test_single_agent(self, sample_agent: ZapAgent) -> None:
        """Test creating Zap with a single agent."""
        zap = Zap(agents=[sample_agent])
        assert len(zap.agents) == 1
        assert zap.list_agents() == ["TestAgent"]

    def test_multiple_agents(self, multi_agent_list: list[ZapAgent]) -> None:
        """Test creating Zap with multiple agents."""
        zap = Zap(agents=multi_agent_list)
        assert len(zap.agents) == 3

    def test_default_task_queue(self, sample_agent: ZapAgent) -> None:
        """Test that default task_queue is 'zap-agents'."""
        zap = Zap(agents=[sample_agent])
        assert zap.task_queue == "zap-agents"

    def test_custom_task_queue(self, sample_agent: ZapAgent) -> None:
        """Test specifying custom task_queue."""
        zap = Zap(agents=[sample_agent], task_queue="my-queue")
        assert zap.task_queue == "my-queue"

    def test_temporal_client_defaults_none(self, sample_agent: ZapAgent) -> None:
        """Test that temporal_client defaults to None."""
        zap = Zap(agents=[sample_agent])
        assert zap.temporal_client is None


class TestZapDuplicateNameValidation:
    """Test duplicate agent name detection."""

    def test_duplicate_names_rejected(self) -> None:
        """Test that duplicate agent names raise ZapConfigurationError."""
        agents = [
            ZapAgent(name="Agent", prompt="test1"),
            ZapAgent(name="Agent", prompt="test2"),
        ]
        with pytest.raises(ZapConfigurationError) as exc_info:
            Zap(agents=agents)
        assert "Duplicate agent names" in str(exc_info.value)
        assert "Agent" in str(exc_info.value)

    def test_multiple_duplicates_reported(self) -> None:
        """Test that multiple duplicates are detected."""
        agents = [
            ZapAgent(name="A", prompt="test"),
            ZapAgent(name="B", prompt="test"),
            ZapAgent(name="A", prompt="test"),
            ZapAgent(name="B", prompt="test"),
        ]
        with pytest.raises(ZapConfigurationError) as exc_info:
            Zap(agents=agents)
        assert "Duplicate agent names" in str(exc_info.value)


class TestZapSubAgentValidation:
    """Test sub-agent reference validation."""

    def test_valid_sub_agent_references(self, agent_with_sub_agents: list[ZapAgent]) -> None:
        """Test that valid sub-agent references are accepted."""
        zap = Zap(agents=agent_with_sub_agents)
        assert len(zap.agents) == 3

    def test_unknown_sub_agent_rejected(self) -> None:
        """Test that referencing unknown sub-agent raises error."""
        agents = [
            ZapAgent(name="Main", prompt="test", sub_agents=["Unknown"]),
        ]
        with pytest.raises(ZapConfigurationError) as exc_info:
            Zap(agents=agents)
        assert "unknown sub-agent 'Unknown'" in str(exc_info.value)

    def test_self_reference_rejected(self) -> None:
        """Test that agent cannot reference itself as sub-agent."""
        agents = [
            ZapAgent(name="Main", prompt="test", sub_agents=["Main"]),
        ]
        with pytest.raises(ZapConfigurationError) as exc_info:
            Zap(agents=agents)
        assert "cannot reference itself" in str(exc_info.value)


class TestZapCircularDependencyValidation:
    """Test circular dependency detection."""

    def test_simple_circular_dependency(self) -> None:
        """Test that A -> B -> A is detected."""
        agents = [
            ZapAgent(name="A", prompt="test", sub_agents=["B"]),
            ZapAgent(name="B", prompt="test", sub_agents=["A"]),
        ]
        with pytest.raises(ZapConfigurationError) as exc_info:
            Zap(agents=agents)
        assert "Circular dependency" in str(exc_info.value)

    def test_three_node_cycle(self) -> None:
        """Test that A -> B -> C -> A is detected."""
        agents = [
            ZapAgent(name="A", prompt="test", sub_agents=["B"]),
            ZapAgent(name="B", prompt="test", sub_agents=["C"]),
            ZapAgent(name="C", prompt="test", sub_agents=["A"]),
        ]
        with pytest.raises(ZapConfigurationError) as exc_info:
            Zap(agents=agents)
        assert "Circular dependency" in str(exc_info.value)

    def test_chain_without_cycle(self) -> None:
        """Test that A -> B -> C (no cycle) is valid."""
        agents = [
            ZapAgent(name="A", prompt="test", sub_agents=["B"]),
            ZapAgent(name="B", prompt="test", sub_agents=["C"]),
            ZapAgent(name="C", prompt="test"),
        ]
        zap = Zap(agents=agents)
        assert len(zap.agents) == 3

    def test_diamond_without_cycle(self) -> None:
        """Test diamond pattern (A -> B, A -> C, B -> D, C -> D) is valid."""
        agents = [
            ZapAgent(name="A", prompt="test", sub_agents=["B", "C"]),
            ZapAgent(name="B", prompt="test", sub_agents=["D"]),
            ZapAgent(name="C", prompt="test", sub_agents=["D"]),
            ZapAgent(name="D", prompt="test"),
        ]
        zap = Zap(agents=agents)
        assert len(zap.agents) == 4


class TestZapGetAgent:
    """Test get_agent method."""

    def test_get_existing_agent(self, zap_instance: Zap) -> None:
        """Test getting an existing agent."""
        agent = zap_instance.get_agent("TestAgent")
        assert agent.name == "TestAgent"

    def test_get_unknown_agent(self, zap_instance: Zap) -> None:
        """Test that getting unknown agent raises error."""
        with pytest.raises(AgentNotFoundError) as exc_info:
            zap_instance.get_agent("Unknown")
        assert "not found" in str(exc_info.value)
        assert "Available agents" in str(exc_info.value)


class TestZapListAgents:
    """Test list_agents method."""

    def test_list_agents_single(self, zap_instance: Zap) -> None:
        """Test listing agents with single agent."""
        assert zap_instance.list_agents() == ["TestAgent"]

    def test_list_agents_multiple(self, multi_agent_list: list[ZapAgent]) -> None:
        """Test listing agents with multiple agents."""
        zap = Zap(agents=multi_agent_list)
        names = zap.list_agents()
        assert len(names) == 3
        assert "MainAgent" in names
        assert "HelperAgent" in names
        assert "ReviewerAgent" in names


class TestZapStartStop:
    """Test start and stop methods."""

    @pytest.mark.asyncio
    async def test_start_sets_started_flag(self, zap_with_mock_client: Zap) -> None:
        """Test that start() sets the _started flag."""
        assert zap_with_mock_client._started is False
        await zap_with_mock_client.start()
        assert zap_with_mock_client._started is True

    @pytest.mark.asyncio
    async def test_start_twice_raises_error(self, zap_with_mock_client: Zap) -> None:
        """Test that calling start() twice raises error."""
        await zap_with_mock_client.start()
        with pytest.raises(RuntimeError) as exc_info:
            await zap_with_mock_client.start()
        assert "already been started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_stop_clears_started_flag(self, zap_with_mock_client: Zap) -> None:
        """Test that stop() clears the _started flag."""
        await zap_with_mock_client.start()
        assert zap_with_mock_client._started is True
        await zap_with_mock_client.stop()
        assert zap_with_mock_client._started is False

    @pytest.mark.asyncio
    async def test_stop_before_start_is_noop(self, zap_instance: Zap) -> None:
        """Test that stop() before start() is a no-op."""
        await zap_instance.stop()  # Should not raise
        assert zap_instance._started is False

    @pytest.mark.asyncio
    async def test_start_initializes_tool_registry(self, zap_with_mock_client: Zap) -> None:
        """Test that start() initializes the tool registry."""
        assert zap_with_mock_client._tool_registry is None
        await zap_with_mock_client.start()
        assert zap_with_mock_client._tool_registry is not None

    @pytest.mark.asyncio
    async def test_stop_clears_tool_registry(self, zap_with_mock_client: Zap) -> None:
        """Test that stop() clears the tool registry."""
        await zap_with_mock_client.start()
        assert zap_with_mock_client._tool_registry is not None
        await zap_with_mock_client.stop()
        assert zap_with_mock_client._tool_registry is None


class TestZapExecuteTask:
    """Test execute_task method."""

    @pytest.mark.asyncio
    async def test_execute_before_start_raises(self, zap_instance: Zap) -> None:
        """Test that execute_task before start raises error."""
        with pytest.raises(ZapNotStartedError) as exc_info:
            await zap_instance.execute_task(agent_name="TestAgent", task="Hello")
        assert "not been started" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_without_task_raises(self, zap_with_mock_client: Zap) -> None:
        """Test that execute_task without task raises ValueError."""
        await zap_with_mock_client.start()
        with pytest.raises(ValueError) as exc_info:
            await zap_with_mock_client.execute_task(agent_name="TestAgent")
        assert "task argument is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_new_task_without_agent_raises(self, zap_with_mock_client: Zap) -> None:
        """Test that new task without agent_name raises ValueError."""
        await zap_with_mock_client.start()
        with pytest.raises(ValueError) as exc_info:
            await zap_with_mock_client.execute_task(task="Hello")
        assert "agent_name is required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_unknown_agent_raises(self, zap_with_mock_client: Zap) -> None:
        """Test that unknown agent raises AgentNotFoundError."""
        await zap_with_mock_client.start()
        with pytest.raises(AgentNotFoundError):
            await zap_with_mock_client.execute_task(agent_name="Unknown", task="Hello")

    @pytest.mark.asyncio
    async def test_execute_returns_pending_task(self, zap_with_mock_client: Zap) -> None:
        """Test that execute_task returns a task with PENDING status."""
        await zap_with_mock_client.start()
        task = await zap_with_mock_client.execute_task(agent_name="TestAgent", task="Hello")
        assert isinstance(task, Task)
        assert task.status == TaskStatus.PENDING
        assert task.agent_name == "TestAgent"

    @pytest.mark.asyncio
    async def test_execute_task_id_format(self, zap_with_mock_client: Zap) -> None:
        """Test that task ID has correct format."""
        await zap_with_mock_client.start()
        task = await zap_with_mock_client.execute_task(agent_name="TestAgent", task="Hello")
        assert task.id.startswith("TestAgent-")
        # Should have 12 hex chars after the dash
        parts = task.id.split("-")
        assert len(parts) == 2
        assert len(parts[1]) == 12

    @pytest.mark.asyncio
    async def test_execute_calls_start_workflow(
        self, zap_with_mock_client: Zap, mock_temporal_client: MagicMock
    ) -> None:
        """Test that execute_task calls temporal start_workflow."""
        await zap_with_mock_client.start()
        await zap_with_mock_client.execute_task(agent_name="TestAgent", task="Hello")

        mock_temporal_client.start_workflow.assert_called_once()
        call_kwargs = mock_temporal_client.start_workflow.call_args.kwargs
        assert call_kwargs["task_queue"] == "zap-agents"
        assert call_kwargs["id"].startswith("TestAgent-")

    @pytest.mark.asyncio
    async def test_follow_up_sends_signal(
        self, zap_with_mock_client: Zap, mock_temporal_client: MagicMock
    ) -> None:
        """Test that follow_up sends signal to workflow."""
        # Set up mock handle
        mock_handle = AsyncMock()
        mock_handle.signal = AsyncMock()
        mock_handle.query = AsyncMock(side_effect=["thinking", []])
        mock_temporal_client.get_workflow_handle.return_value = mock_handle

        await zap_with_mock_client.start()
        task = await zap_with_mock_client.execute_task(
            follow_up_on_task="TestAgent-abc123", task="Follow up"
        )

        mock_temporal_client.get_workflow_handle.assert_called_with("TestAgent-abc123")
        mock_handle.signal.assert_called_once()
        assert task.agent_name == "TestAgent"

    @pytest.mark.asyncio
    async def test_follow_up_task_not_found_raises(
        self, zap_with_mock_client: Zap, mock_temporal_client: MagicMock
    ) -> None:
        """Test that follow_up on non-existent task raises TaskNotFoundError."""
        mock_temporal_client.get_workflow_handle.side_effect = Exception("Not found")

        await zap_with_mock_client.start()
        with pytest.raises(TaskNotFoundError):
            await zap_with_mock_client.execute_task(
                follow_up_on_task="nonexistent-123", task="Follow up"
            )


class TestZapGetTask:
    """Test get_task method."""

    @pytest.mark.asyncio
    async def test_get_task_before_start_raises(self, zap_instance: Zap) -> None:
        """Test that get_task before start raises error."""
        with pytest.raises(ZapNotStartedError):
            await zap_instance.get_task("some-id")

    @pytest.mark.asyncio
    async def test_get_task_queries_workflow(
        self, zap_with_mock_client: Zap, mock_temporal_client: MagicMock
    ) -> None:
        """Test that get_task queries the workflow."""
        # Set up mock handle
        mock_handle = AsyncMock()
        # Query order: status, result, error, history, sub_agent_conversations
        mock_handle.query = AsyncMock(
            side_effect=[
                "completed",
                "Task result",
                None,
                [{"role": "user"}],
                {},  # empty sub-agent conversations
            ]
        )
        mock_temporal_client.get_workflow_handle.return_value = mock_handle

        await zap_with_mock_client.start()
        task = await zap_with_mock_client.get_task("TestAgent-abc123")

        assert task.id == "TestAgent-abc123"
        assert task.agent_name == "TestAgent"
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Task result"
        assert task.sub_tasks == []
        assert task._task_fetcher is not None

    @pytest.mark.asyncio
    async def test_get_task_not_found_raises(
        self, zap_with_mock_client: Zap, mock_temporal_client: MagicMock
    ) -> None:
        """Test that get_task raises TaskNotFoundError for unknown task."""
        mock_temporal_client.get_workflow_handle.side_effect = Exception("Not found")

        await zap_with_mock_client.start()
        with pytest.raises(TaskNotFoundError):
            await zap_with_mock_client.get_task("unknown-123")


class TestZapCancelTask:
    """Test cancel_task method."""

    @pytest.mark.asyncio
    async def test_cancel_task_before_start_raises(self, zap_instance: Zap) -> None:
        """Test that cancel_task before start raises error."""
        with pytest.raises(ZapNotStartedError):
            await zap_instance.cancel_task("some-id")

    @pytest.mark.asyncio
    async def test_cancel_task_not_implemented(self, zap_with_mock_client: Zap) -> None:
        """Test that cancel_task raises NotImplementedError (skeleton)."""
        await zap_with_mock_client.start()
        with pytest.raises(NotImplementedError):
            await zap_with_mock_client.cancel_task("some-id")


class TestZapContextSupport:
    """Test context passing in execute_task."""

    @pytest.mark.asyncio
    async def test_execute_with_dict_context(self, mock_temporal_client: MagicMock) -> None:
        """Test that dict context is passed to dynamic prompt."""
        agent = ZapAgent(
            name="Test",
            prompt=lambda ctx: f"You assist {ctx['name']} from {ctx['company']}.",
        )
        zap = Zap(agents=[agent], temporal_client=mock_temporal_client)
        await zap.start()

        await zap.execute_task(
            agent_name="Test",
            task="Hello",
            context={"name": "Alice", "company": "Acme"},
        )

        # Verify workflow was started with resolved prompt
        call_args = mock_temporal_client.start_workflow.call_args
        workflow_input = call_args[0][1]
        assert workflow_input.system_prompt == "You assist Alice from Acme."

    @pytest.mark.asyncio
    async def test_execute_with_typed_context(self, mock_temporal_client: MagicMock) -> None:
        """Test that typed context is passed to dynamic prompt."""

        @dataclass
        class MyContext:
            user_name: str
            company: str

        def resolve_prompt(ctx: MyContext) -> str:
            return f"You assist {ctx.user_name} from {ctx.company}."

        agent: ZapAgent[MyContext] = ZapAgent(
            name="Test",
            prompt=resolve_prompt,
        )
        zap: Zap[MyContext] = Zap(agents=[agent], temporal_client=mock_temporal_client)
        await zap.start()

        await zap.execute_task(
            agent_name="Test",
            task="Hello",
            context=MyContext(user_name="Bob", company="TechCo"),
        )

        call_args = mock_temporal_client.start_workflow.call_args
        workflow_input = call_args[0][1]
        assert workflow_input.system_prompt == "You assist Bob from TechCo."

    @pytest.mark.asyncio
    async def test_execute_static_prompt_ignores_context(
        self, mock_temporal_client: MagicMock
    ) -> None:
        """Test that static prompt works regardless of context."""
        agent = ZapAgent(name="Test", prompt="Static prompt")
        zap = Zap(agents=[agent], temporal_client=mock_temporal_client)
        await zap.start()

        await zap.execute_task(agent_name="Test", task="Hello", context={"ignored": "value"})

        call_args = mock_temporal_client.start_workflow.call_args
        workflow_input = call_args[0][1]
        assert workflow_input.system_prompt == "Static prompt"

    @pytest.mark.asyncio
    async def test_execute_without_context_uses_empty_dict(
        self, mock_temporal_client: MagicMock
    ) -> None:
        """Test that missing context defaults to empty dict."""
        agent = ZapAgent(name="Test", prompt="Static prompt")
        zap = Zap(agents=[agent], temporal_client=mock_temporal_client)
        await zap.start()

        await zap.execute_task(agent_name="Test", task="Hello")

        call_args = mock_temporal_client.start_workflow.call_args
        workflow_input = call_args[0][1]
        assert workflow_input.system_prompt == "Static prompt"

    @pytest.mark.asyncio
    async def test_dynamic_prompt_without_context_warns(
        self, mock_temporal_client: MagicMock
    ) -> None:
        """Test that dynamic prompt without context emits warning."""
        agent = ZapAgent(
            name="Test",
            prompt=lambda ctx: f"User: {ctx.get('name', 'unknown')}",
        )
        zap = Zap(agents=[agent], temporal_client=mock_temporal_client)
        await zap.start()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await zap.execute_task(agent_name="Test", task="Hello")

            assert len(w) == 1
            assert "dynamic prompt but no context was provided" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

    @pytest.mark.asyncio
    async def test_dynamic_prompt_with_context_no_warning(
        self, mock_temporal_client: MagicMock
    ) -> None:
        """Test that dynamic prompt with context does not warn."""
        agent = ZapAgent(
            name="Test",
            prompt=lambda ctx: f"User: {ctx.get('name', 'unknown')}",
        )
        zap = Zap(agents=[agent], temporal_client=mock_temporal_client)
        await zap.start()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await zap.execute_task(agent_name="Test", task="Hello", context={"name": "Alice"})

            # No warnings should be emitted
            assert len(w) == 0


class TestZapGetAgentTools:
    """Tests for Zap.get_agent_tools method."""

    @pytest.mark.asyncio
    async def test_get_agent_tools_before_start_raises(self, zap_instance: Zap) -> None:
        """Test that get_agent_tools before start raises ZapNotStartedError."""
        with pytest.raises(ZapNotStartedError):
            await zap_instance.get_agent_tools("TestAgent")

    @pytest.mark.asyncio
    async def test_get_agent_tools_unknown_agent_raises(self, zap_with_mock_client: Zap) -> None:
        """Test that unknown agent raises AgentNotFoundError."""
        await zap_with_mock_client.start()
        with pytest.raises(AgentNotFoundError):
            await zap_with_mock_client.get_agent_tools("UnknownAgent")

    @pytest.mark.asyncio
    async def test_get_agent_tools_returns_tool_names(self, zap_with_mock_client: Zap) -> None:
        """Test that get_agent_tools returns tool names from registry."""
        await zap_with_mock_client.start()

        # Mock the tool registry
        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = ["tool_a", "tool_b", "tool_c"]
        zap_with_mock_client._tool_registry = mock_registry

        result = await zap_with_mock_client.get_agent_tools("TestAgent")

        assert result == ["tool_a", "tool_b", "tool_c"]
        mock_registry.get_tool_names.assert_called_once_with("TestAgent")

    @pytest.mark.asyncio
    async def test_get_agent_tools_no_registry_returns_empty(
        self, zap_with_mock_client: Zap
    ) -> None:
        """Test that get_agent_tools returns empty list if no registry."""
        await zap_with_mock_client.start()

        # Set registry to None (shouldn't happen in practice, but defensive)
        zap_with_mock_client._tool_registry = None

        result = await zap_with_mock_client.get_agent_tools("TestAgent")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_agent_tools_valid_agent_empty_tools(self, zap_with_mock_client: Zap) -> None:
        """Test that get_agent_tools returns empty list for agent with no tools."""
        await zap_with_mock_client.start()

        # Mock the tool registry to return empty
        mock_registry = MagicMock()
        mock_registry.get_tool_names.return_value = []
        zap_with_mock_client._tool_registry = mock_registry

        result = await zap_with_mock_client.get_agent_tools("TestAgent")

        assert result == []
