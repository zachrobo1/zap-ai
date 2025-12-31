"""Tests for tool execution activity."""

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from zap_ai.activities import (
    AgentConfigOutput,
    ToolExecutionError,
    ToolExecutionInput,
    ToolNotFoundError,
    get_agent_config_activity,
    get_tool_registry,
    set_tool_registry,
    tool_execution_activity,
)


@pytest.fixture
def activity_env() -> ActivityEnvironment:
    """Create an activity testing environment."""
    return ActivityEnvironment()


@pytest.fixture
def mock_registry(mocker: Any) -> Generator[MagicMock, None, None]:
    """Create a mock tool registry and patch it."""
    mock = MagicMock()

    # Default mock for get_client_for_tool
    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(return_value="tool result")
    mock.get_client_for_tool.return_value = mock_client

    # Patch get_tool_registry to return our mock
    mocker.patch(
        "zap_ai.activities.tool_execution.get_tool_registry",
        return_value=mock,
    )
    yield mock


class TestToolExecutionInput:
    """Tests for ToolExecutionInput dataclass."""

    def test_create_minimal_input(self) -> None:
        """Test creating input with only required fields."""
        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
        )
        assert input.agent_name == "TestAgent"
        assert input.tool_name == "search"
        assert input.arguments == {}

    def test_create_full_input(self) -> None:
        """Test creating input with all fields."""
        args = {"query": "test", "limit": 10}
        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
            arguments=args,
        )
        assert input.arguments == args


class TestToolExecutionExceptions:
    """Tests for tool execution exception classes."""

    def test_tool_execution_error(self) -> None:
        """Test ToolExecutionError exception."""
        error = ToolExecutionError("Tool failed")
        assert str(error) == "Tool failed"
        assert isinstance(error, Exception)

    def test_tool_not_found_error(self) -> None:
        """Test ToolNotFoundError exception."""
        error = ToolNotFoundError("Tool 'search' not found")
        assert str(error) == "Tool 'search' not found"
        assert isinstance(error, Exception)


class TestToolRegistry:
    """Tests for global tool registry functions."""

    def test_set_and_get_registry(self) -> None:
        """Test setting and getting the global registry."""
        mock_registry = MagicMock()
        set_tool_registry(mock_registry)

        result = get_tool_registry()
        assert result == mock_registry

    def test_get_registry_when_not_set(self) -> None:
        """Test that getting registry when not set raises error."""
        # Reset registry
        set_tool_registry(None)

        with pytest.raises(RuntimeError, match="Tool registry not initialized"):
            get_tool_registry()


class TestToolExecutionActivity:
    """Tests for the tool execution activity function."""

    @pytest.mark.asyncio
    async def test_activity_calls_tool_via_registry(
        self, activity_env: ActivityEnvironment, mock_registry: MagicMock
    ) -> None:
        """Test that activity calls the tool via registry."""
        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
            arguments={"query": "test"},
        )

        result = await activity_env.run(tool_execution_activity, input)

        mock_registry.get_client_for_tool.assert_called_once_with("TestAgent", "search")
        assert result == "tool result"

    @pytest.mark.asyncio
    async def test_activity_returns_string_result_directly(
        self, activity_env: ActivityEnvironment, mock_registry: MagicMock
    ) -> None:
        """Test that string results are returned directly."""
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(return_value="string result")
        mock_registry.get_client_for_tool.return_value = mock_client

        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
            arguments={},
        )

        result = await activity_env.run(tool_execution_activity, input)
        assert result == "string result"

    @pytest.mark.asyncio
    async def test_activity_serializes_non_string_result(
        self, activity_env: ActivityEnvironment, mock_registry: MagicMock
    ) -> None:
        """Test that non-string results are JSON serialized."""
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(return_value={"key": "value"})
        mock_registry.get_client_for_tool.return_value = mock_client

        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
            arguments={},
        )

        result = await activity_env.run(tool_execution_activity, input)
        assert result == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_activity_raises_tool_not_found_for_key_error(
        self, activity_env: ActivityEnvironment, mock_registry: MagicMock
    ) -> None:
        """Test that KeyError from registry raises ToolNotFoundError."""
        mock_registry.get_client_for_tool.side_effect = KeyError("Tool not found")

        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="unknown_tool",
            arguments={},
        )

        with pytest.raises(ToolNotFoundError):
            await activity_env.run(tool_execution_activity, input)

    @pytest.mark.asyncio
    async def test_activity_raises_tool_execution_error_for_value_error(
        self, activity_env: ActivityEnvironment, mock_registry: MagicMock
    ) -> None:
        """Test that ValueError from registry raises ToolExecutionError."""
        mock_registry.get_client_for_tool.side_effect = ValueError(
            "message_agent is handled by workflow"
        )

        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="message_agent",
            arguments={},
        )

        with pytest.raises(ToolExecutionError):
            await activity_env.run(tool_execution_activity, input)

    @pytest.mark.asyncio
    async def test_activity_raises_tool_execution_error_on_call_failure(
        self, activity_env: ActivityEnvironment, mock_registry: MagicMock
    ) -> None:
        """Test that tool call failure raises ToolExecutionError."""
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(side_effect=Exception("Tool failed"))
        mock_registry.get_client_for_tool.return_value = mock_client

        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
            arguments={},
        )

        with pytest.raises(ToolExecutionError, match="Failed to execute tool"):
            await activity_env.run(tool_execution_activity, input)

    @pytest.mark.asyncio
    async def test_activity_raises_when_registry_not_initialized(
        self, activity_env: ActivityEnvironment, mocker: Any
    ) -> None:
        """Test that activity raises RuntimeError when registry not set."""
        mocker.patch(
            "zap_ai.activities.tool_execution.get_tool_registry",
            side_effect=RuntimeError("Tool registry not initialized"),
        )

        input = ToolExecutionInput(
            agent_name="TestAgent",
            tool_name="search",
            arguments={},
        )

        with pytest.raises(RuntimeError, match="Tool registry not initialized"):
            await activity_env.run(tool_execution_activity, input)


class TestAgentConfigOutput:
    """Tests for AgentConfigOutput dataclass."""

    def test_create_minimal_output(self) -> None:
        """Test creating output with only required fields."""
        output = AgentConfigOutput(
            agent_name="TestAgent",
            prompt="You are helpful.",
            model="gpt-4o",
            max_iterations=50,
        )
        assert output.agent_name == "TestAgent"
        assert output.prompt == "You are helpful."
        assert output.model == "gpt-4o"
        assert output.max_iterations == 50
        assert output.tools == []

    def test_create_full_output(self) -> None:
        """Test creating output with tools."""
        tools = [{"type": "function", "function": {"name": "search"}}]
        output = AgentConfigOutput(
            agent_name="TestAgent",
            prompt="You are helpful.",
            model="anthropic/claude-sonnet-4-5-20250929",
            max_iterations=25,
            tools=tools,
        )
        assert output.tools == tools


class TestGetAgentConfigActivity:
    """Tests for the get_agent_config activity function."""

    @pytest.mark.asyncio
    async def test_activity_raises_when_registry_not_set(
        self, activity_env: ActivityEnvironment, mocker: Any
    ) -> None:
        """Test that activity raises RuntimeError when registry not initialized."""
        mocker.patch(
            "zap_ai.activities.tool_execution.get_tool_registry",
            side_effect=RuntimeError("Tool registry not initialized"),
        )

        with pytest.raises(RuntimeError, match="Tool registry not initialized"):
            await activity_env.run(get_agent_config_activity, "TestAgent")

    @pytest.mark.asyncio
    async def test_activity_returns_config_from_registry(
        self, activity_env: ActivityEnvironment, mocker: Any
    ) -> None:
        """Test that activity returns config from registry when set."""
        from zap_ai.mcp.tool_registry import AgentConfig

        # Create mock registry
        mock_registry = MagicMock()
        mock_config = AgentConfig(
            agent_name="TestAgent",
            prompt="You are a test agent.",
            model="anthropic/claude-sonnet-4-5-20250929",
            max_iterations=30,
        )
        mock_tools = [{"type": "function", "function": {"name": "search"}}]
        mock_registry.get_agent_config.return_value = mock_config
        mock_registry.get_tools_for_agent.return_value = mock_tools

        mocker.patch(
            "zap_ai.activities.tool_execution.get_tool_registry",
            return_value=mock_registry,
        )

        result = await activity_env.run(get_agent_config_activity, "TestAgent")

        assert result.agent_name == "TestAgent"
        assert result.prompt == "You are a test agent."
        assert result.model == "anthropic/claude-sonnet-4-5-20250929"
        assert result.max_iterations == 30
        assert result.tools == mock_tools

    @pytest.mark.asyncio
    async def test_activity_raises_when_agent_not_found(
        self, activity_env: ActivityEnvironment, mocker: Any
    ) -> None:
        """Test that activity raises KeyError when agent not in registry."""
        # Create mock registry that returns None for config
        mock_registry = MagicMock()
        mock_registry.get_agent_config.return_value = None

        mocker.patch(
            "zap_ai.activities.tool_execution.get_tool_registry",
            return_value=mock_registry,
        )

        with pytest.raises(KeyError, match="not found"):
            await activity_env.run(get_agent_config_activity, "UnknownAgent")

    @pytest.mark.asyncio
    async def test_activity_handles_missing_tools_gracefully(
        self, activity_env: ActivityEnvironment, mocker: Any
    ) -> None:
        """Test that activity handles KeyError when getting tools."""
        from zap_ai.mcp.tool_registry import AgentConfig

        # Create mock registry
        mock_registry = MagicMock()
        mock_config = AgentConfig(
            agent_name="TestAgent",
            prompt="Test prompt",
            model="gpt-4o",
            max_iterations=50,
        )
        mock_registry.get_agent_config.return_value = mock_config
        mock_registry.get_tools_for_agent.side_effect = KeyError("Agent not found")

        mocker.patch(
            "zap_ai.activities.tool_execution.get_tool_registry",
            return_value=mock_registry,
        )

        result = await activity_env.run(get_agent_config_activity, "TestAgent")

        # Should return config with empty tools list
        assert result.agent_name == "TestAgent"
        assert result.tools == []
