"""Tests for MCP tool registry."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from zap_ai.mcp.tool_registry import AgentConfig, AgentTools, ToolRegistry


class TestAgentTools:
    """Tests for AgentTools dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        tools = AgentTools(agent_name="TestAgent")

        assert tools.agent_name == "TestAgent"
        assert tools.mcp_tools == []
        assert tools.message_agent_tool is None

    def test_all_tools_without_message_agent(self) -> None:
        """Test all_tools returns only mcp_tools when no message_agent."""
        mcp_tools = [
            {"type": "function", "function": {"name": "tool1"}},
            {"type": "function", "function": {"name": "tool2"}},
        ]
        tools = AgentTools(agent_name="TestAgent", mcp_tools=mcp_tools)

        assert tools.all_tools == mcp_tools

    def test_all_tools_with_message_agent(self) -> None:
        """Test all_tools includes message_agent tool."""
        mcp_tools = [{"type": "function", "function": {"name": "tool1"}}]
        message_tool = {"type": "function", "function": {"name": "message_agent"}}
        tools = AgentTools(
            agent_name="TestAgent",
            mcp_tools=mcp_tools,
            message_agent_tool=message_tool,
        )

        all_tools = tools.all_tools
        assert len(all_tools) == 2
        assert message_tool in all_tools

    def test_tool_names(self) -> None:
        """Test tool_names property."""
        mcp_tools = [
            {"type": "function", "function": {"name": "search"}},
            {"type": "function", "function": {"name": "calculate"}},
        ]
        tools = AgentTools(agent_name="TestAgent", mcp_tools=mcp_tools)

        assert tools.tool_names == ["search", "calculate"]

    def test_has_tool(self) -> None:
        """Test has_tool method."""
        mcp_tools = [{"type": "function", "function": {"name": "search"}}]
        tools = AgentTools(agent_name="TestAgent", mcp_tools=mcp_tools)

        assert tools.has_tool("search") is True
        assert tools.has_tool("unknown") is False

    def test_can_message_sub_agents_true(self) -> None:
        """Test can_message_sub_agents when message_agent tool present."""
        tools = AgentTools(
            agent_name="TestAgent",
            message_agent_tool={"type": "function", "function": {"name": "message_agent"}},
        )

        assert tools.can_message_sub_agents is True

    def test_can_message_sub_agents_false(self) -> None:
        """Test can_message_sub_agents when no message_agent tool."""
        tools = AgentTools(agent_name="TestAgent")

        assert tools.can_message_sub_agents is False


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_init_default_client_manager(self) -> None:
        """Test initialization creates default ClientManager."""
        registry = ToolRegistry()

        assert registry._client_manager is not None
        assert registry._agent_tools == {}
        assert registry._initialized is False

    def test_init_with_client_manager(self) -> None:
        """Test initialization with provided ClientManager."""
        mock_manager = MagicMock()
        registry = ToolRegistry(client_manager=mock_manager)

        assert registry._client_manager is mock_manager

    def test_client_manager_property(self) -> None:
        """Test client_manager property."""
        registry = ToolRegistry()

        assert registry.client_manager is registry._client_manager

    def test_message_agent_tool_name_constant(self) -> None:
        """Test MESSAGE_AGENT_TOOL_NAME constant."""
        assert ToolRegistry.MESSAGE_AGENT_TOOL_NAME == "message_agent"

    @pytest.mark.asyncio
    async def test_register_agents_basic(self) -> None:
        """Test registering agents."""
        mock_manager = MagicMock()
        mock_manager.register_agent = AsyncMock()
        mock_manager.get_tools_for_agent = MagicMock(return_value=[])

        registry = ToolRegistry(client_manager=mock_manager)

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.sub_agents = []
        mock_agent.mcp_clients = []

        await registry.register_agents([mock_agent], {"TestAgent": mock_agent})

        assert registry._initialized is True
        assert "TestAgent" in registry._agent_tools

    @pytest.mark.asyncio
    async def test_register_agents_creates_message_agent_tool(self) -> None:
        """Test that agents with sub-agents get message_agent tool."""
        mock_manager = MagicMock()
        mock_manager.register_agent = AsyncMock()
        mock_manager.get_tools_for_agent = MagicMock(return_value=[])

        registry = ToolRegistry(client_manager=mock_manager)

        # Create mock agents
        main_agent = MagicMock()
        main_agent.name = "MainAgent"
        main_agent.sub_agents = ["HelperAgent"]
        main_agent.mcp_clients = []

        helper_agent = MagicMock()
        helper_agent.name = "HelperAgent"
        helper_agent.sub_agents = []
        helper_agent.mcp_clients = []
        helper_agent.discovery_prompt = "I help with tasks"

        agents = [main_agent, helper_agent]
        agent_map = {a.name: a for a in agents}

        await registry.register_agents(agents, agent_map)

        # MainAgent should have message_agent tool
        main_tools = registry._agent_tools["MainAgent"]
        assert main_tools.message_agent_tool is not None
        assert main_tools.message_agent_tool["function"]["name"] == "message_agent"

        # HelperAgent should not have message_agent tool
        helper_tools = registry._agent_tools["HelperAgent"]
        assert helper_tools.message_agent_tool is None

    @pytest.mark.asyncio
    async def test_register_agents_raises_if_already_initialized(self) -> None:
        """Test that registering again raises error."""
        mock_manager = MagicMock()
        mock_manager.register_agent = AsyncMock()
        mock_manager.get_tools_for_agent = MagicMock(return_value=[])

        registry = ToolRegistry(client_manager=mock_manager)
        registry._initialized = True

        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"

        with pytest.raises(RuntimeError, match="already initialized"):
            await registry.register_agents([mock_agent], {})

    def test_get_tools_for_agent_success(self) -> None:
        """Test getting tools for a registered agent."""
        registry = ToolRegistry()
        mcp_tools = [{"type": "function", "function": {"name": "search"}}]
        registry._agent_tools["TestAgent"] = AgentTools(agent_name="TestAgent", mcp_tools=mcp_tools)

        tools = registry.get_tools_for_agent("TestAgent")

        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "search"

    def test_get_tools_for_agent_raises_for_unknown(self) -> None:
        """Test error when agent not registered."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_tools_for_agent("UnknownAgent")

    def test_get_agent_tools_success(self) -> None:
        """Test getting AgentTools object."""
        registry = ToolRegistry()
        agent_tools = AgentTools(agent_name="TestAgent")
        registry._agent_tools["TestAgent"] = agent_tools

        result = registry.get_agent_tools("TestAgent")

        assert result is agent_tools

    def test_get_agent_tools_raises_for_unknown(self) -> None:
        """Test error when agent not registered."""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get_agent_tools("UnknownAgent")

    def test_has_message_agent_tool_true(self) -> None:
        """Test has_message_agent_tool returns True when present."""
        registry = ToolRegistry()
        registry._agent_tools["TestAgent"] = AgentTools(
            agent_name="TestAgent",
            message_agent_tool={"type": "function", "function": {"name": "message_agent"}},
        )

        assert registry.has_message_agent_tool("TestAgent") is True

    def test_has_message_agent_tool_false(self) -> None:
        """Test has_message_agent_tool returns False when not present."""
        registry = ToolRegistry()
        registry._agent_tools["TestAgent"] = AgentTools(agent_name="TestAgent")

        assert registry.has_message_agent_tool("TestAgent") is False

    def test_has_message_agent_tool_unknown_agent(self) -> None:
        """Test has_message_agent_tool returns False for unknown agent."""
        registry = ToolRegistry()

        assert registry.has_message_agent_tool("UnknownAgent") is False

    def test_is_message_agent_tool(self) -> None:
        """Test is_message_agent_tool method."""
        registry = ToolRegistry()

        assert registry.is_message_agent_tool("message_agent") is True
        assert registry.is_message_agent_tool("other_tool") is False

    def test_get_client_for_tool_success(self) -> None:
        """Test getting client for a tool."""
        mock_client = MagicMock()
        mock_manager = MagicMock()
        mock_manager.get_client_for_tool = MagicMock(return_value=mock_client)

        registry = ToolRegistry(client_manager=mock_manager)
        registry._agent_tools["TestAgent"] = AgentTools(agent_name="TestAgent")

        result = registry.get_client_for_tool("TestAgent", "search")

        assert result is mock_client
        mock_manager.get_client_for_tool.assert_called_once_with("TestAgent", "search")

    def test_get_client_for_tool_raises_for_message_agent(self) -> None:
        """Test error when trying to get client for message_agent."""
        registry = ToolRegistry()

        with pytest.raises(ValueError, match="not an MCP tool"):
            registry.get_client_for_tool("TestAgent", "message_agent")

    def test_list_agents(self) -> None:
        """Test listing registered agents."""
        registry = ToolRegistry()
        registry._agent_tools["Agent1"] = AgentTools(agent_name="Agent1")
        registry._agent_tools["Agent2"] = AgentTools(agent_name="Agent2")

        agents = registry.list_agents()

        assert set(agents) == {"Agent1", "Agent2"}

    def test_get_tool_count(self) -> None:
        """Test getting tool count."""
        registry = ToolRegistry()
        mcp_tools = [
            {"type": "function", "function": {"name": "tool1"}},
            {"type": "function", "function": {"name": "tool2"}},
        ]
        registry._agent_tools["TestAgent"] = AgentTools(agent_name="TestAgent", mcp_tools=mcp_tools)

        assert registry.get_tool_count("TestAgent") == 2

    def test_get_tool_count_unknown_agent(self) -> None:
        """Test get_tool_count returns 0 for unknown agent."""
        registry = ToolRegistry()

        assert registry.get_tool_count("UnknownAgent") == 0

    def test_get_tool_count_with_message_agent(self) -> None:
        """Test get_tool_count includes message_agent tool."""
        registry = ToolRegistry()
        mcp_tools = [{"type": "function", "function": {"name": "tool1"}}]
        message_tool = {"type": "function", "function": {"name": "message_agent"}}
        registry._agent_tools["TestAgent"] = AgentTools(
            agent_name="TestAgent",
            mcp_tools=mcp_tools,
            message_agent_tool=message_tool,
        )

        assert registry.get_tool_count("TestAgent") == 2

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        """Test shutting down the registry."""
        mock_manager = MagicMock()
        mock_manager.disconnect_all = AsyncMock()

        registry = ToolRegistry(client_manager=mock_manager)
        registry._initialized = True
        registry._agent_tools["TestAgent"] = AgentTools(agent_name="TestAgent")

        await registry.shutdown()

        mock_manager.disconnect_all.assert_called_once()
        assert registry._agent_tools == {}
        assert registry._initialized is False

    @pytest.mark.asyncio
    async def test_build_sub_agent_list(self) -> None:
        """Test _build_sub_agent_list method."""
        registry = ToolRegistry()

        main_agent = MagicMock()
        main_agent.name = "MainAgent"
        main_agent.sub_agents = ["Helper1", "Helper2"]

        helper1 = MagicMock()
        helper1.name = "Helper1"
        helper1.discovery_prompt = "Helps with task 1"

        helper2 = MagicMock()
        helper2.name = "Helper2"
        helper2.discovery_prompt = None

        agent_map = {
            "MainAgent": main_agent,
            "Helper1": helper1,
            "Helper2": helper2,
        }

        result = registry._build_sub_agent_list(main_agent, agent_map)

        assert result == [
            ("Helper1", "Helps with task 1"),
            ("Helper2", None),
        ]

    @pytest.mark.asyncio
    async def test_build_sub_agent_list_handles_missing_agent(self) -> None:
        """Test _build_sub_agent_list handles missing sub-agent gracefully."""
        registry = ToolRegistry()

        main_agent = MagicMock()
        main_agent.name = "MainAgent"
        main_agent.sub_agents = ["MissingAgent"]

        agent_map = {"MainAgent": main_agent}

        result = registry._build_sub_agent_list(main_agent, agent_map)

        # Should still include the agent name but with None prompt
        assert result == [("MissingAgent", None)]

    @pytest.mark.asyncio
    async def test_converts_mcp_tools_to_litellm(self) -> None:
        """Test that MCP tools are converted to LiteLLM format."""
        mock_manager = MagicMock()
        mock_manager.register_agent = AsyncMock()
        mock_manager.get_tools_for_agent = MagicMock(
            return_value=[
                {
                    "name": "search",
                    "description": "Search tool",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ]
        )

        registry = ToolRegistry(client_manager=mock_manager)

        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.sub_agents = []
        mock_agent.mcp_clients = []

        await registry.register_agents([mock_agent], {"TestAgent": mock_agent})

        tools = registry.get_tools_for_agent("TestAgent")

        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "search"
        assert "parameters" in tools[0]["function"]


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_create_agent_config(self) -> None:
        """Test creating an AgentConfig."""
        config = AgentConfig(
            agent_name="TestAgent",
            prompt="You are a test agent.",
            model="gpt-4o",
            max_iterations=25,
        )

        assert config.agent_name == "TestAgent"
        assert config.prompt == "You are a test agent."
        assert config.model == "gpt-4o"
        assert config.max_iterations == 25


class TestToolRegistryAgentConfig:
    """Tests for ToolRegistry agent config functionality."""

    @pytest.mark.asyncio
    async def test_register_stores_agent_config(self) -> None:
        """Test that registering an agent stores its config."""
        mock_manager = MagicMock()
        mock_manager.register_agent = AsyncMock()
        mock_manager.get_tools_for_agent = MagicMock(return_value=[])

        registry = ToolRegistry(client_manager=mock_manager)

        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.prompt = "You are a helpful assistant."
        mock_agent.model = "anthropic/claude-sonnet-4-5-20250929"
        mock_agent.max_iterations = 30
        mock_agent.sub_agents = []
        mock_agent.mcp_clients = []

        await registry.register_agents([mock_agent], {"TestAgent": mock_agent})

        agent_tools = registry._agent_tools["TestAgent"]
        assert agent_tools.config is not None
        assert agent_tools.config.agent_name == "TestAgent"
        assert agent_tools.config.prompt == "You are a helpful assistant."
        assert agent_tools.config.model == "anthropic/claude-sonnet-4-5-20250929"
        assert agent_tools.config.max_iterations == 30

    def test_get_agent_config_success(self) -> None:
        """Test getting agent config for registered agent."""
        registry = ToolRegistry()
        config = AgentConfig(
            agent_name="TestAgent",
            prompt="Test prompt",
            model="gpt-4o",
            max_iterations=50,
        )
        registry._agent_tools["TestAgent"] = AgentTools(
            agent_name="TestAgent",
            config=config,
        )

        result = registry.get_agent_config("TestAgent")

        assert result is not None
        assert result.agent_name == "TestAgent"
        assert result.prompt == "Test prompt"
        assert result.model == "gpt-4o"
        assert result.max_iterations == 50

    def test_get_agent_config_returns_none_for_unknown(self) -> None:
        """Test get_agent_config returns None for unknown agent."""
        registry = ToolRegistry()

        result = registry.get_agent_config("UnknownAgent")

        assert result is None

    def test_get_agent_config_returns_none_if_no_config(self) -> None:
        """Test get_agent_config returns None if agent has no config."""
        registry = ToolRegistry()
        registry._agent_tools["TestAgent"] = AgentTools(
            agent_name="TestAgent",
            config=None,
        )

        result = registry.get_agent_config("TestAgent")

        assert result is None
