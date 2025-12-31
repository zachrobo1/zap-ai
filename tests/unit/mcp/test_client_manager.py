"""Tests for MCP client manager."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from zap_ai.mcp.client_manager import (
    AgentToolMapping,
    ClientConnectionError,
    ClientManager,
    ConnectedClient,
    ToolNotFoundError,
)


class TestConnectedClient:
    """Tests for ConnectedClient dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        mock_client = MagicMock()
        connected = ConnectedClient(client=mock_client, source="test")

        assert connected.client is mock_client
        assert connected.source == "test"
        assert connected.tools == {}
        assert connected.connected is False


class TestAgentToolMapping:
    """Tests for AgentToolMapping dataclass."""

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        mapping = AgentToolMapping(agent_name="TestAgent")

        assert mapping.agent_name == "TestAgent"
        assert mapping.tool_to_client == {}
        assert mapping.all_tools_litellm == []


class TestClientManager:
    """Tests for ClientManager class."""

    def test_init(self) -> None:
        """Test initialization."""
        manager = ClientManager()

        assert manager._connected_clients == {}
        assert manager._agent_mappings == {}
        assert manager._shutdown is False

    def test_is_agent_registered_false(self) -> None:
        """Test is_agent_registered returns False for unknown agent."""
        manager = ClientManager()

        assert manager.is_agent_registered("UnknownAgent") is False

    def test_list_registered_agents_empty(self) -> None:
        """Test list_registered_agents when empty."""
        manager = ClientManager()

        assert manager.list_registered_agents() == []

    @pytest.mark.asyncio
    async def test_register_agent_connects_client(self) -> None:
        """Test that register_agent connects the client."""
        manager = ClientManager()

        # Create mock client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.mcp_clients = [mock_client]

        mapping = await manager.register_agent(mock_agent)

        assert mapping.agent_name == "TestAgent"
        assert manager.is_agent_registered("TestAgent")
        mock_client.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_agent_discovers_tools(self) -> None:
        """Test that register_agent discovers tools from client."""
        manager = ClientManager()

        # Create mock tool
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "Search tool"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        # Create mock client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.mcp_clients = [mock_client]

        mapping = await manager.register_agent(mock_agent)

        assert "search" in mapping.tool_to_client
        tools = manager.get_tools_for_agent("TestAgent")
        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    @pytest.mark.asyncio
    async def test_register_agent_returns_cached_mapping(self) -> None:
        """Test that registering same agent returns cached mapping."""
        manager = ClientManager()

        # Create mock client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.mcp_clients = [mock_client]

        mapping1 = await manager.register_agent(mock_agent)
        mapping2 = await manager.register_agent(mock_agent)

        assert mapping1 is mapping2
        # Should only connect once
        assert mock_client.__aenter__.call_count == 1

    @pytest.mark.asyncio
    async def test_register_agent_raises_after_shutdown(self) -> None:
        """Test that register_agent raises after shutdown."""
        manager = ClientManager()
        await manager.disconnect_all()

        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.mcp_clients = []

        with pytest.raises(RuntimeError, match="shut down"):
            await manager.register_agent(mock_agent)

    @pytest.mark.asyncio
    async def test_connect_client_raises_on_failure(self) -> None:
        """Test that connection failure raises ClientConnectionError."""
        manager = ClientManager()

        # Create mock client that fails
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
        mock_client.__aexit__ = AsyncMock()

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.mcp_clients = [mock_client]

        with pytest.raises(ClientConnectionError, match="Connection failed"):
            await manager.register_agent(mock_agent)

    def test_get_client_for_tool_success(self) -> None:
        """Test getting client for a known tool."""
        manager = ClientManager()

        # Manually set up mapping
        mock_client = MagicMock()
        connected = ConnectedClient(
            client=mock_client,
            source="test",
            tools={"search": {"name": "search"}},
            connected=True,
        )
        mapping = AgentToolMapping(agent_name="TestAgent")
        mapping.tool_to_client["search"] = connected
        manager._agent_mappings["TestAgent"] = mapping

        result = manager.get_client_for_tool("TestAgent", "search")

        assert result is mock_client

    def test_get_client_for_tool_raises_for_unknown_agent(self) -> None:
        """Test error when agent not registered."""
        manager = ClientManager()

        with pytest.raises(KeyError, match="not registered"):
            manager.get_client_for_tool("UnknownAgent", "search")

    def test_get_client_for_tool_raises_for_unknown_tool(self) -> None:
        """Test error when tool not found."""
        manager = ClientManager()
        manager._agent_mappings["TestAgent"] = AgentToolMapping(agent_name="TestAgent")

        with pytest.raises(ToolNotFoundError, match="not found"):
            manager.get_client_for_tool("TestAgent", "unknown_tool")

    def test_get_tools_for_agent_success(self) -> None:
        """Test getting tools for an agent."""
        manager = ClientManager()

        # Manually set up mapping
        mock_client = MagicMock()
        tool_def = {"name": "search", "description": "Search"}
        connected = ConnectedClient(
            client=mock_client,
            source="test",
            tools={"search": tool_def},
            connected=True,
        )
        mapping = AgentToolMapping(agent_name="TestAgent")
        mapping.tool_to_client["search"] = connected
        manager._agent_mappings["TestAgent"] = mapping

        tools = manager.get_tools_for_agent("TestAgent")

        assert len(tools) == 1
        assert tools[0]["name"] == "search"

    def test_get_tools_for_agent_raises_for_unknown_agent(self) -> None:
        """Test error when agent not registered."""
        manager = ClientManager()

        with pytest.raises(KeyError, match="not registered"):
            manager.get_tools_for_agent("UnknownAgent")

    @pytest.mark.asyncio
    async def test_disconnect_all(self) -> None:
        """Test disconnecting all clients."""
        manager = ClientManager()

        # Create mock client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.mcp_clients = [mock_client]

        await manager.register_agent(mock_agent)
        await manager.disconnect_all()

        assert manager._shutdown is True
        assert manager._connected_clients == {}
        assert manager._agent_mappings == {}
        mock_client.__aexit__.assert_called()

    @pytest.mark.asyncio
    async def test_disconnect_all_is_idempotent(self) -> None:
        """Test that disconnect_all can be called multiple times."""
        manager = ClientManager()
        await manager.disconnect_all()
        await manager.disconnect_all()  # Should not raise

        assert manager._shutdown is True

    @pytest.mark.asyncio
    async def test_reconnect_agent(self) -> None:
        """Test reconnecting an agent."""
        manager = ClientManager()

        # Create mock client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock()
        mock_client.__aexit__ = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[])

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.mcp_clients = [mock_client]

        # First registration
        await manager.register_agent(mock_agent)
        assert mock_client.__aenter__.call_count == 1

        # Reconnect
        await manager.reconnect_agent(mock_agent)
        assert mock_client.__aenter__.call_count == 2

    @pytest.mark.asyncio
    async def test_tool_to_dict_handles_dict(self) -> None:
        """Test _tool_to_dict with dict input."""
        manager = ClientManager()
        tool_dict = {"name": "test", "description": "Test"}

        result = manager._tool_to_dict(tool_dict)

        assert result == tool_dict

    @pytest.mark.asyncio
    async def test_tool_to_dict_handles_object(self) -> None:
        """Test _tool_to_dict with object input."""
        manager = ClientManager()

        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object"}

        result = manager._tool_to_dict(mock_tool)

        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool"
        assert result["inputSchema"] == {"type": "object"}

    @pytest.mark.asyncio
    async def test_handles_tool_name_collision(self) -> None:
        """Test that first tool wins on name collision."""
        manager = ClientManager()

        # Create two mock clients with same tool name
        mock_tool = MagicMock()
        mock_tool.name = "search"
        mock_tool.description = "First search"
        mock_tool.inputSchema = {}

        mock_client1 = AsyncMock()
        mock_client1.__aenter__ = AsyncMock()
        mock_client1.__aexit__ = AsyncMock()
        mock_client1.list_tools = AsyncMock(return_value=[mock_tool])

        mock_tool2 = MagicMock()
        mock_tool2.name = "search"
        mock_tool2.description = "Second search"
        mock_tool2.inputSchema = {}

        mock_client2 = AsyncMock()
        mock_client2.__aenter__ = AsyncMock()
        mock_client2.__aexit__ = AsyncMock()
        mock_client2.list_tools = AsyncMock(return_value=[mock_tool2])

        # Create mock agent with both clients
        mock_agent = MagicMock()
        mock_agent.name = "TestAgent"
        mock_agent.mcp_clients = [mock_client1, mock_client2]

        await manager.register_agent(mock_agent)

        # First one should win
        tools = manager.get_tools_for_agent("TestAgent")
        assert len(tools) == 1
        assert tools[0]["description"] == "First search"
