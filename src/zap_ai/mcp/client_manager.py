"""FastMCP client lifecycle management."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zap_ai.exceptions import ClientConnectionError, ToolNotFoundError

if TYPE_CHECKING:
    from fastmcp import Client

    from zap_ai.core.agent import ZapAgent


@dataclass
class ConnectedClient:
    """
    Wrapper around a connected FastMCP client.

    Tracks connection state and available tools for a single client.

    Attributes:
        client: The FastMCP Client instance.
        tools: Dict mapping tool names to their MCP tool definitions.
        connected: Whether the client is currently connected.
        source: Original source string/path used to create the client.
    """

    client: Client
    source: str
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    connected: bool = False


@dataclass
class AgentToolMapping:
    """
    Maps an agent's tools to their source clients.

    Attributes:
        agent_name: Name of the agent.
        tool_to_client: Dict mapping tool names to ConnectedClient instances.
        all_tools_litellm: All tools for this agent in LiteLLM format.
    """

    agent_name: str
    tool_to_client: dict[str, ConnectedClient] = field(default_factory=dict)
    all_tools_litellm: list[dict[str, Any]] = field(default_factory=list)


class ClientManager:
    """
    Manages FastMCP client connections and tool discovery.

    ClientManager handles the lifecycle of MCP client connections, including:
    - Lazy connection on first use
    - Tool discovery and caching
    - Mapping tools back to their source clients
    - Graceful shutdown of all connections

    The manager is designed to be shared across the Zap instance and
    potentially multiple concurrent workflow executions.

    Example:
        ```python
        manager = ClientManager()

        # Register an agent's clients
        await manager.register_agent(agent)

        # Get the client for a specific tool
        client = manager.get_client_for_tool("MyAgent", "search")
        result = await client.call_tool("search", {"query": "test"})

        # Shutdown all connections
        await manager.disconnect_all()
        ```

    Thread Safety:
        This class uses asyncio locks for connection management but is
        designed for single-threaded async use within Temporal activities.
    """

    def __init__(self) -> None:
        """Initialize the client manager."""
        # Connected clients keyed by their id() to handle same client in multiple agents
        self._connected_clients: dict[int, ConnectedClient] = {}

        # Agent name -> AgentToolMapping
        self._agent_mappings: dict[str, AgentToolMapping] = {}

        # Lock for connection operations
        self._connection_lock = asyncio.Lock()

        # Track if manager has been shut down
        self._shutdown = False

    async def register_agent(self, agent: ZapAgent) -> AgentToolMapping:
        """
        Register an agent and connect its MCP clients.

        Connects all MCP clients for the agent, discovers their tools,
        and builds the tool-to-client mapping.

        Args:
            agent: The ZapAgent to register.

        Returns:
            AgentToolMapping with all discovered tools.

        Raises:
            ClientConnectionError: If any client fails to connect.
            RuntimeError: If manager has been shut down.
        """
        if self._shutdown:
            raise RuntimeError("ClientManager has been shut down")

        async with self._connection_lock:
            # Check if already registered
            if agent.name in self._agent_mappings:
                return self._agent_mappings[agent.name]

            mapping = AgentToolMapping(agent_name=agent.name)

            for client in agent.mcp_clients:
                connected = await self._connect_client(client)

                # Map each tool to this client
                for tool_name, tool_def in connected.tools.items():
                    if tool_name in mapping.tool_to_client:
                        # Tool name collision - use first one, log warning
                        # In production, might want to prefix with client source
                        continue
                    mapping.tool_to_client[tool_name] = connected

            self._agent_mappings[agent.name] = mapping
            return mapping

    async def _connect_client(self, client: Client) -> ConnectedClient:
        """
        Connect a FastMCP client and discover its tools.

        Uses client id() as key to deduplicate same client instance
        used by multiple agents.

        Args:
            client: FastMCP Client to connect.

        Returns:
            ConnectedClient wrapper with discovered tools.

        Raises:
            ClientConnectionError: If connection or tool discovery fails.
        """
        client_id = id(client)

        # Return existing connection if available
        if client_id in self._connected_clients:
            existing = self._connected_clients[client_id]
            if existing.connected:
                return existing

        # Create new connected client wrapper
        # Get source for debugging (FastMCP Client may have different attributes)
        source = getattr(client, "source", None) or str(client)

        connected = ConnectedClient(
            client=client,
            source=str(source),
            connected=False,
        )

        try:
            # FastMCP requires entering async context
            # Note: We need to keep the context alive, so we don't use 'async with'
            # Instead, we manually initialize and track state
            await client.__aenter__()
            connected.connected = True

            # Discover tools
            tools_response = await client.list_tools()

            # Build tool mapping
            # FastMCP returns Tool objects; convert to dicts
            for tool in tools_response:
                tool_dict = self._tool_to_dict(tool)
                connected.tools[tool_dict["name"]] = tool_dict

        except Exception as e:
            # Clean up partial connection
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
            raise ClientConnectionError(f"Failed to connect to MCP client '{source}': {e}") from e

        self._connected_clients[client_id] = connected
        return connected

    def _tool_to_dict(self, tool: Any) -> dict[str, Any]:
        """
        Convert a FastMCP Tool object to a dict.

        FastMCP's Tool class has name, description, and inputSchema attributes.
        This handles both dict and object formats.

        Args:
            tool: FastMCP Tool object or dict.

        Returns:
            Dict with name, description, inputSchema keys.
        """
        if isinstance(tool, dict):
            return tool

        # FastMCP Tool object
        return {
            "name": getattr(tool, "name", "unknown"),
            "description": getattr(tool, "description", ""),
            "inputSchema": getattr(tool, "inputSchema", {}),
        }

    def get_client_for_tool(self, agent_name: str, tool_name: str) -> Client:
        """
        Get the FastMCP client that provides a specific tool.

        Args:
            agent_name: Name of the agent.
            tool_name: Name of the tool to find.

        Returns:
            The FastMCP Client instance for the tool.

        Raises:
            ToolNotFoundError: If tool not found for this agent.
            KeyError: If agent not registered.
        """
        if agent_name not in self._agent_mappings:
            raise KeyError(f"Agent '{agent_name}' not registered with ClientManager")

        mapping = self._agent_mappings[agent_name]

        if tool_name not in mapping.tool_to_client:
            available = list(mapping.tool_to_client.keys())
            raise ToolNotFoundError(
                f"Tool '{tool_name}' not found for agent '{agent_name}'. "
                f"Available tools: {available}"
            )

        return mapping.tool_to_client[tool_name].client

    def get_tools_for_agent(self, agent_name: str) -> list[dict[str, Any]]:
        """
        Get all MCP tool definitions for an agent.

        Returns tools in MCP format (not LiteLLM format).
        Use ToolRegistry for LiteLLM-formatted tools.

        Args:
            agent_name: Name of the agent.

        Returns:
            List of MCP tool definition dicts.

        Raises:
            KeyError: If agent not registered.
        """
        if agent_name not in self._agent_mappings:
            raise KeyError(f"Agent '{agent_name}' not registered with ClientManager")

        mapping = self._agent_mappings[agent_name]
        tools: list[dict[str, Any]] = []

        # Collect unique tools (avoid duplicates from shared clients)
        seen_tools: set[str] = set()
        for tool_name, connected in mapping.tool_to_client.items():
            if tool_name not in seen_tools:
                tools.append(connected.tools[tool_name])
                seen_tools.add(tool_name)

        return tools

    def is_agent_registered(self, agent_name: str) -> bool:
        """Check if an agent has been registered."""
        return agent_name in self._agent_mappings

    def list_registered_agents(self) -> list[str]:
        """Return list of registered agent names."""
        return list(self._agent_mappings.keys())

    async def disconnect_all(self) -> None:
        """
        Disconnect all MCP clients and clean up resources.

        Should be called during Zap shutdown. Safe to call multiple times.
        """
        async with self._connection_lock:
            self._shutdown = True

            for connected in self._connected_clients.values():
                if connected.connected:
                    try:
                        await connected.client.__aexit__(None, None, None)
                        connected.connected = False
                    except Exception:
                        # Log but don't raise during shutdown
                        pass

            self._connected_clients.clear()
            self._agent_mappings.clear()

    async def reconnect_agent(self, agent: ZapAgent) -> AgentToolMapping:
        """
        Force reconnection of an agent's clients.

        Useful for recovering from connection failures.

        Args:
            agent: The agent to reconnect.

        Returns:
            Fresh AgentToolMapping.
        """
        async with self._connection_lock:
            # Remove existing mapping
            if agent.name in self._agent_mappings:
                del self._agent_mappings[agent.name]

            # Remove connected clients for this agent
            for client in agent.mcp_clients:
                client_id = id(client)
                if client_id in self._connected_clients:
                    connected = self._connected_clients[client_id]
                    if connected.connected:
                        try:
                            await connected.client.__aexit__(None, None, None)
                        except Exception:
                            pass
                    del self._connected_clients[client_id]

        # Re-register
        return await self.register_agent(agent)
