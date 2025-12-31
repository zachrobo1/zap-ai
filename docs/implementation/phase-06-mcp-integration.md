# Phase 6: MCP Integration

**Goal:** Enable tool discovery and schema conversion for FastMCP clients.

**Dependencies:** Phase 1 (Foundation) must be complete.

**Overview:**
This phase implements the bridge between FastMCP (MCP protocol) and LiteLLM (OpenAI function calling format). The key components are:
1. **Schema Converter** - Transforms MCP tool schemas to LiteLLM format
2. **Client Manager** - Handles FastMCP client lifecycle and connection pooling
3. **Tool Registry** - Discovers, caches, and provides tools for agents

---

## Task 6.1: Implement MCP Schema Converter

**File:** `src/zap_ai/mcp/schema_converter.py`

The schema converter handles two critical transformations:
1. Converting MCP tool definitions to LiteLLM/OpenAI function format
2. Generating the special `transfer_to_agent` tool for sub-agent delegation

**Schema Format Comparison:**

```python
# MCP Tool Format (from FastMCP client.list_tools())
{
    "name": "search_database",
    "description": "Search the database for records",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results", "default": 10}
        },
        "required": ["query"]
    }
}

# LiteLLM/OpenAI Function Format (for tools parameter)
{
    "type": "function",
    "function": {
        "name": "search_database",
        "description": "Search the database for records",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10}
            },
            "required": ["query"]
        }
    }
}
```

**Implementation:**

```python
"""MCP to LiteLLM schema conversion utilities."""

from __future__ import annotations

from typing import Any


class SchemaConversionError(Exception):
    """Raised when schema conversion fails."""
    pass


def mcp_tool_to_litellm(mcp_tool: dict[str, Any]) -> dict[str, Any]:
    """
    Convert an MCP tool definition to LiteLLM/OpenAI function calling format.

    The MCP protocol uses `inputSchema` while OpenAI uses `parameters`.
    Both use JSON Schema internally, so the conversion is structural.

    Args:
        mcp_tool: MCP tool definition dict with keys:
            - name (str): Tool name
            - description (str, optional): Tool description
            - inputSchema (dict, optional): JSON Schema for parameters

    Returns:
        LiteLLM-compatible tool definition dict.

    Raises:
        SchemaConversionError: If required fields are missing or invalid.

    Example:
        ```python
        mcp_tool = {
            "name": "get_weather",
            "description": "Get weather for a city",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
        litellm_tool = mcp_tool_to_litellm(mcp_tool)
        # Returns: {"type": "function", "function": {...}}
        ```
    """
    # Validate required field
    if "name" not in mcp_tool:
        raise SchemaConversionError("MCP tool missing required 'name' field")

    name = mcp_tool["name"]
    if not isinstance(name, str) or not name.strip():
        raise SchemaConversionError(f"Invalid tool name: {name!r}")

    # Extract and validate description
    description = mcp_tool.get("description", "")
    if not isinstance(description, str):
        description = str(description)

    # Extract input schema, defaulting to empty object schema
    input_schema = mcp_tool.get("inputSchema", {
        "type": "object",
        "properties": {},
        "required": []
    })

    # Ensure schema has required structure
    if not isinstance(input_schema, dict):
        raise SchemaConversionError(
            f"Tool '{name}' has invalid inputSchema: expected dict, got {type(input_schema).__name__}"
        )

    # Ensure type is "object" (required by OpenAI)
    if input_schema.get("type") != "object":
        # Wrap non-object schemas
        input_schema = {
            "type": "object",
            "properties": {"value": input_schema},
            "required": ["value"]
        }

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        }
    }


def mcp_tools_to_litellm(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert a list of MCP tools to LiteLLM format.

    Args:
        mcp_tools: List of MCP tool definitions.

    Returns:
        List of LiteLLM-compatible tool definitions.

    Raises:
        SchemaConversionError: If any tool conversion fails.
    """
    return [mcp_tool_to_litellm(tool) for tool in mcp_tools]


def create_message_agent_tool(
    available_agents: list[tuple[str, str | None]],
) -> dict[str, Any]:
    """
    Create the message_agent tool for multi-turn sub-agent conversations.

    This tool allows an agent to have conversations with its configured sub-agents.
    Unlike a "transfer" pattern, the parent agent stays in control and can:
    - Start new conversations with sub-agents
    - Continue existing conversations with follow-up messages
    - Have multiple concurrent conversations with different sub-agents

    The sub-agent is an assistant, not a replacement. The parent agent synthesizes
    results and decides next steps.

    Args:
        available_agents: List of (agent_name, discovery_prompt) tuples.
            - agent_name: The sub-agent's name (used in enum)
            - discovery_prompt: Description of what the sub-agent does.
              If None, the agent won't appear in the description but
              will still be callable.

    Returns:
        LiteLLM-compatible tool definition for message_agent.

    Raises:
        ValueError: If available_agents is empty.

    Example:
        ```python
        agents = [
            ("ResearchAgent", "Use for web research and data gathering"),
            ("WriterAgent", "Use for drafting and editing text"),
        ]
        tool = create_message_agent_tool(agents)
        # Creates tool with agent_name enum and conversation_id support
        ```
    """
    if not available_agents:
        raise ValueError("Cannot create message_agent tool with no available agents")

    # Build agent descriptions for the tool description
    agent_descriptions: list[str] = []
    agent_names: list[str] = []

    for agent_name, discovery_prompt in available_agents:
        agent_names.append(agent_name)
        if discovery_prompt:
            agent_descriptions.append(f"  - **{agent_name}**: {discovery_prompt}")
        else:
            agent_descriptions.append(f"  - **{agent_name}**: (no description)")

    description_text = "\n".join(agent_descriptions)

    return {
        "type": "function",
        "function": {
            "name": "message_agent",
            "description": (
                "Send a message to a sub-agent and receive their response. "
                "Use this to collaborate with specialized agents on parts of your task.\n\n"
                "You remain in control - sub-agents are assistants that help with specific "
                "subtasks. You can have multi-turn conversations by reusing the conversation_id "
                "returned in the response.\n\n"
                "Available agents:\n"
                f"{description_text}\n\n"
                "Usage patterns:\n"
                "- Start new conversation: provide agent_name and message\n"
                "- Continue conversation: provide conversation_id and message\n"
                "- You can have multiple concurrent conversations with different agents"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {
                        "type": "string",
                        "enum": agent_names,
                        "description": (
                            "The name of the sub-agent to message. Required when starting "
                            "a new conversation. Optional when continuing an existing conversation."
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "The message to send to the sub-agent. Be specific and include "
                            "relevant context. For follow-ups, you can reference prior messages "
                            "in the conversation."
                        ),
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": (
                            "Optional. The conversation_id from a previous message_agent response. "
                            "Provide this to continue an existing conversation with a sub-agent. "
                            "If omitted, a new conversation is started."
                        ),
                    },
                },
                "required": ["message"],
            },
        }
    }


def validate_litellm_tool(tool: dict[str, Any]) -> bool:
    """
    Validate that a tool definition conforms to LiteLLM/OpenAI format.

    Args:
        tool: Tool definition to validate.

    Returns:
        True if valid.

    Raises:
        SchemaConversionError: If validation fails.
    """
    if not isinstance(tool, dict):
        raise SchemaConversionError(f"Tool must be dict, got {type(tool).__name__}")

    if tool.get("type") != "function":
        raise SchemaConversionError(f"Tool type must be 'function', got {tool.get('type')!r}")

    function = tool.get("function")
    if not isinstance(function, dict):
        raise SchemaConversionError("Tool missing 'function' dict")

    if "name" not in function:
        raise SchemaConversionError("Tool function missing 'name'")

    if "parameters" in function:
        params = function["parameters"]
        if not isinstance(params, dict):
            raise SchemaConversionError("Tool parameters must be dict")
        if params.get("type") != "object":
            raise SchemaConversionError("Tool parameters type must be 'object'")

    return True
```

**Checklist:**
- [ ] Create `src/zap_ai/mcp/schema_converter.py`
- [ ] Implement `SchemaConversionError` exception
- [ ] Implement `mcp_tool_to_litellm()` with full validation
- [ ] Implement `mcp_tools_to_litellm()` batch converter
- [ ] Implement `create_message_agent_tool()` for multi-turn sub-agent conversations
- [ ] Implement `validate_litellm_tool()` for validation
- [ ] Add comprehensive docstrings with examples
- [ ] Handle edge cases: missing fields, non-object schemas, empty descriptions

---

## Task 6.2: Implement Client Manager

**File:** `src/zap_ai/mcp/client_manager.py`

The ClientManager handles the lifecycle of FastMCP client connections. It:
1. Connects clients lazily on first use
2. Caches connected clients to avoid reconnection overhead
3. Maps tools to their source clients for execution
4. Handles graceful disconnection

**Key Considerations:**
- FastMCP Client requires async context manager (`async with client:`)
- Multiple agents may share the same MCP client instance
- Tool names must be mapped back to their source client for execution
- Connection errors should be handled gracefully

**Implementation:**

```python
"""FastMCP client lifecycle management."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import Client
    from zap_ai.core.agent import ZapAgent


class ClientConnectionError(Exception):
    """Raised when client connection fails."""
    pass


class ToolNotFoundError(Exception):
    """Raised when a tool cannot be found in any client."""
    pass


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
        source = getattr(client, 'source', None) or str(client)

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
            raise ClientConnectionError(
                f"Failed to connect to MCP client '{source}': {e}"
            ) from e

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
```

**Checklist:**
- [ ] Create `src/zap_ai/mcp/client_manager.py`
- [ ] Implement `ClientConnectionError` and `ToolNotFoundError` exceptions
- [ ] Implement `ConnectedClient` dataclass
- [ ] Implement `AgentToolMapping` dataclass
- [ ] Implement `ClientManager` class with `__init__`
- [ ] Implement `register_agent()` - connects clients and discovers tools
- [ ] Implement `_connect_client()` - handles FastMCP client connection
- [ ] Implement `_tool_to_dict()` - converts FastMCP Tool objects
- [ ] Implement `get_client_for_tool()` - maps tool name to client
- [ ] Implement `get_tools_for_agent()` - returns MCP tool definitions
- [ ] Implement `disconnect_all()` - graceful shutdown
- [ ] Implement `reconnect_agent()` - for error recovery
- [ ] Add asyncio lock for thread safety
- [ ] Handle client deduplication (same client in multiple agents)
- [ ] Add comprehensive docstrings

---

## Task 6.3: Implement Tool Registry

**File:** `src/zap_ai/mcp/tool_registry.py`

The ToolRegistry is the high-level interface for tool management. It:
1. Uses ClientManager for MCP client connections
2. Converts MCP tools to LiteLLM format
3. Adds the `transfer_to_agent` tool when agents have sub-agents
4. Provides a clean interface for the workflow to get tools

**Implementation:**

```python
"""Tool registry for managing agent tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zap_ai.mcp.client_manager import ClientManager
from zap_ai.mcp.schema_converter import (
    mcp_tools_to_litellm,
    create_message_agent_tool,
)

if TYPE_CHECKING:
    from zap_ai.core.agent import ZapAgent


@dataclass
class AgentTools:
    """
    Complete tool set for an agent.

    Attributes:
        agent_name: Name of the agent.
        mcp_tools: Tools from MCP clients in LiteLLM format.
        message_agent_tool: The message_agent tool if agent has sub-agents.
        all_tools: Combined list of all tools in LiteLLM format.
    """
    agent_name: str
    mcp_tools: list[dict[str, Any]] = field(default_factory=list)
    message_agent_tool: dict[str, Any] | None = None

    @property
    def all_tools(self) -> list[dict[str, Any]]:
        """Get all tools including message_agent tool if present."""
        tools = self.mcp_tools.copy()
        if self.message_agent_tool:
            tools.append(self.message_agent_tool)
        return tools

    @property
    def tool_names(self) -> list[str]:
        """Get list of all tool names."""
        return [t["function"]["name"] for t in self.all_tools]

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self.tool_names

    @property
    def can_message_sub_agents(self) -> bool:
        """Check if this agent can message sub-agents."""
        return self.message_agent_tool is not None


class ToolRegistry:
    """
    High-level registry for agent tools.

    ToolRegistry provides a unified interface for managing all tools
    available to agents, including:
    - MCP tools from FastMCP clients
    - The message_agent tool for multi-turn sub-agent conversations

    It handles:
    - Client connection via ClientManager
    - Schema conversion from MCP to LiteLLM format
    - Caching of converted tool definitions
    - Sub-agent messaging tool generation

    Example:
        ```python
        registry = ToolRegistry()

        # Register agents (connects MCP clients)
        await registry.register_agents(agents, agent_map)

        # Get tools for LLM inference
        tools = registry.get_tools_for_agent("MyAgent")
        response = await litellm.completion(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        # Check if a tool is the message_agent tool
        if registry.is_message_agent_tool("message_agent"):
            # Handle sub-agent conversation
            pass
        ```

    Thread Safety:
        Registration should happen during Zap.start() before workflows run.
        After registration, all methods are read-only and thread-safe.
    """

    # Special tool name for sub-agent messaging
    MESSAGE_AGENT_TOOL_NAME = "message_agent"

    def __init__(self, client_manager: ClientManager | None = None) -> None:
        """
        Initialize the tool registry.

        Args:
            client_manager: Optional ClientManager instance. If not provided,
                a new one is created.
        """
        self._client_manager = client_manager or ClientManager()
        self._agent_tools: dict[str, AgentTools] = {}
        self._initialized = False

    @property
    def client_manager(self) -> ClientManager:
        """Get the underlying ClientManager."""
        return self._client_manager

    async def register_agents(
        self,
        agents: list[ZapAgent],
        agent_map: dict[str, ZapAgent],
    ) -> None:
        """
        Register all agents and discover their tools.

        This method should be called once during Zap.start(). It:
        1. Connects all MCP clients via ClientManager
        2. Discovers and converts tool schemas
        3. Generates transfer_to_agent tools for agents with sub-agents

        Args:
            agents: List of all agents to register.
            agent_map: Dict mapping agent names to ZapAgent instances.
                Used to look up sub-agent discovery_prompts.

        Raises:
            RuntimeError: If already initialized.
            ClientConnectionError: If any MCP client fails to connect.
        """
        if self._initialized:
            raise RuntimeError("ToolRegistry already initialized")

        for agent in agents:
            await self._register_single_agent(agent, agent_map)

        self._initialized = True

    async def _register_single_agent(
        self,
        agent: ZapAgent,
        agent_map: dict[str, ZapAgent],
    ) -> None:
        """
        Register a single agent's tools.

        Args:
            agent: The agent to register.
            agent_map: Full agent map for sub-agent lookup.
        """
        # Register with client manager (connects clients, discovers tools)
        await self._client_manager.register_agent(agent)

        # Get MCP tools and convert to LiteLLM format
        mcp_tools = self._client_manager.get_tools_for_agent(agent.name)
        litellm_tools = mcp_tools_to_litellm(mcp_tools)

        # Create message_agent tool if agent has sub-agents
        message_agent_tool = None
        if agent.sub_agents:
            available_agents = self._build_sub_agent_list(agent, agent_map)
            message_agent_tool = create_message_agent_tool(available_agents)

        self._agent_tools[agent.name] = AgentTools(
            agent_name=agent.name,
            mcp_tools=litellm_tools,
            message_agent_tool=message_agent_tool,
        )

    def _build_sub_agent_list(
        self,
        agent: ZapAgent,
        agent_map: dict[str, ZapAgent],
    ) -> list[tuple[str, str | None]]:
        """
        Build list of (name, discovery_prompt) for sub-agents.

        Args:
            agent: The parent agent.
            agent_map: Full agent map.

        Returns:
            List of (agent_name, discovery_prompt) tuples.
        """
        result: list[tuple[str, str | None]] = []
        for sub_name in agent.sub_agents:
            sub_agent = agent_map.get(sub_name)
            if sub_agent:
                result.append((sub_name, sub_agent.discovery_prompt))
            else:
                # Shouldn't happen if validation passed, but handle gracefully
                result.append((sub_name, None))
        return result

    def get_tools_for_agent(self, agent_name: str) -> list[dict[str, Any]]:
        """
        Get all tools for an agent in LiteLLM format.

        Args:
            agent_name: Name of the agent.

        Returns:
            List of LiteLLM-formatted tool definitions.
            Returns empty list if agent has no tools.

        Raises:
            KeyError: If agent not registered.
        """
        if agent_name not in self._agent_tools:
            raise KeyError(
                f"Agent '{agent_name}' not registered. "
                f"Available: {list(self._agent_tools.keys())}"
            )
        return self._agent_tools[agent_name].all_tools

    def get_agent_tools(self, agent_name: str) -> AgentTools:
        """
        Get the AgentTools object for an agent.

        Provides more detailed access than get_tools_for_agent().

        Args:
            agent_name: Name of the agent.

        Returns:
            AgentTools object with full tool information.

        Raises:
            KeyError: If agent not registered.
        """
        if agent_name not in self._agent_tools:
            raise KeyError(f"Agent '{agent_name}' not registered")
        return self._agent_tools[agent_name]

    def has_message_agent_tool(self, agent_name: str) -> bool:
        """
        Check if an agent has the message_agent tool.

        Args:
            agent_name: Name of the agent.

        Returns:
            True if agent has sub-agents and thus can message them.
        """
        if agent_name not in self._agent_tools:
            return False
        return self._agent_tools[agent_name].message_agent_tool is not None

    def is_message_agent_tool(self, tool_name: str) -> bool:
        """
        Check if a tool name is the special message_agent tool.

        Args:
            tool_name: Name of the tool to check.

        Returns:
            True if this is the message_agent tool.
        """
        return tool_name == self.MESSAGE_AGENT_TOOL_NAME

    def get_client_for_tool(self, agent_name: str, tool_name: str) -> Any:
        """
        Get the FastMCP client for executing a tool.

        This is a passthrough to ClientManager for convenience.

        Args:
            agent_name: Name of the agent.
            tool_name: Name of the tool.

        Returns:
            FastMCP Client instance.

        Raises:
            ToolNotFoundError: If tool not found.
            ValueError: If tool is the message_agent tool (handled specially).
        """
        if self.is_message_agent_tool(tool_name):
            raise ValueError(
                f"'{self.MESSAGE_AGENT_TOOL_NAME}' is not an MCP tool. "
                "Handle sub-agent messaging in the workflow."
            )
        return self._client_manager.get_client_for_tool(agent_name, tool_name)

    def list_agents(self) -> list[str]:
        """Return list of registered agent names."""
        return list(self._agent_tools.keys())

    def get_tool_count(self, agent_name: str) -> int:
        """
        Get total number of tools for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Number of tools (including transfer tool if present).
        """
        if agent_name not in self._agent_tools:
            return 0
        return len(self._agent_tools[agent_name].all_tools)

    async def shutdown(self) -> None:
        """
        Shutdown the registry and disconnect all clients.

        Delegates to ClientManager.disconnect_all().
        """
        await self._client_manager.disconnect_all()
        self._agent_tools.clear()
        self._initialized = False
```

**Checklist:**
- [ ] Create `src/zap_ai/mcp/tool_registry.py`
- [ ] Implement `AgentTools` dataclass with `all_tools` and `can_message_sub_agents` properties
- [ ] Implement `ToolRegistry` class with `__init__`
- [ ] Implement `register_agents()` - main registration entry point
- [ ] Implement `_register_single_agent()` - per-agent registration
- [ ] Implement `_build_sub_agent_list()` - builds sub-agent tuples
- [ ] Implement `get_tools_for_agent()` - returns LiteLLM tools
- [ ] Implement `get_agent_tools()` - returns AgentTools object
- [ ] Implement `has_message_agent_tool()` and `is_message_agent_tool()`
- [ ] Implement `get_client_for_tool()` - passthrough to ClientManager
- [ ] Implement `shutdown()` - cleanup method
- [ ] Define `MESSAGE_AGENT_TOOL_NAME` constant
- [ ] Add comprehensive docstrings with examples

---

## Task 6.4: Update MCP Module Init

**File:** `src/zap_ai/mcp/__init__.py`

```python
"""
MCP client management and tool handling.

This module provides the bridge between FastMCP (MCP protocol) and
LiteLLM (OpenAI function calling format).

Main components:
- ToolRegistry: High-level interface for agent tool management
- ClientManager: FastMCP client lifecycle management
- Schema converters: MCP to LiteLLM format conversion
"""

from zap_ai.mcp.client_manager import (
    ClientManager,
    ClientConnectionError,
    ToolNotFoundError,
    ConnectedClient,
    AgentToolMapping,
)
from zap_ai.mcp.schema_converter import (
    SchemaConversionError,
    mcp_tool_to_litellm,
    mcp_tools_to_litellm,
    create_message_agent_tool,
    validate_litellm_tool,
)
from zap_ai.mcp.tool_registry import (
    ToolRegistry,
    AgentTools,
)

__all__ = [
    # Main classes
    "ToolRegistry",
    "ClientManager",
    "AgentTools",
    # Supporting classes
    "ConnectedClient",
    "AgentToolMapping",
    # Exceptions
    "ClientConnectionError",
    "ToolNotFoundError",
    "SchemaConversionError",
    # Functions
    "mcp_tool_to_litellm",
    "mcp_tools_to_litellm",
    "create_message_agent_tool",
    "validate_litellm_tool",
]
```

**Checklist:**
- [ ] Update `src/zap_ai/mcp/__init__.py`
- [ ] Import all public classes and functions
- [ ] Define `__all__` list
- [ ] Add module docstring

---

## Phase 6 Verification

After completing all tasks, verify:

1. **Schema conversion works:**
   ```python
   from zap_ai.mcp import mcp_tool_to_litellm, create_message_agent_tool

   # MCP to LiteLLM
   mcp_tool = {
       "name": "test",
       "description": "Test tool",
       "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}
   }
   litellm_tool = mcp_tool_to_litellm(mcp_tool)
   assert litellm_tool["type"] == "function"
   assert litellm_tool["function"]["name"] == "test"
   assert litellm_tool["function"]["parameters"]["type"] == "object"

   # Message agent tool
   agents = [("SubAgent", "Does sub-agent things")]
   msg_tool = create_message_agent_tool(agents)
   assert msg_tool["function"]["name"] == "message_agent"
   assert "SubAgent" in msg_tool["function"]["parameters"]["properties"]["agent_name"]["enum"]
   assert "conversation_id" in msg_tool["function"]["parameters"]["properties"]
   assert "message" in msg_tool["function"]["parameters"]["required"]
   ```

2. **ClientManager connects and discovers:**
   ```python
   from zap_ai.mcp import ClientManager
   from fastmcp import Client

   manager = ClientManager()

   # Mock agent with MCP client
   agent = ZapAgent(
       name="TestAgent",
       prompt="Test",
       mcp_clients=[Client("./test_server.py")],
   )

   mapping = await manager.register_agent(agent)
   assert manager.is_agent_registered("TestAgent")

   # Should be able to get client for discovered tools
   client = manager.get_client_for_tool("TestAgent", "some_tool")

   await manager.disconnect_all()
   ```

3. **ToolRegistry provides complete tools:**
   ```python
   from zap_ai.mcp import ToolRegistry

   registry = ToolRegistry()

   agents = [
       ZapAgent(name="Main", prompt="Main agent", sub_agents=["Helper"]),
       ZapAgent(name="Helper", prompt="Helper", discovery_prompt="Helps with things"),
   ]
   agent_map = {a.name: a for a in agents}

   await registry.register_agents(agents, agent_map)

   # Main agent should have message_agent tool
   tools = registry.get_tools_for_agent("Main")
   tool_names = [t["function"]["name"] for t in tools]
   assert "message_agent" in tool_names

   # Helper agent should NOT have message_agent tool (no sub-agents)
   assert not registry.has_message_agent_tool("Helper")

   await registry.shutdown()
   ```

4. **Error handling works:**
   ```python
   from zap_ai.mcp import mcp_tool_to_litellm, SchemaConversionError

   # Missing name should raise
   try:
       mcp_tool_to_litellm({"description": "no name"})
       assert False, "Should have raised"
   except SchemaConversionError:
       pass

   # Unknown tool should raise
   try:
       manager.get_client_for_tool("Agent", "unknown_tool")
       assert False, "Should have raised"
   except ToolNotFoundError:
       pass
   ```
