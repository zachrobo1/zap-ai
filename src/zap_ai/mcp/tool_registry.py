"""Tool registry for managing agent tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from zap_ai.mcp.client_manager import ClientManager
from zap_ai.mcp.schema_converter import (
    create_message_agent_tool,
    mcp_tools_to_litellm,
)

if TYPE_CHECKING:
    from zap_ai.core.agent import ZapAgent


@dataclass
class AgentConfig:
    """
    Complete configuration for an agent.

    Attributes:
        agent_name: Name of the agent.
        prompt: System prompt for the agent.
        model: LLM model identifier.
        max_iterations: Maximum agentic loop iterations.
    """

    agent_name: str
    prompt: str
    model: str
    max_iterations: int


@dataclass
class AgentTools:
    """
    Complete tool set for an agent.

    Attributes:
        agent_name: Name of the agent.
        mcp_tools: Tools from MCP clients in LiteLLM format.
        message_agent_tool: The message_agent tool if agent has sub-agents.
        config: Agent configuration (prompt, model, etc.).
    """

    agent_name: str
    mcp_tools: list[dict[str, Any]] = field(default_factory=list)
    message_agent_tool: dict[str, Any] | None = None
    config: AgentConfig | None = None

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
        3. Generates message_agent tools for agents with sub-agents

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

        # Store agent config for sub-agent workflow spawning
        config = AgentConfig(
            agent_name=agent.name,
            prompt=agent.prompt,
            model=agent.model,
            max_iterations=agent.max_iterations,
        )

        self._agent_tools[agent.name] = AgentTools(
            agent_name=agent.name,
            mcp_tools=litellm_tools,
            message_agent_tool=message_agent_tool,
            config=config,
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
                f"Agent '{agent_name}' not registered. Available: {list(self._agent_tools.keys())}"
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

    def get_tool_names(self, agent_name: str) -> list[str]:
        """
        Get list of tool names available to an agent.

        Useful for validating approval patterns.

        Args:
            agent_name: Name of the agent.

        Returns:
            List of tool names available to the agent.
            Returns empty list if agent has no tools.

        Raises:
            KeyError: If agent not registered.
        """
        if agent_name not in self._agent_tools:
            raise KeyError(
                f"Agent '{agent_name}' not registered. Available: {list(self._agent_tools.keys())}"
            )
        return self._agent_tools[agent_name].tool_names

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

    def get_agent_config(self, agent_name: str) -> AgentConfig | None:
        """
        Get configuration for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            AgentConfig or None if agent not registered.
        """
        agent_tools = self._agent_tools.get(agent_name)
        if agent_tools:
            return agent_tools.config
        return None

    def get_tool_count(self, agent_name: str) -> int:
        """
        Get total number of tools for an agent.

        Args:
            agent_name: Name of the agent.

        Returns:
            Number of tools (including message_agent tool if present).
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
