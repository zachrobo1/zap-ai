"""Tool execution activity for Temporal workflows.

This module provides the tool execution activity that calls MCP tools
via FastMCP clients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from temporalio import activity

from zap_ai.exceptions import ToolExecutionError, ToolNotFoundError
from zap_ai.tracing import ObservationType, TraceContext, get_tracing_provider

if TYPE_CHECKING:
    from zap_ai.mcp import ToolRegistry


@dataclass
class AgentConfigOutput:
    """
    Output from get_agent_config_activity.

    Attributes:
        agent_name: Name of the agent.
        prompt: System prompt.
        model: LLM model identifier.
        max_iterations: Maximum agentic loop iterations.
        tools: List of tool definitions in LiteLLM format.
    """

    agent_name: str
    prompt: str
    model: str
    max_iterations: int
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolExecutionInput:
    """
    Input for the tool execution activity.

    Attributes:
        agent_name: Name of the agent executing the tool.
        tool_name: Name of the MCP tool to execute.
        arguments: Arguments to pass to the tool.
        trace_context: Optional trace context for observability.
    """

    agent_name: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    trace_context: dict[str, Any] | None = None


# Global reference to tool registry (set during worker initialization)
_tool_registry: ToolRegistry | None = None


def set_tool_registry(registry: ToolRegistry) -> None:
    """
    Set the global tool registry for activities.

    Called during worker initialization to provide activities
    access to MCP clients.

    Args:
        registry: ToolRegistry instance.
    """
    global _tool_registry
    _tool_registry = registry


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    if _tool_registry is None:
        raise RuntimeError(
            "Tool registry not initialized. Call set_tool_registry() during worker setup."
        )
    return _tool_registry


@activity.defn
async def tool_execution_activity(input: ToolExecutionInput) -> str:
    """
    Execute an MCP tool as a Temporal activity.

    Looks up the tool's source client via the ToolRegistry and executes
    the tool with the provided arguments.

    Args:
        input: ToolExecutionInput with agent name, tool name, and arguments.

    Returns:
        Tool execution result as a JSON string.

    Raises:
        ToolNotFoundError: If the tool doesn't exist for this agent.
        ToolExecutionError: If tool execution fails.
        RuntimeError: If tool registry is not initialized.
    """
    tracer = get_tracing_provider()
    parent_context = None

    # Reconstruct trace context if provided
    if input.trace_context:
        parent_context = TraceContext.from_dict(input.trace_context)

    activity.logger.info(
        f"Executing tool '{input.tool_name}' for agent '{input.agent_name}' "
        f"with args: {json.dumps(input.arguments)[:200]}"
    )

    registry = get_tool_registry()

    # Get the client for this tool
    try:
        client = registry.get_client_for_tool(input.agent_name, input.tool_name)
    except ValueError as e:
        # message_agent tool is handled by the workflow, not here
        raise ToolExecutionError(str(e)) from e
    except KeyError as e:
        raise ToolNotFoundError(str(e)) from e
    except Exception as e:
        raise ToolNotFoundError(
            f"Tool '{input.tool_name}' not found for agent '{input.agent_name}': {e}"
        ) from e

    # Execute the tool via FastMCP client
    async def _execute_tool() -> str:
        result = await client.call_tool(input.tool_name, input.arguments)

        activity.logger.info(f"Tool '{input.tool_name}' executed successfully")

        # FastMCP returns various types; serialize to JSON string
        if isinstance(result, str):
            return result
        return json.dumps(result, default=str)

    try:
        # Wrap in tool span if we have trace context
        if parent_context:
            async with tracer.start_observation(
                name=f"tool-{input.tool_name}",
                observation_type=ObservationType.TOOL,
                parent_context=parent_context,
                metadata={
                    "tool_name": input.tool_name,
                    "agent_name": input.agent_name,
                },
                input_data=input.arguments,
            ):
                return await _execute_tool()
        else:
            return await _execute_tool()

    except Exception as e:
        activity.logger.error(f"Tool execution failed: {e}")
        raise ToolExecutionError(f"Failed to execute tool '{input.tool_name}': {e}") from e


@activity.defn
async def get_agent_config_activity(agent_name: str) -> AgentConfigOutput:
    """
    Get configuration for an agent.

    Used by workflows to get sub-agent config before starting child workflows.

    Args:
        agent_name: Name of the agent to get config for.

    Returns:
        AgentConfigOutput with prompt, model, and tools.

    Raises:
        KeyError: If agent not found.
        RuntimeError: If tool registry is not initialized.
    """
    activity.logger.info(f"Getting config for agent '{agent_name}'")

    registry = get_tool_registry()

    # Get agent config
    config = registry.get_agent_config(agent_name)
    if not config:
        raise KeyError(f"Agent '{agent_name}' not found in registry")

    # Get tools
    try:
        tools = registry.get_tools_for_agent(agent_name)
    except KeyError:
        tools = []

    return AgentConfigOutput(
        agent_name=agent_name,
        prompt=config.prompt,
        model=config.model,
        max_iterations=config.max_iterations,
        tools=tools,
    )
