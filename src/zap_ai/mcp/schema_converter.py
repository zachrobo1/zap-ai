"""MCP to LiteLLM schema conversion utilities."""

from __future__ import annotations

from typing import Any

from zap_ai.exceptions import SchemaConversionError


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
    input_schema = mcp_tool.get("inputSchema", {"type": "object", "properties": {}, "required": []})

    # Ensure schema has required structure
    if not isinstance(input_schema, dict):
        raise SchemaConversionError(
            f"Tool '{name}' has invalid inputSchema: "
            f"expected dict, got {type(input_schema).__name__}"
        )

    # Ensure type is "object" (required by OpenAI)
    if input_schema.get("type") != "object":
        # Wrap non-object schemas
        input_schema = {
            "type": "object",
            "properties": {"value": input_schema},
            "required": ["value"],
        }

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        },
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
        },
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
