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
    AgentToolMapping,
    ClientConnectionError,
    ClientManager,
    ConnectedClient,
    ToolNotFoundError,
)
from zap_ai.mcp.schema_converter import (
    SchemaConversionError,
    create_message_agent_tool,
    mcp_tool_to_litellm,
    mcp_tools_to_litellm,
    validate_litellm_tool,
)
from zap_ai.mcp.tool_registry import (
    AgentConfig,
    AgentTools,
    ToolRegistry,
)

__all__ = [
    # Main classes
    "ToolRegistry",
    "ClientManager",
    "AgentConfig",
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
