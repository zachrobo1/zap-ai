"""Tests for MCP schema converter."""

import pytest

from zap_ai.mcp.schema_converter import (
    SchemaConversionError,
    create_message_agent_tool,
    mcp_tool_to_litellm,
    mcp_tools_to_litellm,
    validate_litellm_tool,
)


class TestMcpToolToLitellm:
    """Tests for mcp_tool_to_litellm function."""

    def test_converts_basic_tool(self) -> None:
        """Test converting a basic MCP tool."""
        mcp_tool = {
            "name": "get_weather",
            "description": "Get weather for a city",
            "inputSchema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }

        result = mcp_tool_to_litellm(mcp_tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get weather for a city"
        assert result["function"]["parameters"]["type"] == "object"
        assert "city" in result["function"]["parameters"]["properties"]

    def test_handles_missing_description(self) -> None:
        """Test tool with no description."""
        mcp_tool = {
            "name": "simple_tool",
            "inputSchema": {"type": "object", "properties": {}},
        }

        result = mcp_tool_to_litellm(mcp_tool)

        assert result["function"]["name"] == "simple_tool"
        assert result["function"]["description"] == ""

    def test_handles_missing_input_schema(self) -> None:
        """Test tool with no inputSchema."""
        mcp_tool = {"name": "no_params"}

        result = mcp_tool_to_litellm(mcp_tool)

        assert result["function"]["name"] == "no_params"
        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"] == {}

    def test_raises_on_missing_name(self) -> None:
        """Test error when name is missing."""
        mcp_tool = {"description": "no name field"}

        with pytest.raises(SchemaConversionError, match="missing required 'name' field"):
            mcp_tool_to_litellm(mcp_tool)

    def test_raises_on_empty_name(self) -> None:
        """Test error when name is empty string."""
        mcp_tool = {"name": "   "}

        with pytest.raises(SchemaConversionError, match="Invalid tool name"):
            mcp_tool_to_litellm(mcp_tool)

    def test_raises_on_invalid_name_type(self) -> None:
        """Test error when name is not a string."""
        mcp_tool = {"name": 123}

        with pytest.raises(SchemaConversionError, match="Invalid tool name"):
            mcp_tool_to_litellm(mcp_tool)

    def test_raises_on_invalid_input_schema_type(self) -> None:
        """Test error when inputSchema is not a dict."""
        mcp_tool = {"name": "bad_schema", "inputSchema": "not a dict"}

        with pytest.raises(SchemaConversionError, match="invalid inputSchema"):
            mcp_tool_to_litellm(mcp_tool)

    def test_wraps_non_object_schema(self) -> None:
        """Test that non-object schemas are wrapped."""
        mcp_tool = {
            "name": "string_input",
            "inputSchema": {"type": "string", "description": "A string value"},
        }

        result = mcp_tool_to_litellm(mcp_tool)

        # Should wrap in object with "value" property
        params = result["function"]["parameters"]
        assert params["type"] == "object"
        assert "value" in params["properties"]
        assert params["properties"]["value"]["type"] == "string"
        assert params["required"] == ["value"]

    def test_coerces_description_to_string(self) -> None:
        """Test that non-string descriptions are converted."""
        mcp_tool = {"name": "test", "description": 123}

        result = mcp_tool_to_litellm(mcp_tool)

        assert result["function"]["description"] == "123"


class TestMcpToolsToLitellm:
    """Tests for mcp_tools_to_litellm function."""

    def test_converts_list_of_tools(self) -> None:
        """Test converting multiple tools."""
        mcp_tools = [
            {"name": "tool_a", "description": "Tool A"},
            {"name": "tool_b", "description": "Tool B"},
        ]

        result = mcp_tools_to_litellm(mcp_tools)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "tool_a"
        assert result[1]["function"]["name"] == "tool_b"

    def test_empty_list(self) -> None:
        """Test converting empty list."""
        result = mcp_tools_to_litellm([])
        assert result == []

    def test_raises_on_invalid_tool(self) -> None:
        """Test that invalid tool raises error."""
        mcp_tools = [
            {"name": "good_tool"},
            {"description": "missing name"},
        ]

        with pytest.raises(SchemaConversionError):
            mcp_tools_to_litellm(mcp_tools)


class TestCreateMessageAgentTool:
    """Tests for create_message_agent_tool function."""

    def test_creates_message_agent_tool(self) -> None:
        """Test creating message_agent tool."""
        agents = [
            ("ResearchAgent", "Use for research tasks"),
            ("WriterAgent", "Use for writing tasks"),
        ]

        result = create_message_agent_tool(agents)

        assert result["type"] == "function"
        assert result["function"]["name"] == "message_agent"
        assert "ResearchAgent" in result["function"]["description"]
        assert "WriterAgent" in result["function"]["description"]

    def test_includes_agent_names_in_enum(self) -> None:
        """Test that agent names are in enum."""
        agents = [("Agent1", "Desc 1"), ("Agent2", "Desc 2")]

        result = create_message_agent_tool(agents)

        agent_name_prop = result["function"]["parameters"]["properties"]["agent_name"]
        assert agent_name_prop["enum"] == ["Agent1", "Agent2"]

    def test_includes_required_message_parameter(self) -> None:
        """Test that message is required."""
        agents = [("TestAgent", "Test")]

        result = create_message_agent_tool(agents)

        assert "message" in result["function"]["parameters"]["required"]

    def test_includes_conversation_id_parameter(self) -> None:
        """Test that conversation_id is included."""
        agents = [("TestAgent", "Test")]

        result = create_message_agent_tool(agents)

        props = result["function"]["parameters"]["properties"]
        assert "conversation_id" in props
        assert props["conversation_id"]["type"] == "string"

    def test_handles_none_discovery_prompt(self) -> None:
        """Test handling agents without discovery prompt."""
        agents = [("HiddenAgent", None)]

        result = create_message_agent_tool(agents)

        assert "HiddenAgent" in result["function"]["description"]
        assert "(no description)" in result["function"]["description"]

    def test_raises_on_empty_agents(self) -> None:
        """Test error when no agents provided."""
        with pytest.raises(ValueError, match="no available agents"):
            create_message_agent_tool([])

    def test_description_includes_usage_patterns(self) -> None:
        """Test that description includes usage patterns."""
        agents = [("TestAgent", "Test")]

        result = create_message_agent_tool(agents)

        desc = result["function"]["description"]
        assert "Start new conversation" in desc
        assert "Continue conversation" in desc


class TestValidateLitellmTool:
    """Tests for validate_litellm_tool function."""

    def test_valid_tool_returns_true(self) -> None:
        """Test that valid tool returns True."""
        tool = {
            "type": "function",
            "function": {
                "name": "test",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        assert validate_litellm_tool(tool) is True

    def test_raises_on_non_dict(self) -> None:
        """Test error when tool is not a dict."""
        with pytest.raises(SchemaConversionError, match="must be dict"):
            validate_litellm_tool("not a dict")  # type: ignore

    def test_raises_on_wrong_type(self) -> None:
        """Test error when type is not 'function'."""
        tool = {"type": "other", "function": {"name": "test"}}

        with pytest.raises(SchemaConversionError, match="must be 'function'"):
            validate_litellm_tool(tool)

    def test_raises_on_missing_function(self) -> None:
        """Test error when function dict is missing."""
        tool = {"type": "function"}

        with pytest.raises(SchemaConversionError, match="missing 'function' dict"):
            validate_litellm_tool(tool)

    def test_raises_on_missing_name(self) -> None:
        """Test error when function name is missing."""
        tool = {"type": "function", "function": {}}

        with pytest.raises(SchemaConversionError, match="missing 'name'"):
            validate_litellm_tool(tool)

    def test_raises_on_invalid_parameters(self) -> None:
        """Test error when parameters is not a dict."""
        tool = {
            "type": "function",
            "function": {"name": "test", "parameters": "not a dict"},
        }

        with pytest.raises(SchemaConversionError, match="parameters must be dict"):
            validate_litellm_tool(tool)

    def test_raises_on_wrong_parameters_type(self) -> None:
        """Test error when parameters type is not 'object'."""
        tool = {
            "type": "function",
            "function": {"name": "test", "parameters": {"type": "string"}},
        }

        with pytest.raises(SchemaConversionError, match="type must be 'object'"):
            validate_litellm_tool(tool)

    def test_tool_without_parameters_is_valid(self) -> None:
        """Test that tool without parameters is valid."""
        tool = {"type": "function", "function": {"name": "test"}}

        assert validate_litellm_tool(tool) is True


class TestMcpModuleImports:
    """Tests for MCP module imports."""

    def test_imports_from_module(self) -> None:
        """Test that all exports are importable from module."""
        from zap_ai.mcp import (
            AgentToolMapping,
            AgentTools,
            ClientConnectionError,
            ClientManager,
            ConnectedClient,
            SchemaConversionError,
            ToolNotFoundError,
            ToolRegistry,
            create_message_agent_tool,
            mcp_tool_to_litellm,
            mcp_tools_to_litellm,
            validate_litellm_tool,
        )

        assert ToolRegistry is not None
        assert ClientManager is not None
        assert AgentTools is not None
        assert ConnectedClient is not None
        assert AgentToolMapping is not None
        assert ClientConnectionError is not None
        assert ToolNotFoundError is not None
        assert SchemaConversionError is not None
        assert mcp_tool_to_litellm is not None
        assert mcp_tools_to_litellm is not None
        assert create_message_agent_tool is not None
        assert validate_litellm_tool is not None

    def test_all_exports_defined(self) -> None:
        """Test that __all__ is properly defined."""
        from zap_ai import mcp

        assert hasattr(mcp, "__all__")
        expected = [
            "ToolRegistry",
            "ClientManager",
            "AgentTools",
            "ConnectedClient",
            "AgentToolMapping",
            "ClientConnectionError",
            "ToolNotFoundError",
            "SchemaConversionError",
            "mcp_tool_to_litellm",
            "mcp_tools_to_litellm",
            "create_message_agent_tool",
            "validate_litellm_tool",
        ]
        for name in expected:
            assert name in mcp.__all__
