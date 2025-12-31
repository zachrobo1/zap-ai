# Phase 8: Testing

**Goal:** Add tests for critical components.

**Dependencies:** All previous phases should be complete.

---

## Task 8.1: Unit Tests for Schema Converter

**File:** `tests/test_schema_converter.py`

```python
"""Tests for MCP to LiteLLM schema conversion."""

import pytest

from zap_ai.mcp.schema_converter import (
    mcp_tool_to_litellm,
    mcp_tools_to_litellm,
    create_message_agent_tool,
    validate_litellm_tool,
    SchemaConversionError,
)


class TestMcpToolToLitellm:
    """Tests for mcp_tool_to_litellm function."""

    def test_basic_conversion(self):
        """Test basic MCP to LiteLLM conversion."""
        mcp_tool = {
            "name": "search",
            "description": "Search for something",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            }
        }

        result = mcp_tool_to_litellm(mcp_tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "search"
        assert result["function"]["description"] == "Search for something"
        assert result["function"]["parameters"]["type"] == "object"
        assert "query" in result["function"]["parameters"]["properties"]

    def test_missing_name_raises(self):
        """Test that missing name raises SchemaConversionError."""
        with pytest.raises(SchemaConversionError):
            mcp_tool_to_litellm({"description": "no name"})

    def test_empty_name_raises(self):
        """Test that empty name raises SchemaConversionError."""
        with pytest.raises(SchemaConversionError):
            mcp_tool_to_litellm({"name": "", "description": "empty name"})

    def test_missing_description_defaults_empty(self):
        """Test that missing description defaults to empty string."""
        mcp_tool = {
            "name": "test",
            "inputSchema": {"type": "object", "properties": {}},
        }

        result = mcp_tool_to_litellm(mcp_tool)
        assert result["function"]["description"] == ""

    def test_missing_input_schema_defaults_empty(self):
        """Test that missing inputSchema defaults to empty object."""
        mcp_tool = {"name": "test", "description": "Test tool"}

        result = mcp_tool_to_litellm(mcp_tool)
        assert result["function"]["parameters"]["type"] == "object"
        assert result["function"]["parameters"]["properties"] == {}

    def test_non_object_schema_wrapped(self):
        """Test that non-object schemas are wrapped in object."""
        mcp_tool = {
            "name": "test",
            "inputSchema": {"type": "string"},
        }

        result = mcp_tool_to_litellm(mcp_tool)
        assert result["function"]["parameters"]["type"] == "object"
        assert "value" in result["function"]["parameters"]["properties"]


class TestMcpToolsToLitellm:
    """Tests for batch conversion."""

    def test_batch_conversion(self):
        """Test converting multiple tools."""
        mcp_tools = [
            {"name": "tool1", "description": "First tool"},
            {"name": "tool2", "description": "Second tool"},
        ]

        results = mcp_tools_to_litellm(mcp_tools)

        assert len(results) == 2
        assert results[0]["function"]["name"] == "tool1"
        assert results[1]["function"]["name"] == "tool2"

    def test_empty_list(self):
        """Test converting empty list."""
        assert mcp_tools_to_litellm([]) == []


class TestCreateMessageAgentTool:
    """Tests for message_agent tool generation."""

    def test_basic_creation(self):
        """Test creating message_agent tool."""
        agents = [
            ("ResearchAgent", "Does research"),
            ("WriterAgent", "Writes content"),
        ]

        result = create_message_agent_tool(agents)

        assert result["type"] == "function"
        assert result["function"]["name"] == "message_agent"
        assert "ResearchAgent" in str(result["function"]["description"])
        assert "WriterAgent" in str(result["function"]["description"])

        # Check enum
        enum = result["function"]["parameters"]["properties"]["agent_name"]["enum"]
        assert "ResearchAgent" in enum
        assert "WriterAgent" in enum

    def test_conversation_id_parameter(self):
        """Test that conversation_id is included."""
        agents = [("Test", "Test agent")]

        result = create_message_agent_tool(agents)

        props = result["function"]["parameters"]["properties"]
        assert "conversation_id" in props
        assert "message" in result["function"]["parameters"]["required"]

    def test_empty_agents_raises(self):
        """Test that empty agents list raises ValueError."""
        with pytest.raises(ValueError):
            create_message_agent_tool([])

    def test_none_discovery_prompt(self):
        """Test agent with None discovery prompt."""
        agents = [("TestAgent", None)]

        result = create_message_agent_tool(agents)
        assert "TestAgent" in result["function"]["parameters"]["properties"]["agent_name"]["enum"]


class TestValidateLitellmTool:
    """Tests for tool validation."""

    def test_valid_tool(self):
        """Test validating a correct tool definition."""
        tool = {
            "type": "function",
            "function": {
                "name": "test",
                "parameters": {
                    "type": "object",
                    "properties": {},
                }
            }
        }

        assert validate_litellm_tool(tool) is True

    def test_wrong_type_raises(self):
        """Test that wrong type raises error."""
        with pytest.raises(SchemaConversionError):
            validate_litellm_tool({"type": "not_function"})

    def test_missing_function_raises(self):
        """Test that missing function raises error."""
        with pytest.raises(SchemaConversionError):
            validate_litellm_tool({"type": "function"})

    def test_missing_name_raises(self):
        """Test that missing function name raises error."""
        with pytest.raises(SchemaConversionError):
            validate_litellm_tool({
                "type": "function",
                "function": {}
            })
```

**Checklist:**
- [ ] Create `tests/test_schema_converter.py`
- [ ] Test MCP → LiteLLM conversion
- [ ] Test message_agent tool generation
- [ ] Test edge cases (missing fields, non-object schemas)
- [ ] Test validation function

---

## Task 8.2: Unit Tests for Agent Validation

**File:** `tests/test_agent_validation.py`

```python
"""Tests for agent and Zap validation."""

import pytest

from zap_ai import Zap, ZapAgent, ZapConfigurationError


class TestZapAgentValidation:
    """Tests for ZapAgent field validation."""

    def test_valid_name(self):
        """Test that valid names are accepted."""
        agent = ZapAgent(name="ValidName", prompt="Test")
        assert agent.name == "ValidName"

        agent = ZapAgent(name="valid_name", prompt="Test")
        assert agent.name == "valid_name"

        agent = ZapAgent(name="valid-name-123", prompt="Test")
        assert agent.name == "valid-name-123"

    def test_name_with_spaces_raises(self):
        """Test that names with spaces raise ValueError."""
        with pytest.raises(ValueError, match="cannot contain spaces"):
            ZapAgent(name="Invalid Name", prompt="Test")

    def test_name_with_slash_raises(self):
        """Test that names with slashes raise ValueError."""
        with pytest.raises(ValueError, match="cannot contain forward slashes"):
            ZapAgent(name="invalid/name", prompt="Test")

    def test_name_with_special_chars_raises(self):
        """Test that names with special characters raise ValueError."""
        with pytest.raises(ValueError, match="invalid characters"):
            ZapAgent(name="invalid@name", prompt="Test")

    def test_duplicate_sub_agents_raises(self):
        """Test that duplicate sub-agent references raise ValueError."""
        with pytest.raises(ValueError, match="Duplicate sub-agent"):
            ZapAgent(name="Test", prompt="Test", sub_agents=["A", "A"])


class TestZapValidation:
    """Tests for Zap configuration validation."""

    def test_duplicate_names_detected(self):
        """Test that duplicate agent names are detected."""
        agents = [
            ZapAgent(name="Agent1", prompt="Test"),
            ZapAgent(name="Agent1", prompt="Test"),  # Duplicate
        ]

        with pytest.raises(ZapConfigurationError, match="Duplicate agent names"):
            Zap(agents=agents)

    def test_unknown_sub_agent_detected(self):
        """Test that unknown sub-agent references are detected."""
        agents = [
            ZapAgent(name="Main", prompt="Test", sub_agents=["Unknown"]),
        ]

        with pytest.raises(ZapConfigurationError, match="unknown sub-agent"):
            Zap(agents=agents)

    def test_self_reference_detected(self):
        """Test that self-referencing is detected."""
        agents = [
            ZapAgent(name="Main", prompt="Test", sub_agents=["Main"]),
        ]

        with pytest.raises(ZapConfigurationError, match="cannot reference itself"):
            Zap(agents=agents)

    def test_circular_dependency_detected(self):
        """Test that circular dependencies are detected."""
        agents = [
            ZapAgent(name="A", prompt="Test", sub_agents=["B"]),
            ZapAgent(name="B", prompt="Test", sub_agents=["A"]),
        ]

        with pytest.raises(ZapConfigurationError, match="Circular dependency"):
            Zap(agents=agents)

    def test_longer_cycle_detected(self):
        """Test that longer cycles are detected."""
        agents = [
            ZapAgent(name="A", prompt="Test", sub_agents=["B"]),
            ZapAgent(name="B", prompt="Test", sub_agents=["C"]),
            ZapAgent(name="C", prompt="Test", sub_agents=["A"]),
        ]

        with pytest.raises(ZapConfigurationError, match="Circular dependency"):
            Zap(agents=agents)

    def test_valid_configuration(self):
        """Test that valid configuration passes validation."""
        agents = [
            ZapAgent(name="Main", prompt="Test", sub_agents=["Helper"]),
            ZapAgent(name="Helper", prompt="Test", discovery_prompt="Helps"),
        ]

        zap = Zap(agents=agents)
        assert len(zap.agents) == 2
        assert "Main" in zap.list_agents()
        assert "Helper" in zap.list_agents()
```

**Checklist:**
- [ ] Create `tests/test_agent_validation.py`
- [ ] Test ZapAgent name validation
- [ ] Test duplicate name detection
- [ ] Test unknown sub-agent detection
- [ ] Test circular dependency detection
- [ ] Test valid configuration acceptance

---

## Task 8.3: Unit Tests for Message Types

**File:** `tests/test_message_types.py`

```python
"""Tests for LLM message types."""

import pytest

from zap_ai.llm.message_types import Message, ToolCall, InferenceResult


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_from_litellm(self):
        """Test parsing from LiteLLM format."""
        litellm_tc = {
            "id": "call_123",
            "function": {
                "name": "search",
                "arguments": '{"query": "test"}',
            }
        }

        tc = ToolCall.from_litellm(litellm_tc)

        assert tc.id == "call_123"
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}

    def test_to_litellm(self):
        """Test serialization to LiteLLM format."""
        tc = ToolCall(
            id="call_123",
            name="search",
            arguments={"query": "test"},
            arguments_raw='{"query": "test"}',
        )

        result = tc.to_litellm()

        assert result["id"] == "call_123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"

    def test_invalid_json_arguments(self):
        """Test handling of invalid JSON arguments."""
        litellm_tc = {
            "id": "call_123",
            "function": {
                "name": "search",
                "arguments": "not valid json",
            }
        }

        tc = ToolCall.from_litellm(litellm_tc)
        assert tc.arguments == {}  # Defaults to empty dict


class TestMessage:
    """Tests for Message dataclass."""

    def test_factory_methods(self):
        """Test message factory methods."""
        sys_msg = Message.system("System prompt")
        assert sys_msg.role == "system"
        assert sys_msg.content == "System prompt"

        user_msg = Message.user("User input")
        assert user_msg.role == "user"
        assert user_msg.content == "User input"

        asst_msg = Message.assistant("Response")
        assert asst_msg.role == "assistant"
        assert asst_msg.content == "Response"

    def test_tool_result(self):
        """Test tool result message."""
        msg = Message.tool_result("call_123", "search", "Result")

        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"
        assert msg.name == "search"
        assert msg.content == "Result"

    def test_to_litellm(self):
        """Test serialization to LiteLLM format."""
        msg = Message.user("Hello")
        result = msg.to_litellm()

        assert result == {"role": "user", "content": "Hello"}

    def test_from_litellm(self):
        """Test parsing from LiteLLM format."""
        litellm_msg = {"role": "user", "content": "Hello"}
        msg = Message.from_litellm(litellm_msg)

        assert msg.role == "user"
        assert msg.content == "Hello"


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_has_tool_calls(self):
        """Test has_tool_calls property."""
        result_no_tools = InferenceResult(content="Hello")
        assert not result_no_tools.has_tool_calls

        result_with_tools = InferenceResult(
            tool_calls=[ToolCall(id="1", name="test", arguments={})]
        )
        assert result_with_tools.has_tool_calls

    def test_is_complete(self):
        """Test is_complete property."""
        result_complete = InferenceResult(content="Hello")
        assert result_complete.is_complete

        result_not_complete = InferenceResult(
            tool_calls=[ToolCall(id="1", name="test", arguments={})]
        )
        assert not result_not_complete.is_complete

    def test_to_message(self):
        """Test conversion to Message."""
        result = InferenceResult(content="Hello")
        msg = result.to_message()

        assert msg.role == "assistant"
        assert msg.content == "Hello"
```

**Checklist:**
- [ ] Create `tests/test_message_types.py`
- [ ] Test ToolCall serialization/deserialization
- [ ] Test Message factory methods
- [ ] Test InferenceResult properties
- [ ] Test edge cases (invalid JSON, missing fields)

---

## Task 8.4: Integration Tests with Mock Temporal

**File:** `tests/test_workflow_integration.py`

```python
"""Integration tests for agent workflow with mock Temporal."""

import pytest
from datetime import timedelta

from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from zap_ai.workflows.agent_workflow import AgentWorkflow
from zap_ai.workflows.models import AgentWorkflowInput
from zap_ai.activities.inference import inference_activity, InferenceInput, InferenceOutput
from zap_ai.activities.tool_execution import tool_execution_activity


# Mock activities for testing
async def mock_inference_activity(input: InferenceInput) -> InferenceOutput:
    """Mock inference that returns a simple response."""
    return InferenceOutput(
        content=f"Response to: {input.messages[-1].get('content', '')}",
        tool_calls=[],
        finish_reason="stop",
    )


@pytest.fixture
async def workflow_env():
    """Create a time-skipping workflow environment."""
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


@pytest.mark.asyncio
async def test_basic_workflow_execution(workflow_env):
    """Test that a basic workflow executes and completes."""
    async with Worker(
        workflow_env.client,
        task_queue="test-queue",
        workflows=[AgentWorkflow],
        activities=[mock_inference_activity],
    ):
        result = await workflow_env.client.execute_workflow(
            AgentWorkflow.run,
            AgentWorkflowInput(
                agent_name="TestAgent",
                initial_task="Hello, agent!",
            ),
            id="test-workflow-1",
            task_queue="test-queue",
            execution_timeout=timedelta(seconds=30),
        )

        assert result is not None
        assert "Hello, agent!" in result


@pytest.mark.asyncio
async def test_workflow_queries(workflow_env):
    """Test that workflow queries work correctly."""
    async with Worker(
        workflow_env.client,
        task_queue="test-queue",
        workflows=[AgentWorkflow],
        activities=[mock_inference_activity],
    ):
        handle = await workflow_env.client.start_workflow(
            AgentWorkflow.run,
            AgentWorkflowInput(
                agent_name="TestAgent",
                initial_task="Hello",
            ),
            id="test-workflow-2",
            task_queue="test-queue",
        )

        # Wait briefly for workflow to start
        await workflow_env.sleep(timedelta(milliseconds=100))

        # Query status
        status = await handle.query(AgentWorkflow.get_status)
        assert status in ["pending", "running", "completed"]


@pytest.mark.asyncio
async def test_workflow_signal(workflow_env):
    """Test that workflow signals work correctly."""
    async with Worker(
        workflow_env.client,
        task_queue="test-queue",
        workflows=[AgentWorkflow],
        activities=[mock_inference_activity],
    ):
        handle = await workflow_env.client.start_workflow(
            AgentWorkflow.run,
            AgentWorkflowInput(
                agent_name="TestAgent",
                initial_task="Initial task",
            ),
            id="test-workflow-3",
            task_queue="test-queue",
        )

        # Send follow-up signal
        await handle.signal(AgentWorkflow.add_message, "Follow-up message")

        # Wait and get result
        result = await handle.result()
        assert result is not None
```

**Checklist:**
- [ ] Create `tests/test_workflow_integration.py`
- [ ] Set up WorkflowEnvironment fixture
- [ ] Test basic workflow execution
- [ ] Test query functionality
- [ ] Test signal handling
- [ ] Use mock activities for isolation

---

## Task 8.5: Integration Tests with Mock MCP

**File:** `tests/test_mcp_integration.py`

```python
"""Integration tests for MCP client management."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from zap_ai import ZapAgent
from zap_ai.mcp import ClientManager, ToolRegistry


class MockMCPClient:
    """Mock FastMCP client for testing."""

    def __init__(self, tools=None):
        self.tools = tools or []
        self._entered = False

    async def __aenter__(self):
        self._entered = True
        return self

    async def __aexit__(self, *args):
        self._entered = False

    async def list_tools(self):
        return self.tools

    async def call_tool(self, name, arguments):
        return f"Result for {name}({arguments})"


@pytest.fixture
def mock_tool():
    """Create a mock MCP tool."""
    return MagicMock(
        name="search",
        description="Search for something",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


@pytest.fixture
def mock_client(mock_tool):
    """Create a mock MCP client."""
    return MockMCPClient(tools=[mock_tool])


@pytest.mark.asyncio
async def test_client_manager_register_agent(mock_client, mock_tool):
    """Test registering an agent with ClientManager."""
    manager = ClientManager()

    agent = ZapAgent(
        name="TestAgent",
        prompt="Test prompt",
        mcp_clients=[mock_client],
    )

    mapping = await manager.register_agent(agent)

    assert manager.is_agent_registered("TestAgent")
    assert "search" in mapping.tool_to_client


@pytest.mark.asyncio
async def test_client_manager_get_client_for_tool(mock_client, mock_tool):
    """Test getting client for a specific tool."""
    manager = ClientManager()

    agent = ZapAgent(
        name="TestAgent",
        prompt="Test",
        mcp_clients=[mock_client],
    )

    await manager.register_agent(agent)

    client = manager.get_client_for_tool("TestAgent", "search")
    assert client is mock_client


@pytest.mark.asyncio
async def test_tool_registry_with_sub_agents():
    """Test tool registry creates message_agent tool for sub-agents."""
    mock_client = MockMCPClient(tools=[])

    agents = [
        ZapAgent(
            name="Main",
            prompt="Main agent",
            mcp_clients=[mock_client],
            sub_agents=["Helper"],
        ),
        ZapAgent(
            name="Helper",
            prompt="Helper agent",
            discovery_prompt="Helps with things",
        ),
    ]
    agent_map = {a.name: a for a in agents}

    registry = ToolRegistry()
    await registry.register_agents(agents, agent_map)

    # Main should have message_agent tool
    assert registry.has_message_agent_tool("Main")

    tools = registry.get_tools_for_agent("Main")
    tool_names = [t["function"]["name"] for t in tools]
    assert "message_agent" in tool_names

    # Helper should NOT have message_agent tool
    assert not registry.has_message_agent_tool("Helper")


@pytest.mark.asyncio
async def test_client_manager_disconnect_all(mock_client):
    """Test disconnecting all clients."""
    manager = ClientManager()

    agent = ZapAgent(
        name="TestAgent",
        prompt="Test",
        mcp_clients=[mock_client],
    )

    await manager.register_agent(agent)
    assert mock_client._entered

    await manager.disconnect_all()
    assert not mock_client._entered
```

**Checklist:**
- [ ] Create `tests/test_mcp_integration.py`
- [ ] Create MockMCPClient for testing
- [ ] Test ClientManager registration
- [ ] Test tool discovery
- [ ] Test ToolRegistry with sub-agents
- [ ] Test client disconnection

---

## Phase 8 Verification

After completing all tasks, verify:

1. **All tests pass:**
   ```bash
   pytest tests/ -v
   ```

2. **Coverage is reasonable:**
   ```bash
   pytest tests/ --cov=zap_ai --cov-report=html
   ```

3. **Tests are isolated:**
   - Unit tests don't require external services
   - Integration tests use mocks appropriately

4. **Tests are fast:**
   - Unit tests complete quickly
   - Integration tests use time-skipping where possible
