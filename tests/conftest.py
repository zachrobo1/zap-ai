"""Shared test fixtures for Zap AI tests."""

from typing import Any, Generator
from unittest.mock import MagicMock

import pytest

from zap_ai import Task, TaskStatus, Zap, ZapAgent
from zap_ai.activities.inference import InferenceOutput
from zap_ai.activities.tool_execution import AgentConfigOutput


@pytest.fixture
def mock_litellm_completion(mocker: Any) -> MagicMock:
    """Mock the LiteLLM completion call.

    Returns a mock that can be configured per test to return
    specific responses.
    """
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked response"
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].finish_reason = "stop"

    mock = mocker.patch("litellm.acompletion", return_value=mock_response)
    return mock


@pytest.fixture
def mock_tool_registry(mocker: Any) -> Generator[MagicMock, None, None]:
    """Mock the tool registry for activity tests.

    Sets up a mock registry and patches it into the tool_execution module.
    """
    mock_registry = MagicMock()

    # Default mock for get_tools_for_agent
    mock_registry.get_tools_for_agent.return_value = []

    # Default mock for get_agent_config
    mock_config = MagicMock()
    mock_config.prompt = "You are a test agent."
    mock_config.model = "gpt-4o"
    mock_config.max_iterations = 50
    mock_registry.get_agent_config.return_value = mock_config

    # Default mock for call_tool
    mock_client = MagicMock()
    mock_client.call_tool = MagicMock(return_value="mock result")
    mock_registry.get_client_for_tool.return_value = mock_client

    mocker.patch(
        "zap_ai.activities.tool_execution.get_tool_registry",
        return_value=mock_registry,
    )
    yield mock_registry


@pytest.fixture
def mock_inference_output() -> InferenceOutput:
    """Create a mock inference output for testing."""
    return InferenceOutput(
        content="Mocked response",
        tool_calls=[],
        finish_reason="stop",
    )


@pytest.fixture
def mock_agent_config_output() -> AgentConfigOutput:
    """Create a mock agent config output for testing."""
    return AgentConfigOutput(
        agent_name="TestAgent",
        prompt="You are a test agent.",
        model="gpt-4o",
        max_iterations=50,
        tools=[],
    )


@pytest.fixture
def sample_agent() -> ZapAgent:
    """A minimal valid ZapAgent for testing."""
    return ZapAgent(name="TestAgent", prompt="You are a helpful assistant.")


@pytest.fixture
def sample_agent_with_model() -> ZapAgent:
    """A ZapAgent with a custom model specified."""
    return ZapAgent(
        name="CustomModelAgent",
        prompt="You are a custom model agent.",
        model="claude-3-sonnet-20240229",
    )


@pytest.fixture
def sample_task() -> Task:
    """A sample task for testing."""
    return Task(
        id="TestAgent-abc123",
        agent_name="TestAgent",
        status=TaskStatus.PENDING,
    )


@pytest.fixture
def sample_task_with_history() -> Task:
    """A sample task with conversation history."""
    return Task(
        id="TestAgent-abc123",
        agent_name="TestAgent",
        status=TaskStatus.THINKING,
        history=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {
                "role": "assistant",
                "content": "Using tool",
                "tool_calls": [{"id": "1", "function": {"name": "search"}}],
            },
            {"role": "tool", "content": "result", "tool_call_id": "1"},
        ],
    )


@pytest.fixture
def multi_agent_list() -> list[ZapAgent]:
    """A list of agents for testing multi-agent scenarios."""
    return [
        ZapAgent(name="MainAgent", prompt="You are the main agent."),
        ZapAgent(name="HelperAgent", prompt="You help the main agent."),
        ZapAgent(name="ReviewerAgent", prompt="You review work."),
    ]


@pytest.fixture
def agent_with_sub_agents() -> list[ZapAgent]:
    """A list of agents with sub-agent relationships."""
    return [
        ZapAgent(
            name="MainAgent",
            prompt="You are the main agent.",
            sub_agents=["HelperAgent", "ReviewerAgent"],
        ),
        ZapAgent(
            name="HelperAgent",
            prompt="You help the main agent.",
            discovery_prompt="I can help with research tasks.",
        ),
        ZapAgent(
            name="ReviewerAgent",
            prompt="You review work.",
            discovery_prompt="I can review and provide feedback.",
        ),
    ]


@pytest.fixture
def zap_instance(sample_agent: ZapAgent) -> Zap:
    """A Zap instance with a single agent."""
    return Zap(agents=[sample_agent])


@pytest.fixture
def zap_with_sub_agents(agent_with_sub_agents: list[ZapAgent]) -> Zap:
    """A Zap instance with sub-agent relationships."""
    return Zap(agents=agent_with_sub_agents)
