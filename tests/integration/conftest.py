"""Integration test fixtures for Temporal integration tests."""

import uuid
from collections.abc import AsyncGenerator

import pytest
from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from zap_ai.activities import InferenceInput, InferenceOutput, ToolExecutionInput
from zap_ai.activities.tool_execution import AgentConfigOutput
from zap_ai.workflows import AgentWorkflow


@activity.defn(name="inference_activity")
async def mock_inference_activity(input: InferenceInput) -> InferenceOutput:
    """Mock inference that returns a simple response without calling LLM."""
    last_content = input.messages[-1].get("content", "") if input.messages else ""
    return InferenceOutput(
        content=f"Integration test response to: {last_content}",
        tool_calls=[],
        finish_reason="stop",
    )


# Separate mock for tool-calling tests - note this must have a different name
# since it will be registered as a different activity
async def _mock_inference_with_tools(input: InferenceInput) -> InferenceOutput:
    """Mock inference that returns tool calls on first call, then text response."""
    # Check if we've already made a tool call (assistant message with tool_calls exists)
    has_assistant_with_tools = any(
        m.get("role") == "assistant" and m.get("tool_calls") for m in input.messages
    )
    if not has_assistant_with_tools:
        # First call: return tool call
        return InferenceOutput(
            content=None,
            tool_calls=[
                {
                    "id": "call_test_123",
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "arguments": '{"timezone": "UTC"}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )
    # Subsequent calls: return text response
    return InferenceOutput(
        content="The time is 12:00 UTC.",
        tool_calls=[],
        finish_reason="stop",
    )


@activity.defn(name="tool_execution_activity")
async def mock_tool_execution_activity(input: ToolExecutionInput) -> str:
    """Mock tool execution that returns a simple result."""
    return f"Mock tool result for {input.tool_name}"


@activity.defn(name="get_agent_config_activity")
async def mock_get_agent_config_activity(agent_name: str) -> AgentConfigOutput:
    """Mock agent config activity."""
    return AgentConfigOutput(
        agent_name=agent_name,
        prompt=f"You are agent {agent_name}.",
        model="gpt-4o",
        max_iterations=50,
        tools=[],
    )


@pytest.fixture
async def temporal_client() -> AsyncGenerator[Client, None]:
    """
    Connect to the real Temporal server.

    Assumes Temporal is running at localhost:7233 (started by CI or locally).
    """
    client = await Client.connect("localhost:7233")
    yield client


@pytest.fixture
async def integration_worker(temporal_client: Client) -> AsyncGenerator[Worker, None]:
    """
    Create a worker connected to real Temporal with mock activities.

    Uses a unique task queue per test to avoid conflicts.
    """
    task_queue = f"integration-test-{uuid.uuid4().hex[:8]}"

    worker = Worker(
        temporal_client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        activities=[
            mock_inference_activity,
            mock_tool_execution_activity,
            mock_get_agent_config_activity,
        ],
    )

    async with worker:
        yield worker


@pytest.fixture
def task_queue(integration_worker: Worker) -> str:
    """Get the task queue for the current test's worker."""
    return integration_worker.task_queue


# Create a decorated version of the tool-calling mock
mock_inference_activity_with_tools = activity.defn(name="inference_activity")(
    _mock_inference_with_tools
)


@pytest.fixture
async def integration_worker_with_tools(
    temporal_client: Client,
) -> AsyncGenerator[Worker, None]:
    """
    Create a worker that uses the tool-calling mock inference activity.

    This mock returns a tool call on first invocation, then a text response.
    """
    task_queue = f"integration-test-tools-{uuid.uuid4().hex[:8]}"

    worker = Worker(
        temporal_client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        activities=[
            mock_inference_activity_with_tools,
            mock_tool_execution_activity,
            mock_get_agent_config_activity,
        ],
    )

    async with worker:
        yield worker


@pytest.fixture
def task_queue_tools(integration_worker_with_tools: Worker) -> str:
    """Get the task queue for the tool-calling worker."""
    return integration_worker_with_tools.task_queue
