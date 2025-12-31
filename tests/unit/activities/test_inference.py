"""Tests for inference activity."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from temporalio.testing import ActivityEnvironment

from zap_ai.activities import InferenceInput, InferenceOutput, inference_activity


@pytest.fixture
def activity_env() -> ActivityEnvironment:
    """Create an activity testing environment."""
    return ActivityEnvironment()


class TestInferenceInput:
    """Tests for InferenceInput dataclass."""

    def test_create_minimal_input(self) -> None:
        """Test creating input with only required fields."""
        input = InferenceInput(
            agent_name="TestAgent",
            model="gpt-4o",
        )
        assert input.agent_name == "TestAgent"
        assert input.model == "gpt-4o"
        assert input.messages == []
        assert input.tools == []

    def test_create_full_input(self) -> None:
        """Test creating input with all fields."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        tools = [{"type": "function", "function": {"name": "search"}}]

        input = InferenceInput(
            agent_name="TestAgent",
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
        )

        assert input.model == "gpt-4o-mini"
        assert input.messages == messages
        assert input.tools == tools


class TestInferenceOutput:
    """Tests for InferenceOutput dataclass."""

    def test_create_with_content(self) -> None:
        """Test creating output with content."""
        output = InferenceOutput(
            content="Hello!",
            tool_calls=[],
            finish_reason="stop",
        )
        assert output.content == "Hello!"
        assert output.tool_calls == []
        assert output.finish_reason == "stop"

    def test_create_with_tool_calls(self) -> None:
        """Test creating output with tool calls."""
        tool_calls = [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "test"}'},
            }
        ]
        output = InferenceOutput(
            content=None,
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )
        assert output.content is None
        assert output.tool_calls == tool_calls
        assert output.finish_reason == "tool_calls"

    def test_default_values(self) -> None:
        """Test default values."""
        output = InferenceOutput(content="Test")
        assert output.tool_calls == []
        assert output.finish_reason == "stop"


class TestInferenceActivity:
    """Tests for the inference activity function."""

    @pytest.fixture
    def mock_complete(self, mocker: Any) -> MagicMock:
        """Mock the complete function."""
        mock_result = MagicMock()
        mock_result.content = "Hello! How can I help you?"
        mock_result.tool_calls = []
        mock_result.finish_reason = "stop"

        mock = mocker.patch(
            "zap_ai.activities.inference.complete",
            new_callable=AsyncMock,
            return_value=mock_result,
        )
        return mock

    @pytest.mark.asyncio
    async def test_activity_calls_complete(
        self, activity_env: ActivityEnvironment, mock_complete: MagicMock
    ) -> None:
        """Test that activity calls the complete function."""
        input = InferenceInput(
            agent_name="TestAgent",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, world!"}],
            tools=[],
        )

        result = await activity_env.run(inference_activity, input)

        mock_complete.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello, world!"}],
            tools=None,
        )
        assert result is not None
        assert isinstance(result, InferenceOutput)
        assert result.content == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_activity_passes_tools_to_complete(
        self, activity_env: ActivityEnvironment, mock_complete: MagicMock
    ) -> None:
        """Test that activity passes tools to complete function."""
        tools = [{"type": "function", "function": {"name": "search"}}]
        input = InferenceInput(
            agent_name="TestAgent",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=tools,
        )

        await activity_env.run(inference_activity, input)

        mock_complete.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=tools,
        )

    @pytest.mark.asyncio
    async def test_activity_returns_tool_calls(
        self, activity_env: ActivityEnvironment, mocker: Any
    ) -> None:
        """Test that activity returns tool calls from LLM."""
        mock_tool_call = MagicMock()
        mock_tool_call.to_litellm.return_value = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query": "test"}'},
        }

        mock_result = MagicMock()
        mock_result.content = None
        mock_result.tool_calls = [mock_tool_call]
        mock_result.finish_reason = "tool_calls"

        mocker.patch(
            "zap_ai.activities.inference.complete",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        input = InferenceInput(
            agent_name="TestAgent",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Search for test"}],
            tools=[{"type": "function", "function": {"name": "search"}}],
        )

        result = await activity_env.run(inference_activity, input)

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["function"]["name"] == "search"
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_activity_handles_empty_tools(
        self, activity_env: ActivityEnvironment, mock_complete: MagicMock
    ) -> None:
        """Test that activity passes None for empty tools list."""
        input = InferenceInput(
            agent_name="TestAgent",
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[],
        )

        await activity_env.run(inference_activity, input)

        mock_complete.assert_called_once_with(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=None,
        )
