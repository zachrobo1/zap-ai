"""Tests for LLM provider."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zap_ai.llm.message_types import Message
from zap_ai.llm.provider import (
    LLMProviderError,
    complete,
    convert_messages_to_litellm,
)


class TestComplete:
    """Tests for complete function."""

    @pytest.mark.asyncio
    async def test_complete_returns_inference_result(self) -> None:
        """Test that complete returns InferenceResult."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hello back!", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            result = await complete(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert result.content == "Hello back!"
            assert result.finish_reason == "stop"
            assert result.tool_calls == []
            assert result.usage["total_tokens"] == 15

    @pytest.mark.asyncio
    async def test_complete_parses_tool_calls(self) -> None:
        """Test that complete parses tool calls from response."""
        mock_function = MagicMock()
        mock_function.name = "search"
        mock_function.arguments = '{"query": "test"}'

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.function = mock_function

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content=None, tool_calls=[mock_tool_call]),
                finish_reason="tool_calls",
            )
        ]
        mock_response.usage = None

        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            result = await complete(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Search for test"}],
                tools=[{"type": "function", "function": {"name": "search"}}],
            )

            assert result.content is None
            assert len(result.tool_calls) == 1
            assert result.tool_calls[0].id == "call_123"
            assert result.tool_calls[0].name == "search"
            assert result.tool_calls[0].arguments == {"query": "test"}
            assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_complete_passes_tools_to_litellm(self) -> None:
        """Test that tools are passed to LiteLLM."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="OK", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = None

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            await complete(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Weather?"}],
                tools=tools,
            )

            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["tools"] == tools
            assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_complete_without_tools(self) -> None:
        """Test that tool_choice is not set when no tools."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Hi", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = None

        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            await complete(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello"}],
            )

            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert "tools" not in call_kwargs
            assert "tool_choice" not in call_kwargs

    @pytest.mark.asyncio
    async def test_complete_passes_temperature(self) -> None:
        """Test that temperature is passed to LiteLLM."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Creative", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = None

        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            await complete(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Be creative"}],
                temperature=1.5,
            )

            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["temperature"] == 1.5

    @pytest.mark.asyncio
    async def test_complete_passes_max_tokens(self) -> None:
        """Test that max_tokens is passed when specified."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Short", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = None

        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            await complete(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Short response"}],
                max_tokens=50,
            )

            call_kwargs = mock_litellm.acompletion.call_args.kwargs
            assert call_kwargs["max_tokens"] == 50

    @pytest.mark.asyncio
    async def test_complete_raises_on_error(self) -> None:
        """Test that errors are wrapped in LLMProviderError."""
        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(side_effect=Exception("API Error"))

            with pytest.raises(LLMProviderError, match="LLM call failed"):
                await complete(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Hello"}],
                )

    @pytest.mark.asyncio
    async def test_complete_handles_missing_usage(self) -> None:
        """Test handling response without usage info."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Test", tool_calls=None),
                finish_reason="stop",
            )
        ]
        mock_response.usage = None

        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            result = await complete(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test"}],
            )

            assert result.usage == {}

    @pytest.mark.asyncio
    async def test_complete_handles_missing_finish_reason(self) -> None:
        """Test handling response without finish_reason."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Test", tool_calls=None),
                finish_reason=None,
            )
        ]
        mock_response.usage = None

        with patch("zap_ai.llm.provider.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            result = await complete(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Test"}],
            )

            assert result.finish_reason == "stop"


class TestConvertMessagesToLitellm:
    """Tests for convert_messages_to_litellm function."""

    def test_converts_message_objects(self) -> None:
        """Test converting Message objects."""
        messages = [
            Message.system("You are helpful"),
            Message.user("Hello"),
            Message.assistant(content="Hi there"),
        ]

        result = convert_messages_to_litellm(messages)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are helpful"}
        assert result[1] == {"role": "user", "content": "Hello"}
        assert result[2] == {"role": "assistant", "content": "Hi there"}

    def test_passes_through_dicts(self) -> None:
        """Test that dicts are passed through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        result = convert_messages_to_litellm(messages)

        assert result == messages

    def test_handles_mixed_types(self) -> None:
        """Test handling mix of Message objects and dicts."""
        messages = [
            Message.user("Hello"),
            {"role": "assistant", "content": "Hi"},
        ]

        result = convert_messages_to_litellm(messages)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi"}

    def test_raises_on_unknown_type(self) -> None:
        """Test that unknown types raise ValueError."""
        messages = ["not a message"]  # type: ignore

        with pytest.raises(ValueError, match="Unknown message type"):
            convert_messages_to_litellm(messages)

    def test_handles_empty_list(self) -> None:
        """Test handling empty message list."""
        result = convert_messages_to_litellm([])
        assert result == []

    def test_converts_tool_calls(self) -> None:
        """Test that tool calls are properly converted."""
        from zap_ai.llm.message_types import ToolCall

        tc = ToolCall(
            id="call_1",
            name="search",
            arguments={"q": "test"},
            arguments_raw='{"q": "test"}',
        )
        messages = [Message.assistant(tool_calls=[tc])]

        result = convert_messages_to_litellm(messages)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["function"]["name"] == "search"


class TestLLMProviderError:
    """Tests for LLMProviderError exception."""

    def test_error_is_exception(self) -> None:
        """Test that LLMProviderError is an Exception."""
        error = LLMProviderError("Test error")
        assert isinstance(error, Exception)

    def test_error_message(self) -> None:
        """Test error message is preserved."""
        error = LLMProviderError("API rate limit exceeded")
        assert str(error) == "API rate limit exceeded"
