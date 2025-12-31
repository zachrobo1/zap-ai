"""Tests for LLM message types."""

from zap_ai.llm.message_types import InferenceResult, Message, ToolCall


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_from_litellm_parses_function_call(self) -> None:
        """Test parsing tool call from LiteLLM format."""
        raw = {
            "id": "call_123",
            "function": {"name": "search", "arguments": '{"query": "test"}'},
        }
        tc = ToolCall.from_litellm(raw)

        assert tc.id == "call_123"
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}
        assert tc.arguments_raw == '{"query": "test"}'

    def test_from_litellm_handles_dict_arguments(self) -> None:
        """Test parsing when arguments is already a dict."""
        raw = {
            "id": "call_456",
            "function": {"name": "calculate", "arguments": {"x": 1, "y": 2}},
        }
        tc = ToolCall.from_litellm(raw)

        assert tc.name == "calculate"
        assert tc.arguments == {"x": 1, "y": 2}
        assert tc.arguments_raw == '{"x": 1, "y": 2}'

    def test_from_litellm_handles_invalid_json(self) -> None:
        """Test parsing when arguments is invalid JSON."""
        raw = {
            "id": "call_789",
            "function": {"name": "broken", "arguments": "not valid json{"},
        }
        tc = ToolCall.from_litellm(raw)

        assert tc.name == "broken"
        assert tc.arguments == {}
        assert tc.arguments_raw == "not valid json{"

    def test_from_litellm_handles_missing_fields(self) -> None:
        """Test parsing with missing fields."""
        raw: dict = {}
        tc = ToolCall.from_litellm(raw)

        assert tc.id == ""
        assert tc.name == ""
        assert tc.arguments == {}

    def test_to_litellm_round_trip(self) -> None:
        """Test converting back to LiteLLM format."""
        tc = ToolCall(
            id="call_abc",
            name="fetch",
            arguments={"url": "http://example.com"},
            arguments_raw='{"url": "http://example.com"}',
        )
        result = tc.to_litellm()

        assert result["id"] == "call_abc"
        assert result["type"] == "function"
        assert result["function"]["name"] == "fetch"
        assert result["function"]["arguments"] == '{"url": "http://example.com"}'


class TestMessage:
    """Tests for Message dataclass."""

    def test_system_factory(self) -> None:
        """Test creating system message."""
        msg = Message.system("You are helpful")

        assert msg.role == "system"
        assert msg.content == "You are helpful"
        assert msg.tool_calls == []

    def test_user_factory(self) -> None:
        """Test creating user message."""
        msg = Message.user("Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_factory_with_content(self) -> None:
        """Test creating assistant message with content."""
        msg = Message.assistant(content="Hi there")

        assert msg.role == "assistant"
        assert msg.content == "Hi there"
        assert msg.tool_calls == []

    def test_assistant_factory_with_tool_calls(self) -> None:
        """Test creating assistant message with tool calls."""
        tc = ToolCall(id="call_1", name="test", arguments={}, arguments_raw="{}")
        msg = Message.assistant(tool_calls=[tc])

        assert msg.role == "assistant"
        assert msg.content is None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "test"

    def test_tool_result_factory(self) -> None:
        """Test creating tool result message."""
        msg = Message.tool_result(tool_call_id="call_1", name="search", content="Found 5 results")

        assert msg.role == "tool"
        assert msg.content == "Found 5 results"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "search"

    def test_to_litellm_user_message(self) -> None:
        """Test converting user message to LiteLLM format."""
        msg = Message.user("Test")
        result = msg.to_litellm()

        assert result == {"role": "user", "content": "Test"}

    def test_to_litellm_assistant_with_tool_calls(self) -> None:
        """Test converting assistant message with tool calls."""
        tc = ToolCall(
            id="call_1", name="search", arguments={"q": "test"}, arguments_raw='{"q": "test"}'
        )
        msg = Message.assistant(tool_calls=[tc])
        result = msg.to_litellm()

        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "search"

    def test_to_litellm_tool_result(self) -> None:
        """Test converting tool result to LiteLLM format."""
        msg = Message.tool_result("call_1", "search", "Result here")
        result = msg.to_litellm()

        assert result["role"] == "tool"
        assert result["content"] == "Result here"
        assert result["tool_call_id"] == "call_1"
        assert result["name"] == "search"

    def test_from_litellm_parses_message(self) -> None:
        """Test parsing message from LiteLLM format."""
        raw = {"role": "user", "content": "Hello"}
        msg = Message.from_litellm(raw)

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_from_litellm_parses_tool_calls(self) -> None:
        """Test parsing message with tool calls."""
        raw = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "function": {"name": "test", "arguments": "{}"}}],
        }
        msg = Message.from_litellm(raw)

        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "test"

    def test_from_litellm_handles_empty(self) -> None:
        """Test parsing empty/minimal message."""
        raw: dict = {}
        msg = Message.from_litellm(raw)

        assert msg.role == ""
        assert msg.content is None


class TestInferenceResult:
    """Tests for InferenceResult dataclass."""

    def test_has_tool_calls_true(self) -> None:
        """Test has_tool_calls with tool calls present."""
        tc = ToolCall(id="1", name="test", arguments={}, arguments_raw="{}")
        result = InferenceResult(tool_calls=[tc])

        assert result.has_tool_calls is True

    def test_has_tool_calls_false(self) -> None:
        """Test has_tool_calls with no tool calls."""
        result = InferenceResult(content="Hello")

        assert result.has_tool_calls is False

    def test_is_complete_true(self) -> None:
        """Test is_complete when no tool calls."""
        result = InferenceResult(content="Done", finish_reason="stop")

        assert result.is_complete is True

    def test_is_complete_false(self) -> None:
        """Test is_complete when has tool calls."""
        tc = ToolCall(id="1", name="test", arguments={}, arguments_raw="{}")
        result = InferenceResult(tool_calls=[tc], finish_reason="tool_calls")

        assert result.is_complete is False

    def test_to_message_with_content(self) -> None:
        """Test converting result to message with content."""
        result = InferenceResult(content="Response", finish_reason="stop")
        msg = result.to_message()

        assert msg.role == "assistant"
        assert msg.content == "Response"
        assert msg.tool_calls == []

    def test_to_message_with_tool_calls(self) -> None:
        """Test converting result to message with tool calls."""
        tc = ToolCall(id="call_1", name="search", arguments={}, arguments_raw="{}")
        result = InferenceResult(tool_calls=[tc])
        msg = result.to_message()

        assert msg.role == "assistant"
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_default_values(self) -> None:
        """Test default values are set correctly."""
        result = InferenceResult()

        assert result.content is None
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert result.usage == {}

    def test_usage_tracking(self) -> None:
        """Test usage dict is preserved."""
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        result = InferenceResult(content="test", usage=usage)

        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 20
        assert result.usage["total_tokens"] == 30


class TestLLMModuleImports:
    """Tests for LLM module imports."""

    def test_imports_from_module(self) -> None:
        """Test that all exports are importable from module."""
        from zap_ai.llm import (
            InferenceResult,
            LLMProviderError,
            Message,
            ToolCall,
            complete,
            convert_messages_to_litellm,
        )

        assert Message is not None
        assert ToolCall is not None
        assert InferenceResult is not None
        assert complete is not None
        assert convert_messages_to_litellm is not None
        assert LLMProviderError is not None

    def test_all_exports_defined(self) -> None:
        """Test that __all__ is properly defined."""
        from zap_ai import llm

        assert hasattr(llm, "__all__")
        assert "Message" in llm.__all__
        assert "ToolCall" in llm.__all__
        assert "InferenceResult" in llm.__all__
        assert "complete" in llm.__all__
        assert "LLMProviderError" in llm.__all__
