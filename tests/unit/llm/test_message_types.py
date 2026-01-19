"""Tests for LLM message types."""

from zap_ai.llm.message_types import (
    ImageContent,
    InferenceResult,
    Message,
    TextContent,
    ToolCall,
    content_has_images,
)


class TestTextContent:
    """Tests for TextContent dataclass."""

    def test_to_litellm(self) -> None:
        """Test conversion to LiteLLM format."""
        content = TextContent(text="Hello world")
        result = content.to_litellm()

        assert result == {"type": "text", "text": "Hello world"}

    def test_from_litellm(self) -> None:
        """Test parsing from LiteLLM format."""
        data = {"type": "text", "text": "Test message"}
        content = TextContent.from_litellm(data)

        assert content.text == "Test message"
        assert content.type == "text"

    def test_type_is_always_text(self) -> None:
        """Test that type field is always 'text'."""
        content = TextContent(text="anything")
        assert content.type == "text"


class TestImageContent:
    """Tests for ImageContent dataclass."""

    def test_from_url(self) -> None:
        """Test creating image from URL."""
        img = ImageContent.from_url("https://example.com/image.png")

        assert img.url == "https://example.com/image.png"
        assert img.type == "image_url"
        assert img.detail is None
        assert img.format is None

    def test_from_url_with_detail(self) -> None:
        """Test creating image from URL with detail level."""
        img = ImageContent.from_url("https://example.com/img.jpg", detail="high")

        assert img.url == "https://example.com/img.jpg"
        assert img.detail == "high"

    def test_from_base64(self) -> None:
        """Test creating image from base64 data."""
        img = ImageContent.from_base64("abc123xyz", mime_type="image/jpeg")

        assert img.url == "data:image/jpeg;base64,abc123xyz"
        assert img.format == "image/jpeg"

    def test_from_base64_default_png(self) -> None:
        """Test that from_base64 defaults to PNG mime type."""
        img = ImageContent.from_base64("data")

        assert "image/png" in img.url
        assert img.format == "image/png"

    def test_to_litellm_basic(self) -> None:
        """Test basic conversion to LiteLLM format."""
        img = ImageContent(url="https://example.com/img.png")
        result = img.to_litellm()

        assert result == {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        }

    def test_to_litellm_with_detail(self) -> None:
        """Test conversion with detail level."""
        img = ImageContent(url="https://example.com/img.png", detail="low")
        result = img.to_litellm()

        assert result["image_url"]["detail"] == "low"

    def test_to_litellm_with_format(self) -> None:
        """Test conversion with explicit format."""
        img = ImageContent(url="gs://bucket/img", format="image/webp")
        result = img.to_litellm()

        assert result["image_url"]["format"] == "image/webp"

    def test_from_litellm(self) -> None:
        """Test parsing from LiteLLM format."""
        data = {
            "type": "image_url",
            "image_url": {
                "url": "https://example.com/photo.jpg",
                "detail": "auto",
            },
        }
        img = ImageContent.from_litellm(data)

        assert img.url == "https://example.com/photo.jpg"
        assert img.detail == "auto"

    def test_from_litellm_string_url(self) -> None:
        """Test parsing when image_url is just a string."""
        data = {"type": "image_url", "image_url": "https://example.com/img.png"}
        img = ImageContent.from_litellm(data)

        assert img.url == "https://example.com/img.png"


class TestContentHasImages:
    """Tests for content_has_images function."""

    def test_string_content_has_no_images(self) -> None:
        """Test that string content returns False."""
        assert content_has_images("Hello world") is False

    def test_none_content_has_no_images(self) -> None:
        """Test that None content returns False."""
        assert content_has_images(None) is False

    def test_text_only_list_has_no_images(self) -> None:
        """Test list with only text content returns False."""
        content = [TextContent(text="Hello"), TextContent(text="World")]
        assert content_has_images(content) is False

    def test_list_with_image_returns_true(self) -> None:
        """Test list with image content returns True."""
        content = [
            TextContent(text="Describe this:"),
            ImageContent.from_url("https://example.com/img.png"),
        ]
        assert content_has_images(content) is True

    def test_image_only_list_returns_true(self) -> None:
        """Test list with only images returns True."""
        content = [ImageContent.from_url("https://example.com/img.png")]
        assert content_has_images(content) is True


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

    def test_user_factory_multimodal(self) -> None:
        """Test creating user message with multimodal content."""
        content = [
            TextContent(text="Describe this:"),
            ImageContent.from_url("https://example.com/img.png"),
        ]
        msg = Message.user(content)

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_user_with_images(self) -> None:
        """Test creating user message with images helper."""
        img = ImageContent.from_url("https://example.com/img.png")
        msg = Message.user_with_images("What's in this image?", [img])

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ImageContent)

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

    def test_to_litellm_multimodal_message(self) -> None:
        """Test converting multimodal message to LiteLLM format."""
        content = [
            TextContent(text="What's in this image?"),
            ImageContent.from_url("https://example.com/img.png"),
        ]
        msg = Message.user(content)
        result = msg.to_litellm()

        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0] == {"type": "text", "text": "What's in this image?"}
        assert result["content"][1]["type"] == "image_url"

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

    def test_from_litellm_parses_multimodal(self) -> None:
        """Test parsing multimodal message from LiteLLM format."""
        raw = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            ],
        }
        msg = Message.from_litellm(raw)

        assert msg.role == "user"
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextContent)
        assert isinstance(msg.content[1], ImageContent)

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

    def test_has_images_text_content(self) -> None:
        """Test has_images returns False for text-only content."""
        msg = Message.user("Hello")
        assert msg.has_images() is False

    def test_has_images_multimodal_with_image(self) -> None:
        """Test has_images returns True when images present."""
        content = [
            TextContent(text="Describe:"),
            ImageContent.from_url("https://example.com/img.png"),
        ]
        msg = Message.user(content)
        assert msg.has_images() is True

    def test_has_images_multimodal_text_only(self) -> None:
        """Test has_images returns False for list with only text."""
        content = [TextContent(text="Hello"), TextContent(text="World")]
        msg = Message.user(content)
        assert msg.has_images() is False

    def test_get_text_content_string(self) -> None:
        """Test get_text_content returns string content as-is."""
        msg = Message.user("Hello world")
        assert msg.get_text_content() == "Hello world"

    def test_get_text_content_none(self) -> None:
        """Test get_text_content returns None for None content."""
        msg = Message.assistant()
        assert msg.get_text_content() is None

    def test_get_text_content_multimodal(self) -> None:
        """Test get_text_content extracts text from multimodal content."""
        content = [
            TextContent(text="First part"),
            ImageContent.from_url("https://example.com/img.png"),
            TextContent(text="Second part"),
        ]
        msg = Message.user(content)
        assert msg.get_text_content() == "First part Second part"


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
        """Test that message types are importable from llm module."""
        from zap_ai.llm import (
            InferenceResult,
            Message,
            ToolCall,
        )

        assert Message is not None
        assert ToolCall is not None
        assert InferenceResult is not None

    def test_provider_imports_from_provider_module(self) -> None:
        """Test that provider exports are importable from provider module."""
        from zap_ai.llm.provider import (
            LLMProviderError,
            complete,
            convert_messages_to_litellm,
        )

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
        # Provider exports are NOT in llm.__all__ (import from llm.provider instead)
        assert "complete" not in llm.__all__
        assert "LLMProviderError" not in llm.__all__
