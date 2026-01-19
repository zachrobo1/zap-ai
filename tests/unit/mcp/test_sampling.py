"""Tests for MCP sampling support."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zap_ai.mcp.sampling import LiteLLMSamplingHandler, create_mcp_client
from zap_ai.tracing import TraceContext


class TestLiteLLMSamplingHandler:
    """Tests for LiteLLMSamplingHandler class."""

    def test_init_defaults(self) -> None:
        """Test default initialization values."""
        handler = LiteLLMSamplingHandler()

        assert handler.default_model == "gpt-4o"
        assert handler.default_temperature == 0.7
        assert handler.default_max_tokens is None

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        handler = LiteLLMSamplingHandler(
            default_model="claude-3-opus",
            default_temperature=0.5,
            default_max_tokens=1000,
        )

        assert handler.default_model == "claude-3-opus"
        assert handler.default_temperature == 0.5
        assert handler.default_max_tokens == 1000

    @pytest.mark.asyncio
    async def test_call_converts_messages_with_text_content(self, mocker) -> None:
        """Test __call__ properly converts TextContent messages."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="test response")

        handler = LiteLLMSamplingHandler()

        # Create mock message with TextContent
        msg = MagicMock(role="user")
        msg.content = MagicMock(text="Hello, world!")
        params = MagicMock(systemPrompt="You are helpful", modelPreferences=None)

        result = await handler([msg], params, MagicMock())

        assert result == "test response"
        call_args = mock_complete.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_call_handles_string_content(self, mocker) -> None:
        """Test __call__ handles plain string content."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response")

        handler = LiteLLMSamplingHandler()

        # Message with direct string content (no .text attribute)
        msg = MagicMock(role="assistant")
        msg.content = "Plain text content"
        params = MagicMock(systemPrompt=None, modelPreferences=None)

        result = await handler([msg], params, MagicMock())

        assert result == "response"
        call_args = mock_complete.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 1  # No system prompt
        assert messages[0]["content"] == "Plain text content"

    @pytest.mark.asyncio
    async def test_call_extracts_model_from_hints(self, mocker) -> None:
        """Test that model is extracted from modelPreferences hints."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response")

        handler = LiteLLMSamplingHandler()

        # Create mock with model hint
        hint = MagicMock()
        hint.name = "claude-3-sonnet"
        prefs = MagicMock(hints=[hint])
        params = MagicMock(systemPrompt=None, modelPreferences=prefs)
        msg = MagicMock(role="user", content="test")

        await handler([msg], params, MagicMock())

        assert mock_complete.call_args.kwargs["model"] == "claude-3-sonnet"

    @pytest.mark.asyncio
    async def test_call_uses_default_model_without_hints(self, mocker) -> None:
        """Test that default model is used when no hints provided."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response")

        handler = LiteLLMSamplingHandler(default_model="gpt-4-turbo")

        params = MagicMock(systemPrompt=None, modelPreferences=None)
        msg = MagicMock(role="user", content="test")

        await handler([msg], params, MagicMock())

        assert mock_complete.call_args.kwargs["model"] == "gpt-4-turbo"

    @pytest.mark.asyncio
    async def test_call_uses_default_model_with_empty_hints(self, mocker) -> None:
        """Test that default model is used when hints list is empty."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response")

        handler = LiteLLMSamplingHandler()

        prefs = MagicMock(hints=[])
        params = MagicMock(systemPrompt=None, modelPreferences=prefs)
        msg = MagicMock(role="user", content="test")

        await handler([msg], params, MagicMock())

        assert mock_complete.call_args.kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_call_passes_temperature(self, mocker) -> None:
        """Test that temperature is passed to complete."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response")

        handler = LiteLLMSamplingHandler()

        params = MagicMock(
            systemPrompt=None,
            modelPreferences=None,
            temperature=0.3,
            maxTokens=None,
        )
        msg = MagicMock(role="user", content="test")

        await handler([msg], params, MagicMock())

        assert mock_complete.call_args.kwargs["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_call_passes_max_tokens(self, mocker) -> None:
        """Test that max_tokens is passed to complete."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response")

        handler = LiteLLMSamplingHandler()

        params = MagicMock(
            systemPrompt=None,
            modelPreferences=None,
            temperature=None,
            maxTokens=500,
        )
        msg = MagicMock(role="user", content="test")

        await handler([msg], params, MagicMock())

        assert mock_complete.call_args.kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_call_returns_empty_string_when_content_none(self, mocker) -> None:
        """Test that empty string is returned when result.content is None."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content=None)

        handler = LiteLLMSamplingHandler()

        params = MagicMock(systemPrompt=None, modelPreferences=None)
        msg = MagicMock(role="user", content="test")

        result = await handler([msg], params, MagicMock())

        assert result == ""


class TestCreateMcpClient:
    """Tests for create_mcp_client factory function."""

    @pytest.fixture
    def mock_fastmcp(self):
        """Mock fastmcp module to avoid importing it (which triggers beartype pollution)."""
        mock_client_class = MagicMock()
        mock_fastmcp_module = MagicMock()
        mock_fastmcp_module.Client = mock_client_class

        with patch.dict(sys.modules, {"fastmcp": mock_fastmcp_module}):
            yield mock_client_class

    def test_none_handler(self, mock_fastmcp) -> None:
        """Test creating client with no sampling handler."""
        create_mcp_client("server.py")

        mock_fastmcp.assert_called_once_with("server.py", sampling_handler=None)

    def test_litellm_string_handler(self, mock_fastmcp) -> None:
        """Test creating client with 'litellm' string handler."""
        create_mcp_client("server.py", sampling_handler="litellm")

        call_args = mock_fastmcp.call_args
        handler = call_args.kwargs["sampling_handler"]
        assert isinstance(handler, LiteLLMSamplingHandler)
        assert handler.default_model == "gpt-4o"

    def test_litellm_with_custom_model(self, mock_fastmcp) -> None:
        """Test 'litellm' handler with custom sampling_model."""
        create_mcp_client(
            "server.py",
            sampling_handler="litellm",
            sampling_model="claude-3-opus",
        )

        call_args = mock_fastmcp.call_args
        handler = call_args.kwargs["sampling_handler"]
        assert isinstance(handler, LiteLLMSamplingHandler)
        assert handler.default_model == "claude-3-opus"

    def test_custom_callable_handler(self, mock_fastmcp) -> None:
        """Test creating client with custom callable handler."""

        async def custom_handler(messages, params, context):
            return "custom response"

        create_mcp_client("server.py", sampling_handler=custom_handler)

        call_args = mock_fastmcp.call_args
        assert call_args.kwargs["sampling_handler"] is custom_handler

    def test_passes_additional_kwargs(self, mock_fastmcp) -> None:
        """Test that additional kwargs are passed to Client."""
        create_mcp_client("server.py", timeout=30, extra_param="value")

        mock_fastmcp.assert_called_once_with(
            "server.py",
            sampling_handler=None,
            timeout=30,
            extra_param="value",
        )

    def test_returns_client_instance(self, mock_fastmcp) -> None:
        """Test that function returns the Client instance."""
        mock_client = MagicMock()
        mock_fastmcp.return_value = mock_client

        result = create_mcp_client("server.py")

        assert result is mock_client


class TestSamplingTraceNesting:
    """Tests for sampling trace nesting behavior."""

    @pytest.mark.asyncio
    async def test_nests_under_parent_context_when_set(self, mocker) -> None:
        """Test that sampling nests under parent context when set via set_trace_context."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response", usage=None)

        # Create mock tracer
        mock_tracer = MagicMock()
        mock_sampling_ctx = MagicMock()
        mock_gen_ctx = MagicMock()

        # Set up async context manager for start_observation
        mock_observation_cm = MagicMock()
        mock_observation_cm.__aenter__ = AsyncMock(return_value=mock_sampling_ctx)
        mock_observation_cm.__aexit__ = AsyncMock(return_value=None)
        mock_tracer.start_observation.return_value = mock_observation_cm

        mock_tracer.start_generation = AsyncMock(return_value=mock_gen_ctx)
        mock_tracer.end_generation = AsyncMock()

        mocker.patch("zap_ai.mcp.sampling.get_tracing_provider", return_value=mock_tracer)

        handler = LiteLLMSamplingHandler(enable_tracing=True)

        # Set parent context directly on handler (simulating what tool_execution does)
        parent_ctx = TraceContext(trace_id="parent-trace", span_id="parent-span")
        handler.set_trace_context(parent_ctx)

        params = MagicMock(systemPrompt=None, modelPreferences=None)
        msg = MagicMock(role="user", content="test")

        result = await handler([msg], params, MagicMock())

        assert result == "response"
        # Should call start_observation with parent context, not start_trace
        mock_tracer.start_observation.assert_called_once()
        call_kwargs = mock_tracer.start_observation.call_args.kwargs
        assert call_kwargs["parent_context"] is parent_ctx
        assert "mcp-sampling" in call_kwargs["name"]
        # Should NOT call start_trace
        mock_tracer.start_trace.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_error_when_no_parent_context(self, mocker) -> None:
        """Test that sampling raises error when tracing enabled but no parent context."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response", usage=None)

        # Create mock tracer
        mock_tracer = MagicMock()
        mocker.patch("zap_ai.mcp.sampling.get_tracing_provider", return_value=mock_tracer)

        # Handler with tracing enabled but NO trace context set
        handler = LiteLLMSamplingHandler(enable_tracing=True)

        params = MagicMock(systemPrompt=None, modelPreferences=None)
        msg = MagicMock(role="user", content="test")

        with pytest.raises(RuntimeError) as exc_info:
            await handler([msg], params, MagicMock())

        assert "no parent trace context is available" in str(exc_info.value)
        # Should not call any tracing methods
        mock_tracer.start_trace.assert_not_called()
        mock_tracer.start_observation.assert_not_called()

    @pytest.mark.asyncio
    async def test_generation_nested_under_sampling_span(self, mocker) -> None:
        """Test that generation is nested under the sampling span."""
        mock_complete = mocker.patch(
            "zap_ai.llm.provider.complete",
            new_callable=AsyncMock,
        )
        mock_complete.return_value = MagicMock(content="response", usage={"total_tokens": 10})

        # Create mock tracer
        mock_tracer = MagicMock()
        mock_sampling_ctx = MagicMock()
        mock_gen_ctx = MagicMock()

        # Set up async context manager for start_observation
        mock_observation_cm = MagicMock()
        mock_observation_cm.__aenter__ = AsyncMock(return_value=mock_sampling_ctx)
        mock_observation_cm.__aexit__ = AsyncMock(return_value=None)
        mock_tracer.start_observation.return_value = mock_observation_cm

        mock_tracer.start_generation = AsyncMock(return_value=mock_gen_ctx)
        mock_tracer.end_generation = AsyncMock()

        mocker.patch("zap_ai.mcp.sampling.get_tracing_provider", return_value=mock_tracer)

        handler = LiteLLMSamplingHandler(enable_tracing=True)

        # Set parent context directly on handler
        parent_ctx = TraceContext(trace_id="parent-trace", span_id="parent-span")
        handler.set_trace_context(parent_ctx)

        params = MagicMock(systemPrompt=None, modelPreferences=None)
        msg = MagicMock(role="user", content="test")

        await handler([msg], params, MagicMock())

        # Verify generation uses sampling context as parent
        mock_tracer.start_generation.assert_called_once()
        gen_call_kwargs = mock_tracer.start_generation.call_args.kwargs
        assert gen_call_kwargs["parent_context"] is mock_sampling_ctx

        # Verify end_generation was called with correct context
        mock_tracer.end_generation.assert_called_once()
        end_call_kwargs = mock_tracer.end_generation.call_args.kwargs
        assert end_call_kwargs["context"] is mock_gen_ctx

    def test_set_and_clear_trace_context(self) -> None:
        """Test set_trace_context and clear_trace_context methods."""
        handler = LiteLLMSamplingHandler(enable_tracing=True)

        # Initially None
        assert handler._trace_context is None

        # Set context
        ctx = TraceContext(trace_id="test-trace", span_id="test-span")
        handler.set_trace_context(ctx)
        assert handler._trace_context is ctx

        # Clear context
        handler.clear_trace_context()
        assert handler._trace_context is None
