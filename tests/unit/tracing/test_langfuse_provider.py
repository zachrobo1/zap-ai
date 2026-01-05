"""Tests for LangfuseTracingProvider with mocked SDK."""

from unittest.mock import MagicMock, patch

import pytest

from zap_ai.tracing.protocol import ObservationType, TraceContext

# Skip all tests in this module if langfuse is not installed
try:
    import langfuse  # noqa: F401

    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

pytestmark = pytest.mark.skipif(not HAS_LANGFUSE, reason="langfuse not installed")


@pytest.fixture
def sample_context():
    """Create a sample TraceContext."""
    return TraceContext(
        trace_id="00000000000000000000000000000001",
        span_id="0000000000000001",
        provider_data={
            "langfuse_trace_id": "00000000000000000000000000000001",
            "langfuse_root_span_id": "span-123",
        },
    )


class TestLangfuseTracingProvider:
    """Tests for LangfuseTracingProvider."""

    @pytest.fixture
    def mock_langfuse_instance(self):
        """Create a mock Langfuse instance."""
        mock = MagicMock()
        # Mock create_trace_id
        mock.create_trace_id = MagicMock(return_value="00000000000000000000000000000001")
        # Mock start_span
        mock_span = MagicMock()
        mock_span.id = "span-123"
        mock_span.update_trace = MagicMock()
        mock_span.end = MagicMock()
        mock_span.update = MagicMock()
        mock_span.create_event = MagicMock()
        mock.start_span = MagicMock(return_value=mock_span)
        # Mock start_observation (used for generations)
        mock_gen = MagicMock()
        mock_gen.id = "gen-456"
        mock_gen.update = MagicMock()
        mock_gen.end = MagicMock()
        mock.start_observation = MagicMock(return_value=mock_gen)
        return mock

    @pytest.fixture
    def provider(self, mock_langfuse_instance):
        """Create provider with mocked Langfuse."""
        from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider

        provider = LangfuseTracingProvider.__new__(LangfuseTracingProvider)
        provider._langfuse = mock_langfuse_instance
        provider._active_observations = {}
        return provider

    def test_observation_type_mapping(self, provider):
        """All observation types should map to Langfuse types."""
        assert provider._observation_type_to_langfuse(ObservationType.SPAN) == "span"
        assert provider._observation_type_to_langfuse(ObservationType.GENERATION) == "generation"
        assert provider._observation_type_to_langfuse(ObservationType.TOOL) == "tool"
        assert provider._observation_type_to_langfuse(ObservationType.AGENT) == "agent"

    @pytest.mark.asyncio
    async def test_start_trace(self, provider, mock_langfuse_instance):
        """start_trace should create a Langfuse span (trace is implicit in v3)."""
        async with provider.start_trace(
            name="test-trace",
            session_id="session-123",
            user_id="user-456",
            metadata={"key": "value"},
            tags=["tag1"],
        ) as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id is not None
            assert ctx.span_id is not None
            assert ctx.provider_data is not None

            # In v3, start_span is called instead of trace
            mock_langfuse_instance.start_span.assert_called()
            call_kwargs = mock_langfuse_instance.start_span.call_args.kwargs
            assert call_kwargs["name"] == "test-trace"
            assert call_kwargs["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_start_observation_span(self, provider, sample_context, mock_langfuse_instance):
        """start_observation should create a Langfuse span."""
        async with provider.start_observation(
            name="iteration-0",
            observation_type=ObservationType.SPAN,
            parent_context=sample_context,
            metadata={"iteration": 0},
            input_data={"task": "test"},
        ) as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id == sample_context.trace_id
            assert ctx.span_id is not None

            mock_langfuse_instance.start_span.assert_called()
            call_kwargs = mock_langfuse_instance.start_span.call_args.kwargs
            assert call_kwargs["name"] == "iteration-0"
            assert "observation_type" in call_kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_start_observation_tool(self, provider, sample_context, mock_langfuse_instance):
        """start_observation with TOOL type should work."""
        async with provider.start_observation(
            name="tool-search",
            observation_type=ObservationType.TOOL,
            parent_context=sample_context,
            input_data={"query": "test"},
        ) as ctx:
            assert ctx.trace_id == sample_context.trace_id

            call_kwargs = mock_langfuse_instance.start_span.call_args.kwargs
            assert call_kwargs["name"] == "tool-search"
            assert call_kwargs["input"] == {"query": "test"}

    @pytest.mark.asyncio
    async def test_start_generation(self, provider, sample_context, mock_langfuse_instance):
        """start_generation should create a Langfuse generation via start_observation."""
        ctx = await provider.start_generation(
            name="inference-agent",
            parent_context=sample_context,
            model="gpt-4o",
            input_messages=[{"role": "user", "content": "Hello"}],
            metadata={"temperature": 0.7},
        )

        assert isinstance(ctx, TraceContext)
        assert ctx.trace_id == sample_context.trace_id
        assert "langfuse_generation_id" in ctx.provider_data

        # In v3, we use start_observation with as_type='generation'
        mock_langfuse_instance.start_observation.assert_called_once()
        call_kwargs = mock_langfuse_instance.start_observation.call_args.kwargs
        assert call_kwargs["name"] == "inference-agent"
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["as_type"] == "generation"
        assert call_kwargs["input"] == [{"role": "user", "content": "Hello"}]
        assert call_kwargs["metadata"] == {"temperature": 0.7}

    @pytest.mark.asyncio
    async def test_end_generation(self, provider, sample_context, mock_langfuse_instance):
        """end_generation should end the Langfuse generation."""
        # First start a generation
        gen_ctx = await provider.start_generation(
            name="inference",
            parent_context=sample_context,
            model="gpt-4o",
            input_messages=[],
        )

        # Get the mock generation
        mock_gen = mock_langfuse_instance.start_observation.return_value

        await provider.end_generation(
            context=gen_ctx,
            output={"content": "Hello!", "tool_calls": []},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        mock_gen.update.assert_called_once()
        mock_gen.end.assert_called_once()
        call_kwargs = mock_gen.update.call_args.kwargs
        assert call_kwargs["output"] == {"content": "Hello!", "tool_calls": []}
        # v3 uses usage_details format
        assert call_kwargs["usage_details"] == {"input": 10, "output": 5}

    @pytest.mark.asyncio
    async def test_add_event(self, provider, mock_langfuse_instance):
        """add_event should add an event to the observation."""
        # First create a trace to have an active observation
        async with provider.start_trace(name="test-trace") as ctx:
            mock_span = mock_langfuse_instance.start_span.return_value

            await provider.add_event(
                context=ctx,
                name="status-change",
                attributes={"old": "running", "new": "completed"},
            )

            mock_span.create_event.assert_called_once()
            call_kwargs = mock_span.create_event.call_args.kwargs
            assert call_kwargs["name"] == "status-change"
            assert call_kwargs["metadata"] == {"old": "running", "new": "completed"}

    @pytest.mark.asyncio
    async def test_set_error(self, provider, sample_context, mock_langfuse_instance):
        """set_error should mark the observation as errored."""
        gen_ctx = await provider.start_generation(
            name="inference",
            parent_context=sample_context,
            model="gpt-4o",
            input_messages=[],
        )

        mock_gen = mock_langfuse_instance.start_observation.return_value

        test_error = ValueError("API error")
        await provider.set_error(context=gen_ctx, error=test_error)

        mock_gen.update.assert_called_once()
        call_kwargs = mock_gen.update.call_args.kwargs
        assert call_kwargs["level"] == "ERROR"
        assert "API error" in call_kwargs["status_message"]

    @pytest.mark.asyncio
    async def test_flush(self, provider, mock_langfuse_instance):
        """flush should call Langfuse flush."""
        await provider.flush()
        mock_langfuse_instance.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown(self, provider, mock_langfuse_instance):
        """shutdown should call Langfuse shutdown."""
        await provider.shutdown()
        mock_langfuse_instance.shutdown.assert_called_once()


class TestLangfuseProviderInit:
    """Test provider initialization."""

    def test_init_creates_langfuse(self):
        """Provider should create Langfuse client."""
        with patch("zap_ai.tracing.langfuse_provider.Langfuse") as mock_cls:
            mock_instance = MagicMock()
            mock_cls.return_value = mock_instance

            from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider

            provider = LangfuseTracingProvider(
                public_key="pk-test",
                secret_key="sk-test",
                host="https://custom.langfuse.com",
            )

            mock_cls.assert_called_once_with(
                public_key="pk-test",
                secret_key="sk-test",
                host="https://custom.langfuse.com",
            )
            assert provider._langfuse is mock_instance

    def test_init_with_defaults(self):
        """Provider should accept env var defaults."""
        with patch("zap_ai.tracing.langfuse_provider.Langfuse") as mock_cls:
            from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider

            LangfuseTracingProvider()

            mock_cls.assert_called_once_with(
                public_key=None,
                secret_key=None,
                host=None,
            )
