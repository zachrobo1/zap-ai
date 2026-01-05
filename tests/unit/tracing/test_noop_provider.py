"""Tests for NoOpTracingProvider."""

import pytest

from zap_ai.tracing.noop_provider import NoOpTracingProvider
from zap_ai.tracing.protocol import ObservationType, TraceContext


@pytest.fixture
def provider():
    """Create a NoOpTracingProvider instance."""
    return NoOpTracingProvider()


@pytest.fixture
def sample_context():
    """Create a sample TraceContext."""
    return TraceContext(trace_id="trace-123", span_id="span-456")


class TestNoOpTracingProvider:
    """Tests for NoOpTracingProvider."""

    @pytest.mark.asyncio
    async def test_start_trace(self, provider):
        """start_trace should return a valid context."""
        async with provider.start_trace(name="test-trace") as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id is not None
            assert ctx.span_id is not None

    @pytest.mark.asyncio
    async def test_start_trace_with_all_params(self, provider):
        """start_trace should accept all optional params."""
        async with provider.start_trace(
            name="test-trace",
            session_id="session-123",
            user_id="user-456",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
        ) as ctx:
            assert isinstance(ctx, TraceContext)

    @pytest.mark.asyncio
    async def test_start_observation(self, provider, sample_context):
        """start_observation should return a valid child context."""
        async with provider.start_observation(
            name="test-span",
            observation_type=ObservationType.SPAN,
            parent_context=sample_context,
        ) as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id == sample_context.trace_id
            assert ctx.span_id != sample_context.span_id

    @pytest.mark.asyncio
    async def test_start_observation_with_all_params(self, provider, sample_context):
        """start_observation should accept all optional params."""
        async with provider.start_observation(
            name="test-tool",
            observation_type=ObservationType.TOOL,
            parent_context=sample_context,
            metadata={"tool": "search"},
            input_data={"query": "test"},
        ) as ctx:
            assert isinstance(ctx, TraceContext)

    @pytest.mark.asyncio
    async def test_start_observation_different_types(self, provider, sample_context):
        """start_observation should work with all observation types."""
        for obs_type in ObservationType:
            async with provider.start_observation(
                name=f"test-{obs_type.value}",
                observation_type=obs_type,
                parent_context=sample_context,
            ) as ctx:
                assert isinstance(ctx, TraceContext)

    @pytest.mark.asyncio
    async def test_start_generation(self, provider, sample_context):
        """start_generation should return a valid context."""
        ctx = await provider.start_generation(
            name="test-gen",
            parent_context=sample_context,
            model="gpt-4o",
            input_messages=[{"role": "user", "content": "Hello"}],
        )
        assert isinstance(ctx, TraceContext)
        assert ctx.trace_id == sample_context.trace_id

    @pytest.mark.asyncio
    async def test_start_generation_with_metadata(self, provider, sample_context):
        """start_generation should accept metadata."""
        ctx = await provider.start_generation(
            name="test-gen",
            parent_context=sample_context,
            model="gpt-4o",
            input_messages=[],
            metadata={"temperature": 0.7},
        )
        assert isinstance(ctx, TraceContext)

    @pytest.mark.asyncio
    async def test_end_generation(self, provider, sample_context):
        """end_generation should complete without error."""
        gen_ctx = await provider.start_generation(
            name="test-gen",
            parent_context=sample_context,
            model="gpt-4o",
            input_messages=[],
        )
        await provider.end_generation(
            context=gen_ctx,
            output={"content": "Hello!"},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

    @pytest.mark.asyncio
    async def test_add_event(self, provider, sample_context):
        """add_event should complete without error."""
        await provider.add_event(
            context=sample_context,
            name="test-event",
            attributes={"key": "value"},
        )

    @pytest.mark.asyncio
    async def test_add_event_without_attributes(self, provider, sample_context):
        """add_event should work without attributes."""
        await provider.add_event(
            context=sample_context,
            name="test-event",
        )

    @pytest.mark.asyncio
    async def test_set_error(self, provider, sample_context):
        """set_error should complete without error."""
        await provider.set_error(
            context=sample_context,
            error=ValueError("test error"),
        )

    @pytest.mark.asyncio
    async def test_flush(self, provider):
        """flush should complete without error."""
        await provider.flush()

    @pytest.mark.asyncio
    async def test_shutdown(self, provider):
        """shutdown should complete without error."""
        await provider.shutdown()

    @pytest.mark.asyncio
    async def test_full_workflow(self, provider):
        """Test a complete trace workflow."""
        async with provider.start_trace(name="test-task") as trace_ctx:
            # Start an iteration span
            async with provider.start_observation(
                name="iteration-0",
                observation_type=ObservationType.SPAN,
                parent_context=trace_ctx,
            ) as iter_ctx:
                # Run inference
                gen_ctx = await provider.start_generation(
                    name="inference",
                    parent_context=iter_ctx,
                    model="gpt-4o",
                    input_messages=[{"role": "user", "content": "Hello"}],
                )
                await provider.end_generation(
                    context=gen_ctx,
                    output={"content": "Hi there!"},
                    usage={"prompt_tokens": 5, "completion_tokens": 3},
                )

                # Execute a tool
                async with provider.start_observation(
                    name="tool-search",
                    observation_type=ObservationType.TOOL,
                    parent_context=iter_ctx,
                    input_data={"query": "test"},
                ):
                    pass

        await provider.flush()
        await provider.shutdown()
