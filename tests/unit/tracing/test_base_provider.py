"""Tests for BaseTracingProvider ABC."""

import pytest

from zap_ai.tracing.base import BaseTracingProvider
from zap_ai.tracing.protocol import ObservationType, TraceContext


class MinimalProvider(BaseTracingProvider):
    """Minimal implementation for testing."""

    async def _start_trace_impl(
        self,
        name,
        session_id=None,
        user_id=None,
        metadata=None,
        tags=None,
    ):
        return self._create_context(), None

    async def _start_observation_impl(
        self,
        name,
        observation_type,
        parent_context,
        metadata=None,
        input_data=None,
    ):
        return self._create_child_context(parent_context), None

    async def start_generation(
        self,
        name,
        parent_context,
        model,
        input_messages,
        metadata=None,
    ):
        return self._create_child_context(parent_context)

    async def end_generation(self, context, output, usage=None):
        pass


class ProviderWithCleanup(BaseTracingProvider):
    """Provider that tracks cleanup calls for testing."""

    def __init__(self):
        self.trace_cleanups: list[tuple[TraceContext, str]] = []
        self.observation_cleanups: list[tuple[TraceContext, str]] = []

    async def _start_trace_impl(
        self,
        name,
        session_id=None,
        user_id=None,
        metadata=None,
        tags=None,
    ):
        ctx = self._create_context()
        return ctx, f"trace-cleanup-{name}"

    async def _end_trace_cleanup(self, context, cleanup_data):
        self.trace_cleanups.append((context, cleanup_data))

    async def _start_observation_impl(
        self,
        name,
        observation_type,
        parent_context,
        metadata=None,
        input_data=None,
    ):
        ctx = self._create_child_context(parent_context)
        return ctx, f"obs-cleanup-{name}"

    async def _end_observation_cleanup(self, context, cleanup_data):
        self.observation_cleanups.append((context, cleanup_data))

    async def start_generation(
        self,
        name,
        parent_context,
        model,
        input_messages,
        metadata=None,
    ):
        return self._create_child_context(parent_context)

    async def end_generation(self, context, output, usage=None):
        pass


class TestBaseTracingProviderUtilities:
    """Tests for utility methods."""

    def test_generate_trace_id(self):
        provider = MinimalProvider()
        trace_id = provider._generate_trace_id()
        assert len(trace_id) == 32  # UUID hex
        assert trace_id.isalnum()

    def test_generate_trace_id_unique(self):
        provider = MinimalProvider()
        ids = [provider._generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique

    def test_generate_span_id_full(self):
        provider = MinimalProvider()
        span_id = provider._generate_span_id(w3c_format=False)
        assert len(span_id) == 32

    def test_generate_span_id_w3c(self):
        provider = MinimalProvider()
        span_id = provider._generate_span_id(w3c_format=True)
        assert len(span_id) == 16

    def test_generate_span_id_default_full(self):
        provider = MinimalProvider()
        span_id = provider._generate_span_id()
        assert len(span_id) == 32

    def test_create_context(self):
        provider = MinimalProvider()
        ctx = provider._create_context()
        assert isinstance(ctx, TraceContext)
        assert ctx.trace_id is not None
        assert ctx.span_id is not None
        assert len(ctx.trace_id) == 32
        assert len(ctx.span_id) == 32

    def test_create_context_with_trace_id(self):
        provider = MinimalProvider()
        ctx = provider._create_context(trace_id="custom-trace")
        assert ctx.trace_id == "custom-trace"
        assert ctx.span_id is not None

    def test_create_context_with_span_id(self):
        provider = MinimalProvider()
        ctx = provider._create_context(span_id="custom-span")
        assert ctx.trace_id is not None
        assert ctx.span_id == "custom-span"

    def test_create_context_with_provider_data(self):
        provider = MinimalProvider()
        ctx = provider._create_context(provider_data={"key": "value"})
        assert ctx.provider_data == {"key": "value"}

    def test_create_context_with_all_values(self):
        provider = MinimalProvider()
        ctx = provider._create_context(
            trace_id="custom-trace",
            span_id="custom-span",
            provider_data={"key": "value"},
        )
        assert ctx.trace_id == "custom-trace"
        assert ctx.span_id == "custom-span"
        assert ctx.provider_data == {"key": "value"}

    def test_create_child_context(self):
        provider = MinimalProvider()
        parent = TraceContext(trace_id="parent-trace", span_id="parent-span")
        child = provider._create_child_context(parent)
        assert child.trace_id == "parent-trace"
        assert child.span_id != "parent-span"
        assert len(child.span_id) == 32

    def test_create_child_context_with_span_id(self):
        provider = MinimalProvider()
        parent = TraceContext(trace_id="parent-trace", span_id="parent-span")
        child = provider._create_child_context(parent, span_id="custom-child-span")
        assert child.trace_id == "parent-trace"
        assert child.span_id == "custom-child-span"

    def test_create_child_context_with_provider_data(self):
        provider = MinimalProvider()
        parent = TraceContext(trace_id="parent-trace", span_id="parent-span")
        child = provider._create_child_context(parent, provider_data={"child_key": "child_value"})
        assert child.trace_id == "parent-trace"
        assert child.provider_data == {"child_key": "child_value"}


class TestBaseTracingProviderContextManagers:
    """Tests for context manager behavior."""

    @pytest.mark.asyncio
    async def test_start_trace_yields_context(self):
        provider = MinimalProvider()
        async with provider.start_trace(name="test") as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id is not None
            assert ctx.span_id is not None

    @pytest.mark.asyncio
    async def test_start_trace_with_metadata(self):
        provider = MinimalProvider()
        async with provider.start_trace(
            name="test",
            session_id="session-123",
            user_id="user-456",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
        ) as ctx:
            assert isinstance(ctx, TraceContext)

    @pytest.mark.asyncio
    async def test_start_observation_yields_context(self):
        provider = MinimalProvider()
        parent = TraceContext(trace_id="trace", span_id="span")
        async with provider.start_observation(
            name="obs",
            observation_type=ObservationType.SPAN,
            parent_context=parent,
        ) as ctx:
            assert isinstance(ctx, TraceContext)
            assert ctx.trace_id == parent.trace_id

    @pytest.mark.asyncio
    async def test_start_observation_with_metadata(self):
        provider = MinimalProvider()
        parent = TraceContext(trace_id="trace", span_id="span")
        async with provider.start_observation(
            name="obs",
            observation_type=ObservationType.TOOL,
            parent_context=parent,
            metadata={"key": "value"},
            input_data={"input": "data"},
        ) as ctx:
            assert isinstance(ctx, TraceContext)

    @pytest.mark.asyncio
    async def test_start_observation_all_types(self):
        provider = MinimalProvider()
        parent = TraceContext(trace_id="trace", span_id="span")

        for obs_type in ObservationType:
            async with provider.start_observation(
                name=f"obs-{obs_type.value}",
                observation_type=obs_type,
                parent_context=parent,
            ) as ctx:
                assert isinstance(ctx, TraceContext)

    @pytest.mark.asyncio
    async def test_nested_observations(self):
        provider = MinimalProvider()
        async with provider.start_trace(name="trace") as trace_ctx:
            assert trace_ctx.trace_id is not None

            async with provider.start_observation(
                name="outer",
                observation_type=ObservationType.SPAN,
                parent_context=trace_ctx,
            ) as outer_ctx:
                assert outer_ctx.trace_id == trace_ctx.trace_id

                async with provider.start_observation(
                    name="inner",
                    observation_type=ObservationType.TOOL,
                    parent_context=outer_ctx,
                ) as inner_ctx:
                    assert inner_ctx.trace_id == trace_ctx.trace_id

    @pytest.mark.asyncio
    async def test_cleanup_called_on_trace_exit(self):
        provider = ProviderWithCleanup()
        async with provider.start_trace(name="test-trace") as ctx:
            pass

        assert len(provider.trace_cleanups) == 1
        cleanup_ctx, cleanup_data = provider.trace_cleanups[0]
        assert cleanup_ctx.trace_id == ctx.trace_id
        assert cleanup_data == "trace-cleanup-test-trace"

    @pytest.mark.asyncio
    async def test_cleanup_called_on_observation_exit(self):
        provider = ProviderWithCleanup()
        parent = TraceContext(trace_id="trace", span_id="span")
        async with provider.start_observation(
            name="test-obs",
            observation_type=ObservationType.SPAN,
            parent_context=parent,
        ):
            pass

        assert len(provider.observation_cleanups) == 1
        cleanup_ctx, cleanup_data = provider.observation_cleanups[0]
        assert cleanup_ctx.trace_id == parent.trace_id
        assert cleanup_data == "obs-cleanup-test-obs"

    @pytest.mark.asyncio
    async def test_cleanup_called_on_exception(self):
        provider = ProviderWithCleanup()

        with pytest.raises(ValueError):
            async with provider.start_trace(name="test-trace"):
                raise ValueError("Test error")

        # Cleanup should still be called
        assert len(provider.trace_cleanups) == 1


class TestBaseTracingProviderDefaults:
    """Tests for default method implementations."""

    @pytest.mark.asyncio
    async def test_add_event_is_noop(self):
        provider = MinimalProvider()
        ctx = TraceContext(trace_id="t", span_id="s")
        await provider.add_event(ctx, "event")  # Should not raise

    @pytest.mark.asyncio
    async def test_add_event_with_attributes(self):
        provider = MinimalProvider()
        ctx = TraceContext(trace_id="t", span_id="s")
        await provider.add_event(ctx, "event", attributes={"key": "value"})  # Should not raise

    @pytest.mark.asyncio
    async def test_set_error_is_noop(self):
        provider = MinimalProvider()
        ctx = TraceContext(trace_id="t", span_id="s")
        await provider.set_error(ctx, ValueError("test"))  # Should not raise

    @pytest.mark.asyncio
    async def test_flush_is_noop(self):
        provider = MinimalProvider()
        await provider.flush()  # Should not raise

    @pytest.mark.asyncio
    async def test_shutdown_is_noop(self):
        provider = MinimalProvider()
        await provider.shutdown()  # Should not raise


class TestCannotInstantiateABC:
    """Test that ABC cannot be instantiated directly."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError, match="abstract"):
            BaseTracingProvider()

    def test_must_implement_start_trace_impl(self):
        class IncompleteProvider(BaseTracingProvider):
            async def _start_observation_impl(self, *args, **kwargs):
                pass

            async def start_generation(self, *args, **kwargs):
                pass

            async def end_generation(self, *args, **kwargs):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()

    def test_must_implement_start_observation_impl(self):
        class IncompleteProvider(BaseTracingProvider):
            async def _start_trace_impl(self, *args, **kwargs):
                pass

            async def start_generation(self, *args, **kwargs):
                pass

            async def end_generation(self, *args, **kwargs):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()

    def test_must_implement_start_generation(self):
        class IncompleteProvider(BaseTracingProvider):
            async def _start_trace_impl(self, *args, **kwargs):
                pass

            async def _start_observation_impl(self, *args, **kwargs):
                pass

            async def end_generation(self, *args, **kwargs):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()

    def test_must_implement_end_generation(self):
        class IncompleteProvider(BaseTracingProvider):
            async def _start_trace_impl(self, *args, **kwargs):
                pass

            async def _start_observation_impl(self, *args, **kwargs):
                pass

            async def start_generation(self, *args, **kwargs):
                pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteProvider()


class TestGenerationMethods:
    """Tests for generation-related methods."""

    @pytest.mark.asyncio
    async def test_start_generation(self):
        provider = MinimalProvider()
        parent = TraceContext(trace_id="trace", span_id="span")
        ctx = await provider.start_generation(
            name="gen",
            parent_context=parent,
            model="gpt-4",
            input_messages=[{"role": "user", "content": "hello"}],
        )
        assert isinstance(ctx, TraceContext)
        assert ctx.trace_id == parent.trace_id

    @pytest.mark.asyncio
    async def test_start_generation_with_metadata(self):
        provider = MinimalProvider()
        parent = TraceContext(trace_id="trace", span_id="span")
        ctx = await provider.start_generation(
            name="gen",
            parent_context=parent,
            model="gpt-4",
            input_messages=[{"role": "user", "content": "hello"}],
            metadata={"key": "value"},
        )
        assert isinstance(ctx, TraceContext)

    @pytest.mark.asyncio
    async def test_end_generation(self):
        provider = MinimalProvider()
        ctx = TraceContext(trace_id="trace", span_id="span")
        await provider.end_generation(
            ctx,
            output={"content": "response"},
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )  # Should not raise
