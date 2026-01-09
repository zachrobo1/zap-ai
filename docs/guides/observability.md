# Observability

Zap supports tracing via a pluggable provider system. Tracing helps you understand agent behavior, debug issues, and monitor performance.

## Langfuse Integration

[Langfuse](https://langfuse.com/) is currently the supported tracing provider.

### Setup

1. **Install with Langfuse support:**

    ```bash
    pip install zap-ai[langfuse]
    ```

2. **Configure environment variables:**

    ```bash
    export LANGFUSE_PUBLIC_KEY="pk-..."
    export LANGFUSE_SECRET_KEY="sk-..."
    # Optional: self-hosted Langfuse
    export LANGFUSE_HOST="https://cloud.langfuse.com"
    ```

3. **Enable tracing in your application:**

    ```python
    from zap_ai import Zap, ZapAgent
    from zap_ai.tracing import set_tracing_provider
    from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider

    # Initialize the provider
    provider = LangfuseTracingProvider()
    set_tracing_provider(provider)

    # Your normal Zap setup
    agent = ZapAgent(name="MyAgent", prompt="...")
    zap = Zap(agents=[agent])

    async def main():
        await zap.start()

        task = await zap.execute_task(
            agent_name="MyAgent",
            task="Do something",
        )

        # ... wait for completion ...

        # Important: flush traces before shutdown
        await provider.flush()
        await zap.stop()
    ```

### What Gets Traced

Each task execution creates a trace containing:

| Observation Type | Description |
|-----------------|-------------|
| **Task/Trace** | Root span for the entire task execution |
| **Iteration** | Each agentic loop iteration |
| **Generation** | LLM inference calls with token usage |
| **Tool** | Tool executions with inputs/outputs |
| **Agent** | Sub-agent delegations (child workflows) |

### Viewing Traces

After running your agent, view traces in the Langfuse dashboard:

1. Go to [cloud.langfuse.com](https://cloud.langfuse.com) (or your self-hosted instance)
2. Navigate to **Traces**
3. Click on a trace to see the full execution timeline

You'll see:

- Complete conversation flow
- LLM prompts and responses
- Token usage and costs
- Tool call inputs and outputs
- Sub-agent delegation chains
- Timing for each operation

## Custom Tracing Providers

### Architecture: Protocol vs ABC

Zap's tracing system has two related but distinct components:

| Component | Purpose | When to Use |
|-----------|---------|-------------|
| `TracingProvider` (Protocol) | Defines the interface contract | Used internally for type hints. You don't need to interact with this directly. |
| `BaseTracingProvider` (ABC) | Implementation helper with utilities | **Use this when building custom providers.** |

Any class inheriting from `BaseTracingProvider` automatically satisfies the `TracingProvider` protocol, so type checking works seamlessly.

### Building a Custom Provider

Zap provides an abstract base class `BaseTracingProvider` that you can extend to implement custom tracing backends. This approach gives you:

- **Utility methods** for generating trace/span IDs and creating contexts
- **Default implementations** for optional methods (`add_event`, `set_error`, `flush`, `shutdown`)
- **Clear interface** showing exactly which methods you need to implement

### Minimal Implementation

To create a custom provider, inherit from `BaseTracingProvider` and implement four methods:

```python
from zap_ai.tracing import BaseTracingProvider
from zap_ai.tracing.protocol import ObservationType, TraceContext

class MyTracingProvider(BaseTracingProvider):
    """Custom tracing provider example."""

    async def _start_trace_impl(
        self,
        name,
        session_id=None,
        user_id=None,
        metadata=None,
        tags=None,
    ):
        # Create your trace and return (context, cleanup_data)
        # cleanup_data is passed to _end_trace_cleanup when the trace ends
        ctx = self._create_context()
        return ctx, None  # None = no cleanup needed

    async def _start_observation_impl(
        self,
        name,
        observation_type,
        parent_context,
        metadata=None,
        input_data=None,
    ):
        # Create child observation, preserving the trace_id
        ctx = self._create_child_context(parent_context)
        return ctx, None

    async def start_generation(
        self,
        name,
        parent_context,
        model,
        input_messages,
        metadata=None,
    ):
        # Track LLM generation
        return self._create_child_context(parent_context)

    async def end_generation(self, context, output, usage=None):
        # Record generation output and token usage
        pass
```

### Optional Methods

You can override these methods for additional functionality:

```python
class MyTracingProvider(BaseTracingProvider):
    # ... required methods ...

    async def _end_trace_cleanup(self, context, cleanup_data):
        """Called when a trace context manager exits."""
        # cleanup_data is whatever you returned from _start_trace_impl
        pass

    async def _end_observation_cleanup(self, context, cleanup_data):
        """Called when an observation context manager exits."""
        pass

    async def add_event(self, context, name, attributes=None):
        """Log events within observations."""
        pass

    async def set_error(self, context, error):
        """Mark an observation as errored."""
        pass

    async def flush(self):
        """Flush any buffered trace data."""
        pass

    async def shutdown(self):
        """Cleanup resources."""
        pass
```

### Utility Methods

The base class provides these utility methods:

- `_generate_trace_id()` - Generate a unique 32-character hex trace ID
- `_generate_span_id(w3c_format=False)` - Generate a span ID (16 chars if W3C format, 32 otherwise)
- `_create_context(trace_id=None, span_id=None, provider_data=None)` - Create a new `TraceContext`
- `_create_child_context(parent, span_id=None, provider_data=None)` - Create a child context preserving the parent's `trace_id`

### Register Your Provider

```python
from zap_ai.tracing import set_tracing_provider

provider = MyTracingProvider()
set_tracing_provider(provider)
```

## Disabling Tracing

By default, Zap uses a no-op tracing provider that does nothing. To explicitly disable tracing after enabling it:

```python
from zap_ai.tracing import reset_tracing_provider

reset_tracing_provider()
```

## Best Practices

1. **Always flush before shutdown** - Call `await provider.flush()` to ensure all traces are sent
2. **Use meaningful task names** - Task IDs include the agent name, making traces easier to filter
3. **Add metadata** - Use context to add user IDs or session info that appears in traces
4. **Monitor in production** - Tracing has minimal overhead and is safe for production use

## API Reference

See the [Tracing API](../api/tracing.md) for full documentation.
