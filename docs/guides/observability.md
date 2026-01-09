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

Zap uses a protocol-based system for tracing. You can implement your own provider:

```python
from zap_ai.tracing.protocol import TracingProvider, TraceContext, ObservationType

class MyTracingProvider(TracingProvider):
    async def start_trace(self, name: str, metadata: dict) -> TraceContext:
        # Start a new trace
        ...

    async def start_observation(
        self,
        parent_context: TraceContext,
        name: str,
        observation_type: ObservationType,
        metadata: dict,
    ) -> TraceContext:
        # Start a child observation
        ...

    async def end_observation(
        self,
        context: TraceContext,
        output: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        # End an observation
        ...

    async def flush(self) -> None:
        # Flush any buffered data
        ...
```

Then register it:

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
