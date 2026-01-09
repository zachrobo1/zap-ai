# Tracing API

The tracing module provides observability for agent execution.

## Registry Functions

::: zap_ai.tracing.set_tracing_provider
    options:
      show_source: true

::: zap_ai.tracing.get_tracing_provider
    options:
      show_source: true

::: zap_ai.tracing.reset_tracing_provider
    options:
      show_source: true

## Protocol

::: zap_ai.tracing.protocol.TracingProvider
    options:
      show_source: true

## Abstract Base Class

::: zap_ai.tracing.base.BaseTracingProvider
    options:
      show_source: true

::: zap_ai.tracing.protocol.TraceContext
    options:
      show_source: true

::: zap_ai.tracing.protocol.ObservationType
    options:
      show_source: true

## Providers

### LangfuseTracingProvider

::: zap_ai.tracing.langfuse_provider.LangfuseTracingProvider
    options:
      show_source: true

### NoOpTracingProvider

::: zap_ai.tracing.noop_provider.NoOpTracingProvider
    options:
      show_source: true

## Usage Example

```python
from zap_ai.tracing import set_tracing_provider, reset_tracing_provider
from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider

# Enable Langfuse tracing
provider = LangfuseTracingProvider()
set_tracing_provider(provider)

# ... run your agents ...

# Flush before shutdown
await provider.flush()

# Optionally reset to no-op
reset_tracing_provider()
```
