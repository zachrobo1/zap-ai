# MCP Sampling Support

MCP (Model Context Protocol) allows servers to request LLM completions from clients. Zap provides built-in sampling handlers that reuse your existing LiteLLM configuration, making it easy to give MCP servers access to AI capabilities.

## Why MCP Sampling?

MCP servers often need to perform LLM inference—summarizing content, generating responses, or processing data. Instead of each server managing its own LLM credentials and configuration, MCP sampling lets the client (Zap) handle all LLM interactions:

- **Centralized credentials** - API keys stay in one place
- **Consistent configuration** - Same model, temperature across servers
- **Provider flexibility** - Switch LLM providers without changing servers
- **Cost visibility** - All LLM calls go through your client

## Quick Start

```python
from zap_ai import Zap, ZapAgent
from zap_ai.mcp.sampling import create_mcp_client

# Create MCP client with sampling support
client = create_mcp_client(
    "path/to/mcp_server.py",
    sampling_handler="litellm",  # Use Zap's LLM provider
    sampling_model="gpt-4o",      # Default model for sampling
)

agent = ZapAgent(
    name="SamplingAgent",
    prompt="You are a helpful assistant.",
    mcp_clients=[client],
)

zap = Zap(agents=[agent])
await zap.start()
```

## How It Works

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         MCP Sampling Flow                                 │
│                                                                          │
│  1. MCP Server needs LLM completion                                      │
│         │                                                                │
│         ▼                                                                │
│  2. Server sends sampling request to Client                              │
│     • Messages, system prompt                                            │
│     • Model preferences (hints)                                          │
│     • Temperature, max_tokens                                            │
│         │                                                                │
│         ▼                                                                │
│  3. LiteLLMSamplingHandler receives request                              │
│     • Converts messages to LiteLLM format                                │
│     • Extracts model from hints or uses default                          │
│     • Calls Zap's complete() function                                    │
│         │                                                                │
│         ▼                                                                │
│  4. LLM provider returns completion                                      │
│         │                                                                │
│         ▼                                                                │
│  5. Response returned to MCP Server                                      │
└──────────────────────────────────────────────────────────────────────────┘
```

## API Reference

### create_mcp_client()

Factory function for creating MCP clients with sampling support:

```python
from zap_ai.mcp.sampling import create_mcp_client

client = create_mcp_client(
    source="path/to/server.py",
    sampling_handler="litellm",
    sampling_model="gpt-4o",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | required | Path to MCP server or URL |
| `sampling_handler` | `str \| Callable \| None` | `None` | Handler specification |
| `sampling_model` | `str` | `"gpt-4o"` | Default model for "litellm" handler |
| `**kwargs` | | | Additional FastMCP Client arguments |

**Handler options:**
- `None` - No sampling support (default)
- `"litellm"` - Use built-in LiteLLMSamplingHandler
- `Callable` - Custom async handler function

### LiteLLMSamplingHandler

Direct access to the sampling handler class:

```python
from zap_ai.mcp.sampling import LiteLLMSamplingHandler
from fastmcp import Client

handler = LiteLLMSamplingHandler(
    default_model="anthropic/claude-sonnet-4-5-20250929",
    default_temperature=0.7,
    default_max_tokens=1000,
)

client = Client("server.py", sampling_handler=handler)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `default_model` | `str` | `"gpt-4o"` | LiteLLM model identifier |
| `default_temperature` | `float` | `0.7` | Sampling temperature (0.0-2.0) |
| `default_max_tokens` | `int \| None` | `None` | Max tokens to generate |

## Custom Sampling Handlers

Create a custom handler for specialized behavior:

```python
from zap_ai.mcp.sampling import create_mcp_client

async def my_handler(messages, params, context):
    """Custom sampling handler with logging."""
    import logging
    logging.info(f"Sampling request: {len(messages)} messages")

    # Call your preferred LLM
    response = await my_llm_call(messages, params)

    logging.info(f"Response: {len(response)} chars")
    return response

client = create_mcp_client(
    "server.py",
    sampling_handler=my_handler,
)
```

## Using FastMCP's Built-in Handlers

FastMCP provides its own sampling handlers you can use directly:

```python
from fastmcp import Client
from fastmcp.client.sampling import AnthropicSamplingHandler

# Use FastMCP's Anthropic handler directly
client = Client(
    "server.py",
    sampling_handler=AnthropicSamplingHandler(
        default_model="claude-sonnet-4-5-20250929"
    ),
)

agent = ZapAgent(
    name="MyAgent",
    mcp_clients=[client],
    ...
)
```

## Best Practices

### Model Selection

```python
# Good: Use model hints in server requests
# Server can request specific models, handler uses default as fallback

# Configure a sensible default
client = create_mcp_client(
    "server.py",
    sampling_handler="litellm",
    sampling_model="gpt-4o",  # Fast, capable default
)
```

### Temperature Settings

| Use Case | Recommended Temperature |
|----------|------------------------|
| Factual responses | 0.0 - 0.3 |
| Balanced creativity | 0.5 - 0.7 |
| Creative generation | 0.8 - 1.0 |

### Error Handling

The handler returns empty string if the LLM returns no content:

```python
handler = LiteLLMSamplingHandler()
result = await handler(messages, params, context)
# result is "" if LLM returns None
```

## See Also

- [MCP Protocol](https://modelcontextprotocol.io) - Model Context Protocol specification
- [FastMCP Documentation](https://github.com/jlowin/fastmcp) - FastMCP client library
- [LiteLLM Providers](https://docs.litellm.ai/docs/providers) - Supported LLM providers
