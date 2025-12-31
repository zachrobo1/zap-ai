# Zap Examples

This folder contains example applications demonstrating how to use the Zap AI agent platform.

## Prerequisites

Before running any example, you need:

1. **Set up your API key** - copy `.env.example` to `.env` in the project root:
   ```bash
   cp .env.example .env
   # Edit .env and set ANTHROPIC_API_KEY=your-key-here
   ```

2. **Temporal Server** running locally:
   ```bash
   temporal server start-dev
   ```

> **Note:** The examples run the Temporal worker inline (in the same process).
> This is required because the worker needs access to the MCP tool registry
> which is initialized when `zap.start()` is called.

## Examples

### 1. Simple Agent (`simple_agent.py`)

A basic single-agent example demonstrating:
- Creating an agent with MCP tools
- Executing a task
- Polling for completion

```bash
python examples/simple_agent.py
```

### 2. Multi-Agent (`multi_agent.py`)

A multi-agent example with delegation demonstrating:
- Multiple specialized agents
- Sub-agent relationships
- Task delegation with `message_agent`

```bash
python examples/multi_agent.py
```

### 3. Tools Server (`tools.py`)

A FastMCP tools server providing:
- `get_current_time()` - Get current UTC time
- `calculate()` - Basic arithmetic
- `search_web()` - Simulated web search

Run as a standalone MCP server:
```bash
python examples/tools.py
```

Or test with FastMCP's dev mode:
```bash
fastmcp dev examples/tools.py
```

## Project Structure

```
examples/
├── README.md           # This file
├── tools.py            # MCP tools server
├── simple_agent.py     # Single agent example
└── multi_agent.py      # Multi-agent delegation example
```

## Customization

### Using Different Models

Change the `model` parameter to use different LLM providers:

```python
# Anthropic (requires ANTHROPIC_API_KEY)
model="anthropic/claude-sonnet-4-5-20250929"

# OpenAI (requires OPENAI_API_KEY)
model="gpt-4o"

# Azure OpenAI (requires AZURE_API_KEY + AZURE_API_BASE)
model="azure/gpt-4"
```

See [LiteLLM Providers](https://docs.litellm.ai/docs/providers) for full list.

### Adding Custom Tools

Create your own MCP tools:

```python
from fastmcp import FastMCP

mcp = FastMCP("My Tools")

@mcp.tool()
def my_custom_tool(arg: str) -> str:
    """Description of what this tool does."""
    return f"Result: {arg}"
```

Then reference it in your agent:

```python
from fastmcp import Client

agent = ZapAgent(
    name="MyAgent",
    prompt="...",
    mcp_clients=[Client("path/to/my_tools.py")],
)
```
