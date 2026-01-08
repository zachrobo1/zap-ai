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

### 1. Simple Agent (`simple_agent/`)

A basic single-agent example demonstrating:
- Creating an agent with MCP tools
- Executing a task
- Polling for completion

```bash
python examples/simple_agent/main.py
```

### 2. Multi-Agent (`multi_agent/`)

A multi-agent example with delegation demonstrating:
- Multiple specialized agents
- Sub-agent relationships
- Task delegation with `message_agent`

```bash
python examples/multi_agent/main.py
```

### 3. Langfuse Tracing (`langfuse_tracing/`)

Demonstrates observability with Langfuse tracing:
- Configure Langfuse tracing provider
- Traces for each task execution
- LLM inference and tool calls visible in dashboard

**Additional setup:**
1. Install with Langfuse support: `pip install zap-ai[langfuse]`
2. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env`
3. View traces at https://cloud.langfuse.com

```bash
python examples/langfuse_tracing/main.py
```

### 4. Conversation History Access (`conversation_history/`)

Demonstrates the conversation history inspection API:
- `get_text_content()` - Extract all text from conversation
- `get_tool_calls()` - Get tool calls with `ToolCallInfo` objects
- `get_turns()` / `get_turn(n)` - Navigate conversation by turns
- `turn_count()` - Get number of conversation turns
- `get_sub_tasks()` - Fetch sub-task Task objects (multi-agent)

```bash
python examples/conversation_history/main.py
```

## Project Structure

```
examples/
├── README.md                 # This file
├── simple_agent/
│   ├── main.py               # Single agent example
│   └── tools.py              # MCP tools server
├── multi_agent/
│   ├── main.py               # Multi-agent delegation example
│   └── tools.py              # MCP tools server
├── langfuse_tracing/
│   └── main.py               # Langfuse observability example
└── conversation_history/
    └── main.py               # Conversation history inspection example
```

The `langfuse_tracing` and `conversation_history` examples reuse tools from `simple_agent`.

## Tools

Each example includes a `tools.py` FastMCP server providing:
- `get_current_time()` - Get current UTC time
- `calculate()` - Basic arithmetic
- `search_web()` - Simulated web search

Run as a standalone MCP server:
```bash
python examples/simple_agent/tools.py
```

Or test with FastMCP's dev mode:
```bash
fastmcp dev examples/simple_agent/tools.py
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
