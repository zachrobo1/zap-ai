# Zap - Zach's Agent Platform ⚡️

Zap is an opinionated library for building **resilient AI agents** on top of [Temporal](https://temporal.io/). It provides a scalable, fault-tolerant way to create AI agents that can power demanding use cases and complex architectures.

## Why Zap?

LLM providers can't yet guarantee production-level SLAs. API calls fail, rate limits hit, and connections drop. Zap solves this by running your agents on Temporal—a fault-tolerant code execution platform that captures state and retries failed steps automatically.

**Key Benefits:**

- **Automatic retries** with configurable policies for LLM and tool calls
- **State persistence** - agents survive crashes and can resume mid-conversation
- **Sub-agent delegation** - compose complex systems from specialized agents
- **Human-in-the-loop approvals** - require human oversight for high-stakes tool calls
- **MCP integration** - easily add tools via the Model Context Protocol
- **Provider agnostic** - use any LLM supported by LiteLLM (OpenAI, Anthropic, etc.)
- **Observability** - built-in tracing support with Langfuse integration
- **Dynamic prompts** - context-aware prompts resolved at runtime

## Built On

- [Temporal](https://docs.temporal.io/) - Fault-tolerant workflow orchestration
- [LiteLLM](https://docs.litellm.ai/docs/) - Unified LLM provider interface
- [FastMCP](https://gofastmcp.com/) - Model Context Protocol client for tool integration

## Quick Example

```python
import asyncio
from zap_ai import Zap, ZapAgent, TaskStatus
from fastmcp import Client

# Define an agent with tools
agent = ZapAgent(
    name="Assistant",
    prompt="You are a helpful assistant.",
    model="gpt-4o",
    mcp_clients=[Client("./tools.py")],
)

# Create and start the platform
zap = Zap(agents=[agent])

async def main():
    await zap.start()

    task = await zap.execute_task(
        agent_name="Assistant",
        task="What's the weather like today?",
    )

    while not task.status.is_terminal():
        await asyncio.sleep(1)
        task = await zap.get_task(task.id)

    print(task.result)

asyncio.run(main())
```

## Next Steps

- [Installation](getting-started/installation.md) - Get Zap installed
- [Quick Start](getting-started/quickstart.md) - Build your first agent
- [Approval Workflows](guides/approval-workflows.md) - Human-in-the-loop oversight
- [API Reference](api/index.md) - Detailed API documentation
