# Quick Start

This guide walks you through creating your first Zap agent.

## 1. Create an MCP Tools Server

First, create a tools file that your agent can use. Zap uses [FastMCP](https://gofastmcp.com/) for tool integration.

```python title="tools.py"
from fastmcp import FastMCP

mcp = FastMCP("My Tools")

@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, call a real weather API
    return f"The weather in {city} is sunny, 72°F"

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        result = eval(expression)  # Use a safe evaluator in production
        return str(result)
    except Exception as e:
        return f"Error: {e}"
```

## 2. Define Your Agent

```python title="main.py"
import asyncio
from zap_ai import Zap, ZapAgent, TaskStatus
from fastmcp import Client

# Create an MCP client pointing to your tools
tools_client = Client("./tools.py")

# Define the agent
agent = ZapAgent(
    name="WeatherBot",
    prompt="You are a helpful weather assistant. Use your tools to answer questions about weather.",
    model="gpt-4o",  # Or "anthropic/claude-sonnet-4-5-20250929"
    mcp_clients=[tools_client],
)

# Create the Zap instance
zap = Zap(agents=[agent])
```

## 3. Run a Task

```python title="main.py (continued)"
async def main():
    # Start Zap (connects to Temporal, initializes MCP clients)
    await zap.start()

    # Execute a task
    task = await zap.execute_task(
        agent_name="WeatherBot",
        task="What's the weather like in San Francisco?",
    )

    print(f"Task started: {task.id}")

    # Poll for completion
    while not task.status.is_terminal():
        await asyncio.sleep(1)
        task = await zap.get_task(task.id)
        print(f"Status: {task.status.value}")

    # Check result
    if task.status == TaskStatus.COMPLETED:
        print(f"Result: {task.result}")
    else:
        print(f"Error: {task.error}")

    await zap.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 4. Run the Worker

In a separate terminal, start the Temporal worker:

```python title="worker.py"
import asyncio
from zap_ai.worker import run_worker

asyncio.run(run_worker())
```

```bash
python worker.py
```

## 5. Execute Your Agent

```bash
python main.py
```

You should see output like:

```
Task started: WeatherBot-a1b2c3d4e5f6
Status: thinking
Status: awaiting_tool
Status: thinking
Status: completed
Result: The weather in San Francisco is sunny, 72°F.
```

## Follow-Up Messages

Continue an existing conversation:

```python
# First task
task = await zap.execute_task(
    agent_name="WeatherBot",
    task="What's the weather in NYC?",
)

# ... wait for completion ...

# Follow-up on the same conversation
task = await zap.execute_task(
    follow_up_on_task=task.id,
    task="How about Los Angeles?",
)
```

## Next Steps

- [Dynamic Prompts](../guides/dynamic-prompts.md) - Pass context to agents at runtime
- [Multi-Agent Systems](../guides/multi-agent.md) - Build systems with multiple specialized agents
- [Observability](../guides/observability.md) - Add tracing with Langfuse
- [API Reference](../api/index.md) - Full API documentation
