# Zap - Zach's Agent Platform

Zap is an opinionated library for building **resilient AI agents** on top of [Temporal](https://temporal.io/). It provides a scalable, fault-tolerant way to create AI agents that can power demanding use cases and complex architectures.

## Why Zap?

LLM providers can't yet guarantee production-level SLAs. API calls fail, rate limits hit, and connections drop. Zap solves this by running your agents on Temporal—a fault-tolerant code execution platform that captures state and retries failed steps automatically.

**Key Benefits:**
- **Automatic retries** with configurable policies for LLM and tool calls
- **State persistence** - agents survive crashes and can resume mid-conversation
- **Sub-agent delegation** - compose complex systems from specialized agents
- **MCP integration** - easily add tools via the Model Context Protocol
- **Provider agnostic** - use any LLM supported by LiteLLM (OpenAI, Anthropic, etc.)

## Built On

- [Temporal](https://docs.temporal.io/) - Fault-tolerant workflow orchestration
- [LiteLLM](https://docs.litellm.ai/docs/) - Unified LLM provider interface
- [FastMCP](https://gofastmcp.com/) - Model Context Protocol client for tool integration

## Installation

```bash
pip install zap-ai
```

Or with uv:
```bash
uv add zap-ai
```

## Prerequisites

1. **Temporal Server** - You need a running Temporal cluster:
   ```bash
   # Local development (requires Docker)
   temporal server start-dev

   # Or use Temporal Cloud: https://temporal.io/cloud
   ```

2. **LLM API Keys** - Set environment variables for your LLM provider:
   ```bash
   export OPENAI_API_KEY="sk-..."
   # Or ANTHROPIC_API_KEY, AZURE_API_KEY, etc.
   ```

## Quick Start

### 1. Define Your Agents

```python
import asyncio
from zap_ai import ZapAgent, Zap, TaskStatus
from fastmcp import Client

# Create MCP clients for tool access
search_client = Client("https://example.com/search-mcp")
database_client = Client("./my_db_server.py")

# Define a main agent with access to tools and a sub-agent
main_agent = ZapAgent(
    name="MainAgent",
    prompt="You are a helpful research assistant. Use your tools to find information and delegate complex lookups to the LookupAgent.",
    model="gpt-4o",  # Any LiteLLM-supported model
    mcp_clients=[search_client],
    sub_agents=["LookupAgent"],  # Can delegate to this agent
)

# Define a specialized sub-agent
lookup_agent = ZapAgent(
    name="LookupAgent",
    prompt="You are a database specialist. Query the database to find detailed information.",
    discovery_prompt="Use this agent for database lookups and detailed data retrieval",
    model="gpt-4o-mini",  # Can use different models per agent
    mcp_clients=[database_client],
)

# Create the Zap instance (validates configuration at build time)
zap = Zap(agents=[main_agent, lookup_agent])
```

### 2. Start the Platform

```python
async def main():
    # Initialize Temporal connection and MCP clients
    await zap.start()

    # Execute a task
    task = await zap.execute_task(
        agent_name="MainAgent",
        task="Research the latest developments in quantum computing and summarize the key findings.",
    )

    print(f"Task started: {task.id}")

    # Poll for completion
    while True:
        task = await zap.get_task(task.id)
        print(f"Status: {task.status}")

        if task.status == TaskStatus.COMPLETED:
            print(f"Result: {task.result}")
            break
        elif task.status == TaskStatus.FAILED:
            print(f"Error: {task.error}")
            break

        await asyncio.sleep(2)

asyncio.run(main())
```

### 3. Follow Up on Tasks

```python
# Continue an existing conversation
await zap.execute_task(
    follow_up_on_task=task.id,
    task="Now compare these findings to classical computing approaches.",
)
```

### 4. Run the Worker (Separate Process)

```python
# worker.py
from zap_ai.worker import run_worker
import asyncio

asyncio.run(run_worker())
```

```bash
python worker.py
```

## Configuration

### ZapAgent Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | required | Unique identifier (no spaces) |
| `prompt` | `str` | required | System prompt for the agent |
| `model` | `str` | `"gpt-4o"` | LiteLLM model identifier |
| `mcp_clients` | `list[Client]` | `[]` | FastMCP clients for tool access |
| `sub_agents` | `list[str]` | `[]` | Names of agents this agent can delegate to |
| `discovery_prompt` | `str` | `None` | Description shown to parent agents |
| `max_iterations` | `int` | `50` | Maximum agentic loop iterations |

### Zap Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[ZapAgent]` | required | All agents in the system |
| `temporal_client` | `Client` | `None` | Custom Temporal client (auto-connects if None) |
| `task_queue` | `str` | `"zap-agents"` | Temporal task queue name |

## Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         Zap Orchestrator                         │
│  • Validates agent configuration at build time                   │
│  • Manages Temporal client connection                            │
│  • Routes tasks to appropriate agent workflows                   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Temporal Workflow (per task)                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │   Receive   │───▶│   LLM       │───▶│   Tool Execution    │ │
│  │   Message   │    │  Inference  │    │   (parallel)        │ │
│  └─────────────┘    └─────────────┘    └─────────────────────┘ │
│         ▲                                        │              │
│         │                                        │              │
│         └────────────────────────────────────────┘              │
│                     (agentic loop)                              │
│                                                                  │
│  Features:                                                       │
│  • Signals: Receive follow-up messages                          │
│  • Queries: Check status, get history                           │
│  • Continue-as-new: Handle long conversations                   │
│  • Child workflows: Delegate to sub-agents                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Activities                               │
│  ┌────────────────────┐    ┌─────────────────────────────────┐ │
│  │  Inference         │    │  Tool Execution                  │ │
│  │  (LiteLLM)         │    │  (FastMCP clients)               │ │
│  │  • Retry on failure│    │  • Parallel execution            │ │
│  │  • Provider agnostic│   │  • Schema conversion             │ │
│  └────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### The Agentic Loop

1. **Receive Message** - Initial task or follow-up signal
2. **LLM Inference** - Call the model with conversation history and available tools
3. **Tool Execution** - Execute any requested tools in parallel
4. **Repeat** - Continue until the agent responds without tool calls
5. **Wait** - Idle until a follow-up signal arrives or timeout

### Sub-Agent Delegation

When an agent has sub-agents configured, a special `transfer_to_agent` tool is automatically added. When called:

1. A **child workflow** is spawned for the sub-agent
2. The parent workflow waits for completion
3. The sub-agent's result is returned as a tool response
4. The parent continues with the new context

### State Management

- **Conversation history** is stored in the workflow state
- **Continue-as-new** prevents event history from growing unbounded
- **Queries** allow external access to current status and history
- **Signals** enable follow-up messages to running workflows

## Task Status

| Status | Description |
|--------|-------------|
| `PENDING` | Task created, not yet started |
| `RUNNING` | Agent is processing (inference or between tool calls) |
| `AWAITING_TOOL` | Waiting for tool execution to complete |
| `DELEGATED` | Running in a sub-agent (child workflow) |
| `COMPLETED` | Task finished successfully |
| `FAILED` | Task failed with an error |

## Limitations

- **No streaming** - Currently uses query-based polling for status. Real-time streaming may be added in future versions.
- **Temporal required** - You need a running Temporal cluster (local dev server or Temporal Cloud).
- **MCP tools only** - Tools must be exposed via MCP servers (FastMCP makes this easy).

## Future Plans

- Real-time streaming via callbacks/webhooks
- Human-in-the-loop tools (approval workflows)
- Hooks system for custom logic injection
- Expose agents as MCP servers for agent-to-agent communication
- Execution statistics and debugging tools

## Contributing

Contributions are welcome! Please see [IMPLEMENTATION.md](IMPLEMENTATION.md) for the development roadmap and open tasks.

## License

MIT License - see [LICENSE](LICENSE) for details.
