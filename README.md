# Zap - Zach's Agent Platform ⚡️

[![PyPI version](https://img.shields.io/pypi/v/zap-ai)](https://pypi.org/project/zap-ai/)
[![Coverage](https://codecov.io/gh/zachrobo1/zap-ai/graph/badge.svg)](https://codecov.io/gh/zachrobo1/zap-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Zap is an opinionated library for building **resilient AI agents** on top of [Temporal](https://temporal.io/). It provides a scalable, fault-tolerant way to create AI agents that can power demanding use cases and complex architectures.

Looking for the full docs? Find them [here](https://zachrobo1.github.io/zap-ai/).

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
from zap_ai import Zap, ZapAgent, Task, TaskStatus
from fastmcp import Client

# Create MCP clients for tool access
search_client = Client("https://example.com/search-mcp")
database_client = Client("./my_db_server.py")

# Define a main agent with access to tools and a sub-agent
main_agent = ZapAgent(
    name="MainAgent",
    prompt="You are a helpful research assistant. Use your tools to find information and delegate complex lookups to the LookupAgent.",
    model="anthropic/claude-sonnet-4-5-20250929",  # Any LiteLLM-supported model
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

### 4. Dynamic Prompts with Context

Agents can use dynamic prompts that receive context at execution time:

```python
from dataclasses import dataclass

@dataclass
class UserContext:
    user_name: str
    company: str

# Agent with dynamic prompt
assistant = ZapAgent[UserContext](
    name="PersonalAssistant",
    prompt=lambda ctx: f"You are {ctx.user_name}'s assistant at {ctx.company}. Be helpful and professional.",
    model="gpt-4o",
)

zap = Zap(agents=[assistant])
await zap.start()

# Pass context when executing
task = await zap.execute_task(
    agent_name="PersonalAssistant",
    task="Draft an email to my team about the project update.",
    context=UserContext(user_name="Alice", company="Acme Corp"),
)
```

### 5. Run the Worker (Separate Process)

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
| `prompt` | `str \| Callable` | required | System prompt - static string or `callable(context) -> str` |
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

### execute_task Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent_name` | `str` | required* | Agent to execute the task (*not needed for follow-ups) |
| `task` | `str` | required | The task description/prompt |
| `follow_up_on_task` | `str` | `None` | Continue an existing conversation |
| `context` | `TContext` | `{}` | Context for dynamic prompts |

## Architecture

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                         Zap Orchestrator                        │
│  • Validates agent configuration at build time                  │
│  • Manages Temporal client connection                           │
│  • Resolves dynamic prompts with context                        │
│  • Routes tasks to appropriate agent workflows                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Temporal Workflow (per task)                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Receive   │───▶│   LLM       │───▶│   Tool Execution    │  │
│  │   Message   │    │  Inference  │    │   (parallel)        │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│         ▲                                        │              │
│         │                                        │              │
│         └────────────────────────────────────────┘              │
│                     (agentic loop)                              │
│                                                                 │
│  Features:                                                      │
│  • Signals: Receive follow-up messages                          │
│  • Queries: Check status, get history                           │
│  • Continue-as-new: Handle long conversations                   │
│  • Child workflows: Delegate to sub-agents                      │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Activities                              │
│  ┌────────────────────┐    ┌─────────────────────────────────┐  │
│  │  Inference         │    │  Tool Execution                 │  │
│  │  (LiteLLM)         │    │  (FastMCP clients)              │  │
│  │  • Retry on failure│    │  • Parallel execution           │  │
│  │  • Provider agnostic│   │  • Schema conversion            │  │
│  │  • Tracing support │    │  • Tracing support              │  │
│  └────────────────────┘    └─────────────────────────────────┘  │
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
| `PENDING` | Task created, workflow hasn't started yet |
| `THINKING` | Agent is processing (LLM inference in progress) |
| `AWAITING_TOOL` | Waiting for tool execution (includes sub-agent delegation) |
| `AWAITING_APPROVAL` | Waiting for human approval on a tool call |
| `COMPLETED` | Task finished successfully |
| `FAILED` | Task failed with an error |

## Human-in-the-Loop Approvals

For high-stakes operations, require human approval before tool execution:

```python
from datetime import timedelta
from zap_ai import ApprovalRules

# Execute with approval rules
task = await zap.execute_task(
    agent_name="FinancialAgent",
    task="Transfer $50,000 to vendor",
    approval_rules=ApprovalRules(
        patterns=["transfer_*", "delete_*"],  # Glob patterns
        timeout=timedelta(days=7),
    ),
)

# Check for pending approvals
task = await zap.get_task(task.id)
if task.status == TaskStatus.AWAITING_APPROVAL:
    pending = await task.get_pending_approvals()
    for req in pending:
        print(f"Tool: {req['tool_name']}")
        print(f"Args: {req['tool_args']}")

        # Approve or reject
        await task.approve(req['id'])
        # Or: await task.reject(req['id'], reason="Amount too high")
```

**Features:**
- **Durable** - Pending approvals survive worker restarts
- **Timeouts** - Auto-reject after configurable duration
- **Pattern matching** - Use glob patterns to match tool names
- **Tool discovery** - Preview which tools match your patterns

```python
# Discover tools and validate patterns
tools = await zap.get_agent_tools("FinancialAgent")
rules = ApprovalRules(patterns=["transfer_*"])
print(rules.preview_matches(tools))  # See what matches
```

## Conversation History API

The `Task` object provides methods to inspect the conversation history:

```python
task = await zap.get_task(task_id)

# Get all text content (user + assistant messages, excluding tool calls)
text = task.get_text_content()

# Get all tool calls with their results
tool_calls = task.get_tool_calls()
for tc in tool_calls:
    print(f"{tc.name}({tc.arguments}) -> {tc.result}")

# Navigate by conversation turns
for turn in task.get_turns():
    print(f"Turn {turn.turn_number}:")
    print(f"  User: {turn.user_message}")
    print(f"  Assistant messages: {len(turn.assistant_messages)}")
    print(f"  Tool calls: {len(turn.tool_messages)}")

# Get a specific turn
first_turn = task.get_turn(0)

# Count turns
print(f"Total turns: {task.turn_count()}")

# For multi-agent scenarios: fetch sub-task details
sub_tasks = await task.get_sub_tasks()
for sub in sub_tasks:
    print(f"Sub-task {sub.id}: {sub.status}")
```

### Conversation Types

| Type | Description |
|------|-------------|
| `ToolCallInfo` | Tool call with `id`, `name`, `arguments`, and `result` |
| `ConversationTurn` | A turn with `turn_number`, `user_message`, `assistant_messages`, `tool_messages` |

## Observability

Zap supports tracing via a pluggable provider system. Langfuse is included out of the box, and you can implement custom providers by extending `BaseTracingProvider`.

### Langfuse Integration

1. **Install with Langfuse support:**
   ```bash
   pip install zap-ai[langfuse]
   ```

2. **Configure environment variables:**
   ```bash
   export LANGFUSE_PUBLIC_KEY="pk-..."
   export LANGFUSE_SECRET_KEY="sk-..."
   # Optional: export LANGFUSE_HOST="https://cloud.langfuse.com"
   ```

3. **Enable tracing in your application:**
   ```python
   from zap_ai.tracing import set_tracing_provider
   from zap_ai.tracing.langfuse_provider import LangfuseTracingProvider

   # Initialize the provider
   provider = LangfuseTracingProvider()
   set_tracing_provider(provider)

   # Your Zap code here...

   # Flush traces on shutdown
   await provider.flush()
   ```

### What Gets Traced

- **Task execution** - Each task becomes a trace
- **Agentic loop iterations** - Spans for each iteration
- **LLM calls** - Generation observations with token usage
- **Tool calls** - Tool observations with inputs/outputs
- **Sub-agent delegations** - Nested agent spans

## Limitations

- **No streaming** - Currently uses query-based polling for status. Real-time streaming may be added in future versions.
- **Temporal required** - You need a running Temporal cluster (local dev server or Temporal Cloud).
- **MCP tools only** - Tools must be exposed via MCP servers (FastMCP makes this easy).

## Future Plans

- Real-time streaming via callbacks/webhooks
- Hooks system for custom logic injection
- Expose agents as MCP servers for agent-to-agent communication
- Approval UI dashboard for managing approval queues

## Examples

The [`examples/`](examples/) folder contains working examples:

- **`simple_agent/`** - Basic single-agent setup with MCP tools
- **`multi_agent/`** - Multi-agent delegation with `message_agent`
- **`langfuse_tracing/`** - Observability with Langfuse integration
- **`conversation_history/`** - Conversation history inspection API

See the [examples README](examples/README.md) for detailed setup instructions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on [GitHub](https://github.com/zachrobo1/zap-ai).

## License

MIT License - see [LICENSE](LICENSE) for details.
