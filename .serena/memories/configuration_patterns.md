# Configuration Patterns

Reference for configuring Zap, Temporal, LLM providers, and MCP clients.

## Environment Variables

### Required for LLM Providers
```bash
# Anthropic (Claude models)
ANTHROPIC_API_KEY=sk-ant-api03-...

# OpenAI (GPT models)
OPENAI_API_KEY=sk-...

# At least one provider API key is required
```

### Optional - Temporal Configuration
```bash
# Temporal server address (default: localhost:7233)
TEMPORAL_ADDRESS=localhost:7233

# Task queue name (default: zap-task-queue)
ZAP_TASK_QUEUE=my-custom-queue

# Temporal Cloud (production)
TEMPORAL_ADDRESS=mycompany.tmprl.cloud:7233
TEMPORAL_NAMESPACE=mycompany.production
TEMPORAL_TLS_CERT=path/to/cert.pem
TEMPORAL_TLS_KEY=path/to/key.pem
```

### Optional - Observability
```bash
# Langfuse tracing
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # Or self-hosted URL
```

### Loading Environment Variables
```python
# Use python-dotenv for local development
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file

# In production, set via environment (Docker, K8s, etc.)
```

## LiteLLM Model Configuration

### Model Naming Convention
LiteLLM requires provider prefixes for some models:

```python
# Anthropic models (prefix required)
model="anthropic/claude-3-5-sonnet-20241022"
model="anthropic/claude-3-opus-20240229"
model="anthropic/claude-3-haiku-20240307"

# OpenAI models (prefix optional)
model="gpt-4o"           # ✓ Works
model="openai/gpt-4o"    # ✓ Also works
model="gpt-4o-mini"
model="gpt-3.5-turbo"
```

### Model Selection by Use Case

**Vision tasks** (image understanding):
```python
model="gpt-4o"                                      # OpenAI, fast & capable
model="anthropic/claude-3-5-sonnet-20241022"       # Anthropic, best quality
model="gpt-4o-mini"                                 # Budget option
```

**Text-only tasks**:
```python
model="gpt-4o"                                      # Fast, high quality
model="anthropic/claude-3-5-sonnet-20241022"       # Best for complex reasoning
model="gpt-3.5-turbo"                               # Budget option
model="anthropic/claude-3-haiku-20240307"          # Fast & cheap
```

**Long context**:
```python
model="anthropic/claude-3-5-sonnet-20241022"       # 200k tokens
model="gpt-4o"                                      # 128k tokens
```

### Model Parameters

```python
agent = ZapAgent(
    name="MyAgent",
    model="gpt-4o",
    temperature=0.7,        # Default: 0.7 (0.0-2.0, higher = more creative)
    max_tokens=4000,        # Default: 4000 (max response length)
    prompt="...",
)
```

**Temperature guidance**:
- `0.0` - Deterministic, factual tasks (data extraction, classification)
- `0.3-0.5` - Balanced (Q&A, summaries)
- `0.7-0.9` - Creative (writing, brainstorming)
- `1.0+` - Very creative (experimental, rarely needed)

## Temporal Configuration

### Local Development
```python
from zap_ai import Zap

# Default: connects to localhost:7233
zap = Zap(agents=[agent])
await zap.start()
```

### Production (Temporal Cloud)
```python
from temporalio.client import Client, TLSConfig
from zap_ai import Zap

# Create client with TLS
temporal_client = await Client.connect(
    target="mycompany.tmprl.cloud:7233",
    namespace="mycompany.production",
    tls=TLSConfig(
        client_cert=b"...",  # From file or env
        client_private_key=b"...",
    )
)

# Pass to Zap
zap = Zap(
    agents=[agent],
    temporal_client=temporal_client,
    task_queue="production-tasks"
)
await zap.start()
```

### Custom Task Queues
```python
# Use different task queues for different agent types
zap = Zap(
    agents=[agent],
    task_queue="high-priority-tasks"  # Default: "zap-task-queue"
)

# Worker must use same queue
from zap_ai.worker import create_worker
worker = await create_worker(zap, task_queue="high-priority-tasks")
```

## Worker Configuration

### Basic Worker (Development)
```python
from zap_ai import Zap, ZapAgent
from zap_ai.worker import run_worker

agent = ZapAgent(name="MyAgent", prompt="...", model="gpt-4o")
zap = Zap(agents=[agent])

# Run worker (blocks, runs forever)
await run_worker(zap)
```

### Worker with Custom Sandbox
```python
from zap_ai.worker import create_worker

# Add passthrough modules (imports allowed in workflows)
worker = await create_worker(
    zap,
    sandbox_config={
        "passthrough_modules": [
            "myapp",              # Your app modules
            "pydantic",           # Common libraries
            "dataclasses",
        ]
    }
)

await worker.run()
```

### Production Worker (Separate Process)
```python
# worker.py - separate from main app
from zap_ai.worker import create_production_runner, production_restrictions

async def main():
    # Configure agents
    agents = [
        ZapAgent(name="Agent1", ...),
        ZapAgent(name="Agent2", ...),
    ]
    
    # Production config
    runner = await create_production_runner(
        agents=agents,
        temporal_address=os.getenv("TEMPORAL_ADDRESS"),
        task_queue=os.getenv("ZAP_TASK_QUEUE", "zap-task-queue"),
        sandbox_config=production_restrictions()  # More permissive
    )
    
    await runner.run()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Worker Patterns

**Inline worker** (same process as Zap client):
```python
# Good for development, simple deployments
zap = Zap(agents=[agent])
await zap.start()

# In background task or separate thread
asyncio.create_task(run_worker(zap))

# Use Zap in same process
task = await zap.execute_task(...)
```

**Separate worker** (different process):
```python
# Production pattern - scale workers independently
# Process 1: API server
zap = Zap(agents=[agent])
await zap.start()
task = await zap.execute_task(...)  # Returns immediately

# Process 2: Worker (can scale horizontally)
await run_worker(zap)  # Executes workflows
```

## MCP Client Configuration

### File-Based MCP Server
```python
from fastmcp import Client

agent = ZapAgent(
    name="MyAgent",
    mcp_clients=[
        Client("./tools.py"),           # Relative to project root
        Client("/abs/path/to/tools.py"), # Absolute path
    ]
)
```

### HTTP-Based MCP Server
```python
agent = ZapAgent(
    name="MyAgent",
    mcp_clients=[
        Client("https://api.example.com/mcp"),
    ]
)
```

### MCP with Sampling Support
```python
# MCP server that needs LLM access
from zap_ai.mcp.sampling import create_mcp_client

client = await create_mcp_client(
    "path/to/tools.py",
    sampling_handler=LiteLLMSamplingHandler(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
)

agent = ZapAgent(
    name="MyAgent",
    mcp_clients=[client]
)
```

### Multiple MCP Servers
```python
# Agent can use tools from multiple servers
agent = ZapAgent(
    name="MultiToolAgent",
    mcp_clients=[
        Client("./database_tools.py"),    # Database operations
        Client("./email_tools.py"),       # Email sending
        Client("./slack_tools.py"),       # Slack integration
    ]
)

# Tools from all servers are available
# Naming conflicts: last server wins
```

## Complete Example

```python
import os
from dotenv import load_dotenv
from zap_ai import Zap, ZapAgent
from zap_ai.tracing import set_tracing_provider, LangfuseTracingProvider
from fastmcp import Client

# Load environment
load_dotenv()

# Configure tracing (optional)
set_tracing_provider(LangfuseTracingProvider())

# Define agents
agent = ZapAgent(
    name="Assistant",
    prompt="You are a helpful assistant.",
    model="anthropic/claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=4000,
    mcp_clients=[
        Client("./tools/database.py"),
        Client("./tools/email.py"),
    ]
)

# Create Zap instance
zap = Zap(
    agents=[agent],
    task_queue=os.getenv("ZAP_TASK_QUEUE", "zap-task-queue")
)

# Start and use
await zap.start()

try:
    task = await zap.execute_task(
        agent_name="Assistant",
        task="Help me with something",
        context={"user_id": "123"}
    )
    print(task.result)
finally:
    await zap.stop()
```
