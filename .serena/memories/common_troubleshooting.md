# Common Troubleshooting

Quick reference for debugging common Zap issues.

## Temporal Issues

### Connection Refused
**Error**: `Failed to connect to Temporal server at localhost:7233`

**Cause**: Temporal server not running

**Solution**:
```bash
# Start Temporal server in separate terminal
temporal server start-dev

# Or check if running
lsof -i :7233
```

### Worker Not Picking Up Tasks
**Symptoms**: Task stays in PENDING, workflow never starts

**Possible Causes**:
1. Worker not running or crashed
2. Worker connected to different task queue
3. Worker sandbox restrictions blocking imports

**Solutions**:
```python
# Check task queue matches
zap = Zap(agents=[agent], task_queue="my-queue")  # Must match worker
worker = await create_worker(zap, task_queue="my-queue")

# Check worker logs for sandbox errors
# Add passthrough modules if needed
from zap_ai.worker import create_worker

worker = await create_worker(
    zap,
    sandbox_config={
        "passthrough_modules": ["mymodule", "another_module"]
    }
)
```

### Workflow Event History Too Large
**Error**: Event history exceeds limits after many iterations

**Cause**: Long-running conversations without continue-as-new

**Solution**: This is handled automatically by Zap via continue-as-new, but if you see this error:
- Check if you're using a very old version
- Reduce max_iterations in workflow config
- Break long tasks into smaller sub-tasks

## MCP/Tool Issues

### Tool Not Found
**Error**: `ToolNotFoundError: Tool 'my_tool' not found for agent 'MyAgent'`

**Causes**:
1. MCP server path incorrect
2. Tool not registered in MCP server
3. MCP client connection failed

**Solutions**:
```bash
# Test MCP server in isolation
fastmcp dev path/to/tools.py
# Should list all available tools

# Check MCP client configuration
agent = ZapAgent(
    name="MyAgent",
    mcp_clients=[
        Client("./tools.py"),  # Relative path from project root
        # or
        Client("https://example.com/mcp")  # HTTP endpoint
    ]
)

# Check agent tools are loading
tools = await zap.get_agent_tools("MyAgent")
print([t["name"] for t in tools])
```

### Client Connection Error
**Error**: `ClientConnectionError: Failed to connect to MCP server`

**Causes**:
1. File path doesn't exist
2. HTTP endpoint unreachable
3. MCP server startup failure

**Solutions**:
```bash
# Verify file exists
ls -la path/to/tools.py

# Test HTTP endpoint
curl https://example.com/mcp

# Check MCP server logs
fastmcp dev path/to/tools.py  # Look for startup errors
```

### Schema Conversion Failure
**Error**: `SchemaConversionError: Cannot convert MCP schema to LiteLLM format`

**Cause**: Tool uses unsupported parameter types

**Solution**:
- MCP tools must use JSON-serializable types (str, int, float, bool, list, dict)
- Avoid complex types like datetime, custom classes, etc.
- Use Pydantic models for validation, but convert to dict/list for MCP

## LLM Provider Issues

### Vision Not Supported
**Error**: `VisionNotSupportedError: Model 'gpt-3.5-turbo' does not support vision`

**Cause**: Sending images to non-vision model

**Solution**:
```python
# Use vision-capable models
agent = ZapAgent(
    name="VisionAgent",
    model="gpt-4o",           # ✓ Supports vision
    # model="gpt-3.5-turbo",  # ✗ No vision support
)

# Vision-capable models:
# - gpt-4o, gpt-4o-mini (OpenAI)
# - claude-3.5-sonnet, claude-3-opus, claude-3-haiku (Anthropic)
```

### Missing API Key
**Error**: `LLMProviderError: API key not found`

**Cause**: Environment variables not set

**Solution**:
```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your keys
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Verify keys are loaded
python -c "import os; print(os.getenv('ANTHROPIC_API_KEY'))"
```

### Rate Limit Errors
**Error**: `RateLimitError: Rate limit exceeded`

**Cause**: Too many requests to LLM provider

**Solution**: Temporal automatically retries with exponential backoff. If persistent:
- Use a higher rate limit tier with your provider
- Reduce concurrency (run fewer agents in parallel)
- Add delays between tasks

## Testing Issues

### Integration Tests Hanging
**Symptoms**: Tests run forever, never complete

**Cause**: Temporal server not running or workflow stuck

**Solution**:
```bash
# Ensure Temporal running
temporal server start-dev

# Check if workflow is stuck in Temporal UI
open http://localhost:8233

# Add timeout to tests
@pytest.mark.timeout(30)  # Fail after 30 seconds
async def test_my_workflow():
    ...
```

### Sandbox Restrictions
**Error**: `RestrictedWorkflowAccessError: Cannot import module 'X'`

**Cause**: Temporal sandbox blocks non-deterministic imports

**Solution**:
```python
# Add to sandbox passthrough
worker = await create_worker(
    zap,
    sandbox_config={
        "passthrough_modules": ["mymodule"]
    }
)

# Or use production restrictions (more permissive)
from zap_ai.worker import create_worker, production_restrictions

worker = await create_worker(
    zap,
    sandbox_config=production_restrictions()
)
```

### Mock Issues
**Problem**: Mocks not working as expected in unit tests

**Solutions**:
```python
# Mock external dependencies
from unittest.mock import AsyncMock, MagicMock, patch

# Mock LiteLLM
with patch('zap_ai.llm.provider.completion') as mock_completion:
    mock_completion.return_value = {"choices": [...]}
    ...

# Mock MCP clients
mock_client = MagicMock()
mock_client.call_tool = AsyncMock(return_value={"result": "..."})
```

## Debugging Techniques

### Enable Verbose Logging
```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("zap_ai").setLevel(logging.DEBUG)
```

### Inspect Task History
```python
task = await zap.get_task(task_id)

# View conversation turns
for i, turn in enumerate(task.get_turns()):
    print(f"Turn {i}: {turn.role}")
    print(f"  Content: {turn.content[:100]}...")
    
# View tool calls
for call in task.get_tool_calls():
    print(f"{call.name}({call.arguments}) → {call.result}")
```

### Use Temporal UI
```bash
# Start Temporal with UI
temporal server start-dev

# Open in browser
open http://localhost:8233

# View workflow execution:
# - Event history
# - Activity execution
# - Pending signals (approvals)
```

### Test MCP Tools in Isolation
```bash
# Use FastMCP dev mode to test tools
fastmcp dev path/to/tools.py

# Interactive REPL for testing tools
# Call tools directly without LLM
```

### Check Workflow State
```python
# Query workflow state
from temporalio.client import Client as TemporalClient

temporal_client = await TemporalClient.connect("localhost:7233")
handle = temporal_client.get_workflow_handle(task.id)

# Query current state
state = await handle.query("get_state")
print(state)
```

## When to File a Bug

If you've tried the above and still have issues, file a bug with:
1. Full error message and stack trace
2. Minimal reproducible example
3. Zap version (`pip show zap-ai`)
4. Python version (`python --version`)
5. Temporal version (`temporal server --version`)
6. Relevant logs (worker, Temporal UI)

File at: https://github.com/zachrobo1/zap-ai/issues
