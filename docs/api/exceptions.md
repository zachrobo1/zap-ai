# Exceptions

All Zap exceptions inherit from `ZapError`.

## Exception Hierarchy

```
ZapError
├── ZapConfigurationError
├── ZapNotStartedError
├── AgentNotFoundError
├── TaskNotFoundError
├── ToolNotFoundError
├── ToolExecutionError
├── ClientConnectionError
├── SchemaConversionError
└── LLMProviderError
```

## ZapError

::: zap_ai.exceptions.ZapError
    options:
      show_source: true

## Configuration Errors

::: zap_ai.exceptions.ZapConfigurationError
    options:
      show_source: true

::: zap_ai.exceptions.ZapNotStartedError
    options:
      show_source: true

## Not Found Errors

::: zap_ai.exceptions.AgentNotFoundError
    options:
      show_source: true

::: zap_ai.exceptions.TaskNotFoundError
    options:
      show_source: true

::: zap_ai.exceptions.ToolNotFoundError
    options:
      show_source: true

## Execution Errors

::: zap_ai.exceptions.ToolExecutionError
    options:
      show_source: true

::: zap_ai.exceptions.ClientConnectionError
    options:
      show_source: true

::: zap_ai.exceptions.SchemaConversionError
    options:
      show_source: true

::: zap_ai.exceptions.LLMProviderError
    options:
      show_source: true

## Handling Errors

```python
from zap_ai import Zap, ZapAgent
from zap_ai.exceptions import (
    ZapNotStartedError,
    AgentNotFoundError,
    TaskNotFoundError,
)

zap = Zap(agents=[...])

try:
    # This will fail - Zap not started
    await zap.execute_task(agent_name="Test", task="Hello")
except ZapNotStartedError:
    print("Call zap.start() first!")

await zap.start()

try:
    # This will fail - agent doesn't exist
    await zap.execute_task(agent_name="NonExistent", task="Hello")
except AgentNotFoundError as e:
    print(f"Agent not found: {e}")

try:
    # This will fail - task doesn't exist
    await zap.get_task("invalid-task-id")
except TaskNotFoundError:
    print("Task not found")
```
