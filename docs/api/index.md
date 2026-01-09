# API Reference

This section contains auto-generated documentation from the Zap source code.

## Main Exports

The following are available from `zap_ai`:

```python
from zap_ai import (
    # Main classes
    Zap,
    ZapAgent,
    Task,
    TaskStatus,

    # Conversation types
    ToolCallInfo,
    ConversationTurn,

    # Type helpers
    TContext,
    DefaultContext,
    DynamicPrompt,

    # Exceptions
    ZapConfigurationError,
    ZapNotStartedError,
    AgentNotFoundError,
    TaskNotFoundError,
)
```

## Modules

| Module | Description |
|--------|-------------|
| [Core](core.md) | Main classes: `Zap`, `ZapAgent`, `Task`, `TaskStatus` |
| [Conversation](conversation.md) | History parsing: `ConversationTurn`, `ToolCallInfo` |
| [Tracing](tracing.md) | Observability: tracing providers and protocols |
| [Worker](worker.md) | Temporal worker utilities |
| [Exceptions](exceptions.md) | Error types |

## Quick Links

- [`Zap`](core.md#zap_ai.Zap) - Main orchestrator class
- [`ZapAgent`](core.md#zap_ai.ZapAgent) - Agent configuration
- [`Task`](core.md#zap_ai.Task) - Task execution tracking
- [`TaskStatus`](core.md#zap_ai.TaskStatus) - Task status enum
