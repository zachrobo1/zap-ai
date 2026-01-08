"""
Zap - Zach's Agent Platform

A library for building resilient AI agents on Temporal.

Example:
    ```python
    from zap_ai import Zap, ZapAgent, TaskStatus
    from fastmcp import Client

    agent = ZapAgent(
        name="MyAgent",
        prompt="You are a helpful assistant.",
        mcp_clients=[Client("./tools.py")],
    )

    zap = Zap(agents=[agent])
    await zap.start()

    task = await zap.execute_task(
        agent_name="MyAgent",
        task="Help me with something",
    )

    while not task.status.is_terminal():
        await asyncio.sleep(1)
        task = await zap.get_task(task.id)

    print(task.result)
    ```
"""

from importlib.metadata import version

from zap_ai.core import (
    AgentNotFoundError,
    DefaultContext,
    DynamicPrompt,
    Task,
    TaskNotFoundError,
    TaskStatus,
    TContext,
    Zap,
    ZapAgent,
    ZapConfigurationError,
    ZapNotStartedError,
)

__version__ = version("zap-ai")

__all__ = [
    # Main classes
    "Zap",
    "ZapAgent",
    "Task",
    "TaskStatus",
    # Exceptions
    "ZapConfigurationError",
    "ZapNotStartedError",
    "AgentNotFoundError",
    "TaskNotFoundError",
    # Types
    "TContext",
    "DefaultContext",
    "DynamicPrompt",
    # Metadata
    "__version__",
]
