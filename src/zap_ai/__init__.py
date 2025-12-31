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

from zap_ai.core import (
    AgentNotFoundError,
    Task,
    TaskNotFoundError,
    TaskStatus,
    Zap,
    ZapAgent,
    ZapConfigurationError,
    ZapNotStartedError,
)

__version__ = "0.1.0"

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
    # Metadata
    "__version__",
]
