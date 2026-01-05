"""Temporal worker setup for Zap agents.

This module provides functions to create and run Temporal workers
that execute Zap agent workflows and activities.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
    SandboxRestrictions,
)

from zap_ai.activities.inference import inference_activity
from zap_ai.activities.tool_execution import (
    get_agent_config_activity,
    set_tool_registry,
    tool_execution_activity,
)
from zap_ai.tracing import TracingProvider, set_tracing_provider
from zap_ai.workflows.agent_workflow import AgentWorkflow

if TYPE_CHECKING:
    from zap_ai.core.zap import Zap


# Configure sandbox to pass through problematic modules
# beartype causes circular import issues in the sandbox
_SANDBOX_RESTRICTIONS = SandboxRestrictions.default.with_passthrough_modules(
    "beartype",
)


async def create_worker(
    client: Client,
    task_queue: str,
    tool_registry: Any | None = None,
    tracing_provider: TracingProvider | None = None,
) -> Worker:
    """
    Create a Temporal worker for Zap agents.

    Args:
        client: Temporal client instance.
        task_queue: Task queue name to listen on.
        tool_registry: Optional ToolRegistry for activities.
            If None, stub activities will work but real tool
            execution will fail.
        tracing_provider: Optional TracingProvider for observability.
            If None, tracing is disabled (NoOpTracingProvider used).

    Returns:
        Configured Worker instance (not started).

    Example:
        ```python
        from temporalio.client import Client
        from zap_ai.worker import create_worker

        client = await Client.connect("localhost:7233")
        worker = await create_worker(client, "zap-agents")

        async with worker:
            await asyncio.Event().wait()
        ```
    """
    # Set global registry for activities (None is OK for stubs)
    set_tool_registry(tool_registry)

    # Set global tracing provider for activities
    if tracing_provider:
        set_tracing_provider(tracing_provider)

    return Worker(
        client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        activities=[
            inference_activity,
            tool_execution_activity,
            get_agent_config_activity,
        ],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_SANDBOX_RESTRICTIONS),
    )


async def run_worker(
    temporal_address: str = "localhost:7233",
    task_queue: str = "zap-agents",
    tool_registry: Any | None = None,
    tracing_provider: TracingProvider | None = None,
) -> None:
    """
    Run a Temporal worker for Zap agents.

    This is a blocking function that runs the worker until interrupted.

    Args:
        temporal_address: Temporal server address.
        task_queue: Task queue name to listen on.
        tool_registry: Optional ToolRegistry for activities.
            If None, stub activities will work but real tool
            execution will fail.
        tracing_provider: Optional TracingProvider for observability.
            If None, tracing is disabled (NoOpTracingProvider used).

    Example:
        ```python
        import asyncio
        from zap_ai.worker import run_worker

        # Run with defaults (stubs only)
        asyncio.run(run_worker())

        # Run with custom configuration
        asyncio.run(run_worker(
            temporal_address="temporal.example.com:7233",
            task_queue="my-agents",
        ))
        ```
    """
    # Connect to Temporal
    client = await Client.connect(temporal_address)

    # Create worker
    worker = await create_worker(client, task_queue, tool_registry, tracing_provider)

    print(f"Starting Zap worker on task queue '{task_queue}'...")
    print(f"Connected to Temporal at {temporal_address}")

    async with worker:
        print("Worker running. Press Ctrl+C to stop.")
        # Run forever until interrupted
        await asyncio.Event().wait()


async def run_worker_with_zap(
    zap: Zap,
    temporal_address: str = "localhost:7233",
) -> None:
    """
    Run a Temporal worker using a Zap instance's configuration.

    This is a convenience function that uses the Zap instance's
    task queue, tool registry, and tracing provider.

    Args:
        zap: Initialized Zap instance (start() must have been called).
        temporal_address: Temporal server address.

    Raises:
        RuntimeError: If Zap has not been started.

    Example:
        ```python
        import asyncio
        from zap_ai import Zap, ZapAgent
        from zap_ai.worker import run_worker_with_zap

        agents = [ZapAgent(name="Main", prompt="You are helpful")]
        zap = Zap(agents=agents)

        async def main():
            await zap.start()
            await run_worker_with_zap(zap)

        asyncio.run(main())
        ```
    """
    if not zap._started:
        raise RuntimeError("Zap must be started before running worker")

    # Get tool registry if available (Phase 6+)
    tool_registry = getattr(zap, "_tool_registry", None)

    # Get tracing provider if available (Phase 10+)
    tracing_provider = getattr(zap, "_tracing", None)

    await run_worker(
        temporal_address=temporal_address,
        task_queue=zap.task_queue,
        tool_registry=tool_registry,
        tracing_provider=tracing_provider,
    )
