# Phase 4: Worker

**Goal:** Worker setup to run and test the workflow loop with stub activities.

**Dependencies:** Phase 2 (Temporal Workflows), Phase 3 (Activities - Stubs)

---

## Overview

This phase creates the Temporal worker that runs Zap agent workflows. After this phase, you can run the full workflow loop with mock LLM responses and mock tool results.

**Checkpoint:** At this point, you can:
- Start a Temporal worker that processes Zap workflows
- Execute workflows with stub activities
- Verify the full agentic loop works before adding real integrations

---

## Task 4.1: Implement Worker Setup

**File:** `src/zap_ai/worker/worker.py`

Create the worker setup for running Temporal activities and workflows:

```python
"""Temporal worker setup for Zap agents.

This module provides functions to create and run Temporal workers
that execute Zap agent workflows and activities.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from temporalio.client import Client
from temporalio.worker import Worker

from zap_ai.activities.inference import inference_activity
from zap_ai.activities.tool_execution import (
    set_tool_registry,
    tool_execution_activity,
)
from zap_ai.workflows.agent_workflow import AgentWorkflow

if TYPE_CHECKING:
    from zap_ai.core.zap import Zap


async def create_worker(
    client: Client,
    task_queue: str,
    tool_registry: Any | None = None,
) -> Worker:
    """
    Create a Temporal worker for Zap agents.

    Args:
        client: Temporal client instance.
        task_queue: Task queue name to listen on.
        tool_registry: Optional ToolRegistry for activities.
            If None, stub activities will work but real tool
            execution will fail.

    Returns:
        Configured Worker instance (not started).
    """
    # Set global registry for activities (None is OK for stubs)
    set_tool_registry(tool_registry)

    return Worker(
        client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        activities=[
            inference_activity,
            tool_execution_activity,
        ],
    )


async def run_worker(
    temporal_address: str = "localhost:7233",
    task_queue: str = "zap-agents",
    tool_registry: Any | None = None,
) -> None:
    """
    Run a Temporal worker for Zap agents.

    This is a blocking function that runs the worker until interrupted.

    Args:
        temporal_address: Temporal server address.
        task_queue: Task queue name to listen on.
        tool_registry: Optional ToolRegistry for activities.
    """
    client = await Client.connect(temporal_address)
    worker = await create_worker(client, task_queue, tool_registry)

    print(f"Starting Zap worker on task queue '{task_queue}'...")
    print(f"Connected to Temporal at {temporal_address}")

    async with worker:
        print("Worker running. Press Ctrl+C to stop.")
        await asyncio.Event().wait()


async def run_worker_with_zap(
    zap: Zap,
    temporal_address: str = "localhost:7233",
) -> None:
    """
    Run a Temporal worker using a Zap instance's configuration.

    Args:
        zap: Initialized Zap instance (start() must have been called).
        temporal_address: Temporal server address.
    """
    if not zap._started:
        raise RuntimeError("Zap must be started before running worker")

    tool_registry = getattr(zap, "_tool_registry", None)

    await run_worker(
        temporal_address=temporal_address,
        task_queue=zap.task_queue,
        tool_registry=tool_registry,
    )
```

**Checklist:**
- [x] Create `src/zap_ai/worker/worker.py`
- [x] Implement `create_worker()` factory function
- [x] Implement `run_worker()` blocking runner
- [x] Implement `run_worker_with_zap()` convenience function
- [x] Set global tool registry for activities
- [x] Register workflows and activities with worker
- [x] Add console output for worker status

---

## Task 4.2: Add Worker to Exports

**File:** `src/zap_ai/worker/__init__.py`

```python
"""Temporal worker setup for Zap agents."""

from zap_ai.worker.worker import (
    create_worker,
    run_worker,
    run_worker_with_zap,
)

__all__ = [
    "create_worker",
    "run_worker",
    "run_worker_with_zap",
]
```

**Checklist:**
- [x] Update `src/zap_ai/worker/__init__.py`
- [x] Export `create_worker`, `run_worker`, `run_worker_with_zap`
- [x] Define `__all__` list

---

## Task 4.3: Add CLI Entry Point

**File:** `src/zap_ai/worker/__main__.py`

Create a CLI entry point for running the worker:

```python
"""CLI entry point for running Zap worker.

Usage:
    python -m zap_ai.worker
    python -m zap_ai.worker --temporal-address localhost:7233
    python -m zap_ai.worker --task-queue my-agents
"""

from __future__ import annotations

import argparse
import asyncio

from zap_ai.worker import run_worker


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run a Zap agent worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--temporal-address",
        default="localhost:7233",
        help="Temporal server address",
    )
    parser.add_argument(
        "--task-queue",
        default="zap-agents",
        help="Task queue name",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            run_worker(
                temporal_address=args.temporal_address,
                task_queue=args.task_queue,
            )
        )
    except KeyboardInterrupt:
        print("\nWorker stopped.")


if __name__ == "__main__":
    main()
```

**Usage:**
```bash
# Run with defaults
python -m zap_ai.worker

# Custom configuration
python -m zap_ai.worker --temporal-address localhost:7233 --task-queue my-agents

# Show help
python -m zap_ai.worker --help
```

**Checklist:**
- [x] Create `src/zap_ai/worker/__main__.py`
- [x] Parse command-line arguments
- [x] Run worker with provided configuration
- [x] Handle KeyboardInterrupt gracefully

---

## Phase 4 Verification

After completing all tasks, verify:

1. **Worker module imports work:**
   ```python
   from zap_ai.worker import create_worker, run_worker, run_worker_with_zap
   ```

2. **CLI help works:**
   ```bash
   python -m zap_ai.worker --help
   ```

3. **Worker can be created (unit test):**
   ```python
   from temporalio.testing import WorkflowEnvironment
   from zap_ai.worker import create_worker

   async with await WorkflowEnvironment.start_time_skipping() as env:
       worker = await create_worker(env.client, "test-queue")
       assert worker is not None
   ```

4. **Full workflow loop with stubs (integration test):**
   ```python
   from temporalio.testing import WorkflowEnvironment
   from temporalio.worker import Worker
   from zap_ai.workflows import AgentWorkflow, AgentWorkflowInput
   from zap_ai.activities import inference_activity, tool_execution_activity

   async with await WorkflowEnvironment.start_time_skipping() as env:
       async with Worker(
           env.client,
           task_queue="test-queue",
           workflows=[AgentWorkflow],
           activities=[inference_activity, tool_execution_activity],
       ):
           result = await env.client.execute_workflow(
               AgentWorkflow.run,
               AgentWorkflowInput(agent_name="Test", initial_task="Hello"),
               id="test-1",
               task_queue="test-queue",
           )
           assert "[STUB]" in result
   ```

5. **All tests pass:**
   ```bash
   uv run pytest -v
   ```

---

## Checkpoint: Running the Full Loop

At this point, you can run the complete workflow loop:

**Terminal 1 - Start Temporal (dev server):**
```bash
temporal server start-dev
```

**Terminal 2 - Start Worker:**
```bash
python -m zap_ai.worker
```

**Terminal 3 - Execute Workflow:**
```python
import asyncio
from temporalio.client import Client
from zap_ai.workflows import AgentWorkflow, AgentWorkflowInput

async def main():
    client = await Client.connect("localhost:7233")

    result = await client.execute_workflow(
        AgentWorkflow.run,
        AgentWorkflowInput(agent_name="Test", initial_task="Hello, world!"),
        id="test-workflow-1",
        task_queue="zap-agents",
    )

    print(f"Result: {result}")

asyncio.run(main())
```

Expected output: `Result: [STUB] Processed: Hello, world!`

---

## Notes

- The worker registers both workflows and activities
- Tool registry is optional for stub activities
- Real tool execution requires Phase 6 (MCP Integration)
- Real LLM inference requires Phase 5 (LLM Integration)
- The `run_worker_with_zap()` function is for production use after Phase 7
