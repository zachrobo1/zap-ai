# Worker API

Utilities for running the Temporal worker that executes agent workflows.

## run_worker

::: zap_ai.worker.run_worker
    options:
      show_source: true

## create_worker

::: zap_ai.worker.create_worker
    options:
      show_source: true

## Usage

### Simple Worker

For most use cases, use `run_worker()`:

```python title="worker.py"
import asyncio
from zap_ai.worker import run_worker

asyncio.run(run_worker())
```

Run it:

```bash
python worker.py
```

### Custom Worker

For more control, use `create_worker()`:

```python
import asyncio
from temporalio.client import Client
from zap_ai.worker import create_worker

async def main():
    # Connect to Temporal
    client = await Client.connect("localhost:7233")

    # Create worker with custom task queue
    worker = await create_worker(
        client,
        task_queue="my-custom-queue",
    )

    # Run the worker
    await worker.run()

asyncio.run(main())
```

### Inline Worker (for examples)

Run the worker in the same process as your application:

```python
import asyncio
from temporalio.client import Client
from zap_ai import Zap, ZapAgent
from zap_ai.worker import create_worker

async def main():
    agent = ZapAgent(name="MyAgent", prompt="...")
    zap = Zap(agents=[agent])

    # Connect to Temporal
    temporal_client = await Client.connect("localhost:7233")

    await zap.start()

    # Create worker with tool registry from Zap
    worker = await create_worker(
        temporal_client,
        task_queue=zap.task_queue,
        tool_registry=zap._tool_registry,
    )

    # Run worker in background
    worker_task = asyncio.create_task(worker.run())

    try:
        # Execute tasks
        task = await zap.execute_task(
            agent_name="MyAgent",
            task="Hello!",
        )
        # ... handle task ...
    finally:
        worker_task.cancel()
        await zap.stop()

asyncio.run(main())
```

!!! note
    Running the worker inline is useful for examples and testing but not recommended for production. In production, run workers as separate processes for better resource management and fault isolation.
