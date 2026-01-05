"""Langfuse tracing example for Zap.

This example demonstrates:
- Configuring Zap with Langfuse tracing
- Traces appear in Langfuse dashboard for each task
- Observability for LLM calls, tool executions

Prerequisites:
1. Copy .env.example to .env and set:
   - ANTHROPIC_API_KEY
   - LANGFUSE_PUBLIC_KEY
   - LANGFUSE_SECRET_KEY
2. Start Temporal server: temporal server start-dev
3. Run this script: python main.py
4. View traces at https://cloud.langfuse.com

Note: Install with `pip install zap-ai[langfuse]` for Langfuse support.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from temporalio.client import Client as TemporalClient

from zap_ai import Zap, ZapAgent
from zap_ai.tracing import LangfuseTracingProvider
from zap_ai.worker import create_worker

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")


def check_env_vars() -> bool:
    """Check required environment variables are set."""
    required = ["ANTHROPIC_API_KEY"]
    langfuse_vars = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]

    missing = [var for var in required if not os.environ.get(var)]
    if missing:
        print(f"Error: Missing required env vars: {missing}")
        print("Set them in .env or export directly")
        return False

    missing_langfuse = [var for var in langfuse_vars if not os.environ.get(var)]
    if missing_langfuse:
        print(f"Warning: Langfuse vars not set: {missing_langfuse}")
        print("Tracing will be disabled.")
        print("To enable tracing, set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")
        return True

    return True


def create_tracing_provider() -> LangfuseTracingProvider | None:
    """Create Langfuse tracing provider if configured."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        print("Langfuse not configured - running without tracing")
        return None

    host = os.environ.get("LANGFUSE_HOST")  # Optional, defaults to cloud

    print("Langfuse tracing enabled!")
    print(f"  Host: {host or 'https://cloud.langfuse.com'}")

    return LangfuseTracingProvider(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


async def main() -> None:
    if not check_env_vars():
        return

    # Create tracing provider (None if not configured)
    tracing = create_tracing_provider()

    # Path to the tools server (reuse from simple_agent)
    tools_path = Path(__file__).parent.parent / "simple_agent" / "tools.py"

    if not tools_path.exists():
        print(f"Error: Tools file not found at {tools_path}")
        print("Make sure simple_agent/tools.py exists")
        return

    # Create an agent with tools
    assistant = ZapAgent(
        name="TracedAssistant",
        prompt="""You are a helpful assistant with access to tools.

You can:
- Get the current time using get_current_time
- Perform calculations using calculate
- Search the web using search_web

Be concise and helpful in your responses.""",
        model="anthropic/claude-sonnet-4-5-20250929",
        mcp_clients=[Client(str(tools_path))],
        max_iterations=10,
    )

    # Create Zap with tracing provider
    zap = Zap(
        agents=[assistant],
        tracing_provider=tracing,  # Pass the tracing provider here
    )

    # Connect to Temporal and create worker
    temporal_client = await TemporalClient.connect("localhost:7233")
    worker_task = None

    try:
        # Start Zap
        print("\nStarting Zap...")
        await zap.start()
        print(f"Registered agents: {zap.list_agents()}")

        # Create worker with tracing provider
        worker = await create_worker(
            temporal_client,
            task_queue=zap.task_queue,
            tool_registry=zap._tool_registry,
            tracing_provider=tracing,  # Pass tracing to worker too
        )

        # Start worker in background
        worker_task = asyncio.create_task(worker.run())
        print("Worker started in background")

        # Execute a task
        print("\n" + "=" * 50)
        print("Executing task (check Langfuse dashboard for trace)")
        print("=" * 50)

        task = await zap.execute_task(
            agent_name="TracedAssistant",
            task="What time is it? Also, what is 123 multiplied by 456?",
        )
        print(f"Task started: {task.id}")
        if tracing:
            print(f"Trace ID will match workflow ID: {task.id}")

        # Poll for completion
        print("\nWaiting for completion...")
        while not task.status.is_terminal():
            await asyncio.sleep(1)
            task = await zap.get_task(task.id)
            print(f"  Status: {task.status.value}")

        # Show result
        print("\n" + "=" * 50)
        if task.is_complete():
            print("Task completed successfully!")
            print(f"\nResult:\n{task.result}")
        else:
            print(f"Task failed: {task.error}")

        # Flush traces before exit
        if tracing:
            print("\nFlushing traces to Langfuse...")
            await tracing.flush()
            print("Done! Check your Langfuse dashboard to see the trace.")

    finally:
        # Clean shutdown
        print("\nShutting down...")
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        if tracing:
            await tracing.shutdown()

        await zap.stop()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
