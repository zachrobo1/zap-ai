"""Simple single-agent example for Zap.

This example demonstrates:
- Creating a ZapAgent with tools
- Starting Zap and executing a task
- Polling for task completion

Prerequisites:
1. Copy .env.example to .env and set ANTHROPIC_API_KEY
2. Start Temporal server: temporal server start-dev
3. Run this script: python main.py

Note: LiteLLM requires the 'anthropic/' prefix for Claude models.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from temporalio.client import Client as TemporalClient

from zap_ai import Zap, ZapAgent
from zap_ai.worker import create_worker

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")


async def main() -> None:
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Set it in .env or export ANTHROPIC_API_KEY=your-key")
        return

    # Path to the tools server
    tools_path = Path(__file__).parent / "tools.py"

    # Create an agent with tools
    assistant = ZapAgent(
        name="Assistant",
        prompt="""You are a helpful assistant with access to tools.

You can:
- Get the current time using get_current_time
- Perform calculations using calculate
- Search the web using search_web

Be concise and helpful in your responses.""",
        model="anthropic/claude-sonnet-4-5-20250929",  # LiteLLM format
        mcp_clients=[Client(str(tools_path))],
        max_iterations=10,
    )

    # Create Zap instance
    zap = Zap(agents=[assistant])

    # Connect to Temporal and create worker
    temporal_client = await TemporalClient.connect("localhost:7233")
    worker_task = None

    try:
        # Start Zap (connects to Temporal, initializes tools)
        print("Starting Zap...")
        await zap.start()
        print(f"Registered agents: {zap.list_agents()}")

        # Create worker with the tool registry from Zap
        # This runs the worker in the same process so it has access to MCP clients
        worker = await create_worker(
            temporal_client,
            task_queue=zap.task_queue,
            tool_registry=zap._tool_registry,
        )

        # Start worker in background
        worker_task = asyncio.create_task(worker.run())
        print("Worker started in background")

        # Execute a task
        print("\nExecuting task...")
        task = await zap.execute_task(
            agent_name="Assistant",
            task="What time is it? Also, what is 42 multiplied by 17?",
        )
        print(f"Task started: {task.id}")
        print(f"Initial status: {task.status}")

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

        # Show conversation history
        if task.history:
            print("\n" + "=" * 50)
            print("Conversation history:")
            for msg in task.history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    print(f"\n[{role}]: {content[:200]}...")

    finally:
        # Clean shutdown
        print("\nShutting down...")
        # Cancel worker task
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        await zap.stop()
        print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
