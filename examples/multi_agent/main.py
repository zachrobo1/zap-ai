"""Multi-agent example with delegation for Zap.

This example demonstrates:
- Creating multiple agents with different specializations
- Setting up sub-agent relationships for delegation
- Using the message_agent tool to delegate tasks
- Langfuse tracing for observability across sub-agent delegation

The setup:
- Coordinator: Main agent that routes tasks to specialists
- Researcher: Agent that searches for information
- Calculator: Agent specialized in math calculations

Prerequisites:
1. Copy .env.example to .env and set ANTHROPIC_API_KEY
2. Optionally set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY for tracing
3. Start Temporal server: temporal server start-dev
4. Run this script: python main.py
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


def create_tracing_provider() -> LangfuseTracingProvider | None:
    """Create Langfuse tracing provider if configured."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        print("Langfuse not configured - running without tracing")
        return None

    host = os.environ.get("LANGFUSE_HOST")
    print("Langfuse tracing enabled!")
    print(f"  Host: {host or 'https://cloud.langfuse.com'}")

    return LangfuseTracingProvider(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


async def main() -> None:
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Set it in .env or export ANTHROPIC_API_KEY=your-key")
        return

    # Create tracing provider (None if not configured)
    tracing = create_tracing_provider()

    # Path to the tools server
    tools_path = Path(__file__).parent / "tools.py"
    tools_client = Client(str(tools_path))

    # Create specialist agents
    researcher = ZapAgent(
        name="Researcher",
        prompt="""You are a research specialist. Your job is to search for
information and provide comprehensive answers.

Use the search_web tool to find relevant information.
Summarize findings clearly and cite your sources.""",
        model="anthropic/claude-sonnet-4-5-20250929",
        mcp_clients=[tools_client],
        discovery_prompt="I can search the web and research any topic.",
    )

    calculator = ZapAgent(
        name="Calculator",
        prompt="""You are a math specialist. Your job is to perform calculations
and explain mathematical concepts.

Use the calculate tool for arithmetic operations.
Show your work and explain the steps.""",
        model="anthropic/claude-sonnet-4-5-20250929",
        mcp_clients=[tools_client],
        discovery_prompt="I can perform calculations and solve math problems.",
    )

    # Create coordinator agent that can delegate to specialists
    coordinator = ZapAgent(
        name="Coordinator",
        prompt="""You are a helpful coordinator that routes tasks to the right specialist.

You have access to specialist agents:
- Researcher: For web searches and research tasks
- Calculator: For math calculations

Analyze the user's request and delegate to the appropriate specialist
using the message_agent tool. If the task requires multiple specialists,
coordinate between them.

Always summarize the final results for the user.""",
        model="anthropic/claude-sonnet-4-5-20250929",
        mcp_clients=[tools_client],  # Also has direct tool access
        sub_agents=["Researcher", "Calculator"],  # Can delegate to these
        max_iterations=15,
    )

    # Create Zap with all agents and tracing
    zap = Zap(
        agents=[coordinator, researcher, calculator],
        tracing_provider=tracing,
    )

    # Connect to Temporal and create worker
    temporal_client = await TemporalClient.connect("localhost:7233")
    worker = None
    worker_task = None

    try:
        print("Starting Zap...")
        await zap.start()
        print(f"Registered agents: {zap.list_agents()}")

        # Create worker with the tool registry from Zap
        # This runs the worker in the same process so it has access to MCP clients
        worker = await create_worker(
            temporal_client,
            task_queue=zap.task_queue,
            tool_registry=zap._tool_registry,
            tracing_provider=tracing,
        )

        # Start worker in background
        worker_task = asyncio.create_task(worker.run())
        print("Worker started in background")

        # Execute a task that requires delegation
        print("\nExecuting multi-step task...")
        task = await zap.execute_task(
            agent_name="Coordinator",
            task="""I need help with two things:
1. Search for information about the Pythagorean theorem
2. Calculate 3^2 + 4^2 to verify that it equals 5^2

Please coordinate with the appropriate specialists.""",
        )
        print(f"Task started: {task.id}")

        # Poll for completion
        print("\nWaiting for completion...")
        poll_count = 0
        while not task.status.is_terminal():
            await asyncio.sleep(2)
            task = await zap.get_task(task.id)
            poll_count += 1
            print(f"  [{poll_count}] Status: {task.status.value}")

            # Safety limit
            if poll_count > 30:
                print("Timeout waiting for completion")
                break

        # Show result
        print("\n" + "=" * 60)
        if task.is_complete():
            print("Task completed successfully!")
            print(f"\nFinal Result:\n{task.result}")
        else:
            print(f"Task ended with status: {task.status.value}")
            if task.error:
                print(f"Error: {task.error}")

        # Flush traces before exit
        if tracing:
            print("\nFlushing traces to Langfuse...")
            await tracing.flush()
            print("Done! Check your Langfuse dashboard to see the trace.")

    finally:
        print("\nShutting down...")
        # Cancel worker task
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
