"""MCP Sampling Example for Zap with Langfuse Tracing.

This example demonstrates:
- Creating an MCP client with sampling support
- Using the LiteLLMSamplingHandler
- Running an agent with MCP tools that can request LLM completions
- Tracing sampling calls with Langfuse

Prerequisites:
1. Copy .env.example to .env and set:
   - ANTHROPIC_API_KEY
   - LANGFUSE_PUBLIC_KEY (optional, for tracing)
   - LANGFUSE_SECRET_KEY (optional, for tracing)
2. Start Temporal server: temporal server start-dev
3. Run this script: python main.py
4. View traces at https://cloud.langfuse.com (if Langfuse is configured)

Note: The tools server in this example demonstrates how an MCP server
can request LLM completions from the client.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from temporalio.client import Client as TemporalClient

from zap_ai import Zap, ZapAgent
from zap_ai.mcp.sampling import create_mcp_client
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

    host = os.environ.get("LANGFUSE_HOST")  # Optional, defaults to cloud

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

    # Path to the sampling tools server
    tools_path = Path(__file__).parent / "tools.py"

    # Create MCP client with sampling support and tracing
    tools_client = create_mcp_client(
        str(tools_path),
        sampling_handler="litellm",
        sampling_model="anthropic/claude-sonnet-4-5-20250929",
        enable_tracing=tracing is not None,
    )

    # Create agent with sampling-enabled MCP client
    assistant = ZapAgent(
        name="SamplingAssistant",
        prompt="""You are a helpful assistant with access to AI-powered tools.

You can:
- Summarize text using summarize_text (uses LLM via sampling)
- Generate content using generate_content (uses LLM via sampling)
- Get the current time using get_current_time

Be concise and helpful in your responses.""",
        model="anthropic/claude-sonnet-4-5-20250929",
        mcp_clients=[tools_client],
        max_iterations=10,
    )

    # Create Zap instance with tracing
    zap = Zap(agents=[assistant], tracing_provider=tracing)

    # Connect to Temporal
    temporal_client = await TemporalClient.connect("localhost:7233")
    worker_task = None

    try:
        print("Starting Zap with MCP sampling support...")
        await zap.start()
        print(f"Registered agents: {zap.list_agents()}")

        worker = await create_worker(
            temporal_client,
            task_queue=zap.task_queue,
            tool_registry=zap._tool_registry,
            tracing_provider=tracing,
        )
        worker_task = asyncio.create_task(worker.run())
        print("Worker started")

        print("\nExecuting task...")
        task = await zap.execute_task(
            agent_name="SamplingAssistant",
            task="Summarize the following: 'AI has transformed healthcare and finance.'",
        )
        print(f"Task started: {task.id}")

        while not task.status.is_terminal():
            await asyncio.sleep(1)
            task = await zap.get_task(task.id)
            print(f"  Status: {task.status.value}")

        print("\n" + "=" * 50)
        if task.is_complete():
            print("Task completed!")
            print(f"\nResult:\n{task.result}")
        else:
            print(f"Task failed: {task.error}")

        # Flush traces before exit
        if tracing:
            print("\nFlushing traces to Langfuse...")
            await tracing.flush()
            print("Done! Check your Langfuse dashboard to see the trace.")

    finally:
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
