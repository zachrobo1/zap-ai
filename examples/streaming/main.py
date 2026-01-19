"""Streaming events demo for Zap.

This example demonstrates real-time event streaming with real MCP servers
that perform actual work. It connects to mcp-server-fetch for web fetching
and uses local tools for text analysis.

Prerequisites:
1. Copy .env.example to .env and set ANTHROPIC_API_KEY
2. Start Temporal server: temporal server start-dev
3. Run this script: python examples/streaming/main.py

Note: mcp-server-fetch will be auto-installed via uvx on first run.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import UvxStdioTransport
from temporalio.client import Client as TemporalClient

from zap_ai import Zap, ZapAgent
from zap_ai.streaming.events import (
    CompletedEvent,
    ErrorEvent,
    ThinkingEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from zap_ai.worker import create_worker

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")


class StreamingDisplay:
    """Helper class for formatting streaming events with timestamps."""

    def __init__(self) -> None:
        self.start_time = time.time()
        self.tool_start_times: dict[str, float] = {}
        self.event_count = 0
        self.tool_call_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_iteration = 0

    def elapsed(self) -> str:
        """Format elapsed time as [MM:SS.s]."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"[{minutes:02d}:{seconds:04.1f}]"

    def handle_event(
        self,
        event: ThinkingEvent | ToolCallEvent | ToolResultEvent | CompletedEvent | ErrorEvent,
    ) -> None:
        """Handle and display a streaming event."""
        self.event_count += 1

        match event:
            case ThinkingEvent(iteration=iteration):
                self.last_iteration = iteration
                print(f"\n{self.elapsed()} \U0001f4ad Thinking (iteration {iteration})")

            case ToolCallEvent(name=name, phrase=phrase, tool_call_id=tool_call_id):
                self.tool_call_count += 1
                self.tool_start_times[tool_call_id] = time.time()
                print(f"\n{self.elapsed()} \U0001f527 {name}")
                print(f"          {phrase}")

            case ToolResultEvent(name=name, tool_call_id=tool_call_id, success=success):
                start = self.tool_start_times.pop(tool_call_id, time.time())
                duration = time.time() - start
                if success:
                    self.success_count += 1
                    print(f"\n{self.elapsed()} \u2705 {name} completed ({duration:.1f}s)")
                else:
                    self.failure_count += 1
                    print(f"\n{self.elapsed()} \u274c {name} failed ({duration:.1f}s)")

            case CompletedEvent(result=result):
                print(f"\n{self.elapsed()} \U0001f389 Task Completed!")
                print("\nResult:")
                print("\u2500" * 40)
                print(result)

            case ErrorEvent(error=error):
                print(f"\n{self.elapsed()} \u274c Task Failed!")
                print(f"\nError: {error}")

    def print_summary(self) -> None:
        """Print final summary statistics."""
        total_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print("\U0001f4ca SUMMARY")
        print("=" * 80)
        print(f"Total time:      {total_time:.1f}s")
        print(f"Iterations:      {self.last_iteration}")
        print(
            f"Tool calls:      {self.tool_call_count} "
            f"({self.success_count} succeeded, {self.failure_count} failed)"
        )
        print(f"Events received: {self.event_count}")


async def main() -> None:
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Set it in .env or export ANTHROPIC_API_KEY=your-key")
        return

    # Print header
    print("=" * 80)
    print("\U0001f534 STREAMING EVENTS DEMO")
    print("=" * 80)

    # Path to local tools
    tools_path = Path(__file__).parent / "tools.py"

    # Connect to mcp-server-fetch via uvx for real web fetching
    # This provides real network latency, making streaming natural
    fetch_transport = UvxStdioTransport("mcp-server-fetch")
    fetch_client = Client(fetch_transport)

    # Local tools for text analysis
    tools_client = Client(str(tools_path))

    # Create researcher agent with both MCP servers
    researcher = ZapAgent(
        name="Researcher",
        prompt="""You are a research assistant that can fetch web pages and analyze content.

You have access to:
- fetch: Fetch and read web pages (returns markdown)
- analyze_text: Analyze text for statistics (word count, reading level, etc.)
- summarize_bullet_points: Extract key points from text
- calculate: Perform mathematical calculations

When researching a topic:
1. Fetch relevant web pages
2. Analyze the content for key statistics
3. Summarize the main points
4. Provide a concise research summary""",
        model="anthropic/claude-sonnet-4-5-20250929",
        mcp_clients=[fetch_client, tools_client],
        max_iterations=10,
    )

    # Create Zap instance
    zap = Zap(agents=[researcher])

    # Connect to Temporal
    temporal_client = await TemporalClient.connect("localhost:7233")
    worker_task = None

    # Configure streaming behavior:
    # - poll_interval: 0.1 (default) - how often to poll for events
    # - include_thinking: True (default) - include ThinkingEvent
    # - include_tool_events: True (default) - include ToolCallEvent/ToolResultEvent

    # Research task that triggers multiple tool calls
    task = """
Research the Python programming language:
1. Fetch the Wikipedia page for Python (https://en.wikipedia.org/wiki/Python_(programming_language))
2. Analyze the text for key statistics
3. Summarize the main points

Provide a concise research summary.
"""

    print(f"\nTask: {task.strip()}")
    print("-" * 80)

    display = StreamingDisplay()

    try:
        # Start Zap
        print("\nStarting Zap...")
        await zap.start()

        # Create and start worker
        worker = await create_worker(
            temporal_client,
            task_queue=zap.task_queue,
            tool_registry=zap._tool_registry,
        )
        worker_task = asyncio.create_task(worker.run())

        # Stream events
        async for event in zap.stream_task(
            agent_name="Researcher",
            task=task,
            poll_interval=0.1,
            include_thinking=True,
            include_tool_events=True,
        ):
            display.handle_event(event)

            # Terminal events end the stream
            if isinstance(event, (CompletedEvent, ErrorEvent)):
                break

        # Print summary
        display.print_summary()

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise
    finally:
        # Clean shutdown
        print("\nShutting down...")
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
        await zap.stop()
        print("Done!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
