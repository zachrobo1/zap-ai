"""Example: Accessing Conversation History

This example demonstrates the conversation history inspection API:
- get_text_content() - Extract all text from conversation
- get_tool_calls() - Get all tool calls with their results
- get_turns() / get_turn(n) - Access conversation by turns
- turn_count() - Get number of turns
- get_sub_tasks() - Fetch sub-task Task objects (for multi-agent scenarios)

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

    # Reuse tools from simple_agent example
    tools_path = Path(__file__).parent.parent / "simple_agent" / "tools.py"

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

    # Connect to Temporal
    temporal_client = await TemporalClient.connect("localhost:7233")
    worker_task = None

    try:
        # Start Zap
        print("Starting Zap...")
        await zap.start()

        # Create worker with the tool registry from Zap
        worker = await create_worker(
            temporal_client,
            task_queue=zap.task_queue,
            tool_registry=zap._tool_registry,
        )

        # Start worker in background
        worker_task = asyncio.create_task(worker.run())
        print("Worker started")

        # Execute a task that will use tools
        print("\nExecuting task...")
        task = await zap.execute_task(
            agent_name="Assistant",
            task="What time is it? Also, calculate 15 multiplied by 7 for me.",
        )
        print(f"Task started: {task.id}")

        # Poll for completion
        print("Waiting for completion...")
        while not task.status.is_terminal():
            await asyncio.sleep(1)
            task = await zap.get_task(task.id)
            print(f"  Status: {task.status.value}")

        if not task.is_complete():
            print(f"\nTask failed: {task.error}")
            return

        # ============================================================
        # CONVERSATION HISTORY DEMO
        # ============================================================

        print("\n" + "=" * 60)
        print("CONVERSATION HISTORY DEMO")
        print("=" * 60)

        # 1. get_text_content() - Extract all text from conversation
        print("\n--- get_text_content() ---")
        text_content = task.get_text_content()
        print(text_content)

        # 2. get_tool_calls() - Get all tool calls with their results
        print("\n--- get_tool_calls() ---")
        tool_calls = task.get_tool_calls()
        if tool_calls:
            for tc in tool_calls:
                print(f"  Tool: {tc.name}")
                print(f"    ID: {tc.id}")
                print(f"    Arguments: {tc.arguments}")
                result_preview = (
                    tc.result[:100] + "..." if tc.result and len(tc.result) > 100 else tc.result
                )
                print(f"    Result: {result_preview}")
                print()
        else:
            print("  No tool calls in this conversation")

        # 3. get_turns() - Access conversation by turns
        print("\n--- get_turns() ---")
        turns = task.get_turns()
        for turn in turns:
            print(f"\nTurn {turn.turn_number}:")
            if turn.user_message:
                content = turn.user_message.get("content", "")
                preview = content[:80] + "..." if len(content) > 80 else content
                print(f"  User: {preview}")
            for msg in turn.assistant_messages:
                content = msg.get("content")
                if content:
                    preview = content[:80] + "..." if len(content) > 80 else content
                    print(f"  Assistant: {preview}")
            if turn.tool_messages:
                print(f"  Tool responses: {len(turn.tool_messages)}")

        # 4. get_turn(n) - Get a specific turn
        print("\n--- get_turn(0) ---")
        first_turn = task.get_turn(0)
        if first_turn:
            print(f"First turn exists: turn_number={first_turn.turn_number}")
            print(f"  Has user message: {first_turn.user_message is not None}")
            print(f"  Assistant messages: {len(first_turn.assistant_messages)}")
            print(f"  Tool messages: {len(first_turn.tool_messages)}")

        # 5. turn_count() - Get number of turns
        print("\n--- turn_count() ---")
        print(f"Total conversation turns: {task.turn_count()}")

        # 6. sub_tasks and get_sub_tasks() - For multi-agent scenarios
        print("\n--- sub_tasks ---")
        print(f"Sub-task IDs: {task.sub_tasks}")
        sub_tasks = await task.get_sub_tasks()
        print(f"Sub-task Task objects: {sub_tasks}")
        print("(Empty in single-agent scenario - see multi_agent example for sub-tasks)")

        # Final result
        print("\n" + "=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(task.result)

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
    asyncio.run(main())
