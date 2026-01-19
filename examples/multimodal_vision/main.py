"""Multimodal vision example for Zap.

This example demonstrates:
- Sending images to vision-capable agents
- Using TextContent and ImageContent for multimodal tasks
- Vision support validation (model capability checking)

Prerequisites:
1. Copy .env.example to .env and set ANTHROPIC_API_KEY
2. Start Temporal server: temporal server start-dev
3. Run this script: python main.py

Note: Requires a vision-capable model like Claude 3+ models.
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from temporalio.client import Client as TemporalClient

from zap_ai import ImageContent, TextContent, Zap, ZapAgent
from zap_ai.worker import create_worker

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Sample image URL (Wikipedia cat photo)
SAMPLE_IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
)


async def main() -> None:
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Set it in .env or export ANTHROPIC_API_KEY=your-key")
        return

    # Create a vision-capable agent
    vision_agent = ZapAgent(
        name="VisionAgent",
        prompt="""You are a helpful assistant that can analyze images.
When given an image, describe what you see clearly and concisely.
Focus on the main subjects and notable details.""",
        model="anthropic/claude-sonnet-4-5-20250929",  # Vision-capable
        max_iterations=5,
    )

    # Create Zap instance
    zap = Zap(agents=[vision_agent])

    # Connect to Temporal and create worker
    temporal_client = await TemporalClient.connect("localhost:7233")
    worker_task = None

    try:
        # Start Zap
        print("Starting Zap...")
        await zap.start()
        print(f"Registered agents: {zap.list_agents()}")

        # Create worker
        worker = await create_worker(
            temporal_client,
            task_queue=zap.task_queue,
            tool_registry=zap.tool_registry,
        )

        # Start worker in background
        worker_task = asyncio.create_task(worker.run())
        print("Worker started in background")

        # --- Test 1: Image analysis with multimodal content ---
        print("\n" + "=" * 50)
        print("Test 1: Analyzing image URL")
        print("=" * 50)

        task = await zap.execute_task(
            agent_name="VisionAgent",
            task_content=[
                TextContent(text="What do you see in this image? Be brief."),
                ImageContent.from_url(SAMPLE_IMAGE_URL),
            ],
        )
        print(f"Task started: {task.id}")

        # Poll for completion
        while not task.status.is_terminal():
            await asyncio.sleep(1)
            task = await zap.get_task(task.id)
            print(f"  Status: {task.status.value}")

        if task.is_complete():
            print(f"\nResult:\n{task.result}")
        else:
            print(f"\nError: {task.error}")

        # --- Test 2: Using Message.user_with_images helper ---
        print("\n" + "=" * 50)
        print("Test 2: Using user_with_images helper")
        print("=" * 50)

        # Alternative: use helper to create multimodal content
        from zap_ai.llm.message_types import Message

        msg = Message.user_with_images(
            "Describe the animal in this photo:",
            [ImageContent.from_url(SAMPLE_IMAGE_URL)],
        )

        # Pass the content directly (user_with_images always creates non-None content)
        assert msg.content is not None, "user_with_images always creates content"
        task = await zap.execute_task(
            agent_name="VisionAgent",
            task_content=msg.content,
        )
        print(f"Task started: {task.id}")

        while not task.status.is_terminal():
            await asyncio.sleep(1)
            task = await zap.get_task(task.id)

        if task.is_complete():
            print(f"\nResult:\n{task.result}")
        else:
            print(f"\nError: {task.error}")

    finally:
        # Clean shutdown
        print("\n" + "=" * 50)
        print("Shutting down...")
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
