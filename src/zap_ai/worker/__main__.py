"""CLI entry point for running Zap worker.

Usage:
    python -m zap_ai.worker
    python -m zap_ai.worker --temporal-address localhost:7233
    python -m zap_ai.worker --task-queue my-agents
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from zap_ai.worker import run_worker

# Load .env from current directory or parent directories
load_dotenv()
# Also try project root if running from subdirectory
load_dotenv(Path.cwd() / ".env")


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
