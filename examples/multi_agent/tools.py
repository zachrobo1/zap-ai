"""Example MCP tools server for Zap agents.

This module provides simple tools that agents can use:
- get_current_time: Get the current date and time
- calculate: Perform basic arithmetic
- search_web: Simulate a web search (returns mock results)

Run directly to test: python tools.py
"""

import datetime
from typing import Literal

from fastmcp import FastMCP

mcp = FastMCP("Example Tools")


@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time.

    Args:
        timezone: Timezone name (currently only UTC supported).

    Returns:
        Current datetime as a formatted string.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


@mcp.tool()
def calculate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    a: float,
    b: float,
) -> str:
    """Perform a basic arithmetic calculation.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide).
        a: First number.
        b: Second number.

    Returns:
        Result of the calculation as a string.
    """
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        result = a / b
    else:
        return f"Error: Unknown operation '{operation}'"

    return f"{a} {operation} {b} = {result}"


@mcp.tool()
def search_web(query: str, num_results: int = 3) -> str:
    """Search the web for information (simulated).

    In a real implementation, this would call a search API.
    Currently returns mock results for demonstration.

    Args:
        query: The search query.
        num_results: Number of results to return (1-5).

    Returns:
        Search results as formatted text.
    """
    # Mock results - in production, call a real search API
    mock_results = [
        {
            "title": f"Result 1 for '{query}'",
            "snippet": f"This is a relevant result about {query}...",
            "url": "https://example.com/result1",
        },
        {
            "title": f"Wikipedia: {query}",
            "snippet": f"A comprehensive article about {query}...",
            "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
        },
        {
            "title": f"{query} - Latest News",
            "snippet": f"Breaking news and updates about {query}...",
            "url": "https://news.example.com/article",
        },
    ]

    results = mock_results[: min(num_results, len(mock_results))]
    output = f"Search results for '{query}':\n\n"

    for i, r in enumerate(results, 1):
        output += f"{i}. {r['title']}\n"
        output += f"   {r['snippet']}\n"
        output += f"   URL: {r['url']}\n\n"

    return output


if __name__ == "__main__":
    # Run as MCP server for testing
    # Use: python tools.py
    # Or test with: fastmcp dev tools.py
    print("Starting MCP server...")
    print("Tools available: get_current_time, calculate, search_web")
    print("Press Ctrl+C to stop")
    mcp.run()
