"""MCP tools server demonstrating sampling support.

This server provides tools that use MCP sampling to request
LLM completions from the client.

Run as standalone: python tools.py
Test with FastMCP: fastmcp dev tools.py
"""

import datetime

from fastmcp import Context, FastMCP
from mcp.types import SamplingMessage, TextContent

mcp = FastMCP("Sampling Tools")


@mcp.tool()
async def summarize_text(text: str, ctx: Context) -> str:
    """Summarize the given text using LLM sampling.

    Args:
        text: The text to summarize.
        ctx: FastMCP context (injected automatically).

    Returns:
        A concise summary of the input text.
    """
    result = await ctx.sample(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(type="text", text=f"Summarize in 1-2 sentences:\n\n{text}"),
            )
        ],
        system_prompt="You create concise summaries.",
    )
    return f"Summary: {result}"


@mcp.tool()
async def generate_content(
    topic: str,
    style: str = "informative",
    ctx: Context = None,
) -> str:
    """Generate content about a topic using LLM sampling.

    Args:
        topic: The topic to generate content about.
        style: Writing style (informative, casual, formal).
        ctx: FastMCP context (injected automatically).

    Returns:
        Generated content about the topic.
    """
    result = await ctx.sample(
        messages=[
            SamplingMessage(
                role="user",
                content=TextContent(
                    type="text", text=f"Write a short {style} paragraph about: {topic}"
                ),
            )
        ],
        system_prompt=f"You are a content writer. Write in a {style} style.",
    )
    return result


@mcp.tool()
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current date and time (no sampling needed).

    Args:
        timezone: Timezone name (currently only UTC supported).

    Returns:
        Current datetime as a formatted string.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    return f"Current time ({timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


if __name__ == "__main__":
    print("Starting MCP server with sampling support...")
    print("Tools: summarize_text, generate_content, get_current_time")
    print("Press Ctrl+C to stop")
    mcp.run()
