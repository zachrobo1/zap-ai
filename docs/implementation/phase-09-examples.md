# Phase 9: Examples & Documentation

**Goal:** Provide working examples and complete documentation.

**Dependencies:** All previous phases should be complete.

---

## Task 9.1: Create Basic Example

**File:** `examples/basic_agent.py`

Create a simple example with a single agent:

```python
"""
Basic Zap Agent Example

This example demonstrates:
- Creating a single agent
- Executing a task
- Polling for completion
"""

import asyncio
from zap_ai import Zap, ZapAgent, TaskStatus


async def main():
    # Create a simple agent
    agent = ZapAgent(
        name="AssistantAgent",
        prompt="""You are a helpful assistant. Answer questions clearly and concisely.

When asked to perform calculations, show your work step by step.""",
        model="gpt-4o-mini",  # Use a fast, cost-effective model
    )

    # Create Zap instance
    zap = Zap(agents=[agent])

    # Start Zap (connects to Temporal, initializes tools)
    print("Starting Zap...")
    await zap.start()
    print("Zap started successfully!")

    # Execute a task
    print("\nExecuting task...")
    task = await zap.execute_task(
        agent_name="AssistantAgent",
        task="What is 25 * 17? Show your work.",
    )
    print(f"Task started with ID: {task.id}")

    # Poll for completion
    while not task.is_complete():
        print(f"  Status: {task.status.value}")
        await asyncio.sleep(2)
        task = await zap.get_task(task.id)

    # Print result
    if task.is_successful():
        print(f"\nResult:\n{task.result}")
    else:
        print(f"\nError: {task.error}")

    # Follow-up question
    print("\nAsking follow-up question...")
    task = await zap.execute_task(
        follow_up_on_task=task.id,
        task="Now divide that result by 5.",
    )

    while not task.is_complete():
        await asyncio.sleep(2)
        task = await zap.get_task(task.id)

    if task.is_successful():
        print(f"\nFollow-up result:\n{task.result}")

    # Cleanup
    await zap.stop()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
```

**Checklist:**
- [ ] Create `examples/basic_agent.py`
- [ ] Single agent with no tools
- [ ] Execute task and poll for result
- [ ] Demonstrate follow-up question
- [ ] Include helpful comments

---

## Task 9.2: Create Multi-Agent Example

**File:** `examples/multi_agent.py`

Create an example with multiple agents demonstrating sub-agent messaging:

```python
"""
Multi-Agent Zap Example

This example demonstrates:
- Multiple specialized agents
- Sub-agent messaging
- Multi-turn conversations with sub-agents
"""

import asyncio
from zap_ai import Zap, ZapAgent, TaskStatus


async def main():
    # Create specialized agents
    research_agent = ZapAgent(
        name="ResearchAgent",
        prompt="""You are a research specialist. When asked to research a topic:
1. Gather relevant information
2. Organize findings clearly
3. Cite sources when possible

Be thorough but concise.""",
        model="gpt-4o",
        discovery_prompt="Use this agent to research topics and gather information.",
    )

    writer_agent = ZapAgent(
        name="WriterAgent",
        prompt="""You are a professional writer. When given research or notes:
1. Create well-structured content
2. Use clear, engaging language
3. Maintain consistent tone

Adapt your style to the requested format.""",
        model="gpt-4o",
        discovery_prompt="Use this agent to write and format content.",
    )

    # Main orchestrator agent with access to sub-agents
    main_agent = ZapAgent(
        name="OrchestratorAgent",
        prompt="""You are an orchestrator that coordinates research and writing tasks.

When given a complex task:
1. Break it down into subtasks
2. Delegate research to ResearchAgent
3. Delegate writing to WriterAgent
4. Synthesize the results

Use message_agent to collaborate with your sub-agents. You can have
multi-turn conversations with them using conversation_id.""",
        model="gpt-4o",
        sub_agents=["ResearchAgent", "WriterAgent"],
    )

    # Create Zap instance with all agents
    zap = Zap(agents=[main_agent, research_agent, writer_agent])

    print("Starting Zap with multi-agent configuration...")
    await zap.start()
    print(f"Agents: {zap.list_agents()}")

    # Execute a complex task
    print("\nExecuting complex task...")
    task = await zap.execute_task(
        agent_name="OrchestratorAgent",
        task="""Write a short blog post about the benefits of meditation.

First, research the topic to gather key facts and benefits.
Then, write a 3-paragraph blog post based on your research.""",
    )
    print(f"Task ID: {task.id}")

    # Poll with more detailed status
    while not task.is_complete():
        task = await zap.get_task(task.id)
        print(f"  Status: {task.status.value}")

        # Show sub-agent activity if delegated
        if task.status == TaskStatus.DELEGATED:
            print("    (Working with sub-agent...)")

        await asyncio.sleep(3)

    # Print result
    if task.is_successful():
        print(f"\n{'='*60}")
        print("FINAL RESULT:")
        print('='*60)
        print(task.result)
        print('='*60)

        # Show conversation history
        print(f"\nTotal messages in history: {len(task.history)}")
        print(f"Tool calls made: {task.get_tool_calls_count()}")
    else:
        print(f"\nError: {task.error}")

    await zap.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

**Checklist:**
- [ ] Create `examples/multi_agent.py`
- [ ] Multiple specialized agents
- [ ] Main orchestrator with sub_agents
- [ ] Demonstrate delegation via message_agent
- [ ] Show detailed status during execution

---

## Task 9.3: Create MCP Server Example

**File:** `examples/example_mcp_server.py`

Create a simple FastMCP server that can be used with agents:

```python
"""
Example MCP Server

This creates a simple MCP server with tools that can be used by Zap agents.

Run this server, then use the client in agent_with_tools.py to connect.

Usage:
    python example_mcp_server.py
"""

from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("Example Tools")


@mcp.tool()
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression like "2 + 2" or "10 * 5"

    Returns:
        The result of the calculation
    """
    try:
        # Simple eval for basic math (in production, use a proper parser)
        allowed_chars = set("0123456789+-*/.(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get the current weather for a city (mock data).

    Args:
        city: Name of the city

    Returns:
        Weather information
    """
    # Mock weather data
    weather_data = {
        "new york": "72°F, Partly Cloudy",
        "london": "58°F, Rainy",
        "tokyo": "68°F, Clear",
        "paris": "65°F, Overcast",
    }

    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    else:
        return f"Weather data not available for {city}"


@mcp.tool()
def search_knowledge_base(query: str, category: str = "general") -> str:
    """
    Search a mock knowledge base.

    Args:
        query: Search query
        category: Category to search in (general, tech, science)

    Returns:
        Search results
    """
    # Mock knowledge base
    kb = {
        "general": {
            "python": "Python is a high-level programming language known for readability.",
            "ai": "Artificial Intelligence is the simulation of human intelligence by machines.",
        },
        "tech": {
            "temporal": "Temporal is a workflow orchestration platform for reliable distributed systems.",
            "mcp": "Model Context Protocol enables tools for AI assistants.",
        },
        "science": {
            "physics": "Physics is the natural science studying matter, energy, and their interactions.",
            "biology": "Biology is the study of living organisms and their vital processes.",
        },
    }

    category_data = kb.get(category.lower(), kb["general"])
    query_lower = query.lower()

    for key, value in category_data.items():
        if key in query_lower:
            return f"Found: {value}"

    return f"No results found for '{query}' in category '{category}'"


if __name__ == "__main__":
    # Run the server
    mcp.run()
```

**Additional file:** `examples/agent_with_tools.py`

```python
"""
Agent with MCP Tools Example

This example shows how to connect an agent to an MCP server.

Prerequisites:
    1. Run the MCP server: python example_mcp_server.py
    2. Then run this script

Or use the server file path directly (FastMCP will start it automatically).
"""

import asyncio
from fastmcp import Client
from zap_ai import Zap, ZapAgent, TaskStatus


async def main():
    # Create MCP client pointing to the server file
    # FastMCP will automatically start the server as a subprocess
    tools_client = Client("./example_mcp_server.py")

    # Create an agent with tool access
    agent = ZapAgent(
        name="ToolAgent",
        prompt="""You are a helpful assistant with access to several tools:
- calculate: For mathematical calculations
- get_weather: To check weather in cities
- search_knowledge_base: To look up information

Use these tools to help answer user questions accurately.""",
        model="gpt-4o",
        mcp_clients=[tools_client],
    )

    # Create and start Zap
    zap = Zap(agents=[agent])
    await zap.start()

    print("Agent ready with tools!")
    print("Available tools will be discovered from the MCP server.\n")

    # Test the tools
    tasks = [
        "What is 123 * 456?",
        "What's the weather like in Tokyo?",
        "Tell me about Temporal workflow orchestration.",
    ]

    for task_text in tasks:
        print(f"Task: {task_text}")

        task = await zap.execute_task(
            agent_name="ToolAgent",
            task=task_text,
        )

        while not task.is_complete():
            await asyncio.sleep(1)
            task = await zap.get_task(task.id)

        if task.is_successful():
            print(f"Result: {task.result}\n")
        else:
            print(f"Error: {task.error}\n")

    await zap.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

**Checklist:**
- [ ] Create `examples/example_mcp_server.py`
- [ ] Create `examples/agent_with_tools.py`
- [ ] Implement sample tools (calculate, weather, search)
- [ ] Show how to connect agent to MCP server
- [ ] Include usage instructions in docstrings

---

## Task 9.4: Add Docstrings to All Public APIs

Ensure all public classes and functions have comprehensive docstrings:

**Checklist for each module:**

### Core Module (`src/zap_ai/core/`)
- [ ] `ZapAgent` class docstring with example
- [ ] `Task` class docstring with example
- [ ] `TaskStatus` enum with descriptions
- [ ] `Zap` class docstring with full usage example
- [ ] All exception classes documented

### MCP Module (`src/zap_ai/mcp/`)
- [ ] `ToolRegistry` class docstring with example
- [ ] `ClientManager` class docstring
- [ ] `mcp_tool_to_litellm()` with example
- [ ] `create_message_agent_tool()` with example
- [ ] All exception classes documented

### LLM Module (`src/zap_ai/llm/`)
- [ ] `Message` class with factory method examples
- [ ] `ToolCall` class with serialization examples
- [ ] `InferenceResult` class
- [ ] `complete()` function with example

### Workflows Module (`src/zap_ai/workflows/`)
- [ ] `AgentWorkflow` class with high-level overview
- [ ] All dataclasses documented
- [ ] Signals and queries documented

### Activities Module (`src/zap_ai/activities/`)
- [ ] `inference_activity` with retry policy info
- [ ] `tool_execution_activity` with retry policy info
- [ ] Input/Output dataclasses documented

### Worker Module (`src/zap_ai/worker/`)
- [ ] `run_worker()` with usage example
- [ ] `create_worker()` documented
- [ ] CLI usage documented

---

## Phase 9 Verification

After completing all tasks, verify:

1. **Examples run successfully:**
   ```bash
   # Start Temporal first
   temporal server start-dev

   # In another terminal, start worker
   python -m zap_ai.worker

   # In another terminal, run examples
   python examples/basic_agent.py
   python examples/multi_agent.py
   python examples/agent_with_tools.py
   ```

2. **Documentation renders correctly:**
   ```bash
   # Generate docs (if using sphinx or similar)
   cd docs && make html

   # Or check docstrings with pydoc
   python -m pydoc zap_ai.Zap
   ```

3. **Examples are self-contained:**
   - Each example can be run independently
   - Examples include all necessary imports
   - Prerequisites are documented

4. **API documentation is complete:**
   - All public classes have docstrings
   - Examples are included where helpful
   - Parameter types are documented
