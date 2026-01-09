# Multi-Agent Systems

Zap supports building multi-agent systems where agents can delegate tasks to other specialized agents. This enables complex workflows with separation of concerns.

## Defining Sub-Agents

Use the `sub_agents` parameter to specify which agents an agent can delegate to:

```python
from zap_ai import Zap, ZapAgent
from fastmcp import Client

# Specialized agent for database queries
db_agent = ZapAgent(
    name="DatabaseAgent",
    prompt="You are a database specialist. Query the database to find information.",
    model="gpt-4o-mini",
    mcp_clients=[Client("./db_tools.py")],
    discovery_prompt="Use for database queries and data lookups",  # Shown to parent agents
)

# Specialized agent for writing
writer_agent = ZapAgent(
    name="WriterAgent",
    prompt="You are a technical writer. Create clear, well-structured content.",
    model="gpt-4o",
    discovery_prompt="Use for writing reports, summaries, and documentation",
)

# Main orchestrator agent
main_agent = ZapAgent(
    name="Coordinator",
    prompt="You coordinate research tasks. Delegate to specialists as needed.",
    model="gpt-4o",
    sub_agents=["DatabaseAgent", "WriterAgent"],  # Can delegate to these
)

zap = Zap(agents=[main_agent, db_agent, writer_agent])
```

## How Delegation Works

When an agent has sub-agents configured, Zap automatically adds a `message_agent` tool. The parent agent can call this tool to delegate tasks:

```
User: "Create a report on Q4 sales"

Coordinator thinks: I need to get data first, then write a report.

Coordinator calls: message_agent(
    agent_name="DatabaseAgent",
    task="Get Q4 sales figures by region"
)

DatabaseAgent runs, returns results...

Coordinator calls: message_agent(
    agent_name="WriterAgent",
    task="Write a report on these Q4 sales figures: [data]"
)

WriterAgent runs, returns formatted report...

Coordinator returns: "Here's your Q4 sales report: [report]"
```

## Discovery Prompts

The `discovery_prompt` tells parent agents when to use each sub-agent:

```python
analyst = ZapAgent(
    name="DataAnalyst",
    prompt="You analyze data and find patterns.",
    discovery_prompt="Use for statistical analysis, trend detection, and data insights",
)
```

This description appears in the `message_agent` tool, helping the parent agent make informed delegation decisions.

## Accessing Sub-Task Results

After a task completes, you can inspect sub-task details:

```python
task = await zap.get_task(task_id)

# Get sub-task IDs
print(f"Sub-tasks: {task.sub_tasks}")

# Fetch full sub-task details
sub_tasks = await task.get_sub_tasks()
for sub in sub_tasks:
    print(f"{sub.agent_name}: {sub.status} - {sub.result}")
```

## Architecture

Sub-agent delegation uses Temporal child workflows:

```
┌─────────────────────────────────────────┐
│         Parent Workflow                 │
│  ┌─────────────────────────────────┐    │
│  │  message_agent("DatabaseAgent") │────┼──► Child Workflow (DatabaseAgent)
│  └─────────────────────────────────┘    │         │
│              ▲                          │         │
│              └──────────────────────────┼─────────┘ (result returned)
│                                         │
│  ┌─────────────────────────────────┐    │
│  │  message_agent("WriterAgent")   │────┼──► Child Workflow (WriterAgent)
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

Benefits:

- **Isolation** - Each agent runs in its own workflow with separate state
- **Fault tolerance** - Sub-agent failures don't crash the parent
- **Observability** - Each sub-task is tracked independently

## Best Practices

1. **Clear responsibilities** - Each agent should have a focused purpose
2. **Descriptive discovery prompts** - Help the coordinator make good delegation decisions
3. **Appropriate models** - Use smaller/faster models for simple tasks
4. **Limit nesting depth** - Avoid deeply nested delegation chains

## Example: Research Pipeline

```python
# Search agent - finds information
searcher = ZapAgent(
    name="Searcher",
    prompt="Search for information on the given topic.",
    mcp_clients=[Client("./search_tools.py")],
    discovery_prompt="Use to search and gather information from external sources",
)

# Analyzer - processes and synthesizes
analyzer = ZapAgent(
    name="Analyzer",
    prompt="Analyze information and extract key insights.",
    discovery_prompt="Use to analyze data and identify patterns or conclusions",
)

# Writer - creates final output
writer = ZapAgent(
    name="Writer",
    prompt="Write clear, well-structured reports.",
    discovery_prompt="Use to write final reports and summaries",
)

# Coordinator - orchestrates the pipeline
coordinator = ZapAgent(
    name="ResearchCoordinator",
    prompt="""You coordinate research projects. For each request:
1. Use Searcher to gather relevant information
2. Use Analyzer to process and analyze the data
3. Use Writer to create the final deliverable""",
    sub_agents=["Searcher", "Analyzer", "Writer"],
)

zap = Zap(agents=[coordinator, searcher, analyzer, writer])
```
