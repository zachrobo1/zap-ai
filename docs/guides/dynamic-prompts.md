# Dynamic Prompts

Zap supports dynamic prompts that receive context at execution time. This is useful for personalizing agent behavior based on user information, session data, or other runtime values.

## Basic Usage

Instead of a static string, pass a callable to the `prompt` parameter:

```python
from zap_ai import Zap, ZapAgent

# Agent with dynamic prompt
agent = ZapAgent(
    name="Assistant",
    prompt=lambda ctx: f"You are {ctx['user_name']}'s personal assistant.",
    model="gpt-4o",
)

zap = Zap(agents=[agent])
await zap.start()

# Pass context when executing
task = await zap.execute_task(
    agent_name="Assistant",
    task="Help me draft an email.",
    context={"user_name": "Alice"},
)
```

## Typed Context

For type safety, use a dataclass or Pydantic model and specify the generic type:

```python
from dataclasses import dataclass
from zap_ai import Zap, ZapAgent

@dataclass
class UserContext:
    user_name: str
    company: str
    role: str

# Specify the context type for better IDE support
agent = ZapAgent[UserContext](
    name="WorkAssistant",
    prompt=lambda ctx: f"""You are an assistant for {ctx.user_name},
a {ctx.role} at {ctx.company}. Be professional and helpful.""",
    model="gpt-4o",
)

zap = Zap(agents=[agent])
await zap.start()

task = await zap.execute_task(
    agent_name="WorkAssistant",
    task="Summarize our Q4 goals.",
    context=UserContext(
        user_name="Alice",
        company="Acme Corp",
        role="Product Manager",
    ),
)
```

## Multiple Agents with Shared Context

When using multiple agents, they can all receive the same context type:

```python
@dataclass
class SessionContext:
    user_id: str
    language: str
    timezone: str

researcher = ZapAgent[SessionContext](
    name="Researcher",
    prompt=lambda ctx: f"Research assistant. Respond in {ctx.language}.",
    model="gpt-4o",
)

writer = ZapAgent[SessionContext](
    name="Writer",
    prompt=lambda ctx: f"Technical writer. Use {ctx.timezone} for timestamps.",
    model="gpt-4o",
    discovery_prompt="Use for writing and formatting content",
)

zap = Zap(agents=[researcher, writer])
```

## Default Context

If no context is provided, an empty dict `{}` is used. Agents with dynamic prompts will receive a warning if called without context:

```python
# This will work but emit a warning
task = await zap.execute_task(
    agent_name="Assistant",
    task="Hello",
    # context not provided - warning emitted
)
```

To avoid warnings, always provide context or use a static prompt for agents that don't need runtime customization.

## Checking for Dynamic Prompts

You can check if an agent uses a dynamic prompt:

```python
agent = ZapAgent(
    name="Test",
    prompt=lambda ctx: f"Hello {ctx['name']}",
)

if agent.is_dynamic_prompt():
    print("This agent requires context")
```

## API Reference

See the full API documentation:

- [`ZapAgent`](../api/core.md#zap_ai.ZapAgent) - Agent configuration
- [`Zap.execute_task`](../api/core.md#zap_ai.Zap.execute_task) - Execute with context
