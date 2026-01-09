# Conversation API

Types and utilities for working with conversation history.

## ToolCallInfo

Information about a tool call in the conversation.

::: zap_ai.ToolCallInfo
    options:
      show_source: true

## ConversationTurn

A single turn in the conversation.

::: zap_ai.ConversationTurn
    options:
      show_source: true

## Usage Example

```python
from zap_ai import Zap, ToolCallInfo, ConversationTurn

# After task completion
task = await zap.get_task(task_id)

# Get all tool calls
tool_calls: list[ToolCallInfo] = task.get_tool_calls()
for tc in tool_calls:
    print(f"Called {tc.name} with {tc.arguments}")
    print(f"Result: {tc.result}")

# Navigate by turns
turns: list[ConversationTurn] = task.get_turns()
for turn in turns:
    print(f"Turn {turn.turn_number}:")
    if turn.user_message:
        print(f"  User: {turn.user_message.get('content')}")
    for msg in turn.assistant_messages:
        if msg.get("content"):
            print(f"  Assistant: {msg['content']}")
```
