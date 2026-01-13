# Human-in-the-Loop Approval Workflows

For high-stakes AI operations—financial transactions, data deletions, external communications—you often need human oversight before an agent acts. Zap provides durable approval workflows that survive crashes, support timeouts, and integrate seamlessly with Temporal's fault-tolerant execution model.

## Why Approval Workflows?

Most agent frameworks either:
- Block a process waiting for input (won't survive restarts)
- Require external state management (complexity leak)
- Don't support approvals at all

Zap leverages Temporal's native signals and durable execution to provide approvals that:

- **Survive restarts** - Pending approvals persist across worker restarts
- **Support timeouts** - Auto-reject after configurable duration
- **Require no external infrastructure** - No HTTP callbacks or external databases
- **Integrate with observability** - All decisions are traced

## Quick Start

```python
from datetime import timedelta
from zap_ai import Zap, ZapAgent, ApprovalRules, TaskStatus

# Define an agent with financial tools
agent = ZapAgent(
    name="FinancialAgent",
    prompt="You are a financial assistant.",
    mcp_clients=[Client("./financial_tools.py")],
)

zap = Zap(agents=[agent])
await zap.start()

# Execute with approval rules
task = await zap.execute_task(
    agent_name="FinancialAgent",
    task="Transfer $50,000 to vendor account",
    approval_rules=ApprovalRules(
        patterns=["transfer_*", "delete_*"],
        timeout=timedelta(days=7),
    ),
)

# Poll for approval status
while task.status == TaskStatus.THINKING:
    await asyncio.sleep(1)
    task = await zap.get_task(task.id)

if task.status == TaskStatus.AWAITING_APPROVAL:
    # Check what needs approval
    pending = await task.get_pending_approvals()
    for req in pending:
        print(f"Tool: {req['tool_name']}")
        print(f"Args: {req['tool_args']}")
        print(f"Requested: {req['requested_at']}")

        # Review and decide
        if req['tool_args']['amount'] < 100000:
            await task.approve(req['id'])
        else:
            await task.reject(req['id'], reason="Amount exceeds limit")
```

## How It Works

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Agent Workflow                                 │
│                                                                          │
│  1. LLM returns tool call matching pattern                               │
│         │                                                                │
│         ▼                                                                │
│  2. Create ApprovalRequest                                               │
│     • Assign unique ID                                                   │
│     • Store in pending_approvals                                         │
│     • Set status = AWAITING_APPROVAL                                     │
│         │                                                                │
│         ▼                                                                │
│  3. workflow.wait_condition()  ◄───── Durable wait (survives restarts)   │
│     │                     │                                              │
│     │                     ▼                                              │
│     │            Timeout? ─────► Auto-reject, continue loop              │
│     │                                                                    │
│     ▼                                                                    │
│  4. Signal received (approve/reject)                                     │
│     │                     │                                              │
│     │ approved            │ rejected                                     │
│     ▼                     ▼                                              │
│  Execute tool        Return rejection message                            │
│     │                     │                                              │
│     └─────────────────────┴─────► Continue agentic loop                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `ApprovalRules` | Configuration with glob patterns and timeout |
| `ApprovalRequest` | Pending approval with ID, tool details, timestamps |
| `approve_execution` signal | Temporal signal to approve/reject |
| `get_pending_approvals` query | Query current pending approvals |

## API Reference

### ApprovalRules

Configuration for which tools require approval:

```python
from datetime import timedelta
from zap_ai import ApprovalRules

rules = ApprovalRules(
    patterns=["transfer_*", "delete_*", "send_email"],
    timeout=timedelta(hours=24),  # Default: 7 days
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patterns` | `list[str]` | required | Glob patterns matching tool names |
| `timeout` | `timedelta` | 7 days | Time before auto-rejection |

**Pattern matching:**
- `"transfer_*"` - matches `transfer_funds`, `transfer_stock`
- `"*_file"` - matches `delete_file`, `upload_file`
- `"send_email"` - exact match

### Task Methods

#### get_pending_approvals()

Query pending approval requests:

```python
pending = await task.get_pending_approvals()
# Returns: list[dict]
# [
#     {
#         "id": "uuid-...",
#         "tool_name": "transfer_funds",
#         "tool_args": {"amount": 50000, "to": "vendor"},
#         "requested_at": "2024-01-15T10:30:00Z",
#         "timeout_at": "2024-01-22T10:30:00Z",
#         "context": {"agent_name": "FinancialAgent", "workflow_id": "..."}
#     }
# ]
```

#### approve(approval_id)

Approve a pending tool execution:

```python
await task.approve(pending[0]['id'])
```

#### reject(approval_id, reason=None)

Reject a pending tool execution:

```python
await task.reject(pending[0]['id'], reason="Amount exceeds policy limit")
```

## Tool Discovery

Before setting approval rules, you can discover which tools an agent has access to:

```python
# Get available tools for an agent
tools = await zap.get_agent_tools("FinancialAgent")
# ['transfer_funds', 'check_balance', 'delete_transaction', 'send_email']

# Preview which patterns would match
rules = ApprovalRules(patterns=["transfer_*", "delete_*", "typo_tool"])
print(rules.preview_matches(tools))
# {
#     'transfer_*': ['transfer_funds'],
#     'delete_*': ['delete_transaction'],
#     'typo_tool': []  # No matches - potential typo!
# }

# Find patterns that don't match any tools
unmatched = rules.get_unmatched_patterns(tools)
# ['typo_tool']
```

If you execute a task with patterns that don't match any tools, Zap will emit a warning.

## Timeout Behavior

When an approval times out:

1. The pending approval is auto-rejected
2. The tool result contains: `[Tool call rejected: approval timeout after ...]`
3. The agentic loop continues (the agent sees the rejection and can adapt)

This "reject and continue" behavior ensures agents aren't permanently stuck waiting for human input.

## Multi-Agent Scenarios

Approval rules apply to the agent they're configured for:

```python
# Parent agent has approval rules for transfer_*
task = await zap.execute_task(
    agent_name="Coordinator",
    task="Get data from helper, then transfer funds",
    approval_rules=ApprovalRules(patterns=["transfer_*"]),
)
```

**Behavior:**
- `message_agent` (sub-agent delegation) does NOT trigger approval
- Sub-agent's tools execute normally
- Only parent's `transfer_*` tools require approval

To require approvals on sub-agent tools, you would need to:
1. Execute the sub-agent directly with its own approval rules, or
2. Implement custom approval logic in the sub-agent

## Best Practices

### Pattern Design

```python
# Good: Specific patterns for dangerous operations
ApprovalRules(patterns=[
    "transfer_*",      # All transfers
    "delete_*",        # All deletions
    "send_*",          # All sends (email, SMS, etc.)
    "execute_command", # Shell commands
])

# Bad: Too broad - will catch everything
ApprovalRules(patterns=["*"])
```

### Timeout Configuration

| Use Case | Recommended Timeout |
|----------|---------------------|
| Interactive demo | 5 minutes |
| Business hours approval | 24 hours |
| Async review queue | 7 days (default) |
| Batch processing | 1 hour |

### Approval Response Handling

Agents receive rejection reasons in tool results. Write prompts that handle rejections gracefully:

```python
agent = ZapAgent(
    name="FinancialAgent",
    prompt="""You are a financial assistant.

    If a tool call is rejected, inform the user about the rejection
    and ask if they'd like to try a different approach or modify
    the request parameters.""",
)
```

## Example: Approval Dashboard

Build a simple approval dashboard:

```python
async def approval_dashboard(zap: Zap, task_ids: list[str]):
    """Simple approval dashboard for pending tasks."""
    while True:
        for task_id in task_ids:
            task = await zap.get_task(task_id)

            if task.status != TaskStatus.AWAITING_APPROVAL:
                continue

            pending = await task.get_pending_approvals()
            for req in pending:
                print(f"\n=== Approval Request ===")
                print(f"Task: {task_id}")
                print(f"Tool: {req['tool_name']}")
                print(f"Args: {req['tool_args']}")
                print(f"Agent: {req['context']['agent_name']}")

                decision = input("Approve? (y/n/skip): ")
                if decision.lower() == 'y':
                    await task.approve(req['id'])
                    print("Approved!")
                elif decision.lower() == 'n':
                    reason = input("Rejection reason: ")
                    await task.reject(req['id'], reason=reason)
                    print("Rejected!")

        await asyncio.sleep(5)
```

## See Also

- [Multi-Agent Systems](multi-agent.md) - Sub-agent delegation
- [Observability](observability.md) - Tracing approval decisions
- [API Reference](../api/core.md) - Full Task API
