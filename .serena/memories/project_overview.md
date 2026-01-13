# Zap - Zach's Agent Platform ⚡️

## Purpose
Zap is an opinionated library for building **resilient AI agents** on top of Temporal. It provides a scalable, fault-tolerant way to create AI agents that can power demanding use cases and complex architectures.

**Why Zap?** LLM providers can't yet guarantee production-level SLAs. API calls fail, rate limits hit, and connections drop. Zap solves this by running agents on Temporal—a fault-tolerant code execution platform that captures state and retries failed steps automatically.

## Key Features
- **Automatic retries** with configurable policies for LLM and tool calls
- **State persistence** - agents survive crashes and can resume mid-conversation
- **Sub-agent delegation** - compose complex systems from specialized agents
- **Human-in-the-loop approvals** - require human oversight for high-stakes tool calls
- **MCP integration** - easily add tools via the Model Context Protocol
- **Provider agnostic** - use any LLM supported by LiteLLM (OpenAI, Anthropic, etc.)
- **Observability** - built-in tracing support with Langfuse integration (extensible via BaseTracingProvider ABC)
- **Dynamic prompts** - context-aware prompts resolved at runtime
- **Conversation history API** - rich access to turns, tool calls, and text content

## Tech Stack
- **Python 3.11+** - Core language
- **Temporal** - Fault-tolerant workflow orchestration
- **LiteLLM** - Unified LLM provider interface (supports OpenAI, Anthropic, etc.)
- **FastMCP** - Model Context Protocol client for tool integration
- **Pydantic** - Data validation and settings management

### Optional Dependencies
- **Langfuse** - Tracing/observability (install with `pip install zap-ai[langfuse]`)

### Development Dependencies
- pytest, pytest-asyncio, pytest-mock, pytest-cov (testing)
- ruff (linting & formatting)
- pre-commit (git hooks)
- mkdocs, mkdocs-material, mkdocstrings (documentation)

## Package Structure
```
src/zap_ai/
├── __init__.py         # Public API exports (Zap, ZapAgent, Task, TaskStatus, ApprovalRules)
├── exceptions.py       # All custom exceptions (consolidated, all inherit from ZapError)
├── utils.py            # Shared utility functions (e.g., parse_tool_arguments)
├── core/               # Core models (Zap, ZapAgent, Task)
│   ├── __init__.py     # Re-exports core types
│   ├── agent.py        # ZapAgent configuration model
│   ├── task.py         # Task execution tracking model
│   ├── types.py        # Type aliases (TContext, DynamicPrompt)
│   ├── validation.py   # Extracted validation functions
│   └── zap.py          # Main Zap orchestrator (start, execute_task, get_task, get_agent_tools)
├── conversation/       # Conversation history parsing
│   ├── __init__.py     # Exports ConversationTurn, ToolCallInfo
│   ├── models.py       # ConversationTurn, ToolCallInfo dataclasses
│   └── parser.py       # History parsing functions (get_turns, get_tool_calls, etc.)
├── workflows/          # Temporal workflow definitions
│   ├── __init__.py
│   ├── agent_workflow.py # Main agentic loop workflow (includes approval handling)
│   └── models.py       # Workflow input/state models (ApprovalRules, ApprovalRequest, ApprovalDecision)
├── activities/         # Temporal activities (inference, tool execution)
│   ├── __init__.py
│   ├── inference.py    # LLM inference activity (via LiteLLM)
│   ├── tool_execution.py # MCP tool execution activity
│   └── agent_config.py # Agent configuration activity
├── llm/                # LLM provider abstraction
│   ├── __init__.py
│   ├── provider.py     # LiteLLM wrapper
│   └── message_types.py # Message, ToolCall, InferenceResult
├── mcp/                # MCP client management and tool registry
│   ├── __init__.py
│   ├── client_manager.py # FastMCP client lifecycle
│   ├── schema_converter.py # MCP to LiteLLM schema conversion
│   └── tool_registry.py # Tool discovery and caching (get_tools_for_agent)
├── tracing/            # Observability providers
│   ├── __init__.py     # Global provider registry (set_tracing_provider, get_tracing_provider)
│   ├── protocol.py     # TracingProvider protocol, TraceContext
│   ├── base.py         # BaseTracingProvider ABC for custom providers
│   ├── noop_provider.py # No-op fallback
│   └── langfuse_provider.py # Langfuse implementation
└── worker/             # Worker process for running workflows
    ├── __init__.py
    ├── worker.py       # Worker creation functions (run_worker)
    └── __main__.py     # CLI entry point (python -m zap_ai.worker)
```

## Main Classes

### Core
- `Zap` - Main orchestrator that manages agents and Temporal connections
- `ZapAgent[TContext]` - Configuration for an AI agent (name, prompt, model, tools, sub-agents)
- `Task` - Represents an executing or completed task with rich inspection methods

### Task Inspection
- `TaskStatus` - Enum: PENDING, THINKING, AWAITING_TOOL, AWAITING_APPROVAL, COMPLETED, FAILED
- `ToolCallInfo` - Information about a tool call and its result
- `ConversationTurn` - A single turn in the conversation

### Approval Workflows
- `ApprovalRules` - Glob patterns for tools requiring approval, with timeout
- `ApprovalRequest` - Pending approval request with tool name, args, timestamps
- `ApprovalDecision` - Approval/rejection decision with reason

### Tracing
- `TracingProvider` - Protocol for tracing implementations
- `BaseTracingProvider` - ABC for custom provider implementation
- `LangfuseTracingProvider` - Langfuse integration
- `NoOpTracingProvider` - Fallback (no-op)

## Exception Hierarchy
All exceptions inherit from `ZapError`:
- `ZapConfigurationError` - Invalid Zap configuration
- `ZapNotStartedError` - Operations before start()
- `AgentNotFoundError` - Unknown agent reference
- `TaskNotFoundError` - Unknown task reference
- `ToolNotFoundError` - Tool not found
- `ToolExecutionError` - Tool execution failure
- `ClientConnectionError` - MCP client connection failure
- `SchemaConversionError` - Schema conversion failure
- `LLMProviderError` - LLM provider failure

## Architecture Flow
1. User creates `ZapAgent` configurations and a `Zap` instance
2. `zap.start()` connects to Temporal and initializes MCP clients
3. `zap.execute_task()` starts a Temporal workflow for the agent
4. The workflow runs an agentic loop:
   - LLM inference → check approval rules → tool execution → repeat
5. Approval pauses: if tool matches `ApprovalRules`, workflow waits for signal
6. Sub-agents are executed as child workflows (via `message_agent` tool)
7. State is persisted in Temporal, surviving crashes
8. `continue-as-new` prevents event history from growing unbounded

## Task Methods (on Task object)
- `get_text_content()` - All text from user + assistant messages
- `get_tool_calls()` - All tool calls with results
- `get_turns()` / `get_turn(n)` - Conversation turns
- `turn_count()` - Number of turns
- `get_sub_tasks()` - Fetch child task objects
- `get_pending_approvals()` - Pending approval requests
- `approve(id)` / `reject(id, reason)` - Respond to approvals

## Release & Documentation
- **GitHub Pages docs**: https://zachrobo1.github.io/zap-ai/
- **PyPI**: https://pypi.org/project/zap-ai/
- Uses **release-please** for versioning with conventional commits
- MkDocs with Material theme for documentation (auto-deployed via GitHub Actions)
