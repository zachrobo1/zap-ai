# Zap - Zach's Agent Platform

## Purpose
Zap is an opinionated library for building **resilient AI agents** on top of Temporal. It provides a scalable, fault-tolerant way to create AI agents that can power demanding use cases and complex architectures.

## Key Benefits
- **Automatic retries** with configurable policies for LLM and tool calls
- **State persistence** - agents survive crashes and can resume mid-conversation
- **Sub-agent delegation** - compose complex systems from specialized agents
- **MCP integration** - easily add tools via the Model Context Protocol
- **Provider agnostic** - use any LLM supported by LiteLLM

## Tech Stack
- **Python 3.11+** - Core language
- **Temporal** - Fault-tolerant workflow orchestration
- **LiteLLM** - Unified LLM provider interface (supports OpenAI, Anthropic, etc.)
- **FastMCP** - Model Context Protocol client for tool integration
- **Pydantic** - Data validation and settings management
- **python-dotenv** - Environment variable management

### Optional Dependencies
- **Langfuse** - Tracing/observability (install with `pip install zap-ai[langfuse]`)

### Development Dependencies
- pytest, pytest-asyncio, pytest-mock, pytest-cov (testing)
- ruff (linting & formatting)
- pre-commit (git hooks)

## Package Structure
```
src/zap_ai/
├── __init__.py         # Public API exports
├── exceptions.py       # All custom exceptions (consolidated)
├── utils.py            # Shared utility functions (e.g., parse_tool_arguments)
├── core/               # Core models (Zap, ZapAgent, Task)
│   ├── __init__.py     # Re-exports core types
│   ├── agent.py        # ZapAgent configuration model
│   ├── task.py         # Task execution tracking model
│   ├── types.py        # Type aliases (TContext, DynamicPrompt)
│   ├── validation.py   # Extracted validation functions
│   └── zap.py          # Main Zap orchestrator
├── conversation/       # Conversation history parsing
│   ├── __init__.py     # Exports ConversationTurn, ToolCallInfo
│   ├── models.py       # ConversationTurn, ToolCallInfo dataclasses
│   └── parser.py       # History parsing functions (get_turns, etc.)
├── workflows/          # Temporal workflow definitions
│   ├── __init__.py
│   ├── agent_workflow.py # Main agentic loop workflow
│   └── models.py       # Workflow input/state models
├── activities/         # Temporal activities (inference, tool execution)
│   ├── __init__.py
│   ├── inference.py    # LLM inference activity
│   └── tool_execution.py # MCP tool execution activity
├── llm/                # LLM provider abstraction
│   ├── __init__.py
│   ├── provider.py     # LiteLLM wrapper
│   └── message_types.py # Message, ToolCall, InferenceResult
├── mcp/                # MCP client management and tool registry
│   ├── __init__.py
│   ├── client_manager.py # FastMCP client lifecycle
│   ├── schema_converter.py # MCP to LiteLLM schema conversion
│   └── tool_registry.py # Tool discovery and caching
├── tracing/            # Observability providers (Langfuse, noop)
│   ├── __init__.py     # Global provider registry
│   ├── protocol.py     # TracingProvider protocol, TraceContext
│   ├── noop_provider.py # No-op fallback
│   └── langfuse_provider.py # Langfuse implementation
└── worker/             # Worker process for running workflows
    ├── __init__.py
    ├── worker.py       # Worker creation functions
    └── __main__.py     # CLI entry point
```

## Main Classes
- `Zap` - Main orchestrator that manages agents and Temporal connections
- `ZapAgent` - Configuration for an AI agent (name, prompt, model, tools, sub-agents)
- `Task` - Represents an executing or completed task
- `TaskStatus` - Enum for task states (PENDING, THINKING, AWAITING_TOOL, COMPLETED, FAILED)
- `ToolCallInfo` - Information about a tool call and its result
- `ConversationTurn` - A single turn in the conversation

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

## Architecture
1. User creates `ZapAgent` configurations and a `Zap` instance
2. `zap.start()` connects to Temporal and initializes MCP clients
3. `zap.execute_task()` starts a Temporal workflow for the agent
4. The workflow runs an agentic loop: LLM inference → tool execution → repeat
5. Sub-agents are executed as child workflows
6. State is persisted in Temporal, surviving crashes
