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
- **MCP sampling support** - Handle LLM requests from MCP servers via `LiteLLMSamplingHandler` (import from `zap_ai.mcp.sampling`)
- **Dynamic prompts** - context-aware prompts resolved at runtime
- **Context injection to MCP tools** - pass context (user_id, tenant, etc.) to tools via dependency injection, hidden from LLM, with full type safety
- **Conversation history API** - rich access to turns, tool calls, and text content
- **Multimodal/Vision support** - send images to vision-capable models with automatic capability validation
- **Streaming support** - async generator API for real-time event streaming during task execution

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
├── __init__.py         # Public API exports
├── exceptions.py       # All custom exceptions (consolidated, all inherit from ZapError)
├── utils.py            # Shared utility functions
├── core/               # Core models (Zap, ZapAgent, Task)
├── conversation/       # Conversation history parsing
├── workflows/          # Temporal workflow definitions
├── activities/         # Temporal activities (inference, tool execution)
├── llm/                # LLM provider abstraction
├── mcp/                # MCP client management and tool registry
│   ├── context.py      # ZapContext dependency injection helpers
│   └── sampling.py     # MCP sampling handlers
├── streaming/          # Streaming event support
├── tracing/            # Observability providers
└── worker/             # Worker process for running workflows
```

## Architecture Flow
1. User creates `ZapAgent` configurations and a `Zap` instance
2. `zap.start()` connects to Temporal and initializes MCP clients
3. `zap.execute_task()` starts a Temporal workflow for the agent
4. The workflow runs an agentic loop:
   - LLM inference → check approval rules → tool execution → repeat
   - Events are emitted to workflow state at each step (thinking, tool_call, tool_result, completed, error)
5. Approval pauses: if tool matches `ApprovalRules`, workflow waits for signal
6. Sub-agents are executed as child workflows (via `message_agent` tool)
7. State is persisted in Temporal, surviving crashes
8. `continue-as-new` prevents event history from growing unbounded
9. `stream_task()` polls workflow via Temporal query to yield events in real-time

## Exception Hierarchy
All exceptions inherit from `ZapError`:
- `ZapConfigurationError`, `ZapNotStartedError`, `AgentNotFoundError`, `TaskNotFoundError`
- `VisionNotSupportedError`, `ToolNotFoundError`, `ToolExecutionError`
- `ClientConnectionError`, `SchemaConversionError`, `LLMProviderError`

## Context Injection Pattern

Context passed to `execute_task()` serves two purposes:
1. **Dynamic prompts** - Resolve `prompt=lambda ctx: f"..."` at runtime
2. **MCP tool injection** - Pass context to tools via FastMCP's dependency injection

Context is injected to MCP tools using the `Depends()` pattern with these helpers:
- `ZapContext` - Full context dict (or typed object if metadata present)
- `ZapContextValue(key, default)` - Factory for extracting specific values
- `TypedZapContext(context_type)` - Factory for type-safe deserialization

The context dependencies are hidden from the LLM schema - it only sees regular tool parameters.

See `api_quick_reference.md` for code examples and detailed API signatures.

## Release & Documentation
- **GitHub Pages docs**: https://zachrobo1.github.io/zap-ai/
- **PyPI**: https://pypi.org/project/zap-ai/
- Uses **release-please** for versioning with conventional commits
- MkDocs with Material theme for documentation (auto-deployed via GitHub Actions)
