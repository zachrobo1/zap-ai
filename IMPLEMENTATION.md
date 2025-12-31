# Zap Implementation Roadmap

This document provides an overview of the implementation tasks required to build the Zap AI agent platform. Detailed task specifications are organized into separate phase documents for easier management.

## Overview

**Target Architecture:**
```
src/zap_ai/
├── __init__.py                 # Public API exports
├── core/
│   ├── __init__.py
│   ├── agent.py                # ZapAgent Pydantic model
│   ├── zap.py                  # Main Zap orchestrator
│   ├── task.py                 # Task and TaskStatus models
│   └── config.py               # Configuration dataclasses
├── workflows/
│   ├── __init__.py
│   ├── agent_workflow.py       # Main AgentWorkflow (entity pattern)
│   └── models.py               # Workflow input/output dataclasses
├── activities/
│   ├── __init__.py
│   ├── inference.py            # LLM inference activity
│   └── tool_execution.py       # MCP tool execution activity
├── mcp/
│   ├── __init__.py
│   ├── client_manager.py       # FastMCP client lifecycle
│   ├── schema_converter.py     # MCP → LiteLLM tool format
│   └── tool_registry.py        # Tool discovery and caching
├── llm/
│   ├── __init__.py
│   ├── provider.py             # LiteLLM wrapper
│   └── message_types.py        # Message dataclasses
├── tracing/                    # Optional observability (Phase 10)
│   ├── __init__.py             # Public exports + global registry
│   ├── protocol.py             # TracingProvider protocol
│   ├── noop_provider.py        # No-op provider (default)
│   └── langfuse_provider.py    # Langfuse implementation
└── worker/
    ├── __init__.py
    └── worker.py               # Temporal worker setup
```

---

## Implementation Order

The implementation follows a **Temporal-first** approach: get the workflow loop working with stubs, then incrementally plug in real LLM and MCP functionality.

### Phase 1: Foundation
**Goal:** Core data models and project structure.

| File | Description |
|------|-------------|
| [phase-01-foundation.md](docs/implementation/phase-01-foundation.md) | ZapAgent, Task, TaskStatus, Zap skeleton |

### Phase 2: Temporal Workflows
**Goal:** Entity workflow pattern with agentic loop structure. Uses stub activity calls.

| File | Description |
|------|-------------|
| [phase-02-temporal-workflows.md](docs/implementation/phase-02-temporal-workflows.md) | AgentWorkflow with signals, queries, continue-as-new |

### Phase 3: Activities (Stubs)
**Goal:** Activity interfaces with stub implementations that return mock data.

| File | Description |
|------|-------------|
| [phase-03-activities.md](docs/implementation/phase-03-activities.md) | inference_activity, tool_execution_activity stubs |

### Phase 4: Worker
**Goal:** Worker setup to run and test the workflow loop.

| File | Description |
|------|-------------|
| [phase-04-worker.md](docs/implementation/phase-04-worker.md) | Temporal worker configuration and CLI |

**Checkpoint:** At this point, you can run the full workflow loop with mock LLM responses and mock tool results. This validates the Temporal patterns before adding complexity.

### Phase 5: LLM Integration
**Goal:** Replace inference stub with real LiteLLM calls.

| File | Description |
|------|-------------|
| [phase-05-llm-integration.md](docs/implementation/phase-05-llm-integration.md) | LiteLLM wrapper, message types, real inference activity |

### Phase 6: MCP Integration
**Goal:** Replace tool execution stub with real FastMCP calls.

| File | Description |
|------|-------------|
| [phase-06-mcp-integration.md](docs/implementation/phase-06-mcp-integration.md) | Schema converter, ClientManager, ToolRegistry |

### Phase 7: Orchestration
**Goal:** Complete the Zap class with full functionality.

| File | Description |
|------|-------------|
| [phase-07-orchestration.md](docs/implementation/phase-07-orchestration.md) | Zap.start(), execute_task(), get_task() |

### Phase 8: Testing
**Goal:** Unit and integration tests.

| File | Description |
|------|-------------|
| [phase-08-testing.md](docs/implementation/phase-08-testing.md) | Tests for all components |

### Phase 9: Examples
**Goal:** Working examples and documentation.

| File | Description |
|------|-------------|
| [phase-09-examples.md](docs/implementation/phase-09-examples.md) | Basic agent, multi-agent examples |

### Phase 10: Tracing (Optional)
**Goal:** Add comprehensive observability via Langfuse integration.

| File | Description |
|------|-------------|
| [phase-10-tracing.md](docs/implementation/phase-10-tracing.md) | TracingProvider protocol, Langfuse implementation, context propagation |

**Note:** Phase 10 is optional and can be implemented independently after core functionality works. Tracing is opt-in via the `Zap` constructor.

**Technical Decisions:** [technical-decisions.md](docs/implementation/technical-decisions.md)

---

## Dependency Graph

```
Phase 1 (Foundation)
    │
    └──► Phase 2 (Temporal Workflows)
              │
              └──► Phase 3 (Activities - Stubs)
                        │
                        └──► Phase 4 (Worker)
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
          Phase 5 (LLM)              Phase 6 (MCP)
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                        Phase 7 (Orchestration)
                                  │
                                  ▼
                        Phase 8 (Testing)
                                  │
                                  ▼
                        Phase 9 (Examples)
                                  │
                                  ▼
                        Phase 10 (Tracing) [Optional]
```

**Key insight:** Phases 5 (LLM) and 6 (MCP) can be done in parallel after Phase 4, since they're independent integrations that plug into the existing activity interfaces. Phase 10 (Tracing) is optional and can be done any time after Phase 7.

---

## Quick Reference: Task Counts

| Phase | Focus | Tasks | Status |
|-------|-------|-------|--------|
| 1 | Foundation | 7 tasks | ⬜ Not started |
| 2 | Temporal Workflows | 9 tasks | ⬜ Not started |
| 3 | Activities (Stubs) | 3 tasks | ⬜ Not started |
| 4 | Worker | 3 tasks | ⬜ Not started |
| 5 | LLM Integration | 4 tasks | ⬜ Not started |
| 6 | MCP Integration | 4 tasks | ⬜ Not started |
| 7 | Orchestration | 5 tasks | ⬜ Not started |
| 8 | Testing | 5 tasks | ⬜ Not started |
| 9 | Examples | 4 tasks | ⬜ Not started |
| 10 | Tracing (Optional) | 16 tasks | ⬜ Not started |
| **Total** | | **~60 tasks** | |

---

## Key Technical Decisions

These decisions have been made and are documented in detail in [technical-decisions.md](docs/implementation/technical-decisions.md):

- **Parallel tool execution:** Use `asyncio.gather()` for concurrent activity calls
- **Sub-agent communication:** Collaborative messaging via `message_agent` (not full delegation)
- **Observability:** Query-based polling (no streaming initially)
- **Validation:** Build-time with Pydantic
- **Tracing:** Abstract `TracingProvider` protocol with Langfuse as first implementation (opt-in)
- **Testing:** Unit tests required for all code; defensive tests for error-prone areas (see [Testing Guidelines](docs/implementation/technical-decisions.md#testing-guidelines))

---

## Why Temporal-First?

Building the Temporal workflow loop with stubs first provides several benefits:

1. **Validate patterns early** - Test entity workflow, signals, queries, and continue-as-new before adding LLM complexity
2. **Faster iteration** - Mock responses are instant; no API costs or rate limits during development
3. **Clear interfaces** - Activity stubs define the contract between workflow and integrations
4. **Easier debugging** - When something breaks, you know it's in the new integration, not the workflow
5. **Parallel development** - LLM and MCP integrations can be built simultaneously after Phase 4

---

## Running the Examples

Once all phases are complete:

```bash
# 1. Start Temporal server
temporal server start-dev

# 2. Start the Zap worker (in another terminal)
python -m zap_ai.worker

# 3. Run an example (in another terminal)
python examples/basic_agent.py
```