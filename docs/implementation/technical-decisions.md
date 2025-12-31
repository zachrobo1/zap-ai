# Technical Decisions

This document captures the key technical decisions made during the design of Zap.

---

## Already Decided

### Parallel Tool Execution
**Decision:** Use `asyncio.gather()` for concurrent activity calls

**Rationale:**
- LLM may request multiple tool calls in a single response
- Tools are independent and can execute concurrently
- Reduces total latency for multi-tool responses
- Temporal activities handle individual retries

**Implementation:**
```python
async def _handle_tool_calls(self, tool_calls):
    tasks = [self._execute_mcp_tool(tc) for tc in mcp_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

---

### Sub-Agent Communication
**Decision:** Collaborative messaging (not full delegation)

**Rationale:**
- Parent agent maintains control and context
- Enables multi-turn conversations with sub-agents
- Supports concurrent conversations with multiple sub-agents
- Better observability and debugging
- Follows successful patterns from Anthropic, Google ADK

**Tool Name:** `message_agent` (not `transfer_to_agent`)

---

### Observability
**Decision:** Query-based polling (no streaming initially)

**Rationale:**
- Simpler implementation
- Works well with Temporal's query mechanism
- No need for websocket infrastructure
- Streaming can be added later via callbacks/webhooks

**Future:** May add real-time streaming via callbacks/webhooks in future versions.

---

### Validation
**Decision:** Build-time validation with Pydantic

**Rationale:**
- Catch configuration errors early
- Clear error messages
- Type safety with IDE support
- Validation happens in `Zap.__post_init__()` and `ZapAgent` validators

**Validations:**
- Agent name format (no spaces, alphanumeric)
- Duplicate agent name detection
- Sub-agent reference validation
- Circular dependency detection

---

### Tracing
**Decision:** Abstract `TracingProvider` protocol with Langfuse as first implementation (opt-in)

**Rationale:**
- Langfuse provides excellent agentic tracing with spans, generations, and token tracking
- Abstract protocol allows swapping to OpenTelemetry or custom backends
- Opt-in via `Zap(tracing_provider=...)` keeps core lightweight
- Context propagation via serialized `TraceContext` in activity inputs solves Temporal's process isolation

**Key Design Points:**
- `TracingProvider` protocol defines standard interface
- `TraceContext` dataclass is JSON-serializable for Temporal boundaries
- `NoOpTracingProvider` is the default (zero overhead when tracing disabled)
- Global registry (`set_tracing_provider`) makes provider accessible in activities
- Langfuse optional dependency: `pip install zap-ai[langfuse]`

**Context Propagation Strategy:**
```python
# Activity inputs include optional trace context
@dataclass
class InferenceInput:
    agent_name: str
    model: str
    messages: list[dict]
    tools: list[dict]
    trace_context: dict[str, Any] | None = None  # For observability
```

**Trace Hierarchy:**
```
Trace: task-AgentName-workflowId
  +-- Span: workflow-lifecycle (WORKFLOW)
  +-- Span: iteration-0 (ITERATION)
  |     +-- Generation: inference-AgentName (INFERENCE)
  |     +-- Span: tool-search (TOOL_CALL)
  +-- Span: iteration-1 (ITERATION)
        +-- Span: sub-agent-Helper (SUB_AGENT)
```

---

## Sub-Agent Conversation Design

### Philosophy: Collaboration over Delegation

The parent agent stays in control. Sub-agents are assistants that help with specific subtasks, not replacements that take over completely. This follows the successful patterns from:
- Anthropic's orchestrator-worker pattern
- Google ADK's hierarchical decomposition
- The "Generator-Critic" iterative refinement pattern

### Why Not Full Delegation?
- Parent loses ability to course-correct
- Can't synthesize results from multiple sub-agents
- No iterative refinement possible
- Harder to debug and observe

### The `message_agent` Tool

Instead of "transferring" to a sub-agent, the parent "messages" them:

```python
# Start a new conversation
response = message_agent(
    agent_name="ResearchAgent",
    message="Find information about quantum computing advances in 2024"
)
# Returns: {conversation_id: "conv-abc123", response: "I found several key advances..."}

# Continue the conversation
response = message_agent(
    conversation_id="conv-abc123",
    message="Can you focus specifically on error correction?"
)
# Returns: {conversation_id: "conv-abc123", response: "Regarding error correction..."}

# Start a different conversation (can run concurrently)
response = message_agent(
    agent_name="WriterAgent",
    message="Draft a summary of these findings: ..."
)
# Returns: {conversation_id: "conv-xyz789", response: "Here's a draft summary..."}
```

### How It Works (Temporal Implementation)

```
Parent Workflow                          Child Workflow (Sub-Agent)
     │                                         │
     │  message_agent(agent="Research",        │
     │                message="Find X")        │
     │─────────────────────────────────────────►│
     │     [Start child workflow OR             │ ◄── Process message
     │      send signal to existing]            │     Run inference
     │                                          │     Execute tools
     │                                          │
     │◄─────────────────────────────────────────│
     │  {conversation_id, response}             │ ◄── Return response, then WAIT
     │                                          │     (with timeout)
     │  [Parent continues, may call             │
     │   other tools or sub-agents]             │
     │                                          │
     │  message_agent(conv_id="...",            │
     │                message="Follow up")      │
     │─────────────────────────────────────────►│
     │     [Send signal to child workflow]      │ ◄── Receive signal
     │                                          │     Process follow-up
     │                                          │     Execute tools
     │◄─────────────────────────────────────────│
     │  {conversation_id, response}             │ ◄── Return response, WAIT again
     │                                          │
     │  [Parent done with this sub-agent]       │
     │                                          │ ◄── Eventually times out
     │                                          │     or parent explicitly ends
```

### Key Implementation Details

1. **conversation_id = child workflow ID**
   - When starting: generate new ID, start child workflow
   - When continuing: use existing ID, send signal to child workflow

2. **Child workflow lifecycle:**
   - Receives initial message, processes, returns response
   - Waits for follow-up signal (configurable timeout, e.g., 5 minutes)
   - On signal: processes, returns response, waits again
   - On timeout: workflow completes gracefully

3. **State tracking in parent:**
   ```python
   @dataclass
   class SubAgentConversation:
       conversation_id: str  # Child workflow ID
       agent_name: str
       messages: list[dict]  # History of this conversation
       is_active: bool  # Whether child is still waiting
   ```

4. **Response format:**
   ```python
   {
       "conversation_id": "MainAgent-ResearchAgent-abc123",
       "agent_name": "ResearchAgent",
       "response": "Here's what I found...",
       "is_complete": False  # True if child timed out / completed
   }
   ```

### Advantages of This Design

- **Multi-turn**: Parent can have extended conversations with sub-agents
- **Concurrent**: Parent can message multiple sub-agents simultaneously
- **Controlled**: Parent decides when/if to continue conversations
- **Debuggable**: Each conversation is a separate workflow with its own history
- **Resilient**: Child workflows have their own retry policies

---

## Implementation Notes

### MCP Schema Conversion

```python
# MCP format
{"name": "tool", "inputSchema": {"type": "object", "properties": {...}}}

# LiteLLM format
{"type": "function", "function": {"name": "tool", "parameters": {...}}}
```

### Continue-as-new State

```python
@dataclass
class ConversationState:
    messages: list[dict]
    iteration_count: int
    pending_signals: list[str]
    # Track active sub-agent conversations
    sub_agent_conversations: dict[str, SubAgentConversation]
```

### Retry Policies

| Activity | Max Attempts | Initial Interval | Max Interval | Notes |
|----------|--------------|------------------|--------------|-------|
| Inference | 5 | 1s | 30s | Exponential backoff |
| Tool Execution | 3 | 1s | 10s | Exponential backoff |
| Sub-agent messaging | N/A | N/A | N/A | Handled by child workflow |

---

## Testing Guidelines

### Philosophy

Every piece of functionality must have unit tests. Tests are not optional—they are part of the definition of "done" for any task.

### Minimum Requirements

**Happy Path Testing (Required for ALL code):**
- Every function, method, and class must have at least one test covering the expected use case
- Tests should verify correct inputs produce correct outputs
- Tests should be fast, isolated, and deterministic

**Defensive Code Testing (Required for error-prone areas):**

Areas that require additional defensive testing:
| Area | What to Test |
|------|--------------|
| Input validation | Invalid agent names, missing required fields, malformed data |
| Temporal boundaries | Serialization/deserialization of dataclasses, activity input/output |
| External integrations | LLM provider errors, MCP client failures, network timeouts |
| State management | Continue-as-new state preservation, sub-agent conversation tracking |
| Configuration | Circular dependencies, duplicate names, invalid references |

### Test Structure

```
tests/
├── unit/                       # Fast, isolated tests
│   ├── core/                   # ZapAgent, Task, Zap validation
│   ├── workflows/              # Workflow logic (mocked activities)
│   ├── activities/             # Activity logic (mocked external calls)
│   ├── mcp/                    # Schema conversion, registry
│   ├── llm/                    # Message types, provider wrapper
│   └── tracing/                # Protocol, providers
├── integration/                # Tests with real Temporal (slower)
│   ├── test_workflow_loop.py   # Full agentic loop with mocks
│   └── test_sub_agents.py      # Parent-child workflow interaction
└── conftest.py                 # Shared fixtures
```

### Testing Patterns

**1. Unit Tests (Fast, Isolated)**
```python
# Test one thing, mock dependencies
def test_zap_agent_validates_name():
    with pytest.raises(ValidationError):
        ZapAgent(name="has spaces", prompt="test")

def test_schema_converter_transforms_mcp_to_litellm():
    mcp_tool = {"name": "search", "inputSchema": {...}}
    result = convert_mcp_to_litellm(mcp_tool)
    assert result["type"] == "function"
```

**2. Activity Tests (Mock External Services)**
```python
@pytest.mark.asyncio
async def test_inference_activity_returns_content(mocker):
    # Mock LiteLLM
    mocker.patch("zap_ai.llm.provider.complete", return_value=mock_result)

    result = await inference_activity(InferenceInput(...))
    assert result.content == "expected"
```

**3. Workflow Tests (Mock Activities)**
```python
@pytest.mark.asyncio
async def test_workflow_completes_without_tools():
    # Use Temporal's testing framework
    async with WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(env.client, ...):
            result = await env.client.execute_workflow(...)
            assert result == "expected"
```

**4. Defensive Tests (Error Cases)**
```python
def test_zap_detects_circular_sub_agent_dependency():
    agent_a = ZapAgent(name="A", prompt="...", sub_agents=["B"])
    agent_b = ZapAgent(name="B", prompt="...", sub_agents=["A"])

    with pytest.raises(CircularDependencyError):
        Zap(agents=[agent_a, agent_b])

@pytest.mark.asyncio
async def test_tool_execution_handles_client_error(mocker):
    mocker.patch.object(client, "call_tool", side_effect=ConnectionError)

    with pytest.raises(ToolExecutionError):
        await tool_execution_activity(ToolExecutionInput(...))
```

### What to Mock

| Component | Mock | Don't Mock |
|-----------|------|------------|
| LLM calls | Always (use fixtures) | Never call real APIs in tests |
| MCP clients | Always (use fixtures) | Never call real MCP servers |
| Temporal | Use `WorkflowEnvironment` | Don't mock workflow internals |
| Dataclasses | Never | Test serialization directly |
| Validation | Never | Test validation logic directly |

### Test Fixtures

Create reusable fixtures in `conftest.py`:

```python
@pytest.fixture
def sample_agent():
    return ZapAgent(name="TestAgent", prompt="You are helpful.")

@pytest.fixture
def mock_inference_result():
    return InferenceResult(
        content="Hello!",
        tool_calls=[],
        finish_reason="stop",
    )

@pytest.fixture
def sample_mcp_tool():
    return {
        "name": "search",
        "description": "Search the web",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
```

### Coverage Expectations

| Area | Minimum Coverage |
|------|-----------------|
| Core models | 100% (validation is critical) |
| Schema conversion | 100% (data transformation) |
| Activities | 90%+ (mock externals, test logic) |
| Workflows | 80%+ (complex async logic) |
| Tracing | 90%+ (protocol compliance) |

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=zap_ai --cov-report=term-missing

# Run only unit tests (fast)
pytest tests/unit/

# Run specific test file
pytest tests/unit/core/test_agent.py
```

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

---

## Getting Started

1. Complete Phase 1 first to establish the foundation
2. Phase 2-4 builds the Temporal workflow loop with stubs
3. Checkpoint after Phase 4: test the workflow with mock data
4. Phases 5-6 can be done in parallel (LLM and MCP integrations)
5. Phase 7 completes the Zap orchestrator
6. Phase 8-9 for testing and examples
7. Phase 10 (optional) adds Langfuse tracing

**Estimated scope:** ~60 implementation tasks across all phases (including optional tracing).
