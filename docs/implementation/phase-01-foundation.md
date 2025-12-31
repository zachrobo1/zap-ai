# Phase 1: Foundation

**Goal:** Set up project structure and core data models.

**Dependencies:** None - this is the first phase.

---

## Task 1.1: Create Module Directory Structure

Create the following directory structure with empty `__init__.py` files:

```
src/zap_ai/
├── __init__.py
├── core/
│   └── __init__.py
├── workflows/
│   └── __init__.py
├── activities/
│   └── __init__.py
├── mcp/
│   └── __init__.py
├── llm/
│   └── __init__.py
└── worker/
    └── __init__.py
```

**Checklist:**
- [ ] Create `src/zap_ai/core/` directory
- [ ] Create `src/zap_ai/workflows/` directory
- [ ] Create `src/zap_ai/activities/` directory
- [ ] Create `src/zap_ai/mcp/` directory
- [ ] Create `src/zap_ai/llm/` directory
- [ ] Create `src/zap_ai/worker/` directory
- [ ] Add empty `__init__.py` to each directory

---

## Task 1.2: Implement ZapAgent Pydantic Model

**File:** `src/zap_ai/core/agent.py`

```python
"""Agent configuration model for Zap."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from fastmcp import Client


class ZapAgent(BaseModel):
    """
    Configuration for an AI agent within the Zap platform.

    ZapAgent defines all the properties needed to run an agent, including
    its system prompt, LLM model, available tools (via MCP clients), and
    which other agents it can delegate to.

    Example:
        ```python
        from zap_ai import ZapAgent
        from fastmcp import Client

        agent = ZapAgent(
            name="ResearchAgent",
            prompt="You are a research assistant...",
            model="gpt-4o",
            mcp_clients=[Client("./tools.py")],
            sub_agents=["WriterAgent"],
        )
        ```

    Attributes:
        name: Unique identifier for the agent. Used as workflow ID prefix.
            Cannot contain spaces or special characters that would be
            invalid in a Temporal workflow ID.
        prompt: System prompt that defines the agent's behavior and personality.
            This is sent as the first message in every conversation.
        model: LiteLLM model identifier (e.g., "gpt-4o", "claude-3-opus-20240229",
            "anthropic/claude-3-sonnet"). See LiteLLM docs for full list.
        mcp_clients: List of FastMCP Client instances that provide tools to
            this agent. Clients are connected during Zap.start().
        sub_agents: List of agent names that this agent can delegate to.
            A special "transfer_to_agent" tool is automatically added when
            this list is non-empty. Referenced agents must exist in the
            Zap instance.
        discovery_prompt: Description shown to parent agents when they can
            delegate to this agent. Used in the transfer_to_agent tool
            description. If None, agent won't appear in transfer tool options.
        max_iterations: Maximum number of agentic loop iterations before
            forcing completion. Prevents infinite loops. Each iteration
            is one LLM call + optional tool execution.
    """

    model_config = {"arbitrary_types_allowed": True}

    # Required fields
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the agent (no spaces allowed)",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="System prompt defining agent behavior",
    )

    # Optional fields with defaults
    model: str = Field(
        default="gpt-4o",
        min_length=1,
        description="LiteLLM model identifier",
    )
    mcp_clients: list[Client] = Field(
        default_factory=list,
        description="FastMCP clients providing tools to this agent",
    )
    sub_agents: list[str] = Field(
        default_factory=list,
        description="Names of agents this agent can delegate to",
    )
    discovery_prompt: str | None = Field(
        default=None,
        description="Description for parent agents (shown in transfer tool)",
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum agentic loop iterations",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """
        Validate that name is suitable for use as a Temporal workflow ID prefix.

        Rules:
        - No spaces (would break workflow ID parsing)
        - No forward slashes (used as delimiter in some contexts)
        - Must be alphanumeric with underscores/hyphens only

        Raises:
            ValueError: If name contains invalid characters.
        """
        if " " in v:
            raise ValueError(
                f"Agent name cannot contain spaces: '{v}'. "
                "Use underscores or hyphens instead."
            )
        if "/" in v:
            raise ValueError(
                f"Agent name cannot contain forward slashes: '{v}'."
            )
        # Allow alphanumeric, underscore, hyphen
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        invalid_chars = set(v) - allowed_chars
        if invalid_chars:
            raise ValueError(
                f"Agent name contains invalid characters: {invalid_chars}. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )
        return v

    @field_validator("sub_agents")
    @classmethod
    def validate_sub_agents_no_duplicates(cls, v: list[str]) -> list[str]:
        """Ensure no duplicate sub-agent references."""
        if len(v) != len(set(v)):
            duplicates = [name for name in v if v.count(name) > 1]
            raise ValueError(f"Duplicate sub-agent references: {set(duplicates)}")
        return v
```

**Checklist:**
- [ ] Create `src/zap_ai/core/agent.py`
- [ ] Implement `ZapAgent` class with all fields
- [ ] Add `validate_name_format` field validator
- [ ] Add `validate_sub_agents_no_duplicates` field validator
- [ ] Add comprehensive docstring with example
- [ ] Add `model_config` for arbitrary_types_allowed

---

## Task 1.3: Implement Task and TaskStatus Models

**File:** `src/zap_ai/core/task.py`

```python
"""Task models for tracking agent execution state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """
    Status of a task execution.

    The lifecycle of a task typically follows:
    PENDING -> RUNNING -> (AWAITING_TOOL <-> RUNNING)* -> COMPLETED

    At any point, a task can transition to FAILED if an unrecoverable
    error occurs.

    Attributes:
        PENDING: Task has been created but workflow hasn't started yet.
        RUNNING: Agent is actively processing (LLM inference in progress
            or between tool calls).
        AWAITING_TOOL: Waiting for one or more tool executions to complete.
            Multiple tools may run in parallel.
        DELEGATED: Task has been delegated to a sub-agent via child workflow.
            Parent workflow is waiting for child to complete.
        COMPLETED: Task finished successfully. Result is available.
        FAILED: Task failed with an error. Error details available in
            Task.error field.
    """

    PENDING = "pending"
    RUNNING = "running"
    AWAITING_TOOL = "awaiting_tool"
    DELEGATED = "delegated"
    COMPLETED = "completed"
    FAILED = "failed"

    def is_terminal(self) -> bool:
        """Return True if this is a terminal (final) status."""
        return self in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    def is_active(self) -> bool:
        """Return True if the task is actively being processed."""
        return self in (TaskStatus.RUNNING, TaskStatus.AWAITING_TOOL, TaskStatus.DELEGATED)


@dataclass
class Task:
    """
    Represents a task execution within the Zap platform.

    A Task is created when you call `zap.execute_task()` and tracks the
    full lifecycle of that execution. Use `zap.get_task(task_id)` to
    retrieve updated task state.

    Example:
        ```python
        task = await zap.execute_task(agent_name="MyAgent", task="Do something")
        print(f"Task ID: {task.id}")

        # Poll for completion
        while not task.status.is_terminal():
            await asyncio.sleep(1)
            task = await zap.get_task(task.id)

        if task.status == TaskStatus.COMPLETED:
            print(f"Result: {task.result}")
        else:
            print(f"Failed: {task.error}")
        ```

    Attributes:
        id: Unique identifier for this task. Format: "{agent_name}-{uuid}".
            Used as the Temporal workflow ID.
        agent_name: Name of the agent executing this task.
        status: Current execution status. See TaskStatus for details.
        result: Final result string if completed, None otherwise.
        history: List of conversation messages in LiteLLM format.
            Each message is a dict with "role" and "content" keys.
            May include tool calls and tool results.
        sub_tasks: List of child task IDs spawned for sub-agent delegation.
        error: Error message if failed, None otherwise.
        created_at: Timestamp when task was created.
        updated_at: Timestamp of last status update.
    """

    # Required fields (set at creation)
    id: str
    agent_name: str

    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None

    # Conversation history (list of LiteLLM message dicts)
    history: list[dict[str, Any]] = field(default_factory=list)

    # Sub-task tracking (child workflow IDs)
    sub_tasks: list[str] = field(default_factory=list)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def is_complete(self) -> bool:
        """Return True if task has reached a terminal state."""
        return self.status.is_terminal()

    def is_successful(self) -> bool:
        """Return True if task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    def get_last_message(self) -> dict[str, Any] | None:
        """Return the most recent message in history, or None if empty."""
        return self.history[-1] if self.history else None

    def get_assistant_messages(self) -> list[dict[str, Any]]:
        """Return all assistant messages from history."""
        return [msg for msg in self.history if msg.get("role") == "assistant"]

    def get_tool_calls_count(self) -> int:
        """Return total number of tool calls made during this task."""
        count = 0
        for msg in self.history:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                count += len(msg["tool_calls"])
        return count
```

**Checklist:**
- [ ] Create `src/zap_ai/core/task.py`
- [ ] Implement `TaskStatus` enum with all states
- [ ] Add `is_terminal()` and `is_active()` helper methods to TaskStatus
- [ ] Implement `Task` dataclass with all fields
- [ ] Add helper methods: `is_complete()`, `is_successful()`, `get_last_message()`, etc.
- [ ] Add comprehensive docstrings with examples

---

## Task 1.4: Implement Zap Class Skeleton

**File:** `src/zap_ai/core/zap.py`

```python
"""Main Zap orchestrator class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Awaitable
from uuid import uuid4

from zap_ai.core.agent import ZapAgent
from zap_ai.core.task import Task, TaskStatus

if TYPE_CHECKING:
    from temporalio.client import Client as TemporalClient


class ZapConfigurationError(Exception):
    """Raised when Zap configuration is invalid."""
    pass


class ZapNotStartedError(Exception):
    """Raised when operations are attempted before calling start()."""
    pass


class AgentNotFoundError(Exception):
    """Raised when referencing an agent that doesn't exist."""
    pass


class TaskNotFoundError(Exception):
    """Raised when referencing a task that doesn't exist."""
    pass


@dataclass
class Zap:
    """
    Main orchestrator for the Zap AI agent platform.

    Zap manages a collection of agents and provides methods to execute
    tasks against them. It handles:
    - Agent configuration validation at build time
    - Temporal client connection management
    - Task execution via Temporal workflows
    - Task status queries

    Example:
        ```python
        from zap_ai import Zap, ZapAgent

        agents = [
            ZapAgent(name="MainAgent", prompt="You are helpful..."),
            ZapAgent(name="HelperAgent", prompt="You assist with..."),
        ]

        zap = Zap(agents=agents)
        await zap.start()

        task = await zap.execute_task(
            agent_name="MainAgent",
            task="Help me with something",
        )
        ```

    Attributes:
        agents: List of ZapAgent configurations. Validated at instantiation.
        temporal_client: Optional pre-configured Temporal client. If None,
            a default connection to localhost:7233 is created in start().
        task_queue: Temporal task queue name for agent workflows.
            Default is "zap-agents".
    """

    # Configuration (set at init)
    agents: list[ZapAgent]
    temporal_client: TemporalClient | None = None
    task_queue: str = "zap-agents"

    # Internal state (populated after init/start)
    _agent_map: dict[str, ZapAgent] = field(default_factory=dict, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate configuration at build time.

        Called automatically after dataclass initialization. Performs:
        1. Duplicate agent name detection
        2. Sub-agent reference validation
        3. Circular dependency detection
        4. Builds internal agent lookup map

        Raises:
            ZapConfigurationError: If any validation fails.
        """
        self._validate_no_duplicate_names()
        self._build_agent_map()
        self._validate_sub_agent_references()
        self._validate_no_circular_dependencies()

    def _validate_no_duplicate_names(self) -> None:
        """
        Check that all agent names are unique.

        Raises:
            ZapConfigurationError: If duplicate names found.
        """
        names = [agent.name for agent in self.agents]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ZapConfigurationError(
                f"Duplicate agent names detected: {set(duplicates)}. "
                "Each agent must have a unique name."
            )

    def _build_agent_map(self) -> None:
        """Build internal name -> agent lookup map."""
        self._agent_map = {agent.name: agent for agent in self.agents}

    def _validate_sub_agent_references(self) -> None:
        """
        Validate that all sub_agent references point to existing agents.

        Raises:
            ZapConfigurationError: If any sub-agent reference is invalid.
        """
        all_names = set(self._agent_map.keys())
        for agent in self.agents:
            for sub_name in agent.sub_agents:
                if sub_name not in all_names:
                    raise ZapConfigurationError(
                        f"Agent '{agent.name}' references unknown sub-agent '{sub_name}'. "
                        f"Available agents: {sorted(all_names)}"
                    )
                if sub_name == agent.name:
                    raise ZapConfigurationError(
                        f"Agent '{agent.name}' cannot reference itself as a sub-agent."
                    )

    def _validate_no_circular_dependencies(self) -> None:
        """
        Detect circular dependencies in sub-agent relationships.

        Uses DFS to find cycles in the agent dependency graph.

        Raises:
            ZapConfigurationError: If a circular dependency is detected.
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {
            agent.name: agent.sub_agents for agent in self.agents
        }

        # Track visited and recursion stack for cycle detection
        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []

        def dfs(node: str) -> bool:
            """Return True if cycle detected."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    # Found cycle - build cycle path for error message
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    raise ZapConfigurationError(
                        f"Circular dependency detected: {' -> '.join(cycle)}"
                    )

            path.pop()
            rec_stack.remove(node)
            return False

        for agent_name in graph:
            if agent_name not in visited:
                dfs(agent_name)

    def get_agent(self, name: str) -> ZapAgent:
        """
        Get an agent by name.

        Args:
            name: The agent name to look up.

        Returns:
            The ZapAgent with the given name.

        Raises:
            AgentNotFoundError: If no agent with that name exists.
        """
        if name not in self._agent_map:
            raise AgentNotFoundError(
                f"Agent '{name}' not found. "
                f"Available agents: {sorted(self._agent_map.keys())}"
            )
        return self._agent_map[name]

    def list_agents(self) -> list[str]:
        """Return list of all agent names."""
        return list(self._agent_map.keys())

    async def start(self) -> None:
        """
        Initialize Temporal connection and MCP clients.

        Must be called before execute_task() or get_task(). This method:
        1. Connects to Temporal (if client not provided)
        2. Initializes the tool registry
        3. Pre-connects all MCP clients and discovers tools

        Raises:
            RuntimeError: If start() has already been called.
            ConnectionError: If Temporal connection fails.

        Example:
            ```python
            zap = Zap(agents=[...])
            await zap.start()  # Must call before using
            ```
        """
        if self._started:
            raise RuntimeError("Zap has already been started. Cannot call start() twice.")

        # TODO: Phase 6 - Implement full start() logic
        # 1. Connect to Temporal if self.temporal_client is None
        # 2. Initialize ToolRegistry
        # 3. Pre-connect all MCP clients

        self._started = True

    def _ensure_started(self) -> None:
        """Raise if start() hasn't been called."""
        if not self._started:
            raise ZapNotStartedError(
                "Zap has not been started. Call 'await zap.start()' first."
            )

    async def execute_task(
        self,
        agent_name: str | None = None,
        task: str | None = None,
        follow_up_on_task: str | None = None,
    ) -> Task:
        """
        Execute a new task or follow up on an existing one.

        For new tasks, starts a Temporal workflow for the specified agent.
        For follow-ups, sends a signal to the existing workflow.

        Args:
            agent_name: Name of the agent to execute the task. Required for
                new tasks, ignored for follow-ups (uses original agent).
            task: The task description/prompt to send to the agent. Required.
            follow_up_on_task: If provided, sends the task as a follow-up
                message to an existing task instead of starting a new one.

        Returns:
            Task object with initial state. Use get_task() to poll for updates.

        Raises:
            ZapNotStartedError: If start() hasn't been called.
            AgentNotFoundError: If agent_name doesn't exist (new tasks only).
            TaskNotFoundError: If follow_up_on_task doesn't exist.
            ValueError: If required arguments are missing.

        Example (new task):
            ```python
            task = await zap.execute_task(
                agent_name="MyAgent",
                task="Analyze this data and summarize findings",
            )
            ```

        Example (follow-up):
            ```python
            task = await zap.execute_task(
                follow_up_on_task=task.id,
                task="Now export the summary to PDF",
            )
            ```
        """
        self._ensure_started()

        if task is None:
            raise ValueError("task argument is required")

        if follow_up_on_task is not None:
            # Follow-up on existing task
            # TODO: Phase 6 - Send signal to existing workflow
            # 1. Get workflow handle by ID
            # 2. Send add_message signal
            # 3. Return updated Task
            raise NotImplementedError("Follow-up tasks not yet implemented")

        # New task
        if agent_name is None:
            raise ValueError("agent_name is required for new tasks")

        # Validate agent exists
        agent = self.get_agent(agent_name)  # Raises AgentNotFoundError if not found

        # Generate task ID
        task_id = f"{agent_name}-{uuid4().hex[:12]}"

        # TODO: Phase 6 - Start Temporal workflow
        # 1. Create AgentWorkflowInput
        # 2. Start workflow with client.start_workflow()

        return Task(
            id=task_id,
            agent_name=agent_name,
            status=TaskStatus.PENDING,
        )

    async def get_task(self, task_id: str) -> Task:
        """
        Get the current state of a task.

        Queries the Temporal workflow for current status, result, and
        conversation history.

        Args:
            task_id: The task ID returned from execute_task().

        Returns:
            Task object with current state.

        Raises:
            ZapNotStartedError: If start() hasn't been called.
            TaskNotFoundError: If no task with that ID exists.

        Example:
            ```python
            task = await zap.get_task(task_id)
            print(f"Status: {task.status}")
            if task.is_complete():
                print(f"Result: {task.result}")
            ```
        """
        self._ensure_started()

        # TODO: Phase 6 - Query Temporal workflow
        # 1. Get workflow handle by ID
        # 2. Query status, result, history
        # 3. Build and return Task object

        raise NotImplementedError("get_task not yet implemented")

    async def cancel_task(self, task_id: str) -> None:
        """
        Cancel a running task.

        Sends a cancellation request to the Temporal workflow. The task
        will transition to FAILED status with a cancellation error.

        Args:
            task_id: The task ID to cancel.

        Raises:
            ZapNotStartedError: If start() hasn't been called.
            TaskNotFoundError: If no task with that ID exists.
        """
        self._ensure_started()

        # TODO: Phase 6 (optional) - Cancel Temporal workflow
        raise NotImplementedError("cancel_task not yet implemented")

    async def stop(self) -> None:
        """
        Gracefully shut down Zap.

        Disconnects MCP clients and closes Temporal connection.
        Does not cancel running tasks.
        """
        if not self._started:
            return

        # TODO: Phase 6 - Implement shutdown
        # 1. Disconnect all MCP clients
        # 2. Close Temporal client if we created it

        self._started = False
```

**Checklist:**
- [ ] Create `src/zap_ai/core/zap.py`
- [ ] Define exception classes: `ZapConfigurationError`, `ZapNotStartedError`, `AgentNotFoundError`, `TaskNotFoundError`
- [ ] Implement `Zap` dataclass with fields: `agents`, `temporal_client`, `task_queue`
- [ ] Implement `__post_init__` with validation calls
- [ ] Implement `_validate_no_duplicate_names()`
- [ ] Implement `_build_agent_map()`
- [ ] Implement `_validate_sub_agent_references()`
- [ ] Implement `_validate_no_circular_dependencies()` with DFS
- [ ] Implement `get_agent()` and `list_agents()` helpers
- [ ] Implement `start()` skeleton with TODO comments
- [ ] Implement `execute_task()` skeleton with argument validation
- [ ] Implement `get_task()` skeleton
- [ ] Implement `cancel_task()` skeleton (optional)
- [ ] Implement `stop()` skeleton
- [ ] Add comprehensive docstrings with examples

---

## Task 1.5: Implement Core Module Init

**File:** `src/zap_ai/core/__init__.py`

```python
"""Core Zap models and orchestrator."""

from zap_ai.core.agent import ZapAgent
from zap_ai.core.task import Task, TaskStatus
from zap_ai.core.zap import (
    Zap,
    ZapConfigurationError,
    ZapNotStartedError,
    AgentNotFoundError,
    TaskNotFoundError,
)

__all__ = [
    "Zap",
    "ZapAgent",
    "Task",
    "TaskStatus",
    "ZapConfigurationError",
    "ZapNotStartedError",
    "AgentNotFoundError",
    "TaskNotFoundError",
]
```

**Checklist:**
- [ ] Create `src/zap_ai/core/__init__.py`
- [ ] Import and re-export all public classes
- [ ] Define `__all__` list

---

## Task 1.6: Update Main Package Init

**File:** `src/zap_ai/__init__.py`

```python
"""
Zap - Zach's Agent Platform

A library for building resilient AI agents on Temporal.

Example:
    ```python
    from zap_ai import Zap, ZapAgent, TaskStatus
    from fastmcp import Client

    agent = ZapAgent(
        name="MyAgent",
        prompt="You are a helpful assistant.",
        mcp_clients=[Client("./tools.py")],
    )

    zap = Zap(agents=[agent])
    await zap.start()

    task = await zap.execute_task(
        agent_name="MyAgent",
        task="Help me with something",
    )

    while not task.status.is_terminal():
        await asyncio.sleep(1)
        task = await zap.get_task(task.id)

    print(task.result)
    ```
"""

from zap_ai.core import (
    Zap,
    ZapAgent,
    Task,
    TaskStatus,
    ZapConfigurationError,
    ZapNotStartedError,
    AgentNotFoundError,
    TaskNotFoundError,
)

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "Zap",
    "ZapAgent",
    "Task",
    "TaskStatus",
    # Exceptions
    "ZapConfigurationError",
    "ZapNotStartedError",
    "AgentNotFoundError",
    "TaskNotFoundError",
    # Metadata
    "__version__",
]
```

**Checklist:**
- [ ] Update `src/zap_ai/__init__.py`
- [ ] Import from `zap_ai.core`
- [ ] Define `__version__`
- [ ] Define `__all__`
- [ ] Add module docstring with usage example

---

## Task 1.7: Create Empty Init Files for Other Modules

These files should be empty or contain minimal docstrings for now. They'll be populated in later phases.

**File:** `src/zap_ai/workflows/__init__.py`
```python
"""Temporal workflow definitions."""
```

**File:** `src/zap_ai/activities/__init__.py`
```python
"""Temporal activity definitions."""
```

**File:** `src/zap_ai/mcp/__init__.py`
```python
"""MCP client management and tool handling."""
```

**File:** `src/zap_ai/llm/__init__.py`
```python
"""LLM provider integration."""
```

**File:** `src/zap_ai/worker/__init__.py`
```python
"""Temporal worker setup."""
```

**Checklist:**
- [ ] Create `src/zap_ai/workflows/__init__.py` with docstring
- [ ] Create `src/zap_ai/activities/__init__.py` with docstring
- [ ] Create `src/zap_ai/mcp/__init__.py` with docstring
- [ ] Create `src/zap_ai/llm/__init__.py` with docstring
- [ ] Create `src/zap_ai/worker/__init__.py` with docstring

---

## Phase 1 Verification

After completing all tasks, verify:

1. **Imports work:**
   ```python
   from zap_ai import Zap, ZapAgent, Task, TaskStatus
   ```

2. **Agent validation works:**
   ```python
   # Should raise on spaces
   ZapAgent(name="Bad Name", prompt="test")  # ZapConfigurationError

   # Should work
   ZapAgent(name="GoodName", prompt="test")
   ```

3. **Zap validation works:**
   ```python
   agents = [
       ZapAgent(name="A", prompt="test", sub_agents=["B"]),
       ZapAgent(name="B", prompt="test", sub_agents=["A"]),  # Circular!
   ]
   Zap(agents=agents)  # ZapConfigurationError: circular dependency
   ```

4. **Start/execute flow:**
   ```python
   zap = Zap(agents=[ZapAgent(name="Test", prompt="test")])

   # Should fail before start
   await zap.execute_task(agent_name="Test", task="hello")  # ZapNotStartedError

   await zap.start()
   task = await zap.execute_task(agent_name="Test", task="hello")
   assert task.id.startswith("Test-")
   assert task.status == TaskStatus.PENDING
   ```
