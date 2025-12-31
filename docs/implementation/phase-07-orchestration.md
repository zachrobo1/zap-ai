# Phase 7: Orchestration

**Goal:** Complete the Zap class with full functionality.

**Dependencies:** Phase 1 (Foundation), Phase 2 (MCP Integration), Phase 4 (Temporal Workflows).

---

## Task 7.1: Verify Agent Validation

**File:** `src/zap_ai/core/zap.py`

The agent validation was already implemented in Phase 1. Verify the following methods exist and work correctly:

```python
def __post_init__(self) -> None:
    """Validate configuration at build time."""
    self._validate_no_duplicate_names()
    self._build_agent_map()
    self._validate_sub_agent_references()
    self._validate_no_circular_dependencies()

def _validate_no_duplicate_names(self) -> None:
    """Check that all agent names are unique."""
    names = [agent.name for agent in self.agents]
    if len(names) != len(set(names)):
        duplicates = [name for name in names if names.count(name) > 1]
        raise ZapConfigurationError(
            f"Duplicate agent names detected: {set(duplicates)}. "
            "Each agent must have a unique name."
        )

def _validate_sub_agent_references(self) -> None:
    """Validate that all sub_agent references point to existing agents."""
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
    """Detect circular dependencies using DFS."""
    # Implementation in Phase 1
```

**Checklist:**
- [ ] Verify `_validate_no_duplicate_names()` works
- [ ] Verify `_validate_sub_agent_references()` works
- [ ] Verify `_validate_no_circular_dependencies()` works
- [ ] All validation happens in `__post_init__`

---

## Task 7.2: Implement Zap.start()

**File:** `src/zap_ai/core/zap.py`

Complete the `start()` method:

```python
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
    """
    if self._started:
        raise RuntimeError("Zap has already been started. Cannot call start() twice.")

    # Connect to Temporal if client not provided
    if self.temporal_client is None:
        from temporalio.client import Client
        self.temporal_client = await Client.connect("localhost:7233")

    # Initialize tool registry
    from zap_ai.mcp import ToolRegistry
    self._tool_registry = ToolRegistry()

    # Register all agents (connects MCP clients, discovers tools)
    await self._tool_registry.register_agents(self.agents, self._agent_map)

    # Set registry for activities
    from zap_ai.activities import set_tool_registry
    set_tool_registry(self._tool_registry)

    self._started = True
```

**Note:** Add `_tool_registry` to class attributes:

```python
@dataclass
class Zap:
    # ... existing fields ...

    # Internal state
    _tool_registry: ToolRegistry | None = field(default=None, init=False, repr=False)
```

**Checklist:**
- [ ] Connect to Temporal if `temporal_client` is None
- [ ] Initialize `ToolRegistry`
- [ ] Call `register_agents()` to connect MCP clients
- [ ] Set global tool registry for activities
- [ ] Set `_started = True`

---

## Task 7.3: Implement Zap.execute_task()

**File:** `src/zap_ai/core/zap.py`

Complete the `execute_task()` method:

```python
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
    """
    self._ensure_started()

    if task is None:
        raise ValueError("task argument is required")

    if follow_up_on_task is not None:
        # Follow-up on existing task
        return await self._follow_up_task(follow_up_on_task, task)

    # New task
    if agent_name is None:
        raise ValueError("agent_name is required for new tasks")

    # Validate agent exists
    agent = self.get_agent(agent_name)

    # Generate task ID
    task_id = f"{agent_name}-{uuid4().hex[:12]}"

    # Get agent configuration
    from zap_ai.workflows.models import AgentWorkflowInput
    from zap_ai.workflows.agent_workflow import AgentWorkflow

    # Start Temporal workflow
    await self.temporal_client.start_workflow(
        AgentWorkflow.run,
        AgentWorkflowInput(
            agent_name=agent_name,
            initial_task=task,
        ),
        id=task_id,
        task_queue=self.task_queue,
    )

    return Task(
        id=task_id,
        agent_name=agent_name,
        status=TaskStatus.PENDING,
    )

async def _follow_up_task(self, task_id: str, message: str) -> Task:
    """Send a follow-up message to an existing task."""
    from zap_ai.workflows.agent_workflow import AgentWorkflow

    try:
        handle = self.temporal_client.get_workflow_handle(task_id)

        # Send signal
        await handle.signal(AgentWorkflow.add_message, message)

        # Query current state
        status_str = await handle.query(AgentWorkflow.get_status)
        history = await handle.query(AgentWorkflow.get_history)

        # Parse agent name from task ID
        agent_name = task_id.split("-")[0]

        return Task(
            id=task_id,
            agent_name=agent_name,
            status=TaskStatus(status_str),
            history=history,
        )

    except Exception as e:
        raise TaskNotFoundError(f"Task '{task_id}' not found: {e}") from e
```

**Checklist:**
- [ ] Validate task argument is provided
- [ ] Handle follow-up case with `_follow_up_task()`
- [ ] Validate agent exists for new tasks
- [ ] Generate task ID as `{agent_name}-{uuid}`
- [ ] Start Temporal workflow with `client.start_workflow()`
- [ ] Return `Task` object with pending status
- [ ] Implement `_follow_up_task()` to send signal

---

## Task 7.4: Implement Zap.get_task()

**File:** `src/zap_ai/core/zap.py`

Complete the `get_task()` method:

```python
async def get_task(self, task_id: str) -> Task:
    """
    Get the current state of a task.

    Queries the Temporal workflow for current status, result, and
    conversation history.
    """
    self._ensure_started()

    from zap_ai.workflows.agent_workflow import AgentWorkflow

    try:
        handle = self.temporal_client.get_workflow_handle(task_id)

        # Query workflow state
        status_str = await handle.query(AgentWorkflow.get_status)
        result = await handle.query(AgentWorkflow.get_result)
        error = await handle.query(AgentWorkflow.get_error)
        history = await handle.query(AgentWorkflow.get_history)

        # Parse agent name from task ID
        agent_name = task_id.split("-")[0]

        return Task(
            id=task_id,
            agent_name=agent_name,
            status=TaskStatus(status_str),
            result=result,
            error=error,
            history=history,
        )

    except Exception as e:
        raise TaskNotFoundError(f"Task '{task_id}' not found: {e}") from e
```

**Checklist:**
- [ ] Get workflow handle by ID
- [ ] Query status, result, error, history
- [ ] Build and return `Task` object
- [ ] Handle workflow not found error

---

## Task 7.5: Implement Zap.stop()

**File:** `src/zap_ai/core/zap.py`

Complete the `stop()` method:

```python
async def stop(self) -> None:
    """
    Gracefully shut down Zap.

    Disconnects MCP clients and closes Temporal connection.
    Does not cancel running tasks.
    """
    if not self._started:
        return

    # Shutdown tool registry (disconnects MCP clients)
    if self._tool_registry:
        await self._tool_registry.shutdown()
        self._tool_registry = None

    # Close Temporal client if we created it
    # Note: If client was passed in, we don't close it
    # (caller is responsible for their own client)

    self._started = False
```

**Checklist:**
- [ ] Check if started
- [ ] Shutdown tool registry
- [ ] Set `_started = False`

---

## Phase 7 Verification

After completing all tasks, verify:

1. **Start initializes everything:**
   ```python
   zap = Zap(agents=[
       ZapAgent(name="Test", prompt="You are helpful"),
   ])
   await zap.start()
   assert zap._started
   assert zap._tool_registry is not None
   ```

2. **Execute task starts workflow:**
   ```python
   task = await zap.execute_task(
       agent_name="Test",
       task="Hello",
   )
   assert task.id.startswith("Test-")
   assert task.status == TaskStatus.PENDING
   ```

3. **Get task returns current state:**
   ```python
   task = await zap.get_task(task.id)
   assert task.status in [TaskStatus.PENDING, TaskStatus.RUNNING, TaskStatus.COMPLETED]
   ```

4. **Follow-up works:**
   ```python
   await zap.execute_task(
       follow_up_on_task=task.id,
       task="Follow up question",
   )
   ```

5. **Stop cleans up:**
   ```python
   await zap.stop()
   assert not zap._started
   ```
