"""Main Zap orchestrator class."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic
from uuid import uuid4

from zap_ai.core.agent import ZapAgent
from zap_ai.core.task import Task, TaskStatus
from zap_ai.core.types import TContext
from zap_ai.tracing import NoOpTracingProvider, TracingProvider

if TYPE_CHECKING:
    from temporalio.client import Client as TemporalClient

    from zap_ai.mcp import ToolRegistry


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
class Zap(Generic[TContext]):
    """
    Main orchestrator for the Zap AI agent platform.

    Zap manages a collection of agents and provides methods to execute
    tasks against them. It supports a generic context type that can be
    passed to agents with dynamic prompts. It handles:
    - Agent configuration validation at build time
    - Temporal client connection management
    - Task execution via Temporal workflows
    - Task status queries

    Example:
        ```python
        from zap_ai import Zap, ZapAgent

        # Simple usage with static prompts
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

        # With typed context and dynamic prompts
        from dataclasses import dataclass

        @dataclass
        class MyContext:
            user_name: str
            company: str

        agent = ZapAgent[MyContext](
            name="Helper",
            prompt=lambda ctx: f"You assist {ctx.user_name} from {ctx.company}.",
        )

        zap: Zap[MyContext] = Zap(agents=[agent])
        await zap.start()

        task = await zap.execute_task(
            agent_name="Helper",
            task="Help me with something",
            context=MyContext(user_name="Alice", company="Acme"),
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
    agents: list[ZapAgent[TContext]]
    temporal_client: TemporalClient | None = None
    task_queue: str = "zap-agents"
    tracing_provider: TracingProvider | None = None

    # Internal state (populated after init/start)
    _agent_map: dict[str, ZapAgent[TContext]] = field(default_factory=dict, init=False, repr=False)
    _started: bool = field(default=False, init=False, repr=False)
    _tool_registry: ToolRegistry | None = field(default=None, init=False, repr=False)
    _owns_temporal_client: bool = field(default=False, init=False, repr=False)
    _tracing: TracingProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate configuration at build time.

        Called automatically after dataclass initialization. Performs:
        1. Duplicate agent name detection
        2. Sub-agent reference validation
        3. Circular dependency detection
        4. Builds internal agent lookup map
        5. Initializes tracing provider

        Raises:
            ZapConfigurationError: If any validation fails.
        """
        self._validate_no_duplicate_names()
        self._build_agent_map()
        self._validate_sub_agent_references()
        self._validate_no_circular_dependencies()

        # Initialize tracing (use NoOp if not configured)
        self._tracing = self.tracing_provider or NoOpTracingProvider()

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
        graph: dict[str, list[str]] = {agent.name: agent.sub_agents for agent in self.agents}

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

    def get_agent(self, name: str) -> ZapAgent[TContext]:
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
                f"Agent '{name}' not found. Available agents: {sorted(self._agent_map.keys())}"
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

        # Connect to Temporal if client not provided
        if self.temporal_client is None:
            from temporalio.client import Client

            self.temporal_client = await Client.connect("localhost:7233")
            self._owns_temporal_client = True

        # Initialize tool registry
        from zap_ai.mcp import ToolRegistry

        self._tool_registry = ToolRegistry()

        # Register all agents (connects MCP clients, discovers tools)
        await self._tool_registry.register_agents(self.agents, self._agent_map)

        # Set registry for activities
        from zap_ai.activities.tool_execution import set_tool_registry

        set_tool_registry(self._tool_registry)

        self._started = True

    def _ensure_started(self) -> None:
        """Raise if start() hasn't been called."""
        if not self._started:
            raise ZapNotStartedError("Zap has not been started. Call 'await zap.start()' first.")

    async def execute_task(
        self,
        agent_name: str | None = None,
        task: str | None = None,
        follow_up_on_task: str | None = None,
        context: TContext | None = None,
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
            context: Optional context to pass to agents with dynamic prompts.
                Defaults to {} if not provided. Note: agents with callable
                prompts should be given appropriate context.

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

        Example (with context):
            ```python
            task = await zap.execute_task(
                agent_name="Helper",
                task="Help me with something",
                context=MyContext(user_name="Alice", company="Acme"),
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
            return await self._follow_up_task(follow_up_on_task, task)

        # New task
        if agent_name is None:
            raise ValueError("agent_name is required for new tasks")

        # Validate agent exists and get agent config
        agent = self.get_agent(agent_name)

        # Use default empty dict if no context provided
        effective_context: TContext = context if context is not None else {}  # type: ignore[assignment]

        # Warn if agent has dynamic prompt but no context provided
        if agent.is_dynamic_prompt() and context is None:
            warnings.warn(
                f"Agent '{agent_name}' has a dynamic prompt but no context was provided. "
                "The prompt will be called with an empty dict. "
                "Consider providing context via execute_task(context=...).",
                UserWarning,
                stacklevel=2,
            )

        # Resolve the prompt with context
        resolved_prompt = agent.resolve_prompt(effective_context)

        # Generate task ID
        task_id = f"{agent_name}-{uuid4().hex[:12]}"

        # Get tools for this agent
        tools: list[dict[str, Any]] = []
        if self._tool_registry:
            tools = self._tool_registry.get_tools_for_agent(agent_name)

        # Start Temporal workflow
        from zap_ai.workflows.agent_workflow import AgentWorkflow
        from zap_ai.workflows.models import AgentWorkflowInput

        await self.temporal_client.start_workflow(  # type: ignore[union-attr]
            AgentWorkflow.run,
            AgentWorkflowInput(
                agent_name=agent_name,
                initial_task=task,
                system_prompt=resolved_prompt,
                model=agent.model,
                tools=tools,
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
        """
        Send a follow-up message to an existing task.

        Args:
            task_id: The task ID to send the message to.
            message: The follow-up message.

        Returns:
            Updated Task object.

        Raises:
            TaskNotFoundError: If the task doesn't exist.
        """
        from zap_ai.workflows.agent_workflow import AgentWorkflow

        try:
            handle = self.temporal_client.get_workflow_handle(task_id)  # type: ignore[union-attr]

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

        from zap_ai.workflows.agent_workflow import AgentWorkflow

        try:
            handle = self.temporal_client.get_workflow_handle(task_id)  # type: ignore[union-attr]

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

        # TODO: Phase 7 (optional) - Cancel Temporal workflow
        raise NotImplementedError("cancel_task not yet implemented")

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

        # Note: We don't close the Temporal client even if we created it,
        # as it may be reused or there may be running workflows.
        # The caller is responsible for client lifecycle.

        self._started = False
