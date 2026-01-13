"""Main Zap orchestrator class."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Generic
from uuid import uuid4

from zap_ai.core.agent import ZapAgent
from zap_ai.core.task import Task, TaskStatus
from zap_ai.core.types import TContext
from zap_ai.core.validation import (
    build_agent_map,
    validate_no_circular_dependencies,
    validate_no_duplicate_names,
    validate_sub_agent_references,
)
from zap_ai.exceptions import (
    AgentNotFoundError,
    TaskNotFoundError,
    ZapNotStartedError,
)
from zap_ai.tracing import NoOpTracingProvider, TracingProvider

if TYPE_CHECKING:
    from temporalio.client import Client as TemporalClient

    from zap_ai.mcp import ToolRegistry
    from zap_ai.workflows.models import ApprovalRequest, ApprovalRules


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
        validate_no_duplicate_names(self.agents)
        self._agent_map = build_agent_map(self.agents)
        validate_sub_agent_references(self.agents, self._agent_map)
        validate_no_circular_dependencies(self.agents, self._agent_map)

        # Initialize tracing (use NoOp if not configured)
        self._tracing = self.tracing_provider or NoOpTracingProvider()

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

    async def get_agent_tools(self, agent_name: str) -> list[str]:
        """
        Get list of tool names available to an agent.

        Useful for validating approval patterns before execution.

        Args:
            agent_name: Name of the agent.

        Returns:
            List of tool names available to the agent.

        Raises:
            ZapNotStartedError: If start() hasn't been called.
            AgentNotFoundError: If agent doesn't exist.

        Example:
            ```python
            tools = await zap.get_agent_tools("financial_agent")
            # ['transfer_funds', 'check_balance', 'delete_transaction']

            # Validate approval patterns
            rules = ApprovalRules(patterns=["transfer_*"])
            print(rules.preview_matches(tools))
            # {'transfer_*': ['transfer_funds']}
            ```
        """
        self._ensure_started()

        # Validate agent exists
        self.get_agent(agent_name)

        if not self._tool_registry:
            return []

        return self._tool_registry.get_tool_names(agent_name)

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
        approval_rules: "ApprovalRules | None" = None,
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
            approval_rules: Optional rules for human-in-the-loop approval.
                When provided, tool calls matching the patterns will pause
                for human approval before execution.

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

        Example (with approval rules):
            ```python
            from zap_ai import ApprovalRules

            task = await zap.execute_task(
                agent_name="FinancialAgent",
                task="Transfer $50,000 to vendor",
                approval_rules=ApprovalRules(patterns=["transfer_*", "delete_*"]),
            )
            # Later, check for pending approvals
            pending = await task.get_pending_approvals()
            await task.approve(pending[0].id)
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

        # Validate approval rules if provided
        if approval_rules:
            tool_names = await self.get_agent_tools(agent_name)
            unmatched = approval_rules.get_unmatched_patterns(tool_names)
            if unmatched:
                warnings.warn(
                    f"Approval patterns don't match any tools: {unmatched}. "
                    f"Available tools: {tool_names}",
                    UserWarning,
                    stacklevel=2,
                )

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
                approval_rules=approval_rules.to_dict() if approval_rules else None,
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

    def _create_task_fetcher(self) -> "Callable[[str], Awaitable[Task]]":
        """Create a task fetcher callback bound to this Zap instance."""

        async def fetcher(task_id: str) -> Task:
            return await self.get_task(task_id)

        return fetcher

    def _create_approval_fetcher(
        self, task_id: str
    ) -> "Callable[[], Awaitable[list[ApprovalRequest]]]":
        """Create an approval fetcher callback for a specific task."""
        from zap_ai.workflows.models import ApprovalRequest

        async def fetcher() -> "list[ApprovalRequest]":
            from zap_ai.workflows.agent_workflow import AgentWorkflow

            handle = self.temporal_client.get_workflow_handle(task_id)  # type: ignore[union-attr]
            pending_dicts = await handle.query(AgentWorkflow.get_pending_approvals)
            return [ApprovalRequest.from_dict(d) for d in pending_dicts]

        return fetcher

    def _create_approval_sender(
        self, task_id: str
    ) -> "Callable[[str, bool, str | None], Awaitable[None]]":
        """Create an approval sender callback for a specific task."""

        async def sender(approval_id: str, approved: bool, reason: str | None) -> None:
            from zap_ai.workflows.agent_workflow import AgentWorkflow

            handle = self.temporal_client.get_workflow_handle(task_id)  # type: ignore[union-attr]
            await handle.signal(
                AgentWorkflow.approve_execution, args=[approval_id, approved, reason]
            )

        return sender

    async def get_task(self, task_id: str) -> Task:
        """
        Get the current state of a task.

        Queries the Temporal workflow for current status, result,
        conversation history, and sub-task information.

        The returned Task object includes:
        - Full conversation history via `task.history`
        - Sub-task IDs via `task.sub_tasks`
        - Ability to fetch sub-task Task objects via `await task.get_sub_tasks()`
        - Convenience methods: `get_text_content()`, `get_tool_calls()`,
          `get_turn()`, `get_turns()`, `turn_count()`

        Args:
            task_id: The task ID returned from execute_task().

        Returns:
            Task object with current state and conversation access.

        Raises:
            ZapNotStartedError: If start() hasn't been called.
            TaskNotFoundError: If no task with that ID exists.

        Example:
            ```python
            task = await zap.get_task(task_id)
            print(f"Status: {task.status}")
            if task.is_complete():
                print(f"Result: {task.result}")

            # Access conversation
            print(task.get_text_content())
            for tool_call in task.get_tool_calls():
                print(f"Called: {tool_call.name}")

            # Access sub-tasks
            sub_tasks = await task.get_sub_tasks()
            for sub in sub_tasks:
                print(f"Sub-task: {sub.id}")
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

            # Query sub-agent conversations to get sub-task IDs
            sub_agent_convs = await handle.query(AgentWorkflow.get_sub_agent_conversations)
            sub_task_ids = [conv["conversation_id"] for conv in sub_agent_convs.values()]

            # Parse agent name from task ID
            agent_name = task_id.split("-")[0]

            return Task(
                id=task_id,
                agent_name=agent_name,
                status=TaskStatus(status_str),
                result=result,
                error=error,
                history=history,
                sub_tasks=sub_task_ids,
                _task_fetcher=self._create_task_fetcher(),
                _approval_fetcher=self._create_approval_fetcher(task_id),
                _approval_sender=self._create_approval_sender(task_id),
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
