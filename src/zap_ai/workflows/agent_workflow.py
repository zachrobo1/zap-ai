"""Agent workflow using Temporal entity pattern."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

from zap_ai.core.task import TaskStatus
from zap_ai.workflows.models import (
    AgentWorkflowInput,
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRules,
    ConversationState,
    SubAgentConversation,
    SubAgentResponse,
)

# Import activities with workflow-safe imports
with workflow.unsafe.imports_passed_through():
    from zap_ai.activities.inference import (
        InferenceInput,
        InferenceOutput,
        inference_activity,
    )
    from zap_ai.activities.tool_execution import (
        AgentConfigOutput,
        ToolExecutionInput,
        get_agent_config_activity,
        tool_execution_activity,
    )
    from zap_ai.tracing import (
        TraceContext,
    )


@workflow.defn
class AgentWorkflow:
    """
    Main agent workflow implementing the agentic loop.

    Uses Temporal's entity pattern for long-running conversations
    with signal handling for follow-up messages.
    """

    def __init__(self) -> None:
        # Status tracking
        self._status: TaskStatus = TaskStatus.PENDING
        self._result: str | None = None
        self._error: str | None = None

        # Conversation state
        self._state: ConversationState = ConversationState()

        # Agent configuration (set in run())
        self._agent_name: str = ""
        self._model: str = "gpt-4o"
        self._system_prompt: str = ""
        self._max_iterations: int = 50
        self._tools: list[dict[str, Any]] = []

        # Approval rules (set in run())
        self._approval_rules: ApprovalRules | None = None

        # Pending messages from signals
        self._pending_messages: list[str] = []

        # Tracing context (set in run())
        self._trace_context: TraceContext | None = None

    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------

    @workflow.query
    def get_status(self) -> str:
        """Query current task status."""
        return self._status.value

    @workflow.query
    def get_result(self) -> str | None:
        """Query task result (None if not complete)."""
        return self._result

    @workflow.query
    def get_error(self) -> str | None:
        """Query error message (None if no error)."""
        return self._error

    @workflow.query
    def get_history(self) -> list[dict[str, Any]]:
        """Query conversation history."""
        return self._state.messages

    @workflow.query
    def get_iteration_count(self) -> int:
        """Query current iteration count."""
        return self._state.iteration_count

    @workflow.query
    def get_sub_agent_conversations(self) -> dict[str, dict[str, Any]]:
        """Query active sub-agent conversations."""
        return {
            k: {
                "conversation_id": v.conversation_id,
                "agent_name": v.agent_name,
                "message_count": len(v.messages),
                "is_active": v.is_active,
            }
            for k, v in self._state.sub_agent_conversations.items()
        }

    @workflow.query
    def get_pending_approvals(self) -> list[dict[str, Any]]:
        """Query pending approval requests."""
        return [req.to_dict() for req in self._state.pending_approvals.values()]

    # -------------------------------------------------------------------------
    # Signals
    # -------------------------------------------------------------------------

    @workflow.signal
    async def add_message(self, message: str) -> None:
        """
        Signal to add a follow-up message.

        Used for task follow-ups from the parent Zap instance.
        """
        self._pending_messages.append(message)

    @workflow.signal
    async def sub_agent_message(self, message: str) -> None:
        """
        Signal to receive a message from parent (for child workflows).

        Used when this workflow is a sub-agent receiving follow-up
        messages from its parent.
        """
        self._pending_messages.append(message)

    @workflow.signal
    async def approve_execution(
        self, approval_id: str, approved: bool, reason: str | None = None
    ) -> None:
        """
        Signal to approve or reject a pending tool execution.

        Args:
            approval_id: ID of the approval request.
            approved: Whether to approve (True) or reject (False).
            reason: Optional reason for rejection.
        """
        if approval_id not in self._state.pending_approvals:
            return  # Unknown approval ID, ignore

        # Record the decision
        self._state.approval_decisions[approval_id] = ApprovalDecision(
            approved=approved,
            reason=reason,
            decided_at=workflow.now(),
        )

        # Remove from pending
        self._state.pending_approvals.pop(approval_id, None)

    # -------------------------------------------------------------------------
    # Main Run Method
    # -------------------------------------------------------------------------

    @workflow.run
    async def run(self, input: AgentWorkflowInput) -> str:
        """
        Main workflow entry point implementing the agentic loop.

        Args:
            input: Workflow input with agent name and initial task.

        Returns:
            Final result string.
        """
        # Initialize or restore state
        self._agent_name = input.agent_name
        self._system_prompt = input.system_prompt
        self._model = input.model
        self._tools = input.tools
        self._max_iterations = input.max_iterations
        self._approval_rules = (
            ApprovalRules.from_dict(input.approval_rules) if input.approval_rules else None
        )

        if input.state:
            self._state = ConversationState.from_dict(input.state)
            # Restore pending messages if any
            self._pending_messages = list(self._state.pending_messages)
            # Restore trace context if any
            if self._state.trace_context:
                self._trace_context = TraceContext.from_dict(self._state.trace_context)
        else:
            # Fresh start - add system prompt and initial task
            self._state.messages.append(
                {
                    "role": "system",
                    "content": self._system_prompt or f"You are agent {input.agent_name}.",
                }
            )
            self._state.messages.append(
                {
                    "role": "user",
                    "content": input.initial_task,
                }
            )

        # Initialize trace context for new tasks
        if not self._trace_context:
            # Check if this is a sub-agent with parent trace context
            if input.parent_trace_context:
                self._trace_context = TraceContext.from_dict(input.parent_trace_context)
            else:
                # Create new trace context with W3C format IDs
                # trace_id: 32 lowercase hex chars, span_id: 16 lowercase hex chars
                workflow_id = workflow.info().workflow_id
                import hashlib

                # Generate a deterministic trace ID from workflow ID (32 hex chars)
                trace_id = hashlib.sha256(workflow_id.encode()).hexdigest()[:32]
                span_id = hashlib.sha256((workflow_id + "-span").encode()).hexdigest()[:16]
                self._trace_context = TraceContext(
                    trace_id=trace_id,
                    span_id=span_id,
                    provider_data={"workflow_id": workflow_id},
                )

        # Store trace context in state for continue-as-new
        self._state.trace_context = self._trace_context.to_dict()

        self._status = TaskStatus.THINKING

        # Main agentic loop
        while self._state.iteration_count < self._max_iterations:
            # Check for continue-as-new
            if workflow.info().is_continue_as_new_suggested():
                # Save pending messages to state before continue-as-new
                self._state.pending_messages = list(self._pending_messages)
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(
                    AgentWorkflowInput(
                        agent_name=self._agent_name,
                        initial_task="",  # Not used for continue-as-new
                        system_prompt=self._system_prompt,
                        model=self._model,
                        tools=self._tools,
                        max_iterations=self._max_iterations,
                        state=self._state.to_dict(),
                        parent_workflow_id=input.parent_workflow_id,
                    )
                )

            # Process any pending messages
            if self._pending_messages:
                for msg in self._pending_messages:
                    self._state.messages.append({"role": "user", "content": msg})
                self._pending_messages.clear()

            # Run inference
            self._status = TaskStatus.THINKING
            try:
                inference_result = await self._run_inference()
            except Exception as e:
                self._error = f"Inference failed: {e}"
                self._status = TaskStatus.FAILED
                return ""

            # Add assistant response to history
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if inference_result.content:
                assistant_msg["content"] = inference_result.content
            if inference_result.tool_calls:
                assistant_msg["tool_calls"] = inference_result.tool_calls
            self._state.messages.append(assistant_msg)

            # Check if we're done (no tool calls)
            if not inference_result.tool_calls:
                self._result = inference_result.content
                self._status = TaskStatus.COMPLETED
                break

            # Handle tool calls
            self._status = TaskStatus.AWAITING_TOOL
            await self._handle_tool_calls(inference_result.tool_calls)

            self._state.iteration_count += 1

        # Max iterations reached
        if self._state.iteration_count >= self._max_iterations:
            self._error = f"Max iterations ({self._max_iterations}) reached"
            self._status = TaskStatus.FAILED

        return self._result or ""

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    async def _run_inference(self) -> InferenceOutput:
        """Run LLM inference activity."""
        return await workflow.execute_activity(
            inference_activity,
            InferenceInput(
                agent_name=self._agent_name,
                model=self._model,
                messages=self._state.messages,
                tools=self._tools,
                trace_context=self._trace_context.to_dict() if self._trace_context else None,
            ),
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=5,
            ),
        )

    # -------------------------------------------------------------------------
    # Tool Execution
    # -------------------------------------------------------------------------

    async def _handle_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """
        Execute all tool calls and add results to history.

        When approval rules are active, tools requiring approval are processed
        sequentially, waiting for human approval before execution.

        Both MCP tools and message_agent calls run concurrently via asyncio.gather
        when no approvals are pending.

        Args:
            tool_calls: List of tool call dicts from LLM response.
        """
        if not tool_calls:
            return

        # If no approval rules, use fast parallel path
        if not self._approval_rules:
            await self._execute_tool_calls_parallel(tool_calls)
            return

        # With approval rules, process tools that need approval sequentially
        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")
            tool_call_id = tc.get("id", "")

            # Check if this tool requires approval
            if self._requires_approval(tool_name):
                result = await self._execute_with_approval(tc)
            elif tool_name == "message_agent":
                result = await self._handle_message_agent(tc)
                result = result.to_tool_result()
            else:
                result = await self._execute_mcp_tool(tc)

            self._state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result if not isinstance(result, Exception) else f"Error: {result}",
                }
            )

    async def _execute_tool_calls_parallel(self, tool_calls: list[dict[str, Any]]) -> None:
        """Execute all tool calls in parallel (no approval checks)."""
        tasks: list = []
        for tc in tool_calls:
            func = tc.get("function", {})
            if func.get("name") == "message_agent":
                tasks.append(self._handle_message_agent(tc))
            else:
                tasks.append(self._execute_mcp_tool(tc))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for tc, result in zip(tool_calls, results):
            tool_call_id = tc.get("id", "")
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")

            if isinstance(result, Exception):
                content = f"Error: {result}"
            elif tool_name == "message_agent":
                content = result.to_tool_result()
            else:
                content = result

            self._state.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content,
                }
            )

    def _requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires human approval."""
        if not self._approval_rules:
            return False
        return self._approval_rules.matches(tool_name)

    async def _execute_with_approval(self, tool_call: dict[str, Any]) -> str:
        """
        Execute a tool call after obtaining human approval.

        Creates an approval request, waits for approval or timeout,
        then executes the tool if approved.

        Args:
            tool_call: The tool call requiring approval.

        Returns:
            Tool result string, or rejection/timeout message.
        """
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        args_raw = func.get("arguments", "{}")

        try:
            arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            arguments = {}

        # Create approval request (use workflow-safe deterministic methods)
        approval_id = str(workflow.uuid4())
        now = workflow.now()
        timeout = self._approval_rules.timeout if self._approval_rules else timedelta(days=7)

        request = ApprovalRequest(
            id=approval_id,
            tool_name=tool_name,
            tool_args=arguments,
            requested_at=now,
            timeout_at=now + timeout,
            context={
                "agent_name": self._agent_name,
                "workflow_id": workflow.info().workflow_id,
            },
        )

        # Add to pending approvals
        self._state.pending_approvals[approval_id] = request

        # Update status to awaiting approval
        self._status = TaskStatus.AWAITING_APPROVAL

        # Wait for approval decision or timeout
        try:
            await workflow.wait_condition(
                lambda: approval_id in self._state.approval_decisions,
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Timeout - auto-reject and continue
            self._state.pending_approvals.pop(approval_id, None)
            self._status = TaskStatus.AWAITING_TOOL
            return f"[Tool call rejected: approval timeout after {timeout}]"

        # Get the decision
        decision = self._state.approval_decisions.get(approval_id)

        # Restore status
        self._status = TaskStatus.AWAITING_TOOL

        if not decision or not decision.approved:
            reason = decision.reason if decision else "Unknown reason"
            return f"[Tool call rejected: {reason}]"

        # Approved - execute the tool
        if tool_name == "message_agent":
            result = await self._handle_message_agent(tool_call)
            return result.to_tool_result()
        else:
            return await self._execute_mcp_tool(tool_call)

    async def _execute_mcp_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a single MCP tool via activity."""
        func = tool_call.get("function", {})
        tool_name = func.get("name", "")
        args_raw = func.get("arguments", "{}")

        try:
            arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            arguments = {}

        return await workflow.execute_activity(
            tool_execution_activity,
            ToolExecutionInput(
                agent_name=self._agent_name,
                tool_name=tool_name,
                arguments=arguments,
                trace_context=self._trace_context.to_dict() if self._trace_context else None,
            ),
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=3,
            ),
        )

    # -------------------------------------------------------------------------
    # Sub-Agent Messaging
    # -------------------------------------------------------------------------

    async def _handle_message_agent(self, tool_call: dict[str, Any]) -> SubAgentResponse:
        """
        Handle message_agent tool call for sub-agent conversations.

        Args:
            tool_call: The message_agent tool call from LLM.

        Returns:
            SubAgentResponse with conversation_id and response.
        """
        func = tool_call.get("function", {})
        args_raw = func.get("arguments", "{}")

        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}

        message = args.get("message", "")
        conversation_id = args.get("conversation_id")
        agent_name = args.get("agent_name")

        if conversation_id:
            # Continue existing conversation
            return await self._continue_sub_agent_conversation(conversation_id, message)
        elif agent_name:
            # Start new conversation
            return await self._start_sub_agent_conversation(agent_name, message)
        else:
            return SubAgentResponse(
                conversation_id="",
                agent_name="",
                response="Error: Either conversation_id or agent_name is required",
                is_complete=True,
            )

    async def _start_sub_agent_conversation(
        self, agent_name: str, message: str
    ) -> SubAgentResponse:
        """Start a new conversation with a sub-agent."""
        # Get the sub-agent's configuration via activity
        agent_config: AgentConfigOutput = await workflow.execute_activity(
            get_agent_config_activity,
            agent_name,
            start_to_close_timeout=timedelta(seconds=30),
        )

        # Generate conversation ID (child workflow ID)
        parent_id = workflow.info().workflow_id
        short_uuid = workflow.uuid4().hex[:8]
        conversation_id = f"{parent_id}-{agent_name}-{short_uuid}"

        # Track the conversation
        self._state.sub_agent_conversations[conversation_id] = SubAgentConversation(
            conversation_id=conversation_id,
            agent_name=agent_name,
            messages=[{"role": "user", "content": message}],
            is_active=True,
        )

        # Start child workflow with full agent config and parent trace context
        child_handle = await workflow.start_child_workflow(
            AgentWorkflow.run,
            AgentWorkflowInput(
                agent_name=agent_name,
                initial_task=message,
                system_prompt=agent_config.prompt,
                model=agent_config.model,
                tools=agent_config.tools,
                max_iterations=agent_config.max_iterations,
                parent_workflow_id=parent_id,
                parent_trace_context=self._trace_context.to_dict() if self._trace_context else None,
            ),
            id=conversation_id,
        )

        # Wait for child workflow to complete (handle is directly awaitable)
        result = await child_handle

        # Update conversation tracking
        conv = self._state.sub_agent_conversations[conversation_id]
        conv.messages.append({"role": "assistant", "content": result})

        return SubAgentResponse(
            conversation_id=conversation_id,
            agent_name=agent_name,
            response=result,
            is_complete=False,  # Child may still be waiting for follow-up
        )

    async def _continue_sub_agent_conversation(
        self, conversation_id: str, message: str
    ) -> SubAgentResponse:
        """Continue an existing conversation with a sub-agent."""
        conv = self._state.sub_agent_conversations.get(conversation_id)

        if not conv:
            return SubAgentResponse(
                conversation_id=conversation_id,
                agent_name="",
                response=f"Error: Unknown conversation_id '{conversation_id}'",
                is_complete=True,
            )

        if not conv.is_active:
            return SubAgentResponse(
                conversation_id=conversation_id,
                agent_name=conv.agent_name,
                response="Error: Conversation has ended",
                is_complete=True,
            )

        # Add message to tracking
        conv.messages.append({"role": "user", "content": message})

        # Get handle to existing child workflow and send signal
        child_handle = workflow.get_external_workflow_handle(conversation_id)
        await child_handle.signal(AgentWorkflow.sub_agent_message, message)

        # Wait a short time and query for result
        # Note: In a production system, you might use a more sophisticated pattern
        await workflow.sleep(timedelta(seconds=1))

        try:
            result = await child_handle.query(AgentWorkflow.get_result)
        except Exception:
            result = "Error: Failed to get response from sub-agent"

        conv.messages.append({"role": "assistant", "content": result or ""})

        return SubAgentResponse(
            conversation_id=conversation_id,
            agent_name=conv.agent_name,
            response=result or "",
            is_complete=False,
        )
