# Phase 4: Temporal Workflows

**Goal:** Implement the core agent workflow with entity pattern.

**Dependencies:** Phase 1 (Foundation), Phase 2 (MCP Integration), Phase 3 (LLM Integration).

---

## Task 4.1: Implement Workflow Models

**File:** `src/zap_ai/workflows/models.py`

Create dataclasses for workflow input/output and state management:

```python
"""Workflow models for agent execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentWorkflowInput:
    """
    Input for starting an agent workflow.

    Attributes:
        agent_name: Name of the agent to run.
        initial_task: The initial task/message from the user.
        state: Optional serialized state for continue-as-new.
        parent_workflow_id: If this is a child workflow, the parent's ID.
    """
    agent_name: str
    initial_task: str
    state: dict[str, Any] | None = None
    parent_workflow_id: str | None = None


@dataclass
class SubAgentConversation:
    """
    Tracks a conversation with a sub-agent.

    Attributes:
        conversation_id: Unique ID for this conversation (child workflow ID).
        agent_name: Name of the sub-agent.
        messages: History of messages in this conversation.
        is_active: Whether the child workflow is still running.
    """
    conversation_id: str
    agent_name: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    is_active: bool = True


@dataclass
class SubAgentResponse:
    """
    Response from a sub-agent conversation.

    Attributes:
        conversation_id: ID of the conversation.
        agent_name: Name of the sub-agent.
        response: The sub-agent's response text.
        is_complete: Whether the conversation has ended.
    """
    conversation_id: str
    agent_name: str
    response: str
    is_complete: bool = False

    def to_tool_result(self) -> str:
        """Format as a tool result string."""
        import json
        return json.dumps({
            "conversation_id": self.conversation_id,
            "agent_name": self.agent_name,
            "response": self.response,
            "is_complete": self.is_complete,
        })


@dataclass
class ConversationState:
    """
    Serializable state for continue-as-new.

    Attributes:
        messages: Full conversation history.
        iteration_count: Number of agentic loop iterations completed.
        pending_messages: Messages received via signal while processing.
        sub_agent_conversations: Active sub-agent conversations.
    """
    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    pending_messages: list[str] = field(default_factory=list)
    sub_agent_conversations: dict[str, SubAgentConversation] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for continue-as-new."""
        return {
            "messages": self.messages,
            "iteration_count": self.iteration_count,
            "pending_messages": self.pending_messages,
            "sub_agent_conversations": {
                k: {
                    "conversation_id": v.conversation_id,
                    "agent_name": v.agent_name,
                    "messages": v.messages,
                    "is_active": v.is_active,
                }
                for k, v in self.sub_agent_conversations.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationState":
        """Deserialize from dict."""
        sub_convs = {}
        for k, v in data.get("sub_agent_conversations", {}).items():
            sub_convs[k] = SubAgentConversation(
                conversation_id=v["conversation_id"],
                agent_name=v["agent_name"],
                messages=v["messages"],
                is_active=v["is_active"],
            )
        return cls(
            messages=data.get("messages", []),
            iteration_count=data.get("iteration_count", 0),
            pending_messages=data.get("pending_messages", []),
            sub_agent_conversations=sub_convs,
        )
```

**Checklist:**
- [ ] Create `src/zap_ai/workflows/models.py`
- [ ] Implement `AgentWorkflowInput` dataclass
- [ ] Implement `SubAgentConversation` dataclass
- [ ] Implement `SubAgentResponse` dataclass with `to_tool_result()`
- [ ] Implement `ConversationState` with serialization methods

---

## Task 4.2: Implement AgentWorkflow Base Structure

**File:** `src/zap_ai/workflows/agent_workflow.py`

Create the main workflow class with entity pattern:

```python
"""Agent workflow using Temporal entity pattern."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import uuid4

from temporalio import workflow
from temporalio.common import RetryPolicy

from zap_ai.core.task import TaskStatus
from zap_ai.workflows.models import (
    AgentWorkflowInput,
    ConversationState,
    SubAgentConversation,
    SubAgentResponse,
)

# Import activities
with workflow.unsafe.imports_passed_through():
    from zap_ai.activities.inference import inference_activity, InferenceInput, InferenceOutput
    from zap_ai.activities.tool_execution import tool_execution_activity, ToolExecutionInput


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

        # Pending messages from signals
        self._pending_messages: list[str] = []

    # ... (queries, signals, and run method in following tasks)
```

**Checklist:**
- [ ] Create `src/zap_ai/workflows/agent_workflow.py`
- [ ] Implement `@workflow.defn` class with `__init__`
- [ ] Define instance variables for state tracking
- [ ] Import activities with `workflow.unsafe.imports_passed_through()`

---

## Task 4.3: Implement Queries

Add query methods to `AgentWorkflow`:

```python
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
```

**Checklist:**
- [ ] Implement `get_status()` query
- [ ] Implement `get_result()` query
- [ ] Implement `get_error()` query
- [ ] Implement `get_history()` query
- [ ] Implement `get_sub_agent_conversations()` query

---

## Task 4.4: Implement Signals

Add signal handlers to `AgentWorkflow`:

```python
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
```

**Checklist:**
- [ ] Implement `add_message()` signal for task follow-ups
- [ ] Implement `sub_agent_message()` signal for child workflows

---

## Task 4.5: Implement the Agentic Loop

Add the main `run()` method to `AgentWorkflow`:

```python
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

        if input.state:
            self._state = ConversationState.from_dict(input.state)
        else:
            # Fresh start - add system prompt and initial task
            # Note: Agent config (model, prompt, tools) must be passed through
            # activity or injected via workflow context
            self._state.messages.append({
                "role": "system",
                "content": self._system_prompt,
            })
            self._state.messages.append({
                "role": "user",
                "content": input.initial_task,
            })

        self._status = TaskStatus.RUNNING

        # Main agentic loop
        while self._state.iteration_count < self._max_iterations:
            # Check for continue-as-new
            if workflow.info().is_continue_as_new_suggested():
                await workflow.wait_condition(workflow.all_handlers_finished)
                workflow.continue_as_new(AgentWorkflowInput(
                    agent_name=self._agent_name,
                    initial_task="",  # Not used for continue-as-new
                    state=self._state.to_dict(),
                ))

            # Process any pending messages
            if self._pending_messages:
                for msg in self._pending_messages:
                    self._state.messages.append({"role": "user", "content": msg})
                self._pending_messages.clear()

            # Run inference
            self._status = TaskStatus.RUNNING
            inference_result = await self._run_inference()

            # Add assistant response to history
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if inference_result.content:
                assistant_msg["content"] = inference_result.content
            if inference_result.tool_calls:
                assistant_msg["tool_calls"] = inference_result.tool_calls
            self._state.messages.append(assistant_msg)

            # Check if we're done (no tool calls)
            if not inference_result.tool_calls:
                # Wait for follow-up or timeout
                self._result = inference_result.content
                self._status = TaskStatus.COMPLETED

                # Wait for potential follow-up messages
                try:
                    await workflow.wait_condition(
                        lambda: len(self._pending_messages) > 0,
                        timeout=timedelta(minutes=5),
                    )
                    # Got a follow-up, continue the loop
                    self._status = TaskStatus.RUNNING
                    continue
                except TimeoutError:
                    # No follow-up, we're done
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

    async def _run_inference(self) -> InferenceOutput:
        """Run LLM inference activity."""
        return await workflow.execute_activity(
            inference_activity,
            InferenceInput(
                agent_name=self._agent_name,
                model=self._model,
                messages=self._state.messages,
                tools=self._tools,
            ),
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=30),
                maximum_attempts=5,
            ),
        )
```

**Checklist:**
- [ ] Implement `run()` method with `@workflow.run` decorator
- [ ] Initialize or restore state from input
- [ ] Implement main agentic loop
- [ ] Check for continue-as-new condition
- [ ] Process pending messages
- [ ] Handle completion with follow-up wait
- [ ] Handle max iterations limit
- [ ] Implement `_run_inference()` helper

---

## Task 4.6: Implement Parallel Tool Execution

Add tool execution handling to `AgentWorkflow`:

```python
    async def _handle_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """
        Execute tool calls in parallel and add results to history.

        Args:
            tool_calls: List of tool call dicts from LLM response.
        """
        import asyncio

        # Separate message_agent calls from regular MCP tools
        mcp_calls = []
        message_agent_calls = []

        for tc in tool_calls:
            func = tc.get("function", {})
            if func.get("name") == "message_agent":
                message_agent_calls.append(tc)
            else:
                mcp_calls.append(tc)

        # Execute MCP tools in parallel
        if mcp_calls:
            tasks = [
                self._execute_mcp_tool(tc)
                for tc in mcp_calls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for tc, result in zip(mcp_calls, results):
                tool_call_id = tc.get("id", "")
                func = tc.get("function", {})
                tool_name = func.get("name", "unknown")

                if isinstance(result, Exception):
                    content = f"Error: {result}"
                else:
                    content = result

                self._state.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": content,
                })

        # Handle message_agent calls (may involve child workflows)
        for tc in message_agent_calls:
            result = await self._handle_message_agent(tc)

            self._state.messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "name": "message_agent",
                "content": result.to_tool_result(),
            })

    async def _execute_mcp_tool(self, tool_call: dict[str, Any]) -> str:
        """Execute a single MCP tool via activity."""
        import json

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
            ),
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=RetryPolicy(
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
                maximum_attempts=3,
            ),
        )
```

**Checklist:**
- [ ] Implement `_handle_tool_calls()` method
- [ ] Separate message_agent from MCP tool calls
- [ ] Use `asyncio.gather()` for parallel MCP tool execution
- [ ] Handle tool execution errors gracefully
- [ ] Add tool results to message history
- [ ] Implement `_execute_mcp_tool()` helper

---

## Task 4.7: Implement Continue-as-New

The continue-as-new logic is already included in the `run()` method. Verify:

```python
    # Check for continue-as-new
    if workflow.info().is_continue_as_new_suggested():
        await workflow.wait_condition(workflow.all_handlers_finished)
        workflow.continue_as_new(AgentWorkflowInput(
            agent_name=self._agent_name,
            initial_task="",  # Not used for continue-as-new
            state=self._state.to_dict(),
        ))
```

**Checklist:**
- [ ] Check `workflow.info().is_continue_as_new_suggested()`
- [ ] Wait for handlers with `workflow.all_handlers_finished()`
- [ ] Call `workflow.continue_as_new()` with serialized state

---

## Task 4.8: Implement Sub-Agent Messaging

Add sub-agent messaging handler to `AgentWorkflow`:

```python
    async def _handle_message_agent(self, tool_call: dict[str, Any]) -> SubAgentResponse:
        """
        Handle message_agent tool call for sub-agent conversations.

        Args:
            tool_call: The message_agent tool call from LLM.

        Returns:
            SubAgentResponse with conversation_id and response.
        """
        import json

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
            return await self._continue_sub_agent_conversation(
                conversation_id, message
            )
        elif agent_name:
            # Start new conversation
            return await self._start_sub_agent_conversation(
                agent_name, message
            )
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
        # Generate conversation ID (child workflow ID)
        parent_id = workflow.info().workflow_id
        short_uuid = uuid4().hex[:8]
        conversation_id = f"{parent_id}-{agent_name}-{short_uuid}"

        # Track the conversation
        self._state.sub_agent_conversations[conversation_id] = SubAgentConversation(
            conversation_id=conversation_id,
            agent_name=agent_name,
            messages=[{"role": "user", "content": message}],
            is_active=True,
        )

        self._status = TaskStatus.DELEGATED

        # Start child workflow
        child_handle = await workflow.start_child_workflow(
            AgentWorkflow.run,
            AgentWorkflowInput(
                agent_name=agent_name,
                initial_task=message,
                parent_workflow_id=parent_id,
            ),
            id=conversation_id,
        )

        # Wait for child to complete first response
        result = await child_handle.result()

        # Update conversation tracking
        conv = self._state.sub_agent_conversations[conversation_id]
        conv.messages.append({"role": "assistant", "content": result})

        self._status = TaskStatus.RUNNING

        return SubAgentResponse(
            conversation_id=conversation_id,
            agent_name=agent_name,
            response=result,
            is_complete=False,  # Child is waiting for follow-up
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

        self._status = TaskStatus.DELEGATED

        # Get handle to existing child workflow and send signal
        child_handle = workflow.get_external_workflow_handle(conversation_id)
        await child_handle.signal(AgentWorkflow.sub_agent_message, message)

        # Wait for response via query polling (child will update its result)
        # Note: In practice, you might use a more sophisticated pattern
        # For now, we wait a bit and query
        await workflow.sleep(timedelta(seconds=1))

        try:
            result = await child_handle.query(AgentWorkflow.get_result)
        except Exception:
            result = "Error: Failed to get response from sub-agent"

        conv.messages.append({"role": "assistant", "content": result or ""})

        self._status = TaskStatus.RUNNING

        return SubAgentResponse(
            conversation_id=conversation_id,
            agent_name=conv.agent_name,
            response=result or "",
            is_complete=False,
        )
```

**Checklist:**
- [ ] Implement `_handle_message_agent()` to dispatch based on conversation_id vs agent_name
- [ ] Implement `_start_sub_agent_conversation()` to start child workflow
- [ ] Generate conversation_id as child workflow ID
- [ ] Track conversations in `_state.sub_agent_conversations`
- [ ] Implement `_continue_sub_agent_conversation()` to send signal to existing child
- [ ] Update status to DELEGATED during sub-agent calls

---

## Task 4.9: Implement Child Workflow Signal/Response Pattern

The child workflow pattern is handled by the same `AgentWorkflow` class:

1. Child receives initial message via `run()` input
2. Child processes and returns response
3. Child waits for follow-up via `sub_agent_message` signal
4. On timeout, child completes gracefully

The wait logic is already in the `run()` method:

```python
    # Wait for potential follow-up messages
    try:
        await workflow.wait_condition(
            lambda: len(self._pending_messages) > 0,
            timeout=timedelta(minutes=5),
        )
        # Got a follow-up, continue the loop
        self._status = TaskStatus.RUNNING
        continue
    except TimeoutError:
        # No follow-up, we're done
        break
```

**Checklist:**
- [ ] Child workflow receives initial task in `run()` input
- [ ] Child processes and sets result
- [ ] Child waits for follow-up with `workflow.wait_condition()`
- [ ] On signal: processes new message, continues loop
- [ ] On timeout: workflow completes gracefully

---

## Phase 4 Verification

After completing all tasks, verify:

1. **Workflow starts and completes:**
   ```python
   async with await WorkflowEnvironment.start_time_skipping() as env:
       async with Worker(
           env.client,
           task_queue="test",
           workflows=[AgentWorkflow],
           activities=[inference_activity, tool_execution_activity],
       ):
           result = await env.client.execute_workflow(
               AgentWorkflow.run,
               AgentWorkflowInput(agent_name="Test", initial_task="Hello"),
               id="test-workflow",
               task_queue="test",
           )
           assert result is not None
   ```

2. **Queries work:**
   ```python
   handle = await env.client.start_workflow(...)
   status = await handle.query(AgentWorkflow.get_status)
   assert status in ["pending", "running", "completed"]
   ```

3. **Signals work:**
   ```python
   await handle.signal(AgentWorkflow.add_message, "Follow up question")
   ```

4. **Sub-agent conversations work:**
   ```python
   # Parent calls message_agent tool
   # Child workflow is started
   # Parent receives response with conversation_id
   # Parent can continue conversation
   ```
