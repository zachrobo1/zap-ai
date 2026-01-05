"""Workflow models for agent execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentWorkflowInput:
    """
    Input for starting an agent workflow.

    Attributes:
        agent_name: Name of the agent to run.
        initial_task: The initial task/message from the user.
        system_prompt: The agent's system prompt.
        model: LLM model identifier (default: gpt-4o).
        tools: List of tool definitions available to the agent.
        max_iterations: Maximum agentic loop iterations (default: 50).
        state: Optional serialized state for continue-as-new.
        parent_workflow_id: If this is a child workflow, the parent's ID.
        parent_trace_context: Parent trace context for sub-agent linking.
    """

    agent_name: str
    initial_task: str
    system_prompt: str = ""
    model: str = "gpt-4o"
    tools: list[dict[str, Any]] = field(default_factory=list)
    max_iterations: int = 50
    state: dict[str, Any] | None = None
    parent_workflow_id: str | None = None
    parent_trace_context: dict[str, Any] | None = None


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
        return json.dumps(
            {
                "conversation_id": self.conversation_id,
                "agent_name": self.agent_name,
                "response": self.response,
                "is_complete": self.is_complete,
            }
        )


@dataclass
class ConversationState:
    """
    Serializable state for continue-as-new.

    Attributes:
        messages: Full conversation history.
        iteration_count: Number of agentic loop iterations completed.
        pending_messages: Messages received via signal while processing.
        sub_agent_conversations: Active sub-agent conversations.
        trace_context: Trace context for continue-as-new preservation.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    pending_messages: list[str] = field(default_factory=list)
    sub_agent_conversations: dict[str, SubAgentConversation] = field(default_factory=dict)
    trace_context: dict[str, Any] | None = None

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
            "trace_context": self.trace_context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationState:
        """Deserialize from dict."""
        sub_convs: dict[str, SubAgentConversation] = {}
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
            trace_context=data.get("trace_context"),
        )
