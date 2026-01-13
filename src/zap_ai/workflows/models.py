"""Workflow models for agent execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fnmatch import fnmatch
from typing import Any


@dataclass
class ApprovalRequest:
    """
    Represents a pending approval request for a tool call.

    Attributes:
        id: Unique identifier for this approval request.
        tool_name: Name of the tool requiring approval.
        tool_args: Arguments passed to the tool.
        requested_at: When the approval was requested.
        timeout_at: When the approval will timeout.
        context: Additional context (agent_name, conversation snippet, etc.).
    """

    id: str
    tool_name: str
    tool_args: dict[str, Any]
    requested_at: datetime
    timeout_at: datetime
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for Temporal."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "requested_at": self.requested_at.isoformat(),
            "timeout_at": self.timeout_at.isoformat(),
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalRequest:
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            tool_name=data["tool_name"],
            tool_args=data["tool_args"],
            requested_at=datetime.fromisoformat(data["requested_at"]),
            timeout_at=datetime.fromisoformat(data["timeout_at"]),
            context=data.get("context", {}),
        )


@dataclass
class ApprovalDecision:
    """
    Records a decision on an approval request.

    Attributes:
        approved: Whether the request was approved.
        reason: Optional reason for rejection.
        decided_at: When the decision was made.
    """

    approved: bool
    reason: str | None
    decided_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for Temporal."""
        return {
            "approved": self.approved,
            "reason": self.reason,
            "decided_at": self.decided_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalDecision:
        """Deserialize from dict."""
        return cls(
            approved=data["approved"],
            reason=data.get("reason"),
            decided_at=datetime.fromisoformat(data["decided_at"]),
        )


@dataclass
class ApprovalRules:
    """
    Configuration for human-in-the-loop approval requirements.

    Attributes:
        patterns: Glob patterns for tool names requiring approval (e.g., "transfer_*", "delete_*").
        timeout: How long to wait for approval before auto-rejecting.
    """

    patterns: list[str]
    timeout: timedelta = field(default_factory=lambda: timedelta(days=7))

    def matches(self, tool_name: str) -> bool:
        """Check if a tool name matches any approval pattern."""
        return any(fnmatch(tool_name, pattern) for pattern in self.patterns)

    def preview_matches(self, tool_names: list[str]) -> dict[str, list[str]]:
        """
        Show which tools each pattern would match.

        Args:
            tool_names: List of available tool names.

        Returns:
            Dict mapping each pattern to the list of tools it matches.
        """
        return {
            pattern: [t for t in tool_names if fnmatch(t, pattern)] for pattern in self.patterns
        }

    def get_unmatched_patterns(self, tool_names: list[str]) -> list[str]:
        """
        Return patterns that don't match any tools (potential typos).

        Args:
            tool_names: List of available tool names.

        Returns:
            List of patterns that match no tools.
        """
        matches = self.preview_matches(tool_names)
        return [p for p, tools in matches.items() if not tools]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for Temporal."""
        return {
            "patterns": self.patterns,
            "timeout_seconds": self.timeout.total_seconds(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalRules:
        """Deserialize from dict."""
        return cls(
            patterns=data["patterns"],
            timeout=timedelta(seconds=data["timeout_seconds"]),
        )


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
        approval_rules: Optional rules for human-in-the-loop approval.
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
    approval_rules: dict[str, Any] | None = None  # Serialized ApprovalRules


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
        pending_approvals: Tool calls awaiting human approval.
        approval_decisions: Decisions made on approval requests.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    iteration_count: int = 0
    pending_messages: list[str] = field(default_factory=list)
    sub_agent_conversations: dict[str, SubAgentConversation] = field(default_factory=dict)
    trace_context: dict[str, Any] | None = None
    pending_approvals: dict[str, ApprovalRequest] = field(default_factory=dict)
    approval_decisions: dict[str, ApprovalDecision] = field(default_factory=dict)

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
            "pending_approvals": {k: v.to_dict() for k, v in self.pending_approvals.items()},
            "approval_decisions": {k: v.to_dict() for k, v in self.approval_decisions.items()},
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

        pending_approvals: dict[str, ApprovalRequest] = {}
        for k, v in data.get("pending_approvals", {}).items():
            pending_approvals[k] = ApprovalRequest.from_dict(v)

        approval_decisions: dict[str, ApprovalDecision] = {}
        for k, v in data.get("approval_decisions", {}).items():
            approval_decisions[k] = ApprovalDecision.from_dict(v)

        return cls(
            messages=data.get("messages", []),
            iteration_count=data.get("iteration_count", 0),
            pending_messages=data.get("pending_messages", []),
            sub_agent_conversations=sub_convs,
            trace_context=data.get("trace_context"),
            pending_approvals=pending_approvals,
            approval_decisions=approval_decisions,
        )
