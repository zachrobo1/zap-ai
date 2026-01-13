"""Tests for human-in-the-loop approval workflow functionality."""

from datetime import datetime, timedelta, timezone

from zap_ai.workflows.models import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRules,
    ConversationState,
)


class TestApprovalRules:
    """Tests for ApprovalRules dataclass."""

    def test_create_minimal_rules(self) -> None:
        """Test creating rules with just patterns."""
        rules = ApprovalRules(patterns=["transfer_*"])
        assert rules.patterns == ["transfer_*"]
        assert rules.timeout == timedelta(days=7)  # default

    def test_create_with_custom_timeout(self) -> None:
        """Test creating rules with custom timeout."""
        rules = ApprovalRules(
            patterns=["delete_*"],
            timeout=timedelta(hours=24),
        )
        assert rules.timeout == timedelta(hours=24)

    def test_matches_exact_tool_name(self) -> None:
        """Test exact match."""
        rules = ApprovalRules(patterns=["transfer_funds"])
        assert rules.matches("transfer_funds") is True
        assert rules.matches("delete_funds") is False

    def test_matches_wildcard_prefix(self) -> None:
        """Test glob pattern with wildcard prefix."""
        rules = ApprovalRules(patterns=["*_funds"])
        assert rules.matches("transfer_funds") is True
        assert rules.matches("delete_funds") is True
        assert rules.matches("funds") is False

    def test_matches_wildcard_suffix(self) -> None:
        """Test glob pattern with wildcard suffix."""
        rules = ApprovalRules(patterns=["delete_*"])
        assert rules.matches("delete_file") is True
        assert rules.matches("delete_user") is True
        assert rules.matches("remove_file") is False

    def test_matches_multiple_patterns(self) -> None:
        """Test matching against multiple patterns."""
        rules = ApprovalRules(patterns=["transfer_*", "delete_*", "send_email"])
        assert rules.matches("transfer_funds") is True
        assert rules.matches("delete_user") is True
        assert rules.matches("send_email") is True
        assert rules.matches("get_balance") is False

    def test_matches_question_mark_wildcard(self) -> None:
        """Test single character wildcard."""
        rules = ApprovalRules(patterns=["file_?"])
        assert rules.matches("file_a") is True
        assert rules.matches("file_1") is True
        assert rules.matches("file_ab") is False

    def test_preview_matches_single_pattern(self) -> None:
        """Test preview_matches with one pattern."""
        rules = ApprovalRules(patterns=["transfer_*"])
        tools = ["transfer_funds", "transfer_stock", "delete_user", "get_balance"]

        result = rules.preview_matches(tools)
        assert result == {"transfer_*": ["transfer_funds", "transfer_stock"]}

    def test_preview_matches_multiple_patterns(self) -> None:
        """Test preview_matches with multiple patterns."""
        rules = ApprovalRules(patterns=["transfer_*", "delete_*"])
        tools = ["transfer_funds", "delete_user", "get_balance"]

        result = rules.preview_matches(tools)
        assert result == {
            "transfer_*": ["transfer_funds"],
            "delete_*": ["delete_user"],
        }

    def test_preview_matches_no_matches(self) -> None:
        """Test preview_matches with pattern that matches nothing."""
        rules = ApprovalRules(patterns=["admin_*"])
        tools = ["transfer_funds", "delete_user"]

        result = rules.preview_matches(tools)
        assert result == {"admin_*": []}

    def test_get_unmatched_patterns_all_match(self) -> None:
        """Test get_unmatched_patterns when all patterns match."""
        rules = ApprovalRules(patterns=["transfer_*", "delete_*"])
        tools = ["transfer_funds", "delete_user"]

        unmatched = rules.get_unmatched_patterns(tools)
        assert unmatched == []

    def test_get_unmatched_patterns_some_unmatched(self) -> None:
        """Test get_unmatched_patterns with unmatched patterns."""
        rules = ApprovalRules(patterns=["transfer_*", "admin_*", "typo_tool"])
        tools = ["transfer_funds", "delete_user"]

        unmatched = rules.get_unmatched_patterns(tools)
        assert sorted(unmatched) == ["admin_*", "typo_tool"]

    def test_serialization_roundtrip(self) -> None:
        """Test to_dict and from_dict roundtrip."""
        rules = ApprovalRules(
            patterns=["transfer_*", "delete_*"],
            timeout=timedelta(hours=48),
        )

        data = rules.to_dict()
        restored = ApprovalRules.from_dict(data)

        assert restored.patterns == rules.patterns
        assert restored.timeout == rules.timeout


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_create_request(self) -> None:
        """Test creating an approval request."""
        now = datetime.now(timezone.utc)
        timeout_at = now + timedelta(days=7)

        request = ApprovalRequest(
            id="approval-123",
            tool_name="transfer_funds",
            tool_args={"amount": 50000, "to": "vendor"},
            requested_at=now,
            timeout_at=timeout_at,
            context={"agent_name": "financial_agent"},
        )

        assert request.id == "approval-123"
        assert request.tool_name == "transfer_funds"
        assert request.tool_args == {"amount": 50000, "to": "vendor"}
        assert request.requested_at == now
        assert request.timeout_at == timeout_at
        assert request.context == {"agent_name": "financial_agent"}

    def test_create_request_empty_context(self) -> None:
        """Test creating request with default empty context."""
        now = datetime.now(timezone.utc)

        request = ApprovalRequest(
            id="approval-123",
            tool_name="delete_file",
            tool_args={"path": "/tmp/file"},
            requested_at=now,
            timeout_at=now + timedelta(days=1),
        )

        assert request.context == {}

    def test_serialization_roundtrip(self) -> None:
        """Test to_dict and from_dict roundtrip."""
        now = datetime.now(timezone.utc)
        timeout_at = now + timedelta(days=7)

        request = ApprovalRequest(
            id="approval-456",
            tool_name="send_email",
            tool_args={"to": "user@example.com", "subject": "Test"},
            requested_at=now,
            timeout_at=timeout_at,
            context={"workflow_id": "task-123"},
        )

        data = request.to_dict()
        restored = ApprovalRequest.from_dict(data)

        assert restored.id == request.id
        assert restored.tool_name == request.tool_name
        assert restored.tool_args == request.tool_args
        assert restored.requested_at == request.requested_at
        assert restored.timeout_at == request.timeout_at
        assert restored.context == request.context


class TestApprovalDecision:
    """Tests for ApprovalDecision dataclass."""

    def test_create_approval(self) -> None:
        """Test creating an approval decision."""
        now = datetime.now(timezone.utc)

        decision = ApprovalDecision(
            approved=True,
            reason=None,
            decided_at=now,
        )

        assert decision.approved is True
        assert decision.reason is None
        assert decision.decided_at == now

    def test_create_rejection(self) -> None:
        """Test creating a rejection decision."""
        now = datetime.now(timezone.utc)

        decision = ApprovalDecision(
            approved=False,
            reason="Amount exceeds limit",
            decided_at=now,
        )

        assert decision.approved is False
        assert decision.reason == "Amount exceeds limit"

    def test_serialization_roundtrip(self) -> None:
        """Test to_dict and from_dict roundtrip."""
        now = datetime.now(timezone.utc)

        decision = ApprovalDecision(
            approved=False,
            reason="Unauthorized vendor",
            decided_at=now,
        )

        data = decision.to_dict()
        restored = ApprovalDecision.from_dict(data)

        assert restored.approved == decision.approved
        assert restored.reason == decision.reason
        assert restored.decided_at == decision.decided_at


class TestConversationStateWithApprovals:
    """Tests for ConversationState with approval fields."""

    def test_empty_state_has_empty_approvals(self) -> None:
        """Test that new state has empty approval dicts."""
        state = ConversationState()

        assert state.pending_approvals == {}
        assert state.approval_decisions == {}

    def test_state_with_pending_approvals(self) -> None:
        """Test state with pending approval requests."""
        now = datetime.now(timezone.utc)

        request = ApprovalRequest(
            id="approval-789",
            tool_name="delete_account",
            tool_args={"user_id": "12345"},
            requested_at=now,
            timeout_at=now + timedelta(days=1),
        )

        state = ConversationState(
            pending_approvals={"approval-789": request},
        )

        assert len(state.pending_approvals) == 1
        assert state.pending_approvals["approval-789"].tool_name == "delete_account"

    def test_state_serialization_with_approvals(self) -> None:
        """Test state serialization with approval data."""
        now = datetime.now(timezone.utc)

        request = ApprovalRequest(
            id="approval-101",
            tool_name="transfer_funds",
            tool_args={"amount": 1000},
            requested_at=now,
            timeout_at=now + timedelta(days=7),
        )

        decision = ApprovalDecision(
            approved=True,
            reason=None,
            decided_at=now,
        )

        state = ConversationState(
            messages=[{"role": "user", "content": "Do something"}],
            pending_approvals={"approval-101": request},
            approval_decisions={"approval-100": decision},
        )

        data = state.to_dict()
        restored = ConversationState.from_dict(data)

        assert len(restored.pending_approvals) == 1
        assert restored.pending_approvals["approval-101"].tool_name == "transfer_funds"
        assert len(restored.approval_decisions) == 1
        assert restored.approval_decisions["approval-100"].approved is True
