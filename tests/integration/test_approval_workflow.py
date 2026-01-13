"""Integration tests for human-in-the-loop approval workflows.

These tests verify that:
1. Tools matching approval patterns trigger the approval flow
2. Approving a pending tool call resumes execution
3. Rejecting a tool call skips it and continues the loop
4. Pending approvals can be queried
5. Sub-agent calls don't trigger parent approval rules (only MCP tools do)
"""

import asyncio

import pytest
from temporalio.client import Client

from zap_ai import ApprovalRules, TaskStatus, Zap, ZapAgent


class TestApprovalWorkflowSimple:
    """Tests for basic approval workflow functionality."""

    @pytest.mark.asyncio
    async def test_approval_workflow_triggers_awaiting_status(
        self,
        temporal_client: Client,
        integration_worker_with_approval,
        task_queue_approval: str,
    ) -> None:
        """Test that a tool matching approval pattern triggers AWAITING_APPROVAL status."""
        agent = ZapAgent(name="ApprovalAgent", prompt="You are a file manager.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue_approval,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="ApprovalAgent",
                task="Delete the test file",
                approval_rules=ApprovalRules(patterns=["delete_*"]),
            )

            # Wait for approval status
            for _ in range(50):
                task = await zap.get_task(task.id)
                if task.status == TaskStatus.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.AWAITING_APPROVAL
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_approval_workflow_approve_completes_task(
        self,
        temporal_client: Client,
        integration_worker_with_approval,
        task_queue_approval: str,
    ) -> None:
        """Test that approving a pending tool call allows task to complete."""
        agent = ZapAgent(name="ApprovalAgent", prompt="You are a file manager.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue_approval,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="ApprovalAgent",
                task="Delete the test file",
                approval_rules=ApprovalRules(patterns=["delete_*"]),
            )

            # Wait for approval status
            for _ in range(50):
                task = await zap.get_task(task.id)
                if task.status == TaskStatus.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.AWAITING_APPROVAL

            # Get pending approvals
            pending = await task.get_pending_approvals()
            assert len(pending) == 1
            assert pending[0]["tool_name"] == "delete_file"

            # Approve the tool call
            await task.approve(pending[0]["id"])

            # Wait for completion
            for _ in range(50):
                task = await zap.get_task(task.id)
                if task.status.is_terminal():
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.COMPLETED
            assert "Task completed after approval" in task.result
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_approval_workflow_reject_continues_loop(
        self,
        temporal_client: Client,
        integration_worker_with_approval,
        task_queue_approval: str,
    ) -> None:
        """Test that rejecting a tool call skips it and continues the agentic loop."""
        agent = ZapAgent(name="ApprovalAgent", prompt="You are a file manager.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue_approval,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="ApprovalAgent",
                task="Delete the test file",
                approval_rules=ApprovalRules(patterns=["delete_*"]),
            )

            # Wait for approval status
            for _ in range(50):
                task = await zap.get_task(task.id)
                if task.status == TaskStatus.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.AWAITING_APPROVAL

            # Get pending approvals
            pending = await task.get_pending_approvals()
            assert len(pending) == 1

            # Reject the tool call
            await task.reject(pending[0]["id"], reason="Not authorized")

            # Wait for completion - the loop continues after rejection
            for _ in range(50):
                task = await zap.get_task(task.id)
                if task.status.is_terminal():
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.COMPLETED

            # The conversation should contain the rejection message
            history = task.history
            tool_messages = [m for m in history if m.get("role") == "tool"]
            assert any("rejected" in str(m.get("content", "")).lower() for m in tool_messages)
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_get_pending_approvals_returns_correct_data(
        self,
        temporal_client: Client,
        integration_worker_with_approval,
        task_queue_approval: str,
    ) -> None:
        """Test that get_pending_approvals() returns correct approval request data."""
        agent = ZapAgent(name="ApprovalAgent", prompt="You are a file manager.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue_approval,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="ApprovalAgent",
                task="Delete the test file",
                approval_rules=ApprovalRules(patterns=["delete_*"]),
            )

            # Wait for approval status
            for _ in range(50):
                task = await zap.get_task(task.id)
                if task.status == TaskStatus.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.1)

            # Get pending approvals
            pending = await task.get_pending_approvals()
            assert len(pending) == 1

            approval = pending[0]
            assert "id" in approval
            assert approval["tool_name"] == "delete_file"
            assert approval["tool_args"] == {"path": "/tmp/test.txt"}
            assert "requested_at" in approval
            assert "timeout_at" in approval
            assert "context" in approval
            assert approval["context"]["agent_name"] == "ApprovalAgent"

            # Clean up by approving
            await task.approve(approval["id"])
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_no_approval_without_rules(
        self,
        temporal_client: Client,
        integration_worker_with_approval,
        task_queue_approval: str,
    ) -> None:
        """Test that tools execute immediately when no approval rules are set."""
        agent = ZapAgent(name="NoApprovalAgent", prompt="You are a file manager.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue_approval,
        )
        await zap.start()

        try:
            # No approval_rules parameter
            task = await zap.execute_task(
                agent_name="NoApprovalAgent",
                task="Delete the test file",
            )

            # Task should complete without hitting AWAITING_APPROVAL
            for _ in range(50):
                task = await zap.get_task(task.id)
                # Should never be AWAITING_APPROVAL
                assert task.status != TaskStatus.AWAITING_APPROVAL
                if task.status.is_terminal():
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.COMPLETED
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_non_matching_pattern_no_approval(
        self,
        temporal_client: Client,
        integration_worker_with_approval,
        task_queue_approval: str,
    ) -> None:
        """Test that tools not matching patterns execute without approval."""
        agent = ZapAgent(name="MismatchAgent", prompt="You are a file manager.")
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue_approval,
        )
        await zap.start()

        try:
            # Pattern doesn't match delete_file
            task = await zap.execute_task(
                agent_name="MismatchAgent",
                task="Delete the test file",
                approval_rules=ApprovalRules(patterns=["transfer_*"]),
            )

            # Task should complete without hitting AWAITING_APPROVAL
            for _ in range(50):
                task = await zap.get_task(task.id)
                assert task.status != TaskStatus.AWAITING_APPROVAL
                if task.status.is_terminal():
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.COMPLETED
        finally:
            await zap.stop()


class TestApprovalWorkflowWithSubAgent:
    """Tests for approval workflows with sub-agent delegation."""

    @pytest.mark.asyncio
    async def test_subagent_delegation_no_approval_then_tool_approval(
        self,
        temporal_client: Client,
        integration_worker_subagent_approval,
        task_queue_subagent_approval: str,
    ) -> None:
        """
        Test that sub-agent calls don't trigger approval, but MCP tools do.

        Sequence:
        1. Agent calls message_agent (no approval needed for delegation)
        2. Agent calls transfer_funds (requires approval)
        3. User approves
        4. Task completes
        """
        sub_agent = ZapAgent(
            name="SubAgent",
            prompt="You are a helper agent.",
            discovery_prompt="Use for helper tasks",
        )
        main_agent = ZapAgent(
            name="MainAgent",
            prompt="You are a coordinator.",
            sub_agents=["SubAgent"],
        )
        zap = Zap(
            agents=[main_agent, sub_agent],
            temporal_client=temporal_client,
            task_queue=task_queue_subagent_approval,
        )
        await zap.start()

        try:
            # Approval rules only on transfer_* - message_agent should not trigger
            task = await zap.execute_task(
                agent_name="MainAgent",
                task="Delegate to sub-agent then transfer funds",
                approval_rules=ApprovalRules(patterns=["transfer_*"]),
            )

            # Wait for approval status (should be triggered by transfer_funds, not message_agent)
            for _ in range(100):
                task = await zap.get_task(task.id)
                if task.status == TaskStatus.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.AWAITING_APPROVAL

            # Verify the pending approval is for transfer_funds, not message_agent
            pending = await task.get_pending_approvals()
            assert len(pending) == 1
            assert pending[0]["tool_name"] == "transfer_funds"
            assert pending[0]["tool_args"] == {"amount": 1000}

            # Approve the transfer
            await task.approve(pending[0]["id"])

            # Wait for completion
            for _ in range(50):
                task = await zap.get_task(task.id)
                if task.status.is_terminal():
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.COMPLETED
            assert "All done with sub-agent and approval" in task.result
        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_subagent_execution_before_approval(
        self,
        temporal_client: Client,
        integration_worker_subagent_approval,
        task_queue_subagent_approval: str,
    ) -> None:
        """
        Test that sub-agent executes successfully before the approval-required tool.

        Verifies that the conversation history contains the sub-agent result
        before the approval is requested.
        """
        sub_agent = ZapAgent(
            name="SubAgent",
            prompt="You are a helper agent.",
            discovery_prompt="Use for helper tasks",
        )
        main_agent = ZapAgent(
            name="MainAgent",
            prompt="You are a coordinator.",
            sub_agents=["SubAgent"],
        )
        zap = Zap(
            agents=[main_agent, sub_agent],
            temporal_client=temporal_client,
            task_queue=task_queue_subagent_approval,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="MainAgent",
                task="Delegate to sub-agent then transfer funds",
                approval_rules=ApprovalRules(patterns=["transfer_*"]),
            )

            # Wait for approval status
            for _ in range(100):
                task = await zap.get_task(task.id)
                if task.status == TaskStatus.AWAITING_APPROVAL:
                    break
                await asyncio.sleep(0.1)

            assert task.status == TaskStatus.AWAITING_APPROVAL

            # At this point, the sub-agent should have already completed
            # The conversation history should contain the message_agent result
            history = task.history
            tool_messages = [m for m in history if m.get("role") == "tool"]

            # Should have at least one tool result from message_agent
            assert len(tool_messages) >= 1
            # The first tool result should be from the sub-agent delegation
            first_tool = tool_messages[0]
            assert "message_agent" in first_tool.get("name", "")

            # Clean up by approving
            pending = await task.get_pending_approvals()
            await task.approve(pending[0]["id"])
        finally:
            await zap.stop()
