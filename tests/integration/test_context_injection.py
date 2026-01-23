"""Integration tests for context injection to MCP tools.

These tests verify that:
1. Context flows through the full Zap workflow to MCP tools
2. Different context types (dict, dataclass, Pydantic) are properly serialized
3. Real MCP servers can access context via dependency injection
"""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from pydantic import BaseModel
from temporalio import activity
from temporalio.client import Client as TemporalClient
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import (
    SandboxedWorkflowRunner,
)

from zap_ai import Zap, ZapAgent
from zap_ai.activities import InferenceOutput, ToolExecutionInput
from zap_ai.activities.inference import InferenceInput
from zap_ai.activities.tool_execution import AgentConfigOutput
from zap_ai.mcp.context import (
    TypedZapContext,
    ZapContext,
    ZapContextValue,
)
from zap_ai.workflows import AgentWorkflow

# =============================================================================
# Workflow Integration Tests - Context flows through Temporal workflow
# =============================================================================

# Store for capturing context passed to tools during workflow tests
_captured_workflow_contexts: dict[str, dict] = {}


async def _mock_inference_with_context_tool(input: InferenceInput) -> InferenceOutput:
    """Mock inference that calls a tool to verify context injection."""
    has_tool_result = any(m.get("role") == "tool" for m in input.messages)
    if has_tool_result:
        return InferenceOutput(
            content="Task completed successfully.",
            tool_calls=[],
            finish_reason="stop",
        )
    return InferenceOutput(
        content=None,
        tool_calls=[
            {
                "id": "call_context_test_123",
                "type": "function",
                "function": {
                    "name": "get_user_info",
                    "arguments": "{}",
                },
            }
        ],
        finish_reason="tool_calls",
    )


mock_inference_activity_context = activity.defn(name="inference_activity")(
    _mock_inference_with_context_tool
)


@activity.defn(name="tool_execution_activity")
async def mock_tool_execution_with_context_capture(input: ToolExecutionInput) -> str:
    """Mock tool execution that captures context for verification."""
    # Store the context for later verification
    test_id = input.arguments.get("_test_id", "default")
    _captured_workflow_contexts[test_id] = input.context or {}

    return json.dumps(
        {
            "tool_name": input.tool_name,
            "context_received": input.context is not None,
            "context": input.context,
        }
    )


@activity.defn(name="get_agent_config_activity")
async def mock_get_agent_config_with_tools(agent_name: str) -> AgentConfigOutput:
    """Mock agent config that includes a test tool."""
    return AgentConfigOutput(
        agent_name=agent_name,
        prompt=f"You are agent {agent_name}.",
        model="gpt-4o",
        max_iterations=50,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_user_info",
                    "description": "Get information about the current user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "_test_id": {
                                "type": "string",
                                "description": "Test identifier",
                            },
                        },
                    },
                },
            }
        ],
    )


@pytest.fixture
async def context_workflow_worker(
    temporal_client: TemporalClient,
    test_sandbox_runner: SandboxedWorkflowRunner,
) -> AsyncGenerator[Worker, None]:
    """Create a worker for context injection workflow testing."""
    task_queue = f"integration-test-context-{uuid.uuid4().hex[:8]}"

    worker = Worker(
        temporal_client,
        task_queue=task_queue,
        workflows=[AgentWorkflow],
        activities=[
            mock_inference_activity_context,
            mock_tool_execution_with_context_capture,
            mock_get_agent_config_with_tools,
        ],
        workflow_runner=test_sandbox_runner,
    )

    _captured_workflow_contexts.clear()

    async with worker:
        yield worker


@pytest.fixture
def context_workflow_task_queue(context_workflow_worker: Worker) -> str:
    """Get the task queue for the context workflow worker."""
    return context_workflow_worker.task_queue


class TestWorkflowContextInjection:
    """Integration tests for context flowing through Temporal workflows to tools."""

    @pytest.mark.asyncio
    async def test_dict_context_flows_through_workflow(
        self,
        temporal_client: TemporalClient,
        context_workflow_worker: Worker,
        context_workflow_task_queue: str,
    ) -> None:
        """Test that dict context passes through workflow to tool execution."""
        agent = ZapAgent(
            name="ContextTestAgent",
            prompt="You are a helpful assistant.",
        )

        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=context_workflow_task_queue,
        )
        await zap.start()

        try:
            context = {"user_id": "user_123", "tenant": "acme_corp", "role": "admin"}

            task = await zap.execute_task(
                agent_name="ContextTestAgent",
                task="Get my user info",
                context=context,
            )

            result = await zap.get_task(task.id)
            while not result.status.is_terminal():
                await asyncio.sleep(0.1)
                result = await zap.get_task(task.id)

            assert result.result is not None

            # Verify context was captured by mock tool execution
            assert "default" in _captured_workflow_contexts
            captured = _captured_workflow_contexts["default"]
            assert captured == context

        finally:
            await zap.stop()

    @pytest.mark.parametrize(
        "context_class,context_data,expected_fields",
        [
            (
                "dataclass",
                {
                    "user_id": "user_456",
                    "tenant": "beta_corp",
                    "permissions": ["read", "write"],
                },
                {
                    "user_id": "user_456",
                    "tenant": "beta_corp",
                    "permissions": ["read", "write"],
                    "__zap_context_version__": "1",
                },
            ),
            (
                "pydantic",
                {
                    "session_id": "sess_789",
                    "user_name": "Charlie",
                    "authenticated": True,
                    "metadata": {"source": "web", "version": "2.0"},
                },
                {
                    "session_id": "sess_789",
                    "user_name": "Charlie",
                    "authenticated": True,
                    "metadata": {"source": "web", "version": "2.0"},
                    "__zap_context_version__": "1",
                },
            ),
        ],
        ids=["dataclass", "pydantic"],
    )
    @pytest.mark.asyncio
    async def test_typed_context_serialized_through_workflow(
        self,
        temporal_client: TemporalClient,
        context_workflow_worker: Worker,
        context_workflow_task_queue: str,
        context_class: str,
        context_data: dict,
        expected_fields: dict,
    ) -> None:
        """Test that dataclass and Pydantic contexts are serialized through workflow."""
        if context_class == "dataclass":

            @dataclass
            class UserContext:
                user_id: str
                tenant: str
                permissions: list[str]

            context_instance = UserContext(**context_data)
            agent = ZapAgent[UserContext](
                name="DataclassContextAgent",
                prompt="You are a helpful assistant.",
            )
            context_type_suffix = "UserContext"

        else:  # pydantic

            class SessionContext(BaseModel):
                session_id: str
                user_name: str
                authenticated: bool
                metadata: dict[str, str]

            context_instance = SessionContext(**context_data)
            agent = ZapAgent[SessionContext](
                name="PydanticContextAgent",
                prompt="You are a helpful assistant.",
            )
            context_type_suffix = "SessionContext"

        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=context_workflow_task_queue,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name=agent.name,
                task="Get my user info",
                context=context_instance,
            )

            result = await zap.get_task(task.id)
            while not result.status.is_terminal():
                await asyncio.sleep(0.1)
                result = await zap.get_task(task.id)

            assert result.result is not None

            # Verify serialized context includes type metadata
            assert "default" in _captured_workflow_contexts
            captured = _captured_workflow_contexts["default"]

            for key, value in expected_fields.items():
                assert captured[key] == value

            # Verify type metadata is present for deserialization
            assert "__zap_context_type__" in captured
            assert captured["__zap_context_type__"].endswith(context_type_suffix)

        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_no_context_passes_none_through_workflow(
        self,
        temporal_client: TemporalClient,
        context_workflow_worker: Worker,
        context_workflow_task_queue: str,
    ) -> None:
        """Test that running without context passes None/empty to tools."""
        agent = ZapAgent(
            name="NoContextAgent",
            prompt="You are a helpful assistant.",
        )

        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=context_workflow_task_queue,
        )
        await zap.start()

        try:
            task = await zap.execute_task(
                agent_name="NoContextAgent",
                task="Get my user info",
                # No context provided
            )

            result = await zap.get_task(task.id)
            while not result.status.is_terminal():
                await asyncio.sleep(0.1)
                result = await zap.get_task(task.id)

            assert result.result is not None

            # Verify empty context was captured
            assert "default" in _captured_workflow_contexts
            captured = _captured_workflow_contexts["default"]
            assert captured == {}

        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_context_with_dynamic_prompt(
        self,
        temporal_client: TemporalClient,
        context_workflow_worker: Worker,
        context_workflow_task_queue: str,
    ) -> None:
        """Test that context works for both dynamic prompts AND tool injection."""

        @dataclass
        class FullContext:
            user_name: str
            user_id: str
            api_key: str

        def make_prompt(ctx: FullContext) -> str:
            return f"You are a personal assistant for {ctx.user_name}."

        agent = ZapAgent[FullContext](
            name="FullContextAgent",
            prompt=make_prompt,
        )

        zap: Zap[FullContext] = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=context_workflow_task_queue,
        )
        await zap.start()

        try:
            context = FullContext(
                user_name="Diana",
                user_id="user_diana",
                api_key="secret_key_123",
            )

            task = await zap.execute_task(
                agent_name="FullContextAgent",
                task="Get my user info",
                context=context,
            )

            result = await zap.get_task(task.id)
            while not result.status.is_terminal():
                await asyncio.sleep(0.1)
                result = await zap.get_task(task.id)

            assert result.result is not None

            # Verify dynamic prompt was resolved
            assert result.history is not None
            system_message = result.history[0]
            assert system_message["role"] == "system"
            assert "Diana" in system_message["content"]
            assert system_message["content"] == "You are a personal assistant for Diana."

            # Verify context also flowed to the tool
            assert "default" in _captured_workflow_contexts
            captured = _captured_workflow_contexts["default"]
            assert captured["user_name"] == "Diana"
            assert captured["user_id"] == "user_diana"
            assert captured["api_key"] == "secret_key_123"

        finally:
            await zap.stop()


# =============================================================================
# Real MCP Integration Tests - Context flows through real FastMCP server
# =============================================================================


# Test fixture for real MCP test - must be at module level for importability
class UserSession(BaseModel):
    user_id: str
    tenant: str
    roles: list[str]
    preferences: dict[str, bool]


class TestRealMCPContextInjection:
    """Integration tests using real FastMCP servers (only inference is mocked)."""

    @pytest.mark.asyncio
    async def test_pydantic_context_through_real_mcp_server(
        self,
        temporal_client: TemporalClient,
        test_sandbox_runner: SandboxedWorkflowRunner,
    ) -> None:
        """Test Pydantic context flows through real Zap system to real MCP tools.

        This test uses:
        - Real FastMCP server with tools that access context via dependency injection
        - Real tool_execution_activity
        - Real get_agent_config_activity
        - Only mocked inference (to control which tool gets called)
        """
        from fastmcp.dependencies import Depends

        # Create a FastMCP server with a tool that echoes context
        mcp = FastMCP("RealContextServer")

        # Store to capture what context the tool received
        captured_context: dict[str, Any] = {}
        captured_typed_context: Any = None

        # Create typed dependency
        CurrentSession = TypedZapContext(UserSession)

        @mcp.tool()
        async def echo_user_context(
            zap_ctx: dict = Depends(ZapContext),
            session: UserSession = Depends(CurrentSession),
        ) -> str:
            """Return the user context received via Zap."""
            nonlocal captured_typed_context

            # Capture both dict and typed context
            captured_context.update(zap_ctx)
            captured_typed_context = session

            return json.dumps(zap_ctx)

        # Create agent with the real MCP server
        mcp_client = Client(mcp)
        agent = ZapAgent[UserSession](
            name="RealContextAgent",
            prompt="You help users check their context.",
            mcp_clients=[mcp_client],
        )

        # Create Zap instance
        task_queue = f"integration-real-context-{uuid.uuid4().hex[:8]}"
        zap: Zap[UserSession] = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue,
        )

        # Mock inference to call our test tool
        async def mock_inference(input: InferenceInput) -> InferenceOutput:
            has_tool_result = any(m.get("role") == "tool" for m in input.messages)
            if has_tool_result:
                return InferenceOutput(
                    content="Context check complete.",
                    tool_calls=[],
                    finish_reason="stop",
                )
            return InferenceOutput(
                content=None,
                tool_calls=[
                    {
                        "id": "call_echo_ctx",
                        "type": "function",
                        "function": {
                            "name": "echo_user_context",
                            "arguments": "{}",
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        mock_inference_activity = activity.defn(name="inference_activity")(mock_inference)

        # Start Zap to initialize tool registry
        await zap.start()

        try:
            # Create worker with real tool execution but mocked inference
            from zap_ai.activities.tool_execution import (
                get_agent_config_activity,
                set_tool_registry,
                tool_execution_activity,
            )

            # Set the tool registry for the real activity
            set_tool_registry(zap._tool_registry)

            worker = Worker(
                temporal_client,
                task_queue=task_queue,
                workflows=[AgentWorkflow],
                activities=[
                    mock_inference_activity,  # Mocked
                    tool_execution_activity,  # Real - calls actual MCP server
                    get_agent_config_activity,  # Real - gets tools from registry
                ],
                workflow_runner=test_sandbox_runner,
            )

            async with worker:
                # Execute task with Pydantic context
                session = UserSession(
                    user_id="real_user_123",
                    tenant="real_tenant_corp",
                    roles=["admin", "viewer"],
                    preferences={"dark_mode": True, "beta_features": False},
                )

                task = await zap.execute_task(
                    agent_name="RealContextAgent",
                    task="Check my user context",
                    context=session,
                )

                # Wait for completion
                result = await zap.get_task(task.id)
                while not result.status.is_terminal():
                    await asyncio.sleep(0.1)
                    result = await zap.get_task(task.id)

                # Verify the task completed successfully
                assert result.result is not None
                assert "Context check complete" in result.result

                # Verify context was captured by the real MCP tool (dict access)
                assert captured_context["user_id"] == "real_user_123"
                assert captured_context["tenant"] == "real_tenant_corp"
                assert captured_context["roles"] == ["admin", "viewer"]
                assert captured_context["preferences"]["dark_mode"] is True
                assert captured_context["preferences"]["beta_features"] is False

                # Verify typed deserialization worked correctly
                assert captured_typed_context is not None
                assert isinstance(captured_typed_context, UserSession)
                assert captured_typed_context.user_id == "real_user_123"
                assert captured_typed_context.tenant == "real_tenant_corp"
                assert captured_typed_context.roles == ["admin", "viewer"]
                assert captured_typed_context.preferences["dark_mode"] is True
                assert captured_typed_context.preferences["beta_features"] is False

        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_context_value_extraction_through_real_mcp(
        self,
        temporal_client: TemporalClient,
        test_sandbox_runner: SandboxedWorkflowRunner,
    ) -> None:
        """Test ZapContextValue extracts values correctly through real MCP server."""
        from fastmcp.dependencies import Depends

        # Create a FastMCP server with tools that use ZapContextValue
        mcp = FastMCP("ValueExtractionServer")

        # Store to capture extracted values
        captured_values: dict[str, Any] = {}

        # Create value extraction dependencies
        UserId = ZapContextValue("user_id", "anonymous")
        Tenant = ZapContextValue("tenant", "default_tenant")
        MissingKey = ZapContextValue("nonexistent_key", "default_value")

        @mcp.tool()
        async def get_user_details(
            user_id: str = Depends(UserId),
            tenant: str = Depends(Tenant),
            missing: str = Depends(MissingKey),
        ) -> str:
            """Extract individual context values."""
            nonlocal captured_values
            captured_values["user_id"] = user_id
            captured_values["tenant"] = tenant
            captured_values["missing"] = missing
            return json.dumps({"user_id": user_id, "tenant": tenant, "missing": missing})

        # Create agent with the real MCP server (using plain dict context)
        mcp_client = Client(mcp)
        agent = ZapAgent(
            name="ValueExtractionAgent",
            prompt="You help extract context values.",
            mcp_clients=[mcp_client],
        )

        # Create Zap instance
        task_queue = f"integration-value-extract-{uuid.uuid4().hex[:8]}"
        zap = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=task_queue,
        )

        # Mock inference to call our test tool
        async def mock_inference(input: InferenceInput) -> InferenceOutput:
            has_tool_result = any(m.get("role") == "tool" for m in input.messages)
            if has_tool_result:
                return InferenceOutput(
                    content="Value extraction complete.",
                    tool_calls=[],
                    finish_reason="stop",
                )
            return InferenceOutput(
                content=None,
                tool_calls=[
                    {
                        "id": "call_get_details",
                        "type": "function",
                        "function": {
                            "name": "get_user_details",
                            "arguments": "{}",
                        },
                    }
                ],
                finish_reason="tool_calls",
            )

        mock_inference_activity = activity.defn(name="inference_activity")(mock_inference)

        # Start Zap to initialize tool registry
        await zap.start()

        try:
            from zap_ai.activities.tool_execution import (
                get_agent_config_activity,
                set_tool_registry,
                tool_execution_activity,
            )

            set_tool_registry(zap._tool_registry)

            worker = Worker(
                temporal_client,
                task_queue=task_queue,
                workflows=[AgentWorkflow],
                activities=[
                    mock_inference_activity,
                    tool_execution_activity,
                    get_agent_config_activity,
                ],
                workflow_runner=test_sandbox_runner,
            )

            async with worker:
                # Execute task with plain dict context (no type metadata)
                context = {
                    "user_id": "extract_user_123",
                    "tenant": "extract_tenant",
                    "extra_field": "not_extracted",
                }

                task = await zap.execute_task(
                    agent_name="ValueExtractionAgent",
                    task="Get my user details",
                    context=context,
                )

                # Wait for completion
                result = await zap.get_task(task.id)
                while not result.status.is_terminal():
                    await asyncio.sleep(0.1)
                    result = await zap.get_task(task.id)

                # Verify the task completed successfully
                assert result.result is not None
                assert "Value extraction complete" in result.result

                # Verify values were extracted correctly
                assert captured_values["user_id"] == "extract_user_123"
                assert captured_values["tenant"] == "extract_tenant"
                # Missing key should return the default value
                assert captured_values["missing"] == "default_value"

        finally:
            await zap.stop()
