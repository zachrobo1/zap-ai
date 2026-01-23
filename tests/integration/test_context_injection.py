"""Integration tests for context injection to MCP tools.

These tests verify that:
1. Context passed via FastMCP's meta parameter is accessible in tools via dependency injection
2. The ZapContext, ZapContextValue, and TypedZapContext dependencies work correctly
3. Context flows through the full Zap workflow to MCP tools
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
# FastMCP Server Tests - Direct context injection via meta parameter
# =============================================================================


class TestFastMCPContextInjection:
    """Test context injection directly through FastMCP client/server."""

    @pytest.mark.asyncio
    async def test_context_accessible_via_zap_context_dependency(self) -> None:
        """Test that context passed via meta is accessible in tool via ZapContext dependency."""
        from fastmcp.dependencies import Depends

        # Create a FastMCP server with a tool that returns the context
        mcp = FastMCP("ContextTestServer")

        @mcp.tool()
        async def echo_context(zap_ctx: dict = Depends(ZapContext)) -> str:
            """Return the zap context as JSON."""
            return json.dumps(zap_ctx)

        # Connect client directly to server (in-memory)
        client = Client(mcp)

        async with client:
            result = await client.call_tool(
                "echo_context",
                {},
                meta={"zap_context": {"user_id": "user_123", "tenant": "acme"}},
            )

        # Parse the result - FastMCP returns CallToolResult with content list
        assert result.content
        content = result.content[0].text
        parsed = json.loads(content)

        assert parsed == {"user_id": "user_123", "tenant": "acme"}

    @pytest.mark.asyncio
    async def test_zap_context_value_extracts_specific_keys(self) -> None:
        """Test that ZapContextValue dependency correctly extracts individual values."""
        from fastmcp.dependencies import Depends

        mcp = FastMCP("ValueTestServer")

        UserId = ZapContextValue("user_id", "anonymous")
        MissingKey = ZapContextValue("nonexistent", "default_value")

        @mcp.tool()
        async def get_user_id(user_id: str = Depends(UserId)) -> str:
            """Return just the user_id from context."""
            return user_id

        @mcp.tool()
        async def get_missing_key(missing: str = Depends(MissingKey)) -> str:
            """Return a key that doesn't exist, with default."""
            return missing

        client = Client(mcp)

        async with client:
            # Test getting existing value
            result = await client.call_tool(
                "get_user_id",
                {},
                meta={"zap_context": {"user_id": "user_456", "tenant": "beta"}},
            )
            assert result.content[0].text == "user_456"

            # Test getting missing value returns default
            result = await client.call_tool(
                "get_missing_key",
                {},
                meta={"zap_context": {"user_id": "user_456"}},
            )
            assert result.content[0].text == "default_value"

    @pytest.mark.asyncio
    async def test_context_empty_when_no_meta_provided(self) -> None:
        """Test that context is empty dict when no meta parameter is passed."""
        from fastmcp.dependencies import Depends

        mcp = FastMCP("EmptyContextServer")

        @mcp.tool()
        async def check_context(zap_ctx: dict = Depends(ZapContext)) -> str:
            """Return whether context is empty."""
            return "empty" if zap_ctx == {} else f"has_data: {json.dumps(zap_ctx)}"

        client = Client(mcp)

        async with client:
            # Call without meta parameter
            result = await client.call_tool("check_context", {})
            assert result.content[0].text == "empty"

    @pytest.mark.asyncio
    async def test_complex_nested_context(self) -> None:
        """Test that complex nested context structures are preserved."""
        from fastmcp.dependencies import Depends

        mcp = FastMCP("NestedContextServer")

        @mcp.tool()
        async def echo_context(zap_ctx: dict = Depends(ZapContext)) -> str:
            """Return the full context."""
            return json.dumps(zap_ctx)

        client = Client(mcp)

        complex_context = {
            "user_id": "user_789",
            "permissions": ["read", "write", "admin"],
            "metadata": {
                "source": "api",
                "version": "2.0",
                "features": {"dark_mode": True, "notifications": False},
            },
            "tags": ["premium", "beta-tester"],
        }

        async with client:
            result = await client.call_tool(
                "echo_context",
                {},
                meta={"zap_context": complex_context},
            )

        parsed = json.loads(result.content[0].text)
        assert parsed == complex_context
        assert parsed["permissions"] == ["read", "write", "admin"]
        assert parsed["metadata"]["features"]["dark_mode"] is True

    @pytest.mark.asyncio
    async def test_context_with_tool_arguments(self) -> None:
        """Test that context works alongside regular tool arguments."""
        from fastmcp.dependencies import Depends

        mcp = FastMCP("MixedServer")

        Tenant = ZapContextValue("tenant", "default")

        @mcp.tool()
        async def search_for_user(
            query: str,
            limit: int = 10,
            tenant: str = Depends(Tenant),
        ) -> str:
            """Search scoped to the current tenant."""
            return json.dumps(
                {
                    "query": query,
                    "limit": limit,
                    "tenant_scope": tenant,
                }
            )

        client = Client(mcp)

        async with client:
            result = await client.call_tool(
                "search_for_user",
                {"query": "john", "limit": 5},
                meta={"zap_context": {"tenant": "acme_corp", "user_id": "admin"}},
            )

        parsed = json.loads(result.content[0].text)
        assert parsed["query"] == "john"
        assert parsed["limit"] == 5
        assert parsed["tenant_scope"] == "acme_corp"

    @pytest.mark.asyncio
    async def test_dependency_injection_with_zap_context(self) -> None:
        """Test using ZapContext dependency injection helper."""
        from fastmcp.dependencies import Depends

        mcp = FastMCP("DependencyInjectionServer")

        @mcp.tool()
        async def get_user_details(
            zap_ctx: dict = Depends(ZapContext),
        ) -> str:
            """Get user details using dependency injection."""
            return json.dumps(
                {
                    "user_id": zap_ctx.get("user_id"),
                    "tenant": zap_ctx.get("tenant"),
                    "context_present": bool(zap_ctx),
                }
            )

        client = Client(mcp)

        async with client:
            result = await client.call_tool(
                "get_user_details",
                {},
                meta={"zap_context": {"user_id": "user_999", "tenant": "test_corp"}},
            )

        parsed = json.loads(result.content[0].text)
        assert parsed["user_id"] == "user_999"
        assert parsed["tenant"] == "test_corp"
        assert parsed["context_present"] is True

    @pytest.mark.asyncio
    async def test_dependency_injection_with_zap_context_value(self) -> None:
        """Test using ZapContextValue dependency factory."""
        from fastmcp.dependencies import Depends

        mcp = FastMCP("ValueDependencyServer")

        # Create reusable dependencies
        UserId = ZapContextValue("user_id")
        Tenant = ZapContextValue("tenant", "default")

        @mcp.tool()
        async def check_permissions(
            user_id: str | None = Depends(UserId),
            tenant: str = Depends(Tenant),
        ) -> str:
            """Check permissions using injected context values."""
            if not user_id:
                return "No user authenticated"
            return json.dumps({"user": user_id, "tenant": tenant, "has_access": True})

        client = Client(mcp)

        async with client:
            # Test with full context
            result = await client.call_tool(
                "check_permissions",
                {},
                meta={"zap_context": {"user_id": "user_777", "tenant": "enterprise"}},
            )
            parsed = json.loads(result.content[0].text)
            assert parsed["user"] == "user_777"
            assert parsed["tenant"] == "enterprise"

            # Test with partial context (missing tenant, should use default)
            result = await client.call_tool(
                "check_permissions",
                {},
                meta={"zap_context": {"user_id": "user_888"}},
            )
            parsed = json.loads(result.content[0].text)
            assert parsed["user"] == "user_888"
            assert parsed["tenant"] == "default"


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

    @pytest.mark.asyncio
    async def test_dataclass_context_serialized_through_workflow(
        self,
        temporal_client: TemporalClient,
        context_workflow_worker: Worker,
        context_workflow_task_queue: str,
    ) -> None:
        """Test that dataclass context is serialized and passes through workflow."""

        @dataclass
        class UserContext:
            user_id: str
            tenant: str
            permissions: list[str]

        agent = ZapAgent[UserContext](
            name="DataclassContextAgent",
            prompt="You are a helpful assistant.",
        )

        zap: Zap[UserContext] = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=context_workflow_task_queue,
        )
        await zap.start()

        try:
            context = UserContext(
                user_id="user_456",
                tenant="beta_corp",
                permissions=["read", "write"],
            )

            task = await zap.execute_task(
                agent_name="DataclassContextAgent",
                task="Get my user info",
                context=context,
            )

            result = await zap.get_task(task.id)
            while not result.status.is_terminal():
                await asyncio.sleep(0.1)
                result = await zap.get_task(task.id)

            assert result.result is not None

            # Verify serialized context includes type metadata
            assert "default" in _captured_workflow_contexts
            captured = _captured_workflow_contexts["default"]
            assert captured["user_id"] == "user_456"
            assert captured["tenant"] == "beta_corp"
            assert captured["permissions"] == ["read", "write"]

            # Verify type metadata is present for deserialization
            assert "__zap_context_type__" in captured
            assert captured["__zap_context_type__"].endswith("UserContext")
            assert captured["__zap_context_version__"] == "1"

        finally:
            await zap.stop()

    @pytest.mark.asyncio
    async def test_pydantic_context_serialized_through_workflow(
        self,
        temporal_client: TemporalClient,
        context_workflow_worker: Worker,
        context_workflow_task_queue: str,
    ) -> None:
        """Test that Pydantic model context is serialized via model_dump."""

        class SessionContext(BaseModel):
            session_id: str
            user_name: str
            authenticated: bool
            metadata: dict[str, str]

        agent = ZapAgent[SessionContext](
            name="PydanticContextAgent",
            prompt="You are a helpful assistant.",
        )

        zap: Zap[SessionContext] = Zap(
            agents=[agent],
            temporal_client=temporal_client,
            task_queue=context_workflow_task_queue,
        )
        await zap.start()

        try:
            context = SessionContext(
                session_id="sess_789",
                user_name="Charlie",
                authenticated=True,
                metadata={"source": "web", "version": "2.0"},
            )

            task = await zap.execute_task(
                agent_name="PydanticContextAgent",
                task="Get my user info",
                context=context,
            )

            result = await zap.get_task(task.id)
            while not result.status.is_terminal():
                await asyncio.sleep(0.1)
                result = await zap.get_task(task.id)

            assert result.result is not None

            # Verify Pydantic context was serialized correctly with type metadata
            assert "default" in _captured_workflow_contexts
            captured = _captured_workflow_contexts["default"]
            assert captured["session_id"] == "sess_789"
            assert captured["user_name"] == "Charlie"
            assert captured["authenticated"] is True
            assert captured["metadata"] == {"source": "web", "version": "2.0"}

            # Verify type metadata is present for deserialization
            assert "__zap_context_type__" in captured
            assert captured["__zap_context_type__"].endswith("SessionContext")
            assert captured["__zap_context_version__"] == "1"

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
