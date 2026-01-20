"""Integration tests for MCP sampling support.

These tests verify that MCP sampling handlers work correctly when
MCP servers request LLM completions through the Zap client.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from zap_ai.mcp.sampling import LiteLLMSamplingHandler, create_mcp_client


class TestLiteLLMSamplingHandlerIntegration:
    """Integration tests for LiteLLMSamplingHandler with mocked LLM provider."""

    @pytest.mark.asyncio
    async def test_handler_processes_sampling_request(self) -> None:
        """Test that handler correctly processes a sampling request."""
        with patch("zap_ai.llm.provider.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = MagicMock(content="Generated response")

            handler = LiteLLMSamplingHandler(
                default_model="gpt-4o",
                default_temperature=0.5,
            )

            # Simulate MCP sampling message
            msg = MagicMock(role="user")
            msg.content = MagicMock(text="Summarize this document")

            params = MagicMock(
                systemPrompt="You are a helpful assistant",
                modelPreferences=None,
                temperature=None,
                maxTokens=None,
            )

            result = await handler([msg], params, MagicMock())

            assert result == "Generated response"
            mock_complete.assert_called_once()

            # Verify messages were converted correctly
            call_kwargs = mock_complete.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4o"
            assert call_kwargs["temperature"] == 0.5
            assert len(call_kwargs["messages"]) == 2
            assert call_kwargs["messages"][0]["role"] == "system"
            assert call_kwargs["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_handler_respects_model_hints(self) -> None:
        """Test that handler extracts model from modelPreferences hints."""
        with patch("zap_ai.llm.provider.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = MagicMock(content="Response")

            handler = LiteLLMSamplingHandler()

            # Create model hint
            hint = MagicMock()
            hint.name = "anthropic/claude-sonnet-4-5-20250929"
            prefs = MagicMock(hints=[hint])

            msg = MagicMock(role="user", content="Test")
            params = MagicMock(
                systemPrompt=None,
                modelPreferences=prefs,
                temperature=0.3,
                maxTokens=100,
            )

            await handler([msg], params, MagicMock())

            call_kwargs = mock_complete.call_args.kwargs
            assert call_kwargs["model"] == "anthropic/claude-sonnet-4-5-20250929"
            assert call_kwargs["temperature"] == 0.3
            assert call_kwargs["max_tokens"] == 100


class TestCreateMcpClientIntegration:
    """Integration tests for create_mcp_client factory function."""

    @pytest.fixture
    def mock_fastmcp(self):
        """Mock fastmcp module to avoid importing it (which triggers beartype pollution)."""
        mock_client_class = MagicMock()
        mock_fastmcp_module = MagicMock()
        mock_fastmcp_module.Client = mock_client_class

        with patch.dict(sys.modules, {"fastmcp": mock_fastmcp_module}):
            yield mock_client_class

    def test_creates_client_with_litellm_handler(self, mock_fastmcp) -> None:
        """Test that factory creates Client with LiteLLM handler."""
        create_mcp_client(
            "server.py",
            sampling_handler="litellm",
            sampling_model="gpt-4-turbo",
        )

        call_kwargs = mock_fastmcp.call_args.kwargs
        handler = call_kwargs["sampling_handler"]

        assert isinstance(handler, LiteLLMSamplingHandler)
        assert handler.default_model == "gpt-4-turbo"

    def test_creates_client_with_custom_handler(self, mock_fastmcp) -> None:
        """Test that factory accepts custom handler function."""

        async def my_handler(messages, params, context):
            return "custom response"

        create_mcp_client("server.py", sampling_handler=my_handler)

        call_kwargs = mock_fastmcp.call_args.kwargs
        assert call_kwargs["sampling_handler"] is my_handler
