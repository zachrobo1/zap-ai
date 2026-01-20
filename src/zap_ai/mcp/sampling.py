"""MCP sampling handler support for Zap.

This module provides utilities for handling MCP sampling requests,
allowing MCP servers to request LLM completions from the Zap client.

Example:
    ```python
    from zap_ai.mcp.sampling import create_mcp_client

    # Create MCP client with LiteLLM-based sampling support
    client = create_mcp_client(
        "path/to/server.py",
        sampling_handler="litellm",
        sampling_model="gpt-4o",
    )

    # With tracing enabled (requires Langfuse configuration)
    client = create_mcp_client(
        "path/to/server.py",
        sampling_handler="litellm",
        sampling_model="gpt-4o",
        enable_tracing=True,
    )
    ```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from zap_ai.tracing import ObservationType, TraceContext, get_tracing_provider

if TYPE_CHECKING:
    from fastmcp import Client

# Type alias for sampling handler callable
SamplingHandler = Callable[..., str]

# Module-level registry mapping client IDs to their sampling handlers
_client_handlers: dict[int, "LiteLLMSamplingHandler"] = {}


def get_sampling_handler_for_client(client: "Client") -> "LiteLLMSamplingHandler | None":
    """
    Get the LiteLLMSamplingHandler for a FastMCP client if one exists.

    Args:
        client: The FastMCP Client instance.

    Returns:
        The handler if found, None otherwise.
    """
    return _client_handlers.get(id(client))


class LiteLLMSamplingHandler:
    """
    Sampling handler that uses LiteLLM for completions.

    Reuses Zap's LLM provider abstraction, supporting all LiteLLM models
    including OpenAI, Anthropic, and many others.

    Example:
        ```python
        from fastmcp import Client
        from zap_ai.mcp.sampling import LiteLLMSamplingHandler

        handler = LiteLLMSamplingHandler(default_model="gpt-4o")
        client = Client("server.py", sampling_handler=handler)
        ```

    Attributes:
        default_model: Default model to use if not specified in request.
        default_temperature: Default sampling temperature.
        default_max_tokens: Default max tokens (None means no limit).
    """

    def __init__(
        self,
        default_model: str = "gpt-4o",
        default_temperature: float = 0.7,
        default_max_tokens: int | None = None,
        enable_tracing: bool = False,
    ) -> None:
        """
        Initialize the sampling handler.

        Args:
            default_model: Default model identifier for LiteLLM.
            default_temperature: Default sampling temperature (0.0-2.0).
            default_max_tokens: Default maximum tokens to generate.
            enable_tracing: Whether to trace sampling calls with Langfuse.
        """
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.enable_tracing = enable_tracing
        # Trace context set by tool_execution_activity for nesting sampling under tool span
        self._trace_context: TraceContext | None = None

    def set_trace_context(self, context: TraceContext) -> None:
        """
        Set the trace context for nesting sampling under a parent span.

        Called by tool_execution_activity before invoking MCP tools that may
        use sampling. This context is used when tracing is enabled.

        Args:
            context: The parent trace context to nest under.
        """
        self._trace_context = context

    def clear_trace_context(self) -> None:
        """Clear the trace context after tool execution completes."""
        self._trace_context = None

    async def __call__(
        self,
        messages: list[Any],
        params: Any,
        context: Any,
    ) -> str:
        """
        Handle sampling request using LiteLLM.

        Args:
            messages: List of SamplingMessage objects from FastMCP.
            params: CreateSamplingMessageRequestParams from FastMCP.
            context: Context object provided by FastMCP.

        Returns:
            Generated text response.
        """
        # Convert messages to LiteLLM format
        litellm_messages = self._convert_messages(messages, params)

        # Lazy import to avoid polluting Temporal workflow sandbox
        from zap_ai.llm.provider import complete

        # Extract parameters with fallbacks
        model = self._get_model(params)
        temperature = getattr(params, "temperature", None) or self.default_temperature
        max_tokens = getattr(params, "maxTokens", None) or self.default_max_tokens

        if self.enable_tracing:
            return await self._call_with_tracing(model, litellm_messages, temperature, max_tokens)

        result = await complete(
            model=model,
            messages=litellm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return result.content or ""

    async def _call_with_tracing(
        self,
        model: str,
        litellm_messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Execute completion with Langfuse tracing."""
        tracer = get_tracing_provider()

        if not self._trace_context:
            raise RuntimeError(
                "Tracing is enabled but no parent trace context is available. "
                "MCP sampling must occur within a traced tool execution. "
                "Ensure set_trace_context() is called before tool execution."
            )

        async with tracer.start_observation(
            name=f"mcp-sampling-{model}",
            observation_type=ObservationType.SPAN,
            parent_context=self._trace_context,
            metadata={"source": "mcp_sampling"},
        ) as sampling_ctx:
            return await self._execute_generation(
                tracer, sampling_ctx, model, litellm_messages, temperature, max_tokens
            )

    async def _execute_generation(
        self,
        tracer,
        parent_ctx,
        model: str,
        litellm_messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Execute the LLM generation with tracing."""
        gen_ctx = await tracer.start_generation(
            name="completion",
            parent_context=parent_ctx,
            model=model,
            input_messages=litellm_messages,
            metadata={"temperature": temperature},
        )

        try:
            # Lazy import to avoid polluting Temporal workflow sandbox
            from zap_ai.llm.provider import complete

            result = await complete(
                model=model,
                messages=litellm_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = result.content or ""

            await tracer.end_generation(
                context=gen_ctx,
                output={"content": content},
                usage=result.usage,
            )

            return content
        except Exception as e:
            await tracer.set_error(gen_ctx, e)
            raise

    def _get_model(self, params: Any) -> str:
        """Extract model from params or use default."""
        # Check modelPreferences.hints for model suggestions
        prefs = getattr(params, "modelPreferences", None)
        if not prefs:
            return self.default_model

        hints = getattr(prefs, "hints", None)
        if not hints or len(hints) == 0:
            return self.default_model

        # Use first hint as model name
        hint = hints[0]
        name = getattr(hint, "name", None)
        if name:
            return name

        return self.default_model

    def _convert_messages(self, messages: list[Any], params: Any) -> list[dict[str, Any]]:
        """Convert FastMCP SamplingMessage to LiteLLM format."""
        converted: list[dict[str, Any]] = []

        # Add system prompt if provided
        system_prompt = getattr(params, "systemPrompt", None)
        if system_prompt:
            converted.append({"role": "system", "content": system_prompt})

        # Convert messages
        for msg in messages:
            content = msg.content
            # Handle TextContent
            if hasattr(content, "text"):
                content = content.text
            converted.append({"role": msg.role, "content": str(content)})

        return converted


def create_mcp_client(
    source: str,
    sampling_handler: str | SamplingHandler | None = None,
    sampling_model: str = "gpt-4o",
    enable_tracing: bool = False,
    **kwargs: Any,
) -> Client:
    """
    Create a FastMCP Client with optional sampling support.

    This is a convenience wrapper that makes it easy to create MCP clients
    with Zap's built-in sampling handler shortcuts.

    Args:
        source: Client source (file path, URL, etc.).
        sampling_handler: Handler specification:
            - None: No sampling support
            - "litellm": Use LiteLLMSamplingHandler with Zap's LLM provider
            - Callable: Custom handler function
        sampling_model: Model to use with "litellm" handler.
        enable_tracing: Whether to enable Langfuse tracing for sampling calls.
        **kwargs: Additional Client constructor arguments.

    Returns:
        Configured FastMCP Client.

    Example:
        ```python
        from zap_ai import Zap, ZapAgent
        from zap_ai.mcp.sampling import create_mcp_client

        client = create_mcp_client(
            "path/to/mcp_server.py",
            sampling_handler="litellm",
            sampling_model="gpt-4o",
        )

        agent = ZapAgent(
            name="my-agent",
            prompt="You are a helpful assistant.",
            mcp_clients=[client],
        )

        zap = Zap(agents=[agent])
        await zap.start()
        ```
    """
    from fastmcp import Client

    resolved_handler = None
    if sampling_handler == "litellm":
        resolved_handler = LiteLLMSamplingHandler(
            default_model=sampling_model,
            enable_tracing=enable_tracing,
        )
    elif sampling_handler is not None:
        resolved_handler = sampling_handler

    client = Client(source, sampling_handler=resolved_handler, **kwargs)

    # Register the handler if it's a LiteLLMSamplingHandler so we can set trace context later
    if isinstance(resolved_handler, LiteLLMSamplingHandler):
        _client_handlers[id(client)] = resolved_handler

    return client
