"""Agent configuration model for Zap."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic

from pydantic import BaseModel, ConfigDict, Field, field_validator

from zap_ai.core.types import TContext

if TYPE_CHECKING:
    from fastmcp import Client  # noqa: F401


class ZapAgent(BaseModel, Generic[TContext]):
    """
    Configuration for an AI agent within the Zap platform.

    ZapAgent defines all the properties needed to run an agent, including
    its system prompt (static or dynamic), LLM model, available tools
    (via MCP clients), and which other agents it can delegate to.

    The prompt can be either:
    - A static string: "You are a helpful assistant."
    - A callable that receives context: lambda ctx: f"You assist {ctx['user_name']}."

    Example:
        ```python
        from zap_ai import ZapAgent
        from fastmcp import Client

        # Static prompt
        agent = ZapAgent(
            name="ResearchAgent",
            prompt="You are a research assistant...",
            model="gpt-4o",
            mcp_clients=[Client("./tools.py")],
            sub_agents=["WriterAgent"],
        )

        # Dynamic prompt with context
        agent = ZapAgent[MyContext](
            name="PersonalAgent",
            prompt=lambda ctx: f"You are {ctx.user_name}'s assistant.",
        )
        ```

    Attributes:
        name: Unique identifier for the agent. Used as workflow ID prefix.
            Cannot contain spaces or special characters that would be
            invalid in a Temporal workflow ID.
        prompt: System prompt that defines the agent's behavior and personality.
            Can be a string or a callable that takes context and returns a string.
            This is sent as the first message in every conversation.
        model: LiteLLM model identifier (e.g., "gpt-4o", "claude-3-opus-20240229",
            "anthropic/claude-3-sonnet"). See LiteLLM docs for full list.
        mcp_clients: List of FastMCP Client instances that provide tools to
            this agent. Clients are connected during Zap.start().
        sub_agents: List of agent names that this agent can delegate to.
            A special "message_agent" tool is automatically added when
            this list is non-empty. Referenced agents must exist in the
            Zap instance.
        discovery_prompt: Description shown to parent agents when they can
            delegate to this agent. Used in the message_agent tool
            description. If None, agent won't appear in transfer tool options.
        max_iterations: Maximum number of agentic loop iterations before
            forcing completion. Prevents infinite loops. Each iteration
            is one LLM call + optional tool execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Required fields
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the agent (no spaces allowed)",
    )
    prompt: str | Callable[[Any], str] = Field(
        ...,
        description="System prompt - static string or callable(context) -> str",
    )

    # Optional fields with defaults
    model: str = Field(
        default="gpt-4o",
        min_length=1,
        description="LiteLLM model identifier",
    )
    # Note: Using Any instead of Client due to Temporal sandbox restrictions.
    # Importing fastmcp.Client at runtime causes beartype circular import issues
    # when Temporal validates workflows. The Client type is available for static
    # type checking via TYPE_CHECKING import above.
    mcp_clients: list[Any] = Field(
        default_factory=list,
        description="FastMCP clients providing tools to this agent",
    )
    sub_agents: list[str] = Field(
        default_factory=list,
        description="Names of agents this agent can delegate to",
    )
    discovery_prompt: str | None = Field(
        default=None,
        description="Description for parent agents (shown in message_agent tool)",
    )
    max_iterations: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum agentic loop iterations",
    )

    @field_validator("name")
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """
        Validate that name is suitable for use as a Temporal workflow ID prefix.

        Rules:
        - No spaces (would break workflow ID parsing)
        - No forward slashes (used as delimiter in some contexts)
        - Must be alphanumeric with underscores/hyphens only

        Raises:
            ValueError: If name contains invalid characters.
        """
        if " " in v:
            raise ValueError(
                f"Agent name cannot contain spaces: '{v}'. Use underscores or hyphens instead."
            )
        if "/" in v:
            raise ValueError(f"Agent name cannot contain forward slashes: '{v}'.")
        # Allow alphanumeric, underscore, hyphen
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        invalid_chars = set(v) - allowed_chars
        if invalid_chars:
            raise ValueError(
                f"Agent name contains invalid characters: {invalid_chars}. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str | Callable[[Any], str]) -> str | Callable[[Any], str]:
        """Validate prompt - string must be non-empty, callable must be callable."""
        if isinstance(v, str):
            if not v:
                raise ValueError("Prompt string cannot be empty")
            return v
        if callable(v):
            return v
        raise ValueError("Prompt must be a string or callable")

    @field_validator("sub_agents")
    @classmethod
    def validate_sub_agents_no_duplicates(cls, v: list[str]) -> list[str]:
        """Ensure no duplicate sub-agent references."""
        if len(v) != len(set(v)):
            duplicates = [name for name in v if v.count(name) > 1]
            raise ValueError(f"Duplicate sub-agent references: {set(duplicates)}")
        return v

    def is_dynamic_prompt(self) -> bool:
        """Check if this agent uses a dynamic (callable) prompt."""
        return callable(self.prompt)

    def resolve_prompt(self, context: TContext) -> str:
        """
        Resolve the prompt with the given context.

        Args:
            context: The context to pass to a dynamic prompt.

        Returns:
            The resolved system prompt string.

        Raises:
            TypeError: If prompt resolution fails.
        """
        if callable(self.prompt):
            return self.prompt(context)
        return self.prompt
