"""Tests for ZapAgent model validation."""

import pytest
from pydantic import ValidationError

from zap_ai import ZapAgent


class TestZapAgentCreation:
    """Test ZapAgent instantiation with valid inputs."""

    def test_minimal_agent(self) -> None:
        """Test creating an agent with only required fields."""
        agent = ZapAgent(name="TestAgent", prompt="You are helpful.")
        assert agent.name == "TestAgent"
        assert agent.prompt == "You are helpful."

    def test_default_model(self) -> None:
        """Test that default model is gpt-4o."""
        agent = ZapAgent(name="TestAgent", prompt="test")
        assert agent.model == "gpt-4o"

    def test_custom_model(self) -> None:
        """Test specifying a custom model."""
        agent = ZapAgent(name="TestAgent", prompt="test", model="claude-3-opus-20240229")
        assert agent.model == "claude-3-opus-20240229"

    def test_default_max_iterations(self) -> None:
        """Test that default max_iterations is 50."""
        agent = ZapAgent(name="TestAgent", prompt="test")
        assert agent.max_iterations == 50

    def test_custom_max_iterations(self) -> None:
        """Test specifying custom max_iterations."""
        agent = ZapAgent(name="TestAgent", prompt="test", max_iterations=100)
        assert agent.max_iterations == 100

    def test_default_mcp_clients_empty(self) -> None:
        """Test that mcp_clients defaults to empty list."""
        agent = ZapAgent(name="TestAgent", prompt="test")
        assert agent.mcp_clients == []

    def test_default_sub_agents_empty(self) -> None:
        """Test that sub_agents defaults to empty list."""
        agent = ZapAgent(name="TestAgent", prompt="test")
        assert agent.sub_agents == []

    def test_default_discovery_prompt_none(self) -> None:
        """Test that discovery_prompt defaults to None."""
        agent = ZapAgent(name="TestAgent", prompt="test")
        assert agent.discovery_prompt is None

    def test_sub_agents_list(self) -> None:
        """Test setting sub_agents."""
        agent = ZapAgent(name="Main", prompt="test", sub_agents=["Helper", "Reviewer"])
        assert agent.sub_agents == ["Helper", "Reviewer"]

    def test_discovery_prompt(self) -> None:
        """Test setting discovery_prompt."""
        agent = ZapAgent(name="Helper", prompt="test", discovery_prompt="I help with research.")
        assert agent.discovery_prompt == "I help with research."


class TestZapAgentNameValidation:
    """Test name validation rules."""

    def test_valid_alphanumeric_name(self) -> None:
        """Test alphanumeric names are valid."""
        agent = ZapAgent(name="Agent123", prompt="test")
        assert agent.name == "Agent123"

    def test_valid_underscore_name(self) -> None:
        """Test underscores in names are valid."""
        agent = ZapAgent(name="my_agent", prompt="test")
        assert agent.name == "my_agent"

    def test_valid_hyphen_name(self) -> None:
        """Test hyphens in names are valid."""
        agent = ZapAgent(name="my-agent", prompt="test")
        assert agent.name == "my-agent"

    def test_valid_mixed_name(self) -> None:
        """Test names with mixed valid characters."""
        agent = ZapAgent(name="My_Agent-123", prompt="test")
        assert agent.name == "My_Agent-123"

    def test_invalid_space_in_name(self) -> None:
        """Test that spaces in name raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ZapAgent(name="Bad Name", prompt="test")
        assert "cannot contain spaces" in str(exc_info.value)

    def test_invalid_slash_in_name(self) -> None:
        """Test that forward slashes in name raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ZapAgent(name="bad/name", prompt="test")
        assert "cannot contain forward slashes" in str(exc_info.value)

    def test_invalid_special_chars(self) -> None:
        """Test that special characters raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ZapAgent(name="agent@domain", prompt="test")
        assert "invalid characters" in str(exc_info.value)

    def test_invalid_period_in_name(self) -> None:
        """Test that periods raise ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            ZapAgent(name="agent.name", prompt="test")
        assert "invalid characters" in str(exc_info.value)

    def test_empty_name_rejected(self) -> None:
        """Test that empty name is rejected."""
        with pytest.raises(ValidationError):
            ZapAgent(name="", prompt="test")

    def test_name_max_length(self) -> None:
        """Test that name exceeding 100 chars is rejected."""
        with pytest.raises(ValidationError):
            ZapAgent(name="a" * 101, prompt="test")

    def test_name_at_max_length(self) -> None:
        """Test that name of exactly 100 chars is accepted."""
        agent = ZapAgent(name="a" * 100, prompt="test")
        assert len(agent.name) == 100


class TestZapAgentPromptValidation:
    """Test prompt validation rules."""

    def test_empty_prompt_rejected(self) -> None:
        """Test that empty prompt is rejected."""
        with pytest.raises(ValidationError):
            ZapAgent(name="TestAgent", prompt="")

    def test_whitespace_only_prompt_accepted(self) -> None:
        """Test that whitespace-only prompt is technically accepted (has length)."""
        agent = ZapAgent(name="TestAgent", prompt="   ")
        assert agent.prompt == "   "


class TestZapAgentMaxIterationsValidation:
    """Test max_iterations validation rules."""

    def test_min_iterations_valid(self) -> None:
        """Test that min value of 1 is valid."""
        agent = ZapAgent(name="TestAgent", prompt="test", max_iterations=1)
        assert agent.max_iterations == 1

    def test_max_iterations_valid(self) -> None:
        """Test that max value of 500 is valid."""
        agent = ZapAgent(name="TestAgent", prompt="test", max_iterations=500)
        assert agent.max_iterations == 500

    def test_zero_iterations_rejected(self) -> None:
        """Test that 0 iterations is rejected."""
        with pytest.raises(ValidationError):
            ZapAgent(name="TestAgent", prompt="test", max_iterations=0)

    def test_negative_iterations_rejected(self) -> None:
        """Test that negative iterations is rejected."""
        with pytest.raises(ValidationError):
            ZapAgent(name="TestAgent", prompt="test", max_iterations=-1)

    def test_exceeds_max_iterations_rejected(self) -> None:
        """Test that exceeding 500 is rejected."""
        with pytest.raises(ValidationError):
            ZapAgent(name="TestAgent", prompt="test", max_iterations=501)


class TestZapAgentSubAgentsValidation:
    """Test sub_agents validation rules."""

    def test_duplicate_sub_agents_rejected(self) -> None:
        """Test that duplicate sub-agent names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ZapAgent(name="Main", prompt="test", sub_agents=["Helper", "Helper"])
        assert "Duplicate sub-agent references" in str(exc_info.value)

    def test_multiple_duplicates_rejected(self) -> None:
        """Test that multiple duplicate names are all reported."""
        with pytest.raises(ValidationError) as exc_info:
            ZapAgent(
                name="Main",
                prompt="test",
                sub_agents=["A", "B", "A", "C", "B"],
            )
        assert "Duplicate sub-agent references" in str(exc_info.value)

    def test_unique_sub_agents_accepted(self) -> None:
        """Test that unique sub-agent names are accepted."""
        agent = ZapAgent(name="Main", prompt="test", sub_agents=["Helper", "Reviewer", "Writer"])
        assert len(agent.sub_agents) == 3


class TestZapAgentDynamicPrompt:
    """Test dynamic prompt functionality."""

    def test_static_prompt_is_not_dynamic(self) -> None:
        """Test that static string prompt is not considered dynamic."""
        agent = ZapAgent(name="Test", prompt="You are helpful.")
        assert not agent.is_dynamic_prompt()

    def test_callable_prompt_is_dynamic(self) -> None:
        """Test that callable prompt is considered dynamic."""
        agent = ZapAgent(
            name="Test",
            prompt=lambda ctx: f"You assist {ctx['name']}.",
        )
        assert agent.is_dynamic_prompt()

    def test_resolve_static_prompt(self) -> None:
        """Test resolving a static prompt returns the string."""
        agent = ZapAgent(name="Test", prompt="You are helpful.")
        assert agent.resolve_prompt({}) == "You are helpful."
        assert agent.resolve_prompt({"key": "value"}) == "You are helpful."

    def test_resolve_callable_prompt_with_dict(self) -> None:
        """Test resolving a callable prompt with dict context."""
        agent = ZapAgent(
            name="Test",
            prompt=lambda ctx: f"You assist {ctx['name']} from {ctx['company']}.",
        )
        result = agent.resolve_prompt({"name": "Alice", "company": "Acme"})
        assert result == "You assist Alice from Acme."

    def test_resolve_callable_prompt_with_typed_context(self) -> None:
        """Test resolving a callable prompt with typed context."""
        from dataclasses import dataclass

        @dataclass
        class MyContext:
            user_name: str
            company: str

        agent: ZapAgent[MyContext] = ZapAgent(
            name="Test",
            prompt=lambda ctx: f"You assist {ctx.user_name} from {ctx.company}.",
        )
        result = agent.resolve_prompt(MyContext(user_name="Bob", company="TechCo"))
        assert result == "You assist Bob from TechCo."

    def test_callable_prompt_with_empty_dict_default(self) -> None:
        """Test callable prompt that handles empty dict gracefully."""
        agent = ZapAgent(
            name="Test",
            prompt=lambda ctx: f"User: {ctx.get('name', 'unknown')}",
        )
        assert agent.resolve_prompt({}) == "User: unknown"
        assert agent.resolve_prompt({"name": "Alice"}) == "User: Alice"

    def test_non_string_non_callable_rejected(self) -> None:
        """Test that non-string, non-callable prompt is rejected."""
        with pytest.raises(ValidationError):
            ZapAgent(name="Test", prompt=123)  # type: ignore[arg-type]

    def test_none_prompt_rejected(self) -> None:
        """Test that None prompt is rejected."""
        with pytest.raises(ValidationError):
            ZapAgent(name="Test", prompt=None)  # type: ignore[arg-type]
