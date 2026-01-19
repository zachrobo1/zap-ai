"""Tests for streaming event types."""

import pytest

from zap_ai.streaming.events import (
    CompletedEvent,
    ErrorEvent,
    ThinkingEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
    generate_tool_phrase,
    parse_event,
)


class TestThinkingEvent:
    """Tests for ThinkingEvent."""

    def test_create_thinking_event(self) -> None:
        """Test creating a ThinkingEvent."""
        event = ThinkingEvent(iteration=0, task_id="task-123", seq=1)
        assert event.type == "thinking"
        assert event.iteration == 0
        assert event.task_id == "task-123"
        assert event.seq == 1
        assert event.timestamp  # Should have auto-generated timestamp

    def test_default_values(self) -> None:
        """Test default values for optional parameters."""
        event = ThinkingEvent(iteration=5)
        assert event.task_id == ""
        assert event.seq == 0
        assert event.timestamp  # Should have auto-generated timestamp


class TestToolCallEvent:
    """Tests for ToolCallEvent."""

    def test_create_tool_call_event(self) -> None:
        """Test creating a ToolCallEvent."""
        event = ToolCallEvent(
            name="get_weather",
            arguments={"city": "London"},
            phrase="Getting weather for London...",
            tool_call_id="call-123",
            task_id="task-456",
            seq=2,
        )
        assert event.type == "tool_call"
        assert event.name == "get_weather"
        assert event.arguments == {"city": "London"}
        assert event.phrase == "Getting weather for London..."
        assert event.tool_call_id == "call-123"


class TestToolResultEvent:
    """Tests for ToolResultEvent."""

    def test_create_successful_result(self) -> None:
        """Test creating a successful ToolResultEvent."""
        event = ToolResultEvent(
            name="get_weather",
            result="Sunny, 20C",
            tool_call_id="call-123",
            success=True,
            task_id="task-456",
            seq=3,
        )
        assert event.type == "tool_result"
        assert event.name == "get_weather"
        assert event.result == "Sunny, 20C"
        assert event.success is True

    def test_create_failed_result(self) -> None:
        """Test creating a failed ToolResultEvent."""
        event = ToolResultEvent(
            name="get_weather",
            result="Error: API unavailable",
            tool_call_id="call-123",
            success=False,
        )
        assert event.success is False


class TestTokenEvent:
    """Tests for TokenEvent."""

    def test_create_token_event(self) -> None:
        """Test creating a TokenEvent."""
        event = TokenEvent(token="Hello", index=0)
        assert event.type == "token"
        assert event.token == "Hello"
        assert event.index == 0


class TestCompletedEvent:
    """Tests for CompletedEvent."""

    def test_create_completed_event(self) -> None:
        """Test creating a CompletedEvent."""
        event = CompletedEvent(
            result="Task completed successfully",
            task_id="task-123",
            seq=10,
        )
        assert event.type == "completed"
        assert event.result == "Task completed successfully"


class TestErrorEvent:
    """Tests for ErrorEvent."""

    def test_create_error_event(self) -> None:
        """Test creating an ErrorEvent."""
        event = ErrorEvent(
            error="Max iterations reached",
            task_id="task-123",
            seq=10,
        )
        assert event.type == "error"
        assert event.error == "Max iterations reached"


class TestGenerateToolPhrase:
    """Tests for generate_tool_phrase function."""

    def test_no_description(self) -> None:
        """Test phrase generation with no description."""
        phrase = generate_tool_phrase("get_weather", None, {"city": "London"})
        assert phrase == "Calling get_weather..."

    def test_empty_description(self) -> None:
        """Test phrase generation with empty description."""
        phrase = generate_tool_phrase("get_weather", "", {"city": "London"})
        assert phrase == "Calling get_weather..."

    def test_verb_conversion_get(self) -> None:
        """Test 'Get' to 'Getting' conversion."""
        phrase = generate_tool_phrase("get_weather", "Get weather for a city", {"city": "London"})
        assert phrase.startswith("Getting weather")
        assert phrase.endswith("...")

    def test_verb_conversion_search(self) -> None:
        """Test 'Search' to 'Searching' conversion."""
        phrase = generate_tool_phrase("search_users", "Search for users", {"query": "john"})
        assert phrase.startswith("Searching")
        assert phrase.endswith("...")

    def test_verb_conversion_create(self) -> None:
        """Test 'Create' to 'Creating' conversion."""
        phrase = generate_tool_phrase("create_user", "Create a new user", {"name": "Alice"})
        assert phrase.startswith("Creating")

    def test_verb_conversion_delete(self) -> None:
        """Test 'Delete' to 'Deleting' conversion."""
        phrase = generate_tool_phrase("delete_user", "Delete a user", {"user_id": "123"})
        assert phrase.startswith("Deleting")

    def test_verb_conversion_fetch(self) -> None:
        """Test 'Fetch' to 'Fetching' conversion."""
        phrase = generate_tool_phrase("fetch_data", "Fetch data from API", {"endpoint": "/api/v1"})
        assert phrase.startswith("Fetching")

    def test_argument_substitution_braces(self) -> None:
        """Test argument substitution with brace placeholders."""
        phrase = generate_tool_phrase("get_weather", "Get weather for {city}", {"city": "Paris"})
        assert "Paris" in phrase

    def test_argument_substitution_a_key(self) -> None:
        """Test argument substitution with 'a key' pattern."""
        phrase = generate_tool_phrase("get_weather", "Get weather for a city", {"city": "Berlin"})
        assert "Berlin" in phrase

    def test_argument_substitution_the_key(self) -> None:
        """Test argument substitution with 'the key' pattern."""
        phrase = generate_tool_phrase("get_user", "Get the user", {"user": "Alice"})
        assert "Alice" in phrase

    def test_ensures_ellipsis(self) -> None:
        """Test that phrase always ends with ellipsis."""
        phrase = generate_tool_phrase("do_something", "Do something", {})
        assert phrase.endswith("...")

    def test_handles_underscored_keys(self) -> None:
        """Test substitution with underscored key names."""
        phrase = generate_tool_phrase("get_user", "Get the user id", {"user_id": "123"})
        assert "123" in phrase


class TestParseEvent:
    """Tests for parse_event function."""

    def test_parse_thinking_event(self) -> None:
        """Test parsing a thinking event."""
        data = {
            "seq": 1,
            "type": "thinking",
            "timestamp": "2024-01-01T00:00:00Z",
            "iteration": 0,
        }
        event = parse_event(data, "task-123")
        assert isinstance(event, ThinkingEvent)
        assert event.iteration == 0
        assert event.task_id == "task-123"
        assert event.seq == 1

    def test_parse_tool_call_event(self) -> None:
        """Test parsing a tool_call event."""
        data = {
            "seq": 2,
            "type": "tool_call",
            "timestamp": "2024-01-01T00:00:00Z",
            "name": "get_weather",
            "arguments": {"city": "London"},
            "phrase": "Getting weather...",
            "tool_call_id": "call-123",
        }
        event = parse_event(data, "task-123")
        assert isinstance(event, ToolCallEvent)
        assert event.name == "get_weather"
        assert event.arguments == {"city": "London"}

    def test_parse_tool_result_event(self) -> None:
        """Test parsing a tool_result event."""
        data = {
            "seq": 3,
            "type": "tool_result",
            "timestamp": "2024-01-01T00:00:00Z",
            "name": "get_weather",
            "result": "Sunny",
            "tool_call_id": "call-123",
            "success": True,
        }
        event = parse_event(data, "task-123")
        assert isinstance(event, ToolResultEvent)
        assert event.result == "Sunny"
        assert event.success is True

    def test_parse_token_event(self) -> None:
        """Test parsing a token event."""
        data = {
            "seq": 4,
            "type": "token",
            "timestamp": "2024-01-01T00:00:00Z",
            "token": "Hello",
            "index": 0,
        }
        event = parse_event(data, "task-123")
        assert isinstance(event, TokenEvent)
        assert event.token == "Hello"
        assert event.index == 0

    def test_parse_completed_event(self) -> None:
        """Test parsing a completed event."""
        data = {
            "seq": 10,
            "type": "completed",
            "timestamp": "2024-01-01T00:00:00Z",
            "result": "Task done!",
        }
        event = parse_event(data, "task-123")
        assert isinstance(event, CompletedEvent)
        assert event.result == "Task done!"

    def test_parse_error_event(self) -> None:
        """Test parsing an error event."""
        data = {
            "seq": 10,
            "type": "error",
            "timestamp": "2024-01-01T00:00:00Z",
            "error": "Something went wrong",
        }
        event = parse_event(data, "task-123")
        assert isinstance(event, ErrorEvent)
        assert event.error == "Something went wrong"

    def test_parse_unknown_event_raises(self) -> None:
        """Test that parsing unknown event type raises ValueError."""
        data = {
            "seq": 1,
            "type": "unknown_type",
            "timestamp": "2024-01-01T00:00:00Z",
        }
        with pytest.raises(ValueError, match="Unknown event type"):
            parse_event(data, "task-123")

    def test_parse_handles_missing_optional_fields(self) -> None:
        """Test parsing with missing optional fields uses defaults."""
        data = {
            "type": "thinking",
        }
        event = parse_event(data, "task-123")
        assert isinstance(event, ThinkingEvent)
        assert event.iteration == 0
        assert event.seq == 0
