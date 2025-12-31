"""Tests for workflow models."""

import json

from zap_ai.workflows.models import (
    AgentWorkflowInput,
    ConversationState,
    SubAgentConversation,
    SubAgentResponse,
)


class TestAgentWorkflowInput:
    """Tests for AgentWorkflowInput dataclass."""

    def test_create_minimal_input(self) -> None:
        """Test creating input with only required fields."""
        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Hello",
        )
        assert input.agent_name == "TestAgent"
        assert input.initial_task == "Hello"
        assert input.state is None
        assert input.parent_workflow_id is None

    def test_create_full_input(self) -> None:
        """Test creating input with all fields."""
        state = {"messages": [], "iteration_count": 5}
        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Hello",
            state=state,
            parent_workflow_id="parent-123",
        )
        assert input.state == state
        assert input.parent_workflow_id == "parent-123"

    def test_state_can_be_empty_dict(self) -> None:
        """Test that state can be an empty dict."""
        input = AgentWorkflowInput(
            agent_name="TestAgent",
            initial_task="Hello",
            state={},
        )
        assert input.state == {}


class TestSubAgentConversation:
    """Tests for SubAgentConversation dataclass."""

    def test_create_minimal_conversation(self) -> None:
        """Test creating a conversation with only required fields."""
        conv = SubAgentConversation(
            conversation_id="conv-123",
            agent_name="HelperAgent",
        )
        assert conv.conversation_id == "conv-123"
        assert conv.agent_name == "HelperAgent"
        assert conv.messages == []
        assert conv.is_active is True

    def test_create_full_conversation(self) -> None:
        """Test creating a conversation with all fields."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        conv = SubAgentConversation(
            conversation_id="conv-123",
            agent_name="HelperAgent",
            messages=messages,
            is_active=False,
        )
        assert conv.messages == messages
        assert conv.is_active is False


class TestSubAgentResponse:
    """Tests for SubAgentResponse dataclass."""

    def test_create_minimal_response(self) -> None:
        """Test creating a response with only required fields."""
        response = SubAgentResponse(
            conversation_id="conv-123",
            agent_name="HelperAgent",
            response="Here is my response",
        )
        assert response.conversation_id == "conv-123"
        assert response.agent_name == "HelperAgent"
        assert response.response == "Here is my response"
        assert response.is_complete is False

    def test_create_complete_response(self) -> None:
        """Test creating a response with is_complete=True."""
        response = SubAgentResponse(
            conversation_id="conv-123",
            agent_name="HelperAgent",
            response="Done!",
            is_complete=True,
        )
        assert response.is_complete is True

    def test_to_tool_result_format(self) -> None:
        """Test to_tool_result returns valid JSON."""
        response = SubAgentResponse(
            conversation_id="conv-123",
            agent_name="HelperAgent",
            response="My response",
            is_complete=False,
        )
        result = response.to_tool_result()
        parsed = json.loads(result)

        assert parsed["conversation_id"] == "conv-123"
        assert parsed["agent_name"] == "HelperAgent"
        assert parsed["response"] == "My response"
        assert parsed["is_complete"] is False

    def test_to_tool_result_escapes_special_chars(self) -> None:
        """Test to_tool_result handles special characters."""
        response = SubAgentResponse(
            conversation_id="conv-123",
            agent_name="HelperAgent",
            response='Hello "world" with\nnewlines',
            is_complete=False,
        )
        result = response.to_tool_result()
        parsed = json.loads(result)

        assert parsed["response"] == 'Hello "world" with\nnewlines'


class TestConversationState:
    """Tests for ConversationState dataclass."""

    def test_create_empty_state(self) -> None:
        """Test creating an empty state."""
        state = ConversationState()
        assert state.messages == []
        assert state.iteration_count == 0
        assert state.pending_messages == []
        assert state.sub_agent_conversations == {}

    def test_create_full_state(self) -> None:
        """Test creating a state with all fields."""
        messages = [{"role": "user", "content": "Hello"}]
        pending = ["Follow up"]
        sub_conv = SubAgentConversation(
            conversation_id="conv-123",
            agent_name="Helper",
            messages=[{"role": "user", "content": "Help"}],
            is_active=True,
        )

        state = ConversationState(
            messages=messages,
            iteration_count=5,
            pending_messages=pending,
            sub_agent_conversations={"conv-123": sub_conv},
        )

        assert state.messages == messages
        assert state.iteration_count == 5
        assert state.pending_messages == pending
        assert "conv-123" in state.sub_agent_conversations

    def test_to_dict_empty_state(self) -> None:
        """Test serializing empty state to dict."""
        state = ConversationState()
        result = state.to_dict()

        assert result["messages"] == []
        assert result["iteration_count"] == 0
        assert result["pending_messages"] == []
        assert result["sub_agent_conversations"] == {}

    def test_to_dict_full_state(self) -> None:
        """Test serializing full state to dict."""
        sub_conv = SubAgentConversation(
            conversation_id="conv-123",
            agent_name="Helper",
            messages=[{"role": "user", "content": "Help"}],
            is_active=True,
        )

        state = ConversationState(
            messages=[{"role": "user", "content": "Hello"}],
            iteration_count=3,
            pending_messages=["Follow up"],
            sub_agent_conversations={"conv-123": sub_conv},
        )

        result = state.to_dict()

        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["iteration_count"] == 3
        assert result["pending_messages"] == ["Follow up"]
        assert result["sub_agent_conversations"]["conv-123"]["conversation_id"] == "conv-123"
        assert result["sub_agent_conversations"]["conv-123"]["agent_name"] == "Helper"
        assert result["sub_agent_conversations"]["conv-123"]["is_active"] is True

    def test_from_dict_empty_state(self) -> None:
        """Test deserializing empty state from dict."""
        data: dict = {}
        state = ConversationState.from_dict(data)

        assert state.messages == []
        assert state.iteration_count == 0
        assert state.pending_messages == []
        assert state.sub_agent_conversations == {}

    def test_from_dict_full_state(self) -> None:
        """Test deserializing full state from dict."""
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "iteration_count": 3,
            "pending_messages": ["Follow up"],
            "sub_agent_conversations": {
                "conv-123": {
                    "conversation_id": "conv-123",
                    "agent_name": "Helper",
                    "messages": [{"role": "user", "content": "Help"}],
                    "is_active": False,
                }
            },
        }

        state = ConversationState.from_dict(data)

        assert state.messages == [{"role": "user", "content": "Hello"}]
        assert state.iteration_count == 3
        assert state.pending_messages == ["Follow up"]
        assert "conv-123" in state.sub_agent_conversations
        assert state.sub_agent_conversations["conv-123"].agent_name == "Helper"
        assert state.sub_agent_conversations["conv-123"].is_active is False

    def test_roundtrip_serialization(self) -> None:
        """Test that to_dict and from_dict are inverse operations."""
        sub_conv = SubAgentConversation(
            conversation_id="conv-123",
            agent_name="Helper",
            messages=[
                {"role": "user", "content": "Help me"},
                {"role": "assistant", "content": "Sure!"},
            ],
            is_active=True,
        )

        original = ConversationState(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            iteration_count=7,
            pending_messages=["More questions"],
            sub_agent_conversations={"conv-123": sub_conv},
        )

        serialized = original.to_dict()
        restored = ConversationState.from_dict(serialized)

        assert restored.messages == original.messages
        assert restored.iteration_count == original.iteration_count
        assert restored.pending_messages == original.pending_messages

        # Check sub-agent conversation was restored correctly
        restored_conv = restored.sub_agent_conversations["conv-123"]
        assert restored_conv.conversation_id == sub_conv.conversation_id
        assert restored_conv.agent_name == sub_conv.agent_name
        assert restored_conv.messages == sub_conv.messages
        assert restored_conv.is_active == sub_conv.is_active

    def test_from_dict_handles_missing_sub_conversations(self) -> None:
        """Test from_dict handles missing sub_agent_conversations key."""
        data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "iteration_count": 1,
        }

        state = ConversationState.from_dict(data)

        assert state.sub_agent_conversations == {}
        assert state.pending_messages == []
