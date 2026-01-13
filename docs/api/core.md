# Core API

The core module contains the main classes for building and running agents.

## Zap

The main orchestrator class that manages agents and Temporal connections.

::: zap_ai.Zap
    options:
      show_source: true
      members:
        - start
        - stop
        - execute_task
        - get_task
        - get_agent
        - get_agent_tools

## ZapAgent

Configuration for an AI agent.

::: zap_ai.ZapAgent
    options:
      show_source: true
      members:
        - is_dynamic_prompt
        - resolve_prompt

## Task

Represents a task execution.

::: zap_ai.Task
    options:
      show_source: true
      members:
        - is_complete
        - is_successful
        - get_last_message
        - get_assistant_messages
        - get_tool_calls_count
        - get_text_content
        - get_tool_calls
        - get_turns
        - get_turn
        - turn_count
        - get_sub_tasks
        - get_pending_approvals
        - approve
        - reject

## TaskStatus

Enum for task execution states.

::: zap_ai.TaskStatus
    options:
      show_source: true
      members:
        - is_terminal
        - is_active
