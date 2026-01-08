"""Validation utilities for Zap configuration.

This module provides validation functions for agent configurations,
extracted from the Zap class for better separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from zap_ai.exceptions import ZapConfigurationError

if TYPE_CHECKING:
    from zap_ai.core.agent import ZapAgent


def validate_no_duplicate_names(agents: list["ZapAgent"]) -> None:
    """
    Check that all agent names are unique.

    Args:
        agents: List of ZapAgent configurations.

    Raises:
        ZapConfigurationError: If duplicate names found.
    """
    names = [agent.name for agent in agents]
    if len(names) != len(set(names)):
        duplicates = [name for name in names if names.count(name) > 1]
        raise ZapConfigurationError(
            f"Duplicate agent names detected: {set(duplicates)}. "
            "Each agent must have a unique name."
        )


def build_agent_map(agents: list["ZapAgent"]) -> dict[str, "ZapAgent"]:
    """
    Build internal name -> agent lookup map.

    Args:
        agents: List of ZapAgent configurations.

    Returns:
        Dict mapping agent names to agent instances.
    """
    return {agent.name: agent for agent in agents}


def validate_sub_agent_references(
    agents: list["ZapAgent"],
    agent_map: dict[str, "ZapAgent"],
) -> None:
    """
    Validate that all sub_agent references point to existing agents.

    Args:
        agents: List of ZapAgent configurations.
        agent_map: Dict mapping agent names to instances.

    Raises:
        ZapConfigurationError: If any sub-agent reference is invalid.
    """
    all_names = set(agent_map.keys())
    for agent in agents:
        for sub_name in agent.sub_agents:
            if sub_name not in all_names:
                raise ZapConfigurationError(
                    f"Agent '{agent.name}' references unknown sub-agent '{sub_name}'. "
                    f"Available agents: {sorted(all_names)}"
                )
            if sub_name == agent.name:
                raise ZapConfigurationError(
                    f"Agent '{agent.name}' cannot reference itself as a sub-agent."
                )


def validate_no_circular_dependencies(
    agents: list["ZapAgent"],
    agent_map: dict[str, "ZapAgent"],
) -> None:
    """
    Detect circular dependencies in sub-agent relationships.

    Uses DFS to find cycles in the agent dependency graph.

    Args:
        agents: List of ZapAgent configurations.
        agent_map: Dict mapping agent names to instances.

    Raises:
        ZapConfigurationError: If a circular dependency is detected.
    """
    # Build adjacency list
    graph: dict[str, list[str]] = {agent.name: agent.sub_agents for agent in agents}

    # Track visited and recursion stack for cycle detection
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(node: str) -> bool:
        """Return True if cycle detected."""
        visited.add(node)
        rec_stack.add(node)
        path.append(node)

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                # Found cycle - build cycle path for error message
                cycle_start = path.index(neighbor)
                cycle = path[cycle_start:] + [neighbor]
                raise ZapConfigurationError(f"Circular dependency detected: {' -> '.join(cycle)}")

        path.pop()
        rec_stack.remove(node)
        return False

    for agent_name in graph:
        if agent_name not in visited:
            dfs(agent_name)
