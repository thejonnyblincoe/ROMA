"""
Abstract Agent Interface for ROMA v2.0

Framework-agnostic agent interface supporting generic typing for structured outputs.
All agents follow the same interface with different response types.
"""

from abc import abstractmethod
from typing import Any, Protocol, TypeVar

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_responses import (
    AggregatorResult,
    AtomizerResult,
    ExecutorResult,
    PlanModifierResult,
    PlannerResult,
)

T = TypeVar("T")


class Agent[T](Protocol):
    """
    Abstract agent interface for all ROMA agents.

    Uses generic typing to ensure type safety for different agent response types.
    All agents follow the same interface but return different structured outputs.
    """

    @abstractmethod
    async def run(self, task: TaskNode, context: dict[str, Any]) -> T:
        """
        Execute agent with task and context.

        Args:
            task: TaskNode to process
            context: Execution context including relevant results, constraints, etc.

        Returns:
            Structured result of type T (specific to agent type)
        """
        ...


# Type aliases for cleaner code and documentation
AtomizerAgent = Agent[AtomizerResult]
"""Agent that determines if a task is atomic or needs decomposition"""

PlannerAgent = Agent[PlannerResult]
"""Agent that decomposes complex tasks into executable subtasks"""

ExecutorAgent = Agent[ExecutorResult]
"""Agent that executes atomic tasks and returns results"""

AggregatorAgent = Agent[AggregatorResult]
"""Agent that synthesizes multiple subtask results"""

PlanModifierAgent = Agent[PlanModifierResult]
"""Agent that modifies existing plans based on feedback"""


# Agent type registry for factory pattern
AGENT_TYPES = {
    "atomizer": AtomizerResult,
    "planner": PlannerResult,
    "executor": ExecutorResult,
    "aggregator": AggregatorResult,
    "plan_modifier": PlanModifierResult,
}


def get_agent_result_type(agent_type: str) -> type:
    """Get the result type for a given agent type."""
    if agent_type not in AGENT_TYPES:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return AGENT_TYPES[agent_type]
