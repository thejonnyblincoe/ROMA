"""
Configurable Agent Interface - ROMA v2.0 Domain Interface.

Abstract interface for configurable agent operations that infrastructure must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional, TypeVar

from roma.domain.context.task_context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.config.agent_config import AgentConfig

T = TypeVar("T")


class IConfigurableAgent[T](ABC):
    """
    Domain interface for configurable agent operations.

    Abstract interface that defines all agent operations required by the domain.
    Infrastructure layer must implement this interface to provide agent capabilities.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the agent's name."""

    @abstractmethod
    async def run(self, task: TaskNode, context: Optional["TaskContext"] = None) -> T:
        """
        Execute agent with task and optional context.

        Args:
            task: TaskNode to process
            context: Optional TaskContext with rich execution context

        Returns:
            Structured result of type T (specific to agent type)

        Raises:
            AgentExecutionError: If agent execution fails
        """

    @abstractmethod
    def get_config(self) -> AgentConfig:
        """
        Get the agent's configuration.

        Returns:
            AgentConfig object with agent settings
        """

    @abstractmethod
    def get_agent_type(self) -> str:
        """
        Get the agent's type identifier.

        Returns:
            String identifying the agent type (e.g., 'atomizer', 'planner', etc.)
        """

    @abstractmethod
    def get_task_type(self) -> str | None:
        """
        Get the task type this agent handles.

        Returns:
            Optional task type string, or None if agent handles multiple types
        """

    def set_execution_context(self, _execution_id: str) -> None:
        """
        Set execution context for session isolation (optional method).

        Args:
            execution_id: Execution identifier for session isolation

        Note:
            This method is optional and may not be implemented by all agents.
            Implementations should provide a no-op version if not supported.
        """
        # Default no-op implementation for agents that don't support execution context
        return
