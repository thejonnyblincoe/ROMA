"""
Agent Factory Interface - ROMA v2.0 Domain Interface.

Abstract interface for agent creation that infrastructure must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.task_type import TaskType


class IAgentFactory(ABC):
    """
    Domain interface for agent factory operations.

    Abstract interface that defines all agent creation operations required by the domain.
    Infrastructure layer must implement this interface to provide agent creation capabilities.
    """

    @abstractmethod
    async def create_agent(
        self,
        task_type: TaskType,
        agent_type: AgentType,
        config_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """
        Create an agent instance for the specified task and agent type.

        Args:
            task_type: The type of task the agent will handle
            agent_type: The type of agent to create (atomizer, planner, executor, etc.)
            config_overrides: Optional configuration overrides for the agent

        Returns:
            Configured agent instance ready for execution

        Raises:
            AgentCreationError: If agent creation fails
            UnsupportedAgentTypeError: If agent type is not supported
        """

    @abstractmethod
    def is_agent_available(self, task_type: TaskType, agent_type: AgentType) -> bool:
        """
        Check if an agent is available for the specified task and agent type.

        Args:
            task_type: The type of task
            agent_type: The type of agent

        Returns:
            True if agent is available, False otherwise
        """

    @abstractmethod
    def get_supported_task_types(self) -> list[TaskType]:
        """
        Get list of supported task types.

        Returns:
            List of supported TaskType values
        """

    @abstractmethod
    def get_supported_agent_types(self, task_type: TaskType) -> list[AgentType]:
        """
        Get list of supported agent types for a specific task type.

        Args:
            task_type: The task type to check

        Returns:
            List of supported AgentType values for the task type
        """

    @abstractmethod
    def get_agent_config(self, task_type: TaskType, agent_type: AgentType) -> dict[str, Any]:
        """
        Get configuration for a specific agent type.

        Args:
            task_type: The task type
            agent_type: The agent type

        Returns:
            Agent configuration dictionary

        Raises:
            AgentNotFoundError: If agent configuration is not found
        """
