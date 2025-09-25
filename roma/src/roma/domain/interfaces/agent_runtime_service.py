"""
Agent Runtime Service Interface - ROMA v2.0 Domain Interface.

Abstract interface for agent runtime operations that application must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.result_envelope import AnyResultEnvelope
from roma.domain.value_objects.task_type import TaskType


class IAgentRuntimeService(ABC):
    """
    Domain interface for agent runtime service operations.

    Abstract interface that defines all agent runtime operations required by the domain.
    Application layer must implement this interface to provide agent runtime capabilities.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the agent runtime service.

        Performs startup sequence to initialize agent factory and create all runtime agents.
        """

    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown the agent runtime service.

        Gracefully shuts down all agents and cleans up resources.
        """

    @abstractmethod
    async def get_agent(self, task_type: TaskType, agent_type: AgentType) -> Any:
        """
        Get agent for specified task and agent type (lazy creation).

        Args:
            task_type: Type of task (RETRIEVE, WRITE, THINK, etc.)
            agent_type: Type of agent (ATOMIZER, PLANNER, EXECUTOR, etc.)

        Returns:
            Configured agent instance

        Raises:
            AgentCreationError: If agent cannot be created
        """

    @abstractmethod
    async def execute_agent(
        self,
        agent: Any,
        task: TaskNode,
        context: TaskContext | None = None,
        agent_type: AgentType | None = None,
        execution_id: str | None = None,
    ) -> AnyResultEnvelope:
        """
        Execute agent with task and context.

        Args:
            agent: Agent instance to execute
            task: Task to process
            context: Task context (optional)
            agent_type: Type of agent being executed (optional)
            execution_id: Execution identifier (optional)

        Returns:
            Result envelope containing agent output

        Raises:
            AgentExecutionError: If agent execution fails
        """

    @abstractmethod
    def get_runtime_metrics(self) -> dict[str, Any]:
        """
        Get agent runtime metrics.

        Returns:
            Dictionary with runtime metrics (agents created, executed, errors, etc.)
        """

    @abstractmethod
    def get_framework_name(self) -> str:
        """
        Get framework name for agent runtime.

        Returns:
            Framework name string
        """

    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if agent runtime service is initialized.

        Returns:
            True if initialized, False otherwise
        """
