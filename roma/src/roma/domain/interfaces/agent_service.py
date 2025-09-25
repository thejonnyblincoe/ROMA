"""
Agent Service Interface.

Simple unified interface for all agent services using existing domain types.
Each service has a single `run` method that handles their specific agent type.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.result_envelope import AnyResultEnvelope


class BaseAgentServiceInterface(ABC):
    """Base interface for all agent services with unified run method."""

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the agent type this service handles."""

    @abstractmethod
    async def run(self, task: TaskNode, context: TaskContext, **kwargs: Any) -> NodeResult:
        """
        Run the agent service operation.

        Args:
            task: The task node to process
            context: Task execution context (contains execution_id)
            **kwargs: Service-specific parameters

        Returns:
            NodeResult with appropriate action and envelope
        """

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return {"agent_type": self.agent_type.value, "service_name": self.__class__.__name__}


class AtomizerServiceInterface(BaseAgentServiceInterface):
    """Interface for Atomizer agents."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.ATOMIZER


class PlannerServiceInterface(BaseAgentServiceInterface):
    """Interface for Planner agents."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.PLANNER


class ExecutorServiceInterface(BaseAgentServiceInterface):
    """Interface for Executor agents."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.EXECUTOR


class AggregatorServiceInterface(BaseAgentServiceInterface):
    """Interface for Aggregator agents."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.AGGREGATOR

    @abstractmethod
    async def run(
        self,
        task: TaskNode,
        context: TaskContext,
        child_envelopes: list[AnyResultEnvelope] | None = None,
        is_partial: bool = False,
        **kwargs: Any,
    ) -> NodeResult:
        """
        Run aggregation with child results.

        Args:
            task: Parent task being aggregated
            context: Task context (contains execution_id)
            child_envelopes: Child result envelopes
            is_partial: Whether this is partial aggregation
            **kwargs: Additional parameters
        """


class PlanModifierServiceInterface(BaseAgentServiceInterface):
    """Interface for Plan Modifier agents."""

    @property
    def agent_type(self) -> AgentType:
        return AgentType.PLAN_MODIFIER

    @abstractmethod
    async def run(
        self,
        task: TaskNode,
        context: TaskContext,
        failed_children: list[TaskNode] | None = None,
        failure_reason: str | None = None,
        **kwargs: Any,
    ) -> NodeResult:
        """
        Run plan modification based on failures.

        Args:
            task: Original parent task
            context: Task context (contains execution_id)
            failed_children: Failed child tasks
            failure_reason: Reason for replanning
            **kwargs: Additional parameters
        """
