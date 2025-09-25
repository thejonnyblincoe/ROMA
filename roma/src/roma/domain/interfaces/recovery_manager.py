"""
Recovery Manager Interface - ROMA v2.0 Domain Interface.

Abstract interface for recovery management operations that application must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.child_evaluation_result import ChildEvaluationResult
from roma.domain.value_objects.circuit_breaker_state import CircuitBreakerState
from roma.domain.value_objects.recovery_result import RecoveryResult


class IRecoveryManager(ABC):
    """
    Domain interface for recovery manager operations.

    Abstract interface that defines all recovery management operations required by the domain.
    Application layer must implement this interface to provide recovery capabilities.
    """

    @abstractmethod
    async def handle_failure(
        self, failed_node: TaskNode, error: Exception, context: dict[str, Any] | None = None
    ) -> RecoveryResult:
        """
        Handle task failure and determine recovery action.

        Args:
            failed_node: The task that failed
            error: The error that occurred
            context: Additional context for recovery decision

        Returns:
            RecoveryResult with recommended action
        """

    @abstractmethod
    async def record_success(self, task_id: str) -> None:
        """
        Record successful task execution.

        Args:
            task_id: ID of successful task
        """

    @abstractmethod
    def is_permanently_failed(self, task_id: str) -> bool:
        """
        Check if task is permanently failed.

        Args:
            task_id: Task ID to check

        Returns:
            True if permanently failed, False otherwise
        """

    @abstractmethod
    def get_circuit_breaker_state(self) -> CircuitBreakerState:
        """
        Get current circuit breaker state.

        Returns:
            Circuit breaker state (CLOSED, OPEN, HALF_OPEN)
        """

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get recovery manager statistics.

        Returns:
            Dictionary with recovery statistics
        """

    @abstractmethod
    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker to closed state."""

    @abstractmethod
    def clear_permanent_failures(self) -> None:
        """Clear all permanent failure records."""

    @abstractmethod
    def evaluate_terminal_children(
        self, parent_node: TaskNode, child_nodes: list[TaskNode]
    ) -> ChildEvaluationResult:
        """
        Evaluate terminal child nodes to determine parent recovery action.

        Args:
            parent_node: Parent task node
            child_nodes: List of child task nodes

        Returns:
            ChildEvaluationResult with evaluation details
        """
