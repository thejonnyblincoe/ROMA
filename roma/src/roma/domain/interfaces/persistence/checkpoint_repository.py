"""
Checkpoint Repository Interface for Clean Architecture.

Defines the repository interface for checkpoint persistence operations.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.value_objects.persistence import (
    CheckpointAnalytics,
    CheckpointRecord,
    CheckpointSummary,
    CheckpointType,
    RecoveryStatus,
)


class CheckpointRepository(ABC):
    """
    Abstract repository interface for checkpoint persistence.

    Defines operations for creating, retrieving, and managing execution checkpoints
    and recovery states without exposing persistence implementation details.
    """

    @abstractmethod
    async def create_checkpoint(self, checkpoint_record: CheckpointRecord) -> str:
        """
        Create a new checkpoint.

        Args:
            checkpoint_record: Checkpoint record to create

        Returns:
            Checkpoint ID
        """

    @abstractmethod
    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        """
        Get checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint record or None if not found
        """

    @abstractmethod
    async def get_latest_checkpoint(
        self, execution_id: str, checkpoint_type: CheckpointType | None = None
    ) -> CheckpointRecord | None:
        """
        Get the latest checkpoint for an execution.

        Args:
            execution_id: Execution ID
            checkpoint_type: Optional checkpoint type filter

        Returns:
            Latest checkpoint or None
        """

    @abstractmethod
    async def list_checkpoints(
        self, execution_id: str, include_expired: bool = False
    ) -> list[CheckpointSummary]:
        """
        List all checkpoints for an execution.

        Args:
            execution_id: Execution ID
            include_expired: Whether to include expired checkpoints

        Returns:
            List of checkpoint summaries
        """

    @abstractmethod
    async def get_next_sequence_number(self, execution_id: str) -> int:
        """
        Get next sequence number for an execution.

        Args:
            execution_id: Execution ID

        Returns:
            Next sequence number
        """

    @abstractmethod
    async def invalidate_checkpoint(self, checkpoint_id: str) -> None:
        """
        Invalidate a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to invalidate
        """

    @abstractmethod
    async def cleanup_expired_checkpoints(self) -> int:
        """
        Clean up expired checkpoints.

        Returns:
            Number of checkpoints cleaned up
        """

    @abstractmethod
    async def cleanup_old_checkpoints(self, days: int = 30) -> int:
        """
        Clean up old checkpoints.

        Args:
            days: Number of days to keep

        Returns:
            Number of checkpoints deleted
        """

    @abstractmethod
    async def get_checkpoint_analytics(self) -> CheckpointAnalytics:
        """
        Get checkpoint usage analytics.

        Returns:
            Checkpoint analytics
        """


class RecoveryRepository(ABC):
    """
    Abstract repository interface for recovery state persistence.
    """

    @abstractmethod
    async def create_recovery(
        self,
        task_id: str,
        execution_id: str,
        checkpoint_id: str | None,
        recovery_strategy: str,
        error_context: dict[str, Any] | None,
        attempt_number: int,
    ) -> str:
        """
        Create a recovery state.

        Args:
            task_id: Task ID being recovered
            execution_id: Execution ID being recovered
            checkpoint_id: Checkpoint to recover from
            recovery_strategy: Recovery strategy name
            error_context: Error context that triggered recovery
            attempt_number: Recovery attempt number

        Returns:
            Recovery ID
        """

    @abstractmethod
    async def update_recovery_status(
        self,
        recovery_id: str,
        status: RecoveryStatus,
        state_data: dict[str, Any] | None = None,
        recovery_result: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """
        Update recovery operation status.

        Args:
            recovery_id: Recovery ID
            status: New recovery status
            state_data: Updated state data
            recovery_result: Recovery result
            failure_reason: Failure reason if unsuccessful
        """

    @abstractmethod
    async def get_next_attempt_number(self, task_id: str) -> int:
        """
        Get next attempt number for a task.

        Args:
            task_id: Task ID

        Returns:
            Next attempt number
        """
