"""
Checkpoint Service with Clean Architecture.

Application layer service that orchestrates checkpoint operations using repository interfaces.
"""

import asyncio
import gzip
import logging
import pickle
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from roma.domain.interfaces.persistence import CheckpointRepository, RecoveryRepository
from roma.domain.value_objects.persistence import (
    CheckpointAnalytics,
    CheckpointRecord,
    CheckpointSummary,
    CheckpointType,
    RecoveryStatus,
)

logger = logging.getLogger(__name__)


class CheckpointService:
    """
    Application service for managing execution checkpoints and recovery.

    Uses repository interfaces for persistence operations following Clean Architecture.
    Provides high-level checkpoint orchestration without persistence implementation details.
    """

    def __init__(
        self, checkpoint_repository: CheckpointRepository, recovery_repository: RecoveryRepository
    ):
        """
        Initialize checkpoint service with repository dependencies.

        Args:
            checkpoint_repository: Repository for checkpoint persistence
            recovery_repository: Repository for recovery persistence
        """
        self.checkpoint_repo = checkpoint_repository
        self.recovery_repo = recovery_repository
        self._stats = {
            "checkpoints_created": 0,
            "checkpoints_loaded": 0,
            "recovery_operations": 0,
            "data_compressed_bytes": 0,
        }
        self._lock = asyncio.Lock()

    async def create_checkpoint(
        self,
        execution_id: str,
        checkpoint_name: str,
        checkpoint_data: dict[str, Any],
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
        task_graph_snapshot: dict[str, Any] | None = None,
        execution_context: dict[str, Any] | None = None,
        agent_states: dict[str, Any] | None = None,
        large_data: Any | None = None,
        expires_in_hours: int | None = None,
    ) -> str:
        """
        Create a new execution checkpoint.

        Args:
            execution_id: Execution ID this checkpoint belongs to
            checkpoint_name: Human-readable name
            checkpoint_data: Main checkpoint data
            checkpoint_type: Type of checkpoint
            task_graph_snapshot: Task graph state
            execution_context: Execution context
            agent_states: Agent states
            large_data: Large data to be pickled and compressed
            expires_in_hours: Hours until checkpoint expires

        Returns:
            Checkpoint ID
        """
        # Process large data if provided
        processed_large_data = None
        if large_data is not None:
            serialized = pickle.dumps(large_data)
            compressed_data = gzip.compress(serialized)
            processed_large_data = compressed_data

            async with self._lock:
                self._stats["data_compressed_bytes"] += len(compressed_data)

        # Calculate expiration
        expires_at = None
        if expires_in_hours:
            expires_at = (datetime.now(UTC) + timedelta(hours=expires_in_hours)).isoformat()

        # Get next sequence number
        sequence_number = await self.checkpoint_repo.get_next_sequence_number(execution_id)

        # Calculate data size
        data_size_bytes = len(str(checkpoint_data).encode()) + (
            len(processed_large_data) if processed_large_data else 0
        )

        # Create checkpoint record
        checkpoint_record = CheckpointRecord(
            id=str(uuid4()),
            execution_id=execution_id,
            checkpoint_name=checkpoint_name,
            checkpoint_type=checkpoint_type,
            checkpoint_data=checkpoint_data,
            large_data=processed_large_data,
            task_graph_snapshot=task_graph_snapshot,
            execution_context=execution_context,
            agent_states=agent_states,
            sequence_number=sequence_number,
            created_at=datetime.now(UTC).isoformat(),
            expires_at=expires_at,
            data_size_bytes=data_size_bytes,
        )

        # Create checkpoint via repository
        checkpoint_id = await self.checkpoint_repo.create_checkpoint(checkpoint_record)

        async with self._lock:
            self._stats["checkpoints_created"] += 1

        logger.info(f"Created checkpoint {checkpoint_name} for execution {execution_id}")
        return checkpoint_id

    async def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        """
        Get checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint record or None if not found
        """
        checkpoint_record = await self.checkpoint_repo.get_checkpoint(checkpoint_id)

        if checkpoint_record:
            async with self._lock:
                self._stats["checkpoints_loaded"] += 1

        return checkpoint_record

    async def get_latest_checkpoint(
        self, execution_id: str, checkpoint_type: CheckpointType | None = None
    ) -> CheckpointRecord | None:
        """
        Get the latest checkpoint for an execution.

        Args:
            execution_id: Execution ID
            checkpoint_type: Optional checkpoint type filter

        Returns:
            Latest checkpoint record or None
        """
        return await self.checkpoint_repo.get_latest_checkpoint(execution_id, checkpoint_type)

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
        return await self.checkpoint_repo.list_checkpoints(execution_id, include_expired)

    async def start_recovery(
        self,
        task_id: str,
        execution_id: str,
        checkpoint_id: str | None = None,
        recovery_strategy: str = "default",
        error_context: dict[str, Any] | None = None,
    ) -> str:
        """
        Start a recovery operation.

        Args:
            task_id: Task ID being recovered
            execution_id: Execution ID being recovered
            checkpoint_id: Checkpoint to recover from
            recovery_strategy: Recovery strategy name
            error_context: Error context that triggered recovery

        Returns:
            Recovery ID
        """
        # Get next attempt number
        attempt_number = await self.recovery_repo.get_next_attempt_number(task_id)

        # Create recovery via repository
        recovery_id = await self.recovery_repo.create_recovery(
            task_id=task_id,
            execution_id=execution_id,
            checkpoint_id=checkpoint_id,
            recovery_strategy=recovery_strategy,
            error_context=error_context,
            attempt_number=attempt_number,
        )

        async with self._lock:
            self._stats["recovery_operations"] += 1

        logger.info(f"Started recovery operation for task {task_id}, attempt {attempt_number}")
        return recovery_id

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
        await self.recovery_repo.update_recovery_status(
            recovery_id=recovery_id,
            status=status,
            state_data=state_data,
            recovery_result=recovery_result,
            failure_reason=failure_reason,
        )
        logger.debug(f"Updated recovery {recovery_id} status to {status}")

    async def invalidate_checkpoint(self, checkpoint_id: str, reason: str = "Invalidated") -> None:
        """
        Invalidate a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to invalidate
            reason: Reason for invalidation
        """
        await self.checkpoint_repo.invalidate_checkpoint(checkpoint_id)
        logger.info(f"Invalidated checkpoint {checkpoint_id}: {reason}")

    async def cleanup_expired_checkpoints(self) -> int:
        """
        Clean up expired checkpoints.

        Returns:
            Number of checkpoints cleaned up
        """
        cleaned_count = await self.checkpoint_repo.cleanup_expired_checkpoints()
        logger.info(f"Cleaned up {cleaned_count} expired checkpoints")
        return cleaned_count

    async def cleanup_old_checkpoints(self, days: int = 30) -> int:
        """
        Clean up old checkpoints.

        Args:
            days: Number of days to keep

        Returns:
            Number of checkpoints deleted
        """
        deleted_count = await self.checkpoint_repo.cleanup_old_checkpoints(days)
        logger.info(f"Deleted {deleted_count} checkpoints older than {days} days")
        return deleted_count

    async def get_checkpoint_analytics(self) -> CheckpointAnalytics:
        """Get checkpoint usage analytics."""
        analytics = await self.checkpoint_repo.get_checkpoint_analytics()
        return analytics

    def get_stats(self) -> dict[str, Any]:
        """Get service statistics."""
        return dict(self._stats)
