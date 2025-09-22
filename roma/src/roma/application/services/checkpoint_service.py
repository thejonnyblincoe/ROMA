"""
Checkpoint Service for PostgreSQL Persistence
"""

import asyncio
import logging
import pickle
import gzip
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Union
from uuid import uuid4

from sqlalchemy import select, delete, update, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from src.roma.infrastructure.persistence.models.checkpoint_model import (
    ExecutionCheckpointModel,
    RecoveryStateModel,
    CheckpointType,
    RecoveryStatus
)

logger = logging.getLogger(__name__)


class CheckpointService:
    """
    Service for managing execution checkpoints and recovery.

    Features:
    - Create execution state checkpoints
    - Store and compress large state data
    - Manage checkpoint lifecycle and expiration
    - Support recovery operations
    - Cleanup and archival
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """
        Initialize checkpoint service.

        Args:
            session_factory: SQLAlchemy async session factory
        """
        self.session_factory = session_factory
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
        checkpoint_data: Dict[str, Any],
        checkpoint_type: CheckpointType = CheckpointType.AUTOMATIC,
        task_graph_snapshot: Optional[Dict[str, Any]] = None,
        execution_context: Optional[Dict[str, Any]] = None,
        agent_states: Optional[Dict[str, Any]] = None,
        large_data: Optional[Any] = None,
        expires_in_hours: Optional[int] = None
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
        start_time = datetime.now(timezone.utc)

        try:
            async with self.session_factory() as session:
                # Process large data if provided
                compressed_data = None
                if large_data is not None:
                    serialized = pickle.dumps(large_data)
                    compressed_data = gzip.compress(serialized)

                    async with self._lock:
                        self._stats["data_compressed_bytes"] += len(compressed_data)

                # Calculate expiration
                expires_at = None
                if expires_in_hours:
                    expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)

                # Get next sequence number
                seq_query = select(func.max(ExecutionCheckpointModel.sequence_number)).where(
                    ExecutionCheckpointModel.execution_id == execution_id
                )
                seq_result = await session.execute(seq_query)
                max_seq = seq_result.scalar() or 0
                sequence_number = max_seq + 1

                # Create checkpoint
                checkpoint = ExecutionCheckpointModel(
                    execution_id=execution_id,
                    checkpoint_name=checkpoint_name,
                    checkpoint_type=checkpoint_type,
                    checkpoint_data=checkpoint_data,
                    large_data=compressed_data,
                    task_graph_snapshot=task_graph_snapshot,
                    execution_context=execution_context,
                    agent_states=agent_states,
                    sequence_number=sequence_number,
                    expires_at=expires_at,
                    data_size_bytes=len(str(checkpoint_data).encode()) + (len(compressed_data) if compressed_data else 0)
                )

                session.add(checkpoint)
                await session.commit()

                # Calculate creation duration
                creation_duration = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

                # Update creation duration
                await session.execute(
                    update(ExecutionCheckpointModel)
                    .where(ExecutionCheckpointModel.id == checkpoint.id)
                    .values(creation_duration_ms=int(creation_duration))
                )
                await session.commit()

                async with self._lock:
                    self._stats["checkpoints_created"] += 1

                logger.info(f"Created checkpoint {checkpoint_name} for execution {execution_id}")
                return checkpoint.id

        except SQLAlchemyError as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get checkpoint by ID.

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint data or None if not found
        """
        try:
            async with self.session_factory() as session:
                query = select(ExecutionCheckpointModel).where(
                    and_(
                        ExecutionCheckpointModel.id == checkpoint_id,
                        ExecutionCheckpointModel.is_valid == True
                    )
                )

                result = await session.execute(query)
                checkpoint = result.scalar_one_or_none()

                if not checkpoint:
                    return None

                # Check expiration
                if checkpoint.expires_at and checkpoint.expires_at < datetime.now(timezone.utc):
                    logger.warning(f"Checkpoint {checkpoint_id} has expired")
                    return None

                # Decompress large data if present
                large_data = None
                if checkpoint.large_data:
                    try:
                        decompressed = gzip.decompress(checkpoint.large_data)
                        large_data = pickle.loads(decompressed)
                    except Exception as e:
                        logger.error(f"Failed to decompress checkpoint data: {e}")

                async with self._lock:
                    self._stats["checkpoints_loaded"] += 1

                return {
                    "id": checkpoint.id,
                    "execution_id": checkpoint.execution_id,
                    "checkpoint_name": checkpoint.checkpoint_name,
                    "checkpoint_type": checkpoint.checkpoint_type.value,
                    "checkpoint_data": checkpoint.checkpoint_data,
                    "large_data": large_data,
                    "task_graph_snapshot": checkpoint.task_graph_snapshot,
                    "execution_context": checkpoint.execution_context,
                    "agent_states": checkpoint.agent_states,
                    "sequence_number": checkpoint.sequence_number,
                    "recovery_instructions": checkpoint.recovery_instructions,
                    "dependencies": checkpoint.dependencies,
                    "created_at": checkpoint.created_at.isoformat(),
                    "expires_at": checkpoint.expires_at.isoformat() if checkpoint.expires_at else None,
                    "data_size_bytes": checkpoint.data_size_bytes,
                    "creation_duration_ms": checkpoint.creation_duration_ms,
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to get checkpoint: {e}")
            raise

    async def get_latest_checkpoint(
        self,
        execution_id: str,
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest checkpoint for an execution.

        Args:
            execution_id: Execution ID
            checkpoint_type: Optional checkpoint type filter

        Returns:
            Latest checkpoint or None
        """
        try:
            async with self.session_factory() as session:
                query = select(ExecutionCheckpointModel).where(
                    and_(
                        ExecutionCheckpointModel.execution_id == execution_id,
                        ExecutionCheckpointModel.is_valid == True,
                        or_(
                            ExecutionCheckpointModel.expires_at.is_(None),
                            ExecutionCheckpointModel.expires_at > datetime.now(timezone.utc)
                        )
                    )
                )

                if checkpoint_type:
                    query = query.where(ExecutionCheckpointModel.checkpoint_type == checkpoint_type)

                query = query.order_by(desc(ExecutionCheckpointModel.sequence_number))

                result = await session.execute(query)
                checkpoint = result.scalar_one_or_none()

                if checkpoint:
                    return await self.get_checkpoint(checkpoint.id)

                return None

        except SQLAlchemyError as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            raise

    async def list_checkpoints(
        self,
        execution_id: str,
        include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List all checkpoints for an execution.

        Args:
            execution_id: Execution ID
            include_expired: Whether to include expired checkpoints

        Returns:
            List of checkpoint summaries
        """
        try:
            async with self.session_factory() as session:
                query = select(ExecutionCheckpointModel).where(
                    ExecutionCheckpointModel.execution_id == execution_id
                )

                if not include_expired:
                    query = query.where(
                        and_(
                            ExecutionCheckpointModel.is_valid == True,
                            or_(
                                ExecutionCheckpointModel.expires_at.is_(None),
                                ExecutionCheckpointModel.expires_at > datetime.now(timezone.utc)
                            )
                        )
                    )

                query = query.order_by(ExecutionCheckpointModel.sequence_number)

                result = await session.execute(query)
                checkpoints = result.scalars().all()

                checkpoint_list = []
                for cp in checkpoints:
                    checkpoint_list.append({
                        "id": cp.id,
                        "checkpoint_name": cp.checkpoint_name,
                        "checkpoint_type": cp.checkpoint_type.value,
                        "sequence_number": cp.sequence_number,
                        "created_at": cp.created_at.isoformat(),
                        "expires_at": cp.expires_at.isoformat() if cp.expires_at else None,
                        "is_valid": cp.is_valid,
                        "data_size_bytes": cp.data_size_bytes,
                        "creation_duration_ms": cp.creation_duration_ms,
                    })

                return checkpoint_list

        except SQLAlchemyError as e:
            logger.error(f"Failed to list checkpoints: {e}")
            raise

    async def start_recovery(
        self,
        task_id: str,
        execution_id: str,
        checkpoint_id: Optional[str] = None,
        recovery_strategy: str = "default",
        error_context: Optional[Dict[str, Any]] = None
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
        try:
            async with self.session_factory() as session:
                # Get attempt number
                attempt_query = select(func.max(RecoveryStateModel.attempt_number)).where(
                    RecoveryStateModel.task_id == task_id
                )
                attempt_result = await session.execute(attempt_query)
                max_attempt = attempt_result.scalar() or 0
                attempt_number = max_attempt + 1

                # Create recovery state
                recovery = RecoveryStateModel(
                    task_id=task_id,
                    execution_id=execution_id,
                    checkpoint_id=checkpoint_id,
                    status=RecoveryStatus.ACTIVE,
                    state_data={"recovery_started": True},
                    error_context=error_context,
                    recovery_strategy=recovery_strategy,
                    attempt_number=attempt_number
                )

                session.add(recovery)
                await session.commit()

                async with self._lock:
                    self._stats["recovery_operations"] += 1

                logger.info(f"Started recovery operation for task {task_id}, attempt {attempt_number}")
                return recovery.id

        except SQLAlchemyError as e:
            logger.error(f"Failed to start recovery: {e}")
            raise

    async def update_recovery_status(
        self,
        recovery_id: str,
        status: RecoveryStatus,
        state_data: Optional[Dict[str, Any]] = None,
        recovery_result: Optional[Dict[str, Any]] = None,
        failure_reason: Optional[str] = None
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
        try:
            async with self.session_factory() as session:
                update_values = {
                    "status": status,
                }

                if state_data is not None:
                    update_values["state_data"] = state_data

                if recovery_result is not None:
                    update_values["recovery_result"] = recovery_result

                if failure_reason is not None:
                    update_values["failure_reason"] = failure_reason

                if status in [RecoveryStatus.RECOVERED, RecoveryStatus.ABANDONED]:
                    completed_at = datetime.now(timezone.utc)
                    update_values["recovery_completed_at"] = completed_at
                    update_values["success"] = (status == RecoveryStatus.RECOVERED)

                    # Calculate duration
                    recovery_query = select(RecoveryStateModel.recovery_started_at).where(
                        RecoveryStateModel.id == recovery_id
                    )
                    recovery_result_query = await session.execute(recovery_query)
                    started_at = recovery_result_query.scalar()

                    if started_at:
                        duration = (completed_at - started_at).total_seconds() * 1000
                        update_values["recovery_duration_ms"] = int(duration)

                await session.execute(
                    update(RecoveryStateModel)
                    .where(RecoveryStateModel.id == recovery_id)
                    .values(**update_values)
                )

                await session.commit()

                logger.debug(f"Updated recovery {recovery_id} status to {status}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to update recovery status: {e}")
            raise

    async def invalidate_checkpoint(self, checkpoint_id: str, reason: str = "Invalidated") -> None:
        """
        Invalidate a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to invalidate
            reason: Reason for invalidation
        """
        try:
            async with self.session_factory() as session:
                await session.execute(
                    update(ExecutionCheckpointModel)
                    .where(ExecutionCheckpointModel.id == checkpoint_id)
                    .values(is_valid=False)
                )

                await session.commit()

                logger.info(f"Invalidated checkpoint {checkpoint_id}: {reason}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to invalidate checkpoint: {e}")
            raise

    async def cleanup_expired_checkpoints(self) -> int:
        """
        Clean up expired checkpoints.

        Returns:
            Number of checkpoints cleaned up
        """
        try:
            async with self.session_factory() as session:
                # Mark expired checkpoints as invalid
                result = await session.execute(
                    update(ExecutionCheckpointModel)
                    .where(
                        and_(
                            ExecutionCheckpointModel.expires_at < datetime.now(timezone.utc),
                            ExecutionCheckpointModel.is_valid == True
                        )
                    )
                    .values(is_valid=False)
                )

                await session.commit()
                cleaned_count = result.rowcount

                logger.info(f"Cleaned up {cleaned_count} expired checkpoints")
                return cleaned_count

        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup expired checkpoints: {e}")
            raise

    async def cleanup_old_checkpoints(self, days: int = 30) -> int:
        """
        Clean up old checkpoints.

        Args:
            days: Number of days to keep

        Returns:
            Number of checkpoints deleted
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            async with self.session_factory() as session:
                # Delete old checkpoints
                result = await session.execute(
                    delete(ExecutionCheckpointModel)
                    .where(ExecutionCheckpointModel.created_at < cutoff_date)
                )

                await session.commit()
                deleted_count = result.rowcount

                logger.info(f"Deleted {deleted_count} checkpoints older than {days} days")
                return deleted_count

        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            raise

    async def get_checkpoint_analytics(self) -> Dict[str, Any]:
        """Get checkpoint usage analytics."""
        try:
            async with self.session_factory() as session:
                # Total checkpoints
                total_query = select(func.count(ExecutionCheckpointModel.id))
                total_result = await session.execute(total_query)
                total_checkpoints = total_result.scalar()

                # Valid checkpoints
                valid_query = select(func.count(ExecutionCheckpointModel.id)).where(
                    ExecutionCheckpointModel.is_valid == True
                )
                valid_result = await session.execute(valid_query)
                valid_checkpoints = valid_result.scalar()

                # Checkpoint type distribution
                type_query = (
                    select(ExecutionCheckpointModel.checkpoint_type, func.count(ExecutionCheckpointModel.id))
                    .group_by(ExecutionCheckpointModel.checkpoint_type)
                )
                type_result = await session.execute(type_query)
                type_distribution = {cp_type.value: count for cp_type, count in type_result.fetchall()}

                # Storage metrics
                storage_query = select(
                    func.sum(ExecutionCheckpointModel.data_size_bytes),
                    func.avg(ExecutionCheckpointModel.data_size_bytes),
                    func.avg(ExecutionCheckpointModel.creation_duration_ms)
                )
                storage_result = await session.execute(storage_query)
                total_storage, avg_size, avg_duration = storage_result.fetchone()

                return {
                    "total_checkpoints": total_checkpoints,
                    "valid_checkpoints": valid_checkpoints,
                    "type_distribution": type_distribution,
                    "storage_metrics": {
                        "total_storage_bytes": int(total_storage) if total_storage else 0,
                        "avg_checkpoint_size_bytes": float(avg_size) if avg_size else 0,
                        "avg_creation_duration_ms": float(avg_duration) if avg_duration else 0,
                    },
                    "service_stats": dict(self._stats)
                }

        except SQLAlchemyError as e:
            logger.error(f"Failed to get checkpoint analytics: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return dict(self._stats)