"""
SQLAlchemy Implementation of Checkpoint Repository.

Infrastructure layer implementation of checkpoint persistence using SQLAlchemy.
"""

import asyncio
import logging
import pickle
import gzip
from datetime import datetime, timezone, timedelta
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import select, delete, update, func, and_, or_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from roma.domain.interfaces.persistence import CheckpointRepository, RecoveryRepository
from roma.domain.value_objects.persistence import (
    CheckpointType,
    RecoveryStatus,
    CheckpointRecord,
    CheckpointSummary,
    CheckpointAnalytics,
    CheckpointStorageMetrics,
)
from roma.infrastructure.persistence.models.checkpoint_model import (
    ExecutionCheckpointModel,
    RecoveryStateModel,
)

logger = logging.getLogger(__name__)


class SQLAlchemyCheckpointRepository(CheckpointRepository):
    """
    SQLAlchemy implementation of checkpoint repository.

    Handles all checkpoint persistence operations using PostgreSQL.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """
        Initialize repository with session factory.

        Args:
            session_factory: SQLAlchemy async session factory
        """
        self.session_factory = session_factory

    async def create_checkpoint(self, checkpoint_record: CheckpointRecord) -> str:
        """Create a new checkpoint in the database."""
        start_time = datetime.now(timezone.utc)

        try:
            async with self.session_factory() as session:
                try:
                    # Process large data if provided
                    compressed_data = None
                    if checkpoint_record.large_data is not None:
                        serialized = pickle.dumps(checkpoint_record.large_data)
                        compressed_data = gzip.compress(serialized)

                    # Calculate expiration
                    expires_at = None
                    if checkpoint_record.expires_at:
                        expires_at = datetime.fromisoformat(checkpoint_record.expires_at.replace('Z', '+00:00'))

                    # Use PostgreSQL advisory lock to prevent race conditions in sequence number generation
                    # This ensures atomic sequence generation even when no rows exist yet
                    lock_id = abs(hash(checkpoint_record.execution_id)) % 2147483647
                    await session.execute(
                        text("SELECT pg_advisory_xact_lock(:lock_id)"),
                        {"lock_id": lock_id}
                    )

                    # Now safely get the max sequence number under advisory lock
                    seq_query = select(func.max(ExecutionCheckpointModel.sequence_number)).where(
                        ExecutionCheckpointModel.execution_id == checkpoint_record.execution_id
                    )
                    seq_result = await session.execute(seq_query)
                    max_seq = seq_result.scalar() or 0
                    sequence_number = max_seq + 1

                    # Calculate creation duration BEFORE inserting to avoid second commit
                    creation_duration_ms = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)

                    # Create database model with all data including duration
                    checkpoint = ExecutionCheckpointModel(
                        execution_id=checkpoint_record.execution_id,
                        checkpoint_name=checkpoint_record.checkpoint_name,
                        checkpoint_type=checkpoint_record.checkpoint_type,
                        checkpoint_data=checkpoint_record.checkpoint_data,
                        large_data=compressed_data,
                        task_graph_snapshot=checkpoint_record.task_graph_snapshot,
                        execution_context=checkpoint_record.execution_context,
                        agent_states=checkpoint_record.agent_states,
                        sequence_number=sequence_number,
                        expires_at=expires_at,
                        data_size_bytes=checkpoint_record.data_size_bytes,
                        creation_duration_ms=creation_duration_ms
                    )

                    session.add(checkpoint)
                    await session.commit()

                    logger.info(f"Created checkpoint {checkpoint_record.checkpoint_name}")
                    return checkpoint.id

                except Exception as e:
                    await session.rollback()
                    raise

        except SQLAlchemyError as e:
            logger.error(f"Failed to create checkpoint: {e}")
            raise

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[CheckpointRecord]:
        """Get checkpoint by ID."""
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

                return CheckpointRecord(
                    id=checkpoint.id,
                    execution_id=checkpoint.execution_id,
                    checkpoint_name=checkpoint.checkpoint_name,
                    checkpoint_type=checkpoint.checkpoint_type,
                    checkpoint_data=checkpoint.checkpoint_data,
                    large_data=large_data,
                    task_graph_snapshot=checkpoint.task_graph_snapshot,
                    execution_context=checkpoint.execution_context,
                    agent_states=checkpoint.agent_states,
                    sequence_number=checkpoint.sequence_number,
                    recovery_instructions=checkpoint.recovery_instructions,
                    dependencies=checkpoint.dependencies,
                    created_at=checkpoint.created_at.isoformat(),
                    expires_at=checkpoint.expires_at.isoformat() if checkpoint.expires_at else None,
                    data_size_bytes=checkpoint.data_size_bytes,
                    creation_duration_ms=checkpoint.creation_duration_ms,
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to get checkpoint: {e}")
            raise

    async def get_latest_checkpoint(
        self,
        execution_id: str,
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Optional[CheckpointRecord]:
        """Get the latest checkpoint for an execution."""
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
    ) -> List[CheckpointSummary]:
        """List all checkpoints for an execution."""
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

                summaries = []
                for cp in checkpoints:
                    summaries.append(CheckpointSummary(
                        id=cp.id,
                        checkpoint_name=cp.checkpoint_name,
                        checkpoint_type=cp.checkpoint_type,
                        sequence_number=cp.sequence_number,
                        created_at=cp.created_at.isoformat(),
                        expires_at=cp.expires_at.isoformat() if cp.expires_at else None,
                        is_valid=cp.is_valid,
                        data_size_bytes=cp.data_size_bytes,
                        creation_duration_ms=cp.creation_duration_ms,
                    ))

                return summaries

        except SQLAlchemyError as e:
            logger.error(f"Failed to list checkpoints: {e}")
            raise

    async def get_next_sequence_number(self, execution_id: str) -> int:
        """Get next sequence number for an execution."""
        try:
            async with self.session_factory() as session:
                seq_query = select(func.max(ExecutionCheckpointModel.sequence_number)).where(
                    ExecutionCheckpointModel.execution_id == execution_id
                )
                seq_result = await session.execute(seq_query)
                max_seq = seq_result.scalar() or 0
                return max_seq + 1

        except SQLAlchemyError as e:
            logger.error(f"Failed to get next sequence number: {e}")
            raise

    async def invalidate_checkpoint(self, checkpoint_id: str) -> None:
        """Invalidate a checkpoint."""
        try:
            async with self.session_factory() as session:
                await session.execute(
                    update(ExecutionCheckpointModel)
                    .where(ExecutionCheckpointModel.id == checkpoint_id)
                    .values(is_valid=False)
                )

                await session.commit()
                logger.info(f"Invalidated checkpoint {checkpoint_id}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to invalidate checkpoint: {e}")
            raise

    async def cleanup_expired_checkpoints(self) -> int:
        """Clean up expired checkpoints."""
        try:
            async with self.session_factory() as session:
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
        """Clean up old checkpoints."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            async with self.session_factory() as session:
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

    async def get_checkpoint_analytics(self) -> CheckpointAnalytics:
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

                storage_metrics = CheckpointStorageMetrics(
                    total_storage_bytes=int(total_storage) if total_storage else 0,
                    avg_checkpoint_size_bytes=float(avg_size) if avg_size else 0.0,
                    avg_creation_duration_ms=float(avg_duration) if avg_duration else 0.0,
                )

                return CheckpointAnalytics(
                    total_checkpoints=total_checkpoints,
                    valid_checkpoints=valid_checkpoints,
                    type_distribution=type_distribution,
                    storage_metrics=storage_metrics,
                    service_stats={}  # No instance-level stats
                )

        except SQLAlchemyError as e:
            logger.error(f"Failed to get checkpoint analytics: {e}")
            raise


class SQLAlchemyRecoveryRepository(RecoveryRepository):
    """
    SQLAlchemy implementation of recovery repository.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        """
        Initialize repository with session factory.

        Args:
            session_factory: SQLAlchemy async session factory
        """
        self.session_factory = session_factory

    async def create_recovery(
        self,
        task_id: str,
        execution_id: str,
        checkpoint_id: Optional[str],
        recovery_strategy: str,
        error_context: Optional[dict],
        attempt_number: int
    ) -> str:
        """Create a recovery state."""
        try:
            async with self.session_factory() as session:
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

                logger.info(f"Started recovery operation for task {task_id}, attempt {attempt_number}")
                return recovery.id

        except SQLAlchemyError as e:
            logger.error(f"Failed to start recovery: {e}")
            raise

    async def update_recovery_status(
        self,
        recovery_id: str,
        status: RecoveryStatus,
        state_data: Optional[dict] = None,
        recovery_result: Optional[dict] = None,
        failure_reason: Optional[str] = None
    ) -> None:
        """Update recovery operation status."""
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

    async def get_next_attempt_number(self, task_id: str) -> int:
        """Get next attempt number for a task."""
        try:
            async with self.session_factory() as session:
                attempt_query = select(func.max(RecoveryStateModel.attempt_number)).where(
                    RecoveryStateModel.task_id == task_id
                )
                attempt_result = await session.execute(attempt_query)
                max_attempt = attempt_result.scalar() or 0
                return max_attempt + 1

        except SQLAlchemyError as e:
            logger.error(f"Failed to get next attempt number: {e}")
            raise