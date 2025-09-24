"""
Checkpoint and Recovery Models for PostgreSQL Persistence
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4
from enum import Enum

from sqlalchemy import String, DateTime, Integer, Index, Text, Enum as SQLEnum, LargeBinary
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base
# Use domain enums - clean architecture compliance
from roma.domain.value_objects.persistence.checkpoint_type import CheckpointType
from roma.domain.value_objects.persistence.recovery_status import RecoveryStatus


class ExecutionCheckpointModel(Base):
    """
    SQLAlchemy model for storing execution checkpoints.

    This table stores checkpoints that can be used to recover execution
    state after failures or interruptions.
    """

    __tablename__ = "execution_checkpoints"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        nullable=False
    )

    # Checkpoint identification
    execution_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Execution ID this checkpoint belongs to"
    )

    checkpoint_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable checkpoint name"
    )

    checkpoint_type: Mapped[CheckpointType] = mapped_column(
        SQLEnum(CheckpointType),
        nullable=False,
        default=CheckpointType.AUTOMATIC,
        index=True,
        comment="Type of checkpoint"
    )

    # Checkpoint data
    checkpoint_data: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Serialized execution state data"
    )

    # Large binary data (for complex state)
    large_data: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary,
        nullable=True,
        comment="Large binary checkpoint data"
    )

    # Checkpoint metadata
    task_graph_snapshot: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Task graph state at checkpoint time"
    )

    execution_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Execution context at checkpoint time"
    )

    agent_states: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Agent states at checkpoint time"
    )

    # Checkpoint versioning
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Checkpoint format version"
    )

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Sequence number within execution"
    )

    # Recovery information
    recovery_instructions: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Instructions for recovery from this checkpoint"
    )

    dependencies: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Dependencies required for recovery"
    )

    # Checkpoint validity
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When this checkpoint expires"
    )

    is_valid: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        index=True,
        comment="Whether checkpoint is still valid"
    )

    # Performance metadata
    data_size_bytes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Size of checkpoint data in bytes"
    )

    creation_duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Time taken to create checkpoint in milliseconds"
    )

    # Define indexes for common query patterns
    __table_args__ = (
        # Most common: get checkpoints by execution ordered by sequence
        Index("idx_checkpoint_exec_sequence", "execution_id", "sequence_number"),

        # Checkpoint type queries
        Index("idx_checkpoint_type_created", "checkpoint_type", "created_at"),

        # Validity and expiration
        Index("idx_checkpoint_valid_expires", "is_valid", "expires_at"),

        # Recovery queries
        Index("idx_checkpoint_exec_valid", "execution_id", "is_valid", "created_at"),

        # Performance analysis
        Index("idx_checkpoint_size_duration", "data_size_bytes", "creation_duration_ms"),

        # Cleanup queries
        Index("idx_checkpoint_expires", "expires_at"),

        # JSONB indexes for metadata queries
        Index("idx_checkpoint_data_gin", "checkpoint_data", postgresql_using="gin"),
    )


class RecoveryStateModel(Base):
    """
    SQLAlchemy model for tracking recovery operations.

    This table tracks recovery attempts and their outcomes.
    """

    __tablename__ = "recovery_states"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        nullable=False
    )

    # Recovery identification
    task_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Task ID being recovered"
    )

    execution_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Execution ID being recovered"
    )

    checkpoint_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Checkpoint used for recovery"
    )

    # Recovery status
    status: Mapped[RecoveryStatus] = mapped_column(
        SQLEnum(RecoveryStatus),
        nullable=False,
        default=RecoveryStatus.ACTIVE,
        index=True,
        comment="Recovery status"
    )

    # Recovery data
    state_data: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Recovery state data"
    )

    error_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Error context that triggered recovery"
    )

    recovery_strategy: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Recovery strategy used"
    )

    # Recovery timing
    recovery_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
        comment="When recovery started"
    )

    recovery_completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When recovery completed"
    )

    recovery_duration_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Recovery duration in milliseconds"
    )

    # Recovery outcome
    success: Mapped[Optional[bool]] = mapped_column(
        nullable=True,
        comment="Whether recovery was successful"
    )

    recovery_result: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Recovery operation result"
    )

    failure_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Reason for recovery failure"
    )

    # Recovery versioning
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Recovery state format version"
    )

    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Recovery attempt number"
    )

    # Define indexes for common query patterns
    __table_args__ = (
        # Most common: get recovery states by task/execution
        Index("idx_recovery_task_started", "task_id", "recovery_started_at"),
        Index("idx_recovery_exec_started", "execution_id", "recovery_started_at"),

        # Status queries
        Index("idx_recovery_status_started", "status", "recovery_started_at"),

        # Success/failure analysis
        Index("idx_recovery_success_completed", "success", "recovery_completed_at"),

        # Recovery strategy analysis
        Index("idx_recovery_strategy_status", "recovery_strategy", "status"),

        # Performance analysis
        Index("idx_recovery_duration", "recovery_duration_ms"),

        # Cleanup queries
        Index("idx_recovery_completed", "recovery_completed_at"),

        # Attempt tracking
        Index("idx_recovery_task_attempt", "task_id", "attempt_number"),

        # JSONB indexes for metadata queries
        Index("idx_recovery_state_gin", "state_data", postgresql_using="gin"),
        Index("idx_recovery_error_gin", "error_context", postgresql_using="gin"),
    )