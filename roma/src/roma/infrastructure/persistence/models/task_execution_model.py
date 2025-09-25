"""
Task Execution and Relationship Models for PostgreSQL Persistence
"""

from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from roma.domain.value_objects.persistence.task_relationship_type import TaskRelationshipType

# Use domain enums - clean architecture compliance
from roma.domain.value_objects.task_status import TaskStatus

from .base import Base


class TaskExecutionModel(Base):
    """
    SQLAlchemy model for tracking task executions.

    This table stores high-level execution information for tasks,
    providing a view of task lifecycle and current state.
    """

    __tablename__ = "task_executions"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()), nullable=False
    )

    # Task identification
    task_id: Mapped[str] = mapped_column(
        String(255), nullable=False, unique=True, index=True, comment="Unique task identifier"
    )

    # Task details
    goal: Mapped[str] = mapped_column(Text, nullable=False, comment="Task goal/objective")

    task_type: Mapped[str] = mapped_column(
        String(50), nullable=False, index=True, comment="Task type (RETRIEVE, WRITE, THINK, etc.)"
    )

    node_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True, comment="Node type (PLAN, EXECUTE)"
    )

    # Execution state
    status: Mapped[TaskStatus] = mapped_column(
        SQLEnum(TaskStatus),
        nullable=False,
        default=TaskStatus.PENDING,
        index=True,
        comment="Current execution status",
    )

    # Hierarchy and relationships
    parent_task_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True, comment="Parent task ID if this is a subtask"
    )

    root_task_id: Mapped[str | None] = mapped_column(
        String(255), nullable=True, index=True, comment="Root task ID for the entire execution tree"
    )

    depth_level: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Depth in the task hierarchy (0 = root)"
    )

    # Execution timing
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True, comment="When task execution started"
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True, comment="When task execution completed"
    )

    execution_duration_ms: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Execution duration in milliseconds"
    )

    # Results and metadata
    result: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Task execution result"
    )

    task_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Task metadata and configuration"
    )

    error_info: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Error information if task failed"
    )

    # Agent and execution context
    agent_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Agent configuration used for execution"
    )

    execution_context: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Execution context and environment"
    )

    # Performance metrics
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Token usage statistics"
    )

    cost_info: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Cost information"
    )

    # Retry and recovery
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="Number of retry attempts"
    )

    max_retries: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3, comment="Maximum allowed retries"
    )

    # Archival and cleanup
    is_archived: Mapped[bool] = mapped_column(
        nullable=False, default=False, index=True, comment="Whether execution has been archived"
    )

    # Define indexes for common query patterns
    __table_args__ = (
        # Hierarchy queries
        Index("idx_task_exec_parent", "parent_task_id", "created_at"),
        Index("idx_task_exec_root", "root_task_id", "depth_level"),
        # Status and timing queries
        Index("idx_task_exec_status_created", "status", "created_at"),
        Index("idx_task_exec_started", "started_at"),
        Index("idx_task_exec_completed", "completed_at"),
        # Task type analysis
        Index("idx_task_exec_type_status", "task_type", "status"),
        # Performance queries
        Index("idx_task_exec_duration", "execution_duration_ms"),
        # Retry analysis
        Index("idx_task_exec_retries", "retry_count", "status"),
        # Archival
        Index("idx_task_exec_archived", "is_archived", "completed_at"),
        # JSONB indexes for metadata queries
        Index("idx_task_exec_metadata_gin", "task_metadata", postgresql_using="gin"),
        Index("idx_task_exec_result_gin", "result", postgresql_using="gin"),
    )


class TaskRelationshipModel(Base):
    """
    SQLAlchemy model for storing task relationships.

    This table captures the relationships between tasks in the execution graph.
    """

    __tablename__ = "task_relationships"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()), nullable=False
    )

    # Relationship definition
    parent_task_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("task_executions.task_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent task ID",
    )

    child_task_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("task_executions.task_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Child task ID",
    )

    relationship_type: Mapped[TaskRelationshipType] = mapped_column(
        SQLEnum(TaskRelationshipType),
        nullable=False,
        default=TaskRelationshipType.PARENT_CHILD,
        comment="Type of relationship",
    )

    # Relationship metadata
    order_index: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="Order index for ordered relationships"
    )

    relationship_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True, comment="Relationship-specific metadata"
    )

    # Define indexes and constraints
    __table_args__ = (
        # Unique constraint for parent-child pairs
        Index(
            "idx_task_rel_unique",
            "parent_task_id",
            "child_task_id",
            "relationship_type",
            unique=True,
        ),
        # Query optimization indexes
        Index("idx_task_rel_parent", "parent_task_id", "relationship_type"),
        Index("idx_task_rel_child", "child_task_id", "relationship_type"),
        Index("idx_task_rel_type", "relationship_type", "created_at"),
        # Order queries
        Index("idx_task_rel_order", "parent_task_id", "order_index"),
    )
