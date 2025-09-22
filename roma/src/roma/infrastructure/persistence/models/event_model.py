"""
Event Storage Model for PostgreSQL Persistence
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
from uuid import uuid4

from sqlalchemy import String, DateTime, Integer, Index, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class EventModel(Base):
    """
    SQLAlchemy model for storing task events.

    This table stores all events from the ROMA event system for persistence,
    querying, and observability.
    """

    __tablename__ = "events"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
        nullable=False
    )

    # Core event fields
    task_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
        comment="Task ID this event belongs to"
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment="Type of event (task_created, task_completed, etc.)"
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        default=lambda: datetime.now(timezone.utc),
        comment="When the event occurred"
    )

    # Event metadata and details
    event_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Event-specific metadata and data"
    )

    # Event versioning and ordering
    version: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        comment="Event version for schema evolution"
    )

    sequence_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Sequence number within task for ordering"
    )

    # Source and correlation
    source: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Source of the event (agent, system, user)"
    )

    correlation_id: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        index=True,
        comment="Correlation ID for tracking related events"
    )

    # Additional event data
    event_data: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Large event data that doesn't fit in metadata"
    )

    # Performance and retention
    is_archived: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        index=True,
        comment="Whether event has been archived"
    )

    archived_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the event was archived"
    )

    # Define composite indexes for common query patterns
    __table_args__ = (
        # Most common: get events by task_id ordered by timestamp
        Index("idx_events_task_timestamp", "task_id", "timestamp"),

        # Event type queries with time ranges
        Index("idx_events_type_timestamp", "event_type", "timestamp"),

        # Correlation tracking
        Index("idx_events_correlation", "correlation_id", "timestamp"),

        # Task sequence ordering
        Index("idx_events_task_sequence", "task_id", "sequence_number"),

        # Archival queries
        Index("idx_events_archived", "is_archived", "created_at"),

        # Source analysis
        Index("idx_events_source_timestamp", "source", "timestamp"),

        # Metadata queries (GIN index for JSONB)
        Index("idx_events_metadata_gin", "event_metadata", postgresql_using="gin"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.event_metadata,
            "version": self.version,
            "sequence_number": self.sequence_number,
            "source": self.source,
            "correlation_id": self.correlation_id,
            "event_data": self.event_data,
            "is_archived": self.is_archived,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_task_event(cls, event: Any, correlation_id: Optional[str] = None) -> "EventModel":
        """
        Create EventModel from a domain event.

        Args:
            event: Domain event object
            correlation_id: Optional correlation ID

        Returns:
            EventModel instance
        """
        return cls(
            task_id=event.task_id,
            event_type=event.event_type,
            timestamp=event.timestamp,
            event_metadata=event.metadata,
            source=getattr(event, "source", None),
            correlation_id=correlation_id,
            event_data=getattr(event, "event_data", None),
        )