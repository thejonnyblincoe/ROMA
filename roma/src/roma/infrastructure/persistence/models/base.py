"""
SQLAlchemy Base Model Configuration
"""

from datetime import datetime, timezone
from sqlalchemy import DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from typing import Any


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    # Common columns for all tables
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        nullable=False,
        index=True
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
        server_onupdate=func.now(),
        nullable=False,
        index=True
    )

    def __repr__(self) -> str:
        """String representation of the model."""
        class_name = self.__class__.__name__
        attrs = []

        # Get primary key columns
        for column in self.__table__.primary_key.columns:
            value = getattr(self, column.name, None)
            attrs.append(f"{column.name}={value!r}")

        return f"{class_name}({', '.join(attrs)})"