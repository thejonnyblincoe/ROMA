"""
Deadlock Analysis Value Objects for ROMA v2.0.

Contains value objects for deadlock detection results and analysis.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DeadlockType(str, Enum):
    """Types of deadlocks that can be detected."""

    CIRCULAR_DEPENDENCY = "circular_dependency"
    STALLED_EXECUTION = "stalled_execution"
    INFINITE_WAIT = "infinite_wait"


class DeadlockSeverity(str, Enum):
    """Severity levels for deadlock scenarios."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeadlockReport(BaseModel):
    """Value object representing a detected deadlock scenario."""

    deadlock_type: DeadlockType
    affected_nodes: list[str]
    description: str
    severity: DeadlockSeverity
    suggested_actions: list[str]
    detection_time: datetime
    cycle_path: list[str] | None = None  # For circular dependencies
    waiting_duration_seconds: float | None = None  # For stalled execution


class DeadlockSummary(BaseModel):
    """Value object summarizing all deadlock analysis results."""

    total_deadlocks: int
    status: str  # "healthy", "degraded", "deadlocked"
    by_type: dict[str, int] = Field(default_factory=dict)
    by_severity: dict[str, int] = Field(default_factory=dict)
    monitoring_duration_seconds: float
    latest_deadlock_type: str | None = None
    has_critical_deadlocks: bool = False
