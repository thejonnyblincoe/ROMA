"""
Dependency Status enumeration for ROMA v2.0.

Defines the possible states of task dependencies for enhanced dependency resolution.
"""

from enum import Enum


class DependencyStatus(str, Enum):
    """
    Status enumeration for task dependencies.

    Used to track the state of dependencies during validation and execution.
    """

    # Dependency is satisfied and execution can proceed
    COMPLETED = "completed"

    # Dependency failed and blocks execution
    FAILED = "failed"

    # Dependency is not yet started
    PENDING = "pending"

    # Dependency is currently being executed
    EXECUTING = "executing"

    # Dependency is ready but not yet executing
    READY = "ready"

    # Dependency node does not exist in graph
    MISSING = "missing"

    # Dependency status cannot be determined
    UNKNOWN = "unknown"

    @classmethod
    def from_task_status(cls, task_status) -> "DependencyStatus":
        """
        Convert TaskStatus to DependencyStatus.

        Args:
            task_status: TaskStatus enum value

        Returns:
            Corresponding DependencyStatus
        """
        # Import here to avoid circular imports
        from roma.domain.value_objects.task_status import TaskStatus

        mapping = {
            TaskStatus.COMPLETED: cls.COMPLETED,
            TaskStatus.FAILED: cls.FAILED,
            TaskStatus.PENDING: cls.PENDING,
            TaskStatus.EXECUTING: cls.EXECUTING,
            TaskStatus.READY: cls.READY,
        }

        return mapping.get(task_status, cls.UNKNOWN)

    @property
    def is_satisfied(self) -> bool:
        """Check if dependency is satisfied (allows execution)."""
        return self == DependencyStatus.COMPLETED

    @property
    def is_blocking(self) -> bool:
        """Check if dependency blocks execution."""
        return self in [DependencyStatus.FAILED, DependencyStatus.MISSING]

    @property
    def is_pending(self) -> bool:
        """Check if dependency is still in progress."""
        return self in [DependencyStatus.PENDING, DependencyStatus.READY, DependencyStatus.EXECUTING]

    @property
    def requires_wait(self) -> bool:
        """Check if execution should wait for this dependency."""
        return self.is_pending

    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.value