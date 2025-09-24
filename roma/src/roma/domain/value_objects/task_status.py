"""
TaskStatus enumeration for ROMA v2.0

Manages the lifecycle states of task nodes in the execution graph.
"""

from enum import Enum
from typing import Literal, Set


class TaskStatus(str, Enum):
    """
    Status of a task node in the execution graph.

    State transition flow:
    PENDING → READY → EXECUTING → (COMPLETED | FAILED | WAITING_FOR_CHILDREN)
    WAITING_FOR_CHILDREN → AGGREGATING → COMPLETED

    Special states:
    - NEEDS_REPLAN: Triggers replanning when children fail
    - WAITING_FOR_CHILDREN: Parent waiting for child tasks to complete
    - AGGREGATING: Parent collecting results from completed children
    """

    PENDING = "PENDING"                       # Task created, waiting for dependencies
    READY = "READY"                           # Dependencies satisfied, ready to execute
    EXECUTING = "EXECUTING"                   # Currently being processed
    WAITING_FOR_CHILDREN = "WAITING_FOR_CHILDREN"  # Parent waiting for children to complete
    AGGREGATING = "AGGREGATING"               # Parent collecting child results
    COMPLETED = "COMPLETED"                   # Successfully finished
    FAILED = "FAILED"                         # Execution failed
    NEEDS_REPLAN = "NEEDS_REPLAN"            # Requires replanning due to failure
    
    def __str__(self) -> str:
        return self.value
    
    @classmethod
    def from_string(cls, value: str) -> "TaskStatus":
        """
        Convert string to TaskStatus.
        
        Args:
            value: String representation of task status
            
        Returns:
            TaskStatus enum value
            
        Raises:
            ValueError: If value is not a valid task status
        """
        try:
            return cls(value.upper())
        except ValueError:
            valid_statuses = [s.value for s in cls]
            raise ValueError(
                f"Invalid task status '{value}'. Valid statuses: {valid_statuses}"
            )
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (execution finished)."""
        return self in {TaskStatus.COMPLETED, TaskStatus.FAILED}
    
    @property
    def is_active(self) -> bool:
        """Check if this task is currently active (executing or aggregating)."""
        return self in {TaskStatus.EXECUTING, TaskStatus.AGGREGATING}
    
    @property
    def can_transition_to(self) -> Set["TaskStatus"]:
        """
        Get valid transition states from current status.
        
        Returns:
            Set of valid target statuses for transitions
        """
        transitions = {
            TaskStatus.PENDING: {TaskStatus.READY, TaskStatus.FAILED},
            # READY can fail during agent loading or transition to executing
            TaskStatus.READY: {TaskStatus.EXECUTING, TaskStatus.FAILED},
            TaskStatus.EXECUTING: {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.WAITING_FOR_CHILDREN,
                TaskStatus.NEEDS_REPLAN
            },
            TaskStatus.WAITING_FOR_CHILDREN: {
                TaskStatus.AGGREGATING,
                TaskStatus.NEEDS_REPLAN,
                TaskStatus.FAILED
            },
            TaskStatus.AGGREGATING: {TaskStatus.COMPLETED, TaskStatus.FAILED},
            TaskStatus.NEEDS_REPLAN: {TaskStatus.READY, TaskStatus.FAILED},
            TaskStatus.COMPLETED: set(),  # Terminal state
            TaskStatus.FAILED: {TaskStatus.NEEDS_REPLAN, TaskStatus.READY},  # Recovery
        }
        
        return transitions.get(self, set())
    
    def can_transition_to_status(self, target: "TaskStatus") -> bool:
        """
        Check if transition to target status is valid.
        
        Args:
            target: Target status to transition to
            
        Returns:
            True if transition is valid, False otherwise
        """
        return target in self.can_transition_to


# Type hints for use in other modules
TaskStatusLiteral = Literal[
    "PENDING", "READY", "EXECUTING", "WAITING_FOR_CHILDREN",
    "AGGREGATING", "COMPLETED", "FAILED", "NEEDS_REPLAN"
]
