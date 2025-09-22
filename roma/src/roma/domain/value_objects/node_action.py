"""
Node Action Value Object.

Defines the actions that can result from processing a single task node.
Used to communicate the outcome and next steps for node processing.
"""

from enum import Enum


class NodeAction(Enum):
    """Actions that can result from processing a task node."""

    # Planning and subtask creation
    ADD_SUBTASKS = "add_subtasks"  # Planner created subtasks that need to be added to graph

    # Node completion
    COMPLETE = "complete"  # Node execution finished successfully

    # Aggregation
    AGGREGATE = "aggregate"  # Parent node aggregation completed

    # Plan modification
    REPLAN = "replan"  # Plan needs modification (usually from HITL feedback)

    # Error handling and recovery
    RETRY = "retry"  # Node should be retried (transient failure)
    FAIL = "fail"  # Node failed permanently

    # No action needed
    NOOP = "noop"  # No action needed, node state unchanged

    def __str__(self) -> str:
        """String representation."""
        return self.value

    def __repr__(self) -> str:
        """Representation for debugging."""
        return f"NodeAction.{self.name}"

    @classmethod
    def from_string(cls, value: str) -> "NodeAction":
        """Create NodeAction from string value."""
        try:
            return cls(value.lower().strip())
        except ValueError:
            valid_values = [action.value for action in cls]
            raise ValueError(f"Invalid NodeAction '{value}'. Valid values: {valid_values}")

    @property
    def is_terminal(self) -> bool:
        """Check if this action represents a terminal state."""
        return self in {NodeAction.COMPLETE, NodeAction.FAIL}

    @property
    def requires_graph_mutation(self) -> bool:
        """Check if this action requires graph mutations."""
        return self in {NodeAction.ADD_SUBTASKS, NodeAction.REPLAN}

    @property
    def is_error_state(self) -> bool:
        """Check if this action represents an error condition."""
        return self in {NodeAction.FAIL, NodeAction.RETRY}