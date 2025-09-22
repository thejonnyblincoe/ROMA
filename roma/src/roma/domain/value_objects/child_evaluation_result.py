"""
Child Evaluation Result Value Object.

Defines the result of evaluating terminal children for aggregation decisions.
Used by ExecutionOrchestrator to determine whether to aggregate, partial aggregate, or replan.
"""

from enum import Enum


class ChildEvaluationResult(Enum):
    """Result of evaluating terminal children for aggregation decision."""

    AGGREGATE_ALL = "aggregate_all"           # All children completed successfully
    AGGREGATE_PARTIAL = "aggregate_partial"   # Some failed but below threshold, proceed with partial
    REPLAN = "replan"                        # Too many failures, need to replan the parent task

    def __str__(self) -> str:
        """String representation."""
        return self.value

    def __repr__(self) -> str:
        """Debug representation."""
        return f"ChildEvaluationResult.{self.name}"