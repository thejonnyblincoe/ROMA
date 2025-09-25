"""
Execution Result Value Object.

Defines the final result of complete task graph execution.
Contains execution statistics, success status, and final results.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .result_envelope import AnyResultEnvelope


class ExecutionResult(BaseModel):
    """Result of complete execution orchestration."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, validate_assignment=True)

    # Execution status
    success: bool = Field(..., description="Whether execution completed successfully")

    # Node statistics
    total_nodes: int = Field(..., ge=0, description="Total number of nodes in the graph")
    completed_nodes: int = Field(..., ge=0, description="Number of successfully completed nodes")
    failed_nodes: int = Field(..., ge=0, description="Number of failed nodes")

    # Timing information
    execution_time_seconds: float = Field(
        ..., ge=0.0, description="Total execution time in seconds"
    )
    iterations: int = Field(..., ge=0, description="Number of orchestration iterations")

    # Results
    final_result: AnyResultEnvelope | None = Field(
        default=None, description="Final result envelope from root task (if available)"
    )

    error_details: list[dict[str, Any]] = Field(
        default_factory=list, description="List of error details encountered during execution"
    )

    def __str__(self) -> str:
        """String representation."""
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"ExecutionResult({status}: {self.completed_nodes}/{self.total_nodes} completed, "
            f"{self.failed_nodes} failed, {self.execution_time_seconds:.2f}s, {self.iterations} iterations)"
        )

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"ExecutionResult("
            f"success={self.success}, "
            f"total_nodes={self.total_nodes}, "
            f"completed_nodes={self.completed_nodes}, "
            f"failed_nodes={self.failed_nodes}, "
            f"execution_time_seconds={self.execution_time_seconds}, "
            f"iterations={self.iterations}, "
            f"final_result={self.final_result}, "
            f"error_details={len(self.error_details)} errors)"
        )

    @property
    def completion_rate(self) -> float:
        """Calculate completion rate as percentage."""
        if self.total_nodes == 0:
            return 0.0
        return (self.completed_nodes / self.total_nodes) * 100.0

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_nodes == 0:
            return 0.0
        return (self.failed_nodes / self.total_nodes) * 100.0

    @property
    def has_final_result(self) -> bool:
        """Check if final result is available."""
        return self.final_result is not None

    @property
    def has_errors(self) -> bool:
        """Check if execution had any errors."""
        return len(self.error_details) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "execution_time_seconds": self.execution_time_seconds,
            "iterations": self.iterations,
            "completion_rate": self.completion_rate,
            "failure_rate": self.failure_rate,
            "has_final_result": self.has_final_result,
            "has_errors": self.has_errors,
            "final_result": self.final_result,
            "error_details": self.error_details,
        }

    @classmethod
    def success_result(
        cls,
        total_nodes: int,
        completed_nodes: int,
        execution_time_seconds: float,
        iterations: int,
        final_result: AnyResultEnvelope | None = None,
    ) -> "ExecutionResult":
        """Create a successful execution result."""
        return cls(
            success=True,
            total_nodes=total_nodes,
            completed_nodes=completed_nodes,
            failed_nodes=0,
            execution_time_seconds=execution_time_seconds,
            iterations=iterations,
            final_result=final_result,
        )

    @classmethod
    def failure_result(
        cls,
        total_nodes: int,
        completed_nodes: int,
        failed_nodes: int,
        execution_time_seconds: float,
        iterations: int,
        error_details: list[dict[str, Any]],
        final_result: AnyResultEnvelope | None = None,
    ) -> "ExecutionResult":
        """Create a failed execution result."""
        return cls(
            success=False,
            total_nodes=total_nodes,
            completed_nodes=completed_nodes,
            failed_nodes=failed_nodes,
            execution_time_seconds=execution_time_seconds,
            iterations=iterations,
            final_result=final_result,
            error_details=error_details,
        )
