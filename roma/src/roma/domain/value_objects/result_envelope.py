"""
ResultEnvelope for standardized result and artifact handling in ROMA v2.

Provides a unified wrapper for all agent results with consistent artifact management,
metadata tracking, and performance metrics.
"""

from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from roma.domain.entities.artifacts.base_artifact import BaseArtifact
from roma.domain.value_objects.agent_responses import (
    AggregatorResult,
    AtomizerResult,
    ExecutorResult,
    PlanModifierResult,
    PlannerResult,
)
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.media_type import MediaType

# Type variable for generic result content
T = TypeVar("T")


# Use existing artifact system from domain entities
# Artifact = BaseArtifact (imported above)


class ExecutionMetrics(BaseModel):
    """
    Performance and resource metrics for task execution.
    """

    model_config = ConfigDict(frozen=True)

    execution_time: float = Field(..., ge=0.0, description="Total execution time in seconds")
    tokens_used: int = Field(default=0, ge=0, description="Total tokens consumed")
    cost_estimate: float = Field(default=0.0, ge=0.0, description="Estimated execution cost in USD")
    model_calls: int = Field(default=0, ge=0, description="Number of model API calls made")
    cache_hits: int = Field(default=0, ge=0, description="Number of cache hits")
    memory_peak_mb: float | None = Field(
        default=None, ge=0.0, description="Peak memory usage in MB"
    )
    network_requests: int = Field(default=0, ge=0, description="External network requests made")


class ResultEnvelope[T](BaseModel):
    """
    Standardized envelope for all agent results and artifacts.

    Provides consistent structure for results, artifacts, metadata, and metrics
    across all agent types and execution paths.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Core identification
    envelope_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique envelope identifier"
    )
    task_id: str = Field(..., description="Associated task identifier")
    execution_id: str = Field(..., description="Associated execution identifier")
    agent_type: AgentType = Field(..., description="Type of agent that produced this result")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Result creation time"
    )

    # Result content
    result: T | None = Field(
        ..., description="The actual agent result (AtomizerResult, ExecutorResult, etc.)"
    )
    success: bool = Field(default=True, description="Whether execution was successful")
    error_message: str | None = Field(default=None, description="Error description if failed")

    # Artifacts and outputs
    artifacts: list[BaseArtifact] = Field(
        default_factory=list, description="Files and data created during execution"
    )
    output_text: str | None = Field(default=None, description="Primary text output for display")
    structured_data: dict[str, Any] | None = Field(
        default=None, description="Structured data output"
    )

    # Execution context and tracing
    execution_metrics: ExecutionMetrics = Field(..., description="Performance and resource metrics")
    trace_id: str | None = Field(default=None, description="Distributed tracing identifier")
    parent_envelope_id: str | None = Field(
        default=None, description="Parent envelope for subtask tracking"
    )
    context_hash: str | None = Field(default=None, description="Hash of input context for caching")

    # Quality and confidence
    confidence_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Overall confidence in result"
    )
    quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Result quality assessment"
    )

    # Additional metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional envelope metadata"
    )

    @classmethod
    def create_success(
        cls,
        result: T,
        task_id: str,
        execution_id: str,
        agent_type: AgentType,
        execution_metrics: ExecutionMetrics,
        artifacts: list[BaseArtifact] | None = None,
        output_text: str | None = None,
        **kwargs: Any,
    ) -> "ResultEnvelope[T]":
        """
        Create a successful result envelope.

        Args:
            result: The agent result object
            task_id: Associated task ID
            execution_id: Associated execution ID
            agent_type: Type of agent
            execution_metrics: Performance metrics
            artifacts: Optional list of artifacts
            output_text: Optional primary text output
            **kwargs: Additional fields

        Returns:
            ResultEnvelope instance
        """
        return cls(
            result=result,
            task_id=task_id,
            execution_id=execution_id,
            agent_type=agent_type,
            success=True,
            artifacts=artifacts or [],
            output_text=output_text,
            execution_metrics=execution_metrics,
            **kwargs,
        )

    @classmethod
    def create_error(
        cls,
        error_message: str,
        task_id: str,
        execution_id: str,
        agent_type: AgentType,
        execution_metrics: ExecutionMetrics,
        result: T | None = None,
        **kwargs: Any,
    ) -> "ResultEnvelope[T]":
        """
        Create an error result envelope.

        Args:
            error_message: Description of the error
            task_id: Associated task ID
            execution_id: Associated execution ID
            agent_type: Type of agent
            execution_metrics: Performance metrics
            result: Optional partial result
            **kwargs: Additional fields

        Returns:
            ResultEnvelope instance
        """
        return cls(
            result=result,
            task_id=task_id,
            execution_id=execution_id,
            agent_type=agent_type,
            success=False,
            error_message=error_message,
            execution_metrics=execution_metrics,
            artifacts=[],  # No artifacts on error
            **kwargs,
        )

    def add_artifact(self, artifact: BaseArtifact) -> "ResultEnvelope[T]":
        """
        Add an artifact to the envelope (creates new immutable instance).

        Args:
            artifact: Artifact to add

        Returns:
            New ResultEnvelope with added artifact
        """
        new_artifacts = list(self.artifacts) + [artifact]
        return self.model_copy(update={"artifacts": new_artifacts})

    def get_artifacts_by_type(self, media_type: MediaType) -> list[BaseArtifact]:
        """
        Get all artifacts of a specific media type.

        Args:
            media_type: MediaType filter

        Returns:
            List of matching artifacts
        """
        return [artifact for artifact in self.artifacts if artifact.media_type == media_type]

    def get_total_size_bytes(self) -> int:
        """
        Get total size of all artifacts.

        Returns:
            Total size in bytes
        """
        return sum(artifact.get_size_bytes() or 0 for artifact in self.artifacts)

    def has_errors(self) -> bool:
        """
        Check if this envelope represents a failed execution.

        Returns:
            True if execution failed
        """
        return not self.success or self.error_message is not None

    def extract_primary_output(self) -> str:
        """
        Extract the primary output text for display.

        Returns:
            Primary output string
        """
        if self.output_text:
            return self.output_text

        # Handle null result case
        if self.result is None:
            return self.error_message or "No result available"

        # Try to extract from result based on agent type
        if hasattr(self.result, "result"):
            return str(self.result.result)
        elif hasattr(self.result, "synthesized_result"):
            return str(self.result.synthesized_result)
        elif hasattr(self.result, "reasoning"):
            return str(self.result.reasoning)
        else:
            return str(self.result)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return self.model_dump(mode="json", exclude_none=True)


# Type aliases for specific agent result envelopes

AtomizerEnvelope = ResultEnvelope[AtomizerResult]
PlannerEnvelope = ResultEnvelope[PlannerResult]
ExecutorEnvelope = ResultEnvelope[ExecutorResult]
AggregatorEnvelope = ResultEnvelope[AggregatorResult]
PlanModifierEnvelope = ResultEnvelope[PlanModifierResult]

# Generic envelope for any result
AnyResultEnvelope = ResultEnvelope[Any]
