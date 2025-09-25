"""
Node Result Value Object.

Defines the result of processing a single task node through the agent pipeline.
Used to communicate outcomes, actions, and data between orchestration components.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.result_envelope import AnyResultEnvelope


class NodeResult(BaseModel):
    """Result of processing a single node through the agent pipeline."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, validate_assignment=True)

    # Required fields
    task_id: str = Field(..., description="ID of the task this result is for")
    action: NodeAction = Field(..., description="Action to take based on processing result")

    # Optional result data
    envelope: AnyResultEnvelope | None = Field(
        default=None, description="Result envelope from agent execution"
    )

    new_nodes: list[TaskNode] = Field(
        default_factory=list, description="New nodes to add to graph (for ADD_SUBTASKS action)"
    )

    error: str | None = Field(default=None, description="Error message if processing failed")

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the processing"
    )

    # Execution timing
    processing_time_ms: float | None = Field(
        default=None, description="Time taken to process the node in milliseconds"
    )

    # Agent information
    agent_name: str | None = Field(
        default=None, description="Name of the agent that processed the node"
    )

    agent_type: str | None = Field(
        default=None, description="Type of agent (atomizer, planner, executor, etc.)"
    )

    def __str__(self) -> str:
        """String representation."""
        parts = [f"action={self.action}"]

        if self.envelope:
            parts.append(f"envelope={type(self.envelope).__name__}")

        if self.new_nodes:
            parts.append(f"new_nodes={len(self.new_nodes)}")

        if self.error:
            parts.append(f"error='{self.error[:50]}...'")

        return f"NodeResult({', '.join(parts)})"

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"NodeResult("
            f"task_id={self.task_id}, "
            f"action={self.action}, "
            f"envelope={self.envelope}, "
            f"new_nodes={len(self.new_nodes)}, "
            f"error={self.error}, "
            f"metadata={self.metadata}, "
            f"processing_time_ms={self.processing_time_ms}, "
            f"agent_name={self.agent_name}, "
            f"agent_type={self.agent_type})"
        )

    @property
    def is_successful(self) -> bool:
        """Check if the processing was successful."""
        return self.action not in {NodeAction.FAIL} and self.error is None

    @property
    def has_result_data(self) -> bool:
        """Check if the result contains actual data."""
        return self.envelope is not None

    @property
    def has_subtasks(self) -> bool:
        """Check if the result contains subtasks."""
        return len(self.new_nodes) > 0

    def validate_consistency(self) -> None:
        """Validate that the result is internally consistent."""
        if self.action == NodeAction.ADD_SUBTASKS and not self.has_subtasks:
            raise ValueError("ADD_SUBTASKS action requires new_nodes to be provided")

        if self.action == NodeAction.FAIL and not self.error:
            raise ValueError("FAIL action requires error message")

        if self.action == NodeAction.COMPLETE and not self.has_result_data:
            raise ValueError("COMPLETE action should have result envelope")

        if self.new_nodes and self.action != NodeAction.ADD_SUBTASKS:
            raise ValueError(
                f"new_nodes should only be provided with ADD_SUBTASKS action, got {self.action}"
            )

    @classmethod
    def success(
        cls,
        task_id: str,
        envelope: AnyResultEnvelope,
        agent_name: str | None = None,
        agent_type: str | None = None,
        processing_time_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "NodeResult":
        """Create a successful completion result."""
        return cls(
            task_id=task_id,
            action=NodeAction.COMPLETE,
            envelope=envelope,
            agent_name=agent_name,
            agent_type=agent_type,
            processing_time_ms=processing_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def planning_result(
        cls,
        task_id: str,
        subtasks: list[TaskNode],
        envelope: AnyResultEnvelope | None = None,
        agent_name: str | None = None,
        processing_time_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "NodeResult":
        """Create a planning result with subtasks."""
        return cls(
            task_id=task_id,
            action=NodeAction.ADD_SUBTASKS,
            envelope=envelope,
            new_nodes=subtasks,
            agent_name=agent_name,
            agent_type="planner",
            processing_time_ms=processing_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def aggregation_result(
        cls,
        task_id: str,
        envelope: AnyResultEnvelope,
        agent_name: str | None = None,
        processing_time_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "NodeResult":
        """Create an aggregation result."""
        return cls(
            task_id=task_id,
            action=NodeAction.AGGREGATE,  # Keep as AGGREGATE
            envelope=envelope,
            agent_name=agent_name,
            agent_type="aggregator",
            processing_time_ms=processing_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        task_id: str,
        error: str,
        agent_name: str | None = None,
        agent_type: str | None = None,
        processing_time_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "NodeResult":
        """Create a failure result."""
        return cls(
            task_id=task_id,
            action=NodeAction.FAIL,
            error=error,
            agent_name=agent_name,
            agent_type=agent_type,
            processing_time_ms=processing_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def retry(
        cls,
        task_id: str,
        error: str,
        agent_name: str | None = None,
        agent_type: str | None = None,
        processing_time_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "NodeResult":
        """Create a retry result."""
        return cls(
            task_id=task_id,
            action=NodeAction.RETRY,
            error=error,
            agent_name=agent_name,
            agent_type=agent_type,
            processing_time_ms=processing_time_ms,
            metadata=metadata or {},
        )

    @classmethod
    def replan(
        cls,
        task_id: str,
        parent_id: str | None = None,
        reason: str | None = None,
        agent_name: str | None = None,
        agent_type: str | None = None,
        processing_time_ms: float | None = None,
        **kwargs: Any,
    ) -> "NodeResult":
        """Create a replan result to mark node/parent for replanning."""
        return cls(
            task_id=task_id,
            action=NodeAction.REPLAN,
            agent_name=agent_name,
            agent_type=agent_type,
            processing_time_ms=processing_time_ms,
            metadata={"parent_id": parent_id, "reason": reason or "replanning_requested", **kwargs},
        )
