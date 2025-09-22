"""
Node Result Value Object.

Defines the result of processing a single task node through the agent pipeline.
Used to communicate outcomes, actions, and data between orchestration components.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from .node_action import NodeAction
from .result_envelope import AnyResultEnvelope
from ..entities.task_node import TaskNode


class NodeResult(BaseModel):
    """Result of processing a single node through the agent pipeline."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    # Required fields
    action: NodeAction = Field(..., description="Action to take based on processing result")

    # Optional result data
    envelope: Optional[AnyResultEnvelope] = Field(
        default=None,
        description="Result envelope from agent execution"
    )

    new_nodes: List[TaskNode] = Field(
        default_factory=list,
        description="New nodes to add to graph (for ADD_SUBTASKS action)"
    )

    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the processing"
    )

    # Execution timing
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Time taken to process the node in milliseconds"
    )

    # Agent information
    agent_name: Optional[str] = Field(
        default=None,
        description="Name of the agent that processed the node"
    )

    agent_type: Optional[str] = Field(
        default=None,
        description="Type of agent (atomizer, planner, executor, etc.)"
    )

    def __str__(self) -> str:
        """String representation."""
        parts = [f"NodeResult(action={self.action}"]

        if self.envelope:
            parts.append(f"envelope={type(self.envelope).__name__}")

        if self.new_nodes:
            parts.append(f"new_nodes={len(self.new_nodes)}")

        if self.error:
            parts.append(f"error='{self.error[:50]}...'")

        parts.append(")")
        return ", ".join(parts)

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"NodeResult("
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
            raise ValueError(f"new_nodes should only be provided with ADD_SUBTASKS action, got {self.action}")

    @classmethod
    def success(
        cls,
        envelope: AnyResultEnvelope,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "NodeResult":
        """Create a successful completion result."""
        return cls(
            action=NodeAction.COMPLETE,
            envelope=envelope,
            agent_name=agent_name,
            agent_type=agent_type,
            processing_time_ms=processing_time_ms,
            metadata=metadata or {}
        )

    @classmethod
    def planning_result(
        cls,
        subtasks: List[TaskNode],
        envelope: Optional[AnyResultEnvelope] = None,
        agent_name: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "NodeResult":
        """Create a planning result with subtasks."""
        return cls(
            action=NodeAction.ADD_SUBTASKS,
            envelope=envelope,
            new_nodes=subtasks,
            agent_name=agent_name,
            agent_type="planner",
            processing_time_ms=processing_time_ms,
            metadata=metadata or {}
        )

    @classmethod
    def aggregation_result(
        cls,
        envelope: AnyResultEnvelope,
        agent_name: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "NodeResult":
        """Create an aggregation result."""
        return cls(
            action=NodeAction.AGGREGATE,
            envelope=envelope,
            agent_name=agent_name,
            agent_type="aggregator",
            processing_time_ms=processing_time_ms,
            metadata=metadata or {}
        )

    @classmethod
    def failure(
        cls,
        error: str,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "NodeResult":
        """Create a failure result."""
        return cls(
            action=NodeAction.FAIL,
            error=error,
            agent_name=agent_name,
            agent_type=agent_type,
            processing_time_ms=processing_time_ms,
            metadata=metadata or {}
        )

    @classmethod
    def retry(
        cls,
        error: str,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        processing_time_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "NodeResult":
        """Create a retry result."""
        return cls(
            action=NodeAction.RETRY,
            error=error,
            agent_name=agent_name,
            agent_type=agent_type,
            processing_time_ms=processing_time_ms,
            metadata=metadata or {}
        )