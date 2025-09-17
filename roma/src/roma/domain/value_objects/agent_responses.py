"""
Agent Response Models for ROMA v2.0

Pydantic response models for structured agent outputs.
These models are used with Agno's native response_model parameter.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from src.roma.domain.value_objects.node_type import NodeType
from src.roma.domain.value_objects.task_type import TaskType


class AtomizerResult(BaseModel):
    """
    Atomizer decision result.

    Determines whether a task should be executed atomically or decomposed into subtasks.
    """
    model_config = {"frozen": True, "validate_assignment": True}

    is_atomic: bool = Field(..., description="True if task can be executed directly")
    node_type: NodeType = Field(..., description="Determined node type (PLAN or EXECUTE)")
    reasoning: str = Field(..., min_length=1, description="Explanation of the decision")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Decision confidence score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional decision context")

    @field_validator('node_type')
    @classmethod
    def validate_node_type_consistency(cls, v: NodeType, info):
        """Validate consistency between is_atomic and node_type."""
        if hasattr(info, 'data') and 'is_atomic' in info.data:
            is_atomic = info.data['is_atomic']
            expected_node_type = NodeType.EXECUTE if is_atomic else NodeType.PLAN
            if v != expected_node_type:
                raise ValueError(
                    f"Inconsistent atomizer result: is_atomic={is_atomic} "
                    f"but node_type={v}"
                )
        return v


class SubTask(BaseModel):
    """
    Individual subtask in a decomposition plan.
    """
    model_config = {"frozen": True}

    goal: str = Field(..., min_length=1, description="Precise subtask objective")
    task_type: TaskType = Field(..., description="Type of subtask")
    priority: int = Field(default=0, description="Execution priority (higher = more important)")
    dependencies: List[str] = Field(default_factory=list, description="List of subtask IDs this depends on")
    estimated_effort: Optional[int] = Field(default=None, ge=1, le=10, description="Effort estimate (1-10)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional subtask context")


class PlannerResult(BaseModel):
    """
    Planner decomposition result.

    Contains the breakdown of a complex task into executable subtasks.
    """
    model_config = {"frozen": True}

    subtasks: List[SubTask] = Field(..., min_items=1, description="List of planned subtasks")
    strategy: str = Field(default="parallel", description="Execution strategy")
    reasoning: Optional[str] = Field(default=None, description="Planning rationale")
    estimated_total_effort: Optional[int] = Field(default=None, ge=1, description="Total effort estimate")
    dependencies_graph: Optional[Dict[str, List[str]]] = Field(default=None, description="Task dependency mapping")


class ExecutorResult(BaseModel):
    """
    Executor execution result.

    Contains the output of atomic task execution.
    """
    model_config = {"frozen": True}

    result: Any = Field(..., description="Primary execution result")
    sources: List[str] = Field(default_factory=list, description="Information sources used")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata")
    success: bool = Field(default=True, description="Whether execution was successful")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Result confidence score")
    tokens_used: Optional[int] = Field(default=None, ge=0, description="Tokens consumed during execution")
    execution_time: Optional[float] = Field(default=None, ge=0.0, description="Execution time in seconds")


class AggregatorResult(BaseModel):
    """
    Aggregator synthesis result.

    Contains the synthesis of multiple subtask results into a cohesive output.
    """
    model_config = {"frozen": True}

    synthesized_result: str = Field(..., min_length=1, description="Final synthesized output")
    summary: str = Field(..., min_length=1, description="Executive summary")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Synthesis confidence score")
    sources_used: List[str] = Field(default_factory=list, description="Source subtasks referenced")
    gaps_identified: List[str] = Field(default_factory=list, description="Information gaps found")
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Output quality assessment")


class PlanModifierResult(BaseModel):
    """
    Plan modification result.

    Contains modifications to an existing plan based on feedback or new information.
    """
    model_config = {"frozen": True}

    modified_subtasks: List[SubTask] = Field(..., description="Updated subtask list")
    changes_made: List[str] = Field(..., min_items=1, description="Description of changes applied")
    reasoning: str = Field(..., min_length=1, description="Rationale for modifications")
    impact_assessment: Optional[str] = Field(default=None, description="Assessment of change impact")
    new_dependencies: Optional[Dict[str, List[str]]] = Field(default=None, description="Updated dependency graph")


# Convenience functions for creating results
def create_atomic_result(reasoning: str, confidence: float = 1.0) -> AtomizerResult:
    """Create an atomic (EXECUTE) atomizer result."""
    return AtomizerResult(
        is_atomic=True,
        node_type=NodeType.EXECUTE,
        reasoning=reasoning,
        confidence=confidence
    )


def create_composite_result(reasoning: str, confidence: float = 1.0) -> AtomizerResult:
    """Create a composite (PLAN) atomizer result."""
    return AtomizerResult(
        is_atomic=False,
        node_type=NodeType.PLAN,
        reasoning=reasoning,
        confidence=confidence
    )


def create_successful_execution(result: Any, sources: List[str] = None) -> ExecutorResult:
    """Create a successful executor result."""
    return ExecutorResult(
        result=result,
        sources=sources or [],
        success=True
    )


def create_failed_execution(error_msg: str, metadata: Dict[str, Any] = None) -> ExecutorResult:
    """Create a failed executor result."""
    return ExecutorResult(
        result=error_msg,
        metadata=metadata or {},
        success=False,
        confidence=0.0
    )