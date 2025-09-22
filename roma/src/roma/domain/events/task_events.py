"""
Task Events for ROMA v2.0

Immutable event definitions for task lifecycle tracking.
Provides complete observability into task state changes.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from ..value_objects.task_status import TaskStatus
from ..value_objects.task_type import TaskType
from ..value_objects.node_type import NodeType


def utc_now() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class BaseTaskEvent:
    """
    Base class for all task events.
    
    Every event in the system inherits from this base,
    ensuring consistent event structure and metadata.
    """
    
    event_id: str
    task_id: str
    timestamp: datetime
    event_type: str
    metadata: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate event invariants."""
        if not self.event_id:
            raise ValueError("Event ID cannot be empty")
        if not self.task_id:
            raise ValueError("Task ID cannot be empty")
        if not self.event_type:
            raise ValueError("Event type cannot be empty")


@dataclass(frozen=True, slots=True)
class TaskCreatedEvent(BaseTaskEvent):
    """
    Emitted when a new task is created.
    
    This is the first event in every task's lifecycle.
    """
    
    goal: str
    task_type: TaskType
    parent_id: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        task_id: str,
        goal: str,
        task_type: TaskType,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TaskCreatedEvent":
        """Create a TaskCreatedEvent with default values."""
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="task_created",
            metadata=metadata or {},
            goal=goal,
            task_type=task_type,
            parent_id=parent_id
        )


@dataclass(frozen=True, slots=True)
class TaskNodeAddedEvent(BaseTaskEvent):
    """
    Compatibility event for test suites expecting a node-added event name.

    Mirrors TaskCreatedEvent semantics but with a different class name.
    """
    goal: str
    task_type: TaskType
    parent_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        task_id: str,
        goal: str,
        task_type: TaskType,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TaskNodeAddedEvent":
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            # Keep event_type aligned with task creation for downstream compatibility
            event_type="task_created",
            metadata=metadata or {},
            goal=goal,
            task_type=task_type,
            parent_id=parent_id
        )


@dataclass(frozen=True, slots=True)
class TaskStatusChangedEvent(BaseTaskEvent):
    """
    Emitted when a task's status changes.
    
    Tracks all status transitions throughout the task lifecycle.
    """
    
    old_status: TaskStatus
    new_status: TaskStatus
    version: int
    
    @classmethod
    def create(
        cls,
        task_id: str,
        old_status: TaskStatus,
        new_status: TaskStatus,
        version: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TaskStatusChangedEvent":
        """Create a TaskStatusChangedEvent with default values."""
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="task_status_changed",
            metadata=metadata or {},
            old_status=old_status,
            new_status=new_status,
            version=version
        )


@dataclass(frozen=True, slots=True)
class AtomizerEvaluatedEvent(BaseTaskEvent):
    """
    Emitted when atomizer evaluates a task.
    
    Records the atomizer's decision and reasoning.
    """
    
    is_atomic: bool
    node_type: NodeType
    reasoning: str
    confidence: float
    updated_goal: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        task_id: str,
        is_atomic: bool,
        node_type: NodeType,
        reasoning: str,
        confidence: float,
        updated_goal: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AtomizerEvaluatedEvent":
        """Create an AtomizerEvaluatedEvent with default values."""
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="atomizer_evaluated",
            metadata=metadata or {},
            is_atomic=is_atomic,
            node_type=node_type,
            reasoning=reasoning,
            confidence=confidence,
            updated_goal=updated_goal
        )


@dataclass(frozen=True, slots=True) 
class TaskDecomposedEvent(BaseTaskEvent):
    """
    Emitted when a task is decomposed into subtasks.
    
    Records the planning decision and subtask creation.
    """
    
    subtask_count: int
    subtask_ids: list[str]
    planning_reasoning: Optional[str] = None
    
    @classmethod
    def create(
        cls,
        task_id: str,
        subtask_ids: list[str],
        planning_reasoning: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TaskDecomposedEvent":
        """Create a TaskDecomposedEvent with default values."""
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="task_decomposed",
            metadata=metadata or {},
            subtask_count=len(subtask_ids),
            subtask_ids=subtask_ids.copy(),
            planning_reasoning=planning_reasoning
        )


@dataclass(frozen=True, slots=True)
class TaskExecutedEvent(BaseTaskEvent):
    """
    Emitted when a task is executed atomically.
    
    Records execution details and performance metrics.
    """
    
    execution_duration_ms: float
    agent_name: Optional[str] = None
    tools_used: Optional[list[str]] = None
    
    @classmethod
    def create(
        cls,
        task_id: str,
        execution_duration_ms: float,
        agent_name: Optional[str] = None,
        tools_used: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TaskExecutedEvent":
        """Create a TaskExecutedEvent with default values.""" 
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="task_executed",
            metadata=metadata or {},
            execution_duration_ms=execution_duration_ms,
            agent_name=agent_name,
            tools_used=tools_used.copy() if tools_used else None
        )


@dataclass(frozen=True, slots=True)
class TaskCompletedEvent(BaseTaskEvent):
    """
    Emitted when a task completes successfully.
    
    Marks the successful end of a task's lifecycle.
    """
    
    result_summary: str
    total_duration_ms: Optional[float] = None
    
    @classmethod
    def create(
        cls,
        task_id: str,
        result_summary: str,
        total_duration_ms: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TaskCompletedEvent":
        """Create a TaskCompletedEvent with default values."""
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="task_completed",
            metadata=metadata or {},
            result_summary=result_summary,
            total_duration_ms=total_duration_ms
        )


@dataclass(frozen=True, slots=True)
class TaskFailedEvent(BaseTaskEvent):
    """
    Emitted when a task fails.
    
    Records failure details for debugging and recovery.
    """
    
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    recovery_attempted: bool = False
    
    @classmethod
    def create(
        cls,
        task_id: str,
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        recovery_attempted: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "TaskFailedEvent":
        """Create a TaskFailedEvent with default values."""
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="task_failed",
            metadata=metadata or {},
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            recovery_attempted=recovery_attempted
        )


@dataclass(frozen=True, slots=True)
class ResultsAggregatedEvent(BaseTaskEvent):
    """
    Emitted when child task results are aggregated.
    
    Records aggregation process and final result.
    """
    
    child_count: int
    aggregation_strategy: str
    aggregation_duration_ms: float
    
    @classmethod
    def create(
        cls,
        task_id: str,
        child_count: int,
        aggregation_strategy: str,
        aggregation_duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "ResultsAggregatedEvent":
        """Create a ResultsAggregatedEvent with default values."""
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="results_aggregated",
            metadata=metadata or {},
            child_count=child_count,
            aggregation_strategy=aggregation_strategy,
            aggregation_duration_ms=aggregation_duration_ms
        )


# Union type for all task events
@dataclass(frozen=True, slots=True)
class DependencyAddedEvent(BaseTaskEvent):
    """
    Emitted when a dependency edge is added between tasks.

    Records dependency relationships for observability and debugging.
    """

    dependency_id: str
    dependency_type: str = "task_dependency"

    @classmethod
    def create(
        cls,
        task_id: str,
        dependency_id: str,
        dependency_type: str = "task_dependency",
        metadata: Optional[Dict[str, Any]] = None
    ) -> "DependencyAddedEvent":
        """Create a DependencyAddedEvent with default values."""
        return cls(
            event_id=str(uuid4()),
            task_id=task_id,
            timestamp=utc_now(),
            event_type="dependency_added",
            metadata=metadata or {},
            dependency_id=dependency_id,
            dependency_type=dependency_type
        )


TaskEvent = (
    TaskCreatedEvent |
    TaskStatusChangedEvent |
    AtomizerEvaluatedEvent |
    TaskDecomposedEvent |
    TaskExecutedEvent |
    TaskCompletedEvent |
    TaskFailedEvent |
    ResultsAggregatedEvent |
    DependencyAddedEvent
)
