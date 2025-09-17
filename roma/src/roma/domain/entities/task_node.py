"""
Immutable TaskNode entity for ROMA v2.0

Core entity representing a task in the hierarchical execution graph.
Thread-safe through immutability with state transition methods.
"""

from datetime import datetime, timezone
from typing import Any, Dict, FrozenSet, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, field_validator

from ..value_objects.task_type import TaskType
from ..value_objects.node_type import NodeType
from ..value_objects.task_status import TaskStatus


def uuid4_str() -> str:
    """Generate a UUID4 string."""
    return str(uuid4())


class TaskNode(BaseModel):
    """
    Immutable task node representing a unit of work in ROMA's execution graph.
    
    Key principles:
    - Completely immutable (frozen=True) for thread safety
    - State transitions return new instances
    - Version tracking for optimistic concurrency control
    - Typed relationships using frozensets
    
    Lifecycle:
    1. Created with PENDING status
    2. Atomizer determines PLAN or EXECUTE node_type
    3. State transitions through READY → EXECUTING → COMPLETED/FAILED
    4. Parent nodes AGGREGATE results from children
    """
    
    model_config = ConfigDict(frozen=True, validate_assignment=True, use_enum_values=False)
    
    # Identity and structure
    task_id: str = Field(default_factory=uuid4_str, description="Unique task identifier")
    parent_id: Optional[str] = Field(default=None, description="Parent task ID")
    goal: str = Field(default="", min_length=1, description="Task objective")
    
    # MECE classification and atomizer decision  
    task_type: TaskType = Field(default=TaskType.THINK, description="MECE task classification")
    node_type: Optional[NodeType] = Field(default=None, description="Atomizer decision: PLAN or EXECUTE")
    
    # State management
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    version: int = Field(default=0, ge=0, description="Version for optimistic locking")
    
    # Recovery management
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts made")
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries allowed")
    
    # Execution results
    result: Optional[Any] = Field(default=None, description="Task execution result")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Immutable relationships
    dependencies: FrozenSet[str] = Field(default_factory=frozenset, description="Task dependencies")
    children: FrozenSet[str] = Field(default_factory=frozenset, description="Child tasks")
    
    # Timestamps for observability
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start execution timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    
    @field_validator('goal')
    @classmethod
    def validate_goal(cls, v: str) -> str:
        """Validate goal is not empty or whitespace-only."""
        if not v.strip():
            raise ValueError("Task goal cannot be empty")
        return v
    
    
    def transition_to(
        self, 
        status: TaskStatus, 
        **updates: Any
    ) -> "TaskNode":
        """
        Create new instance with status transition and optional updates.
        
        Args:
            status: Target status to transition to
            **updates: Additional field updates
            
        Returns:
            New TaskNode instance with updated status
            
        Raises:
            ValueError: If status transition is invalid
        """
        if not self.status.can_transition_to_status(status):
            raise ValueError(
                f"Invalid transition from {self.status} to {status}. "
                f"Valid transitions: {self.status.can_transition_to}"
            )
        
        # Auto-update timestamps based on status
        timestamp_updates = {}
        if status == TaskStatus.EXECUTING and not self.started_at:
            timestamp_updates['started_at'] = datetime.now(timezone.utc)
        elif status.is_terminal and not self.completed_at:
            timestamp_updates['completed_at'] = datetime.now(timezone.utc)
        
        # Increment version for optimistic locking
        all_updates = {
            'status': status,
            'version': self.version + 1,
            **timestamp_updates,
            **updates
        }
        
        return self.model_copy(update=all_updates)
    
    def with_result(self, result: Any, metadata: Optional[Dict[str, Any]] = None) -> "TaskNode":
        """
        Create new instance with successful completion.
        
        Args:
            result: The execution result
            metadata: Optional metadata to merge
            
        Returns:
            New TaskNode instance with COMPLETED status and result
        """
        updates: Dict[str, Any] = {'result': result}
        if metadata:
            updates['metadata'] = {**self.metadata, **metadata}
            
        return self.transition_to(TaskStatus.COMPLETED, **updates)
    
    def with_error(self, error: str, metadata: Optional[Dict[str, Any]] = None) -> "TaskNode":
        """
        Create new instance with failure.
        
        Args:
            error: Error message or description
            metadata: Optional metadata to merge
            
        Returns:
            New TaskNode instance with FAILED status and error
        """
        updates: Dict[str, Any] = {'error': error}
        if metadata:
            updates['metadata'] = {**self.metadata, **metadata}
            
        return self.transition_to(TaskStatus.FAILED, **updates)
    
    def add_child(self, child_id: str) -> "TaskNode":
        """
        Create new instance with additional child.
        
        Args:
            child_id: ID of child task to add
            
        Returns:
            New TaskNode instance with child added
        """
        if child_id in self.children:
            return self  # No change needed
            
        return self.model_copy(update={
            "children": self.children | {child_id},
            "version": self.version + 1
        })
    
    def remove_child(self, child_id: str) -> "TaskNode":
        """
        Create new instance with child removed.
        
        Args:
            child_id: ID of child task to remove
            
        Returns:
            New TaskNode instance with child removed
        """
        if child_id not in self.children:
            return self  # No change needed
            
        return self.model_copy(update={
            "children": self.children - {child_id},
            "version": self.version + 1
        })
    
    def add_dependency(self, dependency_id: str) -> "TaskNode":
        """
        Create new instance with additional dependency.
        
        Args:
            dependency_id: ID of dependency task to add
            
        Returns:
            New TaskNode instance with dependency added
        """
        if dependency_id in self.dependencies:
            return self  # No change needed
            
        return self.model_copy(update={
            "dependencies": self.dependencies | {dependency_id},
            "version": self.version + 1
        })
    
    def remove_dependency(self, dependency_id: str) -> "TaskNode":
        """
        Create new instance with dependency removed.
        
        Args:
            dependency_id: ID of dependency task to remove
            
        Returns:
            New TaskNode instance with dependency removed
        """
        if dependency_id not in self.dependencies:
            return self  # No change needed
            
        return self.model_copy(update={
            "dependencies": self.dependencies - {dependency_id},
            "version": self.version + 1
        })
    
    def update_metadata(self, **metadata: Any) -> "TaskNode":
        """
        Create new instance with updated metadata.
        
        Args:
            **metadata: Metadata fields to update
            
        Returns:
            New TaskNode instance with merged metadata
        """
        return self.model_copy(update={
            "metadata": {**self.metadata, **metadata},
            "version": self.version + 1
        })
    
    def set_node_type(self, node_type: NodeType) -> "TaskNode":
        """
        Create new instance with node type set (typically by atomizer).
        
        Args:
            node_type: NodeType determined by atomizer
            
        Returns:
            New TaskNode instance with node_type set
            
        Raises:
            ValueError: If node_type conflicts with task_type constraints
        """
        # All task types can be either PLAN or EXECUTE based on atomizer decision
        # No special constraints - the atomizer handles complexity evaluation
        
        return self.model_copy(update={
            "node_type": node_type,
            "version": self.version + 1
        })
    
    # Properties for convenience
    @property
    def is_atomic(self) -> bool:
        """Check if task is atomic (EXECUTE node_type)."""
        return self.node_type == NodeType.EXECUTE
    
    @property
    def is_composite(self) -> bool:
        """Check if task needs decomposition (PLAN node_type)."""
        return self.node_type == NodeType.PLAN
    
    @property
    def is_root(self) -> bool:
        """Check if this is a root task (no parent)."""
        return self.parent_id is None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf task (no children)."""
        return len(self.children) == 0
    
    @property
    def has_dependencies(self) -> bool:
        """Check if task has dependencies."""
        return len(self.dependencies) > 0
    
    @property
    def execution_duration(self) -> Optional[float]:
        """
        Calculate execution duration in seconds if available.
        
        Returns:
            Duration in seconds, or None if not completed
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries
    
    @property
    def retry_exhausted(self) -> bool:
        """Check if all retries have been exhausted."""
        return self.retry_count >= self.max_retries
    
    def increment_retry(self) -> "TaskNode":
        """
        Create new instance with incremented retry count.
        
        Returns:
            New TaskNode instance with incremented retry count
            
        Raises:
            ValueError: If maximum retries already reached
        """
        if self.retry_exhausted:
            raise ValueError(f"Maximum retries ({self.max_retries}) already reached")
            
        return self.model_copy(update={
            "retry_count": self.retry_count + 1,
            "version": self.version + 1
        })
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        node_type_str = f"({self.node_type.value})" if self.node_type else ""
        return f"TaskNode[{self.task_id[:8]}]{node_type_str}: {self.goal[:50]}..."
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TaskNode to dictionary representation."""
        return {
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "goal": self.goal,
            "task_type": self.task_type.value,
            "node_type": self.node_type.value if self.node_type else None,
            "status": self.status.value,
            "version": self.version,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "result": self.result,
            "error": self.error,
            "metadata": dict(self.metadata),
            "dependencies": list(self.dependencies),
            "children": list(self.children),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_duration": self.execution_duration,
            "is_atomic": self.is_atomic,
            "is_composite": self.is_composite,
            "is_root": self.is_root,
            "is_leaf": self.is_leaf,
            "can_retry": self.can_retry,
            "retry_exhausted": self.retry_exhausted
        }
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"TaskNode(task_id='{self.task_id}', "
            f"goal='{self.goal[:30]}...', "
            f"task_type={self.task_type}, "
            f"node_type={self.node_type}, "
            f"status={self.status}, "
            f"version={self.version})"
        )