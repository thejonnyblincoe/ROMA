from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, FrozenSet
from src.roma_dspy.types.task_type import TaskType
from src.roma_dspy.types.node_type import NodeType
from src.roma_dspy.types.task_status import TaskStatus
from datetime import datetime, timezone
from uuid import uuid4

class TaskNode(BaseModel):
    """
    Immutable task node representing a unit of work in ROMA's execution graph.
    
    Key principles:
    - Completely immutable (frozen=True) for thread safety
    - State transitions return new instances
    - Typed relationships using frozensets
    
    Lifecycle:
    1. Created with PENDING status
    2. Atomizer determines PLAN or EXECUTE node_type
    3. State transitions through READY → EXECUTING → COMPLETED/FAILED
    4. Parent nodes AGGREGATE results from children
    """
    
    # Identity and structure
    parent_id: Optional[str] = Field(default=None, description="Parent task ID")
    goal: str = Field(default="", min_length=1, description="Task objective")
    
    # MECE classification and atomizer decision  
    task_type: TaskType = Field(default=TaskType.THINK, description="MECE task classification")
    node_type: Optional[NodeType] = Field(default=None, description="Atomizer decision: PLAN or EXECUTE")
    
    # State management
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    
    # Execution results
    result: Optional[Any] = Field(default=None, description="Task execution result")
    
    # Immutable relationships
    dependencies: FrozenSet[str] = Field(default_factory=frozenset, description="Task dependencies")
    children: FrozenSet[str] = Field(default_factory=frozenset, description="Child tasks")

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
        })
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        node_type_str = f"({self.node_type.value})" if self.node_type else ""
        return f"TaskNode[{self.task_id[:8]}]{node_type_str}: {self.goal[:50]}..."
    