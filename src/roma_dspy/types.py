"""Type definitions and Pydantic models for the hierarchical task decomposition system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TaskStatus(Enum):
    """Status of a task in the execution pipeline."""

    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    """Type of task based on decomposition."""

    ATOMIC = "atomic"  # Cannot be further decomposed
    COMPOSITE = "composite"  # Can be decomposed into subtasks
    ROOT = "root"  # Top-level task


class NodeState(BaseModel):
    """State of a node in the task execution graph."""

    node_id: str = Field(description="Unique identifier for the node")
    task_id: str = Field(description="Associated task ID")
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    retries: int = Field(default=0, description="Number of retry attempts")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """Represents a task in the hierarchical decomposition."""

    id: str = Field(description="Unique task identifier")
    description: str = Field(description="Natural language description of the task")
    type: TaskType = Field(default=TaskType.COMPOSITE)
    parent_id: Optional[str] = Field(default=None, description="Parent task ID")
    children_ids: List[str] = Field(default_factory=list, description="Child task IDs")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")

    # Execution details
    tool: Optional[str] = Field(default=None, description="Tool to use for atomic tasks")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")

    # Results
    result: Optional[Any] = Field(default=None, description="Task execution result")
    status: TaskStatus = Field(default=TaskStatus.PENDING)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    depth: int = Field(default=0, description="Depth in the task tree")
    max_depth: int = Field(default=5, description="Maximum allowed decomposition depth")

    def is_atomic(self) -> bool:
        """Check if task is atomic (cannot be further decomposed)."""
        return self.type == TaskType.ATOMIC or self.depth >= self.max_depth

    def is_ready(self, completed_tasks: List[str]) -> bool:
        """Check if task is ready for execution based on dependencies."""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)


class Plan(BaseModel):
    """Execution plan for a task decomposition."""

    id: str = Field(description="Unique plan identifier")
    root_task_id: str = Field(description="Root task ID")
    tasks: Dict[str, Task] = Field(default_factory=dict, description="All tasks in the plan")
    execution_order: List[str] = Field(
        default_factory=list,
        description="Topological ordering of tasks"
    )

    # Execution strategy
    parallel_enabled: bool = Field(default=True, description="Allow parallel execution")
    max_parallel_tasks: int = Field(default=5, description="Maximum parallel tasks")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    total_tasks: int = Field(default=0)
    completed_tasks: int = Field(default=0)
    failed_tasks: int = Field(default=0)

    def get_ready_tasks(self, completed_task_ids: List[str]) -> List[Task]:
        """Get tasks that are ready for execution."""
        ready = []
        for task_id, task in self.tasks.items():
            if (
                task.status == TaskStatus.PENDING
                and task.is_ready(completed_task_ids)
            ):
                ready.append(task)
        return ready

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Update the status of a task."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].updated_at = datetime.now()

            if status == TaskStatus.COMPLETED:
                self.completed_tasks += 1
            elif status == TaskStatus.FAILED:
                self.failed_tasks += 1


class Result(BaseModel):
    """Result of task execution."""

    task_id: str = Field(description="Task ID this result belongs to")
    success: bool = Field(description="Whether execution was successful")
    output: Optional[Any] = Field(default=None, description="Task output")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    # Execution details
    execution_time: float = Field(description="Execution time in seconds")
    llm_calls: int = Field(default=0, description="Number of LLM calls made")
    tool_calls: int = Field(default=0, description="Number of tool calls made")

    # For composite tasks
    subtask_results: List["Result"] = Field(
        default_factory=list,
        description="Results from subtasks"
    )

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def aggregate(self) -> Any:
        """Aggregate results from subtasks."""
        if not self.subtask_results:
            return self.output

        # Simple aggregation - can be made more sophisticated
        aggregated = {
            "main_output": self.output,
            "subtasks": [
                {
                    "task_id": r.task_id,
                    "success": r.success,
                    "output": r.output,
                }
                for r in self.subtask_results
            ],
        }
        return aggregated


class ToolCall(BaseModel):
    """Represents a tool invocation."""

    tool_name: str = Field(description="Name of the tool")
    parameters: Dict[str, Any] = Field(description="Tool parameters")
    result: Optional[Any] = Field(default=None, description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error if tool failed")
    timestamp: datetime = Field(default_factory=datetime.now)


# Update forward references for recursive models
Result.model_rebuild()