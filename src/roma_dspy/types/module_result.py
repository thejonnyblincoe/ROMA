"""
Module execution result tracking for comprehensive node history.
"""

from datetime import datetime
from typing import Any, Optional, Dict, List
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


class ModuleResult(BaseModel):
    """
    Result of a module execution (atomizer, planner, executor, aggregator).
    Tracks all inputs, outputs, and metadata for complete observability.
    """

    module_name: str = Field(description="Name of the module (atomizer, planner, executor, aggregator)")
    input: Any = Field(description="Input provided to the module")
    output: Any = Field(description="Output produced by the module")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(), description="Execution timestamp")
    duration: float = Field(default=0.0, description="Execution duration in seconds")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        arbitrary_types_allowed = True


class StateTransition(BaseModel):
    """Record of a state transition in the task lifecycle."""

    from_state: str = Field(description="Previous state")
    to_state: str = Field(description="New state")
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    reason: Optional[str] = Field(default=None, description="Reason for transition")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeMetrics(BaseModel):
    """Performance and execution metrics for a task node."""

    atomizer_duration: Optional[float] = Field(default=None, description="Atomizer execution time")
    planner_duration: Optional[float] = Field(default=None, description="Planner execution time")
    executor_duration: Optional[float] = Field(default=None, description="Executor execution time")
    aggregator_duration: Optional[float] = Field(default=None, description="Aggregator execution time")
    total_duration: Optional[float] = Field(default=None, description="Total execution time")
    retry_count: int = Field(default=0, description="Number of retries")
    subtasks_created: int = Field(default=0, description="Number of subtasks created")
    max_depth_reached: int = Field(default=0, description="Maximum recursion depth reached")

    def calculate_total_duration(self) -> float:
        """Calculate total duration from component durations."""
        durations = [
            self.atomizer_duration or 0,
            self.planner_duration or 0,
            self.executor_duration or 0,
            self.aggregator_duration or 0
        ]
        return sum(durations)


class ExecutionEvent(BaseModel):
    """Event in the execution timeline for tracking and visualization."""

    node_id: str = Field(description="ID of the task node")
    module_name: str = Field(description="Module that was executed")
    event_type: str = Field(description="Type of event (start, complete, error)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now())
    duration: Optional[float] = Field(default=None, description="Duration if event is complete")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True