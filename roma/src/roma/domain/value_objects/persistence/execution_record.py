"""
Execution Record Pydantic Models for Strong Typing.

Domain value objects for execution records with strong type safety.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime


class ExecutionRecord(BaseModel):
    """Strongly typed execution record structure."""
    model_config = ConfigDict(frozen=True)

    execution_id: str
    task_id: str
    goal: str
    task_type: str
    node_type: Optional[str] = None
    status: str
    parent_task_id: Optional[str] = None
    root_task_id: Optional[str] = None
    depth_level: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_duration_ms: Optional[int] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_info: Optional[Dict[str, Any]] = None
    agent_config: Optional[Dict[str, Any]] = None
    execution_context: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    token_usage: Optional[Dict[str, Any]] = None
    cost_info: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str
    children: List["ExecutionRecord"] = Field(default_factory=list)


class ExecutionTreeNode(BaseModel):
    """Strongly typed execution tree node structure."""
    model_config = ConfigDict(frozen=True)

    execution_id: str
    task_id: str
    goal: str
    task_type: str
    node_type: Optional[str] = None
    status: str
    depth_level: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    execution_duration_ms: Optional[int] = None
    relationship_type: Optional[str] = None
    order_index: Optional[int] = None
    children: List["ExecutionTreeNode"] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    """Performance metrics for execution analytics."""
    model_config = ConfigDict(frozen=True)

    avg_duration_ms: Optional[float] = None
    min_duration_ms: Optional[int] = None
    max_duration_ms: Optional[int] = None


class AnalysisPeriod(BaseModel):
    """Analysis time period for execution analytics."""
    model_config = ConfigDict(frozen=True)

    start_time: Optional[str] = None
    end_time: Optional[str] = None


class ExecutionAnalytics(BaseModel):
    """Strongly typed execution analytics structure."""
    model_config = ConfigDict(frozen=True)

    total_executions: int
    status_distribution: Dict[str, int] = Field(default_factory=dict)
    task_type_distribution: Dict[str, int] = Field(default_factory=dict)
    performance_metrics: PerformanceMetrics
    analysis_period: AnalysisPeriod