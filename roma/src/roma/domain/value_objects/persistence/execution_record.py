"""
Execution Record Pydantic Models for Strong Typing.

Domain value objects for execution records with strong type safety.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExecutionRecord(BaseModel):
    """Strongly typed execution record structure."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    task_id: str
    goal: str
    task_type: str
    node_type: str | None = None
    status: str
    parent_task_id: str | None = None
    root_task_id: str | None = None
    depth_level: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    execution_duration_ms: int | None = None
    result: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_info: dict[str, Any] | None = None
    agent_config: dict[str, Any] | None = None
    execution_context: dict[str, Any] | None = None
    retry_count: int = 0
    max_retries: int = 3
    token_usage: dict[str, Any] | None = None
    cost_info: dict[str, Any] | None = None
    created_at: str
    updated_at: str
    children: list["ExecutionRecord"] = Field(default_factory=list)


class ExecutionTreeNode(BaseModel):
    """Strongly typed execution tree node structure."""

    model_config = ConfigDict(frozen=True)

    execution_id: str
    task_id: str
    goal: str
    task_type: str
    node_type: str | None = None
    status: str
    depth_level: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    execution_duration_ms: int | None = None
    relationship_type: str | None = None
    order_index: int | None = None
    children: list["ExecutionTreeNode"] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    """Performance metrics for execution analytics."""

    model_config = ConfigDict(frozen=True)

    avg_duration_ms: float | None = None
    min_duration_ms: int | None = None
    max_duration_ms: int | None = None


class AnalysisPeriod(BaseModel):
    """Analysis time period for execution analytics."""

    model_config = ConfigDict(frozen=True)

    start_time: str | None = None
    end_time: str | None = None


class ExecutionAnalytics(BaseModel):
    """Strongly typed execution analytics structure."""

    model_config = ConfigDict(frozen=True)

    total_executions: int
    status_distribution: dict[str, int] = Field(default_factory=dict)
    task_type_distribution: dict[str, int] = Field(default_factory=dict)
    performance_metrics: PerformanceMetrics
    analysis_period: AnalysisPeriod
