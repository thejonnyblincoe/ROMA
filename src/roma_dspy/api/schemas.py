"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Request Schemas
# ============================================================================

class SolveRequest(BaseModel):
    """Request schema for starting a new task execution."""

    goal: str = Field(..., min_length=1, description="Task goal to decompose and execute")
    max_depth: int = Field(default=2, ge=0, le=10, description="Maximum recursion depth")
    config_profile: Optional[str] = Field(default=None, description="Configuration profile name")
    config_overrides: Optional[Dict[str, Any]] = Field(default=None, description="Configuration overrides")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class CheckpointRestoreRequest(BaseModel):
    """Request schema for restoring from checkpoint."""

    checkpoint_id: str = Field(..., description="Checkpoint ID to restore from")
    resume: bool = Field(default=True, description="Resume execution after restore")


class ConfigUpdateRequest(BaseModel):
    """Request schema for updating configuration."""

    profile: Optional[str] = Field(default=None, description="Configuration profile name")
    overrides: Dict[str, Any] = Field(..., description="Configuration overrides")


# ============================================================================
# Response Schemas
# ============================================================================

class TaskNodeResponse(BaseModel):
    """Response schema for a single task node."""

    task_id: str
    goal: str
    status: str
    depth: int
    node_type: Optional[str] = None
    parent_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DAGStatisticsResponse(BaseModel):
    """Response schema for DAG statistics."""

    dag_id: str
    total_tasks: int
    status_counts: Dict[str, int]
    depth_distribution: Dict[int, int]
    num_subgraphs: int
    is_complete: bool


class ExecutionResponse(BaseModel):
    """Response schema for execution metadata."""

    execution_id: str
    status: str
    initial_goal: str
    max_depth: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    created_at: datetime
    updated_at: datetime
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


class ExecutionDetailResponse(ExecutionResponse):
    """Extended execution response with DAG visualization."""

    dag_snapshot: Optional[Dict[str, Any]] = Field(
        default=None,
        deprecated=True,
        description="DEPRECATED: Use checkpoint endpoints (GET /executions/{id}/dag) instead. "
                    "This field now sourced from checkpoints, not Execution.dag_snapshot column. "
                    "Will be removed in v0.3.0."
    )
    statistics: Optional[DAGStatisticsResponse] = None


class ExecutionListResponse(BaseModel):
    """Response schema for listing executions."""

    executions: List[ExecutionResponse]
    total: int
    offset: int
    limit: int


class CheckpointResponse(BaseModel):
    """Response schema for checkpoint metadata."""

    checkpoint_id: str
    execution_id: str
    created_at: datetime
    trigger: str
    state: str
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = None
    compressed: bool


class CheckpointListResponse(BaseModel):
    """Response schema for listing checkpoints."""

    checkpoints: List[CheckpointResponse]
    total: int


class TaskTraceResponse(BaseModel):
    """Response schema for task trace."""

    trace_id: int
    execution_id: str
    task_id: str
    parent_task_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    task_type: str
    node_type: Optional[str] = None
    status: str
    depth: int
    retry_count: int
    goal: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class LMTraceResponse(BaseModel):
    """Response schema for LM trace."""

    trace_id: int
    execution_id: str
    task_id: Optional[str] = None
    module_name: str
    created_at: datetime
    model: str
    temperature: Optional[float] = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response schema for system health check."""

    status: str
    version: str
    uptime_seconds: float
    active_executions: int
    storage_connected: bool
    cache_size: int
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Response schema for API errors."""

    error: str
    detail: Optional[str] = None
    execution_id: Optional[str] = None
    timestamp: datetime


class VisualizationOptions(BaseModel):
    """Options for controlling visualization detail level."""

    show_ids: bool = Field(default=False, description="Show full task IDs")
    show_timing: bool = Field(default=True, description="Show execution times and durations")
    show_tokens: bool = Field(default=True, description="Show token usage and costs")
    max_goal_length: int = Field(default=60, ge=0, description="Maximum goal text length (0 = unlimited)")
    verbose: bool = Field(default=False, description="Enable all details (overrides other flags)")
    fancy: bool = Field(default=True, description="Use Rich library for fancy CLI visualization (default) or plain text")
    show_io: bool = Field(default=False, description="Show full Input/Output panels (off by default)")
    width: Optional[int] = Field(default=None, description="Force console width for rendering (e.g., 180)")


class VisualizationRequest(BaseModel):
    """Request schema for generating visualizations."""

    visualizer_type: str = Field(
        ...,
        description="Type of visualizer: tree, timeline, statistics, context_flow, llm_trace"
    )
    profile: Optional[str] = Field(default=None, description="Configuration profile name to load")
    include_subgraphs: bool = Field(default=True, description="Include subgraph tasks")
    format: str = Field(default="text", description="Output format: text, json")
    data_source: str = Field(
        default="checkpoint",
        description="Data source: 'checkpoint' (DAG snapshots from PostgreSQL) or 'mlflow' (DSPy traces from MLflow)"
    )
    options: Optional[VisualizationOptions] = Field(
        default=None,
        description="Visualization detail options (uses defaults if not provided)"
    )


class VisualizationResponse(BaseModel):
    """Response schema for visualization output."""

    execution_id: str
    visualizer_type: str
    content: str
    format: str
    generated_at: datetime


class MetricsResponse(BaseModel):
    """Response schema for execution metrics."""

    execution_id: str
    total_lm_calls: int
    total_tokens: int
    total_cost_usd: float
    average_latency_ms: float
    task_breakdown: Dict[str, Dict[str, Any]]


class StatusPollingResponse(BaseModel):
    """Response schema for status polling."""

    execution_id: str
    status: str
    progress: float  # 0.0 to 1.0
    current_task_id: Optional[str] = None
    current_task_goal: Optional[str] = None
    completed_tasks: int
    total_tasks: int
    estimated_remaining_seconds: Optional[int] = None
    last_updated: datetime
