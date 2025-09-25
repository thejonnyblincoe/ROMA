"""
Checkpoint Record Pydantic Models for Strong Typing.

Domain value objects for checkpoint records with strong type safety.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .checkpoint_type import CheckpointType


class CheckpointRecord(BaseModel):
    """Strongly typed checkpoint record structure."""

    model_config = ConfigDict(frozen=True)

    id: str
    execution_id: str
    checkpoint_name: str
    checkpoint_type: CheckpointType
    checkpoint_data: dict[str, Any]
    large_data: Any | None = None
    task_graph_snapshot: dict[str, Any] | None = None
    execution_context: dict[str, Any] | None = None
    agent_states: dict[str, Any] | None = None
    sequence_number: int
    recovery_instructions: str | None = None
    dependencies: list[str] | None = None
    created_at: str
    expires_at: str | None = None
    data_size_bytes: int
    creation_duration_ms: int | None = None


class CheckpointSummary(BaseModel):
    """Summary of checkpoint for listing operations."""

    model_config = ConfigDict(frozen=True)

    id: str
    checkpoint_name: str
    checkpoint_type: CheckpointType
    sequence_number: int
    created_at: str
    expires_at: str | None = None
    is_valid: bool = True
    data_size_bytes: int
    creation_duration_ms: int | None = None


class CheckpointStorageMetrics(BaseModel):
    """Storage metrics for checkpoint analytics."""

    model_config = ConfigDict(frozen=True)

    total_storage_bytes: int = 0
    avg_checkpoint_size_bytes: float = 0.0
    avg_creation_duration_ms: float = 0.0


class CheckpointAnalytics(BaseModel):
    """Strongly typed checkpoint analytics structure."""

    model_config = ConfigDict(frozen=True)

    total_checkpoints: int
    valid_checkpoints: int
    type_distribution: dict[str, int] = Field(default_factory=dict)
    storage_metrics: CheckpointStorageMetrics
    service_stats: dict[str, Any] = Field(default_factory=dict)
