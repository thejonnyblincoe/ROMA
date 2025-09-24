"""
Checkpoint Record Pydantic Models for Strong Typing.

Domain value objects for checkpoint records with strong type safety.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from .checkpoint_type import CheckpointType


class CheckpointRecord(BaseModel):
    """Strongly typed checkpoint record structure."""
    model_config = ConfigDict(frozen=True)

    id: str
    execution_id: str
    checkpoint_name: str
    checkpoint_type: CheckpointType
    checkpoint_data: Dict[str, Any]
    large_data: Optional[Any] = None
    task_graph_snapshot: Optional[Dict[str, Any]] = None
    execution_context: Optional[Dict[str, Any]] = None
    agent_states: Optional[Dict[str, Any]] = None
    sequence_number: int
    recovery_instructions: Optional[str] = None
    dependencies: Optional[List[str]] = None
    created_at: str
    expires_at: Optional[str] = None
    data_size_bytes: int
    creation_duration_ms: Optional[int] = None


class CheckpointSummary(BaseModel):
    """Summary of checkpoint for listing operations."""
    model_config = ConfigDict(frozen=True)

    id: str
    checkpoint_name: str
    checkpoint_type: CheckpointType
    sequence_number: int
    created_at: str
    expires_at: Optional[str] = None
    is_valid: bool = True
    data_size_bytes: int
    creation_duration_ms: Optional[int] = None


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
    type_distribution: Dict[str, int] = Field(default_factory=dict)
    storage_metrics: CheckpointStorageMetrics
    service_stats: Dict[str, Any] = Field(default_factory=dict)