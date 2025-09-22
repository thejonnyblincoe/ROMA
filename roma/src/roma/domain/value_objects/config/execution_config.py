"""
Execution Configuration Value Object.

Defines execution and orchestration configuration as a domain value object.
Controls concurrency, limits, timeouts, and memory management for the execution system.
"""

from pydantic.dataclasses import dataclass
from pydantic import field_validator
from typing import Dict, Any, Optional
from roma.domain.value_objects.task_type import TaskType


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution and orchestration configuration."""

    # Concurrency controls
    max_concurrent_tasks: int = 10
    max_parallel_aggregations: int = 3

    # Loop and recursion limits (prevent infinite loops/explosions)
    max_iterations: int = 1000
    max_depth: int = 5
    max_subtasks_per_node: int = 20
    max_tasks_per_level: int = 50

    # Memory management
    result_cache_max_size: int = 10000
    graph_node_limit: int = 5000

    # Timeouts (in seconds)
    task_timeout: int = 300  # 5 minutes per task
    total_timeout: int = 3600  # 1 hour total
    aggregation_timeout: int = 60  # 1 minute for aggregation

    # Error handling
    retry_on_failure: bool = True
    max_retries: int = 3
    exponential_backoff: bool = True
    fallback_to_simpler: bool = True

    # Optimization flags
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour cache TTL
    checkpoint_enabled: bool = False
    checkpoint_interval: int = 100  # tasks

    # HITL settings
    hitl_enabled: bool = False
    hitl_approval_required: bool = False

    # Replanning settings
    replanning_enabled: bool = True
    default_failure_threshold: float = 0.3  # 30% failures trigger replan
    task_type_thresholds: Optional[Dict[TaskType, float]] = None  # Task-specific thresholds
    enable_partial_aggregation: bool = True
    deadlock_detection_enabled: bool = True
    deadlock_check_interval: int = 300  # 5 minutes between deadlock checks
    hitl_replanning_enabled: bool = False  # HITL layer for replanning

    @field_validator("max_concurrent_tasks")
    @classmethod
    def validate_max_concurrent_tasks(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"max_concurrent_tasks must be 1-100, got: {v}")
        return v

    @field_validator("max_parallel_aggregations")
    @classmethod
    def validate_max_parallel_aggregations(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError(f"max_parallel_aggregations must be 1-20, got: {v}")
        return v

    @field_validator("max_iterations")
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError(f"max_iterations must be 1-10000, got: {v}")
        return v

    @field_validator("max_depth")
    @classmethod
    def validate_max_depth(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError(f"max_depth must be 1-20, got: {v}")
        return v

    @field_validator("max_subtasks_per_node")
    @classmethod
    def validate_max_subtasks_per_node(cls, v: int) -> int:
        if v < 1 or v > 100:
            raise ValueError(f"max_subtasks_per_node must be 1-100, got: {v}")
        return v

    @field_validator("max_tasks_per_level")
    @classmethod
    def validate_max_tasks_per_level(cls, v: int) -> int:
        if v < 1 or v > 1000:
            raise ValueError(f"max_tasks_per_level must be 1-1000, got: {v}")
        return v

    @field_validator("result_cache_max_size")
    @classmethod
    def validate_result_cache_max_size(cls, v: int) -> int:
        if v < 100 or v > 100000:
            raise ValueError(f"result_cache_max_size must be 100-100000, got: {v}")
        return v

    @field_validator("graph_node_limit")
    @classmethod
    def validate_graph_node_limit(cls, v: int) -> int:
        if v < 100 or v > 100000:
            raise ValueError(f"graph_node_limit must be 100-100000, got: {v}")
        return v

    @field_validator("task_timeout", "total_timeout", "aggregation_timeout")
    @classmethod
    def validate_timeouts(cls, v: int) -> int:
        if v < 1 or v > 86400:  # 1 second to 1 day
            raise ValueError(f"timeout must be 1-86400 seconds, got: {v}")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        if v < 0 or v > 10:
            raise ValueError(f"max_retries must be 0-10, got: {v}")
        return v

    @field_validator("cache_ttl")
    @classmethod
    def validate_cache_ttl(cls, v: int) -> int:
        if v < 60 or v > 86400:  # 1 minute to 1 day
            raise ValueError(f"cache_ttl must be 60-86400 seconds, got: {v}")
        return v

    @field_validator("checkpoint_interval")
    @classmethod
    def validate_checkpoint_interval(cls, v: int) -> int:
        if v < 1 or v > 10000:
            raise ValueError(f"checkpoint_interval must be 1-10000, got: {v}")
        return v

    @field_validator("default_failure_threshold")
    @classmethod
    def validate_default_failure_threshold(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"default_failure_threshold must be 0.0-1.0, got: {v}")
        return v

    @field_validator("deadlock_check_interval")
    @classmethod
    def validate_deadlock_check_interval(cls, v: int) -> int:
        if v < 10 or v > 3600:  # 10 seconds to 1 hour
            raise ValueError(f"deadlock_check_interval must be 10-3600 seconds, got: {v}")
        return v

    def get_failure_threshold(self, task_type: TaskType) -> float:
        """Get failure threshold for specific task type."""
        if self.task_type_thresholds and task_type in self.task_type_thresholds:
            return self.task_type_thresholds[task_type]
        return self.default_failure_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_parallel_aggregations": self.max_parallel_aggregations,
            "max_iterations": self.max_iterations,
            "max_depth": self.max_depth,
            "max_subtasks_per_node": self.max_subtasks_per_node,
            "max_tasks_per_level": self.max_tasks_per_level,
            "result_cache_max_size": self.result_cache_max_size,
            "graph_node_limit": self.graph_node_limit,
            "task_timeout": self.task_timeout,
            "total_timeout": self.total_timeout,
            "aggregation_timeout": self.aggregation_timeout,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
            "exponential_backoff": self.exponential_backoff,
            "fallback_to_simpler": self.fallback_to_simpler,
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "checkpoint_enabled": self.checkpoint_enabled,
            "checkpoint_interval": self.checkpoint_interval,
            "hitl_enabled": self.hitl_enabled,
            "hitl_approval_required": self.hitl_approval_required,
            "replanning_enabled": self.replanning_enabled,
            "default_failure_threshold": self.default_failure_threshold,
            "task_type_thresholds": self.task_type_thresholds,
            "enable_partial_aggregation": self.enable_partial_aggregation,
            "deadlock_detection_enabled": self.deadlock_detection_enabled,
            "deadlock_check_interval": self.deadlock_check_interval,
            "hitl_replanning_enabled": self.hitl_replanning_enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionConfig":
        """Create from dictionary."""
        return cls(
            max_concurrent_tasks=data.get("max_concurrent_tasks", 10),
            max_parallel_aggregations=data.get("max_parallel_aggregations", 3),
            max_iterations=data.get("max_iterations", 1000),
            max_depth=data.get("max_depth", 5),
            max_subtasks_per_node=data.get("max_subtasks_per_node", 20),
            max_tasks_per_level=data.get("max_tasks_per_level", 50),
            result_cache_max_size=data.get("result_cache_max_size", 10000),
            graph_node_limit=data.get("graph_node_limit", 5000),
            task_timeout=data.get("task_timeout", 300),
            total_timeout=data.get("total_timeout", 3600),
            aggregation_timeout=data.get("aggregation_timeout", 60),
            retry_on_failure=data.get("retry_on_failure", True),
            max_retries=data.get("max_retries", 3),
            exponential_backoff=data.get("exponential_backoff", True),
            fallback_to_simpler=data.get("fallback_to_simpler", True),
            enable_caching=data.get("enable_caching", True),
            cache_ttl=data.get("cache_ttl", 3600),
            checkpoint_enabled=data.get("checkpoint_enabled", False),
            checkpoint_interval=data.get("checkpoint_interval", 100),
            hitl_enabled=data.get("hitl_enabled", False),
            hitl_approval_required=data.get("hitl_approval_required", False),
            replanning_enabled=data.get("replanning_enabled", True),
            default_failure_threshold=data.get("default_failure_threshold", 0.3),
            task_type_thresholds=data.get("task_type_thresholds", None),
            enable_partial_aggregation=data.get("enable_partial_aggregation", True),
            deadlock_detection_enabled=data.get("deadlock_detection_enabled", True),
            deadlock_check_interval=data.get("deadlock_check_interval", 300),
            hitl_replanning_enabled=data.get("hitl_replanning_enabled", False),
        )