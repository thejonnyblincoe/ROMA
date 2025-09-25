"""
Execution State Value Object.

Tracks the ongoing state of a task execution including timing,
node counts, and result caching with thread safety.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.result_envelope import AnyResultEnvelope


class ExecutionState(BaseModel):
    """
    Thread-safe execution state tracker.

    Tracks the state of an ongoing execution with async locks
    for concurrent access safety during parallel node execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # Core execution identity
    execution_id: str = Field(..., description="Unique execution identifier")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="When execution started"
    )

    # Execution context (immutable after creation)
    overall_objective: str = Field(..., description="High-level objective for this execution")
    root_task: TaskNode = Field(..., description="Root task node")

    # Mutable state (protected by lock)
    iterations: int = Field(default=0, ge=0, description="Number of orchestration iterations")
    total_nodes_processed: int = Field(default=0, ge=0, description="Total nodes processed so far")
    completed_node_ids: set[str] = Field(default_factory=set, description="IDs of completed nodes")
    failed_node_ids: set[str] = Field(default_factory=set, description="IDs of failed nodes")
    result_cache: dict[str, AnyResultEnvelope] = Field(
        default_factory=dict, description="Cache of node execution results"
    )

    # Private lock for thread safety
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def __init__(self, **data: Any) -> None:
        """Initialize ExecutionState with lock."""
        super().__init__(**data)
        self._lock = asyncio.Lock()

    async def increment_iteration(self) -> None:
        """Thread-safe iteration increment."""
        async with self._lock:
            self.iterations += 1

    async def add_processed_nodes(self, count: int) -> None:
        """Thread-safe increment of processed nodes count."""
        async with self._lock:
            self.total_nodes_processed += count

    async def mark_node_completed(self, node_id: str) -> None:
        """Thread-safe node completion tracking."""
        async with self._lock:
            self.completed_node_ids.add(node_id)
            # Remove from failed if it was there (recovery case)
            self.failed_node_ids.discard(node_id)

    async def mark_node_failed(self, node_id: str) -> None:
        """Thread-safe node failure tracking."""
        async with self._lock:
            self.failed_node_ids.add(node_id)
            # Remove from completed if it was there (shouldn't happen, but defensive)
            self.completed_node_ids.discard(node_id)

    async def cache_result(self, node_id: str, result: AnyResultEnvelope) -> None:
        """Thread-safe result caching."""
        async with self._lock:
            self.result_cache[node_id] = result

    async def get_cached_result(self, node_id: str) -> AnyResultEnvelope | None:
        """Thread-safe result retrieval."""
        async with self._lock:
            return self.result_cache.get(node_id)

    async def clear_cache(self) -> None:
        """Thread-safe cache clearing."""
        async with self._lock:
            self.result_cache.clear()

    # Read operations (thread-safe without locks due to Python GIL for primitives)
    def get_execution_time(self) -> float:
        """Get current execution time in seconds."""
        return (datetime.now(UTC) - self.start_time).total_seconds()

    def get_completion_stats(self) -> dict[str, int]:
        """Get completion statistics snapshot."""
        # These reads are atomic in Python due to GIL
        return {
            "completed": len(self.completed_node_ids),
            "failed": len(self.failed_node_ids),
            "total_processed": self.total_nodes_processed,
            "cached_results": len(self.result_cache),
            "iterations": self.iterations,
        }

    def is_node_completed(self, node_id: str) -> bool:
        """Check if a node is completed (thread-safe read)."""
        return node_id in self.completed_node_ids

    def is_node_failed(self, node_id: str) -> bool:
        """Check if a node has failed (thread-safe read)."""
        return node_id in self.failed_node_ids

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_completion_stats()
        return (
            f"ExecutionState({self.execution_id}: "
            f"{stats['completed']} completed, {stats['failed']} failed, "
            f"{stats['iterations']} iterations, {self.get_execution_time():.1f}s)"
        )
