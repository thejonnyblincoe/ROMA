"""Checkpoint management for partial recovery and state preservation."""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.roma_dspy.core.engine.dag import TaskDAG
from src.roma_dspy.core.signatures import TaskNode
from src.roma_dspy.types.checkpoint_types import (
    CheckpointState,
    RecoveryStrategy,
    CheckpointTrigger,
    RecoveryError,
    CheckpointCorruptedError,
    CheckpointExpiredError,
    CheckpointNotFoundError
)
from src.roma_dspy.types.checkpoint_models import (
    CheckpointData,
    CheckpointConfig,
    DAGSnapshot,
    TaskSnapshot,
    RecoveryPlan
)

# Remove settings import to avoid circular dependency
# Configuration will be passed explicitly via constructor

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint creation, storage, and recovery operations.

    Supports async context manager for proper resource cleanup:

        async with CheckpointManager(config) as manager:
            checkpoint_id = await manager.create_checkpoint(...)
    """

    def __init__(self, config: Optional[CheckpointConfig] = None) -> None:
        # Use explicit config or create default - no settings dependency
        self.config = config or CheckpointConfig()
        self.storage_path = self.config.storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cleanup_lock = asyncio.Lock()

    async def create_checkpoint(
        self,
        checkpoint_id: Optional[str],
        dag: TaskDAG,
        trigger: CheckpointTrigger,
        current_depth: int = 0,
        max_depth: int = 5,
        solver_config: Optional[Dict[str, Any]] = None,
        failed_task_ids: Optional[Set[str]] = None
    ) -> str:
        """Create a new checkpoint with current system state."""
        if not self.config.enabled:
            logger.debug("Checkpoint system disabled, skipping checkpoint creation")
            return ""

        checkpoint_id = checkpoint_id or self._generate_checkpoint_id()
        logger.info(f"Creating checkpoint {checkpoint_id} triggered by {trigger.value}")

        try:
            # Create DAG snapshot
            dag_snapshot = await self._serialize_dag(dag)

            # Collect preserved results for partial recovery
            preserved_results = await self._collect_preserved_results(dag)

            # Create checkpoint data
            checkpoint_data = CheckpointData(
                checkpoint_id=checkpoint_id,
                trigger=trigger,
                root_dag=dag_snapshot,
                current_depth=current_depth,
                max_depth=max_depth,
                failed_task_ids=failed_task_ids or set(),
                preserved_results=preserved_results,
                solver_config=solver_config or {}
            )

            # Save to storage
            await self._save_checkpoint(checkpoint_data)

            # Cleanup old checkpoints
            await self._cleanup_expired_checkpoints()

            logger.info(f"Checkpoint {checkpoint_id} created successfully")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint {checkpoint_id}: {e}")
            raise RecoveryError(f"Checkpoint creation failed: {e}") from e

    async def load_checkpoint(self, checkpoint_id: str) -> CheckpointData:
        """Load checkpoint data from storage."""
        if not self.config.enabled:
            raise RecoveryError("Checkpoint system is disabled")

        checkpoint_path = self._get_checkpoint_path(checkpoint_id)
        if not checkpoint_path.exists():
            raise CheckpointNotFoundError(f"Checkpoint {checkpoint_id} not found")

        try:
            logger.info(f"Loading checkpoint {checkpoint_id}")
            checkpoint_data = await self._load_checkpoint_data(checkpoint_path)

            # Verify checkpoint integrity
            if self.config.verify_integrity:
                await self._verify_checkpoint_integrity(checkpoint_data)

            # Check expiration
            if await self._is_checkpoint_expired(checkpoint_data):
                checkpoint_data.state = CheckpointState.EXPIRED
                raise CheckpointExpiredError(f"Checkpoint {checkpoint_id} has expired")

            checkpoint_data.state = CheckpointState.VALID
            logger.info(f"Checkpoint {checkpoint_id} loaded successfully")
            return checkpoint_data

        except (CheckpointExpiredError, CheckpointNotFoundError):
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            raise CheckpointCorruptedError(f"Checkpoint corruption: {e}") from e

    async def create_recovery_plan(
        self,
        checkpoint_data: CheckpointData,
        failed_task_ids: Optional[Set[str]] = None,
        strategy: Optional[RecoveryStrategy] = None
    ) -> RecoveryPlan:
        """Create a recovery plan based on checkpoint data and current failure state."""
        strategy = strategy or self.config.default_recovery_strategy
        failed_task_ids = failed_task_ids or checkpoint_data.failed_task_ids

        logger.info(f"Creating recovery plan with strategy {strategy.value} for {len(failed_task_ids)} failed tasks")

        if strategy == RecoveryStrategy.FULL:
            # Full recovery: retry everything
            tasks_to_retry = list(checkpoint_data.root_dag.tasks.keys())
            tasks_to_preserve = []

        elif strategy == RecoveryStrategy.PARTIAL:
            # Partial recovery: only retry failed tasks and their dependents
            tasks_to_retry = await self._calculate_affected_tasks(
                checkpoint_data.root_dag, failed_task_ids
            )
            tasks_to_preserve = [
                task_id for task_id in checkpoint_data.root_dag.tasks.keys()
                if task_id not in tasks_to_retry
            ]

        else:  # SELECTIVE
            # Selective recovery: user-defined scope (default to partial)
            tasks_to_retry = list(failed_task_ids)
            tasks_to_preserve = [
                task_id for task_id in checkpoint_data.root_dag.tasks.keys()
                if task_id not in failed_task_ids
            ]

        recovery_plan = RecoveryPlan(
            checkpoint_id=checkpoint_data.checkpoint_id,
            strategy=strategy,
            tasks_to_retry=tasks_to_retry,
            tasks_to_preserve=tasks_to_preserve,
            preserve_partial_results=self.config.preserve_partial_results
        )

        logger.info(
            f"Recovery plan created: {len(tasks_to_retry)} tasks to retry, "
            f"{len(tasks_to_preserve)} tasks to preserve"
        )
        return recovery_plan

    async def apply_recovery_plan(
        self,
        recovery_plan: RecoveryPlan,
        target_dag: TaskDAG
    ) -> TaskDAG:
        """Apply recovery plan to restore DAG state from checkpoint."""
        logger.info(f"Applying recovery plan for checkpoint {recovery_plan.checkpoint_id}")

        try:
            # Load checkpoint data
            checkpoint_data = await self.load_checkpoint(recovery_plan.checkpoint_id)

            # Restore preserved task results
            if recovery_plan.restore_dag_state:
                await self._restore_dag_state(
                    target_dag,
                    checkpoint_data.root_dag,
                    recovery_plan.tasks_to_preserve
                )

            # Reset retry counters if requested
            if recovery_plan.reset_retry_counts:
                await self._reset_retry_counters(target_dag, recovery_plan.tasks_to_retry)

            # Mark tasks for retry
            await self._prepare_tasks_for_retry(target_dag, recovery_plan.tasks_to_retry)

            logger.info(f"Recovery plan applied successfully")
            return target_dag

        except Exception as e:
            logger.error(f"Failed to apply recovery plan: {e}")
            raise RecoveryError(f"Recovery plan application failed: {e}") from e

    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        checkpoints = []
        for checkpoint_file in self.storage_path.glob("*.json*"):
            try:
                checkpoint_data = await self._load_checkpoint_data(checkpoint_file)
                checkpoints.append({
                    "checkpoint_id": checkpoint_data.checkpoint_id,
                    "created_at": checkpoint_data.created_at,
                    "trigger": checkpoint_data.trigger.value,
                    "state": checkpoint_data.state.value,
                    "task_count": len(checkpoint_data.root_dag.tasks),
                    "failed_tasks": len(checkpoint_data.failed_task_ids)
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")
                continue

        return sorted(checkpoints, key=lambda x: x["created_at"], reverse=True)

    async def cleanup_checkpoints(self, keep_latest: int = 5) -> int:
        """Cleanup old checkpoints, keeping only the most recent ones."""
        async with self._cleanup_lock:
            checkpoints = await self.list_checkpoints()
            if len(checkpoints) <= keep_latest:
                return 0

            removed_count = 0
            for checkpoint in checkpoints[keep_latest:]:
                try:
                    checkpoint_path = self._get_checkpoint_path(checkpoint["checkpoint_id"])
                    if checkpoint_path.exists():
                        checkpoint_path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old checkpoint {checkpoint['checkpoint_id']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint['checkpoint_id']}: {e}")

            logger.info(f"Cleaned up {removed_count} old checkpoints")
            return removed_count

    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Deleted checkpoint {checkpoint_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False

    async def get_checkpoint_size(self, checkpoint_id: str) -> int:
        """Get size of a checkpoint file in bytes."""
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint_id)
            if checkpoint_path.exists():
                return checkpoint_path.stat().st_size
            return 0
        except Exception as e:
            logger.warning(f"Failed to get size for checkpoint {checkpoint_id}: {e}")
            return 0

    async def validate_checkpoint(self, checkpoint_id: str) -> bool:
        """Validate that a checkpoint is not corrupted."""
        try:
            checkpoint_data = await self.load_checkpoint(checkpoint_id)
            # If we can load it successfully, it's valid
            return True
        except Exception as e:
            logger.warning(f"Checkpoint {checkpoint_id} failed validation: {e}")
            return False

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoint storage."""
        try:
            checkpoints = await self.list_checkpoints()
            total_size = 0
            valid_count = 0

            for cp in checkpoints:
                size = await self.get_checkpoint_size(cp["checkpoint_id"])
                total_size += size
                if cp["state"] == "valid":
                    valid_count += 1

            return {
                "total_checkpoints": len(checkpoints),
                "valid_checkpoints": valid_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "storage_path": str(self.storage_path),
                "compression_enabled": self.config.compress_checkpoints
            }
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}

    async def shutdown(self) -> None:
        """Cleanup resources and perform final checkpoint maintenance."""
        try:
            # Final cleanup of expired checkpoints
            await self._cleanup_expired_checkpoints()
            logger.info("CheckpointManager shutdown complete")
        except Exception as e:
            logger.error(f"Error during CheckpointManager shutdown: {e}")

    # Context manager support
    async def __aenter__(self) -> 'CheckpointManager':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        await self.shutdown()

    @classmethod
    def create_default(cls, storage_path: Optional[Path] = None) -> 'CheckpointManager':
        """Create CheckpointManager with sensible defaults."""
        config = CheckpointConfig()
        if storage_path:
            config.storage_path = storage_path
        return cls(config)

    @classmethod
    def create_in_memory(cls) -> 'CheckpointManager':
        """Create CheckpointManager for testing (uses temp directory)."""
        import tempfile
        temp_dir = Path(tempfile.mkdtemp(prefix="roma_test_checkpoints_"))
        config = CheckpointConfig(
            storage_path=temp_dir,
            max_checkpoints=5,  # Smaller limit for testing
            max_age_hours=1.0   # Shorter retention for testing
        )
        return cls(config)

    # Private methods

    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint identifier with microseconds to prevent collisions."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")  # Include microseconds
        unique_id = str(uuid.uuid4())[:8]
        return f"checkpoint_{timestamp}_{unique_id}"

    def _get_checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get file path for checkpoint."""
        extension = ".json.gz" if self.config.compress_checkpoints else ".json"
        return self.storage_path / f"{checkpoint_id}{extension}"

    async def _serialize_dag(self, dag: TaskDAG) -> DAGSnapshot:
        """Serialize DAG state to checkpoint format."""
        tasks = {}
        completed_tasks = set()
        failed_tasks = set()

        for task_id, task in dag.get_all_tasks().items():
            task_snapshot = TaskSnapshot(
                task_id=task.task_id,
                status=task.status.value,
                task_type=task.task_type.value,
                depth=task.depth,
                retry_count=task.retry_count,
                max_retries=task.max_retries,
                result=task.result,
                error=str(task.error) if task.error else None,
                subgraph_id=task.subgraph_id,
                dependencies=[dep.task_id for dep in task.dependencies],
                metadata=task.metadata or {}
            )
            tasks[task_id] = task_snapshot

            if task.status.value == "completed":
                completed_tasks.add(task_id)
            elif task.status.value == "failed":
                failed_tasks.add(task_id)

        # Serialize subgraphs recursively
        subgraphs = {}
        for subgraph_id, subgraph in dag.subgraphs.items():
            subgraphs[subgraph_id] = await self._serialize_dag(subgraph)

        return DAGSnapshot(
            dag_id=dag.dag_id,
            tasks=tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            dependencies={
                task_id: [dep.task_id for dep in task.dependencies]
                for task_id, task in dag.get_all_tasks().items()
            },
            subgraphs=subgraphs
        )

    async def _collect_preserved_results(self, dag: TaskDAG) -> Dict[str, Any]:
        """Collect results from completed tasks for preservation."""
        preserved_results = {}
        for task_id, task in dag.get_all_tasks().items():
            if task.status.value == "completed" and task.result is not None:
                try:
                    # Only preserve serializable results
                    json.dumps(task.result)  # Test serialization
                    preserved_results[task_id] = task.result
                except (TypeError, ValueError):
                    logger.debug(f"Result for task {task_id} is not serializable, skipping preservation")
        return preserved_results

    async def _save_checkpoint(self, checkpoint_data: CheckpointData) -> None:
        """Save checkpoint data to storage with atomic write to prevent corruption."""
        checkpoint_path = self._get_checkpoint_path(checkpoint_data.checkpoint_id)
        temp_path = checkpoint_path.with_suffix(".tmp")
        data = checkpoint_data.model_dump(mode="json")

        try:
            # Write to temporary file first (atomic operation)
            if self.config.compress_checkpoints:
                with gzip.open(temp_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

            # Atomic move to final location
            temp_path.rename(checkpoint_path)
        except Exception:
            # Cleanup temp file on failure
            if temp_path.exists():
                temp_path.unlink()
            raise

    async def _load_checkpoint_data(self, checkpoint_path: Path) -> CheckpointData:
        """Load checkpoint data from file."""
        if checkpoint_path.suffix == ".gz":
            with gzip.open(checkpoint_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        return CheckpointData.model_validate(data)

    async def _verify_checkpoint_integrity(self, checkpoint_data: CheckpointData) -> None:
        """Verify checkpoint data integrity."""
        required_fields = ["checkpoint_id", "created_at", "root_dag"]
        for field in required_fields:
            if not hasattr(checkpoint_data, field) or getattr(checkpoint_data, field) is None:
                raise CheckpointCorruptedError(f"Missing required field: {field}")

        # Verify DAG structure
        if not checkpoint_data.root_dag.tasks:
            raise CheckpointCorruptedError("Checkpoint contains empty DAG")

    async def _is_checkpoint_expired(self, checkpoint_data: CheckpointData) -> bool:
        """Check if checkpoint has expired."""
        max_age = timedelta(hours=self.config.max_age_hours)
        age = datetime.now() - checkpoint_data.created_at
        return age > max_age

    async def _calculate_affected_tasks(
        self,
        dag_snapshot: DAGSnapshot,
        failed_task_ids: Set[str]
    ) -> List[str]:
        """Calculate which tasks need to be retried based on failures."""
        affected_tasks = set(failed_task_ids)

        # Add tasks that depend on failed tasks
        for task_id, task_snapshot in dag_snapshot.tasks.items():
            for dep_id in task_snapshot.dependencies:
                if dep_id in affected_tasks:
                    affected_tasks.add(task_id)

        return list(affected_tasks)

    async def _restore_dag_state(
        self,
        target_dag: TaskDAG,
        checkpoint_dag: DAGSnapshot,
        tasks_to_preserve: List[str]
    ) -> None:
        """Restore DAG state from checkpoint for preserved tasks."""
        for task_id in tasks_to_preserve:
            if task_id in checkpoint_dag.tasks:
                task_snapshot = checkpoint_dag.tasks[task_id]
                if task_snapshot.result is not None:
                    # Restore task result and status
                    try:
                        await target_dag.restore_task_result(
                            task_id,
                            task_snapshot.result,
                            task_snapshot.status
                        )
                        logger.debug(f"Restored state for task {task_id}")
                    except ValueError:
                        logger.warning(f"Task {task_id} not found in target DAG for restoration")

    async def _reset_retry_counters(self, dag: TaskDAG, task_ids: List[str]) -> None:
        """Reset retry counters for specified tasks."""
        for task_id in task_ids:
            try:
                await dag.reset_task_retry_counter(task_id)
                logger.debug(f"Reset retry counter for task {task_id}")
            except ValueError:
                logger.warning(f"Task {task_id} not found for retry counter reset")

    async def _prepare_tasks_for_retry(self, dag: TaskDAG, task_ids: List[str]) -> None:
        """Prepare tasks for retry by resetting their status."""
        for task_id in task_ids:
            try:
                await dag.prepare_task_for_retry(task_id)
                logger.debug(f"Prepared task {task_id} for retry")
            except ValueError:
                logger.warning(f"Task {task_id} not found for retry preparation")

    async def _cleanup_expired_checkpoints(self) -> None:
        """Remove expired checkpoints."""
        cutoff_time = datetime.now() - timedelta(hours=self.config.max_age_hours)
        removed_count = 0

        for checkpoint_file in self.storage_path.glob("checkpoint_*.json*"):
            try:
                checkpoint_data = await self._load_checkpoint_data(checkpoint_file)
                if checkpoint_data.created_at < cutoff_time:
                    checkpoint_file.unlink()
                    removed_count += 1
                    logger.debug(f"Removed expired checkpoint {checkpoint_data.checkpoint_id}")
            except Exception as e:
                logger.warning(f"Failed to process checkpoint file {checkpoint_file}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired checkpoints")