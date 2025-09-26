"""Event-driven controller wiring the scheduler to module runtime."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional, Set, Tuple

from src.roma_dspy.core.engine.dag import TaskDAG
from src.roma_dspy.core.engine.events import EventType, TaskEvent
from src.roma_dspy.core.engine.runtime import ModuleRuntime
from src.roma_dspy.core.engine.scheduler import EventScheduler
from src.roma_dspy.core.signatures import TaskNode
from src.roma_dspy.types import TaskStatus, FailureContext
from src.roma_dspy.types.checkpoint_types import CheckpointTrigger, RecoveryStrategy
from src.roma_dspy.resilience import create_default_retry_policy
from src.roma_dspy.resilience.checkpoint_manager import CheckpointManager
from src.roma_dspy.types.checkpoint_models import CheckpointConfig

logger = logging.getLogger(__name__)


class EventLoopController:
    """Coordinates event scheduling and module execution."""

    def __init__(
        self,
        dag: TaskDAG,
        runtime: ModuleRuntime,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ) -> None:
        self.dag = dag
        self.runtime = runtime
        self.scheduler = EventScheduler(dag, priority_fn=priority_fn)
        self.scheduler.register_processor(EventType.READY, self._handle_ready)
        self.scheduler.register_processor(EventType.COMPLETED, self._handle_completed)
        self.scheduler.register_processor(EventType.SUBGRAPH_COMPLETE, self._handle_subgraph_complete)
        self.scheduler.register_processor(EventType.FAILED, self._handle_failed)
        self._queued: Set[Tuple[str, str]] = set()  # (dag_id, task_id)

        # Initialize checkpoint manager for recovery - pass explicit config
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(CheckpointConfig())
        self._failure_count = 0
        self._max_recovery_attempts = 3
        self._max_queued_tasks = 1000  # Prevent unbounded queue growth

    async def run(self, max_concurrency: int = 1) -> None:
        """Seed initial tasks and process events until completion."""

        await self.enqueue_ready_tasks()
        await self.scheduler.schedule(max_concurrency=max_concurrency)

    async def enqueue_ready_tasks(self) -> None:
        """Inspect DAG hierarchy and queue tasks whose dependencies cleared."""
        # Prevent unbounded queue growth
        if len(self._queued) >= self._max_queued_tasks:
            logger.warning(f"Queue limit reached ({self._max_queued_tasks}), skipping new task enqueue")
            return

        for task, owning_dag in self.dag.iter_ready_nodes():
            key = (owning_dag.dag_id, task.task_id)
            if key in self._queued:
                continue

            await self.scheduler.emit_event(self._make_ready_event(task, owning_dag))
            self._queued.add(key)

    def _make_ready_event(self, task: TaskNode, dag: TaskDAG) -> TaskEvent:
        return TaskEvent(
            priority=self.scheduler.priority_for(task),
            event_type=EventType.READY,
            task_id=task.task_id,
            dag_id=dag.dag_id,
        )

    def _make_completed_event(self, task: TaskNode, dag: TaskDAG) -> TaskEvent:
        return TaskEvent(
            priority=self.scheduler.priority_for(task),
            event_type=EventType.COMPLETED,
            task_id=task.task_id,
            dag_id=dag.dag_id,
        )

    def _make_subgraph_event(self, task: TaskNode, dag: TaskDAG) -> TaskEvent:
        return TaskEvent(
            priority=self.scheduler.priority_for(task),
            event_type=EventType.SUBGRAPH_COMPLETE,
            task_id=task.task_id,
            dag_id=dag.dag_id,
        )

    async def _handle_ready(self, event: TaskEvent) -> Optional[TaskEvent]:
        if not event.task_id or not event.dag_id:
            return None

        owning_dag = self._resolve_dag(event.dag_id)
        if owning_dag is None:
            logger.warning("Received READY event for unknown dag_id=%s", event.dag_id)
            return None

        try:
            task = owning_dag.get_node(event.task_id)
        except ValueError:
            logger.warning("Task %s not found in dag %s", event.task_id, owning_dag.dag_id)
            return None

        self._queued.discard((owning_dag.dag_id, task.task_id))

        if task.should_force_execute():
            updated = await self.runtime.force_execute_async(task, owning_dag)
            return self._make_completed_event(updated, owning_dag)

        if task.status == TaskStatus.PENDING:
            task = await self.runtime.atomize_async(task, owning_dag)
            task = self.runtime.transition_from_atomizing(task, owning_dag)
            key = (owning_dag.dag_id, task.task_id)
            self._queued.add(key)
            return self._make_ready_event(task, owning_dag)

        if task.status == TaskStatus.PLANNING:
            task = await self.runtime.plan_async(task, owning_dag)
            subgraph = owning_dag.get_subgraph(task.subgraph_id) if task.subgraph_id else None

            if subgraph and subgraph.graph.nodes():
                await self.enqueue_ready_tasks()
                return None

            # No subtasks -> treat as completed subgraph
            return self._make_subgraph_event(task, owning_dag)

        if task.status == TaskStatus.EXECUTING:
            task = await self.runtime.execute_async(task, owning_dag)
            return self._make_completed_event(task, owning_dag)

        if task.status == TaskStatus.AGGREGATING:
            subgraph = owning_dag.get_subgraph(task.subgraph_id) if task.subgraph_id else None
            task = await self.runtime.aggregate_async(task, subgraph, owning_dag)
            return self._make_completed_event(task, owning_dag)

        return None

    async def _handle_completed(self, event: TaskEvent) -> Optional[TaskEvent]:
        if not event.task_id or not event.dag_id:
            return None

        owning_dag = self._resolve_dag(event.dag_id)
        if owning_dag is None:
            return None

        try:
            task = owning_dag.get_node(event.task_id)
        except ValueError:
            return None

        parent_info = await owning_dag.check_subgraph_complete(task.task_id)
        await self.enqueue_ready_tasks()

        if parent_info:
            parent_node, parent_dag = parent_info
            return self._make_subgraph_event(parent_node, parent_dag)

        return None

    async def _handle_subgraph_complete(self, event: TaskEvent) -> Optional[TaskEvent]:
        if not event.task_id or not event.dag_id:
            return None

        owning_dag = self._resolve_dag(event.dag_id)
        if owning_dag is None:
            return None

        try:
            task = owning_dag.get_node(event.task_id)
        except ValueError:
            return None

        if task.status != TaskStatus.PLAN_DONE:
            logger.debug(
                "Ignoring SUBGRAPH_COMPLETE for task %s in state %s",
                task.task_id,
                task.status,
            )
            return None

        subgraph = owning_dag.get_subgraph(task.subgraph_id) if task.subgraph_id else None
        task = await self.runtime.aggregate_async(task, subgraph, owning_dag)
        await self.enqueue_ready_tasks()
        return self._make_completed_event(task, owning_dag)

    async def _handle_failed(self, event: TaskEvent) -> Optional[TaskEvent]:
        if not event.task_id or not event.dag_id:
            return None

        owning_dag = self._resolve_dag(event.dag_id)
        if owning_dag is None:
            return None

        try:
            task = owning_dag.get_node(event.task_id)
        except ValueError:
            logger.warning("Task %s not found in dag %s", event.task_id, owning_dag.dag_id)
            return None

        # Implement retry/backoff strategy
        retry_policy = create_default_retry_policy()

        # Check if retry is possible
        if task.can_retry:
            # Create failure context for retry calculation
            failure_context = FailureContext(
                error_type=type(event.data).__name__ if event.data else "Unknown",
                error_message=str(event.data) if event.data else "Task failed",
                task_type=task.task_type,
                metadata={
                    "task_id": task.task_id,
                    "depth": task.depth,
                    "retry_count": task.retry_count
                }
            )

            # Calculate backoff delay
            delay = retry_policy.calculate_delay(
                task.retry_count,
                task.task_type,
                failure_context
            )

            logger.info(
                "Retrying task %s (attempt %d/%d) after %.2fs delay. Error: %s",
                task.task_id,
                task.retry_count + 1,
                task.max_retries,
                delay,
                event.data
            )

            # Apply delay
            if delay > 0:
                await asyncio.sleep(delay)

            # Update task and transition to READY
            try:
                updated_task = task.increment_retry()
                updated_task = updated_task.transition_to(TaskStatus.READY)
                await owning_dag.update_node(updated_task)

                # Remove from queued set and re-add with new state
                key = (owning_dag.dag_id, task.task_id)
                self._queued.discard(key)
                self._queued.add(key)

                return self._make_ready_event(updated_task, owning_dag)

            except ValueError as e:
                logger.error(
                    "Failed to increment retry for task %s: %s",
                    task.task_id,
                    str(e)
                )
                # Fall through to permanent failure handling

        # Task failed permanently - either can't retry or max retries exceeded
        logger.error(
            "Task %s in dag %s permanently failed after %d retries: %s",
            task.task_id,
            owning_dag.dag_id,
            task.retry_count,
            event.data,
        )
        await owning_dag.mark_failed(event.task_id, error=event.data)
        return None

    def _resolve_dag(self, dag_id: str) -> Optional[TaskDAG]:
        if not dag_id:
            return None
        return self.dag.find_dag(dag_id)
