"""Event-driven controller wiring the scheduler to module runtime."""

from __future__ import annotations

import logging
from typing import Callable, Optional, Set, Tuple

from .dag import TaskDAG
from .events import EventType, TaskEvent
from .runtime import ModuleRuntime
from .scheduler import EventScheduler
from ..signatures import TaskNode
from ...types import TaskStatus

logger = logging.getLogger(__name__)


class EventLoopController:
    """Coordinates event scheduling and module execution."""

    def __init__(
        self,
        dag: TaskDAG,
        runtime: ModuleRuntime,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
    ) -> None:
        self.dag = dag
        self.runtime = runtime
        self.scheduler = EventScheduler(dag, priority_fn=priority_fn)
        self.scheduler.register_processor(EventType.READY, self._handle_ready)
        self.scheduler.register_processor(EventType.COMPLETED, self._handle_completed)
        self.scheduler.register_processor(EventType.SUBGRAPH_COMPLETE, self._handle_subgraph_complete)
        self.scheduler.register_processor(EventType.FAILED, self._handle_failed)
        self._queued: Set[Tuple[str, str]] = set()  # (dag_id, task_id)

    async def run(self, max_concurrency: int = 1) -> None:
        """Seed initial tasks and process events until completion."""

        await self.enqueue_ready_tasks()
        await self.scheduler.schedule(max_concurrency=max_concurrency)

    async def enqueue_ready_tasks(self) -> None:
        """Inspect DAG hierarchy and queue tasks whose dependencies cleared."""

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

        # TODO: implement retry/backoff strategy
        logger.error(
            "Task %s in dag %s failed: %s",
            event.task_id,
            owning_dag.dag_id,
            event.data,
        )
        await owning_dag.mark_failed(event.task_id, error=event.data)
        return None

    def _resolve_dag(self, dag_id: str) -> Optional[TaskDAG]:
        if not dag_id:
            return None
        return self.dag.find_dag(dag_id)
