"""Async event scheduler orchestrating TaskDAG execution."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Awaitable, Callable, Dict, Optional

from src.roma_dspy.engine.dag import TaskDAG
from src.roma_dspy.engine.events import EventType, TaskEvent
from src.roma_dspy.signatures.base_models.task_node import TaskNode


EventHandler = Callable[[TaskEvent], Awaitable[Optional[TaskEvent]]]


class EventScheduler:
    """Thin orchestration layer built on top of TaskDAG and events."""

    def __init__(
        self,
        dag: TaskDAG,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
    ) -> None:
        self.dag = dag
        self._priority_fn = priority_fn or (lambda node: node.depth)
        self.event_queue: asyncio.PriorityQueue[TaskEvent] = asyncio.PriorityQueue()
        self.processors: Dict[EventType, EventHandler] = {}
        self._metrics = defaultdict(int)
        self._stop_requested = False
        self._completion_event = asyncio.Event()

    def register_processor(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a coroutine handler for a particular event type."""

        self.processors[event_type] = handler

    async def emit_event(self, event: TaskEvent) -> None:
        """Push an event onto the internal queue."""

        if self._stop_requested and event.event_type != EventType.STOP:
            return
        await self.event_queue.put(event)
        self._metrics[f"queued::{event.event_type.name.lower()}"] += 1

    async def schedule(self, max_concurrency: int = 1) -> None:
        """Consume events until the DAG marks itself complete."""

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")

        self._completion_event.clear()
        workers = [asyncio.create_task(self._worker()) for _ in range(max_concurrency)]

        try:
            await self._completion_event.wait()
        finally:
            self.force_stop()
            stop_event = TaskEvent(priority=0, event_type=EventType.STOP)
            for _ in workers:
                await self.event_queue.put(stop_event)
            await asyncio.gather(*workers, return_exceptions=False)

    async def _process_event(self, event: TaskEvent) -> None:
        """Dispatch an event to its registered processor."""

        handler = self.processors.get(event.event_type)
        if handler is None:
            return

        follow_up = await handler(event)
        self._metrics[f"handled::{event.event_type.name.lower()}"] += 1

        if follow_up is not None:
            await self.emit_event(follow_up)

    async def _worker(self) -> None:
        """Continuously consume events until stop requested."""

        while True:
            event = await self.event_queue.get()
            if event.event_type == EventType.STOP:
                break

            await self._process_event(event)

            if not self._stop_requested and self.dag.is_dag_complete() and self.event_queue.empty():
                self._completion_event.set()

    def force_stop(self) -> None:
        """Signal the scheduler to stop processing new events."""

        if not self._stop_requested:
            self._stop_requested = True
            self._completion_event.set()

    def priority_for(self, node: TaskNode) -> int:
        """Evaluate current priority for a task node."""

        return self._priority_fn(node)

    @property
    def metrics(self) -> Dict[str, int]:
        """Expose basic counters for observability."""

        return dict(self._metrics)
