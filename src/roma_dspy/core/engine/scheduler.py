"""Async event scheduler orchestrating TaskDAG execution."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Dict, Optional

from .dag import TaskDAG
from .events import EventType, TaskEvent
from ..signatures.base_models.task_node import TaskNode

logger = logging.getLogger(__name__)


EventHandler = Callable[[TaskEvent], Awaitable[Optional[TaskEvent]]]


class EventScheduler:
    """Thin orchestration layer built on top of TaskDAG and events."""

    def __init__(
        self,
        dag: TaskDAG,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
        max_queue_size: int = 1000,
    ) -> None:
        self.dag = dag
        self._priority_fn = priority_fn or (lambda node: node.depth)
        self.max_queue_size = max_queue_size
        self.event_queue: asyncio.PriorityQueue[TaskEvent] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.processors: Dict[EventType, EventHandler] = {}
        self._metrics = defaultdict(int)
        self._stop_requested = False
        self._completion_event = asyncio.Event()

        # Queue overflow tracking
        self._overflow_count = 0
        self._last_overflow_time = None

    def register_processor(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a coroutine handler for a particular event type."""

        self.processors[event_type] = handler

    async def emit_event(self, event: TaskEvent) -> None:
        """Push an event onto the internal queue with overflow protection."""

        if self._stop_requested and event.event_type != EventType.STOP:
            return

        try:
            # Try to add event without blocking
            self.event_queue.put_nowait(event)
            self._metrics[f"queued::{event.event_type.name.lower()}"] += 1
        except asyncio.QueueFull:
            # Handle queue overflow
            await self._handle_queue_overflow(event)

    async def _handle_queue_overflow(self, event: TaskEvent) -> None:
        """Handle queue overflow with priority-based dropping strategy."""
        import time

        self._overflow_count += 1
        self._last_overflow_time = time.time()
        self._metrics["queue_overflows"] += 1

        # For critical events (STOP), force them in by dropping lower priority events
        if event.event_type == EventType.STOP:
            logger.warning("Queue overflow: forcing STOP event by dropping oldest event")
            try:
                # Remove oldest event to make space
                dropped_event = self.event_queue.get_nowait()
                self._metrics[f"dropped::{dropped_event.event_type.name.lower()}"] += 1
                # Add the STOP event
                self.event_queue.put_nowait(event)
                self._metrics[f"queued::{event.event_type.name.lower()}"] += 1
            except asyncio.QueueEmpty:
                # Queue became empty during overflow handling
                self.event_queue.put_nowait(event)
                self._metrics[f"queued::{event.event_type.name.lower()}"] += 1
        else:
            # Drop non-critical events
            logger.error(
                f"Event queue full (size: {self.max_queue_size}), "
                f"dropping {event.event_type.name} event. "
                f"Total overflows: {self._overflow_count}"
            )
            self._metrics[f"dropped::{event.event_type.name.lower()}"] += 1

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

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status for monitoring."""
        import time

        return {
            "current_size": self.event_queue.qsize(),
            "max_size": self.max_queue_size,
            "is_full": self.event_queue.full(),
            "overflow_count": self._overflow_count,
            "last_overflow_time": self._last_overflow_time,
            "time_since_last_overflow": (
                time.time() - self._last_overflow_time
                if self._last_overflow_time else None
            )
        }

    @property
    def metrics(self) -> Dict[str, int]:
        """Expose basic counters for observability."""

        # Include queue status in metrics
        queue_status = self.get_queue_status()
        metrics = dict(self._metrics)
        metrics.update({
            "queue_current_size": queue_status["current_size"],
            "queue_max_size": queue_status["max_size"],
            "queue_overflow_count": queue_status["overflow_count"]
        })

        return metrics

    def get_scheduler_state(self) -> Dict[str, Any]:
        """Get current scheduler state for checkpointing."""
        # NOTE: We don't capture queued events to avoid race conditions
        # Events will be regenerated by the event loop based on DAG state during recovery

        return {
            "max_queue_size": self.max_queue_size,
            "stop_requested": self._stop_requested,
            "metrics": dict(self._metrics),
            "overflow_count": self._overflow_count,
            "last_overflow_time": self._last_overflow_time,
            "queue_current_size": self.event_queue.qsize(),  # Safe to call
            "queue_status": self.get_queue_status()
        }

    def restore_scheduler_state(self, state: Dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        try:
            # Restore basic state
            self._stop_requested = state.get("stop_requested", False)
            self._metrics.update(state.get("metrics", {}))
            self._overflow_count = state.get("overflow_count", 0)
            self._last_overflow_time = state.get("last_overflow_time")

            # Note: We don't restore queued events as they should be regenerated
            # by the event loop based on the DAG state

            logger.info("Scheduler state restored from checkpoint")

        except Exception as e:
            logger.error(f"Failed to restore scheduler state: {e}")
            raise
