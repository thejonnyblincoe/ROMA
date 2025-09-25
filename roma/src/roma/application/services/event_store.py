"""
In-Memory Event Store for ROMA v2.0

Simple event storage system for observability without database dependencies.
Can be evolved to persistent storage later.
"""

import asyncio
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from roma.domain.events.task_events import BaseTaskEvent
from roma.domain.interfaces.event_store import EventFilter, IEventStore


class InMemoryEventStore(IEventStore):
    """
    In-memory event store with observability features.

    Features:
    - Fast event append and query
    - Event filtering and search
    - Event subscribers/listeners
    - Memory-efficient with size limits
    - Thread-safe operations
    """

    def __init__(self, max_events_per_task: int = 1000, max_total_events: int = 100000):
        """
        Initialize event store.

        Args:
            max_events_per_task: Maximum events to store per task
            max_total_events: Maximum total events to store
        """
        self.max_events_per_task = max_events_per_task
        self.max_total_events = max_total_events

        # Event storage by task_id
        self._events_by_task: dict[str, deque[BaseTaskEvent]] = defaultdict(
            lambda: deque(maxlen=max_events_per_task)
        )

        # Global event log (FIFO)
        self._global_events: deque[BaseTaskEvent] = deque(maxlen=max_total_events)

        # Event type indices for fast filtering
        self._events_by_type: dict[str, deque[BaseTaskEvent]] = defaultdict(
            lambda: deque(maxlen=max_total_events // 10)
        )

        # Event subscribers (store id + callback)
        self._subscribers: list[tuple[str, Callable[[BaseTaskEvent], None]]] = []
        self._async_subscribers: list[tuple[str, Callable[[BaseTaskEvent], Any]]] = []

        # Statistics
        self._events_by_type_count: defaultdict[str, int] = defaultdict(int)
        self._events_by_task_count: defaultdict[str, int] = defaultdict(int)
        self._total_events = 0

        # Thread safety
        self._lock = asyncio.Lock()

    async def append(self, event: BaseTaskEvent) -> None:
        """
        Append event to store and notify subscribers.

        Args:
            event: Event to append
        """
        async with self._lock:
            # Store in task-specific deque
            self._events_by_task[event.task_id].append(event)

            # Store in global deque
            self._global_events.append(event)

            # Store in type index
            self._events_by_type[event.event_type].append(event)

            # Update statistics
            self._total_events += 1
            self._events_by_type_count[event.event_type] += 1
            self._events_by_task_count[event.task_id] += 1

        # Notify subscribers (outside lock to avoid blocking)
        await self._notify_subscribers(event)

    async def get_events(
        self, task_id: str, event_filter: EventFilter | None = None
    ) -> list[BaseTaskEvent]:
        """
        Get events for a specific task.

        Args:
            task_id: Task ID to get events for
            event_filter: Optional filter criteria

        Returns:
            List of events matching criteria
        """
        async with self._lock:
            events = list(self._events_by_task.get(task_id, []))

        if event_filter:
            events = self._apply_filter(events, event_filter)

        return events

    async def get_all_events(self, event_filter: EventFilter | None = None) -> list[BaseTaskEvent]:
        """
        Get all events across all tasks.

        Args:
            event_filter: Optional filter criteria

        Returns:
            List of events matching criteria
        """
        async with self._lock:
            events = list(self._global_events)

        if event_filter:
            events = self._apply_filter(events, event_filter)

        return events

    async def get_events_by_type(
        self, event_type: str, event_filter: EventFilter | None = None
    ) -> list[BaseTaskEvent]:
        """
        Get events by type.

        Args:
            event_type: Event type to filter by
            event_filter: Optional additional filter criteria

        Returns:
            List of events matching criteria
        """
        async with self._lock:
            events = list(self._events_by_type.get(event_type, []))

        if event_filter:
            events = self._apply_filter(events, event_filter)

        return events

    def _apply_filter(
        self, events: list[BaseTaskEvent], event_filter: EventFilter
    ) -> list[BaseTaskEvent]:
        """Apply filter criteria to event list."""
        filtered_events = events

        if event_filter.event_type:
            filtered_events = [
                e for e in filtered_events if e.event_type == event_filter.event_type
            ]

        if event_filter.start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= event_filter.start_time]

        if event_filter.end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= event_filter.end_time]

        if event_filter.metadata_filters:
            filtered_events = [
                e
                for e in filtered_events
                if self._matches_metadata(e.metadata, event_filter.metadata_filters)
            ]

        return filtered_events

    def _matches_metadata(
        self, event_metadata: dict[str, Any], filter_metadata: dict[str, Any]
    ) -> bool:
        """Check if event metadata matches filter criteria."""
        for key, value in filter_metadata.items():
            if key not in event_metadata or event_metadata[key] != value:
                return False
        return True

    async def subscribe(self, callback: Callable[[BaseTaskEvent], None]) -> str:
        """
        Subscribe to all events (synchronous callback).

        Args:
            callback: Function to call for each event

        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = str(uuid4())
        self._subscribers.append((subscription_id, callback))
        return subscription_id

    async def subscribe_async(self, callback: Callable[[BaseTaskEvent], None]) -> str:
        """
        Subscribe to all events (asynchronous callback).

        Args:
            callback: Async function to call for each event

        Returns:
            Subscription ID for unsubscribing
        """
        subscription_id = str(uuid4())
        self._async_subscribers.append((subscription_id, callback))
        return subscription_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID returned from subscribe

        Returns:
            True if subscription was found and removed
        """
        # Check sync subscribers
        for i, (sub_id, _) in enumerate(self._subscribers):
            if sub_id == subscription_id:
                del self._subscribers[i]
                return True

        # Check async subscribers
        for i, (sub_id, _) in enumerate(self._async_subscribers):
            if sub_id == subscription_id:
                del self._async_subscribers[i]
                return True

        return False

    async def _notify_subscribers(self, event: BaseTaskEvent) -> None:
        """Notify all subscribers of new event."""
        # Notify sync subscribers
        for _, callback in self._subscribers:
            try:
                callback(event)
            except Exception as e:
                # Log error but don't fail event storage
                print(f"Event subscriber error: {e}")

        # Notify async subscribers
        tasks = []
        for _, callback in self._async_subscribers:
            try:
                task: asyncio.Task[Any] = asyncio.create_task(callback(event))
                tasks.append(task)
            except Exception as e:
                print(f"Async event subscriber error: {e}")

        # Don't wait for async subscribers to complete
        if tasks:
            asyncio.gather(*tasks, return_exceptions=True)

    async def get_task_timeline(self, task_id: str) -> list[dict[str, Any]]:
        """
        Get chronological timeline of events for a task.

        Args:
            task_id: Task ID to get timeline for

        Returns:
            List of timeline events with human-readable descriptions
        """
        events = await self.get_events(task_id)
        timeline = []

        for event in events:
            timeline_item = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "description": self._get_event_description(event),
                "metadata": event.metadata,
            }
            timeline.append(timeline_item)

        return timeline

    # Compatibility wrappers for tested interfaces
    async def get_events_by_task_id(
        self, task_id: str, event_filter: EventFilter | None = None
    ) -> list[BaseTaskEvent]:
        """Compatibility method to get events by task id."""
        return await self.get_events(task_id, event_filter)

    async def generate_timeline(self) -> list[dict[str, Any]]:
        """Generate a global, chronological event timeline across all tasks."""
        async with self._lock:
            events = list(self._global_events)
        events.sort(key=lambda e: e.timestamp)
        timeline: list[dict[str, Any]] = []
        for event in events:
            timeline.append(
                {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "task_id": event.task_id,
                    "description": self._get_event_description(event),
                    "metadata": event.metadata,
                }
            )
        return timeline

    def _get_event_description(self, event: BaseTaskEvent) -> str:
        """Generate human-readable description for event."""
        if event.event_type == "task_created":
            return f"Task created with goal: {getattr(event, 'goal', 'Unknown')}"
        elif event.event_type == "task_status_changed":
            old_status = getattr(event, "old_status", "Unknown")
            new_status = getattr(event, "new_status", "Unknown")
            return f"Status changed from {old_status} to {new_status}"
        elif event.event_type == "atomizer_evaluated":
            is_atomic = getattr(event, "is_atomic", False)
            decision = "atomic" if is_atomic else "needs decomposition"
            return f"Atomizer determined task is {decision}"
        elif event.event_type == "task_decomposed":
            count = getattr(event, "subtask_count", 0)
            return f"Task decomposed into {count} subtasks"
        elif event.event_type == "task_executed":
            duration = getattr(event, "execution_duration_ms", 0)
            return f"Task executed in {duration:.1f}ms"
        elif event.event_type == "task_completed":
            return "Task completed successfully"
        elif event.event_type == "task_failed":
            error = getattr(event, "error_message", "Unknown error")
            return f"Task failed: {error}"
        elif event.event_type == "results_aggregated":
            count = getattr(event, "child_count", 0)
            return f"Aggregated results from {count} child tasks"
        else:
            return f"Event: {event.event_type}"

    async def get_statistics(self) -> dict[str, Any]:
        """
        Get event store statistics.

        Returns:
            Dictionary with event statistics
        """
        async with self._lock:
            return {
                "total_events": self._total_events,
                "events_by_type": dict(self._events_by_type_count),
                "events_by_task_count": len(self._events_by_task_count),
                "active_subscribers": len(self._subscribers) + len(self._async_subscribers),
                "memory_usage": {
                    "tasks_tracked": len(self._events_by_task),
                    "global_events": len(self._global_events),
                    "max_events_per_task": self.max_events_per_task,
                    "max_total_events": self.max_total_events,
                },
            }

    async def clear(self, task_id: str | None = None) -> None:
        """
        Clear events from store.

        Args:
            task_id: If provided, clear only events for this task.
                    If None, clear all events.
        """
        async with self._lock:
            if task_id:
                # Clear specific task
                if task_id in self._events_by_task:
                    del self._events_by_task[task_id]
                    self._events_by_task_count.pop(task_id, None)

                    # Update global and type indices (expensive operation)
                    self._rebuild_indices()
            else:
                # Clear all events
                self._events_by_task.clear()
                self._global_events.clear()
                self._events_by_type.clear()
                self._total_events = 0
                self._events_by_type_count.clear()
                self._events_by_task_count.clear()

    def _rebuild_indices(self) -> None:
        """Rebuild global and type indices after selective deletion."""
        # This is expensive - only called when clearing specific tasks
        self._global_events.clear()
        self._events_by_type.clear()

        for events in self._events_by_task.values():
            for event in events:
                self._global_events.append(event)
                self._events_by_type[event.event_type].append(event)


# Global event store instance
_global_event_store: InMemoryEventStore | None = None


def get_event_store() -> InMemoryEventStore:
    """
    Get the global event store instance.

    Returns:
        Global InMemoryEventStore instance
    """
    global _global_event_store
    if _global_event_store is None:
        _global_event_store = InMemoryEventStore()
    return _global_event_store


async def emit_event(event: BaseTaskEvent) -> None:
    """
    Emit event to the global event store.

    Args:
        event: Event to emit
    """
    store = get_event_store()
    await store.append(event)
