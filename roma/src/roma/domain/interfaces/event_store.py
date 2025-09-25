"""
Event Store Interface - ROMA v2.0 Domain Interface.

Abstract interface for event storage operations that infrastructure must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.events.task_events import BaseTaskEvent


class EventFilter:
    """Filter criteria for querying events."""

    def __init__(
        self,
        task_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ):
        self.task_id = task_id
        self.event_type = event_type
        self.limit = limit
        self.offset = offset


class IEventStore(ABC):
    """
    Domain interface for event store operations.

    Abstract interface that defines all event storage operations required by the domain.
    Infrastructure layer must implement this interface to provide event persistence capabilities.
    """

    @abstractmethod
    async def append(self, event: BaseTaskEvent) -> None:
        """
        Append an event to the store.

        Args:
            event: Event to store

        Raises:
            EventStoreError: If event cannot be stored
        """

    @abstractmethod
    async def get_events(self, event_filter: EventFilter | None = None) -> list[BaseTaskEvent]:
        """
        Retrieve events based on filter criteria.

        Args:
            event_filter: Optional filter criteria

        Returns:
            List of events matching the filter criteria
        """

    @abstractmethod
    async def get_events_by_task_id(self, task_id: str) -> list[BaseTaskEvent]:
        """
        Get all events for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            List of events for the specified task
        """

    @abstractmethod
    async def get_events_by_type(self, event_type: str) -> list[BaseTaskEvent]:
        """
        Get all events of a specific type.

        Args:
            event_type: Event type to filter by

        Returns:
            List of events of the specified type
        """

    @abstractmethod
    async def count_events(self, event_filter: EventFilter | None = None) -> int:
        """
        Count events matching filter criteria.

        Args:
            event_filter: Optional filter criteria

        Returns:
            Number of events matching the criteria
        """

    @abstractmethod
    async def clear(self) -> None:
        """
        Clear all events from the store.

        This is typically used for cleanup during testing or shutdown.
        """

    @abstractmethod
    async def get_latest_event(self, task_id: str) -> BaseTaskEvent | None:
        """
        Get the most recent event for a specific task.

        Args:
            task_id: Task identifier

        Returns:
            Most recent event for the task, or None if no events found
        """

    @abstractmethod
    async def get_event_types(self) -> list[str]:
        """
        Get all unique event types in the store.

        Returns:
            List of unique event type strings
        """

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """
        Get statistics about the event store.

        Returns:
            Dictionary with store statistics (total events, types, etc.)
        """
