"""
Event Publisher Interface - ROMA v2.0 Domain Interface.

Abstract interface for event publishing operations that infrastructure must implement.
"""

from abc import ABC, abstractmethod
from typing import Any

from roma.domain.value_objects.task_type import TaskType


class IEventPublisher(ABC):
    """
    Domain interface for event publisher operations.

    Abstract interface that defines all event publishing operations required by the domain.
    Infrastructure layer must implement this interface to provide event emission capabilities.
    """

    @abstractmethod
    async def emit_event(
        self, event_type: str, task_id: str | None = None, metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Emit a generic event.

        Args:
            event_type: Type identifier for the event
            task_id: Associated task ID (optional)
            metadata: Additional event data

        Returns:
            True if event was emitted successfully
        """

    @abstractmethod
    async def emit_task_node_added(
        self, task_id: str, goal: str, task_type: TaskType, parent_id: str | None = None
    ) -> bool:
        """
        Emit task node added event.

        Args:
            task_id: ID of the added task
            goal: Task goal description
            task_type: Type of the task
            parent_id: ID of parent task (if any)

        Returns:
            True if event was emitted successfully
        """

    @abstractmethod
    async def emit_task_status_changed(
        self, task_id: str, old_status: str, new_status: str, goal: str | None = None
    ) -> bool:
        """
        Emit task status changed event.

        Args:
            task_id: ID of the task
            old_status: Previous status
            new_status: New status
            goal: Task goal (optional)

        Returns:
            True if event was emitted successfully
        """

    @abstractmethod
    async def emit_dependency_added(self, from_task_id: str, to_task_id: str) -> bool:
        """
        Emit dependency edge added event.

        Args:
            from_task_id: Source task ID
            to_task_id: Target task ID

        Returns:
            True if event was emitted successfully
        """

    @abstractmethod
    async def emit_service_event(
        self,
        service_name: str,
        event_name: str,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Emit service-specific event.

        Args:
            service_name: Name of the service
            event_name: Name of the service event
            task_id: Associated task ID (optional)
            metadata: Additional event data

        Returns:
            True if event was emitted successfully
        """

    @abstractmethod
    async def emit_system_startup(self, component: str, version: str = "2.0.0") -> bool:
        """
        Emit system startup event.

        Args:
            component: Component name
            version: Component version

        Returns:
            True if event was emitted successfully
        """

    @abstractmethod
    async def emit_system_shutdown(self, component: str, reason: str = "normal") -> bool:
        """
        Emit system shutdown event.

        Args:
            component: Component name
            reason: Shutdown reason

        Returns:
            True if event was emitted successfully
        """

    @abstractmethod
    async def emit_error_event(
        self,
        error_type: str,
        error_message: str,
        component: str,
        task_id: str | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> bool:
        """
        Emit error event.

        Args:
            error_type: Type of error
            error_message: Error message
            component: Component where error occurred
            task_id: Associated task ID (optional)
            additional_data: Additional error data

        Returns:
            True if event was emitted successfully
        """

    @abstractmethod
    def get_publisher_stats(self) -> dict[str, Any]:
        """
        Get event publisher statistics.

        Returns:
            Dictionary with publisher statistics
        """
