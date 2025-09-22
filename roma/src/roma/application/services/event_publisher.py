"""
Universal Event Publisher Service - ROMA v2.0

Centralized event emission service for all components in the system.
Provides type-safe event builders, error handling, and consistent event patterns.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union
from uuid import uuid4

from roma.domain.entities.task_node import TaskNode
from roma.domain.events.task_events import BaseTaskEvent, utc_now
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.agent_type import AgentType
from roma.application.services.event_store import InMemoryEventStore
from roma.application.services.context_builder_service import TaskContext
from roma.infrastructure.agents.configurable_agent import ConfigurableAgent

logger = logging.getLogger(__name__)


class EventPublisher:
    """
    Universal event publisher for all system components.

    Provides centralized, type-safe event emission with error handling.
    Supports event categories and consistent event patterns across the system.
    """

    def __init__(self, event_store: Optional[InMemoryEventStore] = None):
        """
        Initialize event publisher.

        Args:
            event_store: Event store for persisting events
        """
        self._event_store = event_store
        self._events_emitted = 0
        self._failed_emissions = 0

    async def emit_event(
        self,
        event_type: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
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
        try:
            if not self._event_store:
                logger.debug(f"No event store configured, skipping event: {event_type}")
                return False

            event = BaseTaskEvent(
                event_id=str(uuid4()),
                event_type=event_type,
                task_id=task_id or "system-event",
                timestamp=utc_now(),
                metadata=metadata or {}
            )

            await self._event_store.append(event)
            self._events_emitted += 1
            logger.debug(f"Emitted event: {event_type}")
            return True

        except Exception as e:
            self._failed_emissions += 1
            logger.error(f"Failed to emit event {event_type}: {e}")
            # Don't fail the main operation if event emission fails
            return False

    # Agent Runtime Events
    async def emit_runtime_initialized(
        self,
        framework: str = "agno",
        agent_factory_available: bool = False,
        runtime_agents_created: int = 0
    ) -> bool:
        """Emit runtime initialization event."""
        return await self.emit_event(
            "runtime.runtime_initialized",
            metadata={
                "framework": framework,
                "agent_factory_available": agent_factory_available,
                "runtime_agents_created": runtime_agents_created
            }
        )

    async def emit_runtime_shutdown(self, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Emit runtime shutdown event."""
        return await self.emit_event(
            "runtime.runtime_shutdown",
            metadata={"metrics": metrics or {}}
        )

    async def emit_agent_execution_started(
        self,
        task: TaskNode,
        agent: ConfigurableAgent,
        context: Optional[TaskContext] = None
    ) -> bool:
        """Emit agent execution started event."""
        context_files = 0
        if context and context.context_items:
            context_files = len([
                item for item in context.context_items
                if item.item_type.value in ["image_artifact", "audio_artifact", "video_artifact", "file_artifact"]
            ])

        return await self.emit_event(
            "runtime.agent_execution_started",
            task_id=task.task_id,
            metadata={
                "task_type": task.task_type.value,
                "agent_name": agent.name,
                "context_provided": context is not None,
                "context_files": context_files
            }
        )

    async def emit_agent_execution_completed(
        self,
        task: TaskNode,
        agent: ConfigurableAgent,
        success: bool = True
    ) -> bool:
        """Emit agent execution completed event."""
        return await self.emit_event(
            "runtime.agent_execution_completed",
            task_id=task.task_id,
            metadata={
                "task_type": task.task_type.value,
                "agent_name": agent.name,
                "success": success
            }
        )

    async def emit_agent_execution_failed(
        self,
        task: TaskNode,
        agent: ConfigurableAgent,
        error: str
    ) -> bool:
        """Emit agent execution failed event."""
        return await self.emit_event(
            "runtime.agent_execution_failed",
            task_id=task.task_id,
            metadata={
                "task_type": task.task_type.value,
                "agent_name": agent.name,
                "error": error
            }
        )

    # Task Graph Events
    async def emit_task_node_added(
        self,
        task_id: str,
        goal: str,
        task_type: TaskType,
        parent_id: Optional[str] = None
    ) -> bool:
        """Emit task node added event."""
        return await self.emit_event(
            "graph.task_node_added",
            task_id=task_id,
            metadata={
                "goal": goal,
                "task_type": task_type.value,
                "parent_id": parent_id
            }
        )

    async def emit_task_status_changed(
        self,
        task_id: str,
        old_status: str,
        new_status: str,
        goal: Optional[str] = None
    ) -> bool:
        """Emit task status changed event."""
        return await self.emit_event(
            "graph.task_status_changed",
            task_id=task_id,
            metadata={
                "old_status": old_status,
                "new_status": new_status,
                "goal": goal
            }
        )

    async def emit_dependency_added(
        self,
        from_task_id: str,
        to_task_id: str
    ) -> bool:
        """Emit dependency edge added event."""
        return await self.emit_event(
            "graph.dependency_added",
            task_id=to_task_id,
            metadata={
                "from_task_id": from_task_id,
                "to_task_id": to_task_id
            }
        )

    # Service Events
    async def emit_service_event(
        self,
        service_name: str,
        event_name: str,
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Emit service-specific event."""
        event_type = f"service.{service_name}.{event_name}"
        return await self.emit_event(event_type, task_id, metadata)

    # System Events
    async def emit_system_startup(self, component: str, version: str = "2.0.0") -> bool:
        """Emit system startup event."""
        return await self.emit_event(
            "system.startup",
            metadata={
                "component": component,
                "version": version,
                "timestamp": datetime.now().isoformat()
            }
        )

    async def emit_system_shutdown(self, component: str, reason: str = "normal") -> bool:
        """Emit system shutdown event."""
        return await self.emit_event(
            "system.shutdown",
            metadata={
                "component": component,
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        )

    async def emit_error_event(
        self,
        error_type: str,
        error_message: str,
        component: str,
        task_id: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Emit error event."""
        metadata = {
            "error_type": error_type,
            "error_message": error_message,
            "component": component,
            "timestamp": datetime.now().isoformat()
        }
        if additional_data:
            metadata.update(additional_data)

        return await self.emit_event(
            "system.error",
            task_id=task_id,
            metadata=metadata
        )

    # Publisher Statistics
    def get_publisher_stats(self) -> Dict[str, Any]:
        """Get event publisher statistics."""
        return {
            "events_emitted": self._events_emitted,
            "failed_emissions": self._failed_emissions,
            "success_rate": (
                self._events_emitted / (self._events_emitted + self._failed_emissions)
                if (self._events_emitted + self._failed_emissions) > 0 else 1.0
            ),
            "event_store_available": self._event_store is not None
        }


# Global event publisher instance (singleton pattern)
_global_event_publisher: Optional[EventPublisher] = None


def initialize_event_publisher(event_store: Optional[InMemoryEventStore] = None) -> EventPublisher:
    """
    Initialize the global event publisher.

    Args:
        event_store: Event store for persisting events

    Returns:
        Initialized event publisher instance
    """
    global _global_event_publisher
    _global_event_publisher = EventPublisher(event_store)
    return _global_event_publisher


def get_event_publisher() -> Optional[EventPublisher]:
    """
    Get the global event publisher instance.

    Returns:
        Global event publisher or None if not initialized
    """
    return _global_event_publisher


async def emit_event(
    event_type: str,
    task_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Convenience function for emitting events via global publisher.

    Args:
        event_type: Type identifier for the event
        task_id: Associated task ID (optional)
        metadata: Additional event data

    Returns:
        True if event was emitted successfully
    """
    publisher = get_event_publisher()
    if publisher:
        return await publisher.emit_event(event_type, task_id, metadata)

    logger.debug(f"No global event publisher configured, skipping event: {event_type}")
    return False