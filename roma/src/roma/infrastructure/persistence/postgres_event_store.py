"""
PostgreSQL Event Store Implementation
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import uuid4

from sqlalchemy import select, delete, update, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from roma.domain.events.task_events import (
    BaseTaskEvent,
    TaskCreatedEvent,
    TaskNodeAddedEvent,
    TaskStatusChangedEvent,
    AtomizerEvaluatedEvent,
    TaskDecomposedEvent,
    TaskExecutedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    ResultsAggregatedEvent,
    DependencyAddedEvent
)
from roma.domain.value_objects.config.database_config import DatabaseConfig
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.node_type import NodeType
from roma.application.services.event_store import EventFilter
from .connection_manager import DatabaseConnectionManager
from .models.event_model import EventModel
from .models.base import Base

logger = logging.getLogger(__name__)


class PostgreSQLEventStore:
    """
    PostgreSQL-based event store with full compatibility with InMemoryEventStore API.

    Features:
    - Persistent event storage in PostgreSQL
    - Event filtering and search with database queries
    - Event subscribers/listeners with LISTEN/NOTIFY
    - Bulk operations for performance
    - Event archival and cleanup
    - Thread-safe operations
    """

    def __init__(self, connection_manager: DatabaseConnectionManager):
        """
        Initialize PostgreSQL event store.

        Args:
            connection_manager: Database connection manager
        """
        self.connection_manager = connection_manager
        self.config = connection_manager.config

        # Event subscribers (store id + callback)
        self._subscribers: List[tuple[str, Callable[[BaseTaskEvent], None]]] = []
        self._async_subscribers: List[tuple[str, Callable[[BaseTaskEvent], Any]]] = []

        # Statistics
        self._stats = {
            "total_events": 0,
            "events_by_type": defaultdict(int),
            "events_by_task": defaultdict(int),
            "query_count": 0,
            "bulk_insert_count": 0,
        }

        # Thread safety
        self._lock = asyncio.Lock()

        # SQLAlchemy setup (will reuse connection manager's pool settings)
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        """Initialize the event store."""
        logger.info("Initializing PostgreSQL event store")

        # Ensure connection manager is initialized first
        if not self.connection_manager._pool:
            await self.connection_manager.initialize()

        # Create SQLAlchemy engine with minimal pooling to avoid duplication.
        # Main connection pooling is handled by connection_manager for other operations.
        dsn = self.config.get_dsn().replace("postgresql://", "postgresql+asyncpg://")
        self._engine = create_async_engine(
            dsn,
            echo=False,
            # Use minimal pooling since connection_manager handles the main pool
            pool_size=1,
            max_overflow=2,
            pool_timeout=self.config.pool.connection_timeout,
            pool_recycle=self.config.max_connection_age,
        )

        # Create session factory
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        # Create tables if they don't exist
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("PostgreSQL event store initialized")

    async def close(self) -> None:
        """Close the event store."""
        logger.info("Closing PostgreSQL event store")

        # Dispose SQLAlchemy engine
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None

        # Cancel all subscriber tasks
        await self._cleanup_subscriber_tasks()

        self._subscribers.clear()
        self._async_subscribers.clear()

        logger.info("PostgreSQL event store closed")

    async def append(self, event: BaseTaskEvent) -> None:
        """
        Append event to store and notify subscribers.

        Args:
            event: Event to append
        """
        try:
            async with self._session_factory() as session:
                # Create event model
                event_model = EventModel.from_task_event(event)

                # Add to database
                session.add(event_model)
                await session.commit()

                # Update statistics
                async with self._lock:
                    self._stats["total_events"] += 1
                    self._stats["events_by_type"][event.event_type] += 1
                    self._stats["events_by_task"][event.task_id] += 1

            # Notify subscribers (outside transaction)
            await self._notify_subscribers(event)

        except SQLAlchemyError as e:
            logger.error(f"Failed to append event: {e}")
            raise

    async def append_many(self, events: List[BaseTaskEvent]) -> None:
        """
        Append multiple events in a batch operation.

        Args:
            events: List of events to append
        """
        if not events:
            return

        try:
            async with self._session_factory() as session:
                # Create event models
                event_models = [EventModel.from_task_event(event) for event in events]

                # Bulk insert
                session.add_all(event_models)
                await session.commit()

                # Update statistics
                async with self._lock:
                    self._stats["total_events"] += len(events)
                    self._stats["bulk_insert_count"] += 1
                    for event in events:
                        self._stats["events_by_type"][event.event_type] += 1
                        self._stats["events_by_task"][event.task_id] += 1

            # Notify subscribers for each event
            for event in events:
                await self._notify_subscribers(event)

        except SQLAlchemyError as e:
            logger.error(f"Failed to append events batch: {e}")
            raise

    async def get_events(
        self,
        task_id: str,
        event_filter: Optional[EventFilter] = None
    ) -> List[BaseTaskEvent]:
        """
        Get events for a specific task.

        Args:
            task_id: Task ID to get events for
            event_filter: Optional filter criteria

        Returns:
            List of events matching criteria
        """
        try:
            async with self._session_factory() as session:
                # Build query
                query = select(EventModel).where(EventModel.task_id == task_id)

                # Apply filters
                if event_filter:
                    query = self._apply_filter_to_query(query, event_filter)

                # Order by timestamp
                query = query.order_by(EventModel.timestamp)

                # Execute query
                result = await session.execute(query)
                event_models = result.scalars().all()

                # Update statistics
                async with self._lock:
                    self._stats["query_count"] += 1

                # Convert to domain events
                return [self._model_to_event(model) for model in event_models]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get events for task {task_id}: {e}")
            raise

    async def get_all_events(
        self,
        event_filter: Optional[EventFilter] = None
    ) -> List[BaseTaskEvent]:
        """
        Get all events across all tasks.

        Args:
            event_filter: Optional filter criteria

        Returns:
            List of events matching criteria
        """
        try:
            async with self._session_factory() as session:
                # Build query
                query = select(EventModel)

                # Apply filters
                if event_filter:
                    query = self._apply_filter_to_query(query, event_filter)

                # Order by timestamp
                query = query.order_by(EventModel.timestamp)

                # Execute query
                result = await session.execute(query)
                event_models = result.scalars().all()

                # Update statistics
                async with self._lock:
                    self._stats["query_count"] += 1

                # Convert to domain events
                return [self._model_to_event(model) for model in event_models]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get all events: {e}")
            raise

    async def get_events_by_type(
        self,
        event_type: str,
        event_filter: Optional[EventFilter] = None
    ) -> List[BaseTaskEvent]:
        """
        Get events by type.

        Args:
            event_type: Event type to filter by
            event_filter: Optional additional filter criteria

        Returns:
            List of events matching criteria
        """
        try:
            async with self._session_factory() as session:
                # Build query
                query = select(EventModel).where(EventModel.event_type == event_type)

                # Apply additional filters
                if event_filter:
                    query = self._apply_filter_to_query(query, event_filter)

                # Order by timestamp
                query = query.order_by(EventModel.timestamp)

                # Execute query
                result = await session.execute(query)
                event_models = result.scalars().all()

                # Update statistics
                async with self._lock:
                    self._stats["query_count"] += 1

                # Convert to domain events
                return [self._model_to_event(model) for model in event_models]

        except SQLAlchemyError as e:
            logger.error(f"Failed to get events by type {event_type}: {e}")
            raise

    def _apply_filter_to_query(self, query, event_filter: EventFilter):
        """Apply event filter to SQLAlchemy query."""
        if event_filter.task_id:
            query = query.where(EventModel.task_id == event_filter.task_id)

        if event_filter.event_type:
            query = query.where(EventModel.event_type == event_filter.event_type)

        if event_filter.start_time:
            query = query.where(EventModel.timestamp >= event_filter.start_time)

        if event_filter.end_time:
            query = query.where(EventModel.timestamp <= event_filter.end_time)

        if event_filter.metadata_filters:
            # Use JSONB containment operator for metadata filtering
            for key, value in event_filter.metadata_filters.items():
                query = query.where(EventModel.metadata[key].astext == str(value))

        return query

    def _model_to_event(self, model: EventModel) -> BaseTaskEvent:
        """Convert EventModel to proper domain event based on event_type."""
        metadata = model.event_metadata or {}

        # Reconstruct the specific event type based on event_type
        if model.event_type == "task_created":
            return TaskCreatedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                goal=metadata.get("goal", ""),
                task_type=TaskType(metadata.get("task_type", "THINK")),
                parent_id=metadata.get("parent_id")
            )
        elif model.event_type == "task_node_added":
            return TaskNodeAddedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                goal=metadata.get("goal", ""),
                task_type=TaskType(metadata.get("task_type", "THINK")),
                parent_id=metadata.get("parent_id")
            )
        elif model.event_type == "task_status_changed":
            return TaskStatusChangedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                old_status=TaskStatus(metadata.get("old_status", "PENDING")),
                new_status=TaskStatus(metadata.get("new_status", "PENDING")),
                reason=metadata.get("reason")
            )
        elif model.event_type == "atomizer_evaluated":
            return AtomizerEvaluatedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                decision=metadata.get("decision", "EXECUTE"),
                reasoning=metadata.get("reasoning", ""),
                complexity_score=metadata.get("complexity_score", 0),
                confidence=metadata.get("confidence", 0.0)
            )
        elif model.event_type == "task_decomposed":
            return TaskDecomposedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                child_tasks=metadata.get("child_tasks", []),
                decomposition_strategy=metadata.get("decomposition_strategy", ""),
                total_children=metadata.get("total_children", 0)
            )
        elif model.event_type == "task_executed":
            return TaskExecutedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                result=metadata.get("result"),
                execution_time=metadata.get("execution_time", 0.0),
                agent_type=metadata.get("agent_type", "executor")
            )
        elif model.event_type == "task_completed":
            return TaskCompletedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                result=metadata.get("result"),
                total_execution_time=metadata.get("total_execution_time", 0.0),
                child_count=metadata.get("child_count", 0)
            )
        elif model.event_type == "task_failed":
            return TaskFailedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                error_message=metadata.get("error_message", ""),
                error_type=metadata.get("error_type", ""),
                retry_count=metadata.get("retry_count", 0),
                is_recoverable=metadata.get("is_recoverable", False)
            )
        elif model.event_type == "results_aggregated":
            return ResultsAggregatedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                child_results=metadata.get("child_results", []),
                aggregated_result=metadata.get("aggregated_result"),
                aggregation_strategy=metadata.get("aggregation_strategy", "")
            )
        elif model.event_type == "dependency_added":
            return DependencyAddedEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata,
                parent_id=metadata.get("parent_id", ""),
                child_id=metadata.get("child_id", ""),
                dependency_type=metadata.get("dependency_type", "parent_child")
            )
        else:
            # Fallback: create a generic BaseTaskEvent for unknown event types
            logger.warning(f"Unknown event type '{model.event_type}', creating generic BaseTaskEvent")
            return BaseTaskEvent(
                event_id=model.id,
                task_id=model.task_id,
                timestamp=model.timestamp,
                event_type=model.event_type,
                metadata=metadata
            )

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

    async def subscribe_async(
        self,
        callback: Callable[[BaseTaskEvent], None]
    ) -> str:
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
                logger.error(f"Event subscriber error: {e}")

        # Notify async subscribers
        tasks = []
        for _, callback in self._async_subscribers:
            try:
                task = asyncio.create_task(callback(event))
                tasks.append(task)
            except Exception as e:
                logger.error(f"Async event subscriber error: {e}")

        # Don't wait for async subscribers to complete but ensure proper cleanup
        if tasks:
            # Create a background task group for proper lifecycle management
            asyncio.create_task(
                self._cleanup_subscriber_tasks(asyncio.gather(*tasks, return_exceptions=True))
            )

    async def _cleanup_subscriber_tasks(self, task_group) -> None:
        """Cleanup subscriber tasks to prevent memory leaks."""
        try:
            await task_group
        except Exception as e:
            logger.error(f"Error in subscriber task cleanup: {e}")

    async def get_task_timeline(self, task_id: str) -> List[Dict[str, Any]]:
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
                "metadata": event.metadata
            }
            timeline.append(timeline_item)

        return timeline

    # Compatibility methods for InMemoryEventStore interface
    async def get_events_by_task_id(
        self, task_id: str, event_filter: Optional[EventFilter] = None
    ) -> List[BaseTaskEvent]:
        """Compatibility method to get events by task id."""
        return await self.get_events(task_id, event_filter)

    async def generate_timeline(self) -> List[Dict[str, Any]]:
        """Generate a global, chronological event timeline across all tasks."""
        events = await self.get_all_events()
        timeline = []

        for event in events:
            timeline.append({
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "task_id": event.task_id,
                "description": self._get_event_description(event),
                "metadata": event.metadata,
            })

        return timeline

    def _get_event_description(self, event: BaseTaskEvent) -> str:
        """Generate human-readable description for event."""
        if event.event_type == "task_created":
            return f"Task created with goal: {getattr(event, 'goal', 'Unknown')}"
        elif event.event_type == "task_status_changed":
            old_status = getattr(event, 'old_status', 'Unknown')
            new_status = getattr(event, 'new_status', 'Unknown')
            return f"Status changed from {old_status} to {new_status}"
        elif event.event_type == "atomizer_evaluated":
            is_atomic = getattr(event, 'is_atomic', False)
            decision = "atomic" if is_atomic else "needs decomposition"
            return f"Atomizer determined task is {decision}"
        elif event.event_type == "task_decomposed":
            count = getattr(event, 'subtask_count', 0)
            return f"Task decomposed into {count} subtasks"
        elif event.event_type == "task_executed":
            duration = getattr(event, 'execution_duration_ms', 0)
            return f"Task executed in {duration:.1f}ms"
        elif event.event_type == "task_completed":
            return "Task completed successfully"
        elif event.event_type == "task_failed":
            error = getattr(event, 'error_message', 'Unknown error')
            return f"Task failed: {error}"
        elif event.event_type == "results_aggregated":
            count = getattr(event, 'child_count', 0)
            return f"Aggregated results from {count} child tasks"
        else:
            return f"Event: {event.event_type}"

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get event store statistics.

        Returns:
            Dictionary with event statistics
        """
        # Get database statistics
        async with self._session_factory() as session:
            # Total events count
            total_result = await session.execute(select(func.count(EventModel.id)))
            total_events = total_result.scalar()

            # Events by type
            type_result = await session.execute(
                select(EventModel.event_type, func.count(EventModel.id))
                .group_by(EventModel.event_type)
            )
            events_by_type = dict(type_result.fetchall())

            # Active tasks count
            task_result = await session.execute(
                select(func.count(func.distinct(EventModel.task_id)))
            )
            active_tasks = task_result.scalar()

        async with self._lock:
            return {
                "total_events": total_events,
                "events_by_type": events_by_type,
                "active_tasks": active_tasks,
                "active_subscribers": len(self._subscribers) + len(self._async_subscribers),
                "query_count": self._stats["query_count"],
                "bulk_insert_count": self._stats["bulk_insert_count"],
                "database_stats": self.connection_manager.get_stats(),
            }

    async def clear(self, task_id: Optional[str] = None) -> None:
        """
        Clear events from store.

        Args:
            task_id: If provided, clear only events for this task.
                    If None, clear all events.
        """
        try:
            async with self._session_factory() as session:
                if task_id:
                    # Clear specific task
                    await session.execute(
                        delete(EventModel).where(EventModel.task_id == task_id)
                    )
                else:
                    # Clear all events
                    await session.execute(delete(EventModel))

                await session.commit()

            logger.info(f"Cleared events for task: {task_id or 'ALL'}")

        except SQLAlchemyError as e:
            logger.error(f"Failed to clear events: {e}")
            raise

    async def archive_old_events(self, days: int = None) -> int:
        """
        Archive events older than specified days.

        Args:
            days: Number of days to keep (uses config default if None)

        Returns:
            Number of events archived
        """
        if days is None:
            days = self.config.event_retention_days

        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        try:
            async with self._session_factory() as session:
                # Update old events as archived
                result = await session.execute(
                    update(EventModel)
                    .where(
                        and_(
                            EventModel.created_at < cutoff_date,
                            EventModel.is_archived == False
                        )
                    )
                    .values(
                        is_archived=True,
                        archived_at=datetime.now(timezone.utc)
                    )
                )

                await session.commit()
                archived_count = result.rowcount

            logger.info(f"Archived {archived_count} events older than {days} days")
            return archived_count

        except SQLAlchemyError as e:
            logger.error(f"Failed to archive events: {e}")
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()