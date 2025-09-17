"""
Unit tests for EventStore edge cases to improve coverage.

Tests specific edge cases and error conditions in the EventStore implementation.
"""

import pytest
import asyncio
from dataclasses import replace
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque

from src.roma.domain.events.task_events import (
    TaskCreatedEvent, TaskStatusChangedEvent, AtomizerEvaluatedEvent,
    TaskCompletedEvent, TaskFailedEvent, BaseTaskEvent
)
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.node_type import NodeType
from src.roma.application.services.event_store import (
    InMemoryEventStore, EventFilter, get_event_store, emit_event
)


class TestEventStoreEdgeCases:
    """Test edge cases to improve EventStore coverage."""

    @pytest.mark.asyncio
    async def test_get_events_nonexistent_task_id(self):
        """Test getting events for non-existent task ID returns empty list."""
        store = InMemoryEventStore()

        events = await store.get_events("nonexistent-task-id")

        assert events == []
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_get_events_by_type_nonexistent_type(self):
        """Test getting events by non-existent type returns empty list."""
        store = InMemoryEventStore()

        events = await store.get_events_by_type("nonexistent_event_type")

        assert events == []
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_apply_filter_with_all_none_values(self):
        """Test _apply_filter with EventFilter having all None values."""
        store = InMemoryEventStore()

        # Create test events
        event1 = TaskCreatedEvent.create("task-1", "Goal 1", TaskType.THINK)
        event2 = TaskStatusChangedEvent.create("task-1", TaskStatus.PENDING, TaskStatus.READY, 1)
        events = [event1, event2]

        # Filter with all None values should return all events
        filter_none = EventFilter()
        filtered_events = store._apply_filter(events, filter_none)

        assert len(filtered_events) == 2
        assert filtered_events == events

    @pytest.mark.asyncio
    async def test_matches_metadata_missing_key_in_event(self):
        """Test _matches_metadata when event metadata is missing filter key."""
        store = InMemoryEventStore()

        event_metadata = {"existing_key": "value1"}
        filter_metadata = {"missing_key": "value2"}

        result = store._matches_metadata(event_metadata, filter_metadata)

        assert result is False

    @pytest.mark.asyncio
    async def test_matches_metadata_value_mismatch(self):
        """Test _matches_metadata when event metadata value doesn't match filter."""
        store = InMemoryEventStore()

        event_metadata = {"key": "different_value"}
        filter_metadata = {"key": "expected_value"}

        result = store._matches_metadata(event_metadata, filter_metadata)

        assert result is False

    @pytest.mark.asyncio
    async def test_matches_metadata_empty_filter(self):
        """Test _matches_metadata with empty filter metadata."""
        store = InMemoryEventStore()

        event_metadata = {"key": "value"}
        filter_metadata = {}

        result = store._matches_metadata(event_metadata, filter_metadata)

        assert result is True

    @pytest.mark.asyncio
    async def test_subscriber_exception_handling(self):
        """Test that subscriber exceptions don't break event storage."""
        store = InMemoryEventStore()

        # Create failing subscriber
        def failing_callback(event):
            raise ValueError("Subscriber failed")

        # Subscribe the failing callback
        await store.subscribe(failing_callback)

        # Append event should not raise exception despite subscriber failure
        event = TaskCreatedEvent.create("test-task", "Test goal", TaskType.THINK)
        await store.append(event)

        # Event should still be stored
        events = await store.get_events("test-task")
        assert len(events) == 1
        assert events[0] == event

    @pytest.mark.asyncio
    async def test_async_subscriber_exception_handling(self):
        """Test that async subscriber exceptions don't break event storage."""
        store = InMemoryEventStore()

        # Create failing async subscriber
        async def failing_async_callback(event):
            raise ValueError("Async subscriber failed")

        # Subscribe the failing callback
        await store.subscribe_async(failing_async_callback)

        # Append event should not raise exception despite subscriber failure
        event = TaskCreatedEvent.create("test-task", "Test goal", TaskType.THINK)
        await store.append(event)

        # Event should still be stored
        events = await store.get_events("test-task")
        assert len(events) == 1
        assert events[0] == event

    @pytest.mark.asyncio
    async def test_get_event_description_unknown_event_type(self):
        """Test _get_event_description with unknown event type."""
        store = InMemoryEventStore()

        # Create mock event with unknown type
        mock_event = Mock()
        mock_event.event_type = "unknown_event_type"

        description = store._get_event_description(mock_event)

        assert description == "Event: unknown_event_type"

    @pytest.mark.asyncio
    async def test_get_event_description_missing_attributes(self):
        """Test _get_event_description when event is missing expected attributes."""
        store = InMemoryEventStore()

        # Test task_created event without goal attribute
        mock_event = Mock()
        mock_event.event_type = "task_created"
        # Remove goal attribute to test getattr default
        if hasattr(mock_event, 'goal'):
            delattr(mock_event, 'goal')

        description = store._get_event_description(mock_event)
        assert "Unknown" in description

    @pytest.mark.asyncio
    async def test_generate_timeline_empty_store(self):
        """Test generate_timeline with empty event store."""
        store = InMemoryEventStore()

        timeline = await store.generate_timeline()

        assert timeline == []
        assert len(timeline) == 0

    @pytest.mark.asyncio
    async def test_generate_timeline_chronological_order(self):
        """Test generate_timeline returns events in chronological order."""
        store = InMemoryEventStore()

        # Create events with different timestamps
        base_time = datetime.now(timezone.utc)

        event1 = TaskCreatedEvent.create("task-1", "Goal 1", TaskType.THINK)
        event1 = replace(event1, timestamp=base_time + timedelta(seconds=2))  # Later

        event2 = TaskStatusChangedEvent.create("task-2", TaskStatus.PENDING, TaskStatus.READY, 1)
        event2 = replace(event2, timestamp=base_time + timedelta(seconds=1))  # Earlier

        # Append in reverse chronological order
        await store.append(event1)
        await store.append(event2)

        timeline = await store.generate_timeline()

        # Should be sorted by timestamp (earliest first)
        assert len(timeline) == 2
        assert timeline[0]["task_id"] == "task-2"  # Earlier event first
        assert timeline[1]["task_id"] == "task-1"  # Later event second

    @pytest.mark.asyncio
    async def test_clear_nonexistent_task_id(self):
        """Test clearing events for non-existent task ID."""
        store = InMemoryEventStore()

        # Add an event for a different task
        await store.append(TaskCreatedEvent.create("existing-task", "Goal", TaskType.THINK))

        # Clear non-existent task (should not raise error)
        await store.clear("nonexistent-task")

        # Original event should still be there
        events = await store.get_events("existing-task")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_rebuild_indices_complex_scenario(self):
        """Test _rebuild_indices with complex event structure."""
        store = InMemoryEventStore()

        # Create events for multiple tasks and types
        await store.append(TaskCreatedEvent.create("task-1", "Goal 1", TaskType.THINK))
        await store.append(TaskCreatedEvent.create("task-2", "Goal 2", TaskType.RETRIEVE))
        await store.append(TaskStatusChangedEvent.create("task-1", TaskStatus.PENDING, TaskStatus.READY, 1))
        await store.append(TaskCompletedEvent.create("task-2", "Completed"))

        # Clear specific task to trigger rebuild
        await store.clear("task-1")

        # Verify indices were rebuilt correctly
        all_events = await store.get_all_events()
        assert len(all_events) == 2  # Only task-2 events remain

        created_events = await store.get_events_by_type("task_created")
        assert len(created_events) == 1
        assert created_events[0].task_id == "task-2"

        completed_events = await store.get_events_by_type("task_completed")
        assert len(completed_events) == 1
        assert completed_events[0].task_id == "task-2"

    @pytest.mark.asyncio
    async def test_memory_limits_enforcement(self):
        """Test that memory limits are enforced for deques."""
        # Create store with small limits
        store = InMemoryEventStore(max_events_per_task=3, max_total_events=5)

        task_id = "limited-task"

        # Add more events than the per-task limit
        for i in range(5):
            await store.append(TaskCreatedEvent.create(task_id, f"Goal {i}", TaskType.THINK))

        # Should only keep the last 3 events for the task
        task_events = await store.get_events(task_id)
        assert len(task_events) <= 3

        # Global events should be limited too
        all_events = await store.get_all_events()
        assert len(all_events) <= 5

    @pytest.mark.asyncio
    async def test_event_filter_edge_cases_combined(self):
        """Test complex event filter combinations with edge cases."""
        store = InMemoryEventStore()

        base_time = datetime.now(timezone.utc)

        # Create event that matches some but not all filter criteria
        event = TaskCreatedEvent.create(
            "test-task", "Test goal", TaskType.THINK,
            metadata={"source": "user", "priority": "high"}
        )
        event = replace(event, timestamp=base_time)
        await store.append(event)

        # Filter that should exclude the event (wrong event type)
        filter_wrong_type = EventFilter(
            task_id="test-task",
            event_type="task_completed",  # Wrong type
            start_time=base_time - timedelta(minutes=1),
            end_time=base_time + timedelta(minutes=1),
            metadata_filters={"source": "user"}
        )

        filtered_events = await store.get_events("test-task", filter_wrong_type)
        assert len(filtered_events) == 0

        # Filter that should exclude the event (time range)
        filter_wrong_time = EventFilter(
            event_type="task_created",
            start_time=base_time + timedelta(minutes=1),  # Too late
            metadata_filters={"source": "user"}
        )

        filtered_events = await store.get_events("test-task", filter_wrong_time)
        assert len(filtered_events) == 0

    @pytest.mark.asyncio
    async def test_get_events_by_task_id_compatibility(self):
        """Test compatibility method get_events_by_task_id."""
        store = InMemoryEventStore()

        event = TaskCreatedEvent.create("compat-task", "Test goal", TaskType.THINK)
        await store.append(event)

        # Test compatibility method
        events = await store.get_events_by_task_id("compat-task")

        assert len(events) == 1
        assert events[0] == event

        # Test with filter
        filter_obj = EventFilter(event_type="task_created")
        filtered_events = await store.get_events_by_task_id("compat-task", filter_obj)

        assert len(filtered_events) == 1
        assert filtered_events[0] == event


class TestGlobalEventStoreEdgeCases:
    """Test global event store edge cases."""

    @pytest.mark.asyncio
    async def test_get_event_store_creates_singleton(self):
        """Test that get_event_store creates singleton on first call."""
        from src.roma.application.services.event_store import _global_event_store

        # Access the module-level variable to check initial state
        # (this is implementation detail testing for coverage)
        store1 = get_event_store()
        store2 = get_event_store()

        assert store1 is not None
        assert store1 is store2

    @pytest.mark.asyncio
    async def test_emit_event_uses_global_store(self):
        """Test that emit_event uses the global store."""
        # Clear any existing events in global store
        global_store = get_event_store()
        await global_store.clear()

        event = TaskCreatedEvent.create("global-emit-test", "Global test", TaskType.THINK)

        # Emit using the convenience function
        await emit_event(event)

        # Verify it was stored in the global store
        stored_events = await global_store.get_events("global-emit-test")
        assert len(stored_events) == 1
        assert stored_events[0] == event


class TestEventStoreStatisticsEdgeCases:
    """Test statistics calculation edge cases."""

    @pytest.mark.asyncio
    async def test_statistics_with_empty_store(self):
        """Test statistics calculation with empty store."""
        store = InMemoryEventStore()

        stats = await store.get_statistics()

        assert stats["total_events"] == 0
        assert stats["events_by_type"] == {}
        assert stats["events_by_task_count"] == 0
        assert stats["active_subscribers"] == 0
        assert stats["memory_usage"]["tasks_tracked"] == 0
        assert stats["memory_usage"]["global_events"] == 0

    @pytest.mark.asyncio
    async def test_statistics_after_clear(self):
        """Test statistics after clearing events."""
        store = InMemoryEventStore()

        # Add some events
        await store.append(TaskCreatedEvent.create("task-1", "Goal", TaskType.THINK))
        await store.append(TaskStatusChangedEvent.create("task-1", TaskStatus.PENDING, TaskStatus.READY, 1))

        # Verify events exist
        stats_before = await store.get_statistics()
        assert stats_before["total_events"] == 2

        # Clear all events
        await store.clear()

        # Statistics should be reset
        stats_after = await store.get_statistics()
        assert stats_after["total_events"] == 0
        assert stats_after["events_by_type"] == {}
        assert stats_after["events_by_task_count"] == 0