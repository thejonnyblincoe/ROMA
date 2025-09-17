"""
Integration tests for event store system.

Tests event storage, retrieval, filtering, and subscription functionality.
"""

import pytest
import asyncio
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from typing import List

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.node_type import NodeType
from src.roma.domain.events.task_events import (
    TaskCreatedEvent, TaskStatusChangedEvent, AtomizerEvaluatedEvent,
    TaskCompletedEvent, TaskFailedEvent, BaseTaskEvent
)
from src.roma.application.services.event_store import (
    InMemoryEventStore, EventFilter, get_event_store, emit_event
)


class TestEventStoreBasicOperations:
    """Test basic event store operations."""
    
    @pytest.mark.asyncio
    async def test_append_and_retrieve_events(self, clean_event_store: InMemoryEventStore):
        """Test basic event append and retrieval."""
        task_id = "test-task-123"
        
        # Create and append events
        event1 = TaskCreatedEvent.create(
            task_id=task_id,
            goal="Test goal",
            task_type=TaskType.THINK
        )
        event2 = TaskStatusChangedEvent.create(
            task_id=task_id,
            old_status=TaskStatus.PENDING,
            new_status=TaskStatus.READY,
            version=1
        )
        
        await clean_event_store.append(event1)
        await clean_event_store.append(event2)
        
        # Retrieve events
        events = await clean_event_store.get_events(task_id)
        
        assert len(events) == 2
        assert events[0] == event1
        assert events[1] == event2
    
    @pytest.mark.asyncio
    async def test_get_all_events(self, clean_event_store: InMemoryEventStore):
        """Test retrieving all events across tasks."""
        # Create events for different tasks
        event1 = TaskCreatedEvent.create("task-1", "Goal 1", TaskType.THINK)
        event2 = TaskCreatedEvent.create("task-2", "Goal 2", TaskType.RETRIEVE)
        event3 = TaskStatusChangedEvent.create("task-1", TaskStatus.PENDING, TaskStatus.READY, 1)
        
        await clean_event_store.append(event1)
        await clean_event_store.append(event2)
        await clean_event_store.append(event3)
        
        all_events = await clean_event_store.get_all_events()
        
        assert len(all_events) == 3
        assert event1 in all_events
        assert event2 in all_events
        assert event3 in all_events
    
    @pytest.mark.asyncio
    async def test_get_events_by_type(self, clean_event_store: InMemoryEventStore):
        """Test retrieving events by type."""
        task_id = "test-task"
        
        created_event = TaskCreatedEvent.create(task_id, "Goal", TaskType.THINK)
        status_event = TaskStatusChangedEvent.create(task_id, TaskStatus.PENDING, TaskStatus.READY, 1)
        completed_event = TaskCompletedEvent.create(task_id, "Task completed")
        
        await clean_event_store.append(created_event)
        await clean_event_store.append(status_event)
        await clean_event_store.append(completed_event)
        
        # Get only status change events
        status_events = await clean_event_store.get_events_by_type("task_status_changed")
        assert len(status_events) == 1
        assert status_events[0] == status_event
        
        # Get only created events
        created_events = await clean_event_store.get_events_by_type("task_created")
        assert len(created_events) == 1
        assert created_events[0] == created_event
    
    @pytest.mark.asyncio
    async def test_event_order_preservation(self, clean_event_store: InMemoryEventStore):
        """Test that event order is preserved."""
        task_id = "test-task"
        events = []
        
        # Create events with slight time differences
        for i in range(5):
            event = TaskStatusChangedEvent.create(
                task_id,
                TaskStatus.PENDING,
                TaskStatus.READY, 
                version=i
            )
            events.append(event)
            await clean_event_store.append(event)
            await asyncio.sleep(0.001)  # Small delay to ensure different timestamps
        
        retrieved_events = await clean_event_store.get_events(task_id)
        
        assert len(retrieved_events) == 5
        # Events should be in the same order as appended
        for i, event in enumerate(retrieved_events):
            assert event.version == i


class TestEventStoreFiltering:
    """Test event filtering functionality."""
    
    @pytest.mark.asyncio
    async def test_filter_by_event_type(self, clean_event_store: InMemoryEventStore):
        """Test filtering events by event type."""
        task_id = "test-task"
        
        # Add different types of events
        await clean_event_store.append(TaskCreatedEvent.create(task_id, "Goal", TaskType.THINK))
        await clean_event_store.append(TaskStatusChangedEvent.create(task_id, TaskStatus.PENDING, TaskStatus.READY, 1))
        await clean_event_store.append(TaskCompletedEvent.create(task_id, "Completed"))
        
        # Filter for only status changes
        event_filter = EventFilter(event_type="task_status_changed")
        filtered_events = await clean_event_store.get_events(task_id, event_filter)
        
        assert len(filtered_events) == 1
        assert filtered_events[0].event_type == "task_status_changed"
    
    @pytest.mark.asyncio
    async def test_filter_by_time_range(self, clean_event_store: InMemoryEventStore):
        """Test filtering events by time range."""
        task_id = "test-task"
        base_time = datetime.now(timezone.utc)
        
        # Create events with specific timestamps
        old_event = TaskCreatedEvent.create(task_id, "Old goal", TaskType.THINK)
        old_event = replace(old_event, timestamp=base_time - timedelta(hours=2))
        
        recent_event = TaskStatusChangedEvent.create(task_id, TaskStatus.PENDING, TaskStatus.READY, 1)
        recent_event = replace(recent_event, timestamp=base_time - timedelta(minutes=30))
        
        await clean_event_store.append(old_event)
        await clean_event_store.append(recent_event)
        
        # Filter for events in last hour
        event_filter = EventFilter(
            start_time=base_time - timedelta(hours=1)
        )
        filtered_events = await clean_event_store.get_events(task_id, event_filter)
        
        assert len(filtered_events) == 1
        assert filtered_events[0].event_type == "task_status_changed"
    
    @pytest.mark.asyncio
    async def test_filter_by_metadata(self, clean_event_store: InMemoryEventStore):
        """Test filtering events by metadata."""
        task_id = "test-task"
        
        # Create events with different metadata
        event1 = AtomizerEvaluatedEvent.create(
            task_id, True, NodeType.EXECUTE, "Atomic", 0.9,
            metadata={"agent": "rule_based"}
        )
        event2 = AtomizerEvaluatedEvent.create(
            task_id, False, NodeType.PLAN, "Complex", 0.8,
            metadata={"agent": "agno_based"}
        )
        
        await clean_event_store.append(event1)
        await clean_event_store.append(event2)
        
        # Filter by metadata
        event_filter = EventFilter(
            metadata_filters={"agent": "rule_based"}
        )
        filtered_events = await clean_event_store.get_events(task_id, event_filter)
        
        assert len(filtered_events) == 1
        assert filtered_events[0].metadata["agent"] == "rule_based"
    
    @pytest.mark.asyncio
    async def test_combined_filters(self, clean_event_store: InMemoryEventStore):
        """Test combining multiple filter criteria."""
        task_id = "test-task"
        base_time = datetime.now(timezone.utc)
        
        # Create events with specific properties
        event1 = TaskCreatedEvent.create(task_id, "Goal", TaskType.THINK, metadata={"source": "user"})
        event1 = replace(event1, timestamp=base_time - timedelta(minutes=30))
        
        event2 = TaskStatusChangedEvent.create(
            task_id, TaskStatus.PENDING, TaskStatus.READY, 1,
            metadata={"source": "system"}
        )
        event2 = replace(event2, timestamp=base_time - timedelta(minutes=15))
        
        await clean_event_store.append(event1)
        await clean_event_store.append(event2)
        
        # Filter by type, time, and metadata
        event_filter = EventFilter(
            event_type="task_created",
            start_time=base_time - timedelta(hours=1),
            metadata_filters={"source": "user"}
        )
        filtered_events = await clean_event_store.get_events(task_id, event_filter)
        
        assert len(filtered_events) == 1
        assert filtered_events[0].event_type == "task_created"
        assert filtered_events[0].metadata["source"] == "user"


class TestEventStoreSubscriptions:
    """Test event subscription and notification functionality."""
    
    @pytest.mark.asyncio
    async def test_sync_subscription(self, clean_event_store: InMemoryEventStore, test_event_subscriber):
        """Test synchronous event subscription."""
        # Subscribe to events
        subscription_id = await clean_event_store.subscribe(test_event_subscriber)
        assert subscription_id is not None
        
        # Emit event
        event = TaskCreatedEvent.create("test-task", "Goal", TaskType.THINK)
        await clean_event_store.append(event)
        
        # Check subscriber was notified
        assert test_event_subscriber.call_count == 1
        assert len(test_event_subscriber.received_events) == 1
        assert test_event_subscriber.received_events[0] == event
        
        # Unsubscribe
        unsubscribed = await clean_event_store.unsubscribe(subscription_id)
        assert unsubscribed is True
        
        # Emit another event - should not be received
        await clean_event_store.append(TaskCompletedEvent.create("test-task", "Done"))
        assert test_event_subscriber.call_count == 1  # Still 1
    
    @pytest.mark.asyncio
    async def test_async_subscription(self, clean_event_store: InMemoryEventStore, test_event_subscriber):
        """Test asynchronous event subscription."""
        # Subscribe with async callback
        subscription_id = await clean_event_store.subscribe_async(test_event_subscriber.async_callback)
        
        # Emit event
        event = TaskCreatedEvent.create("test-task", "Goal", TaskType.THINK)
        await clean_event_store.append(event)
        
        # Give async callback time to execute
        await asyncio.sleep(0.01)
        
        # Check subscriber was notified
        assert test_event_subscriber.call_count == 1
        assert len(test_event_subscriber.received_events) == 1
    
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, clean_event_store: InMemoryEventStore):
        """Test multiple event subscribers."""
        subscriber1 = []
        subscriber2 = []
        
        def callback1(event):
            subscriber1.append(event)
        
        def callback2(event):
            subscriber2.append(event)
        
        # Subscribe both callbacks
        await clean_event_store.subscribe(callback1)
        await clean_event_store.subscribe(callback2)
        
        # Emit event
        event = TaskCreatedEvent.create("test-task", "Goal", TaskType.THINK)
        await clean_event_store.append(event)
        
        # Both should receive the event
        assert len(subscriber1) == 1
        assert len(subscriber2) == 1
        assert subscriber1[0] == event
        assert subscriber2[0] == event
    
    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self, clean_event_store: InMemoryEventStore):
        """Test unsubscribing nonexistent subscription."""
        result = await clean_event_store.unsubscribe("nonexistent-id")
        assert result is False


class TestEventStoreStatistics:
    """Test event store statistics and monitoring."""
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(self, clean_event_store: InMemoryEventStore):
        """Test that statistics are tracked correctly."""
        # Initially empty
        stats = await clean_event_store.get_statistics()
        assert stats["total_events"] == 0
        assert len(stats["events_by_type"]) == 0
        
        # Add some events
        await clean_event_store.append(TaskCreatedEvent.create("task-1", "Goal 1", TaskType.THINK))
        await clean_event_store.append(TaskCreatedEvent.create("task-2", "Goal 2", TaskType.RETRIEVE))
        await clean_event_store.append(TaskStatusChangedEvent.create("task-1", TaskStatus.PENDING, TaskStatus.READY, 1))
        
        # Check updated statistics
        stats = await clean_event_store.get_statistics()
        assert stats["total_events"] == 3
        assert stats["events_by_type"]["task_created"] == 2
        assert stats["events_by_type"]["task_status_changed"] == 1
        assert stats["events_by_task_count"] == 2  # Two different tasks
    
    @pytest.mark.asyncio
    async def test_memory_usage_statistics(self, clean_event_store: InMemoryEventStore):
        """Test memory usage statistics."""
        # Add events to different tasks
        for i in range(5):
            await clean_event_store.append(
                TaskCreatedEvent.create(f"task-{i}", f"Goal {i}", TaskType.THINK)
            )
        
        stats = await clean_event_store.get_statistics()
        memory_stats = stats["memory_usage"]
        
        assert memory_stats["tasks_tracked"] == 5
        assert memory_stats["global_events"] == 5
        assert memory_stats["max_events_per_task"] > 0
        assert memory_stats["max_total_events"] > 0


class TestEventStoreTimeline:
    """Test event timeline functionality."""
    
    @pytest.mark.asyncio
    async def test_task_timeline_generation(self, clean_event_store: InMemoryEventStore):
        """Test generating chronological timeline for a task."""
        task_id = "timeline-task"
        
        # Create sequence of events
        await clean_event_store.append(TaskCreatedEvent.create(task_id, "Test goal", TaskType.THINK))
        await clean_event_store.append(TaskStatusChangedEvent.create(task_id, TaskStatus.PENDING, TaskStatus.READY, 1))
        await clean_event_store.append(TaskStatusChangedEvent.create(task_id, TaskStatus.READY, TaskStatus.EXECUTING, 2))
        await clean_event_store.append(TaskCompletedEvent.create(task_id, "Task completed successfully"))
        
        # Get timeline
        timeline = await clean_event_store.get_task_timeline(task_id)
        
        assert len(timeline) == 4
        
        # Check timeline structure
        for item in timeline:
            assert "timestamp" in item
            assert "event_type" in item
            assert "description" in item
            assert "metadata" in item
        
        # Check descriptions are human-readable
        assert "Task created" in timeline[0]["description"]
        assert "Status changed" in timeline[1]["description"]
        assert "completed successfully" in timeline[3]["description"]
    
    @pytest.mark.asyncio
    async def test_timeline_descriptions(self, clean_event_store: InMemoryEventStore):
        """Test timeline event descriptions."""
        task_id = "desc-task"
        
        # Test different event types
        events_and_expected = [
            (TaskCreatedEvent.create(task_id, "Research task", TaskType.THINK), "Task created with goal: Research task"),
            (AtomizerEvaluatedEvent.create(task_id, True, NodeType.EXECUTE, "Simple task", 0.9), "Atomizer determined task is atomic"),
            (TaskFailedEvent.create(task_id, "ValueError", "Invalid input"), "Task failed: Invalid input"),
        ]
        
        for event, expected_desc in events_and_expected:
            await clean_event_store.append(event)
        
        timeline = await clean_event_store.get_task_timeline(task_id)
        
        for i, (_, expected_desc) in enumerate(events_and_expected):
            assert expected_desc in timeline[i]["description"]


class TestEventStoreConcurrency:
    """Test event store thread safety and concurrency."""
    
    @pytest.mark.asyncio
    async def test_concurrent_appends(self, clean_event_store: InMemoryEventStore):
        """Test concurrent event appends."""
        task_count = 10
        events_per_task = 5
        
        async def append_events_for_task(task_index: int):
            task_id = f"task-{task_index}"
            for event_index in range(events_per_task):
                event = TaskCreatedEvent.create(
                    task_id, 
                    f"Goal {event_index}", 
                    TaskType.THINK
                )
                await clean_event_store.append(event)
        
        # Run concurrent appends
        tasks = [append_events_for_task(i) for i in range(task_count)]
        await asyncio.gather(*tasks)
        
        # Verify all events were stored
        stats = await clean_event_store.get_statistics()
        assert stats["total_events"] == task_count * events_per_task
        assert stats["events_by_task_count"] == task_count
        
        # Verify each task has correct number of events
        for i in range(task_count):
            events = await clean_event_store.get_events(f"task-{i}")
            assert len(events) == events_per_task
    
    @pytest.mark.asyncio
    async def test_concurrent_reads_and_writes(self, clean_event_store: InMemoryEventStore):
        """Test concurrent reads and writes."""
        task_id = "concurrent-task"
        
        async def writer():
            for i in range(20):
                await clean_event_store.append(
                    TaskCreatedEvent.create(f"{task_id}-{i}", f"Goal {i}", TaskType.THINK)
                )
                await asyncio.sleep(0.001)
        
        async def reader():
            read_counts = []
            for _ in range(20):
                events = await clean_event_store.get_all_events()
                read_counts.append(len(events))
                await asyncio.sleep(0.001)
            return read_counts
        
        # Run writer and reader concurrently
        writer_task = asyncio.create_task(writer())
        reader_task = asyncio.create_task(reader())
        
        await asyncio.gather(writer_task, reader_task)
        
        # Final count should be 20
        final_events = await clean_event_store.get_all_events()
        assert len(final_events) == 20


class TestEventStoreCleanup:
    """Test event store cleanup functionality."""
    
    @pytest.mark.asyncio
    async def test_clear_specific_task(self, clean_event_store: InMemoryEventStore):
        """Test clearing events for specific task."""
        # Add events for multiple tasks
        await clean_event_store.append(TaskCreatedEvent.create("task-1", "Goal 1", TaskType.THINK))
        await clean_event_store.append(TaskCreatedEvent.create("task-2", "Goal 2", TaskType.RETRIEVE))
        await clean_event_store.append(TaskStatusChangedEvent.create("task-1", TaskStatus.PENDING, TaskStatus.READY, 1))
        
        # Clear events for task-1
        await clean_event_store.clear("task-1")
        
        # task-1 events should be gone, task-2 events should remain
        task1_events = await clean_event_store.get_events("task-1")
        task2_events = await clean_event_store.get_events("task-2")
        
        assert len(task1_events) == 0
        assert len(task2_events) == 1
    
    @pytest.mark.asyncio
    async def test_clear_all_events(self, clean_event_store: InMemoryEventStore):
        """Test clearing all events."""
        # Add some events
        await clean_event_store.append(TaskCreatedEvent.create("task-1", "Goal 1", TaskType.THINK))
        await clean_event_store.append(TaskCreatedEvent.create("task-2", "Goal 2", TaskType.RETRIEVE))
        
        # Clear all
        await clean_event_store.clear()
        
        # All events should be gone
        all_events = await clean_event_store.get_all_events()
        stats = await clean_event_store.get_statistics()
        
        assert len(all_events) == 0
        assert stats["total_events"] == 0


class TestGlobalEventStore:
    """Test global event store functions."""
    
    @pytest.mark.asyncio
    async def test_global_event_store_singleton(self):
        """Test that get_event_store returns singleton."""
        store1 = get_event_store()
        store2 = get_event_store()
        
        assert store1 is store2  # Same instance
    
    @pytest.mark.asyncio
    async def test_emit_event_function(self):
        """Test global emit_event function.""" 
        event = TaskCreatedEvent.create("global-test", "Global goal", TaskType.THINK)
        
        await emit_event(event)
        
        # Check event was stored in global store
        global_store = get_event_store()
        events = await global_store.get_events("global-test")
        
        assert len(events) == 1
        assert events[0] == event