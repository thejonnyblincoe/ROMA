"""
Tests for Graph Event Logging (Task 1.2.5)

Testing event emission and handling in DynamicTaskGraph operations.
"""

import pytest
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_type import NodeType
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.events.task_events import BaseTaskEvent


class TestGraphEventLogging:
    """Test event logging capabilities in DynamicTaskGraph."""

    @pytest.fixture
    def sample_node(self):
        """Create sample task node."""
        return TaskNode(
            task_id="test_node",
            goal="Test node for event logging",
            task_type=TaskType.THINK,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )

    def test_event_handler_registration(self):
        """Test adding and removing event handlers."""
        graph = DynamicTaskGraph()
        
        # Mock handler
        events_received = []
        def test_handler(event):
            events_received.append(event)
        
        # Add handler
        graph.add_event_handler(test_handler)
        assert len(graph._event_handlers) == 1
        
        # Remove handler
        graph.remove_event_handler(test_handler)
        assert len(graph._event_handlers) == 0

    @pytest.mark.asyncio
    async def test_node_added_event(self, sample_node):
        """Test event emission when node is added."""
        graph = DynamicTaskGraph()
        
        # Setup event collector
        events_received = []
        async def event_collector(event):
            events_received.append(event)
        
        graph.add_event_handler(event_collector)
        
        # Add node
        await graph.add_node(sample_node)
        
        # Check event was emitted
        assert len(events_received) == 1
        event = events_received[0]
        
        assert event.event_type == "task_created"
        assert event.task_id == "test_node"
        assert event.goal == "Test node for event logging"
        assert event.task_type == TaskType.THINK
        assert event.metadata["status"] == "PENDING"
        assert event.parent_id is None

    @pytest.mark.asyncio
    async def test_status_changed_event(self, sample_node):
        """Test event emission when node status changes."""
        graph = DynamicTaskGraph(root_node=sample_node)
        
        # Setup event collector
        events_received = []
        async def event_collector(event):
            events_received.append(event)
        
        graph.add_event_handler(event_collector)
        
        # Update status: PENDING -> READY -> EXECUTING
        await graph.update_node_status("test_node", TaskStatus.READY)
        await graph.update_node_status("test_node", TaskStatus.EXECUTING)
        
        # Check event was emitted (skip the root node addition event)
        status_events = [e for e in events_received if e.event_type == "task_status_changed"]
        assert len(status_events) == 2  # PENDING->READY and READY->EXECUTING
        
        # Check the final transition to EXECUTING
        final_event = status_events[1]
        assert final_event.event_type == "task_status_changed"
        assert final_event.task_id == "test_node"
        assert final_event.old_status == TaskStatus.READY
        assert final_event.new_status == TaskStatus.EXECUTING
        assert final_event.metadata["goal"] == "Test node for event logging"

    @pytest.mark.asyncio
    async def test_multiple_event_handlers(self, sample_node):
        """Test multiple event handlers receive events."""
        graph = DynamicTaskGraph()
        
        # Setup multiple collectors
        events_a = []
        events_b = []
        
        async def collector_a(event):
            events_a.append(event)
        
        def collector_b(event):  # Sync handler
            events_b.append(event)
        
        graph.add_event_handler(collector_a)
        graph.add_event_handler(collector_b)
        
        # Add node
        await graph.add_node(sample_node)
        
        # Both handlers should receive event
        assert len(events_a) == 1
        assert len(events_b) == 1
        assert events_a[0].event_type == "task_created"
        assert events_b[0].event_type == "task_created"

    @pytest.mark.asyncio
    async def test_event_handler_error_handling(self, sample_node):
        """Test that handler errors don't break graph operations."""
        graph = DynamicTaskGraph()
        
        # Handler that throws error
        async def failing_handler(event):  # noqa: ARG001
            raise ValueError("Handler error")
        
        # Working handler
        events_received = []
        async def working_handler(event):
            events_received.append(event)
        
        graph.add_event_handler(failing_handler)
        graph.add_event_handler(working_handler)
        
        # Add node - should work despite failing handler
        await graph.add_node(sample_node)
        
        # Working handler should still receive event
        assert len(events_received) == 1
        assert events_received[0].event_type == "task_created"
        
        # Node should be added successfully
        assert graph.get_node("test_node") is not None

    @pytest.mark.asyncio
    async def test_complex_workflow_events(self):
        """Test event sequence for complex workflow."""
        graph = DynamicTaskGraph()
        
        # Setup event collector
        events_received = []
        async def event_collector(event):
            events_received.append(event)
        
        graph.add_event_handler(event_collector)
        
        # Create parent node
        parent = TaskNode(
            task_id="parent",
            goal="Parent task",
            task_type=TaskType.THINK,
            node_type=NodeType.PLAN,
            status=TaskStatus.PENDING
        )
        
        # Create child node
        child = TaskNode(
            task_id="child",
            goal="Child task",
            task_type=TaskType.WRITE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING,
            parent_id="parent"
        )
        
        # Add nodes
        await graph.add_node(parent)
        await graph.add_node(child)
        
        # Update statuses with valid transitions
        await graph.update_node_status("parent", TaskStatus.READY)
        await graph.update_node_status("child", TaskStatus.READY)
        await graph.update_node_status("child", TaskStatus.EXECUTING)
        await graph.update_node_status("child", TaskStatus.COMPLETED)
        
        # Verify event sequence - 2 creations + 4 status changes
        assert len(events_received) == 6
        
        # Check event types in order
        expected_types = ["task_created", "task_created", "task_status_changed", "task_status_changed", "task_status_changed", "task_status_changed"]
        actual_types = [e.event_type for e in events_received]
        assert actual_types == expected_types
        
        # Check specific events
        assert events_received[0].task_id == "parent"
        assert events_received[1].task_id == "child"
        assert events_received[2].task_id == "parent"
        assert events_received[2].new_status == TaskStatus.READY
        assert events_received[5].new_status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_event_metadata_completeness(self, sample_node):
        """Test that events contain complete metadata."""
        graph = DynamicTaskGraph()
        
        events_received = []
        async def event_collector(event):
            events_received.append(event)
        
        graph.add_event_handler(event_collector)
        
        # Add node
        await graph.add_node(sample_node)
        
        # Update status
        await graph.update_node_status("test_node", TaskStatus.READY)
        
        # Check task_created event metadata
        add_event = events_received[0]
        assert add_event.goal == "Test node for event logging"
        assert add_event.task_type == TaskType.THINK
        assert add_event.parent_id is None
        assert "status" in add_event.metadata
        assert add_event.timestamp is not None
        
        # Check task_status_changed event metadata  
        status_event = events_received[1]
        assert status_event.old_status == TaskStatus.PENDING
        assert status_event.new_status == TaskStatus.READY
        assert status_event.version > 0
        assert "goal" in status_event.metadata
        assert status_event.timestamp is not None

    @pytest.mark.asyncio
    async def test_event_ordering_consistency(self):
        """Test that events are emitted in correct order."""
        graph = DynamicTaskGraph()
        
        events_received = []
        async def event_collector(event):
            events_received.append((event.event_type, event.task_id, event.timestamp))
        
        graph.add_event_handler(event_collector)
        
        # Rapid sequence of operations
        node1 = TaskNode(task_id="node1", goal="First", task_type=TaskType.THINK, status=TaskStatus.PENDING)
        node2 = TaskNode(task_id="node2", goal="Second", task_type=TaskType.WRITE, status=TaskStatus.PENDING)
        
        await graph.add_node(node1)
        await graph.update_node_status("node1", TaskStatus.READY)
        await graph.add_node(node2)
        await graph.update_node_status("node2", TaskStatus.READY)
        await graph.update_node_status("node2", TaskStatus.EXECUTING)
        
        # Events should be in chronological order
        timestamps = [event[2] for event in events_received]
        assert timestamps == sorted(timestamps)
        
        # Event sequence should be correct
        expected_sequence = [
            ("task_created", "node1"),
            ("task_status_changed", "node1"), 
            ("task_created", "node2"),
            ("task_status_changed", "node2"),
            ("task_status_changed", "node2")
        ]
        actual_sequence = [(event[0], event[1]) for event in events_received]
        assert actual_sequence == expected_sequence


class TestEventLoggingPerformance:
    """Test performance impact of event logging."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_event_logging_performance_impact(self):
        """Test that event logging doesn't significantly impact performance."""
        import time
        
        # Test without event handlers
        graph_no_events = DynamicTaskGraph()
        
        start = time.time()
        for i in range(100):
            node = TaskNode(
                task_id=f"node_{i}",
                goal=f"Node {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.PENDING
            )
            await graph_no_events.add_node(node)
        time_no_events = time.time() - start
        
        # Test with event handlers
        graph_with_events = DynamicTaskGraph()
        
        events_received = []
        async def fast_handler(event):
            events_received.append(event.task_id)
        
        graph_with_events.add_event_handler(fast_handler)
        
        start = time.time()
        for i in range(100):
            node = TaskNode(
                task_id=f"node_{i}",
                goal=f"Node {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.PENDING
            )
            await graph_with_events.add_node(node)
        time_with_events = time.time() - start
        
        # Event logging should add minimal overhead (< 50% increase)
        overhead_ratio = time_with_events / time_no_events
        assert overhead_ratio < 1.5
        
        # All events should be received
        assert len(events_received) == 100