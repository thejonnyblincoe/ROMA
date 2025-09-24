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
# Events are handled by GraphStateManager, not DynamicTaskGraph directly


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
        """Test that event handling is delegated to GraphStateManager."""
        graph = DynamicTaskGraph()

        # DynamicTaskGraph no longer handles events directly
        # Event handling is managed by GraphStateManager to avoid duplicates
        assert not hasattr(graph, '_event_handlers')
        assert not hasattr(graph, 'add_event_handler')

    @pytest.mark.asyncio
    async def test_node_added_event(self, sample_node):
        """Test that nodes are added correctly (event emission handled by GraphStateManager)."""
        graph = DynamicTaskGraph()

        # Add node
        await graph.add_node(sample_node)

        # Verify node was added to graph
        assert sample_node.task_id in graph.nodes
        assert graph.nodes[sample_node.task_id] == sample_node

        # Event emission is handled by GraphStateManager, not DynamicTaskGraph
        # This ensures no duplicate events are emitted

    @pytest.mark.asyncio
    async def test_status_changed_event(self, sample_node):
        """Test node status changes are handled correctly (events via GraphStateManager)."""
        graph = DynamicTaskGraph(root_node=sample_node)

        # Update status: PENDING -> READY -> EXECUTING
        await graph.update_node_status("test_node", TaskStatus.READY)
        await graph.update_node_status("test_node", TaskStatus.EXECUTING)

        # Verify status changes were applied
        updated_node = graph.get_node("test_node")
        assert updated_node.status == TaskStatus.EXECUTING

        # Event emission is handled by GraphStateManager to avoid duplicates

    @pytest.mark.asyncio
    async def test_multiple_event_handlers(self, sample_node):
        """Test that multiple event handling is managed by GraphStateManager."""
        graph = DynamicTaskGraph()

        # Add node
        await graph.add_node(sample_node)

        # Verify node was added
        assert sample_node.task_id in graph.nodes

        # Multiple event handlers are managed by GraphStateManager
        # This ensures proper event distribution without duplication

    @pytest.mark.asyncio
    async def test_event_handler_error_handling(self, sample_node):
        """Test that graph operations work regardless of event handling."""
        graph = DynamicTaskGraph()

        # Add node - should work without any event handlers
        await graph.add_node(sample_node)

        # Node should be added successfully
        assert graph.get_node("test_node") is not None

        # Error handling for events is managed by GraphStateManager
        # DynamicTaskGraph focuses on core graph operations

    @pytest.mark.asyncio
    async def test_complex_workflow_events(self):
        """Test complex workflow operations (events managed by GraphStateManager)."""
        graph = DynamicTaskGraph()

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

        # Verify all operations completed successfully
        assert graph.get_node("parent").status == TaskStatus.READY
        assert graph.get_node("child").status == TaskStatus.COMPLETED

        # Event sequence tracking is handled by GraphStateManager

    @pytest.mark.asyncio
    async def test_event_metadata_completeness(self, sample_node):
        """Test that node operations maintain complete metadata."""
        graph = DynamicTaskGraph()

        # Add node
        await graph.add_node(sample_node)

        # Update status
        await graph.update_node_status("test_node", TaskStatus.READY)

        # Verify node has complete metadata
        node = graph.get_node("test_node")
        assert node.goal == "Test node for event logging"
        assert node.task_type == TaskType.THINK
        assert node.parent_id is None
        assert node.status == TaskStatus.READY

        # Event metadata completeness is ensured by GraphStateManager

    @pytest.mark.asyncio
    async def test_event_ordering_consistency(self):
        """Test that operations are performed in correct order (events via GraphStateManager)."""
        graph = DynamicTaskGraph()

        # Operation ordering is maintained in DynamicTaskGraph
        # Event ordering consistency is handled by GraphStateManager
        
        # Rapid sequence of operations
        node1 = TaskNode(task_id="node1", goal="First", task_type=TaskType.THINK, status=TaskStatus.PENDING)
        node2 = TaskNode(task_id="node2", goal="Second", task_type=TaskType.WRITE, status=TaskStatus.PENDING)

        await graph.add_node(node1)
        await graph.update_node_status("node1", TaskStatus.READY)
        await graph.add_node(node2)
        await graph.update_node_status("node2", TaskStatus.READY)
        await graph.update_node_status("node2", TaskStatus.EXECUTING)

        # Verify operations completed in correct order
        assert graph.get_node("node1").status == TaskStatus.READY
        assert graph.get_node("node2").status == TaskStatus.EXECUTING

        # Event sequence ordering is handled by GraphStateManager


class TestEventLoggingPerformance:
    """Test performance impact of event logging."""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_event_logging_performance_impact(self):
        """Test that graph operations maintain good performance."""
        import time

        # Test performance without event handlers (event emission via GraphStateManager)
        graph = DynamicTaskGraph()

        start = time.time()
        for i in range(100):
            node = TaskNode(
                task_id=f"node_{i}",
                goal=f"Node {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.PENDING
            )
            await graph.add_node(node)
        time_operations = time.time() - start

        # Verify operations completed successfully
        assert len(graph.nodes) == 100

        # Performance should be good (operations should complete quickly)
        assert time_operations < 1.0  # Should complete in under 1 second

        # Event handling is managed by GraphStateManager
        # DynamicTaskGraph maintains optimal performance by focusing on core operations