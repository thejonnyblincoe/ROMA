"""
Unit tests for GraphStateManager.

Following TDD principles - these tests define the interface and expected behavior
before implementation.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from roma.application.services.event_store import InMemoryEventStore
from roma.domain.entities.task_node import TaskNode
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class TestGraphStateManager:
    """Test GraphStateManager core functionality."""

    @pytest.fixture
    def sample_task_nodes(self) -> list[TaskNode]:
        """Create sample task nodes for testing."""
        return [
            TaskNode(
                goal="Root task - research cryptocurrency",
                task_type=TaskType.THINK,
                node_type=NodeType.PLAN,
                status=TaskStatus.PENDING
            ),
            TaskNode(
                goal="Retrieve Bitcoin price data",
                task_type=TaskType.RETRIEVE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            ),
            TaskNode(
                goal="Analyze market trends",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            ),
        ]

    @pytest.fixture
    def mock_event_store(self) -> AsyncMock:
        """Create mock event store for testing."""
        mock_store = AsyncMock(spec=InMemoryEventStore)
        mock_store.append = AsyncMock()
        mock_store.get_events_by_task_id = AsyncMock(return_value=[])
        return mock_store

    @pytest_asyncio.fixture
    async def sample_graph(self, sample_task_nodes) -> DynamicTaskGraph:
        """Create sample graph with nodes for testing."""
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        for node in sample_task_nodes[1:]:
            await graph.add_node(node)
        return graph

    async def create_sample_graph(self, sample_task_nodes):
        """Create sample graph with nodes."""
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        for node in sample_task_nodes[1:]:
            await graph.add_node(node)
        return graph

    def test_graph_state_manager_init(self, mock_event_store):
        """Test GraphStateManager initialization."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        graph = DynamicTaskGraph()
        manager = GraphStateManager(graph=graph, event_store=mock_event_store)

        assert manager.graph == graph
        assert manager.event_store == mock_event_store
        assert manager.version == 0
        assert manager.is_locked is False

    @pytest.mark.asyncio
    async def test_transition_node_status_success(self, mock_event_store, sample_task_nodes):
        """Test successful node status transition."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        sample_graph = await self.create_sample_graph(sample_task_nodes)
        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        node_id = sample_task_nodes[0].task_id
        old_status = sample_task_nodes[0].status
        new_status = TaskStatus.READY

        updated_node = await manager.transition_node_status(node_id, new_status)

        # Verify node was updated
        assert updated_node.status == new_status
        assert updated_node.task_id == node_id
        assert updated_node.version == sample_task_nodes[0].version + 1

        # Verify event was stored
        mock_event_store.append.assert_called_once()
        event_call = mock_event_store.append.call_args[0][0]
        assert event_call.task_id == node_id
        assert event_call.old_status == old_status
        assert event_call.new_status == new_status

        # Verify manager version was incremented
        assert manager.version == 1

    @pytest.mark.asyncio
    async def test_transition_node_status_invalid_node(self, mock_event_store, sample_task_nodes):
        """Test transition with invalid node ID."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        sample_graph = await self.create_sample_graph(sample_task_nodes)
        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        with pytest.raises(ValueError, match="Task node .* not found in graph"):
            await manager.transition_node_status("invalid-id", TaskStatus.READY)

    @pytest.mark.asyncio
    async def test_transition_node_status_invalid_transition(self, sample_graph, mock_event_store, sample_task_nodes):
        """Test transition with invalid status transition."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        node_id = sample_task_nodes[0].task_id

        # Try invalid transition PENDING -> COMPLETED (should go through READY, EXECUTING first)
        with pytest.raises(ValueError, match="Invalid transition"):
            await manager.transition_node_status(node_id, TaskStatus.COMPLETED)

    @pytest.mark.asyncio
    async def test_transition_node_status_concurrent_access(self, sample_graph, mock_event_store, sample_task_nodes):
        """Test concurrent node status transitions are serialized."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        node_id = sample_task_nodes[0].task_id

        # Create concurrent transition tasks
        tasks = [
            manager.transition_node_status(node_id, TaskStatus.READY),
            manager.transition_node_status(node_id, TaskStatus.READY),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # One should succeed, one should fail with concurrent modification
        success_count = sum(1 for r in results if isinstance(r, TaskNode))
        error_count = sum(1 for r in results if isinstance(r, Exception))

        assert success_count == 1
        assert error_count == 1
        assert manager.version == 1

    @pytest.mark.asyncio
    async def test_add_node_with_events(self, sample_graph, mock_event_store):
        """Test adding node with event emission."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        new_node = TaskNode(
            goal="New task for testing",
            task_type=TaskType.WRITE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )

        await manager.add_node(new_node)

        # Verify node was added to graph
        assert manager.graph.get_node(new_node.task_id) == new_node

        # Verify event was emitted (may be called multiple times by different layers)
        assert mock_event_store.append.call_count >= 1
        event_call = mock_event_store.append.call_args[0][0]
        assert event_call.task_id == new_node.task_id

        # Verify manager version was incremented
        assert manager.version == 1

    @pytest.mark.asyncio
    async def test_add_node_with_dependencies(self, sample_graph, mock_event_store, sample_task_nodes):
        """Test adding node with dependency tracking."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        parent_id = sample_task_nodes[0].task_id
        child_node = TaskNode(
            goal="Child task",
            task_type=TaskType.WRITE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING,
            parent_id=parent_id
        )

        await manager.add_node(child_node)

        # Verify dependency was created
        children_ids = manager.graph.get_children(parent_id)
        assert len(children_ids) == 1
        assert children_ids[0] == child_node.task_id

    def test_get_ready_nodes(self, sample_graph, mock_event_store):
        """Test getting ready nodes."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        ready_nodes = manager.get_ready_nodes()

        # All nodes without dependencies should be ready initially
        assert len(ready_nodes) >= 1
        assert ready_nodes[0].status == TaskStatus.PENDING

    def test_get_node_by_id(self, sample_graph, mock_event_store, sample_task_nodes):
        """Test getting node by ID."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        node_id = sample_task_nodes[0].task_id
        retrieved_node = manager.get_node_by_id(node_id)

        assert retrieved_node == sample_task_nodes[0]
        assert retrieved_node.task_id == node_id

    def test_get_node_by_id_not_found(self, sample_graph, mock_event_store):
        """Test getting nonexistent node by ID."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        result = manager.get_node_by_id("nonexistent-id")
        assert result is None

    def test_get_all_nodes(self, sample_graph, mock_event_store, sample_task_nodes):
        """Test getting all nodes."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        all_nodes = manager.get_all_nodes()

        assert len(all_nodes) == len(sample_task_nodes)
        node_ids = {node.task_id for node in all_nodes}
        expected_ids = {node.task_id for node in sample_task_nodes}
        assert node_ids == expected_ids

    def test_has_cycles(self, sample_graph, mock_event_store):
        """Test cycle detection."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        # Linear graph should have no cycles
        assert manager.has_cycles() is False

    @pytest.mark.asyncio
    async def test_get_execution_statistics(self, sample_graph, mock_event_store, sample_task_nodes):
        """Test getting execution statistics."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        manager = GraphStateManager(graph=sample_graph, event_store=mock_event_store)

        # Transition some nodes
        await manager.transition_node_status(sample_task_nodes[0].task_id, TaskStatus.READY)
        await manager.transition_node_status(sample_task_nodes[0].task_id, TaskStatus.EXECUTING)

        stats = manager.get_execution_statistics()

        assert "total_nodes" in stats
        assert "pending_nodes" in stats
        assert "ready_nodes" in stats
        assert "executing_nodes" in stats
        assert "completed_nodes" in stats
        assert "failed_nodes" in stats
        assert "version" in stats

        assert stats["total_nodes"] == len(sample_task_nodes)
        assert stats["executing_nodes"] == 1
        assert stats["version"] == manager.version


class TestGraphStateManagerConcurrency:
    """Test GraphStateManager concurrency and thread safety."""

    @pytest.fixture
    def mock_event_store(self) -> AsyncMock:
        """Create mock event store for testing."""
        mock_store = AsyncMock(spec=InMemoryEventStore)
        mock_store.append = AsyncMock()
        return mock_store

    @pytest.mark.asyncio
    async def test_high_concurrency_node_transitions(self, mock_event_store):
        """Test high concurrency node status transitions."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        graph = DynamicTaskGraph()
        manager = GraphStateManager(graph=graph, event_store=mock_event_store)

        # Create 50 nodes
        nodes = []
        for i in range(50):
            node = TaskNode(
                goal=f"Concurrent task {i}",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            )
            nodes.append(node)
            await manager.add_node(node)

        # Transition all nodes to READY concurrently
        tasks = [
            manager.transition_node_status(node.task_id, TaskStatus.READY)
            for node in nodes
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All transitions should succeed
        success_count = sum(1 for r in results if isinstance(r, TaskNode))
        assert success_count == 50

        # Verify all nodes are READY
        all_nodes = manager.get_all_nodes()
        ready_count = sum(1 for node in all_nodes if node.status == TaskStatus.READY)
        assert ready_count == 50

        # Version should be 50 (initial adds) + 50 (status transitions) = 100
        assert manager.version == 100

    @pytest.mark.asyncio
    async def test_concurrent_add_and_transition(self, mock_event_store):
        """Test concurrent node addition and status transitions."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        graph = DynamicTaskGraph()
        manager = GraphStateManager(graph=graph, event_store=mock_event_store)

        # Create initial node
        root_node = TaskNode(
            goal="Root task",
            task_type=TaskType.THINK,
            node_type=NodeType.PLAN,
            status=TaskStatus.PENDING
        )
        await manager.add_node(root_node)

        # Concurrent operations: add new nodes and transition existing ones
        add_tasks = []
        transition_tasks = []

        for i in range(25):
            # Add new node
            new_node = TaskNode(
                goal=f"New task {i}",
                task_type=TaskType.WRITE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            )
            add_tasks.append(manager.add_node(new_node))

        # Transition root node through states
        transition_tasks = [
            manager.transition_node_status(root_node.task_id, TaskStatus.READY),
            # Note: Need to be careful about sequential dependencies in real concurrent test
        ]

        # Execute all operations concurrently
        all_tasks = add_tasks + transition_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Most operations should succeed
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count >= 25  # At least the adds should succeed

        # Verify final state consistency
        all_nodes = manager.get_all_nodes()
        assert len(all_nodes) >= 26  # 1 root + 25 added nodes

        # Verify no data corruption
        for node in all_nodes:
            assert node.task_id is not None
            assert node.goal is not None
            assert isinstance(node.status, TaskStatus)


class TestGraphStateManagerEventIntegration:
    """Test GraphStateManager integration with event system."""

    @pytest.fixture
    def real_event_store(self) -> InMemoryEventStore:
        """Create real event store for integration testing."""
        return InMemoryEventStore(max_total_events=1000)

    @pytest.mark.asyncio
    async def test_event_emission_on_state_changes(self, real_event_store):
        """Test that events are emitted for all state changes."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        graph = DynamicTaskGraph()
        manager = GraphStateManager(graph=graph, event_store=real_event_store)

        # Add node
        node = TaskNode(
            goal="Test task",
            task_type=TaskType.THINK,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )

        await manager.add_node(node)

        # Transition through states
        await manager.transition_node_status(node.task_id, TaskStatus.READY)
        await manager.transition_node_status(node.task_id, TaskStatus.EXECUTING)
        await manager.transition_node_status(node.task_id, TaskStatus.COMPLETED)

        # Verify events were stored
        events = await real_event_store.get_events_by_task_id(node.task_id)

        # Should have: NodeAdded + 3 StatusChanged events
        assert len(events) >= 4

        # Verify event types and sequence
        event_types = [type(event).__name__ for event in events]
        assert "TaskNodeAddedEvent" in event_types
        assert event_types.count("TaskStatusChangedEvent") == 3

    @pytest.mark.asyncio
    async def test_event_timeline_generation(self, real_event_store):
        """Test event timeline generation from state changes."""
        from roma.application.orchestration.graph_state_manager import GraphStateManager

        graph = DynamicTaskGraph()
        manager = GraphStateManager(graph=graph, event_store=real_event_store)

        # Create and manage a node
        node = TaskNode(
            goal="Timeline test task",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )

        await manager.add_node(node)
        await manager.transition_node_status(node.task_id, TaskStatus.READY)
        await manager.transition_node_status(node.task_id, TaskStatus.EXECUTING)
        await manager.transition_node_status(node.task_id, TaskStatus.COMPLETED)

        # Generate timeline
        timeline = await real_event_store.generate_timeline()

        assert len(timeline) >= 4

        # Verify timeline is chronologically ordered
        timestamps = [entry["timestamp"] for entry in timeline]
        assert timestamps == sorted(timestamps)
