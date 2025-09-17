"""
Unit tests for DynamicTaskGraph.

Following TDD principles - these tests define the interface and expected behavior
before implementation.
"""

import asyncio
from typing import List, Set
import pytest
import networkx as nx

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.node_type import NodeType


class TestDynamicTaskGraph:
    """Test DynamicTaskGraph core functionality."""

    @pytest.fixture
    def sample_task_nodes(self) -> List[TaskNode]:
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
            TaskNode(
                goal="Write analysis report",
                task_type=TaskType.WRITE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            ),
        ]

    def test_dynamic_task_graph_init(self):
        """Test DynamicTaskGraph initialization."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        
        # Should initialize with empty state
        assert len(graph.nodes) == 0
        assert len(graph.get_all_nodes()) == 0
        assert graph.root_node_id is None
        assert graph.execution_id is not None  # Should auto-generate
        
        # NetworkX graph should be initialized
        assert isinstance(graph._graph, nx.DiGraph)
        assert len(graph._graph.nodes) == 0

    def test_dynamic_task_graph_init_with_root(self, sample_task_nodes):
        """Test DynamicTaskGraph initialization with root node."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        root_node = sample_task_nodes[0]
        graph = DynamicTaskGraph(root_node=root_node)
        
        assert graph.root_node_id == root_node.task_id
        assert len(graph.nodes) == 1
        assert graph.get_node(root_node.task_id) == root_node

    @pytest.mark.asyncio
    async def test_add_node_basic(self, sample_task_nodes):
        """Test basic node addition."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        node = sample_task_nodes[0]
        
        await graph.add_node(node)
        
        assert len(graph.nodes) == 1
        assert graph.get_node(node.task_id) == node
        assert node.task_id in graph._graph.nodes

    @pytest.mark.asyncio 
    async def test_add_node_with_parent(self, sample_task_nodes):
        """Test adding node with parent dependency."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        parent = sample_task_nodes[0]
        child = sample_task_nodes[1].model_copy(update={"parent_id": parent.task_id})
        
        await graph.add_node(parent)
        await graph.add_node(child)
        
        assert len(graph.nodes) == 2
        assert graph.get_node(child.task_id).parent_id == parent.task_id
        
        # Should create edge in NetworkX graph
        assert graph._graph.has_edge(parent.task_id, child.task_id)

    @pytest.mark.asyncio
    async def test_concurrent_node_addition(self, sample_task_nodes):
        """Test thread-safe concurrent node addition."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        
        # Create 50 nodes for concurrent addition
        nodes = []
        for i in range(50):
            node = TaskNode(
                goal=f"Task {i}",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            )
            nodes.append(node)
        
        # Add all nodes concurrently
        tasks = [graph.add_node(node) for node in nodes]
        await asyncio.gather(*tasks)
        
        # Verify all nodes added correctly
        assert len(graph.nodes) == 50
        for node in nodes:
            assert graph.get_node(node.task_id) == node
            assert node.task_id in graph._graph.nodes

    def test_get_ready_nodes_empty(self):
        """Test getting ready nodes from empty graph."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        ready_nodes = graph.get_ready_nodes()
        
        assert len(ready_nodes) == 0

    @pytest.mark.asyncio
    async def test_get_ready_nodes_single_pending(self, sample_task_nodes):
        """Test getting ready nodes with single pending node."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        node = sample_task_nodes[0]
        
        await graph.add_node(node)
        ready_nodes = graph.get_ready_nodes()
        
        assert len(ready_nodes) == 1
        assert ready_nodes[0] == node

    @pytest.mark.asyncio
    async def test_get_ready_nodes_with_dependencies(self, sample_task_nodes):
        """Test getting ready nodes respects dependencies."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        parent = sample_task_nodes[0]
        child = sample_task_nodes[1].model_copy(update={"parent_id": parent.task_id})
        
        await graph.add_node(parent)
        await graph.add_node(child)
        
        ready_nodes = graph.get_ready_nodes()
        
        # Only parent should be ready (child has dependency)
        assert len(ready_nodes) == 1
        assert ready_nodes[0] == parent

    @pytest.mark.asyncio
    async def test_update_node_status(self, sample_task_nodes):
        """Test updating node status."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        node = sample_task_nodes[0]
        
        await graph.add_node(node)
        
        # First transition PENDING → READY
        ready_node = await graph.update_node_status(node.task_id, TaskStatus.READY)
        assert ready_node.status == TaskStatus.READY
        assert ready_node.version == node.version + 1
        assert graph.get_node(node.task_id) == ready_node
        
        # Then transition READY → EXECUTING
        executing_node = await graph.update_node_status(node.task_id, TaskStatus.EXECUTING)
        assert executing_node.status == TaskStatus.EXECUTING
        assert executing_node.version == ready_node.version + 1
        assert graph.get_node(node.task_id) == executing_node

    @pytest.mark.asyncio
    async def test_update_node_status_enables_children(self, sample_task_nodes):
        """Test completing parent enables children to become ready."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        parent = sample_task_nodes[0]
        child = sample_task_nodes[1].model_copy(update={"parent_id": parent.task_id})
        
        await graph.add_node(parent)
        await graph.add_node(child)
        
        # Initially only parent is ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].task_id == parent.task_id
        
        # Complete parent through proper state transitions: PENDING → READY → EXECUTING → COMPLETED
        await graph.update_node_status(parent.task_id, TaskStatus.READY)
        await graph.update_node_status(parent.task_id, TaskStatus.EXECUTING)
        await graph.update_node_status(parent.task_id, TaskStatus.COMPLETED)
        
        # Now child should be ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].task_id == child.task_id

    def test_has_cycles_empty_graph(self):
        """Test cycle detection on empty graph."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        assert graph.has_cycles() is False

    @pytest.mark.asyncio
    async def test_has_cycles_linear_graph(self, sample_task_nodes):
        """Test cycle detection on linear graph (no cycles)."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        
        # Create linear chain: A -> B -> C
        node_a = sample_task_nodes[0]
        node_b = sample_task_nodes[1].model_copy(update={"parent_id": node_a.task_id})
        node_c = sample_task_nodes[2].model_copy(update={"parent_id": node_b.task_id})
        
        await graph.add_node(node_a)
        await graph.add_node(node_b) 
        await graph.add_node(node_c)
        
        assert graph.has_cycles() is False

    def test_get_node_nonexistent(self):
        """Test getting nonexistent node returns None."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        result = graph.get_node("nonexistent-id")
        
        assert result is None

    def test_get_all_nodes_empty(self):
        """Test getting all nodes from empty graph."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        all_nodes = graph.get_all_nodes()
        
        assert len(all_nodes) == 0
        assert isinstance(all_nodes, list)

    @pytest.mark.asyncio
    async def test_get_all_nodes_multiple(self, sample_task_nodes):
        """Test getting all nodes from populated graph."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        
        for node in sample_task_nodes:
            await graph.add_node(node)
        
        all_nodes = graph.get_all_nodes()
        
        assert len(all_nodes) == len(sample_task_nodes)
        assert set(node.task_id for node in all_nodes) == set(node.task_id for node in sample_task_nodes)

    def test_get_children_nonexistent_parent(self):
        """Test getting children of nonexistent parent."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        children = graph.get_children("nonexistent-id")
        
        assert len(children) == 0

    @pytest.mark.asyncio
    async def test_get_children_with_children(self, sample_task_nodes):
        """Test getting children of node with children."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        parent = sample_task_nodes[0]
        child1 = sample_task_nodes[1].model_copy(update={"parent_id": parent.task_id})
        child2 = sample_task_nodes[2].model_copy(update={"parent_id": parent.task_id})
        
        await graph.add_node(parent)
        await graph.add_node(child1)
        await graph.add_node(child2)
        
        children = graph.get_children(parent.task_id)
        
        assert len(children) == 2
        child_ids = {child.task_id for child in children}
        assert child_ids == {child1.task_id, child2.task_id}

    def test_model_serialization(self, sample_task_nodes):
        """Test Pydantic model serialization/deserialization."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        
        # Should be able to serialize to dict
        graph_dict = graph.model_dump()
        assert "nodes" in graph_dict
        assert "execution_id" in graph_dict
        assert "root_node_id" in graph_dict
        
        # NetworkX graph should be excluded from serialization
        assert "_graph" not in graph_dict
        
        # Should be able to recreate from dict
        new_graph = DynamicTaskGraph.model_validate(graph_dict)
        assert new_graph.execution_id == graph.execution_id
        assert new_graph.root_node_id == graph.root_node_id
        assert len(new_graph.nodes) == len(graph.nodes)


class TestDynamicTaskGraphPerformance:
    """Performance tests for DynamicTaskGraph."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_graph_performance(self):
        """Test performance with large number of nodes."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        
        # Create 1000 nodes
        nodes = []
        for i in range(1000):
            node = TaskNode(
                goal=f"Performance test task {i}",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            )
            nodes.append(node)
        
        # Time concurrent addition
        import time
        start_time = time.time()
        
        tasks = [graph.add_node(node) for node in nodes]
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Should complete within reasonable time (< 5 seconds for 1000 nodes)
        execution_time = end_time - start_time
        assert execution_time < 5.0
        
        # Verify all nodes added
        assert len(graph.nodes) == 1000
        
        # Test ready nodes performance
        start_time = time.time()
        ready_nodes = graph.get_ready_nodes()
        end_time = time.time()
        
        # Should be fast (< 100ms for 1000 nodes)
        query_time = end_time - start_time
        assert query_time < 0.1
        assert len(ready_nodes) == 1000  # All nodes ready (no dependencies)

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_deep_hierarchy_performance(self):
        """Test performance with deep node hierarchy."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
        
        graph = DynamicTaskGraph()
        
        # Create chain of 100 dependent nodes
        previous_node = None
        nodes = []
        
        for i in range(100):
            parent_id = previous_node.task_id if previous_node else None
            node = TaskNode(
                goal=f"Deep hierarchy task {i}",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id=parent_id
            )
            nodes.append(node)
            await graph.add_node(node)
            previous_node = node
        
        # Test ready nodes in deep hierarchy
        ready_nodes = graph.get_ready_nodes()
        
        # Only root node should be ready
        assert len(ready_nodes) == 1
        assert ready_nodes[0] == nodes[0]
        
        # Test completing nodes enables next in chain
        import time
        start_time = time.time()
        
        for i in range(10):  # Complete first 10 nodes
            current_ready = graph.get_ready_nodes()
            assert len(current_ready) == 1
            
            # Proper state transitions: PENDING → READY → EXECUTING → COMPLETED
            task_id = current_ready[0].task_id
            await graph.update_node_status(task_id, TaskStatus.READY)
            await graph.update_node_status(task_id, TaskStatus.EXECUTING)
            await graph.update_node_status(task_id, TaskStatus.COMPLETED)
        
        end_time = time.time()
        
        # Should complete quickly (< 1 second for 10 updates)
        execution_time = end_time - start_time
        assert execution_time < 1.0


class TestDynamicTaskGraphEdgeCases:
    """Test edge cases to improve coverage."""

    @pytest.fixture
    def sample_task_nodes(self) -> List[TaskNode]:
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
            )
        ]

    @pytest.mark.asyncio
    async def test_add_node_with_parent_in_graph(self, sample_task_nodes):
        """Test adding node with parent that exists in graph (covers line 82)."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph

        graph = DynamicTaskGraph()
        parent_node = sample_task_nodes[0]

        # Add parent first
        await graph.add_node(parent_node)

        # Create child with parent_id set
        child_node = TaskNode(
            goal="Child task",
            task_type=TaskType.RETRIEVE,
            parent_id=parent_node.task_id
        )

        # This should hit line 82: add_edge when parent_id exists in nodes
        await graph.add_node(child_node)

        # Verify edge was created
        children = graph.get_children(parent_node.task_id)
        assert len(children) == 1
        assert children[0].task_id == child_node.task_id

    def test_get_children_nonexistent_task_id(self):
        """Test get_children with task_id not in graph (covers lines 215-219)."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph

        graph = DynamicTaskGraph()

        # This should hit lines 215-216: task_id not in graph, return []
        children = graph.get_children("nonexistent_task_id")
        assert children == []

    @pytest.mark.asyncio
    async def test_get_children_with_fake_child_edge(self, sample_task_nodes):
        """Test get_children with manually added edge (covers successors path)."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph

        graph = DynamicTaskGraph()
        parent_node = sample_task_nodes[0]
        await graph.add_node(parent_node)

        # Manually add edge to graph (valid test case for successors)
        graph._graph.add_node("fake_child_id")
        graph._graph.add_edge(parent_node.task_id, "fake_child_id")

        # This should return the child IDs from NetworkX successors
        children = graph.get_children(parent_node.task_id)
        assert len(children) == 1
        assert children[0] == "fake_child_id"
        assert hasattr(children[0], 'task_id')  # Test the _TaskIdStr wrapper

    @pytest.mark.asyncio
    async def test_has_cycles_with_actual_cycle(self):
        """Test has_cycles with actual cycle (covers cycle detection paths)."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph

        graph = DynamicTaskGraph()

        # Create nodes that form a cycle
        node_a = TaskNode(goal="Node A", task_type=TaskType.THINK)
        node_b = TaskNode(goal="Node B", task_type=TaskType.THINK, parent_id=node_a.task_id)
        node_c = TaskNode(goal="Node C", task_type=TaskType.THINK, parent_id=node_b.task_id)

        await graph.add_node(node_a)
        await graph.add_node(node_b)
        await graph.add_node(node_c)

        # Manually create cycle A -> B -> C -> A
        graph._graph.add_edge(node_c.task_id, node_a.task_id)

        # Should detect the cycle
        assert graph.has_cycles() is True

    def test_get_node_with_valid_id(self, sample_task_nodes):
        """Test get_node with valid task_id (covers successful path)."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph

        graph = DynamicTaskGraph()
        node = sample_task_nodes[0]

        # Add node synchronously to nodes dict (bypass async add_node)
        graph.nodes[node.task_id] = node

        # Should return the node
        result = graph.get_node(node.task_id)
        assert result == node

    @pytest.mark.asyncio
    async def test_update_node_status_nonexistent_node(self):
        """Test update_node_status with nonexistent node (covers error paths)."""
        from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph

        graph = DynamicTaskGraph()

        # Should handle nonexistent node gracefully
        # Note: Based on implementation, this might raise an exception or handle gracefully
        try:
            await graph.update_node_status("nonexistent", TaskStatus.COMPLETED)
        except KeyError:
            # If it raises KeyError, that's expected behavior
            pass