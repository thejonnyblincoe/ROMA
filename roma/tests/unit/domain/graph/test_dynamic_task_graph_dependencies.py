"""
Test dependency handling in DynamicTaskGraph.

Tests the core dependency edge functionality added to fix sibling dependencies.
"""

import pytest
import asyncio
from typing import List

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph


class TestDynamicTaskGraphDependencies:
    """Test dependency handling in DynamicTaskGraph."""

    @pytest.fixture
    def graph(self) -> DynamicTaskGraph:
        """Create a fresh DynamicTaskGraph for testing."""
        return DynamicTaskGraph()

    @pytest.fixture
    def task_nodes(self) -> List[TaskNode]:
        """Create test task nodes."""
        return [
            TaskNode(
                task_id="task_a",
                goal="First task",
                task_type=TaskType.THINK
            ),
            TaskNode(
                task_id="task_b",
                goal="Second task",
                task_type=TaskType.THINK,
                dependencies=frozenset(["task_a"])
            ),
            TaskNode(
                task_id="task_c",
                goal="Third task",
                task_type=TaskType.THINK,
                dependencies=frozenset(["task_a", "task_b"])
            )
        ]

    @pytest.mark.asyncio
    async def test_add_node_with_dependencies_creates_edges(self, graph: DynamicTaskGraph):
        """Test that adding a node with dependencies creates graph edges."""
        # Add task_a first
        task_a = TaskNode(task_id="task_a", goal="First task", task_type=TaskType.THINK)
        await graph.add_node(task_a)

        # Add task_b with dependency on task_a
        task_b = TaskNode(
            task_id="task_b",
            goal="Second task",
            task_type=TaskType.THINK,
            dependencies=frozenset(["task_a"])
        )
        await graph.add_node(task_b)

        # Verify edge was created
        assert "task_a" in graph._graph.predecessors("task_b")
        assert "task_b" in graph._graph.successors("task_a")

    @pytest.mark.asyncio
    async def test_add_node_ignores_missing_dependencies(self, graph: DynamicTaskGraph):
        """Test that dependencies to non-existent nodes are ignored."""
        task_b = TaskNode(
            task_id="task_b",
            goal="Second task",
            task_type=TaskType.THINK,
            dependencies=frozenset(["task_a"])  # task_a doesn't exist yet
        )
        await graph.add_node(task_b)

        # No edge should be created
        assert list(graph._graph.predecessors("task_b")) == []

    @pytest.mark.asyncio
    async def test_add_dependency_edge_creates_edge(self, graph: DynamicTaskGraph):
        """Test the add_dependency_edge method."""
        # Add both nodes first
        task_a = TaskNode(task_id="task_a", goal="First task", task_type=TaskType.THINK)
        task_b = TaskNode(task_id="task_b", goal="Second task", task_type=TaskType.THINK)
        await graph.add_node(task_a)
        await graph.add_node(task_b)

        # Add dependency edge
        await graph.add_dependency_edge("task_a", "task_b")

        # Verify edge was created
        assert "task_a" in graph._graph.predecessors("task_b")
        assert "task_b" in graph._graph.successors("task_a")

        # Verify TaskNode was updated with dependency
        updated_task_b = graph.get_node("task_b")
        assert "task_a" in updated_task_b.dependencies

    @pytest.mark.asyncio
    async def test_add_dependency_edge_missing_source(self, graph: DynamicTaskGraph):
        """Test add_dependency_edge with missing source node."""
        task_b = TaskNode(task_id="task_b", goal="Second task", task_type=TaskType.THINK)
        await graph.add_node(task_b)

        with pytest.raises(KeyError, match="Source task ID task_a not found"):
            await graph.add_dependency_edge("task_a", "task_b")

    @pytest.mark.asyncio
    async def test_add_dependency_edge_missing_target(self, graph: DynamicTaskGraph):
        """Test add_dependency_edge with missing target node."""
        task_a = TaskNode(task_id="task_a", goal="First task", task_type=TaskType.THINK)
        await graph.add_node(task_a)

        with pytest.raises(KeyError, match="Target task ID task_b not found"):
            await graph.add_dependency_edge("task_a", "task_b")

    @pytest.mark.asyncio
    async def test_get_ready_nodes_respects_dependencies(self, graph: DynamicTaskGraph, task_nodes: List[TaskNode]):
        """Test that get_ready_nodes respects dependency order."""
        # Add all nodes
        for node in task_nodes:
            await graph.add_node(node)

        # Initially, only task_a should be ready (no dependencies)
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].task_id == "task_a"

        # Complete task_a (proper status transitions)
        await graph.update_node_status("task_a", TaskStatus.READY)
        await graph.update_node_status("task_a", TaskStatus.EXECUTING)
        await graph.update_node_status("task_a", TaskStatus.COMPLETED)

        # Now task_b should be ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].task_id == "task_b"

        # Complete task_b (proper status transitions)
        await graph.update_node_status("task_b", TaskStatus.READY)
        await graph.update_node_status("task_b", TaskStatus.EXECUTING)
        await graph.update_node_status("task_b", TaskStatus.COMPLETED)

        # Now task_c should be ready
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].task_id == "task_c"

    @pytest.mark.asyncio
    async def test_complex_dependency_graph(self, graph: DynamicTaskGraph):
        """Test a complex diamond-shaped dependency graph."""
        # Create diamond pattern: A -> B,C -> D
        nodes = [
            TaskNode(task_id="A", goal="Start", task_type=TaskType.THINK),
            TaskNode(task_id="B", goal="Branch 1", task_type=TaskType.THINK, dependencies=frozenset(["A"])),
            TaskNode(task_id="C", goal="Branch 2", task_type=TaskType.THINK, dependencies=frozenset(["A"])),
            TaskNode(task_id="D", goal="End", task_type=TaskType.THINK, dependencies=frozenset(["B", "C"]))
        ]

        for node in nodes:
            await graph.add_node(node)

        # Initially only A is ready
        ready = graph.get_ready_nodes()
        assert len(ready) == 1 and ready[0].task_id == "A"

        # Complete A (proper status transitions)
        await graph.update_node_status("A", TaskStatus.READY)
        await graph.update_node_status("A", TaskStatus.EXECUTING)
        await graph.update_node_status("A", TaskStatus.COMPLETED)

        # Now B and C are ready
        ready = graph.get_ready_nodes()
        ready_ids = {node.task_id for node in ready}
        assert ready_ids == {"B", "C"}

        # Complete B (proper status transitions)
        await graph.update_node_status("B", TaskStatus.READY)
        await graph.update_node_status("B", TaskStatus.EXECUTING)
        await graph.update_node_status("B", TaskStatus.COMPLETED)

        # Still only C is ready (D needs both B and C)
        ready = graph.get_ready_nodes()
        assert len(ready) == 1 and ready[0].task_id == "C"

        # Complete C (proper status transitions)
        await graph.update_node_status("C", TaskStatus.READY)
        await graph.update_node_status("C", TaskStatus.EXECUTING)
        await graph.update_node_status("C", TaskStatus.COMPLETED)

        # Now D is ready
        ready = graph.get_ready_nodes()
        assert len(ready) == 1 and ready[0].task_id == "D"

    @pytest.mark.asyncio
    async def test_concurrent_dependency_edge_addition(self, graph: DynamicTaskGraph):
        """Test thread-safe concurrent dependency edge addition."""
        # Add base nodes
        nodes = [TaskNode(task_id=f"task_{i}", goal=f"Task {i}", task_type=TaskType.THINK)
                 for i in range(10)]

        for node in nodes:
            await graph.add_node(node)

        # Concurrently add dependency edges
        tasks = []
        for i in range(1, 10):
            task = graph.add_dependency_edge(f"task_{i-1}", f"task_{i}")
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Verify all edges were created correctly
        for i in range(1, 10):
            predecessors = list(graph._graph.predecessors(f"task_{i}"))
            assert f"task_{i-1}" in predecessors

        # Verify readiness order
        ready = graph.get_ready_nodes()
        assert len(ready) == 1 and ready[0].task_id == "task_0"

    @pytest.mark.asyncio
    async def test_dependency_cycles_detected(self, graph: DynamicTaskGraph):
        """Test that dependency cycles can be detected."""
        # Create nodes with circular dependency
        node_a = TaskNode(task_id="A", goal="Task A", task_type=TaskType.THINK, dependencies=frozenset(["B"]))
        node_b = TaskNode(task_id="B", goal="Task B", task_type=TaskType.THINK, dependencies=frozenset(["A"]))

        # This won't create cycle yet since dependencies are only added if target exists
        await graph.add_node(node_a)
        await graph.add_node(node_b)

        # Manually create the cycle
        await graph.add_dependency_edge("A", "B")  # Will create A -> B edge
        await graph.add_dependency_edge("B", "A")  # Will create B -> A edge (cycle!)

        # Check cycle detection
        assert graph.has_cycles()

    @pytest.mark.asyncio
    async def test_event_emission_for_dependency_edges(self, graph: DynamicTaskGraph):
        """Test that dependency edge addition emits events."""
        events = []

        def capture_event(event):
            events.append(event)

        graph.add_event_handler(capture_event)

        # Add nodes
        task_a = TaskNode(task_id="task_a", goal="First", task_type=TaskType.THINK)
        task_b = TaskNode(task_id="task_b", goal="Second", task_type=TaskType.THINK)
        await graph.add_node(task_a)
        await graph.add_node(task_b)

        # Clear events from node addition
        events.clear()

        # Add dependency edge
        await graph.add_dependency_edge("task_a", "task_b")

        # Verify event was emitted
        assert len(events) == 1
        event = events[0]
        assert event.task_id == "task_b"
        assert event.metadata.get("action") == "dependency_added"
        assert event.metadata.get("dependency_id") == "task_a"