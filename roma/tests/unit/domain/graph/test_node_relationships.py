"""
TDD Tests for Node Relationship Methods (Task 1.2.3)

Following TDD principles - these tests define the interface and expected behavior
for node relationship methods in DynamicTaskGraph.
"""

import pytest

from roma.domain.entities.task_node import TaskNode
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class TestNodeRelationships:
    """Test node relationship methods in DynamicTaskGraph."""

    @pytest.fixture
    async def sample_hierarchy(self):
        """Create a sample task hierarchy for testing.
        
        Structure:
        root
        ├── child1
        │   ├── grandchild1
        │   └── grandchild2
        └── child2
            └── grandchild3
        """
        # Create nodes
        root = TaskNode(
            task_id="root",
            goal="Root task",
            task_type=TaskType.THINK,
            node_type=NodeType.PLAN,
            status=TaskStatus.PENDING
        )

        child1 = TaskNode(
            task_id="child1",
            goal="Child 1 task",
            task_type=TaskType.WRITE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING,
            parent_id="root"
        )

        child2 = TaskNode(
            task_id="child2",
            goal="Child 2 task",
            task_type=TaskType.WRITE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING,
            parent_id="root"
        )

        grandchild1 = TaskNode(
            task_id="grandchild1",
            goal="Grandchild 1 task",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING,
            parent_id="child1"
        )

        grandchild2 = TaskNode(
            task_id="grandchild2",
            goal="Grandchild 2 task",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING,
            parent_id="child1"
        )

        grandchild3 = TaskNode(
            task_id="grandchild3",
            goal="Grandchild 3 task",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING,
            parent_id="child2"
        )

        # Create graph
        graph = DynamicTaskGraph(root_node=root)

        # Add all nodes to create the hierarchy
        nodes = [child1, child2, grandchild1, grandchild2, grandchild3]
        for node in nodes:
            await graph.add_node(node)

        return graph, {
            "root": root,
            "child1": child1,
            "child2": child2,
            "grandchild1": grandchild1,
            "grandchild2": grandchild2,
            "grandchild3": grandchild3
        }

    @pytest.mark.asyncio
    async def test_get_children_root_node(self, sample_hierarchy):
        """Test getting children of root node."""
        graph, nodes = await sample_hierarchy

        children = graph.get_children("root")

        assert len(children) == 2
        assert "child1" in children
        assert "child2" in children

    @pytest.mark.asyncio
    async def test_get_children_internal_node(self, sample_hierarchy):
        """Test getting children of internal node."""
        graph, nodes = await sample_hierarchy

        children = graph.get_children("child1")

        assert len(children) == 2
        assert "grandchild1" in children
        assert "grandchild2" in children

    @pytest.mark.asyncio
    async def test_get_children_leaf_node(self, sample_hierarchy):
        """Test getting children of leaf node."""
        graph, nodes = await sample_hierarchy

        children = graph.get_children("grandchild1")

        assert children == []

    @pytest.mark.asyncio
    async def test_get_ancestors_leaf_node(self, sample_hierarchy):
        """Test getting ancestors of leaf node."""
        graph, nodes = await sample_hierarchy

        ancestors = graph.get_ancestors("grandchild1")

        assert ancestors == ["child1", "root"]

    @pytest.mark.asyncio
    async def test_get_siblings_with_siblings(self, sample_hierarchy):
        """Test getting siblings when node has siblings."""
        graph, nodes = await sample_hierarchy

        siblings = graph.get_siblings("child1")

        assert siblings == ["child2"]

    @pytest.mark.asyncio
    async def test_get_descendants_root_node(self, sample_hierarchy):
        """Test getting all descendants of root node."""
        graph, nodes = await sample_hierarchy

        descendants = graph.get_descendants("root")

        assert len(descendants) == 5
        # Should include all children and grandchildren
        assert "child1" in descendants
        assert "child2" in descendants
        assert "grandchild1" in descendants
        assert "grandchild2" in descendants
        assert "grandchild3" in descendants
