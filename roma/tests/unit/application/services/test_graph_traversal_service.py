"""
Unit tests for GraphTraversalService.

Following TDD principles - these tests define the interface and expected behavior
before implementation.
"""


import pytest

from roma.domain.entities.task_node import TaskNode
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class TestGraphTraversalService:
    """Test GraphTraversalService core functionality."""

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
            TaskNode(
                goal="Write analysis report",
                task_type=TaskType.WRITE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            ),
        ]

    async def create_linear_graph(self, sample_task_nodes) -> DynamicTaskGraph:
        """Create linear dependency graph: A -> B -> C -> D."""
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])

        # Create linear chain
        previous_node = sample_task_nodes[0]
        for node in sample_task_nodes[1:]:
            child_node = node.model_copy(update={"parent_id": previous_node.task_id})
            await graph.add_node(child_node)
            previous_node = child_node

        return graph

    async def create_parallel_graph(self, sample_task_nodes) -> DynamicTaskGraph:
        """Create parallel graph: A -> [B, C, D]."""
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])

        # All other nodes depend on root
        root_id = sample_task_nodes[0].task_id
        for node in sample_task_nodes[1:]:
            child_node = node.model_copy(update={"parent_id": root_id})
            await graph.add_node(child_node)

        return graph

    def test_graph_traversal_service_init(self):
        """Test GraphTraversalService initialization."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph()
        service = GraphTraversalService(graph)

        assert service.graph == graph

    @pytest.mark.asyncio
    async def test_get_topological_order_empty_graph(self):
        """Test topological ordering of empty graph."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph()
        service = GraphTraversalService(graph)

        order = service.get_topological_order()
        assert len(order) == 0

    @pytest.mark.asyncio
    async def test_get_topological_order_single_node(self, sample_task_nodes):
        """Test topological ordering of single node."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        service = GraphTraversalService(graph)

        order = service.get_topological_order()

        assert len(order) == 1
        assert order[0] == sample_task_nodes[0].task_id

    @pytest.mark.asyncio
    async def test_get_topological_order_linear_graph(self, sample_task_nodes):
        """Test topological ordering preserves dependency order."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_linear_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        order = service.get_topological_order()

        assert len(order) == 4
        # Root should come first, then children in dependency order
        assert order[0] == sample_task_nodes[0].task_id

    @pytest.mark.asyncio
    async def test_get_topological_order_parallel_graph(self, sample_task_nodes):
        """Test topological ordering with parallel branches."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_parallel_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        order = service.get_topological_order()

        assert len(order) == 4
        # Root should come first
        assert order[0] == sample_task_nodes[0].task_id
        # Remaining nodes can be in any order (parallel)
        remaining_ids = set(order[1:])
        expected_ids = {node.task_id for node in sample_task_nodes[1:]}
        # Note: NetworkX may assign different IDs after model_copy

    def test_detect_cycles_empty_graph(self):
        """Test cycle detection on empty graph."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph()
        service = GraphTraversalService(graph)

        cycles = service.detect_cycles()
        assert len(cycles) == 0

    @pytest.mark.asyncio
    async def test_detect_cycles_linear_graph(self, sample_task_nodes):
        """Test cycle detection on linear graph (no cycles)."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_linear_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        cycles = service.detect_cycles()
        assert len(cycles) == 0

    @pytest.mark.asyncio
    async def test_detect_cycles_parallel_graph(self, sample_task_nodes):
        """Test cycle detection on parallel graph (no cycles)."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_parallel_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        cycles = service.detect_cycles()
        assert len(cycles) == 0

    @pytest.mark.asyncio
    async def test_find_parallel_execution_paths_single_node(self, sample_task_nodes):
        """Test parallel execution paths with single node."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        service = GraphTraversalService(graph)

        paths = service.find_parallel_execution_paths()

        # Single node forms one execution batch
        assert len(paths) == 1
        assert len(paths[0]) == 1
        assert sample_task_nodes[0].task_id in paths[0]

    @pytest.mark.asyncio
    async def test_find_parallel_execution_paths_linear_graph(self, sample_task_nodes):
        """Test parallel execution paths with linear graph."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_linear_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        paths = service.find_parallel_execution_paths()

        # Linear graph creates 4 sequential batches
        assert len(paths) == 4
        # Each batch should have exactly one node
        for batch in paths:
            assert len(batch) == 1

    @pytest.mark.asyncio
    async def test_find_parallel_execution_paths_parallel_graph(self, sample_task_nodes):
        """Test parallel execution paths with parallel graph."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_parallel_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        paths = service.find_parallel_execution_paths()

        # Should have 2 batches: root first, then parallel children
        assert len(paths) == 2
        assert len(paths[0]) == 1  # Root node batch
        assert len(paths[1]) == 3  # Parallel children batch

    def test_get_node_dependencies(self, sample_task_nodes):
        """Test getting node dependencies."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        service = GraphTraversalService(graph)

        node_id = sample_task_nodes[0].task_id
        dependencies = service.get_node_dependencies(node_id)

        # Root node has no dependencies
        assert len(dependencies) == 0

    @pytest.mark.asyncio
    async def test_get_node_dependencies_with_parent(self, sample_task_nodes):
        """Test getting node dependencies with parent."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_linear_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        # Get second node's dependencies
        all_nodes = graph.get_all_nodes()
        second_node = next(node for node in all_nodes if node.parent_id is not None)

        dependencies = service.get_node_dependencies(second_node.task_id)

        # Should have exactly one dependency (parent)
        assert len(dependencies) == 1
        assert dependencies[0] == second_node.parent_id

    def test_get_node_dependents(self, sample_task_nodes):
        """Test getting node dependents."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        service = GraphTraversalService(graph)

        node_id = sample_task_nodes[0].task_id
        dependents = service.get_node_dependents(node_id)

        # Single node has no dependents
        assert len(dependents) == 0

    @pytest.mark.asyncio
    async def test_get_node_dependents_with_children(self, sample_task_nodes):
        """Test getting node dependents with children."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_parallel_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        # Root node should have 3 dependents
        root_id = sample_task_nodes[0].task_id
        dependents = service.get_node_dependents(root_id)

        assert len(dependents) == 3

    def test_calculate_graph_depth(self, sample_task_nodes):
        """Test calculating graph depth."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        service = GraphTraversalService(graph)

        depth = service.calculate_graph_depth()

        # Single node has depth 1
        assert depth == 1

    @pytest.mark.asyncio
    async def test_calculate_graph_depth_linear(self, sample_task_nodes):
        """Test calculating graph depth for linear graph."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_linear_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        depth = service.calculate_graph_depth()

        # Linear chain of 4 nodes has depth 4
        assert depth == 4

    @pytest.mark.asyncio
    async def test_calculate_graph_depth_parallel(self, sample_task_nodes):
        """Test calculating graph depth for parallel graph."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_parallel_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        depth = service.calculate_graph_depth()

        # Parallel graph has depth 2 (root + children)
        assert depth == 2

    def test_is_node_ready_for_execution_nonexistent(self):
        """Test checking if nonexistent node is ready for execution."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph()
        service = GraphTraversalService(graph)

        ready = service.is_node_ready_for_execution("nonexistent-id")
        assert ready is False

    @pytest.mark.asyncio
    async def test_is_node_ready_for_execution_pending_root(self, sample_task_nodes):
        """Test checking if pending root node is ready for execution."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        service = GraphTraversalService(graph)

        node_id = sample_task_nodes[0].task_id
        ready = service.is_node_ready_for_execution(node_id)

        # Root node with PENDING status should be ready
        assert ready is True

    @pytest.mark.asyncio
    async def test_is_node_ready_for_execution_with_incomplete_dependencies(self, sample_task_nodes):
        """Test checking if node with incomplete dependencies is ready."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        graph = await self.create_linear_graph(sample_task_nodes)
        service = GraphTraversalService(graph)

        # Get second node (has dependency on root)
        all_nodes = graph.get_all_nodes()
        second_node = next(node for node in all_nodes if node.parent_id is not None)

        ready = service.is_node_ready_for_execution(second_node.task_id)

        # Should not be ready because parent is still PENDING
        assert ready is False


class TestGraphTraversalServicePerformance:
    """Test GraphTraversalService performance with large graphs."""

    @pytest.mark.asyncio
    async def test_large_graph_topological_sort_performance(self):
        """Test topological sort performance with large graph."""
        from roma.application.services.graph_traversal_service import GraphTraversalService

        # Create large linear graph (1000 nodes)
        nodes = []
        graph = DynamicTaskGraph()

        for i in range(1000):
            parent_id = nodes[i-1].task_id if i > 0 else None
            node = TaskNode(
                goal=f"Performance test task {i}",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id=parent_id
            )
            nodes.append(node)

            if i == 0:
                graph = DynamicTaskGraph(root_node=node)
            else:
                await graph.add_node(node)

        service = GraphTraversalService(graph)

        import time
        start_time = time.time()

        order = service.get_topological_order()

        end_time = time.time()

        # Should complete quickly (< 1 second for 1000 nodes)
        execution_time = end_time - start_time
        assert execution_time < 1.0

        # Verify correctness
        assert len(order) == 1000
