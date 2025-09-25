"""
Tests for Graph Visualizer (Task 1.2.4)

Following TDD principles to test graph visualization capabilities.
"""

import tempfile
from pathlib import Path

import pytest

from roma.domain.entities.task_node import TaskNode
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType
from roma.tools.graph_visualizer import GraphVisualizer, visualize_graph


class TestGraphVisualizer:
    """Test GraphVisualizer functionality."""

    @pytest.fixture
    async def sample_graph(self):
        """Create sample graph for testing."""
        root = TaskNode(
            task_id="root",
            goal="Root task for visualization testing",
            task_type=TaskType.THINK,
            node_type=NodeType.PLAN,
            status=TaskStatus.COMPLETED
        )

        child1 = TaskNode(
            task_id="child1",
            goal="First child task",
            task_type=TaskType.WRITE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.EXECUTING,
            parent_id="root"
        )

        child2 = TaskNode(
            task_id="child2",
            goal="Second child task",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.FAILED,
            parent_id="root"
        )

        graph = DynamicTaskGraph(root_node=root)
        await graph.add_node(child1)
        await graph.add_node(child2)

        return graph

    def test_visualizer_initialization(self, sample_graph):
        """Test GraphVisualizer initialization."""
        visualizer = GraphVisualizer(sample_graph)

        assert visualizer.graph == sample_graph
        assert GraphVisualizer.STATUS_COLORS[TaskStatus.PENDING] == "#E8E8E8"
        assert GraphVisualizer.TASK_TYPE_SHAPES[TaskType.THINK] == "ellipse"

    @pytest.mark.asyncio
    async def test_render_ascii_basic(self, sample_graph):
        """Test basic ASCII rendering."""
        graph = await sample_graph
        visualizer = GraphVisualizer(graph)

        ascii_output = visualizer.render_ascii()

        # Should contain root and children
        assert "Root task for visualization testing" in ascii_output
        assert "First child task" in ascii_output
        assert "Second child task" in ascii_output

        # Should have tree structure symbols
        assert "└──" in ascii_output or "├──" in ascii_output

        # Should have status symbols
        assert "🟢" in ascii_output  # Completed
        assert "🔵" in ascii_output  # Executing
        assert "🔴" in ascii_output  # Failed

    @pytest.mark.asyncio
    async def test_render_ascii_with_details(self, sample_graph):
        """Test ASCII rendering with detailed information."""
        graph = await sample_graph
        visualizer = GraphVisualizer(graph)

        ascii_output = visualizer.render_ascii(show_details=True)

        # Should include status and type information
        assert "COMPLETED" in ascii_output
        assert "EXECUTING" in ascii_output
        assert "FAILED" in ascii_output
        assert "THINK" in ascii_output
        assert "WRITE" in ascii_output
        assert "RETRIEVE" in ascii_output

    @pytest.mark.asyncio
    async def test_render_ascii_empty_graph(self):
        """Test ASCII rendering of empty graph."""
        empty_graph = DynamicTaskGraph()
        visualizer = GraphVisualizer(empty_graph)

        ascii_output = visualizer.render_ascii()

        assert ascii_output == "Empty graph"

    @pytest.mark.asyncio
    async def test_get_graph_statistics(self, sample_graph):
        """Test graph statistics calculation."""
        graph = await sample_graph
        visualizer = GraphVisualizer(graph)

        stats = visualizer.get_graph_statistics()

        # Basic counts
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 2

        # Status breakdown
        assert stats["status_breakdown"]["COMPLETED"] == 1
        assert stats["status_breakdown"]["EXECUTING"] == 1
        assert stats["status_breakdown"]["FAILED"] == 1

        # Type breakdown
        assert stats["type_breakdown"]["THINK"] == 1
        assert stats["type_breakdown"]["WRITE"] == 1
        assert stats["type_breakdown"]["RETRIEVE"] == 1

        # Structure metrics
        assert stats["max_depth"] == 1  # Children are depth 1
        assert stats["max_fanout"] == 2  # Root has 2 children

    @pytest.mark.asyncio
    async def test_export_summary(self, sample_graph):
        """Test exporting graph summary to file."""
        graph = await sample_graph
        visualizer = GraphVisualizer(graph)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            output_path = visualizer.export_summary(tmp.name)

            # Check file was created
            assert Path(output_path).exists()

            # Check content
            content = Path(output_path).read_text(encoding='utf-8')
            assert "ROMA Task Graph Analysis" in content
            assert "total_nodes: 3" in content
            assert "Root task for visualization testing" in content

            # Cleanup
            Path(output_path).unlink()

    @pytest.mark.asyncio
    async def test_render_graphviz_dot_format(self, sample_graph):
        """Test Graphviz DOT format generation."""
        graph = await sample_graph
        visualizer = GraphVisualizer(graph)

        # Test DOT format (doesn't require graphviz binary)
        try:
            dot_output = visualizer.render_graphviz(output_format="dot")

            # Should be valid DOT syntax
            assert "digraph" in dot_output
            assert "root" in dot_output
            assert "child1" in dot_output
            assert "child2" in dot_output
            assert "->" in dot_output  # Graph edges

        except ImportError:
            # Graphviz not available, skip test
            pytest.skip("Graphviz not available")

    @pytest.mark.asyncio
    async def test_matplotlib_rendering_import_check(self, sample_graph):
        """Test Matplotlib rendering import handling."""
        graph = await sample_graph
        visualizer = GraphVisualizer(graph)

        # Test import error handling
        try:
            # This should either work or raise ImportError
            result = visualizer.render_matplotlib()
            # If it works, result should be None (no save path)
            assert result is None
        except ImportError as e:
            # Should have helpful error message
            assert "matplotlib not available" in str(e).lower()

    @pytest.mark.asyncio
    async def test_visualize_graph_utility(self, sample_graph):
        """Test utility function for quick visualization."""
        graph = await sample_graph

        # Test ASCII method
        ascii_result = visualize_graph(graph, method="ascii")
        assert isinstance(ascii_result, str)
        assert "Root task for visualization testing" in ascii_result

        # Test invalid method
        with pytest.raises(ValueError, match="Unknown visualization method"):
            visualize_graph(graph, method="invalid")


class TestGraphVisualizerPerformance:
    """Test performance characteristics of visualization."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_graph_visualization(self):
        """Test visualization performance with large graph."""
        # Create large graph: 1 root + 50 children
        root = TaskNode(
            task_id="root",
            goal="Root of large graph",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        graph = DynamicTaskGraph(root_node=root)

        # Add 50 children
        for i in range(50):
            child = TaskNode(
                task_id=f"child_{i}",
                goal=f"Child task {i}",
                task_type=TaskType.WRITE,
                status=TaskStatus.PENDING,
                parent_id="root"
            )
            await graph.add_node(child)

        visualizer = GraphVisualizer(graph)

        import time

        # Test ASCII rendering performance
        start_time = time.time()
        ascii_output = visualizer.render_ascii()
        ascii_time = time.time() - start_time

        # Should complete quickly and contain all nodes
        assert ascii_time < 1.0  # Should be very fast
        assert ascii_output.count("Child task") == 50

        # Test statistics calculation performance
        start_time = time.time()
        stats = visualizer.get_graph_statistics()
        stats_time = time.time() - start_time

        assert stats_time < 0.5  # Statistics should be fast
        assert stats["total_nodes"] == 51
        assert stats["max_fanout"] == 50


class TestGraphVisualizerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_graph_statistics(self):
        """Test statistics on empty graph."""
        empty_graph = DynamicTaskGraph()
        visualizer = GraphVisualizer(empty_graph)

        stats = visualizer.get_graph_statistics()

        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert all(count == 0 for count in stats["status_breakdown"].values())
        assert all(count == 0 for count in stats["type_breakdown"].values())

    @pytest.mark.asyncio
    async def test_single_node_visualization(self):
        """Test visualization of single node graph."""
        single_node = TaskNode(
            task_id="single",
            goal="Single node test",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        graph = DynamicTaskGraph(root_node=single_node)
        visualizer = GraphVisualizer(graph)

        ascii_output = visualizer.render_ascii()
        assert "Single node test" in ascii_output

        stats = visualizer.get_graph_statistics()
        assert stats["total_nodes"] == 1
        assert stats["max_depth"] == 0
        assert stats["max_fanout"] == 0

    @pytest.mark.asyncio
    async def test_deep_hierarchy_visualization(self):
        """Test visualization of deep linear hierarchy."""
        # Create chain: root -> child -> grandchild -> great_grandchild
        nodes = []
        for i in range(4):
            parent_id = nodes[i-1].task_id if i > 0 else None
            node = TaskNode(
                task_id=f"level_{i}",
                goal=f"Task at level {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.PENDING,
                parent_id=parent_id
            )
            nodes.append(node)

        graph = DynamicTaskGraph(root_node=nodes[0])
        for node in nodes[1:]:
            await graph.add_node(node)

        visualizer = GraphVisualizer(graph)

        ascii_output = visualizer.render_ascii()
        # Should show hierarchical structure
        assert "Task at level 0" in ascii_output
        assert "Task at level 3" in ascii_output

        stats = visualizer.get_graph_statistics()
        assert stats["max_depth"] == 3  # Deepest node is at depth 3
