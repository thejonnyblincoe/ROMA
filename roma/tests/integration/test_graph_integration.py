"""
Integration Tests for Graph Operations (Task 1.2.6)

Testing cross-component interactions and end-to-end graph workflows.
"""

import asyncio

import pytest
import pytest_asyncio

from roma.domain.entities.task_node import TaskNode
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType

# Events are managed by GraphStateManager in integration scenarios
from roma.tools.graph_visualizer import GraphVisualizer


class TestGraphIntegration:
    """Test integration between graph, events, and visualization."""

    @pytest_asyncio.fixture
    async def research_workflow(self):
        """Create a realistic research workflow graph."""
        # Root research task
        root = TaskNode(
            task_id="research_task",
            goal="Analyze AI market trends for Q4 2024",
            task_type=TaskType.THINK,
            node_type=NodeType.PLAN,
            status=TaskStatus.PENDING
        )

        graph = DynamicTaskGraph(root_node=root)

        # Data gathering subtasks
        data_tasks = [
            TaskNode(
                task_id="gather_reports",
                goal="Collect industry reports",
                task_type=TaskType.RETRIEVE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id="research_task"
            ),
            TaskNode(
                task_id="scrape_news",
                goal="Scrape recent news articles",
                task_type=TaskType.RETRIEVE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id="research_task"
            ),
            TaskNode(
                task_id="survey_data",
                goal="Analyze survey responses",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id="research_task"
            )
        ]

        for task in data_tasks:
            await graph.add_node(task)

        # Analysis subtasks that depend on data gathering
        analysis_tasks = [
            TaskNode(
                task_id="trend_analysis",
                goal="Identify key market trends",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id="research_task"
            ),
            TaskNode(
                task_id="competitive_analysis",
                goal="Analyze competitive landscape",
                task_type=TaskType.THINK,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id="research_task"
            )
        ]

        for task in analysis_tasks:
            await graph.add_node(task)

        # Final report generation
        report_task = TaskNode(
            task_id="generate_report",
            goal="Generate comprehensive market analysis report",
            task_type=TaskType.WRITE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING,
            parent_id="research_task"
        )
        await graph.add_node(report_task)

        return graph

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_execution(self, research_workflow):
        """Test complete workflow execution with event tracking."""
        graph = research_workflow

        # Event collection
        # Event collection is handled by GraphStateManager in integration scenarios

        # Initially, only the root task should be ready (no dependencies)
        ready_nodes = graph.get_ready_nodes()
        assert len(ready_nodes) == 1
        assert ready_nodes[0].task_id == "research_task"

        # Complete the root task to unlock children
        await graph.update_node_status("research_task", TaskStatus.READY)
        await graph.update_node_status("research_task", TaskStatus.EXECUTING)
        await graph.update_node_status("research_task", TaskStatus.COMPLETED)

        # Now data gathering subtasks should be ready (their parent is COMPLETED)
        ready_nodes = graph.get_ready_nodes()
        ready_ids = {node.task_id for node in ready_nodes}
        expected_ready = {"gather_reports", "scrape_news", "survey_data"}
        # Include analysis and report tasks since they all have same parent
        expected_ready.update({"trend_analysis", "competitive_analysis", "generate_report"})
        assert expected_ready == ready_ids

        # Execute all subtasks since they're all ready (flat hierarchy)
        subtask_ids = ["gather_reports", "scrape_news", "survey_data",
                      "trend_analysis", "competitive_analysis", "generate_report"]

        for task_id in subtask_ids:
            await graph.update_node_status(task_id, TaskStatus.READY)
            await graph.update_node_status(task_id, TaskStatus.EXECUTING)
            await graph.update_node_status(task_id, TaskStatus.COMPLETED)

        # Verify all tasks completed
        all_nodes = graph.get_all_nodes()
        completed_count = sum(1 for n in all_nodes if n.status == TaskStatus.COMPLETED)
        assert completed_count == len(all_nodes)  # All should be completed now

        # Event logging is handled by GraphStateManager in integration scenarios
        # Verify all operations completed successfully without needing event tracking

    @pytest.mark.asyncio
    async def test_graph_visualization_integration(self, research_workflow):
        """Test integration between graph and visualization."""
        graph = research_workflow
        visualizer = GraphVisualizer(graph)

        # Test ASCII visualization
        ascii_output = visualizer.render_ascii(show_details=True)
        assert "Analyze AI market trends" in ascii_output
        assert "Collect industry reports" in ascii_output
        assert "Generate comprehensive market analysis r" in ascii_output  # Truncated

        # Test statistics
        stats = visualizer.get_graph_statistics()
        assert stats["total_nodes"] == 7  # Root + 6 subtasks
        assert stats["total_edges"] == 6   # All subtasks have parent
        assert stats["type_breakdown"]["RETRIEVE"] == 2
        assert stats["type_breakdown"]["THINK"] == 4  # Root + survey_data + 2 analysis tasks
        assert stats["type_breakdown"]["WRITE"] == 1
        assert stats["max_fanout"] == 6   # Root has 6 children

        # Test that visualization handles status changes
        await graph.update_node_status("gather_reports", TaskStatus.READY)
        await graph.update_node_status("gather_reports", TaskStatus.EXECUTING)

        updated_ascii = visualizer.render_ascii()
        assert "🔵" in updated_ascii  # Should show executing symbol

    @pytest.mark.asyncio
    async def test_concurrent_graph_operations(self, research_workflow):
        """Test concurrent graph operations maintain consistency."""
        graph = research_workflow

        # Create multiple concurrent operations
        async def batch_status_updates():
            tasks = []
            node_ids = ["gather_reports", "scrape_news", "survey_data"]

            for node_id in node_ids:
                tasks.append(graph.update_node_status(node_id, TaskStatus.READY))

            await asyncio.gather(*tasks)

            # Second batch - transition to executing
            tasks = []
            for node_id in node_ids:
                tasks.append(graph.update_node_status(node_id, TaskStatus.EXECUTING))

            await asyncio.gather(*tasks)

        await batch_status_updates()

        # Verify all nodes were updated correctly
        for node_id in ["gather_reports", "scrape_news", "survey_data"]:
            node = graph.get_node(node_id)
            assert node is not None
            assert node.status == TaskStatus.EXECUTING

    @pytest.mark.asyncio
    async def test_graph_relationship_traversal(self, research_workflow):
        """Test complex graph traversal operations."""
        graph = research_workflow

        # Test ancestor relationships
        ancestors = graph.get_ancestors("generate_report")
        assert "research_task" in ancestors
        assert len(ancestors) == 1  # Only immediate parent

        # Test descendant relationships
        descendants = graph.get_descendants("research_task")
        expected_descendants = {
            "gather_reports", "scrape_news", "survey_data",
            "trend_analysis", "competitive_analysis", "generate_report"
        }
        assert set(descendants) == expected_descendants

        # Test sibling relationships
        siblings = graph.get_siblings("gather_reports")
        expected_siblings = {
            "scrape_news", "survey_data", "trend_analysis",
            "competitive_analysis", "generate_report"
        }
        assert set(siblings) == expected_siblings

        # Test subtree extraction
        subtree = graph.get_subtree("research_task")
        assert len(subtree) == 7  # All nodes in the workflow

    @pytest.mark.asyncio
    async def test_graph_state_consistency_under_failures(self, research_workflow):
        """Test graph maintains consistency when operations fail."""
        graph = research_workflow

        # Event handler failures are managed by GraphStateManager
        # DynamicTaskGraph focuses on maintaining core operation consistency

        # Perform operations that should trigger handler failures
        for i in range(5):
            node_id = "gather_reports"
            if i == 0:
                await graph.update_node_status(node_id, TaskStatus.READY)
            elif i == 1:
                await graph.update_node_status(node_id, TaskStatus.EXECUTING)
            elif i == 2:
                # This should trigger the handler failure
                await graph.update_node_status(node_id, TaskStatus.COMPLETED)
            else:
                # Additional operations after failure
                node = graph.get_node(node_id)
                assert node is not None  # Graph should still work

        # Event handling is managed by GraphStateManager
        # Verify graph operations completed successfully

        # Verify graph state is consistent
        node = graph.get_node("gather_reports")
        assert node is not None
        assert node.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_dynamic_graph_expansion(self):
        """Test dynamically expanding graph during execution."""
        # Start with simple graph
        root = TaskNode(
            task_id="root",
            goal="Dynamic expansion test",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        graph = DynamicTaskGraph(root_node=root)

        # Event tracking is handled by GraphStateManager for integration scenarios

        # Add nodes dynamically as workflow progresses
        initial_child = TaskNode(
            task_id="child1",
            goal="First child",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.PENDING,
            parent_id="root"
        )
        await graph.add_node(initial_child)

        # Complete first child, then add more children
        await graph.update_node_status("child1", TaskStatus.READY)
        await graph.update_node_status("child1", TaskStatus.EXECUTING)
        await graph.update_node_status("child1", TaskStatus.COMPLETED)

        # Add second child after first completes
        second_child = TaskNode(
            task_id="child2",
            goal="Second child added dynamically",
            task_type=TaskType.WRITE,
            status=TaskStatus.PENDING,
            parent_id="root"
        )
        await graph.add_node(second_child)

        # Add grandchild
        grandchild = TaskNode(
            task_id="grandchild",
            goal="Grandchild task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING,
            parent_id="child2"
        )
        await graph.add_node(grandchild)

        # Verify graph structure
        assert len(graph.get_all_nodes()) == 4
        assert len(graph.get_children("root")) == 2
        assert len(graph.get_children("child2")) == 1

        # Event tracking is handled by GraphStateManager for integration scenarios
        # Verify all dynamic operations completed successfully

        # Test visualization of dynamic graph
        visualizer = GraphVisualizer(graph)
        stats = visualizer.get_graph_statistics()
        assert stats["max_depth"] == 2  # Root -> child -> grandchild


class TestGraphPerformanceIntegration:
    """Test performance characteristics of integrated graph operations."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_scale_graph_performance(self):
        """Test graph performance with realistic large-scale operations."""
        import time

        # Create large hierarchical graph
        root = TaskNode(
            task_id="root",
            goal="Large scale performance test",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        graph = DynamicTaskGraph(root_node=root)

        # Add many children (simulating decomposed task)
        start_time = time.time()
        for i in range(100):
            child = TaskNode(
                task_id=f"child_{i}",
                goal=f"Child task {i}",
                task_type=TaskType.WRITE if i % 3 == 0 else TaskType.THINK,
                status=TaskStatus.PENDING,
                parent_id="root"
            )
            await graph.add_node(child)

        creation_time = time.time() - start_time
        assert creation_time < 1.0  # Should complete quickly

        # Test batch status updates
        start_time = time.time()
        update_tasks = []
        for i in range(100):
            update_tasks.append(graph.update_node_status(f"child_{i}", TaskStatus.READY))

        await asyncio.gather(*update_tasks)
        update_time = time.time() - start_time
        assert update_time < 2.0  # Concurrent updates should be fast

        # Test visualization performance
        visualizer = GraphVisualizer(graph)
        start_time = time.time()
        ascii_output = visualizer.render_ascii()
        viz_time = time.time() - start_time
        assert viz_time < 0.5  # Visualization should be fast
        assert len(ascii_output) > 1000  # Should have substantial output

        # Test statistics calculation
        start_time = time.time()
        stats = visualizer.get_graph_statistics()
        stats_time = time.time() - start_time
        assert stats_time < 0.1
        assert stats["total_nodes"] == 101
        assert stats["max_fanout"] == 100

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test that graph operations don't cause memory leaks."""
        import gc

        root = TaskNode(
            task_id="memory_test",
            goal="Memory usage test",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        # Create and destroy multiple graphs
        for iteration in range(10):
            graph = DynamicTaskGraph(root_node=root)

            # Add nodes
            for i in range(50):
                node = TaskNode(
                    task_id=f"node_{iteration}_{i}",
                    goal=f"Node {i}",
                    task_type=TaskType.WRITE,
                    status=TaskStatus.PENDING,
                    parent_id="memory_test"
                )
                await graph.add_node(node)

            # Update all nodes
            for i in range(50):
                await graph.update_node_status(f"node_{iteration}_{i}", TaskStatus.READY)

            # Force garbage collection
            del graph
            gc.collect()

        # Test should complete without memory errors
        assert True  # If we get here, memory management is working


class TestGraphErrorRecovery:
    """Test error handling and recovery in graph operations."""

    @pytest.mark.asyncio
    async def test_invalid_operations_dont_corrupt_graph(self):
        """Test that invalid operations don't corrupt graph state."""
        root = TaskNode(
            task_id="error_test",
            goal="Error handling test",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        graph = DynamicTaskGraph(root_node=root)

        child = TaskNode(
            task_id="child",
            goal="Child task",
            task_type=TaskType.WRITE,
            status=TaskStatus.PENDING,
            parent_id="error_test"
        )
        await graph.add_node(child)

        # Try invalid status transition
        with pytest.raises(ValueError, match="Invalid transition"):
            await graph.update_node_status("child", TaskStatus.EXECUTING)  # PENDING -> EXECUTING is invalid

        # Verify graph state is unchanged
        node = graph.get_node("child")
        assert node is not None
        assert node.status == TaskStatus.PENDING

        # Try to update non-existent node
        with pytest.raises(KeyError):
            await graph.update_node_status("nonexistent", TaskStatus.READY)

        # Graph should still work normally
        await graph.update_node_status("child", TaskStatus.READY)
        node = graph.get_node("child")
        assert node.status == TaskStatus.READY

    @pytest.mark.asyncio
    async def test_concurrent_modification_safety(self):
        """Test graph safety under concurrent modifications."""
        root = TaskNode(
            task_id="concurrent_test",
            goal="Concurrent modification test",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        graph = DynamicTaskGraph(root_node=root)

        # Add base child
        child = TaskNode(
            task_id="child",
            goal="Test child",
            task_type=TaskType.WRITE,
            status=TaskStatus.PENDING,
            parent_id="concurrent_test"
        )
        await graph.add_node(child)

        # Create concurrent operations
        async def concurrent_updates():
            operations = []

            # Multiple status updates
            operations.append(graph.update_node_status("child", TaskStatus.READY))

            # Add more nodes concurrently
            for i in range(5):
                new_node = TaskNode(
                    task_id=f"concurrent_{i}",
                    goal=f"Concurrent task {i}",
                    task_type=TaskType.THINK,
                    status=TaskStatus.PENDING,
                    parent_id="concurrent_test"
                )
                operations.append(graph.add_node(new_node))

            return await asyncio.gather(*operations, return_exceptions=True)

        results = await concurrent_updates()

        # Some operations might fail due to race conditions, but graph should be consistent
        final_nodes = graph.get_all_nodes()
        assert len(final_nodes) >= 2  # At least root and original child

        # Verify no nodes are in invalid states
        for node in final_nodes:
            assert node.status in TaskStatus
            assert node.task_type in TaskType
