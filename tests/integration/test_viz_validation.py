"""
Integration tests for visualization functionality validation.

Tests both programmatic usage and CLI integration.
"""

import asyncio
from pathlib import Path
import pytest
from unittest.mock import Mock, patch

from roma_dspy.config.manager import ConfigManager
from roma_dspy.core.engine.solve import RecursiveSolver
from roma_dspy.core.engine.dag import TaskDAG
from roma_dspy.core.signatures.base_models.task_node import TaskNode
from roma_dspy.types import TaskStatus, NodeType
from roma_dspy.visualizer import (
    TreeVisualizer,
    TimelineVisualizer,
    StatisticsVisualizer,
    ContextFlowVisualizer,
    LLMTraceVisualizer,
    HierarchicalVisualizer,
    RealTimeVisualizer,
)


@pytest.fixture
def sample_dag():
    """Create a sample DAG with hierarchical structure for testing."""
    dag = TaskDAG(execution_id="test-exec-123")

    # Create root task (no parent_id makes it root)
    root = TaskNode(
        task_id="root-1",
        goal="Plan a weekend in Barcelona",
        execution_id="test-exec-123",
        max_depth=2,
        depth=0,
        status=TaskStatus.COMPLETED,
        node_type=NodeType.PLAN,
    )
    dag.add_node(root)

    # Create subtasks (start in PENDING, then set result)
    # Subtask 1: Research attractions
    subtask1 = TaskNode(
        task_id="task-1",
        goal="Research top attractions in Barcelona",
        execution_id="test-exec-123",
        max_depth=2,
        depth=1,
        parent_id="root-1",
        node_type=NodeType.EXECUTE,
    )
    subtask1 = subtask1.restore_state(
        result="Top attractions include Sagrada Familia, Park Güell, and Casa Batlló.",
        status=TaskStatus.COMPLETED
    )

    # Subtask 2: Find restaurants (depends on subtask 1)
    subtask2 = TaskNode(
        task_id="task-2",
        goal="Find highly-rated restaurants near attractions",
        execution_id="test-exec-123",
        max_depth=2,
        depth=1,
        parent_id="root-1",
        dependencies=frozenset(["task-1"]),
        node_type=NodeType.EXECUTE,
    )
    subtask2 = subtask2.restore_state(
        result="Recommended restaurants: Cervecería Catalana, El Xampanyet, Can Culleretes.",
        status=TaskStatus.COMPLETED
    )

    # Subtask 3: Create itinerary (depends on both)
    subtask3 = TaskNode(
        task_id="task-3",
        goal="Create day-by-day itinerary",
        execution_id="test-exec-123",
        max_depth=2,
        depth=1,
        parent_id="root-1",
        dependencies=frozenset(["task-1", "task-2"]),
        node_type=NodeType.EXECUTE,
    )
    subtask3 = subtask3.restore_state(
        result="Day 1: Sagrada Familia + lunch at Cervecería Catalana. Day 2: Park Güell + tapas at El Xampanyet.",
        status=TaskStatus.COMPLETED
    )

    # Create subgraph with all subtasks
    subgraph = dag.create_subgraph(
        parent_task_id="root-1",
        subtasks=[subtask1, subtask2, subtask3]
    )

    # Update root with subgraph reference and result
    root = root.set_subgraph(subgraph.dag_id)  # Use dag_id, not subgraph_id
    root = root.restore_state(result=subtask3.result, status=TaskStatus.COMPLETED)
    dag.update_node(root)

    return dag


class TestVisualizerProgrammatic:
    """Test visualizers programmatically (no API server required)."""

    def test_tree_visualizer_with_dag(self, sample_dag):
        """Test TreeVisualizer with a sample DAG."""
        viz = TreeVisualizer(use_colors=False, show_ids=True, show_timing=False)
        output = viz.visualize(source=sample_dag)

        # Verify output contains expected elements
        assert "HIERARCHICAL TASK DECOMPOSITION TREE" in output
        assert "Plan a weekend in Barcelona" in output
        assert "Research top attractions" in output
        assert "Find highly-rated restaurants" in output
        assert "Create day-by-day itinerary" in output
        assert "EXECUTION STATISTICS" in output
        assert "Total Tasks:" in output

    def test_tree_visualizer_with_colors(self, sample_dag):
        """Test TreeVisualizer with colors enabled."""
        viz = TreeVisualizer(use_colors=True)
        output = viz.visualize(source=sample_dag)

        # Should contain ANSI color codes
        assert "\033[" in output or "Plan a weekend" in output

    def test_tree_visualizer_from_snapshot(self):
        """Test TreeVisualizer with dict snapshot (from storage)."""
        snapshot = {
            "tasks": {
                "root-1": {
                    "id": "root-1",
                    "goal": "Test task",
                    "status": "completed",
                    "depth": 0,
                    "max_depth": 2,
                    "is_root": True,
                    "node_type": "PLAN",
                    "execution_history": ["planner", "aggregator"],
                }
            },
            "subgraphs": {},
            "statistics": {
                "total_tasks": 1,
                "num_subgraphs": 0,
                "is_complete": True,
                "status_counts": {"completed": 1},
                "depth_distribution": {"0": 1},
            },
        }

        viz = TreeVisualizer(use_colors=False)
        output = viz.visualize_from_snapshot(snapshot)

        assert "HIERARCHICAL TASK DECOMPOSITION TREE (FROM STORAGE)" in output
        assert "Test task" in output
        assert "Total Tasks: 1" in output

    def test_timeline_visualizer(self, sample_dag):
        """Test TimelineVisualizer."""
        viz = TimelineVisualizer(width=80)
        output = viz.visualize(source=sample_dag)

        # Timeline visualizer should work even without timing data
        assert output is not None
        assert len(output) > 0

    def test_statistics_visualizer(self, sample_dag):
        """Test StatisticsVisualizer."""
        viz = StatisticsVisualizer()
        output = viz.visualize(source=sample_dag)

        assert "DETAILED EXECUTION STATISTICS" in output
        assert "PERFORMANCE METRICS" in output
        assert "Total Tasks:" in output

    def test_llm_trace_visualizer(self, sample_dag):
        """Test LLMTraceVisualizer."""
        viz = LLMTraceVisualizer(show_metrics=True, show_summary=True, verbose=False)
        output = viz.visualize(source=sample_dag)

        assert "EXECUTION TRACE" in output
        assert "Root Goal:" in output
        assert "Plan a weekend in Barcelona" in output
        assert "EXECUTION SUMMARY" in output

    def test_llm_trace_export_json(self, sample_dag):
        """Test LLMTraceVisualizer JSON export."""
        viz = LLMTraceVisualizer()
        trace_json = viz.export_trace_json(source=sample_dag)

        assert "root_goal" in trace_json
        assert "events" in trace_json
        assert trace_json["root_goal"] == "Plan a weekend in Barcelona"

    def test_hierarchical_visualizer(self, sample_dag):
        """Test HierarchicalVisualizer combining multiple modes."""
        viz = HierarchicalVisualizer(mode="tree", use_colors=False, verbose=False)

        # Test direct visualization (bypassing solver)
        output = viz.tree.visualize(source=sample_dag)

        assert "HIERARCHICAL TASK DECOMPOSITION TREE" in output
        assert "Plan a weekend in Barcelona" in output

    def test_real_time_visualizer_events(self):
        """Test RealTimeVisualizer event tracking."""
        viz = RealTimeVisualizer(use_colors=False, verbose=True)

        # Create a simple task (already completed for simplicity)
        task = TaskNode(
            task_id="test-1",
            goal="Test goal",
            execution_id="test-exec",
            max_depth=2,
            depth=0,
            status=TaskStatus.COMPLETED,  # Start in completed state
        )

        # Test event methods
        viz.on_execution_start(task)
        assert viz.start_time is not None

        viz.on_task_enter(task, depth=0)
        assert len(viz.execution_stack) == 1

        viz.on_module_start(task, "executor", depth=0)
        assert len(viz.events) == 1

        viz.on_module_complete(task, "executor", "result", 0.5, depth=0)
        assert len(viz.events) == 2

        viz.on_task_complete(task, depth=0)
        assert len(viz.execution_stack) == 0

        viz.on_execution_complete(task)


class TestVisualizerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_dag(self):
        """Test visualizers with empty DAG."""
        dag = TaskDAG(execution_id="empty")
        viz = TreeVisualizer()

        output = viz.visualize(source=dag)
        assert "No root task found" in output

    def test_no_execution_data(self):
        """Test visualizers with no data."""
        viz = TreeVisualizer()
        output = viz.visualize(source=None, dag=None)

        assert "No execution data available" in output

    def test_visualizer_with_task_node(self):
        """Test visualizer with TaskNode as source."""
        task = TaskNode(
            task_id="root-1",
            goal="Test root task",
            execution_id="test",
            max_depth=2,
            depth=0,
            status=TaskStatus.COMPLETED,
        )

        dag = TaskDAG(execution_id="test")
        dag.add_node(task)

        viz = TreeVisualizer()
        output = viz.visualize(source=task, dag=dag)

        assert "Test root task" in output

    def test_invalid_snapshot_format(self):
        """Test TreeVisualizer with invalid snapshot."""
        viz = TreeVisualizer()

        # Invalid snapshot - not a dict
        output = viz.visualize_from_snapshot("invalid")
        assert "Invalid snapshot format" in output

        # Dict but missing 'tasks' key
        output = viz.visualize_from_snapshot({"data": {}})
        assert "Invalid snapshot format" in output

        # Empty tasks
        output = viz.visualize_from_snapshot({"tasks": {}})
        assert "No tasks found" in output


class TestContextFlowVisualizer:
    """Test ContextFlowVisualizer (requires runtime)."""

    def test_context_flow_no_runtime(self):
        """Test context flow visualizer without runtime."""
        dag = TaskDAG(execution_id="test")
        viz = ContextFlowVisualizer()

        output = viz.visualize(source=None, dag=dag)
        assert "No runtime context store available" in output

    def test_context_flow_with_mock_runtime(self, sample_dag):
        """Test context flow visualizer with mocked runtime."""
        viz = ContextFlowVisualizer(use_colors=False)

        # Create mock source with runtime
        mock_runtime = Mock()
        mock_runtime.context_store.get_task_index.return_value = 0
        mock_runtime.context_store.get_result.return_value = "Mock result"

        mock_source = Mock()
        mock_source._solver = Mock()
        mock_source._solver.runtime = mock_runtime
        mock_source._solver.last_dag = sample_dag

        output = viz.visualize(source=mock_source)

        # Should generate output even with mock
        assert "CONTEXT FLOW VISUALIZATION" in output


@pytest.mark.asyncio
async def test_end_to_end_solve_with_visualizers():
    """Test solving a simple task and visualizing it."""
    # Load config
    config_mgr = ConfigManager()
    config = config_mgr.load_config(profile="lightweight")

    # Override for testing
    config.runtime.max_depth = 1
    config.runtime.verbose = False
    config.resilience.checkpoint.enabled = False
    config.storage.postgres.enabled = False

    # Create solver
    solver = RecursiveSolver(config=config)

    # Solve a simple task
    with patch('roma_dspy.core.modules.atomizer.Atomizer.forward') as mock_atomizer:
        with patch('roma_dspy.core.modules.executor.Executor.forward') as mock_executor:
            # Mock atomizer to say task is atomic
            mock_atomizer.return_value = Mock(
                is_atomic=True,
                rationale="Simple question"
            )

            # Mock executor result
            mock_executor.return_value = Mock(
                answer="The capital of France is Paris.",
                reasoning="Based on geographic knowledge."
            )

            result = await solver.async_solve("What is the capital of France?", depth=0)

            # Verify we got a result
            assert result is not None
            assert result.status == TaskStatus.COMPLETED

            # Test all visualizers with the result
            dag = solver.last_dag
            assert dag is not None

            # Tree visualizer
            tree_viz = TreeVisualizer(use_colors=False)
            tree_output = tree_viz.visualize(source=dag)
            assert "capital of France" in tree_output

            # Timeline visualizer
            timeline_viz = TimelineVisualizer()
            timeline_output = timeline_viz.visualize(source=dag)
            assert "EXECUTION TIMELINE" in timeline_output

            # Statistics visualizer
            stats_viz = StatisticsVisualizer()
            stats_output = stats_viz.visualize(source=dag)
            assert "DETAILED EXECUTION STATISTICS" in stats_output

            # LLM Trace visualizer
            trace_viz = LLMTraceVisualizer(verbose=False)
            trace_output = trace_viz.visualize(source=dag)
            assert "EXECUTION TRACE" in trace_output

            # Export to JSON
            trace_json = trace_viz.export_trace_json(source=dag)
            assert "root_goal" in trace_json
            assert "capital of France" in trace_json["root_goal"]


def test_visualizer_format_helpers():
    """Test helper formatting methods."""
    viz = RealTimeVisualizer(use_colors=False)

    # Test duration formatting
    assert "ms" in viz.format_duration(0.001)
    assert "s" in viz.format_duration(1.5)
    assert "m" in viz.format_duration(90)

    # Test status emoji
    assert viz.get_status_emoji(TaskStatus.COMPLETED) == "✅"
    assert viz.get_status_emoji(TaskStatus.FAILED) == "❌"
    assert viz.get_status_emoji(TaskStatus.EXECUTING) == "⚡"

    # Test module emoji
    assert viz.get_module_emoji("atomizer") == "🔍"
    assert viz.get_module_emoji("planner") == "📝"
    assert viz.get_module_emoji("executor") == "⚡"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])