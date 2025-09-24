#!/usr/bin/env python3
"""
Test script demonstrating aggregation tracking in ExecutionVisualizer.
Shows which tasks are aggregated vs executed directly.
"""

import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.roma_dspy.signatures.base_models.task_node import TaskNode
from src.roma_dspy.types.task_status import TaskStatus
from src.roma_dspy.types.node_type import NodeType
from src.roma_dspy.types.module_result import ModuleResult
from src.roma_dspy.engine.dag import TaskDAG
from src.roma_dspy.engine.visualizer import ExecutionVisualizer


def create_mock_dag_with_aggregation():
    """Create a mock DAG showing aggregation patterns."""

    # Create DAG
    dag = TaskDAG()

    # Create root task (will aggregate subtasks)
    root_task = TaskNode(
        task_id=str(uuid4()),
        goal="Create a comprehensive marketing strategy",
        depth=0,
        max_depth=2,
        status=TaskStatus.COMPLETED,
        node_type=NodeType.PLAN,
        parent_id=None
    )

    # Add planner execution
    root_task = root_task.record_module_execution(
        "planner",
        ModuleResult(
            module_name="planner",
            input=root_task.goal,
            output=[
                "Analyze target market",
                "Develop content strategy",
                "Plan social media campaigns",
                "Design email marketing"
            ],
            duration=1.5
        )
    )

    # Add root to DAG
    dag.add_node(root_task)

    # Create subgraph for subtasks
    subgraph_id = str(uuid4())
    root_task = root_task.set_subgraph(subgraph_id)
    dag.graph.nodes[root_task.task_id]['task'] = root_task

    subgraph = TaskDAG(dag_id=subgraph_id, parent_dag=dag)
    dag.subgraphs[subgraph_id] = subgraph

    # Subtask 1: PLAN node (will also aggregate its own subtasks)
    subtask1 = TaskNode(
        task_id=str(uuid4()),
        parent_id=root_task.task_id,
        goal="Analyze target market",
        depth=1,
        max_depth=2,
        status=TaskStatus.COMPLETED,
        node_type=NodeType.PLAN
    )

    # This task plans and aggregates
    subtask1 = subtask1.record_module_execution(
        "planner",
        ModuleResult(
            module_name="planner",
            input=subtask1.goal,
            output=["Research demographics", "Study competitors"],
            duration=0.8
        )
    )

    # Create nested subgraph for subtask1
    nested_subgraph_id = str(uuid4())
    subtask1 = subtask1.set_subgraph(nested_subgraph_id)
    nested_subgraph = TaskDAG(dag_id=nested_subgraph_id, parent_dag=subgraph)
    subgraph.subgraphs[nested_subgraph_id] = nested_subgraph

    # Add nested tasks
    nested_task1 = TaskNode(
        task_id=str(uuid4()),
        parent_id=subtask1.task_id,
        goal="Research demographics",
        depth=2,
        max_depth=2,
        status=TaskStatus.COMPLETED,
        node_type=NodeType.EXECUTE
    )
    nested_task1 = nested_task1.record_module_execution(
        "executor",
        ModuleResult(
            module_name="executor",
            input=nested_task1.goal,
            output="Demographics: 25-45 age group, 60% female, urban professionals",
            duration=1.2
        )
    )
    nested_subgraph.add_node(nested_task1)

    nested_task2 = TaskNode(
        task_id=str(uuid4()),
        parent_id=subtask1.task_id,
        goal="Study competitors",
        depth=2,
        max_depth=2,
        status=TaskStatus.COMPLETED,
        node_type=NodeType.EXECUTE
    )
    nested_task2 = nested_task2.record_module_execution(
        "executor",
        ModuleResult(
            module_name="executor",
            input=nested_task2.goal,
            output="Top 3 competitors: Company A (35% market share), Company B (25%), Company C (20%)",
            duration=1.5
        )
    )
    nested_subgraph.add_node(nested_task2)

    # Add aggregator to subtask1
    subtask1 = subtask1.record_module_execution(
        "aggregator",
        ModuleResult(
            module_name="aggregator",
            input="Combine market analysis results",
            output="Complete market analysis: Target demographic is urban professionals aged 25-45, primarily female. Main competitors hold 80% market share.",
            duration=0.3
        )
    )
    subgraph.add_node(subtask1)

    # Subtask 2: EXECUTE node (directly executed, no aggregation)
    subtask2 = TaskNode(
        task_id=str(uuid4()),
        parent_id=root_task.task_id,
        goal="Develop content strategy",
        depth=1,
        max_depth=2,
        status=TaskStatus.COMPLETED,
        node_type=NodeType.EXECUTE
    )
    subtask2 = subtask2.record_module_execution(
        "executor",
        ModuleResult(
            module_name="executor",
            input=subtask2.goal,
            output="Content Strategy: Weekly blog posts, bi-weekly videos, daily social media updates",
            duration=2.0
        )
    )
    subgraph.add_node(subtask2)

    # Subtask 3: EXECUTE node (directly executed)
    subtask3 = TaskNode(
        task_id=str(uuid4()),
        parent_id=root_task.task_id,
        goal="Plan social media campaigns",
        depth=1,
        max_depth=2,
        status=TaskStatus.COMPLETED,
        node_type=NodeType.EXECUTE
    )
    subtask3 = subtask3.record_module_execution(
        "executor",
        ModuleResult(
            module_name="executor",
            input=subtask3.goal,
            output="Social Media Plan: Instagram focus, TikTok for younger audience, LinkedIn for B2B",
            duration=1.8
        )
    )
    subgraph.add_node(subtask3)

    # Subtask 4: EXECUTE node (directly executed)
    subtask4 = TaskNode(
        task_id=str(uuid4()),
        parent_id=root_task.task_id,
        goal="Design email marketing",
        depth=1,
        max_depth=2,
        status=TaskStatus.COMPLETED,
        node_type=NodeType.EXECUTE
    )
    subtask4 = subtask4.record_module_execution(
        "executor",
        ModuleResult(
            module_name="executor",
            input=subtask4.goal,
            output="Email Campaign: Welcome series, weekly newsletters, promotional campaigns",
            duration=1.6
        )
    )
    subgraph.add_node(subtask4)

    # Add aggregator to root task
    root_task = root_task.record_module_execution(
        "aggregator",
        ModuleResult(
            module_name="aggregator",
            input="Combine all marketing strategy components",
            output="Complete Marketing Strategy:\n1. Market Analysis: Urban professionals 25-45\n2. Content: Weekly blogs, bi-weekly videos\n3. Social: Instagram/TikTok/LinkedIn\n4. Email: Welcome series + newsletters",
            duration=0.5
        )
    )

    # Update root task with final result
    root_task = root_task.model_copy(update={
        "result": "Comprehensive Marketing Strategy Document [Full strategy with all components integrated]"
    })
    dag.graph.nodes[root_task.task_id]['task'] = root_task

    return dag, root_task


def test_aggregation_visualizer():
    """Test the ExecutionVisualizer with aggregation tracking."""

    print("=" * 80)
    print("AGGREGATION TRACKING IN EXECUTION VISUALIZER")
    print("=" * 80)
    print("\n📊 This demo shows how the visualizer tracks:")
    print("   🔄 Tasks that aggregate subtask results")
    print("   ⚡ Tasks that execute directly without subtasks")
    print("   📁 Nested aggregation hierarchies")
    print()

    # Create mock DAG with aggregation patterns
    dag, root_task = create_mock_dag_with_aggregation()

    # Create visualizer
    visualizer = ExecutionVisualizer(
        use_colors=True,
        max_output_length=150,
        max_goal_length=60,
        show_timings=True,
        verbose=False
    )

    # Show full execution report
    print("\n" + "=" * 80)
    print("📊 FULL EXECUTION REPORT WITH AGGREGATION TRACKING")
    print("=" * 80)

    try:
        report = visualizer.get_full_execution_report(root_task, dag)
        print(report)
    except Exception as e:
        print(f"Error in get_full_execution_report: {e}")
        import traceback
        traceback.print_exc()

    # Show tree view with aggregation indicators
    print("\n" + "=" * 80)
    print("🌳 EXECUTION TREE WITH AGGREGATION INDICATORS")
    print("=" * 80)
    print("\nLegend:")
    print("  🔄 = Task aggregated subtask results")
    print("  ⚡ = Task executed directly (no subtasks)")
    print("  [PLAN] = Planning node (may have subtasks)")
    print("  [EXECUTE] = Execution node (atomic task)")
    print()

    try:
        tree = visualizer.get_execution_tree_with_details(root_task, dag)
        print(tree)
    except Exception as e:
        print(f"Error in get_execution_tree_with_details: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("✅ AGGREGATION TRACKING SUMMARY")
    print("=" * 80)
    print("\nThe visualizer now clearly shows:")
    print("• Which tasks aggregated results from subtasks (🔄)")
    print("• Which tasks executed directly without decomposition (⚡)")
    print("• The hierarchical aggregation flow")
    print("• Performance metrics for aggregation operations")
    print()


if __name__ == "__main__":
    test_aggregation_visualizer()