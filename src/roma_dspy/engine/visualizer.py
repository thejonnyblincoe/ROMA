"""
Visualization system for hierarchical task decomposition DAGs.
"""

from typing import Dict, Any, Optional, List, Tuple, Set
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from datetime import datetime
import json

from src.roma_dspy.engine.dag import TaskDAG
from src.roma_dspy.engine.tracker import ExecutionTracker
from src.roma_dspy.signatures.base_models.task_node import TaskNode
from src.roma_dspy.types.task_status import TaskStatus
from src.roma_dspy.types.node_type import NodeType


class DAGVisualizer:
    """
    Visualization system for task DAGs with hierarchical nesting.

    Features:
    - Hierarchical graph rendering with nested containers
    - Real-time state coloring
    - Execution history visualization
    - Multiple export formats
    - Interactive visualization support
    """

    # Color scheme for task states
    STATE_COLORS = {
        TaskStatus.PENDING: "#E0E0E0",      # Light gray
        TaskStatus.ATOMIZING: "#FFE082",    # Light amber
        TaskStatus.PLANNING: "#81C784",     # Light green
        TaskStatus.PLAN_DONE: "#4FC3F7",    # Light blue
        TaskStatus.READY: "#AED581",        # Light lime
        TaskStatus.EXECUTING: "#FFD54F",    # Amber
        TaskStatus.AGGREGATING: "#4DD0E1",  # Cyan
        TaskStatus.COMPLETED: "#66BB6A",    # Green
        TaskStatus.FAILED: "#EF5350",       # Red
        TaskStatus.NEEDS_REPLAN: "#FFAB91"  # Light orange
    }

    # Shape for node types
    NODE_SHAPES = {
        NodeType.PLAN: "box",
        NodeType.EXECUTE: "ellipse"
    }

    def __init__(self, dag: Optional[TaskDAG] = None, tracker: Optional[ExecutionTracker] = None):
        """
        Initialize the visualizer.

        Args:
            dag: TaskDAG to visualize
            tracker: Optional ExecutionTracker for enhanced visualization
        """
        self.dag = dag
        self.tracker = tracker
        self.figure = None
        self.axes = None

    def visualize(
        self,
        dag: Optional[TaskDAG] = None,
        show_depth: bool = True,
        show_history: bool = True,
        figsize: Tuple[int, int] = (16, 12),
        save_path: Optional[str] = None
    ) -> None:
        """
        Create a visualization of the task DAG.

        Args:
            dag: TaskDAG to visualize (uses self.dag if not provided)
            show_depth: Whether to show depth indicators
            show_history: Whether to show execution history
            figsize: Figure size for matplotlib
            save_path: Optional path to save the visualization
        """
        dag = dag or self.dag
        if not dag:
            raise ValueError("No DAG provided for visualization")

        # Create figure
        self.figure, self.axes = plt.subplots(figsize=figsize)
        self.axes.set_title(f"Task Decomposition DAG - {dag.dag_id[:8]}", fontsize=16, fontweight='bold')

        # Build NetworkX graph for layout
        display_graph = self._build_display_graph(dag)

        # Calculate layout
        pos = nx.spring_layout(display_graph, k=2, iterations=50)

        # Draw nested subgraphs first (background)
        self._draw_subgraphs(dag, pos)

        # Draw nodes
        self._draw_nodes(dag, display_graph, pos, show_depth, show_history)

        # Draw edges
        self._draw_edges(display_graph, pos)

        # Add legend
        self._add_legend()

        # Add statistics
        if self.tracker:
            self._add_statistics()

        # Clean up axes
        self.axes.axis('off')

        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.tight_layout()
        plt.show()

    def _build_display_graph(self, dag: TaskDAG) -> nx.DiGraph:
        """
        Build a NetworkX graph for display purposes.

        Args:
            dag: TaskDAG to convert

        Returns:
            NetworkX DiGraph for visualization
        """
        display_graph = nx.DiGraph()

        # Add all nodes
        for node_id in dag.graph.nodes():
            task = dag.get_node(node_id)
            display_graph.add_node(
                node_id,
                task=task,
                label=self._create_node_label(task)
            )

        # Add edges
        for from_id, to_id in dag.graph.edges():
            display_graph.add_edge(from_id, to_id)

        return display_graph

    def _create_node_label(self, task: TaskNode) -> str:
        """
        Create a label for a node.

        Args:
            task: TaskNode to label

        Returns:
            Label string
        """
        # Truncate goal for display
        goal_display = task.goal[:30] + "..." if len(task.goal) > 30 else task.goal

        # Add depth indicator
        depth_str = f"[D{task.depth}]" if task.depth > 0 else ""

        # Add forced execution indicator
        forced_str = " ⚡" if task.should_force_execute() else ""

        return f"{depth_str} {goal_display}{forced_str}"

    def _draw_nodes(
        self,
        dag: TaskDAG,
        display_graph: nx.DiGraph,
        pos: Dict,
        show_depth: bool,
        show_history: bool
    ) -> None:
        """
        Draw nodes with appropriate colors and shapes.

        Args:
            dag: TaskDAG being visualized
            display_graph: NetworkX graph for display
            pos: Position dictionary
            show_depth: Whether to show depth indicators
            show_history: Whether to show execution history
        """
        for node_id in display_graph.nodes():
            task = dag.get_node(node_id)
            x, y = pos[node_id]

            # Get node color based on status
            color = self.STATE_COLORS.get(task.status, "#FFFFFF")

            # Determine node shape based on type
            if task.should_force_execute():
                # Forced execution - diamond shape
                node_patch = mpatches.FancyBboxPatch(
                    (x - 0.15, y - 0.1),
                    0.3, 0.2,
                    boxstyle="round,pad=0.02",
                    facecolor=color,
                    edgecolor="black",
                    linewidth=2,
                    transform=self.axes.transData
                )
            elif task.node_type == NodeType.PLAN:
                # Planning node - rectangle
                node_patch = mpatches.FancyBboxPatch(
                    (x - 0.15, y - 0.08),
                    0.3, 0.16,
                    boxstyle="square,pad=0.01",
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.5,
                    transform=self.axes.transData
                )
            else:
                # Execute node - ellipse
                node_patch = mpatches.Ellipse(
                    (x, y),
                    0.3, 0.16,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=1.5,
                    transform=self.axes.transData
                )

            self.axes.add_patch(node_patch)

            # Add label
            label = self._create_node_label(task)
            self.axes.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')

            # Add execution history indicators
            if show_history and task.execution_history:
                modules_executed = list(task.execution_history.keys())
                history_str = ",".join([m[0].upper() for m in modules_executed])
                self.axes.text(x, y - 0.12, f"[{history_str}]", ha='center', va='center', fontsize=6)

    def _draw_edges(self, display_graph: nx.DiGraph, pos: Dict) -> None:
        """
        Draw edges between nodes.

        Args:
            display_graph: NetworkX graph for display
            pos: Position dictionary
        """
        for from_id, to_id in display_graph.edges():
            x1, y1 = pos[from_id]
            x2, y2 = pos[to_id]

            # Draw arrow
            self.axes.annotate(
                '',
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    lw=1.5,
                    connectionstyle="arc3,rad=0.1"
                )
            )

    def _draw_subgraphs(self, dag: TaskDAG, pos: Dict) -> None:
        """
        Draw nested subgraph containers.

        Args:
            dag: TaskDAG being visualized
            pos: Position dictionary
        """
        for subgraph_id, subgraph in dag.subgraphs.items():
            # Find all nodes in this subgraph
            subgraph_nodes = list(subgraph.graph.nodes())
            if not subgraph_nodes:
                continue

            # Calculate bounding box
            xs = [pos.get(n, (0, 0))[0] for n in subgraph_nodes if n in pos]
            ys = [pos.get(n, (0, 0))[1] for n in subgraph_nodes if n in pos]

            if xs and ys:
                min_x, max_x = min(xs) - 0.2, max(xs) + 0.2
                min_y, max_y = min(ys) - 0.2, max(ys) + 0.2

                # Draw container
                rect = FancyBboxPatch(
                    (min_x, min_y),
                    max_x - min_x,
                    max_y - min_y,
                    boxstyle="round,pad=0.05",
                    facecolor='lightblue',
                    edgecolor='blue',
                    alpha=0.2,
                    linewidth=2,
                    linestyle='--'
                )
                self.axes.add_patch(rect)

                # Add subgraph label
                self.axes.text(
                    min_x + 0.05,
                    max_y - 0.05,
                    f"Subgraph: {subgraph_id[:12]}...",
                    fontsize=7,
                    style='italic',
                    color='blue'
                )

    def _add_legend(self) -> None:
        """Add a legend explaining colors and shapes."""
        legend_elements = []

        # Status colors
        for status, color in self.STATE_COLORS.items():
            legend_elements.append(
                mpatches.Patch(color=color, label=status.value)
            )

        # Add legend
        self.axes.legend(
            handles=legend_elements,
            loc='upper left',
            bbox_to_anchor=(1, 1),
            title="Task Status",
            fontsize=8
        )

    def _add_statistics(self) -> None:
        """Add execution statistics to the visualization."""
        if not self.tracker:
            return

        stats = self.tracker.calculate_statistics()
        summary = stats['summary']

        # Create statistics text
        stats_text = f"""
        Execution Statistics:
        Total Nodes: {summary['total_nodes']}
        Completed: {summary['completed']}
        Failed: {summary['failed']}
        Success Rate: {summary['success_rate']:.1%}
        Duration: {summary.get('total_duration', 0):.2f}s
        """

        # Add to plot
        self.axes.text(
            0.02, 0.98,
            stats_text.strip(),
            transform=self.axes.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

    def export_to_graphml(self, dag: Optional[TaskDAG] = None, filepath: str = "dag.graphml") -> None:
        """
        Export DAG to GraphML format for use in other tools.

        Args:
            dag: TaskDAG to export
            filepath: Path to save the GraphML file
        """
        dag = dag or self.dag
        if not dag:
            raise ValueError("No DAG provided for export")

        # Build NetworkX graph with attributes
        export_graph = nx.DiGraph()

        for node_id in dag.graph.nodes():
            task = dag.get_node(node_id)
            export_graph.add_node(
                node_id,
                goal=task.goal,
                status=task.status.value,
                depth=task.depth,
                node_type=task.node_type.value if task.node_type else "unknown",
                result=str(task.result)[:100] if task.result else ""
            )

        for from_id, to_id in dag.graph.edges():
            export_graph.add_edge(from_id, to_id)

        # Export to GraphML
        nx.write_graphml(export_graph, filepath)

    def export_to_json(self, dag: Optional[TaskDAG] = None, filepath: str = "dag.json") -> None:
        """
        Export DAG to JSON format.

        Args:
            dag: TaskDAG to export
            filepath: Path to save the JSON file
        """
        dag = dag or self.dag
        if not dag:
            raise ValueError("No DAG provided for export")

        export_data = dag.export_to_dict()

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

    def create_execution_timeline(self, tracker: Optional[ExecutionTracker] = None) -> None:
        """
        Create a Gantt chart showing execution timeline.

        Args:
            tracker: ExecutionTracker with execution data
        """
        tracker = tracker or self.tracker
        if not tracker:
            raise ValueError("No tracker provided for timeline")

        fig, ax = plt.subplots(figsize=(14, 8))

        # Group events by node
        node_events = {}
        for event in tracker.execution_timeline:
            if event.node_id not in node_events:
                node_events[event.node_id] = []
            node_events[event.node_id].append(event)

        # Plot timeline for each node
        y_pos = 0
        labels = []

        for node_id, events in node_events.items():
            if node_id == "SYSTEM":
                continue

            # Find start and end times
            start_times = [e.timestamp for e in events]
            if not start_times:
                continue

            min_time = min(start_times)
            max_time = max(start_times)

            # Plot bar for execution duration
            if tracker.start_time:
                start_offset = (min_time - tracker.start_time).total_seconds()
                duration = (max_time - min_time).total_seconds()

                # Get node for color
                if node_id in tracker.nodes:
                    node = tracker.nodes[node_id]
                    color = self.STATE_COLORS.get(node.status, "gray")
                else:
                    color = "gray"

                ax.barh(y_pos, duration, left=start_offset, height=0.8, color=color, alpha=0.7)

            labels.append(node_id[:8])
            y_pos += 1

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time (seconds)")
        ax.set_title("Task Execution Timeline")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def visualize_dag(
    dag: TaskDAG,
    tracker: Optional[ExecutionTracker] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Convenience function to visualize a DAG.

    Args:
        dag: TaskDAG to visualize
        tracker: Optional ExecutionTracker
        save_path: Optional path to save the visualization
    """
    visualizer = DAGVisualizer(dag, tracker)
    visualizer.visualize(save_path=save_path)


class ExecutionVisualizer:
    """Provides detailed visualization of task execution."""

    @staticmethod
    def format_plan_details(task: TaskNode) -> str:
        """
        Format the planning details from a task node.

        Args:
            task: The task node with planning information

        Returns:
            Formatted string with plan details
        """
        lines = []

        if 'planner' not in task.execution_history:
            return "No planning information available."

        planner_result = task.execution_history['planner']
        output = planner_result.output

        lines.append("\n📝 PLANNING DETAILS")
        lines.append("=" * 80)

        if 'subtasks' in output:
            lines.append(f"\n📌 Generated {len(output['subtasks'])} subtasks:\n")
            for i, subtask in enumerate(output['subtasks'], 1):
                lines.append(f"  {i}. {subtask.get('goal', 'Unknown goal')}")
                lines.append(f"     Type: {subtask.get('task_type', 'Unknown')}")
                if subtask.get('dependencies'):
                    lines.append(f"     Dependencies: {', '.join(subtask['dependencies'])}")
                lines.append("")

        if 'dependencies' in output:
            lines.append("\n🔗 Dependency Graph:")
            deps = output['dependencies']
            if deps:
                for task_id, dep_list in deps.items():
                    if dep_list:
                        lines.append(f"  {task_id} → {', '.join(dep_list)}")
            else:
                lines.append("  No inter-task dependencies")

        return "\n".join(lines)

    @staticmethod
    def format_execution_details(task: TaskNode) -> str:
        """
        Format the execution details from atomic task nodes.

        Args:
            task: The task node with execution information

        Returns:
            Formatted string with execution details
        """
        lines = []

        if 'executor' not in task.execution_history:
            return ""

        executor_result = task.execution_history['executor']

        lines.append(f"\n⚡ EXECUTION NODE: {task.goal}")  # Show full goal
        lines.append("-" * 80)
        lines.append(f"Duration: {executor_result.duration:.2f}s")
        lines.append(f"\n📤 Output:")

        # Show full output for execute nodes
        output = executor_result.output
        if isinstance(output, dict) and 'output' in output:
            output_text = output['output']
        else:
            output_text = str(output)

        lines.append(output_text)

        return "\n".join(lines)

    @staticmethod
    def format_aggregation_details(task: TaskNode, dag: TaskDAG) -> str:
        """
        Format the aggregation details showing what was passed to the aggregator.

        Args:
            task: The task node with aggregation information
            dag: The DAG containing task relationships

        Returns:
            Formatted string with aggregation details
        """
        lines = []

        if 'aggregator' not in task.execution_history:
            return ""

        aggregator_result = task.execution_history['aggregator']

        lines.append("\n🔄 AGGREGATION DETAILS")
        lines.append("=" * 80)
        lines.append(f"Original Goal: {task.goal}")
        lines.append(f"Duration: {aggregator_result.duration:.2f}s")

        # Get subtasks that were aggregated
        if task.subgraph_id:
            subgraph = dag.get_subgraph(task.subgraph_id)
            if subgraph:
                lines.append(f"\n📥 Inputs to Aggregator ({len(subgraph.graph.nodes())} subtasks):")
                lines.append("-" * 40)

                for i, subtask_node in enumerate(subgraph.get_all_tasks(include_subgraphs=False), 1):
                    lines.append(f"\n{i}. Subtask: {subtask_node.goal}")
                    lines.append(f"   Type: {subtask_node.task_type.value if subtask_node.task_type else 'Unknown'}")
                    lines.append(f"   Status: {subtask_node.status.value}")

                    if subtask_node.result:
                        result_str = str(subtask_node.result)
                        if len(result_str) > 500:
                            # Show first and last parts for very long results
                            lines.append(f"   Result: {result_str[:400]}...")
                            lines.append(f"   ... [truncated {len(result_str) - 500} chars] ...")
                            lines.append(f"   ...{result_str[-100:]}")
                        else:
                            lines.append(f"   Result: {result_str}")
                    else:
                        lines.append("   Result: (No result)")

        lines.append(f"\n📤 Aggregated Output:")
        lines.append("-" * 40)
        lines.append(str(aggregator_result.output))

        return "\n".join(lines)

    @staticmethod
    def get_full_execution_report(task: TaskNode, dag: TaskDAG) -> str:
        """
        Generate a comprehensive execution report showing all details.

        Args:
            task: The root task node
            dag: The DAG containing all task relationships

        Returns:
            Complete formatted execution report
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("📊 COMPLETE EXECUTION REPORT")
        lines.append("=" * 80)

        # Basic task info
        lines.append(f"\n🎯 GOAL: {task.goal}")
        lines.append(f"📌 Status: {task.status.value}")
        lines.append(f"📐 Depth: {task.depth}/{task.max_depth}")

        # Show planning details if this was a PLAN node
        if task.node_type == NodeType.PLAN:
            lines.append(ExecutionVisualizer.format_plan_details(task))

            # Show execution details for all subtasks
            if task.subgraph_id:
                subgraph = dag.get_subgraph(task.subgraph_id)
                if subgraph:
                    lines.append("\n" + "=" * 80)
                    lines.append("⚡ SUBTASK EXECUTIONS")
                    lines.append("=" * 80)

                    for subtask_node in subgraph.get_all_tasks(include_subgraphs=False):
                        if subtask_node.node_type == NodeType.EXECUTE:
                            lines.append(ExecutionVisualizer.format_execution_details(subtask_node))
                        elif subtask_node.node_type == NodeType.PLAN:
                            # Recursively show nested plans
                            lines.append(f"\n📁 NESTED PLAN: {subtask_node.goal}")  # Show full goal
                            lines.append(ExecutionVisualizer.format_plan_details(subtask_node))
                            if subtask_node.subgraph_id:
                                nested_subgraph = subgraph.get_subgraph(subtask_node.subgraph_id)
                                if nested_subgraph:
                                    for nested_task in nested_subgraph.get_all_tasks(include_subgraphs=False):
                                        if nested_task.node_type == NodeType.EXECUTE:
                                            lines.append(ExecutionVisualizer.format_execution_details(nested_task))

            # Show aggregation details
            lines.append(ExecutionVisualizer.format_aggregation_details(task, dag))

        # Show execution details if this was an EXECUTE node
        elif task.node_type == NodeType.EXECUTE:
            lines.append(ExecutionVisualizer.format_execution_details(task))

        # Final result
        if task.result:
            lines.append("\n" + "=" * 80)
            lines.append("✨ FINAL RESULT")
            lines.append("=" * 80)
            lines.append(str(task.result))

        # Performance summary
        if task.metrics:
            lines.append("\n" + "=" * 80)
            lines.append("📈 PERFORMANCE METRICS")
            lines.append("=" * 80)
            if task.metrics.total_duration:
                lines.append(f"  Total Duration: {task.metrics.total_duration:.2f}s")
            if task.metrics.subtasks_created:
                lines.append(f"  Subtasks Created: {task.metrics.subtasks_created}")
            if task.metrics.atomizer_duration:
                lines.append(f"  Atomizer: {task.metrics.atomizer_duration:.2f}s")
            if task.metrics.planner_duration:
                lines.append(f"  Planner: {task.metrics.planner_duration:.2f}s")
            if task.metrics.executor_duration:
                lines.append(f"  Executor: {task.metrics.executor_duration:.2f}s")
            if task.metrics.aggregator_duration:
                lines.append(f"  Aggregator: {task.metrics.aggregator_duration:.2f}s")

        return "\n".join(lines)

    @staticmethod
    def get_execution_tree_with_details(task: TaskNode, dag: TaskDAG, indent: int = 0, visited: Optional[Set] = None) -> str:
        """
        Generate an execution tree with inline details.

        Args:
            task: Current task node
            dag: The DAG containing task relationships
            indent: Current indentation level
            visited: Set of visited task IDs

        Returns:
            Tree with execution details
        """
        if visited is None:
            visited = set()

        if task.task_id in visited:
            return f"{'  ' * indent}↺ {task.task_id[:8]}... (circular reference)"

        visited.add(task.task_id)

        lines = []
        prefix = "  " * indent

        # Status emoji
        status_emoji = {
            "COMPLETED": "✅",
            "FAILED": "❌",
            "EXECUTING": "⚙️",
            "PENDING": "⏳",
            "PLANNING": "📝",
            "PLAN_DONE": "✔️",
            "AGGREGATING": "🔄"
        }.get(task.status.value, "❓")

        # Node type
        node_type_str = ""
        if task.node_type:
            node_type_str = f"[{task.node_type.value}]"

        # Main node line - show full goal
        lines.append(f"{prefix}{status_emoji} {task.goal} {node_type_str}")

        # Add execution details for EXECUTE nodes
        if task.node_type == NodeType.EXECUTE and task.result:
            result_preview = str(task.result)[:100]
            if len(str(task.result)) > 100:
                result_preview += "..."
            lines.append(f"{prefix}  └─ Result: {result_preview}")

        # Add children if it's a PLAN node
        if dag and task.subgraph_id:
            subgraph = dag.get_subgraph(task.subgraph_id)
            if subgraph:
                children = list(subgraph.get_all_tasks(include_subgraphs=False))
                for i, child_task in enumerate(children):
                    is_last = (i == len(children) - 1)
                    connector = "└─" if is_last else "├─"
                    child_tree = ExecutionVisualizer.get_execution_tree_with_details(child_task, subgraph, indent + 2, visited)
                    lines.append(f"{prefix}  {connector} {child_tree}")

        return "\n".join(lines)