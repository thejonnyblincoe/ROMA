"""
Enhanced Execution Visualizer for ROMA-DSPy hierarchical task decomposition.

Provides clean, readable visualization of solver execution with:
- Structured execution reports
- Hierarchical tree views
- Performance metrics
- Smart output formatting
"""

from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime
from collections import defaultdict
import json
import re

from src.roma_dspy.signatures.base_models.task_node import TaskNode
from src.roma_dspy.types.task_status import TaskStatus
from src.roma_dspy.types.node_type import NodeType
from src.roma_dspy.engine.dag import TaskDAG


class ExecutionVisualizer:
    """
    Enhanced visualizer for ROMA-DSPy execution results.
    Provides clean, structured output with improved readability.
    """

    def __init__(
        self,
        use_colors: bool = True,
        max_output_length: int = 200,
        max_goal_length: int = 80,
        show_timings: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the ExecutionVisualizer.

        Args:
            use_colors: Whether to use ANSI color codes
            max_output_length: Maximum length for output previews
            max_goal_length: Maximum length for goal descriptions
            show_timings: Whether to show execution timings
            verbose: Whether to show detailed information
        """
        self.use_colors = use_colors
        self.max_output_length = max_output_length
        self.max_goal_length = max_goal_length
        self.show_timings = show_timings
        self.verbose = verbose

        # ANSI color codes
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'red': '\033[31m',
            'bright_green': '\033[92m',
            'bright_blue': '\033[94m',
            'bright_cyan': '\033[96m',
        }

    def _color(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors and color in self.colors:
            return f"{self.colors[color]}{text}{self.colors['reset']}"
        return text

    def _truncate(self, text: str, max_length: int, suffix: str = "...") -> str:
        """Truncate text to maximum length with suffix."""
        if not text:
            return ""
        text = str(text).strip()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 0.001:
            return f"{seconds*1000000:.0f}μs"
        elif seconds < 1:
            return f"{seconds*1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds / 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"

    def _get_status_symbol(self, status: TaskStatus) -> str:
        """Get symbol for task status."""
        symbols = {
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.PENDING: "⏳",
            TaskStatus.EXECUTING: "⚡",
            TaskStatus.PLANNING: "📝",
            TaskStatus.READY: "🟢",
            TaskStatus.ATOMIZING: "🔍",
            TaskStatus.AGGREGATING: "🔄",
            TaskStatus.PLAN_DONE: "✔️",
            TaskStatus.NEEDS_REPLAN: "🔁"
        }
        return symbols.get(status, "❓")

    def _get_node_type_symbol(self, node_type: NodeType) -> str:
        """Get symbol for node type."""
        if node_type == NodeType.PLAN:
            return "[PLAN]"
        elif node_type == NodeType.EXECUTE:
            return "[EXECUTE]"
        return ""

    def _format_header(self, title: str, width: int = 80) -> str:
        """Format a section header."""
        separator = "=" * width
        return f"\n{self._color(separator, 'cyan')}\n{self._color(title, 'bold')}\n{self._color(separator, 'cyan')}"

    def _format_subheader(self, title: str, emoji: str = "", width: int = 80) -> str:
        """Format a subsection header."""
        separator = "-" * width
        header = f"{emoji} {title}" if emoji else title
        return f"\n{self._color(header, 'bright_blue')}\n{separator}"

    def get_full_execution_report(self, result: Any, dag: TaskDAG) -> str:
        """
        Generate a comprehensive execution report.

        Args:
            result: The execution result
            dag: The task DAG

        Returns:
            Formatted execution report string
        """
        lines = []

        # Main header
        lines.append(self._format_header("📊 COMPLETE EXECUTION REPORT"))

        # Find root task
        root_task = self._find_root_task(dag)
        if not root_task:
            return "No root task found in execution DAG"

        # Goal and status section
        lines.append("")
        lines.append(f"🎯 {self._color('GOAL:', 'bold')} {self._truncate(root_task.goal, self.max_goal_length)}")
        lines.append(f"📌 {self._color('Status:', 'bold')} {self._get_status_symbol(root_task.status)} {self._color(root_task.status.value, 'bright_green')}")
        lines.append(f"📐 {self._color('Depth:', 'bold')} {root_task.depth}/{root_task.max_depth}")

        # Planning information - check if there's planner output in execution history
        if root_task.execution_history and 'planner' in root_task.execution_history:
            planner_result = root_task.execution_history['planner']
            if planner_result.output:
                lines.append(f"📋 {self._color('Planning Context:', 'bold')}")
                lines.append(f"   {self._truncate(str(planner_result.output), 150)}")
        else:
            lines.append(f"{self._color('No planning information available.', 'dim')}")

        # Subtask executions
        lines.append(self._format_header("⚡ SUBTASK EXECUTIONS"))
        lines.extend(self._format_task_executions(dag, root_task))

        # Performance metrics
        lines.append(self._format_header("📈 PERFORMANCE METRICS"))
        lines.extend(self._format_performance_metrics(dag))

        return "\n".join(lines)

    def _format_task_executions(self, dag: TaskDAG, root_task: TaskNode, indent: int = 0) -> List[str]:
        """Format task executions recursively."""
        lines = []

        # Process subgraph if exists
        if root_task.subgraph_id and root_task.subgraph_id in dag.subgraphs:
            subgraph = dag.subgraphs[root_task.subgraph_id]

            # Get all tasks in subgraph
            for task_id in subgraph.graph.nodes():
                task = subgraph.get_node(task_id)
                lines.extend(self._format_single_task_execution(task, subgraph, indent))

        return lines

    def _format_single_task_execution(self, task: TaskNode, dag: TaskDAG, indent: int) -> List[str]:
        """Format a single task execution."""
        lines = []
        indent_str = "  " * indent

        # Task header with aggregation indicator
        node_type = self._get_node_type_symbol(task.node_type) if task.node_type else ""

        # Check if this task has an aggregator (meaning its subtasks were aggregated)
        has_aggregator = task.execution_history and 'aggregator' in task.execution_history
        aggregation_marker = " 🔄 [AGGREGATED]" if has_aggregator else ""

        task_header = f"{indent_str}📁 {self._color(node_type, 'magenta')} {self._color(self._truncate(task.goal, self.max_goal_length), 'bright_cyan')}{self._color(aggregation_marker, 'yellow')}"
        lines.append("")
        lines.append(task_header)

        # Planning context if available from execution history
        if task.execution_history and 'planner' in task.execution_history and self.verbose:
            planner_result = task.execution_history['planner']
            if planner_result.output:
                lines.append(f"{indent_str}   {self._color('Planning:', 'dim')} {self._truncate(str(planner_result.output), 100)}")
        elif task.node_type == NodeType.PLAN and not (task.execution_history and 'planner' in task.execution_history):
            lines.append(f"{indent_str}   {self._color('No planning information available.', 'dim')}")

        # Execution details with special handling for aggregation
        if task.execution_history:
            for module_name, exec_result in task.execution_history.items():
                emoji = self._get_module_emoji(module_name)
                duration_str = self._format_duration(exec_result.duration) if self.show_timings else ""

                # Special formatting for aggregator results
                if module_name == 'aggregator':
                    lines.append("")
                    lines.append(f"{indent_str}🔄 {self._color('AGGREGATION RESULT:', 'yellow')} Combined outputs from subtasks")
                    lines.append(f"{indent_str}{'-' * 80}")

                    if duration_str:
                        lines.append(f"{indent_str}{self._color('Aggregation Duration:', 'dim')} {duration_str}")

                    if exec_result.output:
                        lines.append(f"{indent_str}")
                        lines.append(f"{indent_str}📦 {self._color('Aggregated Output:', 'bold')}")
                        output_lines = self._format_output(exec_result.output, indent + 1)
                        lines.extend(output_lines)
                else:
                    # Regular execution node
                    lines.append("")
                    execution_type = "EXECUTION NODE:" if module_name == 'executor' else f"{module_name.upper()} MODULE:"
                    lines.append(f"{indent_str}{emoji} {self._color(execution_type, 'bold')} {self._truncate(task.goal, self.max_goal_length)}")
                    lines.append(f"{indent_str}{'-' * 80}")

                    if duration_str:
                        lines.append(f"{indent_str}{self._color('Duration:', 'dim')} {duration_str}")

                    # Output
                    if exec_result.output:
                        lines.append(f"{indent_str}")
                        lines.append(f"{indent_str}📤 {self._color('Output:', 'bold')}")
                        output_lines = self._format_output(exec_result.output, indent + 1)
                        lines.extend(output_lines)

        # Process nested subgraphs
        if task.subgraph_id and task.subgraph_id in dag.subgraphs:
            subgraph = dag.subgraphs[task.subgraph_id]
            for sub_task_id in subgraph.graph.nodes():
                sub_task = subgraph.get_node(sub_task_id)
                lines.extend(self._format_single_task_execution(sub_task, subgraph, indent + 1))

        return lines

    def _format_output(self, output: Any, indent: int) -> List[str]:
        """Format task output with proper indentation."""
        lines = []
        indent_str = "  " * indent

        output_str = str(output)
        # Smart truncation - try to keep meaningful content
        if len(output_str) > self.max_output_length:
            # Try to find a natural break point
            truncated = self._truncate(output_str, self.max_output_length, "\n...")
        else:
            truncated = output_str

        # Format multi-line output
        for line in truncated.split('\n'):
            if line.strip():
                lines.append(f"{indent_str}{line}")

        return lines

    def _get_module_emoji(self, module_name: str) -> str:
        """Get emoji for module type."""
        emojis = {
            "atomizer": "🔍",
            "planner": "📝",
            "executor": "⚡",
            "aggregator": "🔄",
            "verifier": "✓"
        }
        return emojis.get(module_name.lower(), "📦")

    def _format_performance_metrics(self, dag: TaskDAG) -> List[str]:
        """Format performance metrics section."""
        lines = []

        # Collect metrics
        total_duration = 0
        subtask_count = 0
        module_durations = defaultdict(float)
        aggregated_count = 0
        executed_directly_count = 0
        plan_nodes = 0
        execute_nodes = 0

        all_tasks = dag.get_all_tasks(include_subgraphs=True)

        for task in all_tasks:
            subtask_count += 1

            # Count node types
            if task.node_type == NodeType.PLAN:
                plan_nodes += 1
            elif task.node_type == NodeType.EXECUTE:
                execute_nodes += 1

            # Check if aggregated
            has_aggregator = task.execution_history and 'aggregator' in task.execution_history
            if has_aggregator:
                aggregated_count += 1
            elif task.node_type == NodeType.EXECUTE:
                executed_directly_count += 1

            if task.execution_history:
                for module_name, exec_result in task.execution_history.items():
                    module_durations[module_name] += exec_result.duration
                    total_duration += exec_result.duration

        # Format metrics
        lines.append(f"  {self._color('Total Duration:', 'bold')} {self._format_duration(total_duration)}")
        lines.append(f"  {self._color('Total Tasks:', 'bold')} {subtask_count}")
        lines.append("")

        # Node type breakdown
        lines.append(f"  {self._color('Task Breakdown:', 'bold')}")
        lines.append(f"    • Plan Nodes: {plan_nodes} {self._color('(decomposed into subtasks)', 'dim')}")
        lines.append(f"    • Execute Nodes: {execute_nodes} {self._color('(executed directly)', 'dim')}")
        lines.append("")

        # Aggregation statistics
        lines.append(f"  {self._color('Aggregation Statistics:', 'bold')}")
        lines.append(f"    • Tasks with Aggregation: {aggregated_count} {self._color('🔄', 'yellow')}")
        lines.append(f"    • Tasks Executed Directly: {executed_directly_count} {self._color('⚡', 'cyan')}")
        lines.append("")

        # Module breakdown
        lines.append(f"  {self._color('Module Execution Times:', 'bold')}")
        for module, duration in sorted(module_durations.items()):
            emoji = self._get_module_emoji(module)
            lines.append(f"    {emoji} {self._color(module.capitalize() + ':', 'dim')} {self._format_duration(duration)}")

        return lines

    def get_execution_tree_with_details(self, result: Any, dag: TaskDAG) -> str:
        """
        Generate a hierarchical tree view with execution details.

        Args:
            result: The execution result
            dag: The task DAG

        Returns:
            Formatted tree view string
        """
        lines = []

        # Header
        lines.append(self._format_header("🌳 EXECUTION TREE WITH DETAILS"))

        # Find root task
        root_task = self._find_root_task(dag)
        if not root_task:
            return "No root task found in execution DAG"

        # Build tree
        self._build_tree_view(root_task, dag, lines, 0, set(), is_last=True, prefix="")

        return "\n".join(lines)

    def _build_tree_view(
        self,
        task: TaskNode,
        dag: TaskDAG,
        lines: List[str],
        depth: int,
        visited: Set[str],
        is_last: bool,
        prefix: str
    ):
        """Build hierarchical tree view recursively."""
        if task.task_id in visited:
            lines.append(f"{prefix}↺ (circular reference)")
            return

        visited.add(task.task_id)

        # Determine connectors
        if depth == 0:
            connector = ""
            child_prefix = ""
        else:
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("    " if is_last else "│   ")

        # Format node with aggregation indicator
        status_symbol = self._get_status_symbol(task.status)
        node_type = self._get_node_type_symbol(task.node_type) if task.node_type else ""
        goal = self._truncate(task.goal, self.max_goal_length)

        # Check if this task aggregated subtasks
        has_aggregator = task.execution_history and 'aggregator' in task.execution_history
        has_subtasks = task.subgraph_id and task.subgraph_id in dag.subgraphs

        aggregation_info = ""
        if has_aggregator and has_subtasks:
            aggregation_info = self._color(" 🔄", 'yellow')
        elif task.node_type == NodeType.EXECUTE:
            aggregation_info = self._color(" ⚡", 'cyan')

        node_line = f"{prefix}{connector}{status_symbol} {goal} {self._color(node_type, 'magenta')}{aggregation_info}"
        lines.append(node_line)

        # Add execution result if available
        if task.result:
            result_preview = self._truncate(str(task.result), 150)
            result_line = f"{child_prefix}└─ Result: {self._color(result_preview, 'dim')}"
            lines.append(result_line)

        # Process subgraph
        if task.subgraph_id and task.subgraph_id in dag.subgraphs:
            subgraph = dag.subgraphs[task.subgraph_id]
            children = list(subgraph.graph.nodes())

            for i, child_id in enumerate(children):
                child_task = subgraph.get_node(child_id)
                is_child_last = (i == len(children) - 1)

                # Recursive call with proper prefix
                self._build_tree_view(
                    child_task,
                    subgraph,
                    lines,
                    depth + 1,
                    visited,
                    is_child_last,
                    child_prefix
                )

    def _find_root_task(self, dag: TaskDAG) -> Optional[TaskNode]:
        """Find the root task in the DAG."""
        for node_id in dag.graph.nodes():
            task = dag.get_node(node_id)
            if task.is_root:
                return task
        # If no explicit root, find node with no predecessors
        for node_id in dag.graph.nodes():
            if not list(dag.graph.predecessors(node_id)):
                return dag.get_node(node_id)
        return None

    def export_to_markdown(self, result: Any, dag: TaskDAG) -> str:
        """Export visualization to Markdown format."""
        lines = []

        lines.append("# ROMA-DSPy Execution Report")
        lines.append("")

        # Temporarily disable colors for markdown
        original_use_colors = self.use_colors
        self.use_colors = False

        # Add execution report
        lines.append("## Execution Summary")
        lines.append("```")
        lines.append(self.get_full_execution_report(result, dag))
        lines.append("```")

        lines.append("")
        lines.append("## Execution Tree")
        lines.append("```")
        lines.append(self.get_execution_tree_with_details(result, dag))
        lines.append("```")

        # Restore color setting
        self.use_colors = original_use_colors

        return "\n".join(lines)

    def export_to_json(self, result: Any, dag: TaskDAG) -> str:
        """Export execution data to JSON format."""
        data = {
            "execution_id": dag.dag_id,
            "metadata": {
                "created_at": str(dag.metadata.get('created_at')),
                "completed_at": str(dag.metadata.get('execution_completed_at')),
            },
            "root_task": None,
            "tasks": [],
            "statistics": dag.get_statistics()
        }

        # Find and process root task
        root_task = self._find_root_task(dag)
        if root_task:
            data["root_task"] = {
                "task_id": root_task.task_id,
                "goal": root_task.goal,
                "status": root_task.status.value,
                "depth": root_task.depth,
                "max_depth": root_task.max_depth,
                "result": str(root_task.result) if root_task.result else None
            }

        # Process all tasks
        all_tasks = dag.get_all_tasks(include_subgraphs=True)
        for task in all_tasks:
            task_data = {
                "task_id": task.task_id,
                "goal": task.goal,
                "status": task.status.value,
                "node_type": task.node_type.value if task.node_type else None,
                "depth": task.depth,
                "execution_history": {}
            }

            # Add execution history
            if task.execution_history:
                for module_name, exec_result in task.execution_history.items():
                    task_data["execution_history"][module_name] = {
                        "duration": exec_result.duration,
                        "output": str(exec_result.output) if exec_result.output else None
                    }

            data["tasks"].append(task_data)

        return json.dumps(data, indent=2)