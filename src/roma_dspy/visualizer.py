"""
Execution visualizer for ROMA-DSPy hierarchical task decomposition.

Provides multiple visualization modes for understanding the solver's execution:
- Real-time execution tracking with colored output
- Hierarchical tree visualization
- Timeline view of module executions
- Statistics and metrics dashboard
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple
from enum import Enum
import time
from collections import defaultdict

from roma_dspy.core.signatures.base_models.task_node import TaskNode
from roma_dspy.types import TaskStatus, NodeType
from roma_dspy.core.engine.dag import TaskDAG


def _resolve_visualization_inputs(
    source: Optional[Any],
    dag: Optional[TaskDAG]
) -> Tuple[Optional[TaskDAG], Optional[TaskNode]]:
    """Resolve caller input into a TaskDAG and optional root task."""
    resolved_dag = dag
    root_task: Optional[TaskNode] = None

    if isinstance(source, TaskDAG):
        resolved_dag = source
    elif isinstance(source, TaskNode):
        root_task = source
    elif source is not None:
        # Check if it's a RecursiveSolverModule wrapper
        if hasattr(source, '_solver') and hasattr(source._solver, 'last_dag'):
            resolved_dag = dag or source._solver.last_dag
        else:
            resolved_dag = dag or getattr(source, "last_dag", None)

    if resolved_dag and root_task is None and hasattr(resolved_dag, "graph"):
        try:
            for node_id in resolved_dag.graph.nodes():
                candidate = resolved_dag.get_node(node_id)
                if candidate.is_root:
                    root_task = candidate
                    break
        except Exception:
            pass

    return resolved_dag, root_task


class ColorCode(Enum):
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


class RealTimeVisualizer:
    """
    Real-time visualization of solver execution.
    Shows each step as it happens with colored output and progress indicators.
    """

    def __init__(self, use_colors: bool = True, verbose: bool = True):
        """
        Initialize real-time visualizer.

        Args:
            use_colors: Whether to use ANSI colors in output
            verbose: Whether to show detailed information
        """
        self.use_colors = use_colors
        self.verbose = verbose
        self.execution_stack: List[Dict[str, Any]] = []
        self.start_time = None
        self.events: List[Dict[str, Any]] = []

    def color(self, text: str, color: ColorCode) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return f"{color.value}{text}{ColorCode.RESET.value}"
        return text

    def get_status_emoji(self, status: TaskStatus) -> str:
        """Get emoji for task status."""
        return {
            TaskStatus.PENDING: "⏳",
            TaskStatus.ATOMIZING: "🔍",
            TaskStatus.PLANNING: "📝",
            TaskStatus.PLAN_DONE: "✔️",
            TaskStatus.READY: "🟢",
            TaskStatus.EXECUTING: "⚡",
            TaskStatus.AGGREGATING: "🔄",
            TaskStatus.COMPLETED: "✅",
            TaskStatus.FAILED: "❌",
            TaskStatus.NEEDS_REPLAN: "🔁"
        }.get(status, "❓")

    def get_module_emoji(self, module_name: str) -> str:
        """Get emoji for module type."""
        return {
            "atomizer": "🔍",
            "planner": "📝",
            "executor": "⚡",
            "aggregator": "🔄",
            "verifier": "✓"
        }.get(module_name.lower(), "📦")

    def format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds / 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.1f}s"

    def print_header(self, title: str, level: int = 0):
        """Print a formatted header."""
        indent = "  " * level
        separator = "=" * (60 - level * 2)

        print(f"\n{indent}{self.color(separator, ColorCode.CYAN)}")
        print(f"{indent}{self.color(f'🚀 {title}', ColorCode.BOLD)}")
        print(f"{indent}{self.color(separator, ColorCode.CYAN)}")

    def on_execution_start(self, task: TaskNode):
        """Called when solver execution starts."""
        self.start_time = time.time()
        self.print_header("SOLVER EXECUTION STARTED")
        print(f"📋 Task: {self.color(task.goal, ColorCode.BRIGHT_CYAN)}")
        print(f"🎯 Max Depth: {self.color(str(task.max_depth), ColorCode.YELLOW)}")
        print()

    def on_task_enter(self, task: TaskNode, depth: int):
        """Called when entering a task."""
        indent = "  " * depth
        prefix = "├── " if depth > 0 else ""

        # Show task entry
        print(f"{indent}{prefix}{self.get_status_emoji(task.status)} "
              f"[Depth {self.color(str(depth), ColorCode.YELLOW)}] "
              f"{self.color(task.goal[:80], ColorCode.BRIGHT_BLUE)}")

        if self.verbose and task.task_id:
            print(f"{indent}    └─ ID: {self.color(task.task_id[:8], ColorCode.DIM)}")

        # Track in execution stack
        self.execution_stack.append({
            'task_id': task.task_id,
            'goal': task.goal,
            'depth': depth,
            'enter_time': time.time()
        })

    def on_module_start(self, task: TaskNode, module_name: str, depth: int):
        """Called when a module starts executing."""
        indent = "  " * (depth + 1)
        emoji = self.get_module_emoji(module_name)

        print(f"{indent}{emoji} {self.color(module_name.upper(), ColorCode.MAGENTA)} starting...")

        # Record event
        self.events.append({
            'type': 'module_start',
            'task_id': task.task_id,
            'module': module_name,
            'timestamp': time.time(),
            'depth': depth
        })

    def on_module_complete(self, task: TaskNode, module_name: str, result: Any, duration: float, depth: int):
        """Called when a module completes execution."""
        indent = "  " * (depth + 1)
        emoji = self.get_module_emoji(module_name)

        # Format result preview
        result_str = str(result)
        if len(result_str) > 60:
            result_str = result_str[:60] + "..."

        print(f"{indent}{emoji} {self.color(module_name.upper(), ColorCode.GREEN)} "
              f"completed in {self.color(self.format_duration(duration), ColorCode.YELLOW)}")

        if self.verbose:
            print(f"{indent}   → Result: {self.color(result_str, ColorCode.DIM)}")

        # Record event
        self.events.append({
            'type': 'module_complete',
            'task_id': task.task_id,
            'module': module_name,
            'timestamp': time.time(),
            'duration': duration,
            'depth': depth
        })

    def on_status_change(self, task: TaskNode, old_status: TaskStatus, new_status: TaskStatus, depth: int):
        """Called when task status changes."""
        indent = "  " * (depth + 1)

        old_emoji = self.get_status_emoji(old_status)
        new_emoji = self.get_status_emoji(new_status)

        print(f"{indent}🔄 Status: {old_emoji} {self.color(old_status.value, ColorCode.DIM)} "
              f"→ {new_emoji} {self.color(new_status.value, ColorCode.BRIGHT_GREEN)}")

    def on_subtasks_created(self, parent: TaskNode, subtasks: List[TaskNode], depth: int):
        """Called when subtasks are created."""
        indent = "  " * (depth + 1)

        print(f"{indent}📂 Created {self.color(str(len(subtasks)), ColorCode.YELLOW)} subtasks:")
        for i, subtask in enumerate(subtasks[:5]):  # Show first 5
            prefix = "├──" if i < len(subtasks) - 1 else "└──"
            print(f"{indent}  {prefix} {subtask.goal[:60]}")

        if len(subtasks) > 5:
            print(f"{indent}      ... and {len(subtasks) - 5} more")

    def on_task_complete(self, task: TaskNode, depth: int):
        """Called when a task completes."""
        if self.execution_stack and self.execution_stack[-1]['task_id'] == task.task_id:
            stack_entry = self.execution_stack.pop()
            duration = time.time() - stack_entry['enter_time']

            indent = "  " * depth
            print(f"{indent}✨ Task completed in {self.color(self.format_duration(duration), ColorCode.GREEN)}")

            if task.result and self.verbose:
                result_preview = str(task.result)[:100]
                print(f"{indent}   📄 Result: {self.color(result_preview, ColorCode.DIM)}")

    def on_execution_complete(self, root_task: TaskNode):
        """Called when entire execution completes."""
        if self.start_time:
            total_duration = time.time() - self.start_time

            self.print_header("EXECUTION COMPLETE")
            print(f"✅ Status: {self.color('SUCCESS', ColorCode.BRIGHT_GREEN)}")
            print(f"⏱️  Total Time: {self.color(self.format_duration(total_duration), ColorCode.YELLOW)}")
            print(f"📊 Events Recorded: {self.color(str(len(self.events)), ColorCode.CYAN)}")
            print()


class TreeVisualizer:
    """
    Enhanced tree visualization for hierarchical task decomposition.
    Shows the complete execution tree with detailed information.
    """

    def __init__(self, use_colors: bool = True, show_ids: bool = False,
                 show_timing: bool = True, show_tokens: bool = True,
                 max_goal_length: int = 60):
        """
        Initialize tree visualizer.

        Args:
            use_colors: Whether to use ANSI colors
            show_ids: Whether to show task IDs
            show_timing: Whether to show timing information
            show_tokens: Whether to show token usage and costs
            max_goal_length: Maximum goal text length (0 = unlimited)
        """
        self.use_colors = use_colors
        self.show_ids = show_ids
        self.show_timing = show_timing
        self.show_tokens = show_tokens
        self.max_goal_length = max_goal_length
        self.rt_viz = RealTimeVisualizer(use_colors=use_colors)  # Reuse color methods

    def visualize(self, source: Optional[Any] = None, dag: Optional[TaskDAG] = None) -> str:
        """
        Generate tree visualization from solver's last DAG or dict snapshot.

        Args:
            source: RecursiveSolver, TaskDAG, TaskNode, or dict snapshot to visualize
            dag: Optional DAG to visualize (defaults to solver.last_dag if available)

        Returns:
            String representation of the tree
        """
        # Check if source is a dict snapshot from storage
        if isinstance(source, dict) and 'tasks' in source:
            return self.visualize_from_snapshot(source)

        dag, root_task = _resolve_visualization_inputs(source, dag)

        if not dag and root_task is None:
            return "No execution data available. Run solve() first."

        # Find root task
        if root_task is None and dag:
            root_task = self._find_root_task(dag)
        if not root_task:
            return "No root task found in DAG."

        lines = []
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
        lines.append(self.rt_viz.color("📊 HIERARCHICAL TASK DECOMPOSITION TREE", ColorCode.BOLD))
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
        lines.append("")

        # Build tree recursively
        self._build_tree(root_task, dag, lines, 0, set(), is_last=True, prefix="")

        # Add statistics
        lines.append("")
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))

        # Add token metrics if enabled
        if self.show_tokens:
            tree_metrics = root_task.get_tree_metrics(dag)
            if tree_metrics.total_tokens > 0:
                lines.append(self.rt_viz.color("💰 TREE TOTALS", ColorCode.BOLD))
                lines.append(f"  Total Prompt Tokens: {tree_metrics.prompt_tokens:,}")
                lines.append(f"  Total Completion Tokens: {tree_metrics.completion_tokens:,}")
                lines.append(f"  Total Tokens: {tree_metrics.total_tokens:,}")
                cost_str = f"  Total Cost: ${tree_metrics.cost:.6f}"
                lines.append(self.rt_viz.color(cost_str, ColorCode.BRIGHT_GREEN))
                lines.append("")

        if dag:
            lines.extend(self._generate_statistics(dag))
        else:
            lines.append(self.rt_viz.color(
                "Detailed statistics unavailable without TaskDAG context.",
                ColorCode.DIM
            ))

        return "\n".join(lines)

    def visualize_from_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """
        Generate tree visualization from dict snapshot (from PostgreSQL storage).

        Args:
            snapshot: Dict snapshot with keys: 'tasks', 'subgraphs', 'statistics', 'dag_id'

        Returns:
            String representation of the tree
        """
        from roma_dspy.core.signatures.base_models.task_node import TaskNode

        # Validate snapshot structure
        if not isinstance(snapshot, dict):
            return "Invalid snapshot format. Expected dict."

        if 'tasks' not in snapshot:
            return "Invalid snapshot format. Expected dict with 'tasks' key."

        tasks_data = snapshot['tasks']
        subgraphs = snapshot.get('subgraphs', {})
        statistics = snapshot.get('statistics', {})

        if not tasks_data:
            return "No tasks found in snapshot."

        # tasks_data is a dict mapping task_id -> task_data
        tasks = tasks_data

        # Find root node (prefer is_root flag, fallback to depth 0)
        root_node_data = None
        for node_data in tasks.values():
            # New format: use is_root field
            if node_data.get('is_root', False):
                root_node_data = node_data
                break
            # Legacy format: use depth 0
            if node_data.get('depth', -1) == 0 and root_node_data is None:
                root_node_data = node_data

        if not root_node_data:
            return "No root task found in snapshot."

        # Build header
        lines = []
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
        lines.append(self.rt_viz.color("📊 HIERARCHICAL TASK DECOMPOSITION TREE (FROM STORAGE)", ColorCode.BOLD))
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
        lines.append("")

        # Build tree recursively from snapshot
        visited = set()
        self._build_tree_from_snapshot(
            root_node_data,
            tasks,
            subgraphs,
            lines,
            0,
            visited,
            is_last=True,
            prefix=""
        )

        # Add statistics
        lines.append("")
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))

        if statistics:
            lines.extend(self._generate_statistics_from_snapshot(statistics))
        else:
            lines.append(self.rt_viz.color(
                "Detailed statistics unavailable in snapshot.",
                ColorCode.DIM
            ))

        return "\n".join(lines)

    def _build_tree_from_snapshot(
        self,
        node_data: Dict[str, Any],
        all_nodes: Dict[str, Any],
        all_subgraphs: Dict[str, Any],
        lines: List[str],
        depth: int,
        visited: Set[str],
        is_last: bool,
        prefix: str
    ):
        """Recursively build tree representation from snapshot dict."""
        # Snapshot uses 'task_id', not 'id'
        node_id = node_data.get('task_id') or node_data.get('id')

        if not node_id:
            lines.append(f"{prefix}⚠️ (task missing ID)")
            return

        if node_id in visited:
            lines.append(f"{prefix}↺ (circular reference to {node_id[:8]}...)")
            return

        visited.add(node_id)

        # Build current node line
        connector = "└── " if is_last else "├── "
        node_line = self._format_node_from_snapshot(node_data, depth)

        lines.append(f"{prefix}{connector}{node_line}")

        # Add task details
        if self.show_ids:
            detail_prefix = prefix + ("    " if is_last else "│   ")
            node_id_short = node_id[:8] if node_id else "unknown"
            lines.append(f"{detail_prefix}ID: {self.rt_viz.color(node_id_short, ColorCode.DIM)}")

        # Add execution history from snapshot (it's a list of module names)
        execution_history = node_data.get('execution_history', [])
        if execution_history and self.show_timing:
            detail_prefix = prefix + ("    " if is_last else "│   ")

            # execution_history is a list of module names
            if isinstance(execution_history, list):
                for module_name in execution_history:
                    emoji = self.rt_viz.get_module_emoji(module_name)
                    lines.append(f"{detail_prefix}{emoji} {module_name}")
            # If it's a dict (from newer snapshots), handle it
            elif isinstance(execution_history, dict):
                for module_name, module_data in execution_history.items():
                    emoji = self.rt_viz.get_module_emoji(module_name)
                    duration = module_data.get('duration', 0.0)
                    duration_str = self.rt_viz.format_duration(duration)

                    # Check for token metrics
                    if self.show_tokens and 'token_metrics' in module_data:
                        metrics = module_data['token_metrics']
                        if metrics and metrics.get('cost', 0) > 0:
                            if metrics.get('total_tokens', 0) > 0:
                                token_str = f"[{metrics['prompt_tokens']}/{metrics['completion_tokens']} tokens, ${metrics['cost']:.6f}]"
                            else:
                                token_str = f"[${metrics['cost']:.6f}]"
                            token_colored = self.rt_viz.color(token_str, ColorCode.CYAN)
                            lines.append(f"{detail_prefix}{emoji} {module_name}: {duration_str} {token_colored}")
                        else:
                            lines.append(f"{detail_prefix}{emoji} {module_name}: {duration_str}")
                    else:
                        lines.append(f"{detail_prefix}{emoji} {module_name}: {duration_str}")

        # Process subgraph if exists
        subgraph_id = node_data.get('subgraph_id')
        if subgraph_id and subgraph_id in all_subgraphs:
            subgraph_data = all_subgraphs[subgraph_id]
            subgraph_tasks_data = subgraph_data.get('tasks', {})

            # Convert subgraph tasks to dict if it's a list (use 'task_id' as key)
            if isinstance(subgraph_tasks_data, list):
                subgraph_tasks = {task.get('task_id', task.get('id')): task for task in subgraph_tasks_data}
            else:
                subgraph_tasks = subgraph_tasks_data

            child_prefix = prefix + ("    " if is_last else "│   ")
            children = list(subgraph_tasks.values())

            for i, child_data in enumerate(children):
                is_child_last = (i == len(children) - 1)
                self._build_tree_from_snapshot(
                    child_data,
                    subgraph_tasks,  # Use subgraph's tasks for children
                    all_subgraphs,   # Pass all subgraphs for nested subgraphs
                    lines,
                    depth + 1,
                    visited,
                    is_child_last,
                    child_prefix
                )

    def _format_node_from_snapshot(self, node_data: Dict[str, Any], depth: int) -> str:
        """Format a single node from snapshot data for display."""
        from roma_dspy.types import TaskStatus, NodeType

        # Status emoji
        status_str = node_data.get('status', 'pending')
        try:
            status = TaskStatus(status_str)
        except ValueError:
            status = TaskStatus.PENDING
        status_emoji = self.rt_viz.get_status_emoji(status)

        # Depth indicator
        max_depth = node_data.get('max_depth', 3)
        depth_str = f"[D{depth}/{max_depth}]"
        depth_colored = self.rt_viz.color(depth_str, ColorCode.YELLOW)

        # Node type
        node_type_str = node_data.get('node_type')
        if node_type_str:
            try:
                node_type = NodeType(node_type_str)
                type_emoji = "📝" if node_type == NodeType.PLAN else "⚡"
                type_str = f"{type_emoji}{node_type.value}"
                type_colored = self.rt_viz.color(type_str, ColorCode.MAGENTA)
            except ValueError:
                type_colored = ""
        else:
            type_colored = ""

        # Goal (truncate if max_goal_length is set and > 0)
        goal = node_data.get('goal', '(no goal)')
        if self.max_goal_length > 0 and len(goal) > self.max_goal_length:
            goal = goal[:self.max_goal_length - 3] + "..."
        goal_colored = self.rt_viz.color(goal, ColorCode.BRIGHT_BLUE)

        # Status
        status_colored = self._color_by_status(status.value, status)

        return f"{status_emoji} {depth_colored} {goal_colored} {type_colored} [{status_colored}]"

    def _generate_statistics_from_snapshot(self, stats: Dict[str, Any]) -> List[str]:
        """Generate statistics summary from snapshot statistics dict."""
        from roma_dspy.types import TaskStatus

        lines = []
        lines.append(self.rt_viz.color("📈 EXECUTION STATISTICS", ColorCode.BOLD))
        lines.append("")

        # Task counts by status
        status_counts = stats.get('status_counts', {})
        if status_counts:
            lines.append("Task Status Distribution:")
            for status_str, count in status_counts.items():
                try:
                    status = TaskStatus(status_str)
                    emoji = self.rt_viz.get_status_emoji(status)
                except ValueError:
                    emoji = "❓"
                lines.append(f"  {emoji} {status_str}: {count}")

        # Depth distribution
        depth_dist = stats.get('depth_distribution', {})
        if depth_dist:
            lines.append("")
            lines.append("Depth Distribution:")
            # Convert keys to int for proper sorting
            depth_items = [(int(k) if isinstance(k, (int, str)) and str(k).isdigit() else k, v)
                          for k, v in depth_dist.items()]
            for depth, count in sorted(depth_items):
                bar = "█" * min(count, 20)
                lines.append(f"  Level {depth}: {bar} ({count} tasks)")

        # Summary
        lines.append("")
        lines.append(f"Total Tasks: {stats.get('total_tasks', 0)}")
        lines.append(f"Subgraphs Created: {stats.get('num_subgraphs', 0)}")
        is_complete = stats.get('is_complete', False)
        lines.append(f"Execution Complete: {'✅ Yes' if is_complete else '❌ No'}")

        return lines

    def _find_root_task(self, dag: TaskDAG) -> Optional[TaskNode]:
        """Find the root task in the DAG."""
        for node_id in dag.graph.nodes():
            task = dag.get_node(node_id)
            if task.is_root:
                return task
        return None

    def _build_tree(self, task: TaskNode, dag: Optional[TaskDAG], lines: List[str],
                    depth: int, visited: Set[str], is_last: bool, prefix: str):
        """Recursively build tree representation."""
        if task.task_id in visited:
            lines.append(f"{prefix}↺ (circular reference)")
            return

        visited.add(task.task_id)

        # Build current node line
        connector = "└── " if is_last else "├── "
        node_line = self._format_node(task, depth)

        lines.append(f"{prefix}{connector}{node_line}")

        # Add task details
        if self.show_ids:
            detail_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(f"{detail_prefix}ID: {self.rt_viz.color(task.task_id[:8], ColorCode.DIM)}")

        if self.show_timing and task.execution_history:
            detail_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(f"{detail_prefix}{self._format_timing(task)}")

        # Add execution history
        if task.execution_history:
            detail_prefix = prefix + ("    " if is_last else "│   ")
            has_token_data = False

            for module_name, result in task.execution_history.items():
                emoji = self.rt_viz.get_module_emoji(module_name)
                duration = self.rt_viz.format_duration(result.duration)

                # Add token metrics if available and enabled
                if self.show_tokens and result.token_metrics:
                    metrics = result.token_metrics
                    # Show cost even if token counts aren't available
                    if metrics.cost > 0:
                        has_token_data = True
                        if metrics.total_tokens > 0:
                            token_str = f"[{metrics.prompt_tokens}/{metrics.completion_tokens} tokens, ${metrics.cost:.6f}]"
                        else:
                            token_str = f"[${metrics.cost:.6f}]"
                        token_colored = self.rt_viz.color(token_str, ColorCode.CYAN)
                        lines.append(f"{detail_prefix}{emoji} {module_name}: {duration} {token_colored}")
                    else:
                        lines.append(f"{detail_prefix}{emoji} {module_name}: {duration}")
                else:
                    lines.append(f"{detail_prefix}{emoji} {module_name}: {duration}")

            # Add node totals if showing tokens and we have data
            if self.show_tokens and has_token_data:
                node_metrics = task.get_node_metrics()
                if node_metrics.total_tokens > 0:
                    total_str = f"💰 Node Total: {node_metrics.total_tokens} tokens, ${node_metrics.cost:.6f}"
                    total_colored = self.rt_viz.color(total_str, ColorCode.BRIGHT_YELLOW)
                    lines.append(f"{detail_prefix}{total_colored}")

        # Process subgraph if exists
        if task.subgraph_id and dag:
            subgraph = dag.get_subgraph(task.subgraph_id)
            if subgraph:
                child_prefix = prefix + ("    " if is_last else "│   ")
                children = list(subgraph.get_all_tasks(include_subgraphs=False))

                for i, child in enumerate(children):
                    is_child_last = (i == len(children) - 1)
                    self._build_tree(child, subgraph, lines, depth + 1,
                                   visited, is_child_last, child_prefix)

    def _format_node(self, task: TaskNode, depth: int) -> str:
        """Format a single node for display."""
        # Status emoji
        status_emoji = self.rt_viz.get_status_emoji(task.status)

        # Depth indicator
        depth_str = f"[D{depth}/{task.max_depth}]"
        depth_colored = self.rt_viz.color(depth_str, ColorCode.YELLOW)

        # Node type
        if task.node_type:
            type_emoji = "📝" if task.node_type == NodeType.PLAN else "⚡"
            type_str = f"{type_emoji}{task.node_type.value}"
            type_colored = self.rt_viz.color(type_str, ColorCode.MAGENTA)
        else:
            type_colored = ""

        # Goal (truncate if max_goal_length is set and > 0)
        goal = task.goal
        if self.max_goal_length > 0 and len(goal) > self.max_goal_length:
            goal = goal[:self.max_goal_length - 3] + "..."
        goal_colored = self.rt_viz.color(goal, ColorCode.BRIGHT_BLUE)

        # Status
        status_colored = self._color_by_status(task.status.value, task.status)

        return f"{status_emoji} {depth_colored} {goal_colored} {type_colored} [{status_colored}]"

    def _color_by_status(self, text: str, status: TaskStatus) -> str:
        """Color text based on task status."""
        color_map = {
            TaskStatus.COMPLETED: ColorCode.BRIGHT_GREEN,
            TaskStatus.FAILED: ColorCode.BRIGHT_RED,
            TaskStatus.EXECUTING: ColorCode.BRIGHT_YELLOW,
            TaskStatus.PLANNING: ColorCode.BRIGHT_CYAN,
            TaskStatus.PENDING: ColorCode.DIM,
        }
        color = color_map.get(status, ColorCode.WHITE)
        return self.rt_viz.color(text, color)

    def _format_timing(self, task: TaskNode) -> str:
        """Format timing information for a task."""
        if task.metrics and task.metrics.total_duration:
            duration = self.rt_viz.format_duration(task.metrics.total_duration)
            return f"⏱️  {self.rt_viz.color(duration, ColorCode.GREEN)}"
        return ""

    def _generate_statistics(self, dag: TaskDAG) -> List[str]:
        """Generate statistics summary."""
        stats = dag.get_statistics()
        lines = []

        lines.append(self.rt_viz.color("📈 EXECUTION STATISTICS", ColorCode.BOLD))
        lines.append("")

        # Task counts by status
        lines.append("Task Status Distribution:")
        for status, count in stats['status_counts'].items():
            emoji = self.rt_viz.get_status_emoji(TaskStatus(status))
            lines.append(f"  {emoji} {status}: {count}")

        # Depth distribution
        lines.append("")
        lines.append("Depth Distribution:")
        for depth, count in sorted(stats['depth_distribution'].items()):
            bar = "█" * min(count, 20)
            lines.append(f"  Level {depth}: {bar} ({count} tasks)")

        # Summary
        lines.append("")
        lines.append(f"Total Tasks: {stats['total_tasks']}")
        lines.append(f"Subgraphs Created: {stats['num_subgraphs']}")
        lines.append(f"Execution Complete: {'✅ Yes' if stats['is_complete'] else '❌ No'}")

        return lines


class TimelineVisualizer:
    """
    Timeline visualization showing execution flow over time.
    Creates a Gantt-chart style view of module executions.
    """

    def __init__(self, width: int = 80):
        """
        Initialize timeline visualizer.

        Args:
            width: Width of the timeline display
        """
        self.width = width
        self.events: List[Dict[str, Any]] = []

    def visualize(self, source: Optional[Any] = None, dag: Optional[TaskDAG] = None) -> str:
        """
        Generate timeline visualization from execution events.

        Args:
            source: RecursiveSolver, TaskDAG, or TaskNode instance
            dag: Optional DAG (defaults to solver.last_dag if available)

        Returns:
            String representation of the timeline
        """
        dag, _ = _resolve_visualization_inputs(source, dag)
        self.events = []
        if not dag:
            return "No execution data available."

        # Collect all execution events
        self._collect_events(dag)

        if not self.events:
            return "No execution events found."

        lines = []
        lines.append("=" * self.width)
        lines.append("📅 EXECUTION TIMELINE")
        lines.append("=" * self.width)
        lines.append("")

        # Sort events by timestamp
        self.events.sort(key=lambda e: e['timestamp'])

        # Calculate time range
        start_time = self.events[0]['timestamp']
        end_time = self.events[-1]['timestamp']
        total_duration = end_time - start_time if end_time > start_time else 1

        # Group events by task
        task_timelines = defaultdict(list)
        for event in self.events:
            task_timelines[event['task_id']].append(event)

        # Create timeline for each task
        for task_id, task_events in task_timelines.items():
            task = dag.get_node(task_id) if task_id in dag.graph.nodes else None
            if task:
                lines.append(self._format_task_timeline(task, task_events, start_time, total_duration))

        lines.append("")
        lines.append(f"Total Duration: {total_duration:.2f}s")
        lines.append("=" * self.width)

        return "\n".join(lines)

    def _collect_events(self, dag: TaskDAG):
        """Recursively collect execution events from DAG."""
        for node_id in dag.graph.nodes():
            task = dag.get_node(node_id)

            # Add events from execution history
            for module_name, result in task.execution_history.items():
                self.events.append({
                    'task_id': task.task_id,
                    'task_goal': task.goal[:30],
                    'module': module_name,
                    'timestamp': result.timestamp.timestamp() if hasattr(result.timestamp, 'timestamp') else time.time(),
                    'duration': result.duration,
                    'depth': task.depth
                })

        # Recurse into subgraphs
        for subgraph in dag.subgraphs.values():
            self._collect_events(subgraph)

    def _format_task_timeline(self, task: TaskNode, events: List[Dict], start_time: float, total_duration: float) -> str:
        """Format timeline for a single task."""
        # Task header
        header = f"[D{task.depth}] {task.goal[:30]}"

        # Create timeline bar
        timeline_width = self.width - 35
        timeline = [" "] * timeline_width

        for event in events:
            # Calculate position on timeline
            event_start = event['timestamp'] - start_time
            event_duration = event.get('duration', 0.1)

            start_pos = int((event_start / total_duration) * timeline_width)
            end_pos = min(int(((event_start + event_duration) / total_duration) * timeline_width), timeline_width - 1)

            # Get module character
            module_char = {
                'atomizer': 'A',
                'planner': 'P',
                'executor': 'E',
                'aggregator': 'G'
            }.get(event['module'], '?')

            # Fill timeline
            for i in range(max(0, start_pos), min(timeline_width, end_pos + 1)):
                timeline[i] = module_char

        return f"{header:30} |{''.join(timeline)}|"


class StatisticsVisualizer:
    """
    Comprehensive statistics and metrics visualization.
    """

    def __init__(self):
        """Initialize statistics visualizer."""
        self.stats = {}

    def visualize(self, source: Optional[Any] = None, dag: Optional[TaskDAG] = None) -> str:
        """
        Generate detailed statistics visualization.

        Args:
            source: RecursiveSolver, TaskDAG, or TaskNode instance
            dag: Optional DAG (defaults to solver.last_dag if available)

        Returns:
            Formatted statistics string
        """
        dag, _ = _resolve_visualization_inputs(source, dag)
        if not dag:
            return "No execution data available."

        # Gather statistics
        self._gather_statistics(dag)

        lines = []
        lines.append("=" * 80)
        lines.append("📊 DETAILED EXECUTION STATISTICS")
        lines.append("=" * 80)
        lines.append("")

        # Module execution stats
        lines.append("MODULE EXECUTION SUMMARY:")
        lines.append("-" * 40)
        for module, stats in self.stats.get('modules', {}).items():
            lines.append(f"  {module.upper()}:")
            lines.append(f"    Executions: {stats['count']}")
            lines.append(f"    Total Time: {stats['total_time']:.3f}s")
            lines.append(f"    Avg Time: {stats['avg_time']:.3f}s")
            lines.append(f"    Min Time: {stats['min_time']:.3f}s")
            lines.append(f"    Max Time: {stats['max_time']:.3f}s")

        # Task depth analysis
        lines.append("")
        lines.append("RECURSION DEPTH ANALYSIS:")
        lines.append("-" * 40)
        depth_stats = self.stats.get('depth_stats', {})
        for depth in sorted(depth_stats.keys()):
            stats = depth_stats[depth]
            lines.append(f"  Depth {depth}:")
            lines.append(f"    Tasks: {stats['count']}")
            lines.append(f"    Completed: {stats['completed']}")
            lines.append(f"    Failed: {stats['failed']}")
            lines.append(f"    Avg Duration: {stats['avg_duration']:.3f}s")

        # Performance metrics
        lines.append("")
        lines.append("PERFORMANCE METRICS:")
        lines.append("-" * 40)
        perf = self.stats.get('performance', {})
        lines.append(f"  Total Tasks: {perf.get('total_tasks', 0)}")
        lines.append(f"  Completed Tasks: {perf.get('completed_tasks', 0)}")
        lines.append(f"  Failed Tasks: {perf.get('failed_tasks', 0)}")
        lines.append(f"  Total Execution Time: {perf.get('total_time', 0):.3f}s")
        lines.append(f"  Parallelization Potential: {perf.get('parallelization', 0):.1%}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def _gather_statistics(self, dag: TaskDAG):
        """Gather comprehensive statistics from DAG."""
        self.stats = {
            'modules': defaultdict(lambda: {'count': 0, 'total_time': 0, 'times': []}),
            'depth_stats': defaultdict(lambda: {
                'count': 0, 'completed': 0, 'failed': 0, 'durations': []
            }),
            'performance': {}
        }

        all_tasks = dag.get_all_tasks(include_subgraphs=True)

        for task in all_tasks:
            # Module statistics
            for module_name, result in task.execution_history.items():
                module_stats = self.stats['modules'][module_name]
                module_stats['count'] += 1
                module_stats['total_time'] += result.duration
                module_stats['times'].append(result.duration)

            # Depth statistics
            depth_stats = self.stats['depth_stats'][task.depth]
            depth_stats['count'] += 1
            if task.status == TaskStatus.COMPLETED:
                depth_stats['completed'] += 1
            elif task.status == TaskStatus.FAILED:
                depth_stats['failed'] += 1

            if task.metrics and task.metrics.total_duration:
                depth_stats['durations'].append(task.metrics.total_duration)

        # Calculate aggregated metrics
        for module, stats in self.stats['modules'].items():
            if stats['times']:
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['min_time'] = min(stats['times'])
                stats['max_time'] = max(stats['times'])

        for depth, stats in self.stats['depth_stats'].items():
            if stats['durations']:
                stats['avg_duration'] = sum(stats['durations']) / len(stats['durations'])
            else:
                stats['avg_duration'] = 0

        # Overall performance metrics
        self.stats['performance']['total_tasks'] = len(all_tasks)
        self.stats['performance']['completed_tasks'] = sum(
            1 for t in all_tasks if t.status == TaskStatus.COMPLETED
        )
        self.stats['performance']['failed_tasks'] = sum(
            1 for t in all_tasks if t.status == TaskStatus.FAILED
        )

        # Calculate total time
        total_time = 0
        for module_stats in self.stats['modules'].values():
            total_time += module_stats['total_time']
        self.stats['performance']['total_time'] = total_time


class ContextFlowVisualizer:
    """
    Visualizer for context flow between subtasks showing index-based dependencies.
    Shows the actual XML-formatted context passed to LLMs.
    """

    def __init__(self, use_colors: bool = True):
        """
        Initialize context flow visualizer.

        Args:
            use_colors: Whether to use ANSI colors
        """
        self.use_colors = use_colors
        self.rt_viz = RealTimeVisualizer(use_colors=use_colors)

    def visualize(self, source: Optional[Any] = None, dag: Optional[TaskDAG] = None,
                  show_full_context: bool = False) -> str:
        """
        Visualize context flow with index-based dependencies.

        Args:
            source: RecursiveSolver, TaskDAG, or TaskNode to visualize
            dag: Optional DAG to visualize
            show_full_context: If True, shows full context; if False, shows preview

        Returns:
            Formatted string showing context flow
        """
        # Access runtime and dag from source
        runtime = None
        resolved_dag = dag

        # Try to get runtime and dag from various source types
        if hasattr(source, '_solver'):
            # DSPy module wrapper
            runtime = source._solver.runtime
            if resolved_dag is None:
                resolved_dag = source._solver.last_dag
        elif hasattr(source, 'runtime'):
            # Direct runtime access
            runtime = source.runtime
            if resolved_dag is None and hasattr(source, 'last_dag'):
                resolved_dag = source.last_dag

        # Fallback to standard resolution if needed
        if resolved_dag is None:
            resolved_dag, _ = _resolve_visualization_inputs(source, dag)

        if not resolved_dag:
            return "No execution data available. Make sure you've run the task first."

        if not runtime:
            return "No runtime context store available."

        lines = []
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
        lines.append(self.rt_viz.color("🔗 CONTEXT FLOW VISUALIZATION (Index-Based)", ColorCode.BOLD))
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
        lines.append("")

        all_tasks = resolved_dag.get_all_tasks(include_subgraphs=True)

        # Group tasks by their parent (subgraph)
        subgraph_tasks = {}
        for task in all_tasks:
            if task.parent_id and task.node_type and task.node_type.value == "EXECUTE":
                if task.parent_id not in subgraph_tasks:
                    subgraph_tasks[task.parent_id] = []
                subgraph_tasks[task.parent_id].append(task)

        # Process each subgraph
        for parent_id, tasks in subgraph_tasks.items():
            parent_task = resolved_dag.get_node(parent_id) if parent_id in resolved_dag.graph else None
            if parent_task and parent_task.subgraph_id:
                lines.append(self.rt_viz.color("─" * 80, ColorCode.CYAN))
                lines.append(f"📦 Subgraph: {self.rt_viz.color(parent_task.goal[:60] + '...', ColorCode.BRIGHT_BLUE)}")
                lines.append(self.rt_viz.color("─" * 80, ColorCode.CYAN))

                # Get tasks with their indices
                indexed_tasks = []
                for task in tasks:
                    idx = runtime.context_store.get_task_index(parent_task.subgraph_id, task.task_id)
                    if idx is not None:
                        indexed_tasks.append((idx, task))

                # Sort by index
                indexed_tasks.sort(key=lambda x: x[0])

                # Display each task
                for idx, task in indexed_tasks:
                    lines.append("")
                    lines.append(self.rt_viz.color(f"[Subtask {idx}] {task.goal[:70]}", ColorCode.BRIGHT_GREEN))

                    if task.dependencies:
                        # Get dependency indices
                        dep_indices = []
                        for dep_id in task.dependencies:
                            dep_idx = runtime.context_store.get_task_index(parent_task.subgraph_id, dep_id)
                            if dep_idx is not None:
                                dep_indices.append(dep_idx)

                        if dep_indices:
                            lines.append(f"  ⬅️  Dependencies: {self.rt_viz.color(str(sorted(dep_indices)), ColorCode.YELLOW)}")

                            # Reconstruct the actual XML context
                            context_parts = []
                            for dep_id in task.dependencies:
                                result_str = runtime.context_store.get_result(dep_id)
                                if result_str:
                                    dep_task = None
                                    try:
                                        dep_task, _ = resolved_dag.find_node(dep_id)
                                    except ValueError:
                                        pass

                                    dep_idx = runtime.context_store.get_task_index(parent_task.subgraph_id, dep_id)
                                    if dep_idx is not None:
                                        context_entry = f'<subtask id="{dep_idx}">'
                                        if dep_task:
                                            context_entry += f"\n    <goal>{dep_task.goal}</goal>"
                                        # Truncate output for display
                                        if show_full_context:
                                            context_entry += f"\n    <output>{result_str}</output>\n  </subtask>"
                                        else:
                                            output_preview = result_str[:150] + "..." if len(result_str) > 150 else result_str
                                            context_entry += f"\n    <output>{output_preview}</output>\n  </subtask>"
                                        context_parts.append(context_entry)

                            if context_parts:
                                full_context = "  <context>\n  " + "\n\n  ".join(context_parts) + "\n  </context>"
                                lines.append(self.rt_viz.color("  📥 Context passed to LLM (XML format):", ColorCode.MAGENTA))
                                for line in full_context.split('\n'):
                                    lines.append(self.rt_viz.color(f"    {line}", ColorCode.DIM))
                    else:
                        lines.append(self.rt_viz.color("  ℹ️  No dependencies (independent task)", ColorCode.DIM))

                    result_str = str(task.result) if task.result else "(no result)"
                    if len(result_str) > 150:
                        result_str = result_str[:150] + "..."
                    lines.append(f"  ✅ Result: {self.rt_viz.color(result_str, ColorCode.GREEN)}")
                    lines.append(self.rt_viz.color("  " + "-" * 76, ColorCode.DIM))

        lines.append("")
        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
        return "\n".join(lines)

    def get_task_context_details(self, source: Any, subtask_index: int) -> str:
        """
        Get detailed context information for a specific task by index.

        Args:
            source: RecursiveSolver or module with runtime
            subtask_index: The index of the task in its subgraph

        Returns:
            Formatted string with context details
        """
        # Access runtime and dag
        runtime = None
        dag = None

        if hasattr(source, '_solver'):
            runtime = source._solver.runtime
            dag = source._solver.last_dag
        elif hasattr(source, 'runtime'):
            runtime = source.runtime
            dag = getattr(source, 'last_dag', None)

        if not runtime or not dag:
            return "No runtime or DAG available."

        # Find the task
        for subgraph_id, index_map in runtime.context_store._index_maps.items():
            if subtask_index in index_map:
                task_id = index_map[subtask_index]
                subgraph = dag.get_subgraph(subgraph_id)
                if subgraph:
                    try:
                        task = subgraph.get_node(task_id)

                        lines = []
                        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
                        lines.append(self.rt_viz.color(f"📋 CONTEXT DETAILS FOR SUBTASK {subtask_index}", ColorCode.BOLD))
                        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))
                        lines.append(f"Goal: {self.rt_viz.color(task.goal, ColorCode.BRIGHT_BLUE)}")
                        lines.append(self.rt_viz.color("-" * 80, ColorCode.DIM))

                        # Get the context from execution history metadata
                        executor_result = task.execution_history.get("executor")
                        if executor_result and executor_result.metadata:
                            context_received = executor_result.metadata.get("context_received")
                            if context_received:
                                lines.append(self.rt_viz.color("Context (from execution metadata):", ColorCode.MAGENTA))
                                lines.append(self.rt_viz.color(context_received, ColorCode.DIM))
                            else:
                                lines.append(self.rt_viz.color("No context was passed to this task.", ColorCode.YELLOW))
                        else:
                            lines.append(self.rt_viz.color("No execution history available.", ColorCode.YELLOW))

                        lines.append(self.rt_viz.color("-" * 80, ColorCode.DIM))
                        result_preview = str(task.result)[:200] + "..." if len(str(task.result)) > 200 else str(task.result)
                        lines.append(f"Result: {self.rt_viz.color(result_preview, ColorCode.GREEN)}")
                        lines.append(self.rt_viz.color("=" * 80, ColorCode.CYAN))

                        return "\n".join(lines)

                    except ValueError:
                        pass

        return f"Subtask {subtask_index} not found."


class LLMTraceVisualizer:
    """
    LLM-friendly execution trace visualizer for prompt optimization.

    Generates a chronological, detailed trace of the entire execution flow
    that is optimized for LLM consumption and analysis. Useful for:
    - Prompt optimization and debugging
    - Understanding module behavior
    - Cost and performance analysis
    - Reproducing execution flows
    """

    def __init__(self, show_metrics: bool = True, show_summary: bool = True, verbose: bool = True):
        """
        Initialize LLM trace visualizer.

        Args:
            show_metrics: Whether to display duration/tokens/cost for each module
            show_summary: Whether to display execution summary at the end
            verbose: Whether to show full inputs/outputs (True) or compact versions (False)
        """
        self.show_metrics = show_metrics
        self.show_summary = show_summary
        self.verbose = verbose

    def visualize(self, source: Optional[Any] = None, dag: Optional[TaskDAG] = None) -> str:
        """
        Generate LLM-friendly execution trace.

        Args:
            source: RecursiveSolver, TaskDAG, or TaskNode to visualize
            dag: Optional DAG to visualize (defaults to solver.last_dag if available)

        Returns:
            String representation of the execution trace
        """
        dag, root_task = _resolve_visualization_inputs(source, dag)

        if not dag and root_task is None:
            return "No execution data available. Run solve() first."

        # Find root task if not provided
        if root_task is None and dag:
            for node_id in dag.graph.nodes():
                task = dag.get_node(node_id)
                if task.is_root:
                    root_task = task
                    break

        if not root_task:
            return "No root task found in DAG."

        lines = []
        lines.append("=== EXECUTION TRACE ===")
        lines.append(f"Root Goal: {root_task.goal}")
        lines.append(f"Max Depth: {root_task.max_depth}")

        # Format timestamp properly
        start_time = self._normalize_timestamp(root_task.created_at)
        lines.append(f"Start Time: {start_time.isoformat()}")
        lines.append("")

        # Collect all execution events chronologically
        events = self._collect_execution_events(dag, root_task)

        # Format trace
        self._format_trace(lines, events, dag)

        # Add summary if enabled
        if self.show_summary:
            lines.append("")
            lines.append("=== EXECUTION SUMMARY ===")
            self._add_summary(lines, events, root_task)

        return "\n".join(lines)

    def _collect_execution_events(self, dag: TaskDAG, root_task: TaskNode) -> List[Dict[str, Any]]:
        """Collect all execution events in chronological order."""
        events = []
        all_tasks = dag.get_all_tasks(include_subgraphs=True)

        for task in all_tasks:
            # Create task entry event
            task_event = {
                'type': 'task_start',
                'timestamp': self._normalize_timestamp(task.created_at),
                'task': task,
                'depth': task.depth
            }
            events.append(task_event)

            # Add module execution events
            for module_name, module_result in task.execution_history.items():
                module_event = {
                    'type': 'module_execution',
                    'timestamp': self._normalize_timestamp(module_result.timestamp),
                    'task': task,
                    'module_name': module_name,
                    'module_result': module_result,
                    'depth': task.depth
                }
                events.append(module_event)

            # Add task completion event
            if task.completed_at:
                completion_event = {
                    'type': 'task_complete',
                    'timestamp': self._normalize_timestamp(task.completed_at),
                    'task': task,
                    'depth': task.depth
                }
                events.append(completion_event)

        # Sort by timestamp
        events.sort(key=lambda e: e['timestamp'])
        return events

    def _normalize_timestamp(self, ts: datetime) -> datetime:
        """Normalize timestamp to timezone-naive for consistent comparison."""
        if ts.tzinfo is not None:
            # Convert to UTC and remove timezone info
            return ts.replace(tzinfo=None)
        return ts

    def _format_trace(self, lines: List[str], events: List[Dict[str, Any]], dag: TaskDAG):
        """Format execution trace from events."""
        # Group events by task to ensure proper ordering
        task_events = {}
        for event in events:
            task_id = event['task'].task_id
            if task_id not in task_events:
                task_events[task_id] = []
            task_events[task_id].append(event)

        # Process tasks in chronological order by their first event
        task_order = []
        for task_id, tevents in task_events.items():
            first_timestamp = min(e['timestamp'] for e in tevents)
            task_order.append((first_timestamp, task_id, tevents))

        task_order.sort(key=lambda x: x[0])

        # Now format each task's execution in order
        for _, task_id, tevents in task_order:
            # Sort this task's events
            tevents.sort(key=lambda e: e['timestamp'])

            task = tevents[0]['task']
            depth = tevents[0]['depth']
            indent = "  " * depth

            # Print task header
            lines.append(f"{indent}[DEPTH {depth}] Task: {task.goal}")
            lines.append(f"{indent}  ID: {task.task_id[:8]}...")
            if task.parent_id:
                lines.append(f"{indent}  Parent: {task.parent_id[:8]}...")
            if task.dependencies:
                dep_list = [dep_id[:8] + "..." for dep_id in task.dependencies]
                lines.append(f"{indent}  Dependencies: {dep_list}")

            # Print module executions for this task
            for event in tevents:
                if event['type'] == 'module_execution':
                    module_name = event['module_name']
                    module_result = event['module_result']

                    lines.append("")
                    lines.append(f"{indent}  MODULE: {module_name.capitalize()}")

                    # Format inputs
                    lines.append(f"{indent}    Input:")
                    self._format_io(lines, module_result.input, indent + "      ")

                    # Format outputs
                    lines.append(f"{indent}    Output:")
                    self._format_io(lines, module_result.output, indent + "      ")

                    # Add metrics if enabled
                    if self.show_metrics:
                        metrics_parts = [f"Duration: {module_result.duration:.2f}s"]
                        if module_result.token_metrics:
                            tm = module_result.token_metrics
                            if tm.total_tokens > 0:
                                metrics_parts.append(f"Tokens: {tm.prompt_tokens}/{tm.completion_tokens}")
                            if tm.cost > 0:
                                metrics_parts.append(f"Cost: ${tm.cost:.6f}")
                        lines.append(f"{indent}    {' | '.join(metrics_parts)}")

            # Print completion status
            lines.append("")
            lines.append(f"{indent}  Status: -> {task.status.value}")
            if task.result and self.verbose:
                result_str = str(task.result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                lines.append(f"{indent}  Result: {result_str}")

            lines.append("")

    def _format_io(self, lines: List[str], data: Any, indent: str):
        """Format input/output data."""
        if data is None:
            lines.append(f"{indent}(none)")
            return

        # Handle dspy.Prediction objects
        if hasattr(data, '__dict__') and hasattr(data, '_store'):
            # DSPy prediction object
            for key, value in data.__dict__.items():
                if key.startswith('_'):
                    continue
                formatted_value = self._format_value(value)
                lines.append(f"{indent}{key}: {formatted_value}")
        elif isinstance(data, dict):
            for key, value in data.items():
                formatted_value = self._format_value(value)
                lines.append(f"{indent}{key}: {formatted_value}")
        elif isinstance(data, str):
            if self.verbose or len(data) <= 200:
                lines.append(f"{indent}{data}")
            else:
                lines.append(f"{indent}{data[:200]}...")
        else:
            formatted_value = self._format_value(data)
            lines.append(f"{indent}{formatted_value}")

    def _format_value(self, value: Any) -> str:
        """Format a single value for display."""
        if value is None:
            return "null"
        elif isinstance(value, str):
            if self.verbose or len(value) <= 100:
                return f'"{value}"'
            else:
                return f'"{value[:100]}..."'
        elif isinstance(value, (list, tuple)):
            if not value:
                return "[]"
            elif len(value) <= 3 or self.verbose:
                formatted_items = [self._format_value(item) for item in value]
                return f"[{', '.join(formatted_items)}]"
            else:
                formatted_items = [self._format_value(item) for item in value[:3]]
                return f"[{', '.join(formatted_items)}, ... ({len(value)} total)]"
        elif isinstance(value, dict):
            if not value:
                return "{}"
            items = list(value.items())
            if len(items) <= 3 or self.verbose:
                formatted = ", ".join(f"{k}: {self._format_value(v)}" for k, v in items)
                return f"{{{formatted}}}"
            else:
                formatted = ", ".join(f"{k}: {self._format_value(v)}" for k, v in items[:3])
                return f"{{{formatted}, ... ({len(items)} total)}}"
        elif hasattr(value, '__dict__'):
            # Pydantic model or object
            if hasattr(value, 'model_dump'):
                return self._format_value(value.model_dump())
            elif hasattr(value, 'dict'):
                return self._format_value(value.dict())
            else:
                return str(value)
        else:
            return str(value)

    def _add_summary(self, lines: List[str], events: List[Dict[str, Any]], root_task: TaskNode):
        """Add execution summary."""
        # Calculate metrics
        total_duration = 0.0
        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        task_count = 0
        completed_count = 0
        failed_count = 0
        max_depth = 0

        seen_tasks = set()

        for event in events:
            task = event['task']

            # Track unique tasks
            if task.task_id not in seen_tasks:
                seen_tasks.add(task.task_id)
                task_count += 1
                if task.status == TaskStatus.COMPLETED:
                    completed_count += 1
                elif task.status == TaskStatus.FAILED:
                    failed_count += 1
                max_depth = max(max_depth, task.depth)

            # Aggregate module metrics
            if event['type'] == 'module_execution':
                module_result = event['module_result']
                total_duration += module_result.duration

                if module_result.token_metrics:
                    tm = module_result.token_metrics
                    total_prompt_tokens += tm.prompt_tokens
                    total_completion_tokens += tm.completion_tokens
                    total_tokens += tm.total_tokens
                    total_cost += tm.cost

        # Format summary
        if root_task.completed_at and root_task.created_at:
            wall_time = (root_task.completed_at - root_task.created_at).total_seconds()
            lines.append(f"Wall Time: {wall_time:.2f}s")

        lines.append(f"Total Module Time: {total_duration:.2f}s")

        if self.show_metrics and total_tokens > 0:
            lines.append(f"Total Tokens: {total_prompt_tokens}/{total_completion_tokens} ({total_tokens} total)")

        if self.show_metrics and total_cost > 0:
            lines.append(f"Total Cost: ${total_cost:.6f}")

        lines.append(f"Tasks: {task_count} total ({completed_count} completed, {failed_count} failed)")
        lines.append(f"Max Depth Reached: {max_depth}")

    def visualize_from_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """
        Generate LLM trace visualization from dict snapshot (from PostgreSQL storage).

        Note: This generates a simplified trace from snapshot data. For full execution
        history with module inputs/outputs, use visualize() with a live DAG object.

        Args:
            snapshot: Dict snapshot with keys: 'tasks', 'subgraphs', 'statistics', 'dag_id'

        Returns:
            String representation of the execution trace
        """
        # Validate snapshot structure
        if not isinstance(snapshot, dict):
            return "Invalid snapshot format. Expected dict."

        if 'tasks' not in snapshot:
            return "Invalid snapshot format. Expected dict with 'tasks' key."

        tasks_data = snapshot['tasks']
        statistics = snapshot.get('statistics', {})

        if not tasks_data:
            return "No tasks found in snapshot."

        # Find root node
        root_node_data = None
        for node_data in tasks_data.values():
            if node_data.get('is_root', False):
                root_node_data = node_data
                break
            if node_data.get('depth', -1) == 0 and root_node_data is None:
                root_node_data = node_data

        if not root_node_data:
            return "No root task found in snapshot."

        lines = []
        lines.append("=== EXECUTION TRACE (FROM STORAGE) ===")
        lines.append(f"Root Goal: {root_node_data.get('goal', '(no goal)')}")
        lines.append(f"Max Depth: {root_node_data.get('max_depth', 'unknown')}")
        lines.append("")
        lines.append("NOTE: This is a simplified trace from snapshot data.")
        lines.append("Full module input/output details are only available with live DAG visualization.")
        lines.append("")

        # Collect tasks sorted by creation time or task_id
        task_list = []
        for task_id, task_data in tasks_data.items():
            task_list.append((task_data.get('depth', 0), task_id, task_data))

        # Sort by depth and task_id
        task_list.sort(key=lambda x: (x[0], x[1]))

        # Format tasks
        for depth, task_id, task_data in task_list:
            indent = "  " * depth
            lines.append(f"{indent}[DEPTH {depth}] Task: {task_data.get('goal', '(no goal)')}")
            lines.append(f"{indent}  ID: {task_id[:8]}...")
            lines.append(f"{indent}  Status: {task_data.get('status', 'unknown')}")

            # Show execution history if available
            execution_history = task_data.get('execution_history', {})
            if execution_history:
                lines.append(f"{indent}  Modules executed:")
                if isinstance(execution_history, list):
                    for module_name in execution_history:
                        lines.append(f"{indent}    - {module_name}")
                elif isinstance(execution_history, dict):
                    for module_name, module_data in execution_history.items():
                        duration = module_data.get('duration', 0.0) if isinstance(module_data, dict) else 0.0
                        if duration > 0:
                            lines.append(f"{indent}    - {module_name}: {duration:.2f}s")
                        else:
                            lines.append(f"{indent}    - {module_name}")

            # Show result if available
            result = task_data.get('result')
            if result and self.verbose:
                result_str = str(result)
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                lines.append(f"{indent}  Result: {result_str}")

            lines.append("")

        # Add summary if enabled and statistics available
        if self.show_summary and statistics:
            lines.append("=== EXECUTION SUMMARY ===")
            lines.append(f"Total Tasks: {statistics.get('total_tasks', 0)}")

            status_counts = statistics.get('status_counts', {})
            if status_counts:
                for status, count in status_counts.items():
                    lines.append(f"  {status}: {count}")

            lines.append(f"Subgraphs Created: {statistics.get('num_subgraphs', 0)}")
            lines.append(f"Execution Complete: {'Yes' if statistics.get('is_complete', False) else 'No'}")

        return "\n".join(lines)

    def export_trace_json(self, source: Optional[Any] = None, dag: Optional[TaskDAG] = None) -> Dict[str, Any]:
        """
        Export execution trace as JSON for programmatic analysis.

        Args:
            source: RecursiveSolver, TaskDAG, or TaskNode to export
            dag: Optional DAG to export

        Returns:
            Dictionary with complete execution trace data
        """
        dag, root_task = _resolve_visualization_inputs(source, dag)

        if not dag or not root_task:
            return {"error": "No execution data available"}

        events = self._collect_execution_events(dag, root_task)

        # Convert events to JSON-serializable format
        trace_data = {
            "root_goal": root_task.goal,
            "max_depth": root_task.max_depth,
            "start_time": self._normalize_timestamp(root_task.created_at).isoformat(),
            "events": []
        }

        for event in events:
            event_data = {
                "type": event['type'],
                "timestamp": self._normalize_timestamp(event['timestamp']).isoformat(),
                "depth": event['depth'],
                "task_id": event['task'].task_id,
                "task_goal": event['task'].goal,
            }

            if event['type'] == 'module_execution':
                module_result = event['module_result']
                event_data['module_name'] = event['module_name']
                event_data['duration'] = module_result.duration

                # Serialize input/output
                event_data['input'] = self._serialize_for_json(module_result.input)
                event_data['output'] = self._serialize_for_json(module_result.output)

                if module_result.token_metrics:
                    tm = module_result.token_metrics
                    event_data['token_metrics'] = {
                        'prompt_tokens': tm.prompt_tokens,
                        'completion_tokens': tm.completion_tokens,
                        'total_tokens': tm.total_tokens,
                        'cost': tm.cost,
                        'model': tm.model
                    }

            elif event['type'] == 'task_complete':
                event_data['status'] = event['task'].status.value
                event_data['result'] = self._serialize_for_json(event['task'].result)

            trace_data['events'].append(event_data)

        return trace_data

    def _serialize_for_json(self, obj: Any) -> Any:
        """Serialize object for JSON export."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif hasattr(obj, 'model_dump'):
            return self._serialize_for_json(obj.model_dump())
        elif hasattr(obj, 'dict'):
            return self._serialize_for_json(obj.dict())
        elif hasattr(obj, '__dict__'):
            return {k: self._serialize_for_json(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
        else:
            return str(obj)


class HierarchicalVisualizer:
    """
    Main visualizer class that combines all visualization modes.
    Can be integrated with RecursiveSolver for real-time visualization.
    """

    def __init__(self, mode: str = "all", use_colors: bool = True, verbose: bool = True):
        """
        Initialize hierarchical visualizer.

        Args:
            mode: Visualization mode ("realtime", "tree", "timeline", "stats", "context", "llm_trace", "all")
            use_colors: Whether to use colored output
            verbose: Whether to show detailed information
        """
        self.mode = mode
        self.realtime = RealTimeVisualizer(use_colors=use_colors, verbose=verbose)
        self.tree = TreeVisualizer(use_colors=use_colors)
        self.timeline = TimelineVisualizer()
        self.stats = StatisticsVisualizer()
        self.context = ContextFlowVisualizer(use_colors=use_colors)
        self.llm_trace = LLMTraceVisualizer(show_metrics=True, show_summary=True, verbose=verbose)

    def visualize_execution(self, solver) -> str:
        """
        Generate complete visualization of solver execution.

        Args:
            solver: RecursiveSolver instance after execution

        Returns:
            Complete visualization string
        """
        output = []

        if self.mode in ["tree", "all"]:
            output.append(self.tree.visualize(solver))
            output.append("")

        if self.mode in ["timeline", "all"]:
            output.append(self.timeline.visualize(solver))
            output.append("")

        if self.mode in ["stats", "all"]:
            output.append(self.stats.visualize(solver))
            output.append("")

        if self.mode in ["llm_trace"]:
            output.append(self.llm_trace.visualize(solver))
            output.append("")

        return "\n".join(output)

    def export_to_html(self, solver, filename: str):
        """
        Export visualization to interactive HTML file.

        Args:
            solver: RecursiveSolver instance
            filename: Output HTML filename
        """
        dag = solver.last_dag
        if not dag:
            print("No execution data available.")
            return

        html_content = self._generate_html(dag)

        with open(filename, 'w') as f:
            f.write(html_content)

        print(f"Visualization exported to {filename}")

    def export_to_json(self, solver, filename: str):
        """
        Export execution data to JSON for further analysis.

        Args:
            solver: RecursiveSolver instance
            filename: Output JSON filename
        """
        dag = solver.last_dag
        if not dag:
            print("No execution data available.")
            return

        data = dag.export_to_dict()

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Execution data exported to {filename}")

    def _generate_html(self, dag: TaskDAG) -> str:
        """Generate interactive HTML visualization."""
        # This is a simplified HTML template
        # In production, you'd want to use a proper templating engine
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ROMA-DSPy Execution Visualization</title>
            <style>
                body { font-family: monospace; background: #1e1e1e; color: #d4d4d4; }
                .tree { margin: 20px; }
                .node { margin: 5px 0; padding: 5px; border-left: 3px solid #569cd6; }
                .depth-0 { margin-left: 0px; }
                .depth-1 { margin-left: 20px; }
                .depth-2 { margin-left: 40px; }
                .depth-3 { margin-left: 60px; }
                .status-completed { background: #1e3a1e; border-color: #4ec9b0; }
                .status-failed { background: #3a1e1e; border-color: #f48771; }
                .status-executing { background: #3a3a1e; border-color: #dcdcaa; }
                .module { display: inline-block; margin: 2px; padding: 2px 6px; background: #264f78; border-radius: 3px; }
                .stats { margin: 20px; padding: 15px; background: #2d2d30; border-radius: 5px; }
                h1, h2 { color: #569cd6; }
                .emoji { font-size: 1.2em; }
            </style>
        </head>
        <body>
            <h1>🚀 ROMA-DSPy Hierarchical Task Decomposition</h1>
            <div class="tree">
                <!-- Tree visualization would be generated here -->
                <pre>{tree}</pre>
            </div>
            <div class="stats">
                <h2>📊 Execution Statistics</h2>
                <pre>{stats}</pre>
            </div>
        </body>
        </html>
        """.format(
            tree=self.tree.visualize(None, dag),
            stats=json.dumps(dag.get_statistics(), indent=2)
        )

        return html
