"""
Execution visualizer for ROMA-DSPy hierarchical task decomposition.

Provides multiple visualization modes for understanding the solver's execution:
- Real-time execution tracking with colored output
- Hierarchical tree visualization
- Timeline view of module executions
- Statistics and metrics dashboard
"""

import json
from datetime import datetime, timezone
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

    def visualize_from_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """
        Generate timeline visualization from dict snapshot (limited functionality).

        Args:
            snapshot: Dict snapshot with keys: 'tasks', 'subgraphs', 'statistics', 'dag_id'

        Returns:
            Message indicating timeline is not available from snapshots
        """
        return (
            "Timeline visualization is not available from storage snapshots.\n"
            "Timeline requires live execution events that are not preserved in snapshots.\n"
            "Use the tree, statistics, or llm_trace visualizers instead."
        )


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

    def visualize_from_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """
        Generate statistics visualization from dict snapshot.

        Args:
            snapshot: Dict snapshot with keys: 'tasks', 'subgraphs', 'statistics', 'dag_id'

        Returns:
            Formatted statistics string
        """
        # Validate snapshot structure
        if not isinstance(snapshot, dict):
            return "Invalid snapshot format. Expected dict."

        statistics = snapshot.get('statistics', {})

        if not statistics:
            return "No statistics found in snapshot."

        lines = []
        lines.append("=" * 80)
        lines.append("📊 EXECUTION STATISTICS (FROM STORAGE)")
        lines.append("=" * 80)
        lines.append("")

        # Task counts by status
        status_counts = statistics.get('status_counts', {})
        if status_counts:
            lines.append("Task Status Distribution:")
            for status_str, count in status_counts.items():
                lines.append(f"  {status_str}: {count}")
            lines.append("")

        # Depth distribution
        depth_dist = statistics.get('depth_distribution', {})
        if depth_dist:
            lines.append("Depth Distribution:")
            # Convert keys to int for proper sorting
            depth_items = [(int(k) if isinstance(k, (int, str)) and str(k).isdigit() else k, v)
                          for k, v in depth_dist.items()]
            for depth, count in sorted(depth_items):
                bar = "█" * min(count, 20)
                lines.append(f"  Level {depth}: {bar} ({count} tasks)")
            lines.append("")

        # Summary
        lines.append("Summary:")
        lines.append(f"  Total Tasks: {statistics.get('total_tasks', 0)}")
        lines.append(f"  Subgraphs Created: {statistics.get('num_subgraphs', 0)}")
        is_complete = statistics.get('is_complete', False)
        lines.append(f"  Execution Complete: {'✅ Yes' if is_complete else '❌ No'}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("Note: Detailed module execution statistics are only available from live DAG visualization.")

        return "\n".join(lines)


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

    def visualize_from_snapshot(self, snapshot: Dict[str, Any]) -> str:
        """
        Generate context flow visualization from dict snapshot (limited functionality).

        Args:
            snapshot: Dict snapshot with keys: 'tasks', 'subgraphs', 'statistics', 'dag_id'

        Returns:
            Message indicating context flow is not available from snapshots
        """
        return (
            "Context flow visualization is not available from storage snapshots.\n"
            "Context flow requires the runtime context store to show dependencies and XML-formatted context.\n"
            "This information is not preserved in database snapshots.\n"
            "Use the tree, statistics, or llm_trace visualizers instead."
        )


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

    def __init__(
        self,
        show_metrics: bool = True,
        show_summary: bool = True,
        verbose: bool = True,
        fancy: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        show_io: bool = False,
        console_width: Optional[int] = None,
    ):
        """
        Initialize LLM trace visualizer.

        Args:
            show_metrics: Whether to display duration/tokens/cost for each module
            show_summary: Whether to display execution summary at the end
            verbose: Whether to show full inputs/outputs (True) or compact versions (False)
            fancy: Whether to use Rich library for beautiful CLI visualization (True) or plain text (False)
            mlflow_tracking_uri: MLflow tracking URI (default: http://localhost:5000)
        """
        self.show_metrics = show_metrics
        self.show_summary = show_summary
        self.verbose = verbose
        self.fancy = fancy
        self.mlflow_tracking_uri = mlflow_tracking_uri or "http://localhost:5000"
        self._mlflow_client = None
        self.mlflow_experiment_name = mlflow_experiment_name
        self.show_io = show_io
        self.console_width = console_width

    def _make_console(self):
        """Create a Rich Console, honoring explicit width if provided.

        force_terminal=True ensures width is respected even when not attached
        to a TTY (e.g., docker exec without -t).
        """
        from rich.console import Console
        if self.console_width:
            return Console(width=self.console_width, force_terminal=True)
        return Console()

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

        # Collect all execution events chronologically
        events = self._collect_execution_events(dag, root_task)

        # Route to appropriate formatter
        if self.fancy:
            return self._format_trace_rich(events, dag, root_task)
        else:
            # Plain text format (existing implementation)
            lines = []
            lines.append("=== EXECUTION TRACE ===")
            lines.append(f"Root Goal: {root_task.goal}")
            lines.append(f"Max Depth: {root_task.max_depth}")

            # Format timestamp properly
            start_time = self._normalize_timestamp(root_task.created_at)
            lines.append(f"Start Time: {start_time.isoformat()}")
            lines.append("")

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

    # ==================== RICH FORMATTING METHODS ====================

    def _format_trace_rich(self, events: List[Dict[str, Any]], dag: TaskDAG, root_task: TaskNode) -> str:
        """Format execution trace using Rich library for beautiful CLI output."""
        try:
            from rich.console import Console
            from rich.tree import Tree

            console = Console()

            # Build rich tree
            tree = self._build_rich_tree(events, dag, root_task)

            # Build summary panel
            if self.show_summary:
                summary = self._format_summary_rich(events, root_task)
            else:
                summary = None

            # Render to string
            with console.capture() as capture:
                console.print(tree)
                if summary:
                    console.print()
                    console.print(summary)

            return capture.get()
        except Exception as e:
            # Fallback to plain text if Rich fails
            return f"Error generating Rich visualization: {e}\nFalling back to plain text.\n\n" + self.visualize(source=dag)

    def _build_rich_tree(self, events: List[Dict[str, Any]], dag: TaskDAG, root_task: TaskNode) -> "Tree":
        """Build Rich Tree from execution events."""
        from rich.tree import Tree

        # Create root with execution info
        root_label = (
            f"📊 [bold cyan]Execution Trace[/bold cyan] "
            f"[dim][{root_task.task_id[:8]}][/dim]"
        )
        tree = Tree(root_label)

        # Group events by task
        task_events = defaultdict(list)
        for event in events:
            task_events[event['task'].task_id].append(event)

        # Build tree recursively
        visited = set()
        self._add_task_to_tree(tree, root_task, task_events, dag, visited, depth=0)

        return tree

    def _add_task_to_tree(self, parent_node: "Tree", task: TaskNode, task_events: Dict,
                         dag: TaskDAG, visited: Set[str], depth: int):
        """Recursively add tasks to Rich tree."""
        if task.task_id in visited:
            parent_node.add("[dim]↻ Circular reference[/dim]")
            return

        visited.add(task.task_id)

        # Create task node with status emoji and goal
        status_emoji = self._get_status_emoji(task.status)
        task_label = f"{status_emoji} [bold blue][D{depth}][/bold blue] {task.goal[:80]}"
        task_node = parent_node.add(task_label)

        # Add module executions
        if task.task_id in task_events:
            events = sorted(task_events[task.task_id], key=lambda e: e.get('timestamp', 0))
            for event in events:
                if event['type'] == 'module_execution':
                    module_panel = self._format_module_rich(event['module_result'])
                    task_node.add(module_panel)

        # Recurse to subtasks
        if task.subgraph_id and dag:
            subgraph = dag.get_subgraph(task.subgraph_id)
            if subgraph:
                for child in subgraph.get_all_tasks(include_subgraphs=False):
                    self._add_task_to_tree(
                        task_node, child, task_events, dag, visited, depth + 1
                    )

    def _format_module_rich(self, module_result) -> "Panel":
        """Format module execution as Rich Panel."""
        from rich.panel import Panel
        import re

        module_name = module_result.module_name
        emoji = self._get_module_emoji(module_name)

        sections = []

        # Input section
        sections.append("[bold]📥 Input:[/bold]")
        sections.append(self._format_value_rich(module_result.input, indent="  "))
        sections.append("")

        # Reasoning (if available)
        reasoning = self._extract_reasoning(module_result)
        if reasoning:
            sections.append("[bold yellow]💭 Reasoning:[/bold yellow]")
            reasoning_text = reasoning[:200] + "..." if len(reasoning) > 200 and not self.verbose else reasoning
            sections.append(f"[dim]  {reasoning_text}[/dim]")
            sections.append("")

        # Tool calls (if available)
        tool_calls = self._extract_tool_calls(module_result)
        if tool_calls:
            sections.append(self._format_tool_calls_rich(tool_calls))
            sections.append("")

        # Output section
        sections.append("[bold]📤 Output:[/bold]")
        sections.append(self._format_value_rich(module_result.output, indent="  "))
        sections.append("")

        # Metrics
        if self.show_metrics:
            metrics_line = self._format_metrics_inline(module_result)
            sections.append(metrics_line)

        content = "\n".join(sections)

        # Create panel
        title = f"{emoji} [bold magenta]{module_name.capitalize()}[/bold magenta]"
        return Panel(
            content,
            title=title,
            border_style="blue",
            padding=(1, 2),
            expand=False
        )

    def _extract_reasoning(self, module_result) -> Optional[str]:
        """Extract reasoning from multiple sources."""
        import re

        # Check metadata first
        if module_result.metadata and 'reasoning' in module_result.metadata:
            return str(module_result.metadata['reasoning'])

        # Check output object
        if hasattr(module_result.output, 'reasoning'):
            return str(module_result.output.reasoning)

        # Parse from messages (DSPy signatures often have reasoning field)
        if module_result.messages:
            for msg in module_result.messages:
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    # Try to extract reasoning with regex
                    patterns = [
                        r'(?:Reasoning|Analysis|Thought):\s*(.+?)(?:\n\n|\n(?=[A-Z])|$)',
                        r'<reasoning>(.*?)</reasoning>',
                    ]
                    for pattern in patterns:
                        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                        if match:
                            return match.group(1).strip()

        return None

    def _extract_tool_calls(self, module_result) -> List[Dict]:
        """Extract tool calls from multiple sources."""
        tool_calls = []

        # Source 1: Explicit metadata
        if module_result.metadata and 'tool_calls' in module_result.metadata:
            calls = module_result.metadata['tool_calls']
            if isinstance(calls, list):
                tool_calls.extend(calls)

        # Source 2: OpenAI-style messages
        if module_result.messages:
            assistant_calls = {}
            for msg in module_result.messages:
                if msg.get('role') == 'assistant' and 'tool_calls' in msg:
                    for tc in msg['tool_calls']:
                        call_id = tc.get('id')
                        assistant_calls[call_id] = {
                            'name': tc.get('function', {}).get('name'),
                            'args': tc.get('function', {}).get('arguments'),
                            'result': None
                        }
                elif msg.get('role') == 'tool':
                    # Match result to call
                    call_id = msg.get('tool_call_id')
                    if call_id in assistant_calls:
                        assistant_calls[call_id]['result'] = msg.get('content')

            tool_calls.extend(assistant_calls.values())

        return tool_calls

    def _format_tool_calls_rich(self, tool_calls: List[Dict]) -> str:
        """Format tool calls with tree structure."""
        import json

        lines = ["🛠️  [bold green]Tool Calls:[/bold green]"]
        for i, call in enumerate(tool_calls):
            is_last = (i == len(tool_calls) - 1)
            connector = "└─" if is_last else "├─"

            name = call.get('name', 'unknown')
            args = call.get('args', '')

            # Format args (truncate if long)
            if isinstance(args, str) and len(args) > 50:
                args = args[:50] + "..."
            elif isinstance(args, dict):
                args = json.dumps(args)
                if len(args) > 50:
                    args = args[:50] + "..."

            lines.append(f"  {connector} [cyan]{name}[/cyan]([dim]{args}[/dim])")

            # Add result if available
            result = call.get('result')
            if result:
                continuation = "  " if is_last else "│ "
                result_preview = result[:80] + "..." if len(result) > 80 else result
                lines.append(f"  {continuation}└─ [green]✓[/green] {result_preview}")

        return "\n".join(lines)

    def _format_metrics_inline(self, module_result) -> str:
        """Format metrics inline with icons."""
        metrics_parts = []

        # Duration
        duration_str = f"⏱️  [yellow]{module_result.duration:.2f}s[/yellow]"
        metrics_parts.append(duration_str)

        # Token metrics
        if module_result.token_metrics:
            tm = module_result.token_metrics
            if tm.total_tokens > 0:
                token_str = f"📊 [cyan]{tm.prompt_tokens}/{tm.completion_tokens} tokens[/cyan]"
                metrics_parts.append(token_str)
            if tm.cost > 0:
                cost_indicator = self._get_cost_indicator(tm.cost)
                cost_str = f"💰 {cost_indicator} [green]${tm.cost:.6f}[/green]"
                metrics_parts.append(cost_str)
            if tm.model:
                metrics_parts.append(f"🤖 [dim]{tm.model}[/dim]")

        return " | ".join(metrics_parts)

    def _format_value_rich(self, value: Any, indent: str = "") -> str:
        """Format value with Rich markup."""
        import json

        if value is None:
            return f"{indent}[dim](none)[/dim]"
        elif isinstance(value, str):
            if len(value) > 150 and not self.verbose:
                return f"{indent}[dim]{value[:150]}...[/dim]"
            return f"{indent}{value}"
        elif isinstance(value, dict):
            json_str = json.dumps(value, indent=2)
            if len(json_str) > 200 and not self.verbose:
                return f"{indent}[dim]{json_str[:200]}...[/dim]"
            return f"{indent}[cyan]{json_str}[/cyan]"
        elif isinstance(value, (list, tuple)):
            if len(str(value)) > 150 and not self.verbose:
                return f"{indent}[dim]{str(value)[:150]}...[/dim]"
            return f"{indent}{str(value)}"
        else:
            return f"{indent}{str(value)}"

    def _format_summary_rich(self, events: List[Dict[str, Any]], root_task: TaskNode) -> "Panel":
        """Create Rich summary panel."""
        from rich.table import Table
        from rich.panel import Panel

        # Calculate metrics (reuse existing logic from _add_summary)
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

        # Create table
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="yellow", justify="right")

        table.add_row("Total Tasks", str(task_count))
        table.add_row("✅ Completed", f"[green]{completed_count}[/green]")
        if failed_count > 0:
            table.add_row("❌ Failed", f"[red]{failed_count}[/red]")
        table.add_row("Max Depth", str(max_depth))
        table.add_row("Duration", f"{total_duration:.2f}s")
        if total_tokens > 0:
            table.add_row("Total Tokens", f"{total_tokens:,}")
        if total_cost > 0:
            cost_indicator = self._get_cost_indicator(total_cost)
            table.add_row("Total Cost", f"{cost_indicator} ${total_cost:.6f}")

        return Panel(
            table,
            title="📊 [bold]Execution Summary[/bold]",
            border_style="green",
            padding=(1, 2)
        )

    def _get_status_emoji(self, status: TaskStatus) -> str:
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

    def _get_module_emoji(self, module_name: str) -> str:
        """Get emoji for module type."""
        return {
            "atomizer": "🔍",
            "planner": "📝",
            "executor": "⚡",
            "aggregator": "🔄",
            "verifier": "✓"
        }.get(module_name.lower(), "📦")

    def _get_cost_indicator(self, cost: float) -> str:
        """Get visual indicator for cost level."""
        if cost < 0.001:
            return "🟢"  # Cheap
        elif cost < 0.01:
            return "🟡"  # Moderate
        else:
            return "🔴"  # Expensive

    # ==================== END RICH FORMATTING ====================

    # ==================== MLFLOW INTEGRATION ====================

    def _get_mlflow_client(self):
        """Lazy initialization of MLflow client."""
        if self._mlflow_client is None:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                self._mlflow_client = mlflow
            except ImportError:
                raise ImportError(
                    "mlflow package required for MLflow visualization. "
                    "Install with: pip install mlflow>=2.18.0"
                )
        return self._mlflow_client

    def _fetch_mlflow_traces(self, execution_id: str) -> List[Any]:
        """
        Fetch all MLflow traces for the given execution_id.

        Strategy:
        1) Find runs where run name equals the execution_id or the tag 'execution_id' matches.
        2) Query traces scoped by those run_ids with include_spans=True so span data is present.
        3) If none found via run_id scoping, fall back to scanning experiments and filtering by
           trace tags ('execution_id' or 'mlflow.trace.session').

        Note: include_spans=True requires experiments to use an HTTP-served artifact root
        (e.g., mlflow-artifacts:/). Old experiments with container-local paths won't return spans
        to the client. The API-level Rich formatter handles both cases gracefully.
        """
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=self.mlflow_tracking_uri)

        try:
            experiments = client.search_experiments()
            exp_ids = [exp.experiment_id for exp in experiments]

            # Prefer scoping to configured experiment (by name), if provided
            if self.mlflow_experiment_name:
                try:
                    exp = next((e for e in experiments if e.name == self.mlflow_experiment_name), None)
                    if exp:
                        exp_ids = [exp.experiment_id]
                except Exception:
                    pass

            # Step 1: find matching runs
            matching_run_ids: Set[str] = set()
            for exp_id in exp_ids:
                # Some servers don't support OR in filter; query twice
                for flt in [
                    f"tags.execution_id = '{execution_id}'",
                    f"tags.mlflow.runName = '{execution_id}'",
                ]:
                    try:
                        runs = client.search_runs([exp_id], filter_string=flt, max_results=200)
                        for r in runs:
                            rid = getattr(getattr(r, 'info', r), 'run_id', None)
                            if rid:
                                matching_run_ids.add(rid)
                    except Exception:
                        # ignore invalid filter errors per server
                        continue

            # Step 2: collect traces for those runs (with spans)
            collected: List[Any] = []
            for rid in matching_run_ids:
                try:
                    # Scope by the run's own experiment to avoid cross-experiment issues
                    try:
                        run = client.get_run(rid)
                        run_exp_ids = [run.info.experiment_id]
                    except Exception:
                        run_exp_ids = exp_ids
                    traces = client.search_traces(experiment_ids=run_exp_ids, run_id=rid, include_spans=True)
                    collected.extend(traces)
                except Exception:
                    # if run-scoped fetch fails, continue
                    continue

            if collected:
                return collected

            # Step 3: fallback — scan each experiment and filter by trace tags
            all_traces: List[Any] = []
            for exp_id in exp_ids:
                try:
                    traces = client.search_traces(experiment_ids=[exp_id], include_spans=True)
                except Exception:
                    # try without spans
                    try:
                        traces = client.search_traces(experiment_ids=[exp_id])
                    except Exception:
                        continue

                for t in traces:
                    info = getattr(t, 'info', None)
                    tags = getattr(info, 'tags', {}) if info else {}
                    if isinstance(tags, dict) and (
                        tags.get('execution_id') == execution_id or tags.get('mlflow.trace.session') == execution_id
                    ):
                        all_traces.append(t)

            if not all_traces:
                raise ValueError(
                    f"No MLflow traces found for execution {execution_id}. "
                    f"Ensure the MLflow experiment stores artifacts via mlflow-artifacts:/ and that "
                    f"run name or tags include the execution_id."
                )
            return all_traces

        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to fetch MLflow traces: {e}")

    def _mlflow_spans_to_events(self, traces: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert MLflow trace spans to event structure for visualization.

        Args:
            traces: List of MLflow Trace objects

        Returns:
            List of event dictionaries sorted chronologically
        """
        events = []

        # Build span map once for O(1) lookups (performance optimization)
        span_map = self._build_span_map(traces)

        for trace in traces:
            # Each trace has a list of spans
            if not hasattr(trace, 'data') or not hasattr(trace.data, 'spans'):
                continue

            for span in trace.data.spans:
                # Extract span timing
                start_time = datetime.fromtimestamp(span.start_time_ns / 1e9, tz=timezone.utc)
                end_time = datetime.fromtimestamp(span.end_time_ns / 1e9, tz=timezone.utc) if span.end_time_ns else start_time
                duration_s = (span.end_time_ns - span.start_time_ns) / 1e9 if span.end_time_ns else 0

                # Extract attributes
                attrs = span.attributes if hasattr(span, 'attributes') else {}

                # Build module_result-like structure
                module_result = type('ModuleResult', (), {
                    'inputs': span.inputs if hasattr(span, 'inputs') else {},
                    'output': span.outputs if hasattr(span, 'outputs') else None,
                    'metadata': attrs,
                    'messages': self._extract_messages_from_span(span),
                    'token_metrics': self._extract_token_metrics_from_span(span),
                    'timestamp': start_time,
                    'duration': duration_s,
                })()

                # Create event
                event = {
                    'type': 'module_execution',
                    'timestamp': start_time,
                    'module_name': span.name,
                    'module_result': module_result,
                    'task_id': attrs.get('task_id'),
                    'depth': self._calculate_span_depth(span, span_map),
                    'span_id': span.span_id if hasattr(span, 'span_id') else None,
                    'parent_span_id': span.parent_id if hasattr(span, 'parent_id') else None,
                }
                events.append(event)

        # Sort chronologically
        events.sort(key=lambda e: e['timestamp'])
        return events

    def _extract_messages_from_span(self, span) -> List[Dict]:
        """Extract OpenAI-style messages from span attributes."""
        attrs = span.attributes if hasattr(span, 'attributes') else {}

        # Check for messages in attributes
        if 'messages' in attrs:
            return attrs['messages']

        # Check for messages in inputs
        if hasattr(span, 'inputs') and isinstance(span.inputs, dict):
            if 'messages' in span.inputs:
                return span.inputs['messages']

        return []

    def _extract_token_metrics_from_span(self, span):
        """Extract token usage metrics from span attributes."""
        attrs = span.attributes if hasattr(span, 'attributes') else {}

        # MLflow DSPy autolog stores token usage in attributes
        token_usage = attrs.get('token_usage', {})
        if not token_usage and hasattr(span, 'outputs'):
            # Sometimes in outputs
            outputs = span.outputs if isinstance(span.outputs, dict) else {}
            token_usage = outputs.get('token_usage', {})

        if token_usage:
            # Note: Use 'cost' not 'cost_usd' to match attribute expected by existing code
            cost_value = attrs.get('cost_usd') or attrs.get('cost') or token_usage.get('cost_usd') or token_usage.get('cost') or 0.0
            return type('TokenMetrics', (), {
                'prompt_tokens': token_usage.get('prompt_tokens', 0),
                'completion_tokens': token_usage.get('completion_tokens', 0),
                'total_tokens': token_usage.get('total_tokens', 0),
                'cost': cost_value,  # Use 'cost' attribute (not cost_usd) for consistency
                'model': attrs.get('model') or token_usage.get('model'),
            })()

        return None

    def _build_span_map(self, traces: List[Any]) -> Dict[str, Any]:
        """
        Build span lookup map from traces.

        Args:
            traces: All traces

        Returns:
            Dict mapping span_id to span object
        """
        span_map = {}
        for trace in traces:
            if hasattr(trace, 'data') and hasattr(trace.data, 'spans'):
                for s in trace.data.spans:
                    if hasattr(s, 'span_id'):
                        span_map[s.span_id] = s
        return span_map

    def _calculate_span_depth(self, span, span_map: Dict[str, Any]) -> int:
        """
        Calculate depth of span in hierarchy by traversing parent_id chain.

        Args:
            span: Current span
            span_map: Pre-built map of span_id to span objects

        Returns:
            Depth level (0 = root)
        """
        depth = 0
        current_span = span

        # Traverse parent chain
        max_depth = 20  # Safety limit
        while hasattr(current_span, 'parent_id') and current_span.parent_id and depth < max_depth:
            parent_id = current_span.parent_id
            if parent_id in span_map:
                current_span = span_map[parent_id]
                depth += 1
            else:
                break

        return depth

    def visualize_from_mlflow(self, execution_id: str) -> str:
        """
        Visualize execution using MLflow traces (most complete data from DSPy autolog).

        This provides the richest visualization with full DSPy module execution traces,
        including tool calls, reasoning chains, and hierarchical span relationships.

        Args:
            execution_id: Execution ID to visualize

        Returns:
            Rich-formatted visualization string (if fancy=True) or plain text

        Raises:
            ImportError: If mlflow not installed
            ValueError: If no traces found for execution_id
        """
        try:
            # Fetch traces from MLflow
            traces = self._fetch_mlflow_traces(execution_id)

            # Convert spans to events
            events = self._mlflow_spans_to_events(traces)

            if not events:
                return f"No span data found in MLflow traces for execution {execution_id}"

            task_model = self._build_mlflow_task_tree_model(traces)

            # Use Rich task-tree if task metadata available, else fallback to span-tree
            if self.fancy:
                if task_model.get("tasks"):
                    return self._format_mlflow_task_tree_rich(execution_id, traces, task_model)
                return self._format_mlflow_traces_rich(execution_id, traces)
            else:
                # Plain text format
                lines = []
                lines.append(f"=== LLM Trace (MLflow) for Execution: {execution_id} ===\n")

                for event in events:
                    module_result = event.get('module_result')
                    if not module_result:
                        continue

                    indent = "  " * event.get('depth', 0)
                    lines.append(f"{indent}[{event['module_name']}]")

                    if self.show_metrics and module_result.token_metrics:
                        tm = module_result.token_metrics
                        lines.append(
                            f"{indent}  Tokens: {tm.total_tokens} "
                            f"(prompt: {tm.prompt_tokens}, completion: {tm.completion_tokens})"
                        )
                        if tm.cost and tm.cost > 0:
                            lines.append(f"{indent}  Cost: ${tm.cost:.6f}")

                    if module_result.duration:
                        lines.append(f"{indent}  Duration: {module_result.duration:.2f}s")

                    lines.append("")

                return "\n".join(lines)

        except ImportError as e:
            return f"MLflow not available: {e}\nInstall with: pip install mlflow>=2.18.0"
        except Exception as e:
            return f"Error fetching MLflow traces: {e}"

    # ==================== END MLFLOW INTEGRATION ====================

    def _format_mlflow_traces_rich(self, execution_id: str, traces: List[Any]) -> str:
        """
        Render MLflow spans in a hierarchical Rich tree preserving parent/child relationships.

        Shows per-span metrics (duration, tokens, cost, model) and provides a
        compact, CLI-friendly overview with a summary panel.
        """
        try:
            from rich.tree import Tree
            from rich.panel import Panel
            from rich.table import Table

            console = self._make_console()

            # Build trees per trace
            roots: List[Tree] = []
            total_tokens = 0
            total_cost = 0.0
            total_spans = 0
            total_duration = 0.0

            for trace in traces:
                info = getattr(trace, 'info', None)
                trace_id = getattr(info, 'trace_id', 'unknown')
                span_list = getattr(getattr(trace, 'data', None), 'spans', []) or []
                total_spans += len(span_list)

                # Map spans by id and by parent
                span_map = {getattr(s, 'span_id', None): s for s in span_list}
                children: Dict[str, List[Any]] = defaultdict(list)
                for s in span_list:
                    pid = getattr(s, 'parent_id', None)
                    if pid:
                        children[pid].append(s)

                # Identify root spans (no parent or parent missing)
                roots_spans = [s for s in span_list if not getattr(s, 'parent_id', None) or getattr(s, 'parent_id') not in span_map]

                # Trace label
                # Try to compute a simple wall duration from root span(s)
                def span_duration(s):
                    return ((getattr(s, 'end_time_ns', 0) or 0) - (getattr(s, 'start_time_ns', 0) or 0)) / 1e9
                root_durations = [span_duration(s) for s in roots_spans if span_duration(s) > 0]
                root_total = sum(root_durations) if root_durations else 0.0

                trace_label = (
                    f"🧵 [bold cyan]Trace[/bold cyan] [dim]{trace_id[:8]}[/dim]  "
                    f"([yellow]{len(span_list)} spans[/yellow] • {root_total:.2f}s)"
                )
                trace_tree = Tree(trace_label)

                # Recursively add spans
                def add_span(node: Tree, span: Any):
                    nonlocal total_tokens, total_cost, total_duration

                    # Duration
                    if hasattr(span, 'start_time_ns') and getattr(span, 'end_time_ns', None):
                        duration = (span.end_time_ns - span.start_time_ns) / 1e9
                    else:
                        duration = 0.0
                    total_duration += max(0.0, duration)

                    # Tokens / cost / model
                    tm = self._extract_token_metrics_from_span(span)
                    tok_str = ""
                    if tm:
                        total_tokens += tm.total_tokens
                        total_cost += tm.cost or 0.0
                        tok_items = []
                        if tm.total_tokens:
                            tok_items.append(f"tokens: {tm.total_tokens}")
                        if tm.prompt_tokens or tm.completion_tokens:
                            tok_items.append(f"p:{tm.prompt_tokens}/c:{tm.completion_tokens}")
                        if tm.model:
                            tok_items.append(f"model: {tm.model}")
                        if tm.cost:
                            tok_items.append(f"cost: ${tm.cost:.6f}")
                        tok_str = " | ".join(tok_items)

                    # Reasoning (snippet)
                    attrs = getattr(span, 'attributes', {}) or {}
                    reasoning = attrs.get('reasoning') if isinstance(attrs, dict) else None

                    name = getattr(span, 'name', 'span')
                    emoji = self._get_module_emoji(name)
                    label = f"• {emoji} [bold magenta]{name}[/bold magenta]  [dim]{duration:.2f}s[/dim]"
                    if tok_str:
                        label += f"  —  {tok_str}"
                    span_node = node.add(label)

                    # Tool calls (if any)
                    tool_calls = self._extract_tool_calls_from_span(span)
                    if tool_calls:
                        tbl = Table(box=None, show_header=True, header_style="bold blue")
                        tbl.add_column("Tool", style="cyan")
                        tbl.add_column("Args", style="yellow")
                        tbl.add_column("Result", style="green")
                        preview_len = 80 if not self.verbose else 160
                        for call in tool_calls[:8] if not self.verbose else tool_calls:
                            tool_name = call.get('tool') or call.get('tool_name') or call.get('name') or 'tool'
                            # Prefer toolkit.tool format if available
                            if call.get('toolkit') and tool_name:
                                tool_name = f"{call['toolkit']}.{tool_name}"
                            args_v = call.get('arguments') or call.get('args') or call.get('input')
                            out_v = call.get('output') or call.get('result')
                            def pv(v):
                                s = str(v)
                                return (s[:preview_len] + '…') if len(s) > preview_len and not self.verbose else s
                            tbl.add_row(str(tool_name), pv(args_v), pv(out_v))
                        span_node.add(Panel(tbl, title="🛠 Tool Calls", border_style="blue", padding=(0,1)))

                    # Inputs/Outputs compact preview (trim unless verbose)
                    inputs = getattr(span, 'inputs', None)
                    outputs = getattr(span, 'outputs', None)
                    def preview(val: Any) -> str:
                        s = str(val)
                        limit = 140 if not self.verbose else 400
                        return (s[:limit] + '…') if len(s) > limit else s

                    if inputs and (self.verbose or self.show_io):
                        span_node.add(Panel(f"[bold]📥 Input[/bold]\n{preview(inputs)}", border_style="blue", padding=(0,1)))
                    if reasoning:
                        span_node.add(Panel(f"[bold yellow]💭 Reasoning[/bold yellow]\n[dim]{preview(reasoning)}[/dim]", border_style="yellow", padding=(0,1)))
                    if outputs is not None and (self.verbose or self.show_io):
                        span_node.add(Panel(f"[bold]📤 Output[/bold]\n{preview(outputs)}", border_style="green", padding=(0,1)))

                    # Children
                    for child in children.get(getattr(span, 'span_id', None), []):
                        add_span(span_node, child)

                for root in roots_spans:
                    add_span(trace_tree, root)

                roots.append(trace_tree)

            # Build summary
            summary = Table(show_header=False, box=None)
            summary.add_row("Total Traces", str(len(traces)))
            summary.add_row("Total Spans", str(total_spans))
            summary.add_row("Total Duration", f"{total_duration:.2f}s")
            if total_tokens:
                summary.add_row("Total Tokens", f"{total_tokens:,}")
            if total_cost:
                summary.add_row("Total Cost", f"${total_cost:.6f}")

            # Render
            with console.capture() as cap:
                exp = f" [dim](exp: {self.mlflow_experiment_name})[/dim]" if getattr(self, 'mlflow_experiment_name', None) else ""
                header = f"📊 [bold green]LLM Traces (MLflow) for Execution[/bold green]: [cyan]{execution_id}[/cyan]{exp}"
                console.print(header)
                for t in roots:
                    console.print(t)
                    console.print()
                console.print(Panel(summary, title="Summary", border_style="green"))
            return cap.get()
        except Exception as e:
            return f"Failed to render MLflow span tree: {e}"

    def build_mlflow_trace_data(self, execution_id: str) -> Dict[str, Any]:
        """Return structured MLflow data suitable for interactive UI consumption."""
        traces = self._fetch_mlflow_traces(execution_id)
        if not traces:
            return {
                "execution_id": execution_id,
                "experiment": self.mlflow_experiment_name,
                "tasks": [],
                "summary": {},
                "traces": [],
                "fallback_spans": [],
            }

        task_model = self._build_mlflow_task_tree_model(traces)

        trace_infos = []
        for tr in traces:
            info = getattr(tr, 'info', None)
            trace_infos.append({
                "trace_id": getattr(info, 'trace_id', None),
                "run_id": getattr(info, 'run_id', None),
                "span_count": len(getattr(getattr(tr, 'data', None), 'spans', []) or []),
            })

        return {
            "execution_id": execution_id,
            "experiment": self.mlflow_experiment_name,
            "tasks": task_model.get("tasks", []),
            "summary": task_model.get("summary", {}),
            "traces": trace_infos,
            "fallback_spans": task_model.get("fallback_spans", []),
        }

    def _format_mlflow_task_tree_rich(self, execution_id: str, traces: List[Any], task_model: Dict[str, Any]) -> str:
        from rich.tree import Tree
        from rich.panel import Panel
        from rich.table import Table

        tasks = task_model.get("tasks", [])
        if not tasks:
            return self._format_mlflow_traces_rich(execution_id, traces)

        console = self._make_console()
        root_label = f"🌳 [bold cyan]Task Execution Tree[/bold cyan] [dim]{execution_id}[/dim]"
        if self.mlflow_experiment_name:
            root_label += f" [dim](exp: {self.mlflow_experiment_name})[/dim]"
        tree = Tree(root_label)

        children_map: Dict[str, List[Dict[str, Any]]] = {}
        lookup: Dict[str, Dict[str, Any]] = {}
        roots: List[Dict[str, Any]] = []

        for task in tasks:
            lookup[task["task_id"]] = task
        for task in tasks:
            parent = task.get("parent_task_id")
            if parent and parent in lookup:
                children_map.setdefault(parent, []).append(task)
            else:
                roots.append(task)

        def render_task(node: Tree, task: Dict[str, Any]):
            goal = task.get("goal") or f"Task {task['task_id'][:8]}"
            metrics = task.get("metrics", {})
            badge = []
            if metrics.get("duration"):
                badge.append(f"{metrics['duration']:.2f}s")
            if metrics.get("tokens"):
                badge.append(f"{metrics['tokens']} tok")
            if metrics.get("cost"):
                badge.append(f"${metrics['cost']:.4f}")
            meta_tags = []
            if task.get('module'):
                meta_tags.append(str(task['module']))
            if task.get('task_type') or task.get('node_type'):
                meta_tags.append("/".join([
                    str(task.get('task_type', '?')),
                    str(task.get('node_type', '?')),
                ]))

            label = f"🧩 [bold]{goal[:80]}[/bold]"
            if badge:
                label += "  [dim]" + " • ".join(badge) + "[/dim]"
            if meta_tags:
                label += "  [dim](" + ", ".join(meta_tags) + ")[/dim]"
            task_node = node.add(label)

            for span in task.get('spans', [])[:8] if not self.verbose else task.get('spans', []):
                extras = []
                if span.get('duration'):
                    extras.append(f"{span['duration']:.2f}s")
                if span.get('tokens'):
                    extras.append(f"{span['tokens']} tok")
                if span.get('model'):
                    extras.append(str(span['model']))
                if span.get('cost'):
                    extras.append(f"${span['cost']:.4f}")
                suffix = "  [dim]" + " • ".join(extras) + "[/dim]" if extras else ""
                task_node.add(f"• {self._get_module_emoji(span.get('name', 'span'))} [magenta]{span.get('name', 'span')}[/magenta]{suffix}")

            for child in children_map.get(task['task_id'], []):
                render_task(task_node, child)

        for root in roots:
            render_task(tree, root)

        summary = task_model.get('summary', {})
        with console.capture() as cap:
            console.print(tree)
            table = Table(show_header=False, box=None)
            if summary.get('total_tasks') is not None:
                table.add_row("Tasks", str(summary['total_tasks']))
            if summary.get('total_spans') is not None:
                table.add_row("Spans", str(summary['total_spans']))
            if summary.get('total_duration') is not None:
                table.add_row("Duration", f"{summary['total_duration']:.2f}s")
            if summary.get('total_tokens'):
                table.add_row("Tokens", f"{summary['total_tokens']:,}")
            if summary.get('total_cost'):
                table.add_row("Cost", f"${summary['total_cost']:.4f}")
            console.print(Panel(table, title="Summary", border_style="green"))
        return cap.get()

    def _build_mlflow_task_tree_model(self, traces: List[Any]) -> Dict[str, Any]:
        """Build structured task + span model from MLflow traces."""
        all_spans: List[Any] = []
        for tr in traces:
            spans = getattr(getattr(tr, 'data', None), 'spans', []) or []
            all_spans.extend(spans)

        def _attr(span, *names):
            attrs = getattr(span, 'attributes', {}) or {}
            if not isinstance(attrs, dict):
                return None
            for name in names:
                if name in attrs:
                    return attrs[name]
            return None

        tasks: Dict[str, Dict[str, Any]] = {}
        span_map = {getattr(s, 'span_id', None): s for s in all_spans if getattr(s, 'span_id', None)}
        fallback_spans = []

        for span in all_spans:
            span_type = _attr(span, 'roma.span_type', 'span_type')
            task_id = _attr(span, 'roma.task_id', 'task_id')
            start_ns = getattr(span, 'start_time_ns', 0) or 0
            end_ns = getattr(span, 'end_time_ns', 0) or 0
            duration = max(0.0, (end_ns - start_ns) / 1e9)
            tm = self._extract_token_metrics_from_span(span)

            if not task_id:
                tool_calls = self._extract_tool_calls_from_span(span)
                fallback_spans.append({
                    'span_id': getattr(span, 'span_id', None),
                    'name': getattr(span, 'name', 'span'),
                    'module': _attr(span, 'roma.module', 'module'),
                    'task_id': _attr(span, 'roma.task_id', 'task_id'),
                    'parent_span_id': getattr(span, 'parent_id', None),
                    'parent_id': getattr(span, 'parent_id', None),
                    'start_time': datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat() if start_ns else None,
                    'start_ts': (start_ns / 1e9) if start_ns else None,
                    'duration': duration,
                    'tokens': tm.total_tokens if tm else None,
                    'cost': tm.cost if tm else None,
                    'model': tm.model if tm else None,
                    'tool_calls': tool_calls,
                    'inputs': getattr(span, 'inputs', None),
                    'outputs': getattr(span, 'outputs', None),
                    'reasoning': _attr(span, 'reasoning'),
                })
                continue

            entry = tasks.setdefault(task_id, {
                'task_id': task_id,
                'parent_task_id': None,
                'goal': None,
                'module': _attr(span, 'roma.module', 'module'),
                'task_type': _attr(span, 'roma.task_type', 'task_type'),
                'node_type': _attr(span, 'roma.node_type', 'node_type'),
                'status': _attr(span, 'roma.status', 'status'),
                'depth': _attr(span, 'roma.depth', 'depth') or 0,
                'metrics': {
                    'duration': 0.0,
                    'tokens': 0,
                    'cost': 0.0,
                },
                'spans': [],
                '_first_span_id': None,
                '_first_start_ns': None,
            })

            if entry['_first_span_id'] is None or (entry['_first_start_ns'] is not None and start_ns < entry['_first_start_ns']):
                entry['_first_span_id'] = getattr(span, 'span_id', None)
                entry['_first_start_ns'] = start_ns

            if entry['goal'] is None:
                goal = _attr(span, 'goal', 'roma.goal', 'task_goal')
                if not goal:
                    inputs = getattr(span, 'inputs', {}) or {}
                    if isinstance(inputs, dict):
                        goal = inputs.get('goal') or inputs.get('original_goal')
                if goal:
                    entry['goal'] = str(goal)

            if span_type != "module_wrapper":
                entry['metrics']['duration'] += duration
                if tm:
                    entry['metrics']['tokens'] += tm.total_tokens or 0
                    entry['metrics']['cost'] += tm.cost or 0.0

                entry['spans'].append({
                    'span_id': getattr(span, 'span_id', None),
                    'parent_id': getattr(span, 'parent_id', None),
                    'name': getattr(span, 'name', 'span'),
                    'start_ns': start_ns,
                    'start_time': datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc).isoformat() if start_ns else None,
                    'duration': duration,
                    'tokens': tm.total_tokens if tm else None,
                    'cost': tm.cost if tm else None,
                    'model': tm.model if tm else None,
                    'tool_calls': self._extract_tool_calls_from_span(span),
                    'inputs': getattr(span, 'inputs', None),
                    'outputs': getattr(span, 'outputs', None),
                    'reasoning': _attr(span, 'reasoning'),
                })

        # Determine task parents
        for task_id, entry in tasks.items():
            first_span = span_map.get(entry['_first_span_id']) if entry['_first_span_id'] else None
            parent_tid = None
            if first_span:
                parent_tid = _attr(first_span, 'roma.parent_task_id', 'parent_task_id')
                if parent_tid == task_id:
                    parent_tid = None
                if not parent_tid:
                    current = first_span
                    visited = 0
                    while current and getattr(current, 'parent_id', None) and visited < 50:
                        parent_span = span_map.get(current.parent_id)
                        if not parent_span:
                            break
                        maybe_parent = _attr(parent_span, 'roma.task_id', 'task_id')
                        if maybe_parent and maybe_parent != task_id:
                            parent_tid = maybe_parent
                            break
                        current = parent_span
                        visited += 1
            if parent_tid and parent_tid in tasks:
                entry['parent_task_id'] = parent_tid

        # Prepare output list
        task_list = []
        for entry in tasks.values():
            entry['spans'].sort(key=lambda sp: sp['start_ns'])
            for sp in entry['spans']:
                start_ns = sp.pop('start_ns', None)
                if start_ns:
                    sp['start_ts'] = start_ns / 1e9
            entry.pop('_first_span_id', None)
            entry.pop('_first_start_ns', None)
            task_list.append(entry)

        summary = {
            'total_tasks': len(task_list),
            'total_spans': sum(len(entry['spans']) for entry in task_list),
            'total_duration': 0.0,
            'total_tokens': 0,
            'total_cost': 0.0,
        }
        for entry in task_list:
            summary['total_duration'] += entry['metrics']['duration']
            summary['total_tokens'] += entry['metrics']['tokens']
            summary['total_cost'] += entry['metrics']['cost']

        return {
            'tasks': task_list,
            'summary': summary,
            'fallback_spans': fallback_spans,
        }
    def _extract_tool_calls_from_span(self, span: Any) -> List[Dict[str, Any]]:
        """Heuristically extract tool call records from an MLflow span.

        Looks in attributes, outputs, inputs, and OpenAI-style assistant messages.
        Each returned item is a dict with keys like: tool/tool_name, toolkit, arguments, output.
        """
        calls: List[Dict[str, Any]] = []
        attrs = getattr(span, 'attributes', {}) or {}
        inputs = getattr(span, 'inputs', {}) or {}
        outputs = getattr(span, 'outputs', {}) or {}

        # 1) Direct metadata field
        if isinstance(attrs, dict) and isinstance(attrs.get('tool_calls'), list):
            for c in attrs['tool_calls']:
                if isinstance(c, dict):
                    calls.append(c)

        # 2) Structured attributes (single tool)
        single = {}
        for key in ('tool', 'tool_name', 'name'):
            if key in attrs:
                single['tool'] = attrs[key]
                break
        for key in ('toolkit', 'tool_class', 'toolkit_class'):
            if key in attrs:
                single['toolkit'] = attrs[key]
                break
        for key in ('arguments', 'args', 'input'):
            if key in attrs:
                single['arguments'] = attrs[key]
                break
        for key in ('output', 'result', 'return'):
            if key in attrs:
                single['output'] = attrs[key]
                break
        if single:
            calls.append(single)

        # 3) Outputs or inputs include tool_calls
        for container in (outputs, inputs):
            if isinstance(container, dict) and isinstance(container.get('tool_calls'), list):
                for c in container['tool_calls']:
                    if isinstance(c, dict):
                        calls.append(c)

        # 4) OpenAI-style assistant messages with tool_calls
        msgs = []
        if isinstance(inputs, dict) and isinstance(inputs.get('messages'), list):
            msgs.extend(inputs['messages'])
        if isinstance(outputs, dict) and isinstance(outputs.get('messages'), list):
            msgs.extend(outputs['messages'])
        for m in msgs:
            if isinstance(m, dict) and m.get('role') == 'assistant':
                tc = m.get('tool_calls')
                if isinstance(tc, list):
                    for c in tc:
                        if isinstance(c, dict):
                            # OpenAI format may nest function/name/arguments
                            name = c.get('function', {}).get('name') if isinstance(c.get('function'), dict) else c.get('name')
                            args = c.get('function', {}).get('arguments') if isinstance(c.get('function'), dict) else c.get('arguments')
                            calls.append({'tool': name, 'arguments': args})

        return calls

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
