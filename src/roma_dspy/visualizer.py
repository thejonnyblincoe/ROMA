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

from .core.signatures.base_models.task_node import TaskNode
from .types import TaskStatus, NodeType
from .core.engine.dag import TaskDAG


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
                 show_timing: bool = True, show_tokens: bool = True):
        """
        Initialize tree visualizer.

        Args:
            use_colors: Whether to use ANSI colors
            show_ids: Whether to show task IDs
            show_timing: Whether to show timing information
            show_tokens: Whether to show token usage and costs
        """
        self.use_colors = use_colors
        self.show_ids = show_ids
        self.show_timing = show_timing
        self.show_tokens = show_tokens
        self.rt_viz = RealTimeVisualizer(use_colors=use_colors)  # Reuse color methods

    def visualize(self, source: Optional[Any] = None, dag: Optional[TaskDAG] = None) -> str:
        """
        Generate tree visualization from solver's last DAG.

        Args:
            source: RecursiveSolver, TaskDAG, or TaskNode to visualize
            dag: Optional DAG to visualize (defaults to solver.last_dag if available)

        Returns:
            String representation of the tree
        """
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

        # Goal (truncate if too long)
        goal = task.goal
        if len(goal) > 60:
            goal = goal[:57] + "..."
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


class HierarchicalVisualizer:
    """
    Main visualizer class that combines all visualization modes.
    Can be integrated with RecursiveSolver for real-time visualization.
    """

    def __init__(self, mode: str = "all", use_colors: bool = True, verbose: bool = True):
        """
        Initialize hierarchical visualizer.

        Args:
            mode: Visualization mode ("realtime", "tree", "timeline", "stats", "all")
            use_colors: Whether to use colored output
            verbose: Whether to show detailed information
        """
        self.mode = mode
        self.realtime = RealTimeVisualizer(use_colors=use_colors, verbose=verbose)
        self.tree = TreeVisualizer(use_colors=use_colors)
        self.timeline = TimelineVisualizer()
        self.stats = StatisticsVisualizer()

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
