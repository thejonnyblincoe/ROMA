"""
Comprehensive execution tracking system for task nodes and module executions.
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import json
from pathlib import Path

from src.roma_dspy.signatures.base_models.task_node import TaskNode
from src.roma_dspy.engine.dag import TaskDAG
from src.roma_dspy.types.module_result import ExecutionEvent, ModuleResult


class ExecutionTracker:
    """
    Comprehensive tracking system for task execution.

    Features:
    - Complete execution history for all nodes
    - Module-level tracking (atomizer, planner, executor, aggregator)
    - Timeline of all execution events
    - Export capabilities for analysis
    - Performance metrics and statistics
    """

    def __init__(self):
        """Initialize the execution tracker."""
        self.nodes: Dict[str, TaskNode] = {}
        self.execution_timeline: List[ExecutionEvent] = []
        self.dags: Dict[str, TaskDAG] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def start_tracking(self) -> None:
        """Mark the start of execution tracking."""
        self.start_time = datetime.now()
        self.add_event(
            ExecutionEvent(
                node_id="SYSTEM",
                module_name="tracker",
                event_type="start",
                metadata={"message": "Execution tracking started"}
            )
        )

    def stop_tracking(self) -> None:
        """Mark the end of execution tracking."""
        self.end_time = datetime.now()
        self.add_event(
            ExecutionEvent(
                node_id="SYSTEM",
                module_name="tracker",
                event_type="stop",
                duration=self.get_total_duration(),
                metadata={"message": "Execution tracking stopped"}
            )
        )

    def track_node(self, node: TaskNode) -> None:
        """
        Track a task node.

        Args:
            node: TaskNode to track
        """
        self.nodes[node.task_id] = node

    def track_dag(self, dag: TaskDAG) -> None:
        """
        Track a DAG and all its nodes.

        Args:
            dag: TaskDAG to track
        """
        self.dags[dag.dag_id] = dag

        # Track all nodes in the DAG
        for task in dag.get_all_tasks(include_subgraphs=True):
            self.track_node(task)

    def track_module_execution(
        self,
        node_id: str,
        module_name: str,
        input_data: Any,
        output_data: Any,
        duration: float,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Track a module execution.

        Args:
            node_id: ID of the task node
            module_name: Name of the module executed
            input_data: Input provided to the module
            output_data: Output produced by the module
            duration: Execution duration in seconds
            error: Error message if execution failed
            metadata: Additional metadata
        """
        # Create execution event
        event = ExecutionEvent(
            node_id=node_id,
            module_name=module_name,
            event_type="complete" if error is None else "error",
            duration=duration,
            metadata={
                "input": str(input_data)[:500],  # Truncate for storage
                "output": str(output_data)[:500],
                "error": error,
                **(metadata or {})
            }
        )

        self.add_event(event)

        # Update node if tracked
        if node_id in self.nodes:
            node = self.nodes[node_id]
            module_result = ModuleResult(
                module_name=module_name,
                input=input_data,
                output=output_data,
                duration=duration,
                error=error,
                metadata=metadata or {}
            )
            updated_node = node.record_module_execution(module_name, module_result)
            self.nodes[node_id] = updated_node

    def add_event(self, event: ExecutionEvent) -> None:
        """
        Add an execution event to the timeline.

        Args:
            event: ExecutionEvent to add
        """
        self.execution_timeline.append(event)

    def get_node_history(self, node_id: str) -> Dict[str, Any]:
        """
        Get complete execution history for a node.

        Args:
            node_id: Task node ID

        Returns:
            Dictionary containing full node history
        """
        if node_id not in self.nodes:
            return {"error": f"Node {node_id} not found"}

        node = self.nodes[node_id]

        # Get all events for this node
        node_events = [
            event for event in self.execution_timeline
            if event.node_id == node_id
        ]

        # Build comprehensive history
        return {
            "node_id": node_id,
            "goal": node.goal,
            "depth": node.depth,
            "status": node.status.value,
            "node_type": node.node_type.value if node.node_type else None,
            "execution_history": {
                name: {
                    "input": result.input,
                    "output": result.output,
                    "duration": result.duration,
                    "error": result.error,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in node.execution_history.items()
            },
            "state_transitions": [
                {
                    "from": t.from_state,
                    "to": t.to_state,
                    "timestamp": t.timestamp.isoformat(),
                    "reason": t.reason
                }
                for t in node.state_transitions
            ],
            "events": [
                {
                    "module": e.module_name,
                    "type": e.event_type,
                    "timestamp": e.timestamp.isoformat(),
                    "duration": e.duration
                }
                for e in node_events
            ],
            "metrics": node.metrics.model_dump(),
            "children": list(node.children),
            "dependencies": list(node.dependencies),
            "result": str(node.result)[:1000] if node.result else None
        }

    def get_module_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for each module type.

        Returns:
            Dictionary with module-level statistics
        """
        module_stats = {
            "atomizer": {"count": 0, "total_duration": 0, "errors": 0},
            "planner": {"count": 0, "total_duration": 0, "errors": 0},
            "executor": {"count": 0, "total_duration": 0, "errors": 0},
            "aggregator": {"count": 0, "total_duration": 0, "errors": 0}
        }

        for event in self.execution_timeline:
            if event.module_name in module_stats:
                stats = module_stats[event.module_name]
                stats["count"] += 1
                if event.duration:
                    stats["total_duration"] += event.duration
                if event.event_type == "error":
                    stats["errors"] += 1

        # Calculate averages
        for module, stats in module_stats.items():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
                stats["error_rate"] = stats["errors"] / stats["count"]

        return module_stats

    def get_depth_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics organized by recursion depth.

        Returns:
            Dictionary with depth-level statistics
        """
        depth_stats = {}

        for node in self.nodes.values():
            depth = node.depth
            if depth not in depth_stats:
                depth_stats[depth] = {
                    "count": 0,
                    "completed": 0,
                    "failed": 0,
                    "forced_execute": 0,
                    "avg_duration": 0,
                    "durations": []
                }

            stats = depth_stats[depth]
            stats["count"] += 1

            if node.status.value == "COMPLETED":
                stats["completed"] += 1
            elif node.status.value == "FAILED":
                stats["failed"] += 1

            if node.should_force_execute():
                stats["forced_execute"] += 1

            if node.execution_duration:
                stats["durations"].append(node.execution_duration)

        # Calculate averages
        for depth, stats in depth_stats.items():
            if stats["durations"]:
                stats["avg_duration"] = sum(stats["durations"]) / len(stats["durations"])
            del stats["durations"]  # Remove raw data

        return depth_stats

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive execution statistics.

        Returns:
            Dictionary with all statistics
        """
        total_nodes = len(self.nodes)
        completed_nodes = sum(
            1 for node in self.nodes.values()
            if node.status.value == "COMPLETED"
        )
        failed_nodes = sum(
            1 for node in self.nodes.values()
            if node.status.value == "FAILED"
        )

        return {
            "summary": {
                "total_nodes": total_nodes,
                "completed": completed_nodes,
                "failed": failed_nodes,
                "success_rate": completed_nodes / total_nodes if total_nodes > 0 else 0,
                "total_duration": self.get_total_duration(),
                "total_events": len(self.execution_timeline)
            },
            "module_statistics": self.get_module_statistics(),
            "depth_statistics": self.get_depth_statistics(),
            "dag_count": len(self.dags)
        }

    def get_total_duration(self) -> Optional[float]:
        """
        Get total execution duration.

        Returns:
            Duration in seconds or None if not complete
        """
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        elif self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return None

    def export_execution_trace(self, format: str = "json") -> Dict[str, Any]:
        """
        Export complete execution trace.

        Args:
            format: Export format (currently only JSON)

        Returns:
            Complete execution trace
        """
        trace = {
            "metadata": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration": self.get_total_duration(),
                "node_count": len(self.nodes),
                "event_count": len(self.execution_timeline)
            },
            "nodes": {
                node_id: self.get_node_history(node_id)
                for node_id in self.nodes.keys()
            },
            "timeline": [
                {
                    "node_id": event.node_id,
                    "module": event.module_name,
                    "type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "duration": event.duration,
                    "metadata": event.metadata
                }
                for event in self.execution_timeline
            ],
            "statistics": self.calculate_statistics()
        }

        if format == "json":
            return trace
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_trace(self, filepath: Path) -> None:
        """
        Save execution trace to file.

        Args:
            filepath: Path to save the trace
        """
        trace = self.export_execution_trace(format="json")

        with open(filepath, "w") as f:
            json.dump(trace, f, indent=2, default=str)

    def get_critical_path(self) -> List[str]:
        """
        Get the critical path (longest execution path) through the DAG.

        Returns:
            List of node IDs representing the critical path
        """
        # Find the path with the longest total duration
        critical_path = []
        max_duration = 0

        # For each DAG, find its critical path
        for dag in self.dags.values():
            paths = self._find_all_paths(dag)
            for path in paths:
                duration = sum(
                    self.nodes[node_id].execution_duration or 0
                    for node_id in path
                    if node_id in self.nodes
                )
                if duration > max_duration:
                    max_duration = duration
                    critical_path = path

        return critical_path

    def _find_all_paths(self, dag: TaskDAG) -> List[List[str]]:
        """
        Find all paths through a DAG.

        Args:
            dag: TaskDAG to analyze

        Returns:
            List of paths (each path is a list of node IDs)
        """
        import networkx as nx

        paths = []
        graph = dag.graph

        # Find nodes with no predecessors (roots)
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]

        # Find nodes with no successors (leaves)
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        # Find all paths from roots to leaves
        for root in roots:
            for leaf in leaves:
                try:
                    all_simple_paths = nx.all_simple_paths(graph, root, leaf)
                    paths.extend(list(all_simple_paths))
                except nx.NetworkXNoPath:
                    continue

        return paths

    def print_summary(self) -> None:
        """Print a summary of the execution."""
        stats = self.calculate_statistics()

        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)

        summary = stats["summary"]
        print(f"\nTotal Nodes: {summary['total_nodes']}")
        print(f"Completed: {summary['completed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success Rate: {summary['success_rate']:.2%}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")

        print("\nModule Statistics:")
        for module, module_stats in stats["module_statistics"].items():
            if module_stats["count"] > 0:
                print(f"  {module}:")
                print(f"    Executions: {module_stats['count']}")
                print(f"    Avg Duration: {module_stats.get('avg_duration', 0):.3f}s")
                print(f"    Error Rate: {module_stats.get('error_rate', 0):.2%}")

        print("\nDepth Distribution:")
        for depth, depth_stats in sorted(stats["depth_statistics"].items()):
            print(f"  Depth {depth}: {depth_stats['count']} nodes")
            if depth_stats.get('forced_execute', 0) > 0:
                print(f"    (Force executed: {depth_stats['forced_execute']})")

        print("=" * 60)