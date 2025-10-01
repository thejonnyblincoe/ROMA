"""
NetworkX-based DAG implementation for task dependency management and execution.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import networkx as nx
from datetime import datetime
from uuid import uuid4

from ..signatures.base_models.task_node import TaskNode
from ...types import TaskStatus, NodeType


class TaskDAG:
    """
    Manages task dependencies as a directed acyclic graph using NetworkX.

    Features:
    - Topological sorting for dependency resolution
    - Nested subgraphs for hierarchical decomposition
    - Parallel execution tracking
    - Comprehensive state management
    """

    def __init__(self, dag_id: Optional[str] = None, parent_dag: Optional['TaskDAG'] = None, execution_id: Optional[str] = None):
        """
        Initialize a new TaskDAG.

        Args:
            dag_id: Unique identifier for this DAG
            parent_dag: Parent DAG if this is a subgraph
            execution_id: Unique identifier for this execution run
        """
        self.dag_id = dag_id or str(uuid4())
        self.execution_id = execution_id or str(uuid4())
        self.graph = nx.DiGraph()
        self.parent_dag = parent_dag
        self.subgraphs: Dict[str, 'TaskDAG'] = {}
        self.metadata = {
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'execution_started_at': None,
            'execution_completed_at': None,
            'execution_id': self.execution_id
        }

    def add_node(
        self,
        task: TaskNode,
        parent_id: Optional[str] = None
    ) -> TaskNode:
        """
        Add a task node to the DAG.

        Args:
            task: TaskNode to add
            parent_id: Optional parent task ID

        Returns:
            Updated TaskNode with depth calculated
        """
        # Calculate depth if parent exists
        if parent_id and parent_id in self.graph:
            parent_task = self.get_node(parent_id)
            task = task.with_incremented_depth(parent_task.depth)

            # Check if max depth reached - force execution
            if task.should_force_execute():
                task = task.set_node_type(NodeType.EXECUTE)
                task = task.transition_to(TaskStatus.READY)

        # Add node to graph with task data
        self.graph.add_node(
            task.task_id,
            task=task,
            added_at=datetime.now()
        )

        # Add edge from parent if specified
        if parent_id:
            self.add_edge(parent_id, task.task_id)

        self.metadata['updated_at'] = datetime.now()
        return task

    def add_edge(
        self,
        from_task_id: str,
        to_task_id: str,
        edge_type: str = "dependency"
    ) -> None:
        """
        Add a dependency edge between tasks.

        Args:
            from_task_id: Source task ID
            to_task_id: Target task ID
            edge_type: Type of edge (dependency, parent-child)
        """
        if from_task_id not in self.graph:
            raise ValueError(f"Source task {from_task_id} not in DAG")
        if to_task_id not in self.graph:
            raise ValueError(f"Target task {to_task_id} not in DAG")

        self.graph.add_edge(
            from_task_id,
            to_task_id,
            edge_type=edge_type,
            created_at=datetime.now()
        )

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(from_task_id, to_task_id)
            raise ValueError(f"Adding edge from {from_task_id} to {to_task_id} would create a cycle")

        self.metadata['updated_at'] = datetime.now()

    def add_dependencies(
        self,
        task_id: str,
        dependency_ids: List[str]
    ) -> TaskNode:
        """
        Add multiple dependencies for a task.

        Args:
            task_id: Task that depends on others
            dependency_ids: List of dependency task IDs

        Returns:
            Updated TaskNode
        """
        if task_id not in self.graph:
            raise ValueError(f"Task {task_id} not in DAG")

        task = self.get_node(task_id)

        for dep_id in dependency_ids:
            if dep_id not in self.graph:
                raise ValueError(f"Dependency {dep_id} not in DAG")

            # Add edge from dependency to task (dep must complete before task)
            self.add_edge(dep_id, task_id, edge_type="dependency")

            # Update task dependencies
            task = task.add_dependency(dep_id)

        # Update node in graph
        self.graph.nodes[task_id]['task'] = task
        return task

    def get_node(self, task_id: str) -> TaskNode:
        """
        Get a task node by ID.

        Args:
            task_id: Task ID to retrieve

        Returns:
            TaskNode instance
        """
        if task_id not in self.graph:
            raise ValueError(f"Task {task_id} not in DAG")
        return self.graph.nodes[task_id]['task']

    def update_node(self, task: TaskNode) -> None:
        """
        Update a task node in the DAG.

        Args:
            task: Updated TaskNode
        """
        if task.task_id not in self.graph:
            raise ValueError(f"Task {task.task_id} not in DAG")

        self.graph.nodes[task.task_id]['task'] = task
        self.graph.nodes[task.task_id]['updated_at'] = datetime.now()
        self.metadata['updated_at'] = datetime.now()

    def get_ready_tasks(self, include_subgraphs: bool = False) -> List[TaskNode]:
        """
        Get tasks that can be processed immediately (dependencies satisfied).

        Args:
            include_subgraphs: Whether to include ready tasks from nested subgraphs

        Returns:
            List of TaskNode instances ready for execution
        """
        ready_tasks: List[TaskNode] = []

        for node_id in self.graph.nodes():
            task = self.graph.nodes[node_id]['task']

            if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                continue

            if self._dependencies_satisfied(node_id):
                ready_tasks.append(task)

        if include_subgraphs:
            for subgraph in self.subgraphs.values():
                ready_tasks.extend(subgraph.get_ready_tasks(include_subgraphs=True))

        return ready_tasks

    def iter_ready_nodes(self) -> List[Tuple[TaskNode, 'TaskDAG']]:
        """Return ready tasks paired with the DAG that owns them."""

        ready_pairs: List[Tuple[TaskNode, 'TaskDAG']] = []

        for node_id in self.graph.nodes():
            task = self.graph.nodes[node_id]['task']
            if task.status not in {TaskStatus.PENDING, TaskStatus.READY}:
                continue
            if self._dependencies_satisfied(node_id):
                ready_pairs.append((task, self))

        for subgraph in self.subgraphs.values():
            ready_pairs.extend(subgraph.iter_ready_nodes())

        return ready_pairs

    def _dependencies_satisfied(self, node_id: str) -> bool:
        """Check whether all predecessors for a node have completed."""

        for pred_id in self.graph.predecessors(node_id):
            pred_task = self.graph.nodes[pred_id]['task']
            if pred_task.status != TaskStatus.COMPLETED:
                return False
        return True

    def get_execution_order(self) -> List[str]:
        """
        Get topological sort of task IDs for execution order.

        Returns:
            List of task IDs in execution order
        """
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            raise ValueError(f"Cannot determine execution order: {e}")

    def create_subgraph(
        self,
        parent_task_id: str,
        subtasks: List[TaskNode],
        dependencies: Optional[Dict[str, List[str]]] = None
    ) -> 'TaskDAG':
        """
        Create a subgraph for a planning node's subtasks.

        Args:
            parent_task_id: ID of the parent planning task
            subtasks: List of subtasks to add to subgraph
            dependencies: Optional dependency mapping {task_id: [dependency_ids]}

        Returns:
            New TaskDAG instance for the subgraph
        """
        subgraph_id = f"{self.dag_id}_sub_{parent_task_id}"
        subgraph = TaskDAG(dag_id=subgraph_id, parent_dag=self, execution_id=self.execution_id)

        # Get parent task for depth calculation
        parent_task = self.get_node(parent_task_id)

        # Add all subtasks to subgraph
        for subtask in subtasks:
            # Set parent_id and calculate depth
            subtask = subtask.model_copy(update={'parent_id': parent_task_id})
            subtask = subgraph.add_node(subtask)

        # Add dependencies if provided
        if dependencies:
            for task_id, dep_ids in dependencies.items():
                if task_id in [t.task_id for t in subtasks]:
                    subgraph.add_dependencies(task_id, dep_ids)

        # Store subgraph reference
        self.subgraphs[subgraph_id] = subgraph

        # Update parent task with subgraph reference
        parent_task = parent_task.set_subgraph(subgraph_id)
        self.update_node(parent_task)

        return subgraph

    def get_subgraph(self, subgraph_id: str) -> Optional['TaskDAG']:
        """
        Get a subgraph by ID.

        Args:
            subgraph_id: Subgraph ID

        Returns:
            TaskDAG instance or None if not found
        """
        return self.subgraphs.get(subgraph_id)

    def get_all_tasks(self, include_subgraphs: bool = True) -> List[TaskNode]:
        """
        Get all tasks in the DAG.

        Args:
            include_subgraphs: Whether to include tasks from subgraphs

        Returns:
            List of all TaskNode instances
        """
        tasks = [self.get_node(node_id) for node_id in self.graph.nodes()]

        if include_subgraphs:
            for subgraph in self.subgraphs.values():
                tasks.extend(subgraph.get_all_tasks(include_subgraphs=True))

        return tasks

    def get_task_dependencies(self, task_id: str) -> List[TaskNode]:
        """
        Get all dependencies for a task.

        Args:
            task_id: Task ID

        Returns:
            List of dependency TaskNode instances
        """
        if task_id not in self.graph:
            raise ValueError(f"Task {task_id} not in DAG")

        return [self.get_node(pred_id) for pred_id in self.graph.predecessors(task_id)]

    def get_task_children(self, task_id: str) -> List[TaskNode]:
        """
        Get all children of a task.

        Args:
            task_id: Task ID

        Returns:
            List of child TaskNode instances
        """
        if task_id not in self.graph:
            raise ValueError(f"Task {task_id} not in DAG")

        return [self.get_node(succ_id) for succ_id in self.graph.successors(task_id)]

    def is_dag_complete(self) -> bool:
        """
        Check if all tasks in the DAG are complete.

        Returns:
            True if all tasks are in terminal state
        """
        for node_id in self.graph.nodes():
            task = self.get_node(node_id)
            if not task.status.is_terminal:
                return False

        # Check subgraphs
        for subgraph in self.subgraphs.values():
            if not subgraph.is_dag_complete():
                return False

        return True

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get DAG execution statistics.

        Returns:
            Dictionary with various statistics
        """
        all_tasks = self.get_all_tasks(include_subgraphs=True)

        status_counts = {}
        for task in all_tasks:
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        depth_distribution = {}
        for task in all_tasks:
            depth = task.depth
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1

        return {
            'dag_id': self.dag_id,
            'total_tasks': len(all_tasks),
            'status_counts': status_counts,
            'depth_distribution': depth_distribution,
            'num_subgraphs': len(self.subgraphs),
            'is_complete': self.is_dag_complete(),
            'metadata': self.metadata
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export DAG structure to dictionary format.

        Returns:
            Dictionary representation of the DAG
        """
        nodes = []
        edges = []

        for node_id in self.graph.nodes():
            task = self.get_node(node_id)
            nodes.append({
                'id': task.task_id,
                'goal': task.goal,
                'status': task.status.value,
                'depth': task.depth,
                'node_type': task.node_type.value if task.node_type else None,
                'execution_history': list(task.execution_history.keys())
            })

        for from_id, to_id, edge_data in self.graph.edges(data=True):
            edges.append({
                'from': from_id,
                'to': to_id,
                'type': edge_data.get('edge_type', 'dependency')
            })

        result = {
            'dag_id': self.dag_id,
            'nodes': nodes,
            'edges': edges,
            'statistics': self.get_statistics()
        }

        # Include subgraphs
        if self.subgraphs:
            result['subgraphs'] = {
                sub_id: subgraph.export_to_dict()
                for sub_id, subgraph in self.subgraphs.items()
            }

        return result

    # ------------------------------------------------------------------
    # Scheduling helpers (async wrappers to support event scheduler)
    # ------------------------------------------------------------------

    async def pop_ready_tasks(self) -> List[TaskNode]:
        """Async wrapper returning tasks whose dependencies are satisfied."""

        return [node for node, _ in self.iter_ready_nodes()]

    async def mark_completed(self, task_id: str, result: Optional[Any] = None) -> TaskNode:
        """Mark a task as completed and persist the update in the DAG."""

        task = self.get_node(task_id)

        updated = task
        if result is not None:
            updated = task.with_result(result)
        else:
            updated = task.transition_to(TaskStatus.COMPLETED)

        self.update_node(updated)
        return updated

    async def mark_failed(self, task_id: str, error: Optional[Any] = None) -> TaskNode:
        """Mark a task as failed; metadata can carry error context."""

        task = self.get_node(task_id)
        metadata = {"error": error} if error is not None else {}
        updated = task.transition_to(TaskStatus.FAILED, transition_metadata=metadata)
        self.update_node(updated)
        return updated

    async def check_subgraph_complete(
        self,
        task_id: str
    ) -> Optional[Tuple[TaskNode, 'TaskDAG']]:
        """Return parent task and owning DAG if its subgraph has completed."""

        task = self.get_node(task_id)
        if not task.parent_id:
            return None

        parent_dag = self.parent_dag or self
        parent_node = parent_dag.get_node(task.parent_id)

        if not parent_node.subgraph_id:
            return None

        subgraph = parent_dag.get_subgraph(parent_node.subgraph_id)
        if not subgraph:
            return None

        if subgraph.is_dag_complete():
            return parent_node, parent_dag

        return None

    def find_dag(self, dag_id: str) -> Optional['TaskDAG']:
        """Locate a DAG (self or subgraph) by ID."""

        if self.dag_id == dag_id:
            return self

        for subgraph in self.subgraphs.values():
            located = subgraph.find_dag(dag_id)
            if located is not None:
                return located

        return None

    def find_node(self, task_id: str) -> Tuple[TaskNode, 'TaskDAG']:
        """Locate a task within this DAG hierarchy and return owning DAG."""

        if task_id in self.graph:
            return self.get_node(task_id), self

        for subgraph in self.subgraphs.values():
            try:
                return subgraph.find_node(task_id)
            except ValueError:
                continue

        raise ValueError(f"Task {task_id} not found in DAG hierarchy")

    def get_execution_id(self) -> str:
        """
        Get the execution ID for this DAG.

        Returns:
            Execution ID string
        """
        return self.execution_id
