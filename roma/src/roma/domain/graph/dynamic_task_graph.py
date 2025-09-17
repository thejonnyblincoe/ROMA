"""
Dynamic Task Graph implementation using NetworkX as the backend.

This module provides a Pydantic-based dynamic DAG for hierarchical task execution
with thread-safe operations and NetworkX integration.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from uuid import uuid4
import networkx as nx
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.events.task_events import (
    BaseTaskEvent,
    TaskNodeAddedEvent,
    TaskStatusChangedEvent
)


def generate_execution_id() -> str:
    """Generate unique execution ID."""
    return str(uuid4())


class DynamicTaskGraph(BaseModel):
    """
    Dynamic Task Graph with NetworkX backend.
    
    Provides a thread-safe, immutable interface for managing hierarchical task
    execution with dynamic graph construction during runtime.
    
    Key Features:
    - NetworkX DiGraph backend for proven graph algorithms
    - Thread-safe concurrent operations with asyncio locks
    - Pydantic validation and serialization
    - Immutable TaskNode management
    - Dependency tracking and resolution
    """
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    # Public fields (serializable)
    execution_id: str = Field(default_factory=generate_execution_id, 
                             description="Unique execution identifier")
    root_node_id: Optional[str] = Field(default=None, 
                                       description="Root node task ID")
    nodes: Dict[str, TaskNode] = Field(default_factory=dict, 
                                      description="Task nodes by ID")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict,
                                              description="Execution metadata")
    
    # Private attributes (excluded from serialization)
    _graph: nx.DiGraph = PrivateAttr(default_factory=nx.DiGraph)  # type: ignore[type-arg]
    _lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _event_handlers: List[Callable[[BaseTaskEvent], Any]] = PrivateAttr(default_factory=list)
    
    def __init__(self, root_node: Optional[TaskNode] = None, **data: Any) -> None:
        """Initialize DynamicTaskGraph with optional root node."""
        # Add root node to data if provided
        if root_node:
            data.setdefault('root_node_id', root_node.task_id)
            nodes = data.get('nodes', {})
            nodes[root_node.task_id] = root_node
            data['nodes'] = nodes
        
        super().__init__(**data)
        
        # Initialize private attributes
        self._graph = nx.DiGraph()
        self._lock = asyncio.Lock()
        
        # Add nodes to NetworkX graph
        for node_id, node in self.nodes.items():
            self._graph.add_node(node_id)
            if node.parent_id and node.parent_id in self.nodes:
                self._graph.add_edge(node.parent_id, node_id)
    
    async def add_node(self, node: TaskNode) -> None:
        """
        Add a node to the graph with thread safety and event logging.
        
        Args:
            node: TaskNode to add to the graph
        """
        async with self._lock:
            # Add to nodes dict
            self.nodes[node.task_id] = node
            
            # Add to NetworkX graph
            self._graph.add_node(node.task_id)
            
            # Add edge if node has parent
            if node.parent_id and node.parent_id in self.nodes:
                self._graph.add_edge(node.parent_id, node.task_id)
            
            # Emit node added event
            await self._emit_event(TaskNodeAddedEvent.create(
                task_id=node.task_id,
                goal=node.goal,
                task_type=node.task_type,
                parent_id=node.parent_id,
                metadata={
                    "status": node.status.value
                }
            ))
    
    def get_node(self, task_id: str) -> Optional[TaskNode]:
        """
        Get a node by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskNode if found, None otherwise
        """
        return self.nodes.get(task_id)
    
    def get_all_nodes(self) -> List[TaskNode]:
        """
        Get all nodes in the graph.
        
        Returns:
            List of all TaskNode objects
        """
        return list(self.nodes.values())
    
    def get_ready_nodes(self) -> List[TaskNode]:
        """
        Get nodes that are ready for execution.
        
        A node is ready if:
        1. Status is PENDING
        2. All dependencies are completed (in-degree == 0 for pending nodes)
        
        Returns:
            List of TaskNode objects ready for execution
        """
        ready_nodes = []
        
        for node_id, node in self.nodes.items():
            if node.status == TaskStatus.PENDING:
                # Check if all dependencies are completed
                predecessors = list(self._graph.predecessors(node_id))
                
                if not predecessors:  # No dependencies
                    ready_nodes.append(node)
                else:
                    # Check if all predecessors are completed
                    all_completed = all(
                        self.nodes[pred_id].status == TaskStatus.COMPLETED 
                        for pred_id in predecessors
                        if pred_id in self.nodes
                    )
                    if all_completed:
                        ready_nodes.append(node)
        
        return ready_nodes
    
    async def update_node_status(self, task_id: str, new_status: TaskStatus) -> TaskNode:
        """
        Update node status with thread safety.
        
        Args:
            task_id: Task identifier
            new_status: New status for the task
            
        Returns:
            Updated TaskNode
            
        Raises:
            KeyError: If task_id not found in graph
        """
        async with self._lock:
            if task_id not in self.nodes:
                raise KeyError(f"Task ID {task_id} not found in graph")
            
            # Create updated node
            current_node = self.nodes[task_id]
            updated_node = current_node.transition_to(new_status)
            
            # Update nodes dict
            self.nodes[task_id] = updated_node
            
            # Emit status change event
            await self._emit_event(TaskStatusChangedEvent.create(
                task_id=task_id,
                old_status=current_node.status,
                new_status=new_status,
                version=updated_node.version,
                metadata={
                    "goal": updated_node.goal
                }
            ))
            
            return updated_node
    
    # Compatibility helper to support tests and callers needing node objects
    def get_children_nodes(self, task_id: str) -> List[TaskNode]:
        """
        Get child nodes of a given task as TaskNode objects.

        Args:
            task_id: Parent task identifier

        Returns:
            List of child TaskNode objects
        """
        if task_id not in self._graph:
            return []

        child_ids = list(self._graph.successors(task_id))
        return [self.nodes[child_id] for child_id in child_ids if child_id in self.nodes]
    
    def has_cycles(self) -> bool:
        """
        Check if graph has cycles using NetworkX.
        
        Returns:
            True if graph contains cycles, False otherwise
        """
        try:
            # NetworkX will raise NetworkXError if cycles exist
            list(nx.topological_sort(self._graph))
            return False
        except nx.NetworkXUnfeasible:
            return True
    
    def get_topological_order(self) -> List[str]:
        """
        Get topological ordering of nodes using NetworkX.
        
        Returns:
            List of task IDs in topological order
            
        Raises:
            NetworkXError: If graph contains cycles
        """
        return list(nx.topological_sort(self._graph))
    
    # Node Relationship Methods (Task 1.2.3)
    
    def get_children(self, task_id: str) -> List[str]:
        """
        Get all direct children of a node as task ID strings.

        Backward-compatibility: returns a list of string-like values that also
        expose a `.task_id` attribute to satisfy tests that access `child.task_id`.

        Args:
            task_id: ID of the parent node

        Returns:
            List of child task IDs. If the parent does not exist, returns an empty list.
        """
        if task_id not in self.nodes and task_id not in self._graph:
            # For consistency with tests expecting empty list on missing parent
            return []

        class _TaskIdStr(str):
            @property
            def task_id(self) -> str:
                return str(self)

        # Use NetworkX successors for direct children
        raw_ids = list(self._graph.successors(task_id))
        return [_TaskIdStr(cid) for cid in raw_ids]
    
    def get_ancestors(self, task_id: str) -> List[str]:
        """
        Get all ancestors (parents, grandparents, etc.) of a node.
        
        Args:
            task_id: ID of the descendant node
            
        Returns:
            List of ancestor task IDs in order (immediate parent first)
            
        Raises:
            KeyError: If task_id not found
        """
        if task_id not in self.nodes:
            raise KeyError(f"Task ID {task_id} not found in graph")
        
        ancestors = []
        current = task_id
        
        # Walk up the parent chain
        while current is not None:
            predecessors = list(self._graph.predecessors(current))
            if not predecessors:
                break
            # Assume single parent (tree structure)
            parent = predecessors[0]
            ancestors.append(parent)
            current = parent
        
        return ancestors
    
    def get_siblings(self, task_id: str) -> List[str]:
        """
        Get all siblings (nodes with same parent) of a node.
        
        Args:
            task_id: ID of the node
            
        Returns:
            List of sibling task IDs (excluding the node itself)
            
        Raises:
            KeyError: If task_id not found
        """
        if task_id not in self.nodes:
            raise KeyError(f"Task ID {task_id} not found in graph")
        
        # Get parent node
        predecessors = list(self._graph.predecessors(task_id))
        if not predecessors:
            # Root node has no siblings
            return []
        
        parent_id = predecessors[0]  # Assume single parent
        siblings = self.get_children(parent_id)
        
        # Remove self from siblings
        return [sibling_id for sibling_id in siblings if sibling_id != task_id]
    
    def get_descendants(self, task_id: str) -> List[str]:
        """
        Get all descendants (children, grandchildren, etc.) of a node.
        
        Args:
            task_id: ID of the ancestor node
            
        Returns:
            List of descendant task IDs in breadth-first order
            
        Raises:
            KeyError: If task_id not found
        """
        if task_id not in self.nodes:
            raise KeyError(f"Task ID {task_id} not found in graph")
        
        # Use NetworkX descendants for all reachable nodes
        descendants = list(nx.descendants(self._graph, task_id))
        
        # Sort by depth (breadth-first ordering)
        depths = nx.single_source_shortest_path_length(self._graph, task_id)
        descendants.sort(key=lambda node_id: depths.get(node_id, float('inf')))
        
        return descendants
    
    def get_subtree(self, task_id: str) -> Dict[str, TaskNode]:
        """
        Get the complete subtree rooted at the given node.
        
        Args:
            task_id: ID of the root node of subtree
            
        Returns:
            Dictionary of {task_id: TaskNode} for subtree
            
        Raises:
            KeyError: If task_id not found
        """
        if task_id not in self.nodes:
            raise KeyError(f"Task ID {task_id} not found in graph")
        
        subtree_ids = [task_id] + self.get_descendants(task_id)
        return {node_id: self.nodes[node_id] for node_id in subtree_ids}
    
    # Event Handling Methods (Task 1.2.5)
    
    def add_event_handler(self, handler: Callable[[BaseTaskEvent], Any]) -> None:
        """
        Add event handler for graph operations.
        
        Args:
            handler: Async callable that receives TaskEvent objects
        """
        self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: Callable[[BaseTaskEvent], Any]) -> None:
        """
        Remove event handler.
        
        Args:
            handler: Handler to remove
        """
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
    
    async def _emit_event(self, event: BaseTaskEvent) -> None:
        """
        Emit event to all registered handlers.
        
        Args:
            event: Event to emit
        """
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                # Log error but don't fail graph operations
                print(f"Event handler error: {e}")

    async def set_node_exact(self, task_id: str, node: TaskNode) -> None:
        """Replace a node in-place without altering its version or timestamps.

        This is intended for rollback scenarios where we must restore the exact
        prior instance atomically.
        """
        async with self._lock:
            if task_id not in self.nodes:
                raise KeyError(f"Task ID {task_id} not found in graph")
            self.nodes[task_id] = node
    
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to rebuild NetworkX graph from nodes."""
        if hasattr(self, '_graph'):
            return  # Already initialized
        
        # Initialize private attributes for deserialization
        self._graph = nx.DiGraph()
        self._lock = asyncio.Lock()
        
        # Rebuild NetworkX graph from nodes
        for node_id, node in self.nodes.items():
            self._graph.add_node(node_id)
            if node.parent_id and node.parent_id in self.nodes:
                self._graph.add_edge(node.parent_id, node_id)
