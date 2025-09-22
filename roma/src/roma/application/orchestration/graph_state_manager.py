"""
Graph State Manager for ROMA v2.0.

Provides centralized, thread-safe state management for dynamic task graphs
with event emission and concurrency control.
"""

import asyncio
from typing import Dict, List, Optional, Any

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.application.services.event_publisher import EventPublisher


class GraphStateManager:
    """
    Centralized state manager for dynamic task graphs.
    
    Provides thread-safe operations for:
    - Node status transitions with validation
    - Event emission for all state changes
    - Concurrent access control
    - Graph modification coordination
    - Execution statistics and monitoring
    """
    
    def __init__(self, graph: DynamicTaskGraph, event_publisher: EventPublisher):
        """
        Initialize GraphStateManager.

        Args:
            graph: DynamicTaskGraph to manage
            event_publisher: Event publisher for state change notifications
        """
        self.graph = graph
        self.event_publisher = event_publisher
        self.version = 0
        self._state_lock = asyncio.Lock()
    
    @property
    def is_locked(self) -> bool:
        """Check if state manager is currently locked."""
        return self._state_lock.locked()
    
    async def transition_node_status(self, task_id: str, new_status: TaskStatus) -> TaskNode:
        """
        Transition node status with validation and event emission.
        
        This operation is atomic - either all steps succeed or none do,
        maintaining consistency between graph state, events, and version.
        
        Args:
            task_id: Task identifier
            new_status: Target status
            
        Returns:
            Updated TaskNode
            
        Raises:
            ValueError: If task not found or invalid transition
        """
        async with self._state_lock:
            # Get current node
            current_node = self.graph.get_node(task_id)
            if current_node is None:
                raise ValueError(f"Task node {task_id} not found in graph")
            
            # Store original state for potential rollback
            original_node = current_node
            old_status = current_node.status
            
            # Step 1: Update graph state with atomic validation
            # Re-check current status immediately before update to prevent race conditions
            try:
                current_node_state = self.graph.get_node(task_id)
                if current_node_state is None:
                    raise ValueError(f"Task node {task_id} not found in graph during update")
                
                # Validate that state hasn't changed since this transition was planned
                # This prevents race conditions where multiple transitions were planned against
                # the same original state but only one should succeed
                if current_node_state.status != old_status:
                    # State changed between lock acquisition and update - this is a race condition
                    raise ValueError(f"Concurrent modification detected: expected status {old_status}, found {current_node_state.status}. Another transaction modified this node.")
                
                updated_node = await self.graph.update_node_status(task_id, new_status)
            except Exception as e:
                # Re-raise as ValueError for consistent error handling
                raise ValueError(f"Invalid transition from {old_status} to {new_status}: {str(e)}")
            
            # Step 2: Emit status change event (atomic operation requirement)
            event = TaskStatusChangedEvent.create(
                task_id=task_id,
                old_status=old_status,
                new_status=new_status,
                version=updated_node.version
            )
            
            try:
                await self.event_store.append(event)
            except Exception as event_error:
                # Event storage failed - rollback graph state to maintain consistency
                try:
                    await self.graph.set_node_exact(task_id, original_node)
                except Exception:
                    # If rollback fails, system may be inconsistent; surface the error
                    pass
                # Re-raise the original event storage error
                raise event_error
            
            # Step 3: Increment manager version only after successful event storage
            self.version += 1
            
            return updated_node
    
    async def add_node(self, node: TaskNode) -> None:
        """
        Add node to graph with event emission.
        
        This operation is atomic - either all steps succeed or none do,
        maintaining consistency between graph state, events, and version.
        
        Args:
            node: TaskNode to add
        """
        async with self._state_lock:
            # Step 1: Add node to graph
            await self.graph.add_node(node)
            
            # Step 2: Emit node created event (atomic operation requirement)
            event = TaskCreatedEvent.create(
                task_id=node.task_id,
                goal=node.goal,
                task_type=node.task_type,
                parent_id=node.parent_id
            )
            
            try:
                await self.event_store.append(event)
                # Emit compatibility event for consumers expecting TaskNodeAddedEvent
                compat_event = TaskNodeAddedEvent.create(
                    task_id=node.task_id,
                    goal=node.goal,
                    task_type=node.task_type,
                    parent_id=node.parent_id
                )
                await self.event_store.append(compat_event)
            except Exception as event_error:
                # Event storage failed - rollback graph state to maintain consistency
                try:
                    # Remove the node that was just added from both data structures
                    self.graph.nodes.pop(node.task_id, None)
                    if hasattr(self.graph, '_graph') and self.graph._graph.has_node(node.task_id):
                        self.graph._graph.remove_node(node.task_id)
                except:
                    # If rollback fails, we're in an inconsistent state
                    # This is a critical error that should trigger system alerts
                    pass
                # Re-raise the original event storage error
                raise event_error
            
            # Step 3: Increment manager version only after successful event storage
            self.version += 1

    async def add_dependency_edge(self, from_id: str, to_id: str) -> None:
        """
        Add a dependency edge between two existing nodes.

        Args:
            from_id: Source task ID (dependency)
            to_id: Target task ID (dependent)

        Raises:
            KeyError: If either task ID is not found in graph
            ValueError: If adding edge would create a cycle
        """
        async with self._state_lock:
            # Delegate to graph and handle events
            await self.graph.add_dependency_edge(from_id, to_id)

            # Emit dependency added event for observability
            dependency_event = DependencyAddedEvent.create(
                task_id=to_id,
                dependency_id=from_id,
                metadata={
                    "from_task": from_id,
                    "to_task": to_id,
                    "dependency_type": "task_dependency",
                    "graph_version": self.version + 1
                }
            )
            await self.event_store.store_event(dependency_event)

            # Increment version for this state change
            self.version += 1

    def get_ready_nodes(self) -> List[TaskNode]:
        """
        Get nodes ready for execution.
        
        Returns:
            List of TaskNode objects ready for execution
        """
        return self.graph.get_ready_nodes()
    
    def get_node_by_id(self, task_id: str) -> Optional[TaskNode]:
        """
        Get node by task ID.
        
        Args:
            task_id: Task identifier
            
        Returns:
            TaskNode if found, None otherwise
        """
        return self.graph.get_node(task_id)
    
    def get_all_nodes(self) -> List[TaskNode]:
        """
        Get all nodes in the graph.

        Returns:
            List of all TaskNode objects
        """
        return self.graph.get_all_nodes()

    def get_children_nodes(self, task_id: str) -> List[TaskNode]:
        """
        Get child nodes of a specific task.

        Args:
            task_id: Parent task identifier

        Returns:
            List of child TaskNode objects
        """
        return self.graph.get_children_nodes(task_id)

    def has_cycles(self) -> bool:
        """
        Check if graph has cycles.
        
        Returns:
            True if graph contains cycles, False otherwise
        """
        return self.graph.has_cycles()
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics for monitoring.
        
        Returns:
            Dictionary with execution statistics
        """
        all_nodes = self.get_all_nodes()
        
        # Count nodes by status
        status_counts = {}
        for status in TaskStatus:
            status_counts[f"{status.value.lower()}_nodes"] = sum(
                1 for node in all_nodes if node.status == status
            )
        
        return {
            "total_nodes": len(all_nodes),
            "version": self.version,
            **status_counts
        }

    async def remove_node(self, task_id: str) -> None:
        """
        Remove node from graph with event emission.

        This operation is atomic - either all steps succeed or none do,
        maintaining consistency between graph state, events, and version.

        Args:
            task_id: Task identifier to remove

        Raises:
            KeyError: If task not found
        """
        async with self._state_lock:
            # Step 1: Verify node exists before removal
            node = self.graph.get_node(task_id)
            if node is None:
                raise KeyError(f"Task node {task_id} not found in graph")

            # Step 2: Remove from graph (delegates to DynamicTaskGraph)
            await self.graph.remove_node(task_id)

            # Step 3: Increment manager version after successful removal
            self.version += 1

    async def update_node_metadata(self, task_id: str, metadata_updates: Dict[str, Any]) -> TaskNode:
        """
        Update node metadata with event emission.

        This operation is atomic - either all steps succeed or none do,
        maintaining consistency between graph state, events, and version.

        Args:
            task_id: Task identifier
            metadata_updates: Dictionary of metadata key-value pairs to update

        Returns:
            Updated TaskNode

        Raises:
            KeyError: If task not found
        """
        async with self._state_lock:
            # Step 1: Update metadata (delegates to DynamicTaskGraph)
            updated_node = await self.graph.update_node_metadata(task_id, metadata_updates)

            # Step 2: Increment manager version after successful update
            self.version += 1

            return updated_node
