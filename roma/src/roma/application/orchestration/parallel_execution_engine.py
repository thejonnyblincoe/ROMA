"""
Parallel Execution Engine for ROMA v2.0.

Implements modified Kahn's algorithm for hierarchical task execution with:
- Dynamic graph expansion during execution
- Semaphore-controlled parallel processing 
- Event-driven state management
- Graceful error handling and recovery
"""

import asyncio
from typing import List, Dict, Any, Optional, Protocol, Set
from dataclasses import dataclass
from datetime import datetime, timezone

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.application.orchestration.graph_state_manager import GraphStateManager


class TaskExecutor(Protocol):
    """Protocol for task execution implementations."""
    
    async def execute_task(self, node: TaskNode) -> Dict[str, Any]:
        """
        Execute a single task node.
        
        Args:
            node: TaskNode to execute
            
        Returns:
            Dictionary with execution results
        """
        ...


@dataclass
class ExecutionResult:
    """Result of graph execution."""
    
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    success: bool
    execution_time_seconds: float
    error_details: Optional[List[Dict[str, Any]]] = None


class ParallelExecutionEngine:
    """
    Parallel execution engine using modified Kahn's algorithm.
    
    Key Features:
    - Modified Kahn's algorithm for DAG traversal
    - Semaphore-controlled concurrency
    - Dynamic node addition during execution
    - Real-time state management through GraphStateManager
    - Comprehensive error handling and recovery
    
    Modified Kahn's Algorithm Flow:
    1. Initialize in-degree counts for all nodes
    2. Identify ready nodes (in-degree = 0, status = PENDING)
    3. Execute ready nodes in parallel (respecting concurrency limit)
    4. Update dependencies and repeat until completion or deadlock
    5. Handle dynamic node addition during execution
    """
    
    def __init__(
        self,
        state_manager: GraphStateManager,
        task_executor: TaskExecutor,
        max_concurrent_tasks: int = 10
    ):
        """
        Initialize ParallelExecutionEngine.
        
        Args:
            state_manager: GraphStateManager for state coordination
            task_executor: TaskExecutor implementation for task execution
            max_concurrent_tasks: Maximum number of concurrent task executions
        """
        self.state_manager = state_manager
        self.task_executor = task_executor
        self.max_concurrent_tasks = max_concurrent_tasks
        self.is_running = False
        
        # Semaphore to control concurrency
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Execution statistics
        self._total_nodes_processed = 0
        self._completed_nodes = 0
        self._failed_nodes = 0
        self._error_details: List[Dict[str, Any]] = []
        # Track task IDs in case state transitions fail but execution outcome is known
        self._completed_task_ids: Set[str] = set()
        self._failed_task_ids: Set[str] = set()
    
    async def execute_graph(self) -> ExecutionResult:
        """
        Execute the entire task graph using modified Kahn's algorithm.
        
        Returns:
            ExecutionResult with execution statistics
        """
        if self.is_running:
            raise RuntimeError("Execution engine is already running")
        
        self.is_running = True
        start_time = datetime.now(timezone.utc)
        
        try:
            # Reset statistics
            self._total_nodes_processed = 0
            self._completed_nodes = 0
            self._failed_nodes = 0
            self._error_details = []
            self._completed_task_ids.clear()
            self._failed_task_ids.clear()

            # If graph is empty at start, wait briefly for dynamic additions
            if not self.state_manager.get_all_nodes():
                for _ in range(50):  # ~1s total
                    await asyncio.sleep(0.02)
                    if self.state_manager.get_all_nodes():
                        break
            
            # Main execution loop (modified Kahn's algorithm)
            while not self._is_execution_complete():
                # Get batch of ready nodes
                ready_batch = self._get_ready_nodes_batch()
                
                if not ready_batch:
                    # No ready nodes but execution not complete - possible deadlock
                    break
                
                # Execute ready nodes in parallel
                await self._execute_batch(ready_batch)
            
            # Calculate final results
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()
            
            all_nodes = self.state_manager.get_all_nodes()
            total_nodes = len(all_nodes)
            
            # Count final statistics from actual graph state
            completed_count_graph = sum(1 for node in all_nodes if node.status == TaskStatus.COMPLETED)
            failed_count_graph = sum(1 for node in all_nodes if node.status == TaskStatus.FAILED)

            # Reconcile with local tracking in case state transitions failed
            completed_count = max(completed_count_graph, len(self._completed_task_ids))
            failed_count = max(failed_count_graph, len(self._failed_task_ids))
            
            success = failed_count == 0 and total_nodes == completed_count
            
            return ExecutionResult(
                total_nodes=total_nodes,
                completed_nodes=completed_count,
                failed_nodes=failed_count,
                success=success,
                execution_time_seconds=execution_time,
                error_details=self._error_details if self._error_details else None
            )
        
        finally:
            self.is_running = False
    
    def _get_ready_nodes_batch(self) -> List[TaskNode]:
        """
        Get batch of nodes ready for execution.
        
        A node is ready if:
        1. Status is PENDING
        2. All dependencies are COMPLETED
        
        Returns:
            List of TaskNode objects ready for execution
        """
        return self.state_manager.get_ready_nodes()
    
    def _is_execution_complete(self) -> bool:
        """
        Check if graph execution is complete.
        
        Execution is complete when all nodes are in terminal states
        (COMPLETED or FAILED) or no more progress can be made.
        
        Returns:
            True if execution is complete, False otherwise
        """
        all_nodes = self.state_manager.get_all_nodes()
        
        if not all_nodes:
            return True  # Empty graph is complete
        
        # Check if all nodes are in terminal states
        terminal_statuses = {TaskStatus.COMPLETED, TaskStatus.FAILED}
        non_terminal_nodes = [
            node for node in all_nodes 
            if node.status not in terminal_statuses
        ]
        
        return len(non_terminal_nodes) == 0
    
    async def _execute_batch(self, batch: List[TaskNode]) -> None:
        """
        Execute a batch of ready nodes in parallel.
        
        Args:
            batch: List of TaskNode objects to execute
        """
        if not batch:
            return
        
        # Create execution tasks with semaphore control
        execution_tasks = [
            self._execute_single_node(node)
            for node in batch
        ]
        
        # Execute all tasks concurrently
        await asyncio.gather(*execution_tasks, return_exceptions=True)
    
    async def _execute_single_node(self, node: TaskNode) -> None:
        """
        Execute a single node with concurrency control.
        
        Args:
            node: TaskNode to execute
        """
        async with self._execution_semaphore:
            try:
                # Transition to READY if still PENDING
                if node.status == TaskStatus.PENDING:
                    await self.state_manager.transition_node_status(
                        node.task_id, TaskStatus.READY
                    )
                
                # Transition to EXECUTING
                await self.state_manager.transition_node_status(
                    node.task_id, TaskStatus.EXECUTING
                )
                
                # Execute the actual task
                execution_result = await self.task_executor.execute_task(node)
                
                # Determine final status based on execution result
                if execution_result.get("status") == "completed":
                    final_status = TaskStatus.COMPLETED
                    self._completed_nodes += 1
                    self._completed_task_ids.add(node.task_id)
                else:
                    final_status = TaskStatus.FAILED
                    self._failed_nodes += 1
                    self._failed_task_ids.add(node.task_id)
                
                # Transition to final status
                await self.state_manager.transition_node_status(
                    node.task_id, final_status
                )
                
                self._total_nodes_processed += 1
                
            except Exception as e:
                # Handle execution errors
                self._failed_nodes += 1
                self._total_nodes_processed += 1
                self._failed_task_ids.add(node.task_id)
                base_error_details = {
                    "error": str(e),
                    "task_id": node.task_id,
                    "goal": node.goal,
                    "phase": "execute_task",
                }
                
                # Attempt state transition but handle failure gracefully
                try:
                    await self.state_manager.transition_node_status(
                        node.task_id, TaskStatus.FAILED
                    )
                except Exception as state_error:
                    # Capture both execution error and transition failure
                    error_details = {
                        **base_error_details,
                        "state_transition_error": str(state_error),
                        "phase": "state_transition_failed",
                    }
                    self._error_details.append(error_details)
                else:
                    # Transition to FAILED succeeded; record original error
                    self._error_details.append(base_error_details)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get current execution statistics.
        
        Returns:
            Dictionary with execution statistics
        """
        return {
            "total_nodes_processed": self._total_nodes_processed,
            "completed_nodes": self._completed_nodes,
            "failed_nodes": self._failed_nodes,
            "is_running": self.is_running,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "current_graph_stats": self.state_manager.get_execution_statistics()
        }
    
    async def stop_execution(self) -> None:
        """
        Stop the execution engine gracefully.
        
        This will allow currently running tasks to complete but prevent
        new tasks from starting.
        """
        self.is_running = False
        
        # Wait for all current executions to complete
        # This is done by waiting for all semaphore slots to be available
        for _ in range(self.max_concurrent_tasks):
            await self._execution_semaphore.acquire()
        
        # Release all acquired slots
        for _ in range(self.max_concurrent_tasks):
            self._execution_semaphore.release()
    
    def is_deadlocked(self) -> bool:
        """
        Check if execution is in a deadlock state.
        
        Deadlock occurs when there are non-terminal nodes but no ready nodes.
        
        Returns:
            True if deadlocked, False otherwise
        """
        if self._is_execution_complete():
            return False
        
        ready_nodes = self._get_ready_nodes_batch()
        return len(ready_nodes) == 0
