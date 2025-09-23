"""
DAG Executor for parallel task execution with dependency management.
"""

from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from src.roma_dspy.engine.dag import TaskDAG
from src.roma_dspy.engine.solve import RecursiveSolver
from src.roma_dspy.signatures.base_models.task_node import TaskNode
from src.roma_dspy.types.task_status import TaskStatus


class DAGExecutor:
    """
    Executes tasks in a DAG with parallel processing and dependency management.

    Features:
    - Parallel execution of independent tasks
    - Dependency tracking and resolution
    - Real-time status updates
    - Failure handling and retry logic
    - Progress monitoring
    """

    def __init__(
        self,
        solver: Optional[RecursiveSolver] = None,
        max_workers: int = 4,
        enable_async: bool = True
    ):
        """
        Initialize DAG executor.

        Args:
            solver: RecursiveSolver instance to use
            max_workers: Maximum number of parallel workers
            enable_async: Whether to use async execution
        """
        self.solver = solver or RecursiveSolver()
        self.max_workers = max_workers
        self.enable_async = enable_async
        self.execution_lock = threading.Lock()
        self.status_callbacks: List[Callable] = []

    def add_status_callback(self, callback: Callable[[TaskNode], None]) -> None:
        """
        Add a callback for task status updates.

        Args:
            callback: Function to call on status changes
        """
        self.status_callbacks.append(callback)

    def _notify_status_change(self, task: TaskNode) -> None:
        """Notify all callbacks of a status change."""
        for callback in self.status_callbacks:
            try:
                callback(task)
            except Exception as e:
                print(f"Error in status callback: {e}")

    def execute_dag(self, dag: TaskDAG) -> Dict[str, Any]:
        """
        Synchronously execute all tasks in the DAG.

        Args:
            dag: TaskDAG to execute

        Returns:
            Execution results and statistics
        """
        if self.enable_async:
            # Run async version in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.async_execute_dag(dag))
            finally:
                loop.close()
        else:
            return self._sync_execute_dag(dag)

    def _sync_execute_dag(self, dag: TaskDAG) -> Dict[str, Any]:
        """
        Execute DAG using thread pool for parallelism.

        Args:
            dag: TaskDAG to execute

        Returns:
            Execution results
        """
        start_time = datetime.now()
        dag.metadata['execution_started_at'] = start_time

        completed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()
        results: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Continue until all tasks are processed
            while True:
                # Find ready tasks
                ready_tasks = self._find_ready_tasks(dag, completed_tasks, failed_tasks)

                if not ready_tasks and not self._has_active_tasks(dag):
                    # No more tasks to execute
                    break

                # Submit ready tasks to executor
                future_to_task = {}
                for task in ready_tasks:
                    # Update status to READY
                    task = task.transition_to(TaskStatus.READY)
                    dag.update_node(task)
                    self._notify_status_change(task)

                    # Submit for execution
                    future = executor.submit(self._execute_task, task, dag)
                    future_to_task[future] = task

                # Wait for tasks to complete
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        result_task = future.result()
                        dag.update_node(result_task)
                        completed_tasks.add(result_task.task_id)
                        results[result_task.task_id] = result_task.result
                        self._notify_status_change(result_task)
                    except Exception as e:
                        # Task failed
                        task = task.transition_to(
                            TaskStatus.FAILED,
                            error=str(e)
                        )
                        dag.update_node(task)
                        failed_tasks.add(task.task_id)
                        results[task.task_id] = {"error": str(e)}
                        self._notify_status_change(task)

        # Update DAG metadata
        end_time = datetime.now()
        dag.metadata['execution_completed_at'] = end_time
        dag.metadata['execution_duration'] = (end_time - start_time).total_seconds()

        return {
            'results': results,
            'statistics': dag.get_statistics(),
            'completed_count': len(completed_tasks),
            'failed_count': len(failed_tasks),
            'total_duration': dag.metadata['execution_duration']
        }

    async def async_execute_dag(self, dag: TaskDAG) -> Dict[str, Any]:
        """
        Asynchronously execute all tasks in the DAG with maximum parallelism.

        Args:
            dag: TaskDAG to execute

        Returns:
            Execution results and statistics
        """
        start_time = datetime.now()
        dag.metadata['execution_started_at'] = start_time

        completed_tasks: Set[str] = set()
        failed_tasks: Set[str] = set()
        results: Dict[str, Any] = {}

        # Continue until all tasks are processed
        while True:
            # Find ready tasks
            ready_tasks = self._find_ready_tasks(dag, completed_tasks, failed_tasks)

            if not ready_tasks:
                # Check if we have active tasks
                if not self._has_active_tasks(dag):
                    break

                # Wait a bit for active tasks to complete
                await asyncio.sleep(0.1)
                continue

            # Execute ready tasks in parallel
            async_tasks = []
            for task in ready_tasks:
                # Update status to READY
                task = task.transition_to(TaskStatus.READY)
                dag.update_node(task)
                self._notify_status_change(task)

                # Create async task
                async_tasks.append(self._async_execute_task(task, dag))

            # Wait for all to complete
            task_results = await asyncio.gather(*async_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(task_results):
                task = ready_tasks[i]

                if isinstance(result, Exception):
                    # Task failed
                    task = task.transition_to(
                        TaskStatus.FAILED,
                        error=str(result)
                    )
                    dag.update_node(task)
                    failed_tasks.add(task.task_id)
                    results[task.task_id] = {"error": str(result)}
                    self._notify_status_change(task)
                else:
                    # Task succeeded
                    dag.update_node(result)
                    completed_tasks.add(result.task_id)
                    results[result.task_id] = result.result
                    self._notify_status_change(result)

        # Update DAG metadata
        end_time = datetime.now()
        dag.metadata['execution_completed_at'] = end_time
        dag.metadata['execution_duration'] = (end_time - start_time).total_seconds()

        return {
            'results': results,
            'statistics': dag.get_statistics(),
            'completed_count': len(completed_tasks),
            'failed_count': len(failed_tasks),
            'total_duration': dag.metadata['execution_duration']
        }

    def _find_ready_tasks(
        self,
        dag: TaskDAG,
        completed: Set[str],
        failed: Set[str]
    ) -> List[TaskNode]:
        """
        Find all tasks ready for execution.

        Args:
            dag: TaskDAG to search
            completed: Set of completed task IDs
            failed: Set of failed task IDs

        Returns:
            List of ready TaskNodes
        """
        ready = []

        for node_id in dag.graph.nodes():
            # Skip if already processed
            if node_id in completed or node_id in failed:
                continue

            task = dag.get_node(node_id)

            # Skip if already active or completed
            if task.status in [
                TaskStatus.EXECUTING,
                TaskStatus.ATOMIZING,
                TaskStatus.PLANNING,
                TaskStatus.AGGREGATING,
                TaskStatus.COMPLETED,
                TaskStatus.FAILED
            ]:
                continue

            # Check dependencies
            dependencies = dag.get_task_dependencies(node_id)
            deps_satisfied = all(
                dep.task_id in completed
                for dep in dependencies
            )

            if deps_satisfied and task.status == TaskStatus.PENDING:
                ready.append(task)

        return ready

    def _has_active_tasks(self, dag: TaskDAG) -> bool:
        """
        Check if DAG has any active tasks.

        Args:
            dag: TaskDAG to check

        Returns:
            True if there are active tasks
        """
        for node_id in dag.graph.nodes():
            task = dag.get_node(node_id)
            if task.status.is_active:
                return True

        # Check subgraphs
        for subgraph in dag.subgraphs.values():
            if self._has_active_tasks(subgraph):
                return True

        return False

    def _execute_task(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """
        Execute a single task synchronously.

        Args:
            task: TaskNode to execute
            dag: Parent DAG

        Returns:
            Completed TaskNode
        """
        return self.solver.solve(task, dag, task.depth)

    async def _async_execute_task(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """
        Execute a single task asynchronously.

        Args:
            task: TaskNode to execute
            dag: Parent DAG

        Returns:
            Completed TaskNode
        """
        return await self.solver.async_solve(task, dag, task.depth)

    def execute_with_monitoring(
        self,
        dag: TaskDAG,
        update_interval: float = 1.0
    ) -> Dict[str, Any]:
        """
        Execute DAG with real-time monitoring.

        Args:
            dag: TaskDAG to execute
            update_interval: Seconds between status updates

        Returns:
            Execution results
        """
        import threading

        # Start monitoring thread
        stop_monitoring = threading.Event()

        def monitor():
            while not stop_monitoring.is_set():
                stats = dag.get_statistics()
                print(f"\rProgress: {stats['status_counts']}", end="", flush=True)
                stop_monitoring.wait(update_interval)

        monitor_thread = threading.Thread(target=monitor)
        monitor_thread.start()

        try:
            # Execute DAG
            results = self.execute_dag(dag)
        finally:
            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()
            print()  # New line after progress

        return results


class ParallelSolver:
    """
    High-level interface for parallel task solving with DAG management.
    """

    def __init__(
        self,
        max_workers: int = 4,
        max_depth: int = 5,
        enable_monitoring: bool = False
    ):
        """
        Initialize parallel solver.

        Args:
            max_workers: Maximum parallel workers
            max_depth: Maximum recursion depth
            enable_monitoring: Enable progress monitoring
        """
        self.solver = RecursiveSolver(max_depth=max_depth)
        self.executor = DAGExecutor(
            solver=self.solver,
            max_workers=max_workers,
            enable_async=True
        )
        self.enable_monitoring = enable_monitoring

    def solve(self, task: str) -> Dict[str, Any]:
        """
        Solve a task with parallel execution.

        Args:
            task: Task goal string

        Returns:
            Solution results and statistics
        """
        # Create initial task and DAG
        root_task = TaskNode(goal=task, depth=0, max_depth=self.solver.max_depth)
        dag = TaskDAG()
        dag.add_node(root_task)

        # Execute with monitoring if enabled
        if self.enable_monitoring:
            return self.executor.execute_with_monitoring(dag)
        else:
            return self.executor.execute_dag(dag)

    async def async_solve(self, task: str) -> Dict[str, Any]:
        """
        Asynchronously solve a task with parallel execution.

        Args:
            task: Task goal string

        Returns:
            Solution results and statistics
        """
        # Create initial task and DAG
        root_task = TaskNode(goal=task, depth=0, max_depth=self.solver.max_depth)
        dag = TaskDAG()
        dag.add_node(root_task)

        # Execute asynchronously
        return await self.executor.async_execute_dag(dag)