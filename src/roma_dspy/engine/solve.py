"""
Recursive solver for hierarchical task decomposition with depth constraints.
"""

import asyncio
import logging
import warnings
from datetime import datetime
from typing import Optional, Union, List, Tuple

import dspy

from src.roma_dspy.engine import TaskDAG
from src.roma_dspy.modules import Aggregator, Atomizer, Executor, Planner, Verifier
from src.roma_dspy.signatures import SubTask, TaskNode
from src.roma_dspy.types import ModuleResult, NodeType, TaskStatus

# Configure logging
logger = logging.getLogger(__name__)

# Suppress DSPy warnings about forward() usage
warnings.filterwarnings("ignore", message="Calling module.forward.*is discouraged")


class RecursiveSolver:
    """
    Implements recursive hierarchical task decomposition algorithm.

    Key features:
    - Maximum recursion depth constraint with forced execution
    - Comprehensive execution tracking for all modules
    - State-based execution flow
    - Nested DAG management for hierarchical decomposition
    - Async and sync execution support
    - Integrated visualization support
    """

    def __init__(
        self,
        atomizer: Optional[Atomizer] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        aggregator: Optional[Aggregator] = None,
        verifier: Optional[Verifier] = None,
        max_depth: int = 2,
        lm: Optional[dspy.LM] = None,
        enable_logging: bool = False
    ):
        """
        Initialize the recursive solver.

        Args:
            atomizer: Module for determining task atomicity
            planner: Module for task decomposition
            executor: Module for atomic task execution
            aggregator: Module for result synthesis
            verifier: Module for result validation (not yet implemented)
            max_depth: Maximum recursion depth
            lm: Language model to use
            enable_logging: Whether to enable debug logging
            visualizer: Optional visualizer for execution tracking
        """
        # Initialize modules with defaults if not provided
        self.atomizer = atomizer or Atomizer(lm=lm)
        self.planner = planner or Planner(lm=lm)
        self.executor = executor or Executor(lm=lm)
        self.aggregator = aggregator or Aggregator(lm=lm)
        self.verifier = verifier  # Optional, not yet implemented

        self.max_depth = max_depth
        self.last_dag = None  # Store last DAG for visualization

        # Configure logging
        if enable_logging:
            logging.basicConfig(level=logging.DEBUG)
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    # ==================== Main Entry Points ====================

    def solve(
        self,
        task: Union[str, TaskNode],
        dag: Optional[TaskDAG] = None,
        depth: int = 0
    ) -> TaskNode:
        """
        Synchronously solve a task using recursive decomposition.

        Args:
            task: Task goal string or TaskNode
            dag: Optional DAG to track execution
            depth: Current recursion depth

        Returns:
            Completed TaskNode with results
        """
        logger.debug(f"Starting solve for task: {task if isinstance(task, str) else task.goal}")

        # Initialize task and DAG
        task, dag = self._initialize_task_and_dag(task, dag, depth)

        # Execute based on current state
        task = self._execute_state_machine(task, dag)

        logger.debug(f"Completed solve with status: {task.status}")
        return task

    async def async_solve(
        self,
        task: Union[str, TaskNode],
        dag: Optional[TaskDAG] = None,
        depth: int = 0
    ) -> TaskNode:
        """
        Asynchronously solve a task using recursive decomposition.

        Args:
            task: Task goal string or TaskNode
            dag: Optional DAG to track execution
            depth: Current recursion depth

        Returns:
            Completed TaskNode with results
        """
        logger.debug(f"Starting async_solve for task: {task if isinstance(task, str) else task.goal}")

        # Initialize task and DAG
        task, dag = self._initialize_task_and_dag(task, dag, depth)

        # Execute based on current state
        task = await self._async_execute_state_machine(task, dag)

        logger.debug(f"Completed async_solve with status: {task.status}")
        return task

    # ==================== Initialization ====================

    def _initialize_task_and_dag(
        self,
        task: Union[str, TaskNode],
        dag: Optional[TaskDAG],
        depth: int
    ) -> Tuple[TaskNode, TaskDAG]:
        """Initialize task node and DAG for execution."""
        # Convert string to TaskNode if needed
        if isinstance(task, str):
            task = TaskNode(goal=task, depth=depth, max_depth=self.max_depth)

        # Create DAG if not provided
        if dag is None:
            dag = TaskDAG()
            dag.add_node(task)
            self.last_dag = dag  # Store for visualization

        return task, dag

    # ==================== State Machine Execution ====================

    def _execute_state_machine(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Execute synchronous state machine for task processing."""
        # Check for forced execution at max depth
        if task.should_force_execute():
            logger.debug(f"Force executing task at max depth: {task.depth}")
            return self._force_execute(task, dag)

        # Process based on current state
        if task.status == TaskStatus.PENDING:
            task = self._atomize(task, dag)

        if task.status == TaskStatus.ATOMIZING:
            task = self._transition_from_atomizing(task, dag)

        if task.status == TaskStatus.PLANNING:
            task = self._plan(task, dag)

        if task.status == TaskStatus.EXECUTING:
            task = self._execute(task, dag)
        elif task.status == TaskStatus.PLAN_DONE:
            task = self._process_subgraph(task, dag)

        return task

    async def _async_execute_state_machine(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Execute asynchronous state machine for task processing."""
        # Check for forced execution at max depth
        if task.should_force_execute():
            logger.debug(f"Force executing task at max depth: {task.depth}")
            return await self._async_force_execute(task, dag)

        # Process based on current state
        if task.status == TaskStatus.PENDING:
            task = await self._async_atomize(task, dag)

        if task.status == TaskStatus.ATOMIZING:
            task = self._transition_from_atomizing(task, dag)

        if task.status == TaskStatus.PLANNING:
            task = await self._async_plan(task, dag)

        if task.status == TaskStatus.EXECUTING:
            task = await self._async_execute(task, dag)
        elif task.status == TaskStatus.PLAN_DONE:
            task = await self._async_process_subgraph(task, dag)

        return task

    def _transition_from_atomizing(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Handle transition from ATOMIZING state."""
        if task.node_type == NodeType.EXECUTE:
            task = task.transition_to(TaskStatus.EXECUTING)
        else:
            task = task.transition_to(TaskStatus.PLANNING)
        dag.update_node(task)
        return task

    # ==================== Module Execution Methods ====================

    def _atomize(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Determine if task is atomic or needs decomposition."""
        logger.debug(f"Atomizing task: {task.goal[:50]}...")

        old_status = task.status
        task = task.transition_to(TaskStatus.ATOMIZING)

        result, duration = self._execute_module(self.atomizer, task.goal)

        task = self._record_module_result(
            task, "atomizer", task.goal,
            {"is_atomic": result.is_atomic, "node_type": result.node_type.value},
            duration
        )

        task = task.set_node_type(result.node_type)
        dag.update_node(task)
        return task

    async def _async_atomize(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Async determine if task is atomic or needs decomposition."""
        logger.debug(f"Async atomizing task: {task.goal[:50]}...")
        task = task.transition_to(TaskStatus.ATOMIZING)

        result, duration = await self._async_execute_module(self.atomizer, task.goal)

        task = self._record_module_result(
            task, "atomizer", task.goal,
            {"is_atomic": result.is_atomic, "node_type": result.node_type.value},
            duration
        )

        task = task.set_node_type(result.node_type)
        dag.update_node(task)
        return task

    def _plan(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Decompose task into subtasks."""
        logger.debug(f"Planning task: {task.goal[:50]}...")

        result, duration = self._execute_module(self.planner, task.goal)

        task = self._record_module_result(
            task, "planner", task.goal,
            {
                "subtasks": [s.model_dump() for s in result.subtasks],
                "dependencies": result.dependencies_graph
            },
            duration
        )

        # Create and link subtasks
        task = self._create_subtask_graph(task, dag, result)

        old_status = task.status
        task = task.transition_to(TaskStatus.PLAN_DONE)

        dag.update_node(task)
        return task

    async def _async_plan(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Async decompose task into subtasks."""
        logger.debug(f"Async planning task: {task.goal[:50]}...")

        result, duration = await self._async_execute_module(self.planner, task.goal)

        task = self._record_module_result(
            task, "planner", task.goal,
            {
                "subtasks": [s.model_dump() for s in result.subtasks],
                "dependencies": result.dependencies_graph
            },
            duration
        )

        # Create and link subtasks
        task = self._create_subtask_graph(task, dag, result)

        task = task.transition_to(TaskStatus.PLAN_DONE)
        dag.update_node(task)
        return task

    def _execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Execute atomic task."""
        logger.debug(f"Executing task: {task.goal[:50]}...")

        result, duration = self._execute_module(self.executor, task.goal)

        task = self._record_module_result(
            task, "executor", task.goal, result.output, duration
        )

        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    async def _async_execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Async execute atomic task."""
        logger.debug(f"Async executing task: {task.goal[:50]}...")

        result, duration = await self._async_execute_module(self.executor, task.goal)

        task = self._record_module_result(
            task, "executor", task.goal, result.output, duration
        )

        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    def _force_execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Force execution of task at max depth."""
        logger.debug(f"Force executing task at depth {task.depth}: {task.goal[:50]}...")

        # Skip atomizer, go straight to execution
        task = task.set_node_type(NodeType.EXECUTE)
        task = task.transition_to(TaskStatus.EXECUTING)
        dag.update_node(task)

        result, duration = self._execute_module(self.executor, task.goal)

        task = self._record_module_result(
            task, "executor", task.goal, result.output, duration,
            metadata={"forced": True, "depth": task.depth}
        )

        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    async def _async_force_execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Force async execution of task at max depth."""
        logger.debug(f"Force async executing task at depth {task.depth}: {task.goal[:50]}...")

        # Skip atomizer, go straight to execution
        task = task.set_node_type(NodeType.EXECUTE)
        task = task.transition_to(TaskStatus.EXECUTING)
        dag.update_node(task)

        result, duration = await self._async_execute_module(self.executor, task.goal)

        task = self._record_module_result(
            task, "executor", task.goal, result.output, duration,
            metadata={"forced": True, "depth": task.depth}
        )

        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    # ==================== Subgraph Processing ====================

    def _process_subgraph(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Process and aggregate subgraph results."""
        subgraph = dag.get_subgraph(task.subgraph_id)
        if subgraph:
            logger.debug(f"Processing subgraph for task: {task.goal[:50]}...")
            self._solve_subgraph(subgraph)
            task = self._aggregate(task, subgraph, dag)
        return task

    async def _async_process_subgraph(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Async process and aggregate subgraph results."""
        subgraph = dag.get_subgraph(task.subgraph_id)
        if subgraph:
            logger.debug(f"Async processing subgraph for task: {task.goal[:50]}...")
            await self._async_solve_subgraph(subgraph)
            task = await self._async_aggregate(task, subgraph, dag)
        return task

    def _solve_subgraph(self, subgraph: TaskDAG) -> None:
        """Recursively solve all tasks in subgraph."""
        execution_order = subgraph.get_execution_order()

        for task_id in execution_order:
            task = subgraph.get_node(task_id)
            dependencies = subgraph.get_task_dependencies(task_id)

            # Execute if dependencies are satisfied
            if all(dep.status == TaskStatus.COMPLETED for dep in dependencies):
                if task.status == TaskStatus.PENDING:
                    solved_task = self.solve(task, subgraph, task.depth)
                    subgraph.update_node(solved_task)

    async def _async_solve_subgraph(self, subgraph: TaskDAG) -> None:
        """Async recursively solve all tasks in subgraph with parallelism."""
        pending_tasks = set(subgraph.graph.nodes())
        completed_tasks = set()

        while pending_tasks:
            # Find tasks ready to execute
            ready_tasks = self._get_ready_tasks(
                subgraph, pending_tasks, completed_tasks
            )

            if not ready_tasks:
                logger.warning("No tasks ready, possible dependency issue")
                break

            # Execute ready tasks in parallel
            solved_tasks = await self._execute_tasks_parallel(ready_tasks, subgraph)

            # Update tracking
            for solved_task in solved_tasks:
                subgraph.update_node(solved_task)
                pending_tasks.remove(solved_task.task_id)
                completed_tasks.add(solved_task.task_id)

    def _aggregate(self, task: TaskNode, subgraph: TaskDAG, dag: TaskDAG) -> TaskNode:
        """Aggregate results from completed subtasks."""
        logger.debug(f"Aggregating results for task: {task.goal[:50]}...")
        task = task.transition_to(TaskStatus.AGGREGATING)

        # Collect subtask results
        subtask_results = self._collect_subtask_results(subgraph)

        # Call aggregator
        result, duration = self._execute_module(
            self.aggregator,
            original_goal=task.goal,
            subtasks_results=subtask_results
        )

        task = self._record_module_result(
            task, "aggregator",
            {"original_goal": task.goal, "subtask_count": len(subtask_results)},
            result.synthesized_result,
            duration
        )

        task = task.with_result(result.synthesized_result)
        dag.update_node(task)
        return task

    async def _async_aggregate(self, task: TaskNode, subgraph: TaskDAG, dag: TaskDAG) -> TaskNode:
        """Async aggregate results from completed subtasks."""
        logger.debug(f"Async aggregating results for task: {task.goal[:50]}...")
        task = task.transition_to(TaskStatus.AGGREGATING)

        # Collect subtask results
        subtask_results = self._collect_subtask_results(subgraph)

        # Call aggregator
        result, duration = await self._async_execute_module(
            self.aggregator,
            original_goal=task.goal,
            subtasks_results=subtask_results
        )

        task = self._record_module_result(
            task, "aggregator",
            {"original_goal": task.goal, "subtask_count": len(subtask_results)},
            result.synthesized_result,
            duration
        )

        task = task.with_result(result.synthesized_result)
        dag.update_node(task)
        return task

    # ==================== Helper Methods ====================

    def _execute_module(self, module, *args, **kwargs):
        """Execute a module and return result with duration."""
        start_time = datetime.now()
        result = module(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()
        return result, duration

    async def _async_execute_module(self, module, *args, **kwargs):
        """Async execute a module and return result with duration."""
        start_time = datetime.now()
        result = await module.aforward(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()
        return result, duration

    def _record_module_result(
        self,
        task: TaskNode,
        module_name: str,
        input_data,
        output_data,
        duration: float,
        metadata: Optional[dict] = None
    ) -> TaskNode:
        """Record module execution result in task."""
        module_result = ModuleResult(
            module_name=module_name,
            input=input_data,
            output=output_data,
            timestamp=datetime.now(),
            duration=duration,
            metadata=metadata or {}
        )
        return task.record_module_execution(module_name, module_result)

    def _create_subtask_graph(self, task: TaskNode, dag: TaskDAG, planner_result) -> TaskNode:
        """Create subtask nodes and subgraph."""
        subtask_nodes = []
        for subtask in planner_result.subtasks:
            subtask_node = TaskNode(
                goal=subtask.goal,
                parent_id=task.task_id,
                depth=task.depth + 1,
                max_depth=self.max_depth
            )
            subtask_nodes.append(subtask_node)

        # Create subgraph with dependencies
        dag.create_subgraph(
            task.task_id,
            subtask_nodes,
            planner_result.dependencies_graph
        )

        # Get updated task with subgraph_id set
        task = dag.get_node(task.task_id)

        # Update metrics
        updated_metrics = task.metrics.model_copy()
        updated_metrics.subtasks_created = len(subtask_nodes)
        task = task.model_copy(update={'metrics': updated_metrics})

        return task

    def _collect_subtask_results(self, subgraph: TaskDAG) -> List[SubTask]:
        """Collect results from all subtasks in subgraph."""
        subtask_results = []
        for subtask_node in subgraph.get_all_tasks(include_subgraphs=False):
            subtask_results.append(
                SubTask(
                    goal=subtask_node.goal,
                    task_type=subtask_node.task_type,
                    dependencies=[],
                    result=str(subtask_node.result) if subtask_node.result else ""
                )
            )
        return subtask_results

    def _get_ready_tasks(
        self,
        subgraph: TaskDAG,
        pending_tasks: set,
        completed_tasks: set
    ) -> List[TaskNode]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        for task_id in pending_tasks:
            task = subgraph.get_node(task_id)
            dependencies = subgraph.get_task_dependencies(task_id)

            if all(dep.task_id in completed_tasks for dep in dependencies):
                ready_tasks.append(task)

        return ready_tasks

    async def _execute_tasks_parallel(
        self,
        tasks: List[TaskNode],
        subgraph: TaskDAG
    ) -> List[TaskNode]:
        """Execute multiple tasks in parallel."""
        async_tasks = []
        for task in tasks:
            if task.status == TaskStatus.PENDING:
                async_tasks.append(self.async_solve(task, subgraph, task.depth))

        if async_tasks:
            return await asyncio.gather(*async_tasks)
        return []

    # ==================== Visualization Methods (Deprecated) ====================

    def get_execution_tree(self, task: Optional[TaskNode] = None) -> str:
        """Get a tree visualization of the task execution."""
        if not self.last_dag:
            return "No execution data available. Run solve() first."

        if task is None:
            task = self._get_root_task()
            if not task:
                return "No root task found in DAG."

        return task.print_tree(self.last_dag)

    def get_execution_summary(self, task: Optional[TaskNode] = None) -> str:
        """Get a detailed summary of the task execution."""
        if not self.last_dag:
            return "No execution data available. Run solve() first."

        if task is None:
            task = self._get_root_task()
            if not task:
                return "No root task found in DAG."

        return task.pretty_print()

    def _get_root_task(self) -> Optional[TaskNode]:
        """Get the root task from the DAG."""
        if not self.last_dag:
            return None

        root_tasks = [
            self.last_dag.get_node(n)
            for n in self.last_dag.graph.nodes()
            if self.last_dag.get_node(n).is_root
        ]

        return root_tasks[0] if root_tasks else None


# ==================== Convenience Functions ====================

def solve(task: Union[str, TaskNode], max_depth: int = 2, **kwargs) -> TaskNode:
    """
    Solve a task using recursive decomposition.

    Args:
        task: Task goal string or TaskNode
        max_depth: Maximum recursion depth
        **kwargs: Additional arguments for RecursiveSolver

    Returns:
        Completed TaskNode with results
    """
    solver = RecursiveSolver(max_depth=max_depth, **kwargs)
    return solver.solve(task)


async def async_solve(task: Union[str, TaskNode], max_depth: int = 2, **kwargs) -> TaskNode:
    """
    Asynchronously solve a task using recursive decomposition.

    Args:
        task: Task goal string or TaskNode
        max_depth: Maximum recursion depth
        **kwargs: Additional arguments for RecursiveSolver

    Returns:
        Completed TaskNode with results
    """
    solver = RecursiveSolver(max_depth=max_depth, **kwargs)
    return await solver.async_solve(task)