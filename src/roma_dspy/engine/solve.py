"""
Recursive solver for hierarchical task decomposition with depth constraints.
"""

from typing import Optional, Union
from datetime import datetime
import asyncio
import dspy

from src.roma_dspy.engine import TaskDAG
from src.roma_dspy.signatures import TaskNode, SubTask
from src.roma_dspy.types import TaskStatus, NodeType, ModuleResult
from src.roma_dspy.modules import Atomizer, Planner, Executor, Aggregator, Verifier


class RecursiveSolver:
    """
    Implements the recursive hierarchical task decomposition algorithm.

    Key features:
    - Maximum recursion depth constraint with forced execution
    - Comprehensive execution tracking for all modules
    - State-based execution flow
    - Nested DAG management for hierarchical decomposition
    """

    def __init__(
        self,
        atomizer: Optional[Atomizer] = None,
        planner: Optional[Planner] = None,
        executor: Optional[Executor] = None,
        aggregator: Optional[Aggregator] = None,
        verifier: Optional[Verifier] = None,
        max_depth: int = 2,
        lm: Optional[dspy.LM] = None
    ):
        """
        Initialize the recursive solver.

        Args:
            atomizer: Module for determining task atomicity
            planner: Module for task decomposition
            executor: Module for atomic task execution
            aggregator: Module for result synthesis
            verifier: Module for result validation
            max_depth: Maximum recursion depth
            lm: Language model to use
        """
        self.atomizer = atomizer or Atomizer(lm=lm)
        self.planner = planner or Planner(lm=lm)
        self.executor = executor or Executor(lm=lm)
        self.aggregator = aggregator or Aggregator(lm=lm)
        #TODO: Add verifier at later stage.
        self.verifier = verifier or Verifier(lm=lm) if verifier else None
        self.max_depth = max_depth
        self.last_dag = None  # Store last DAG for visualization

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
        # import ipdb; ipdb.set_trace()
        # Convert string to TaskNode if needed
        if isinstance(task, str):
            task = TaskNode(goal=task, depth=depth, max_depth=self.max_depth)

        # Create DAG if not provided
        if dag is None:
            dag = TaskDAG()
            dag.add_node(task)
            self.last_dag = dag  # Store for visualization

        # Check for forced execution at max depth
        if task.should_force_execute():
            return self._force_execute(task, dag)

        # State machine execution
        if task.status == TaskStatus.PENDING:
            task = self._atomize(task, dag)

        if task.status == TaskStatus.ATOMIZING:
            if task.node_type == NodeType.EXECUTE:
                task = task.transition_to(TaskStatus.EXECUTING)
            else:
                task = task.transition_to(TaskStatus.PLANNING)
            dag.update_node(task)

        if task.status == TaskStatus.PLANNING:
            task = self._plan(task, dag)

        if task.status == TaskStatus.EXECUTING:
            task = self._execute(task, dag)

        elif task.status == TaskStatus.PLAN_DONE:
            # Execute subtasks in subgraph
            subgraph = dag.get_subgraph(task.subgraph_id)
            if subgraph:
                self._solve_subgraph(subgraph)
                task = self._aggregate(task, subgraph, dag)

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
        # Convert string to TaskNode if needed
        if isinstance(task, str):
            task = TaskNode(goal=task, depth=depth, max_depth=self.max_depth)

        # Create DAG if not provided
        if dag is None:
            dag = TaskDAG()
            dag.add_node(task)
            self.last_dag = dag  # Store for visualization

        # Check for forced execution at max depth
        if task.should_force_execute():
            return await self._async_force_execute(task, dag)

        # State machine execution
        if task.status == TaskStatus.PENDING:
            task = await self._async_atomize(task, dag)

        if task.status == TaskStatus.ATOMIZING:
            if task.node_type == NodeType.EXECUTE:
                task = task.transition_to(TaskStatus.EXECUTING)
            else:
                task = task.transition_to(TaskStatus.PLANNING)
            dag.update_node(task)

        if task.status == TaskStatus.PLANNING:
            task = await self._async_plan(task, dag)

        if task.status == TaskStatus.EXECUTING:
            task = await self._async_execute(task, dag)

        elif task.status == TaskStatus.PLAN_DONE:
            # Execute subtasks in subgraph
            subgraph = dag.get_subgraph(task.subgraph_id)
            if subgraph:
                await self._async_solve_subgraph(subgraph)
                task = await self._async_aggregate(task, subgraph, dag)

        return task

    def _force_execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Force execution of task at max depth."""
        start_time = datetime.now()

        # Skip atomizer, go straight to execution
        task = task.set_node_type(NodeType.EXECUTE)
        task = task.transition_to(TaskStatus.EXECUTING)
        dag.update_node(task)

        # Execute directly
        result = self.executor(task.goal)

        # Record execution
        module_result = ModuleResult(
            module_name="executor",
            input=task.goal,
            output=result.output,
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds(),
            metadata={"forced": True, "depth": task.depth}
        )

        task = task.record_module_execution("executor", module_result)
        task = task.with_result(result.output)
        dag.update_node(task)

        return task

    async def _async_force_execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Force async execution of task at max depth."""
        start_time = datetime.now()

        # Skip atomizer, go straight to execution
        task = task.set_node_type(NodeType.EXECUTE)
        task = task.transition_to(TaskStatus.EXECUTING)
        dag.update_node(task)

        # Execute directly
        result = await self.executor.aforward(task.goal)

        # Record execution
        module_result = ModuleResult(
            module_name="executor",
            input=task.goal,
            output=result.output,
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds(),
            metadata={"forced": True, "depth": task.depth}
        )

        task = task.record_module_execution("executor", module_result)
        task = task.with_result(result.output)
        dag.update_node(task)

        return task

    def _atomize(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Determine if task is atomic or needs decomposition."""
        start_time = datetime.now()
        task = task.transition_to(TaskStatus.ATOMIZING)
        # Call atomizer
        result = self.atomizer(task.goal)

        # Record execution
        module_result = ModuleResult(
            module_name="atomizer",
            input=task.goal,
            output={"is_atomic": result.is_atomic, "node_type": result.node_type.value},
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds()
        )

        task = task.record_module_execution("atomizer", module_result)
        task = task.set_node_type(result.node_type)
        dag.update_node(task)

        return task

    async def _async_atomize(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Async determine if task is atomic or needs decomposition."""
        start_time = datetime.now()
        task = task.transition_to(TaskStatus.ATOMIZING)

        # Call atomizer
        result = await self.atomizer.aforward(task.goal)

        # Record execution
        module_result = ModuleResult(
            module_name="atomizer",
            input=task.goal,
            output={"is_atomic": result.is_atomic, "node_type": result.node_type.value},
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds()
        )

        task = task.record_module_execution("atomizer", module_result)
        task = task.set_node_type(result.node_type)
        dag.update_node(task)

        return task

    def _plan(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Decompose task into subtasks."""
        start_time = datetime.now()

        # Call planner
        result = self.planner(task.goal)

        # Record execution
        module_result = ModuleResult(
            module_name="planner",
            input=task.goal,
            output={
                "subtasks": [s.model_dump() for s in result.subtasks],
                "dependencies": result.dependencies_graph
            },
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds()
        )

        task = task.record_module_execution("planner", module_result)

        # Create subtask nodes
        subtask_nodes = []
        for subtask in result.subtasks:
            subtask_node = TaskNode(
                goal=subtask.goal,
                parent_id=task.task_id,
                depth=task.depth + 1,
                max_depth=self.max_depth
            )
            subtask_nodes.append(subtask_node)

        # Create subgraph with dependencies
        subgraph = dag.create_subgraph(
            task.task_id,
            subtask_nodes,
            result.dependencies_graph
        )

        # Get the updated task with subgraph_id set
        task = dag.get_node(task.task_id)

        # Update metrics while preserving subgraph_id
        updated_metrics = task.metrics.model_copy()
        updated_metrics.subtasks_created = len(subtask_nodes)
        task = task.model_copy(update={'metrics': updated_metrics})

        task = task.transition_to(TaskStatus.PLAN_DONE)
        dag.update_node(task)

        return task

    async def _async_plan(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Async decompose task into subtasks."""
        start_time = datetime.now()

        # Call planner
        result = await self.planner.aforward(task.goal)

        # Record execution
        module_result = ModuleResult(
            module_name="planner",
            input=task.goal,
            output={
                "subtasks": [s.model_dump() for s in result.subtasks],
                "dependencies": result.dependencies_graph
            },
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds()
        )

        task = task.record_module_execution("planner", module_result)

        # Create subtask nodes
        subtask_nodes = []
        for subtask in result.subtasks:
            subtask_node = TaskNode(
                goal=subtask.goal,
                parent_id=task.task_id,
                depth=task.depth + 1,
                max_depth=self.max_depth
            )
            subtask_nodes.append(subtask_node)

        # Create subgraph with dependencies
        subgraph = dag.create_subgraph(
            task.task_id,
            subtask_nodes,
            result.dependencies_graph
        )

        # Get the updated task with subgraph_id set
        task = dag.get_node(task.task_id)

        # Update metrics while preserving subgraph_id
        updated_metrics = task.metrics.model_copy()
        updated_metrics.subtasks_created = len(subtask_nodes)
        task = task.model_copy(update={'metrics': updated_metrics})

        task = task.transition_to(TaskStatus.PLAN_DONE)
        dag.update_node(task)

        return task

    def _execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Execute atomic task."""
        start_time = datetime.now()

        # Call executor
        result = self.executor(task.goal)

        # Record execution
        module_result = ModuleResult(
            module_name="executor",
            input=task.goal,
            output=result.output,
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds()
        )

        task = task.record_module_execution("executor", module_result)
        task = task.with_result(result.output)
        dag.update_node(task)

        return task

    async def _async_execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        """Async execute atomic task."""
        start_time = datetime.now()

        # Call executor
        result = await self.executor.aforward(task.goal)

        # Record execution
        module_result = ModuleResult(
            module_name="executor",
            input=task.goal,
            output=result.output,
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds()
        )

        task = task.record_module_execution("executor", module_result)
        task = task.with_result(result.output)
        dag.update_node(task)

        return task

    def _solve_subgraph(self, subgraph: TaskDAG) -> None:
        """Recursively solve all tasks in subgraph."""
        # Get execution order
        execution_order = subgraph.get_execution_order()

        for task_id in execution_order:
            task = subgraph.get_node(task_id)

            # Check dependencies
            dependencies = subgraph.get_task_dependencies(task_id)
            if all(dep.status == TaskStatus.COMPLETED for dep in dependencies):
                # Recursive solve - let solve handle state transitions
                if task.status == TaskStatus.PENDING:
                    solved_task = self.solve(task, subgraph, task.depth)
                    subgraph.update_node(solved_task)

    async def _async_solve_subgraph(self, subgraph: TaskDAG) -> None:
        """Async recursively solve all tasks in subgraph with parallelism."""
        # Track pending tasks
        pending_tasks = set(subgraph.graph.nodes())
        completed_tasks = set()

        while pending_tasks:
            # Find tasks ready to execute
            ready_tasks = []
            for task_id in pending_tasks:
                task = subgraph.get_node(task_id)

                # Check dependencies
                dependencies = subgraph.get_task_dependencies(task_id)
                if all(dep.task_id in completed_tasks for dep in dependencies):
                    ready_tasks.append(task)

            if not ready_tasks:
                # No tasks ready, might be a dependency issue
                break

            # Execute ready tasks in parallel - let async_solve handle state transitions
            async_tasks = []
            for task in ready_tasks:
                if task.status == TaskStatus.PENDING:
                    async_tasks.append(self.async_solve(task, subgraph, task.depth))

            # Wait for all to complete
            solved_tasks = await asyncio.gather(*async_tasks)

            # Update subgraph
            for solved_task in solved_tasks:
                subgraph.update_node(solved_task)
                pending_tasks.remove(solved_task.task_id)
                completed_tasks.add(solved_task.task_id)

    def _aggregate(self, task: TaskNode, subgraph: TaskDAG, dag: TaskDAG) -> TaskNode:
        """Aggregate results from completed subtasks."""
        start_time = datetime.now()
        task = task.transition_to(TaskStatus.AGGREGATING)

        # Collect subtask results
        subtask_results = []
        for subtask_node in subgraph.get_all_tasks(include_subgraphs=False):
            subtask_results.append(
                SubTask(
                    goal=subtask_node.goal,
                    task_type=subtask_node.task_type,  # Add task_type from node
                    dependencies=[],  # Dependencies not needed for aggregation
                    result=str(subtask_node.result) if subtask_node.result else ""
                )
            )

        # Call aggregator
        result = self.aggregator(
            original_goal=task.goal,
            subtasks_results=subtask_results
        )

        # Record execution
        module_result = ModuleResult(
            module_name="aggregator",
            input={
                "original_goal": task.goal,
                "subtask_count": len(subtask_results)
            },
            output=result.synthesized_result,
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds()
        )

        task = task.record_module_execution("aggregator", module_result)
        task = task.with_result(result.synthesized_result)
        dag.update_node(task)

        return task

    async def _async_aggregate(self, task: TaskNode, subgraph: TaskDAG, dag: TaskDAG) -> TaskNode:
        """Async aggregate results from completed subtasks."""
        start_time = datetime.now()
        task = task.transition_to(TaskStatus.AGGREGATING)

        # Collect subtask results
        subtask_results = []
        for subtask_node in subgraph.get_all_tasks(include_subgraphs=False):
            subtask_results.append(
                SubTask(
                    goal=subtask_node.goal,
                    task_type=subtask_node.task_type,  # Add task_type from node
                    dependencies=[],  # Dependencies not needed for aggregation
                    result=str(subtask_node.result) if subtask_node.result else ""
                )
            )

        # Call aggregator
        result = await self.aggregator.aforward(
            original_goal=task.goal,
            subtasks_results=subtask_results
        )

        # Record execution
        module_result = ModuleResult(
            module_name="aggregator",
            input={
                "original_goal": task.goal,
                "subtask_count": len(subtask_results)
            },
            output=result.synthesized_result,
            timestamp=start_time,
            duration=(datetime.now() - start_time).total_seconds()
        )

        task = task.record_module_execution("aggregator", module_result)
        task = task.with_result(result.synthesized_result)
        dag.update_node(task)

        return task

    def get_execution_tree(self, task: Optional[TaskNode] = None) -> str:
        """
        Get a tree visualization of the task execution.

        Args:
            task: Optional specific task to visualize. If None, uses root task from last DAG.

        Returns:
            Tree representation as string
        """
        if not self.last_dag:
            return "No execution data available. Run solve() first."

        if task is None:
            # Get root task from DAG
            root_tasks = [self.last_dag.get_node(n) for n in self.last_dag.graph.nodes()
                          if self.last_dag.get_node(n).is_root]
            if not root_tasks:
                return "No root task found in DAG."
            task = root_tasks[0]

        return task.print_tree(self.last_dag)

    def get_execution_summary(self, task: Optional[TaskNode] = None) -> str:
        """
        Get a detailed summary of the task execution.

        Args:
            task: Optional specific task to summarize. If None, uses root task from last DAG.

        Returns:
            Detailed summary as string
        """
        if not self.last_dag:
            return "No execution data available. Run solve() first."

        if task is None:
            # Get root task from DAG
            root_tasks = [self.last_dag.get_node(n) for n in self.last_dag.graph.nodes()
                          if self.last_dag.get_node(n).is_root]
            if not root_tasks:
                return "No root task found in DAG."
            task = root_tasks[0]

        return task.pretty_print()


# Convenience functions
def solve(task: Union[str, TaskNode], max_depth: int = 2) -> TaskNode:
    """
    Solve a task using recursive decomposition.

    Args:
        task: Task goal string or TaskNode
        max_depth: Maximum recursion depth

    Returns:
        Completed TaskNode with results
    """
    solver = RecursiveSolver(max_depth=max_depth)
    return solver.solve(task)


async def async_solve(task: Union[str, TaskNode], max_depth: int = 2) -> TaskNode:
    """
    Asynchronously solve a task using recursive decomposition.

    Args:
        task: Task goal string or TaskNode
        max_depth: Maximum recursion depth

    Returns:
        Completed TaskNode with results
    """
    solver = RecursiveSolver(max_depth=max_depth)
    return await solver.async_solve(task)