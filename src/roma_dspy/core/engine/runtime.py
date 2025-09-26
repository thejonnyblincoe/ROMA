"""Runtime helpers for module execution and DAG manipulation."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable, List, Optional

from src.roma_dspy.core.engine.dag import TaskDAG
from src.roma_dspy.core.modules import Aggregator, Atomizer, Executor, Planner, Verifier
from src.roma_dspy.core.signatures import SubTask, TaskNode
from src.roma_dspy.types import ModuleResult, NodeType, TaskStatus, AgentType
from src.roma_dspy.resilience import with_module_resilience, measure_execution_time


SolveFn = Callable[[TaskNode, TaskDAG, int], TaskNode]
AsyncSolveFn = Callable[[TaskNode, TaskDAG, int], Awaitable[TaskNode]]


class ModuleRuntime:
    """Shared module orchestration for both sync and async solvers."""

    def __init__(
        self,
        atomizer: Atomizer,
        planner: Planner,
        executor: Executor,
        aggregator: Aggregator,
        verifier: Optional[Verifier] = None,
    ) -> None:
        self.atomizer = atomizer
        self.planner = planner
        self.executor = executor
        self.aggregator = aggregator
        self.verifier = verifier

    # ------------------------------------------------------------------
    # Core module execution helpers
    # ------------------------------------------------------------------

    def atomize(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        task = task.transition_to(TaskStatus.ATOMIZING)
        try:
            result, duration = self._execute_atomizer(task.goal)
            task = self._record_module_result(
                task,
                "atomizer",
                task.goal,
                {"is_atomic": result.is_atomic, "node_type": result.node_type.value},
                duration,
            )
        except Exception as e:
            self._enhance_error_context(e, AgentType.ATOMIZER, task)
            raise
        task = task.set_node_type(result.node_type)
        dag.update_node(task)
        return task

    async def atomize_async(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        task = task.transition_to(TaskStatus.ATOMIZING)
        try:
            result, duration = await self._async_execute_atomizer(task.goal)
            task = self._record_module_result(
                task,
                "atomizer",
                task.goal,
                {"is_atomic": result.is_atomic, "node_type": result.node_type.value},
                duration,
            )
            task = task.set_node_type(result.node_type)
        except Exception as e:
            self._enhance_error_context(e, AgentType.ATOMIZER, task)
            raise
        dag.update_node(task)
        return task

    def transition_from_atomizing(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        if task.node_type == NodeType.EXECUTE:
            task = task.transition_to(TaskStatus.EXECUTING)
        else:
            task = task.transition_to(TaskStatus.PLANNING)
        dag.update_node(task)
        return task

    def plan(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        try:
            result, duration = self._execute_planner(task.goal)
            task = self._record_module_result(
                task,
                "planner",
                task.goal,
                {"subtasks": len(result.subtasks), "dependencies": len(result.dependencies_graph)},
                duration,
            )
            task = self._create_subtask_graph(task, dag, result)
            task = task.transition_to(TaskStatus.PLAN_DONE)
            dag.update_node(task)
            return task
        except Exception as e:
            self._enhance_error_context(e, AgentType.PLANNER, task)
            raise

    async def plan_async(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        try:
            result, duration = await self._async_execute_planner(task.goal)
            task = self._record_module_result(
                task,
                "planner",
                task.goal,
                {"subtasks": len(result.subtasks), "dependencies": len(result.dependencies_graph)},
                duration,
            )
            task = self._create_subtask_graph(task, dag, result)
            task = task.transition_to(TaskStatus.PLAN_DONE)
            dag.update_node(task)
            return task
        except Exception as e:
            self._enhance_error_context(e, AgentType.PLANNER, task)
            raise

    def execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        try:
            result, duration = self._execute_executor(task.goal)
            task = self._record_module_result(
                task,
                "executor",
                task.goal,
                {"output_length": len(str(result)) if result else 0},
                duration,
            )
            task = task.with_result(result)
            dag.update_node(task)
            return task
        except Exception as e:
            self._enhance_error_context(e, AgentType.EXECUTOR, task)
            raise

    async def execute_async(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        try:
            result, duration = await self._async_execute_executor(task.goal)
            task = self._record_module_result(
                task,
                "executor",
                task.goal,
                {"output_length": len(str(result)) if result else 0},
                duration,
            )
            task = task.with_result(result)
            dag.update_node(task)
            return task
        except Exception as e:
            self._enhance_error_context(e, AgentType.EXECUTOR, task)
            raise

    def force_execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        task = task.set_node_type(NodeType.EXECUTE)
        task = task.transition_to(TaskStatus.EXECUTING)
        dag.update_node(task)
        result, duration = self._execute_module(self.executor, task.goal)
        task = self._record_module_result(
            task,
            "executor",
            task.goal,
            result.output,
            duration,
            metadata={"forced": True, "depth": task.depth},
        )
        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    async def force_execute_async(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        task = task.set_node_type(NodeType.EXECUTE)
        task = task.transition_to(TaskStatus.EXECUTING)
        dag.update_node(task)
        result, duration = await self._async_execute_module(self.executor, task.goal)
        task = self._record_module_result(
            task,
            "executor",
            task.goal,
            result.output,
            duration,
            metadata={"forced": True, "depth": task.depth},
        )
        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    def aggregate(self, task: TaskNode, subgraph: Optional[TaskDAG], dag: TaskDAG) -> TaskNode:
        if task.status != TaskStatus.PLAN_DONE:
            return task
        task = task.transition_to(TaskStatus.AGGREGATING)
        subtask_results = self._collect_subtask_results(subgraph)
        try:
            result, duration = self._execute_aggregator(
                original_goal=task.goal,
                subtasks_results=subtask_results,
            )
            task = self._record_module_result(
                task,
                "aggregator",
                task.goal,
                {"subtask_count": len(subtask_results), "result_length": len(str(result)) if result else 0},
                duration,
            )
        except Exception as e:
            self._enhance_error_context(e, AgentType.AGGREGATOR, task)
            raise
        task = task.with_result(result)
        dag.update_node(task)
        return task

    async def aggregate_async(
        self,
        task: TaskNode,
        subgraph: Optional[TaskDAG],
        dag: TaskDAG,
    ) -> TaskNode:
        if task.status != TaskStatus.PLAN_DONE:
            return task
        task = task.transition_to(TaskStatus.AGGREGATING)
        subtask_results = self._collect_subtask_results(subgraph)
        try:
            result, duration = await self._async_execute_aggregator(
                original_goal=task.goal,
                subtasks_results=subtask_results,
            )
            task = self._record_module_result(
                task,
                "aggregator",
                task.goal,
                {"subtask_count": len(subtask_results), "result_length": len(str(result)) if result else 0},
                duration,
            )
        except Exception as e:
            self._enhance_error_context(e, AgentType.AGGREGATOR, task)
            raise
        task = task.with_result(result)
        dag.update_node(task)
        return task

    # ------------------------------------------------------------------
    # Subgraph helpers
    # ------------------------------------------------------------------

    def process_subgraph(
        self,
        task: TaskNode,
        dag: TaskDAG,
        solve_fn: SolveFn,
    ) -> TaskNode:
        subgraph = dag.get_subgraph(task.subgraph_id) if task.subgraph_id else None
        if subgraph:
            self.solve_subgraph(subgraph, solve_fn)
            task = self.aggregate(task, subgraph, dag)
        return task

    async def process_subgraph_async(
        self,
        task: TaskNode,
        dag: TaskDAG,
        solve_fn: AsyncSolveFn,
    ) -> TaskNode:
        subgraph = dag.get_subgraph(task.subgraph_id) if task.subgraph_id else None
        if subgraph:
            await self.solve_subgraph_async(subgraph, solve_fn)
            task = await self.aggregate_async(task, subgraph, dag)
        return task

    def solve_subgraph(self, subgraph: TaskDAG, solve_fn: SolveFn) -> None:
        for task_id in subgraph.get_execution_order():
            task = subgraph.get_node(task_id)
            dependencies = subgraph.get_task_dependencies(task_id)
            if all(dep.status == TaskStatus.COMPLETED for dep in dependencies):
                if task.status == TaskStatus.PENDING:
                    solved = solve_fn(task, subgraph, task.depth)
                    subgraph.update_node(solved)

    async def solve_subgraph_async(
        self,
        subgraph: TaskDAG,
        solve_fn: AsyncSolveFn,
    ) -> None:
        pending = set(subgraph.graph.nodes())
        completed: set[str] = set()

        while pending:
            ready = self._get_ready_tasks(subgraph, pending, completed)
            if not ready:
                break

            solved_tasks = await self._execute_tasks_parallel(ready, subgraph, solve_fn)
            for solved_task in solved_tasks:
                subgraph.update_node(solved_task)
                pending.remove(solved_task.task_id)
                completed.add(solved_task.task_id)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @measure_execution_time
    @with_module_resilience(module_name="atomizer")
    def _execute_atomizer(self, *args, **kwargs):
        return self.atomizer(*args, **kwargs)

    @measure_execution_time
    @with_module_resilience(module_name="atomizer")
    async def _async_execute_atomizer(self, *args, **kwargs):
        return await self.atomizer.aforward(*args, **kwargs)

    @measure_execution_time
    @with_module_resilience(module_name="planner")
    def _execute_planner(self, *args, **kwargs):
        return self.planner(*args, **kwargs)

    @measure_execution_time
    @with_module_resilience(module_name="planner")
    async def _async_execute_planner(self, *args, **kwargs):
        return await self.planner.aforward(*args, **kwargs)

    @measure_execution_time
    @with_module_resilience(module_name="executor")
    def _execute_executor(self, *args, **kwargs):
        return self.executor(*args, **kwargs)

    @measure_execution_time
    @with_module_resilience(module_name="executor")
    async def _async_execute_executor(self, *args, **kwargs):
        return await self.executor.aforward(*args, **kwargs)

    @measure_execution_time
    @with_module_resilience(module_name="aggregator")
    def _execute_aggregator(self, *args, **kwargs):
        return self.aggregator(*args, **kwargs)

    @measure_execution_time
    @with_module_resilience(module_name="aggregator")
    async def _async_execute_aggregator(self, *args, **kwargs):
        return await self.aggregator.aforward(*args, **kwargs)

    def _record_module_result(
        self,
        task: TaskNode,
        module_name: str,
        input_data,
        output_data,
        duration: float,
        metadata: Optional[dict] = None,
    ) -> TaskNode:
        module_result = ModuleResult(
            module_name=module_name,
            input=input_data,
            output=output_data,
            timestamp=datetime.now(),
            duration=duration,
            metadata=metadata or {},
        )
        return task.record_module_execution(module_name, module_result)

    def _create_subtask_graph(self, task: TaskNode, dag: TaskDAG, planner_result) -> TaskNode:
        subtask_nodes: List[TaskNode] = []
        for subtask in planner_result.subtasks:
            subtask_node = TaskNode(
                goal=subtask.goal,
                parent_id=task.task_id,
                depth=task.depth + 1,
                max_depth=task.max_depth,
            )
            subtask_nodes.append(subtask_node)

        dag.create_subgraph(task.task_id, subtask_nodes, planner_result.dependencies_graph)
        task = dag.get_node(task.task_id)
        updated_metrics = task.metrics.model_copy()
        updated_metrics.subtasks_created = len(subtask_nodes)
        return task.model_copy(update={"metrics": updated_metrics})

    def _collect_subtask_results(self, subgraph: Optional[TaskDAG]) -> List[SubTask]:
        collected: List[SubTask] = []
        if subgraph:
            for node in subgraph.get_all_tasks(include_subgraphs=False):
                collected.append(
                    SubTask(
                        goal=node.goal,
                        task_type=node.task_type,
                        dependencies=[],
                        result=str(node.result) if node.result else "",
                    )
                )
        return collected

    def _get_ready_tasks(
        self,
        subgraph: TaskDAG,
        pending: set[str],
        completed: set[str],
    ) -> List[TaskNode]:
        ready: List[TaskNode] = []
        for task_id in pending:
            task = subgraph.get_node(task_id)
            dependencies = subgraph.get_task_dependencies(task_id)
            if all(dep.task_id in completed for dep in dependencies):
                ready.append(task)
        return ready

    async def _execute_tasks_parallel(
        self,
        tasks: Iterable[TaskNode],
        subgraph: TaskDAG,
        solve_fn: AsyncSolveFn,
    ) -> List[TaskNode]:
        coros = []
        for task in tasks:
            if task.status == TaskStatus.PENDING:
                coros.append(solve_fn(task, subgraph, task.depth))
        return await asyncio.gather(*coros) if coros else []

    def _enhance_error_context(self, error: Exception, agent_type: AgentType, task: Optional[TaskNode]) -> None:
        """Enhance error with agent and task context for better debugging."""
        task_id = task.task_id if task is not None else "unknown"
        error_msg = f"[{agent_type.value.upper()}] Task '{task_id}' failed: {str(error)}"
        if hasattr(error, 'args') and error.args:
            error.args = (error_msg,) + error.args[1:]
        else:
            error.args = (error_msg,)
