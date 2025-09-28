"""Runtime helpers for module execution and DAG manipulation."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable, Iterable, List, Optional

from .dag import TaskDAG
from ..modules import Aggregator, Atomizer, Executor, Planner, Verifier
from ..signatures import SubTask, TaskNode
from ...types import ModuleResult, NodeType, TaskStatus, TokenMetrics


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
        result, duration, token_metrics, messages = self._execute_module(self.atomizer, task.goal)
        task = self._record_module_result(
            task,
            "atomizer",
            task.goal,
            {"is_atomic": result.is_atomic, "node_type": result.node_type.value},
            duration,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.set_node_type(result.node_type)
        dag.update_node(task)
        return task

    async def atomize_async(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        task = task.transition_to(TaskStatus.ATOMIZING)
        result, duration, token_metrics, messages = await self._async_execute_module(self.atomizer, task.goal)
        task = self._record_module_result(
            task,
            "atomizer",
            task.goal,
            {"is_atomic": result.is_atomic, "node_type": result.node_type.value},
            duration,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.set_node_type(result.node_type)
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
        result, duration, token_metrics, messages = self._execute_module(self.planner, task.goal)
        task = self._record_module_result(
            task,
            "planner",
            task.goal,
            {
                "subtasks": [s.model_dump() for s in result.subtasks],
                "dependencies": result.dependencies_graph,
            },
            duration,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = self._create_subtask_graph(task, dag, result)
        task = task.transition_to(TaskStatus.PLAN_DONE)
        dag.update_node(task)
        return task

    async def plan_async(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        result, duration, token_metrics, messages = await self._async_execute_module(self.planner, task.goal)
        task = self._record_module_result(
            task,
            "planner",
            task.goal,
            {
                "subtasks": [s.model_dump() for s in result.subtasks],
                "dependencies": result.dependencies_graph,
            },
            duration,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = self._create_subtask_graph(task, dag, result)
        task = task.transition_to(TaskStatus.PLAN_DONE)
        dag.update_node(task)
        return task

    def execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        result, duration, token_metrics, messages = self._execute_module(self.executor, task.goal)
        task = self._record_module_result(
            task,
            "executor",
            task.goal,
            result.output,
            duration,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    async def execute_async(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        result, duration, token_metrics, messages = await self._async_execute_module(self.executor, task.goal)
        task = self._record_module_result(
            task,
            "executor",
            task.goal,
            result.output,
            duration,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    def force_execute(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        task = task.set_node_type(NodeType.EXECUTE)
        task = task.transition_to(TaskStatus.EXECUTING)
        dag.update_node(task)
        result, duration, token_metrics, messages = self._execute_module(self.executor, task.goal)
        task = self._record_module_result(
            task,
            "executor",
            task.goal,
            result.output,
            duration,
            metadata={"forced": True, "depth": task.depth},
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    async def force_execute_async(self, task: TaskNode, dag: TaskDAG) -> TaskNode:
        task = task.set_node_type(NodeType.EXECUTE)
        task = task.transition_to(TaskStatus.EXECUTING)
        dag.update_node(task)
        result, duration, token_metrics, messages = await self._async_execute_module(self.executor, task.goal)
        task = self._record_module_result(
            task,
            "executor",
            task.goal,
            result.output,
            duration,
            metadata={"forced": True, "depth": task.depth},
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.with_result(result.output)
        dag.update_node(task)
        return task

    def aggregate(self, task: TaskNode, subgraph: Optional[TaskDAG], dag: TaskDAG) -> TaskNode:
        if task.status != TaskStatus.PLAN_DONE:
            return task
        task = task.transition_to(TaskStatus.AGGREGATING)
        subtask_results = self._collect_subtask_results(subgraph)
        result, duration, token_metrics, messages = self._execute_module(
            self.aggregator,
            original_goal=task.goal,
            subtasks_results=subtask_results,
        )
        task = self._record_module_result(
            task,
            "aggregator",
            {"original_goal": task.goal, "subtask_count": len(subtask_results)},
            result.synthesized_result,
            duration,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.with_result(result.synthesized_result)
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
        result, duration, token_metrics, messages = await self._async_execute_module(
            self.aggregator,
            original_goal=task.goal,
            subtasks_results=subtask_results,
        )
        task = self._record_module_result(
            task,
            "aggregator",
            {"original_goal": task.goal, "subtask_count": len(subtask_results)},
            result.synthesized_result,
            duration,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.with_result(result.synthesized_result)
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

    def _execute_module(self, module, *args, **kwargs):
        start_time = datetime.now()
        result = module(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()

        # Extract token metrics from module history if available
        token_metrics = None
        messages = None

        # Check both module.history and module._predictor.history
        history = None
        if hasattr(module, 'history') and module.history:
            history = module.history
        elif hasattr(module, '_predictor') and hasattr(module._predictor, 'history') and module._predictor.history:
            history = module._predictor.history

        if history:
            last_history = history[-1]
            if isinstance(last_history, dict):
                # Extract usage information
                usage = last_history.get('usage', {})
                model = last_history.get('model') or last_history.get('response_model')
                # Use the cost directly from DSPy history
                cost = last_history.get('cost')
                token_metrics = TokenMetrics.from_usage_dict(usage, model, cost)

                # Extract messages if available
                if 'messages' in last_history:
                    messages = last_history.get('messages')

        return result, duration, token_metrics, messages

    async def _async_execute_module(self, module, *args, **kwargs):
        start_time = datetime.now()
        result = await module.aforward(*args, **kwargs)
        duration = (datetime.now() - start_time).total_seconds()

        # Extract token metrics from module history if available
        token_metrics = None
        messages = None

        # Check both module.history and module._predictor.history
        history = None
        if hasattr(module, 'history') and module.history:
            history = module.history
        elif hasattr(module, '_predictor') and hasattr(module._predictor, 'history') and module._predictor.history:
            history = module._predictor.history

        if history:
            last_history = history[-1]
            if isinstance(last_history, dict):
                # Extract usage information
                usage = last_history.get('usage', {})
                model = last_history.get('model') or last_history.get('response_model')
                # Use the cost directly from DSPy history
                cost = last_history.get('cost')
                token_metrics = TokenMetrics.from_usage_dict(usage, model, cost)

                # Extract messages if available
                if 'messages' in last_history:
                    messages = last_history.get('messages')

        return result, duration, token_metrics, messages

    def _record_module_result(
        self,
        task: TaskNode,
        module_name: str,
        input_data,
        output_data,
        duration: float,
        metadata: Optional[dict] = None,
        token_metrics: Optional[TokenMetrics] = None,
        messages: Optional[list] = None,
    ) -> TaskNode:
        module_result = ModuleResult(
            module_name=module_name,
            input=input_data,
            output=output_data,
            timestamp=datetime.now(),
            duration=duration,
            metadata=metadata or {},
            token_metrics=token_metrics,
            messages=messages,
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
