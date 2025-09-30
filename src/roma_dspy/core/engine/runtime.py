"""Runtime helpers for module execution and DAG manipulation."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from .dag import TaskDAG
from ..modules import Aggregator, Atomizer, Executor, Planner, Verifier
from ..signatures import SubTask, TaskNode
from ...types import ModuleResult, NodeType, TaskStatus, TokenMetrics


SolveFn = Callable[[TaskNode, TaskDAG, int], TaskNode]
AsyncSolveFn = Callable[[TaskNode, TaskDAG, int], Awaitable[TaskNode]]


class ContextStore:
    """Thread-safe storage for task execution contexts with O(1) lookup."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._lock = asyncio.Lock()
        # Map subgraph_id -> {index -> task_id}
        self._index_maps: Dict[str, Dict[int, str]] = {}

    async def store_result(self, task_id: str, result: str) -> None:
        """
        Store task result in a thread-safe manner.

        Args:
            task_id: Unique task identifier
            result: Task execution result
        """
        async with self._lock:
            self._store[task_id] = result

    def get_result(self, task_id: str) -> Optional[str]:
        """
        Retrieve task result with O(1) lookup.

        Args:
            task_id: Unique task identifier

        Returns:
            Task result or None if not found
        """
        return self._store.get(task_id)

    def register_index_mapping(self, subgraph_id: str, index: int, task_id: str) -> None:
        """
        Register mapping between subtask index and task_id for a subgraph.

        Args:
            subgraph_id: ID of the subgraph
            index: Integer index of subtask in the list (0-based)
            task_id: Actual task ID
        """
        if subgraph_id not in self._index_maps:
            self._index_maps[subgraph_id] = {}
        self._index_maps[subgraph_id][index] = task_id

    def get_task_id_from_index(self, subgraph_id: str, index: int) -> Optional[str]:
        """
        Get task_id from subtask index within a subgraph.

        Args:
            subgraph_id: ID of the subgraph
            index: Integer index of subtask

        Returns:
            Task ID or None if not found
        """
        return self._index_maps.get(subgraph_id, {}).get(index)

    def get_context_for_dependencies(self, dep_ids: List[str]) -> str:
        """
        Build context string from dependency task results.

        Args:
            dep_ids: List of dependency task IDs

        Returns:
            Formatted context string with all dependency results
        """
        contexts = []
        for dep_id in dep_ids:
            result = self.get_result(dep_id)
            if result:
                contexts.append(f"[Task {dep_id[:8]}]: {result}")
        return "\n\n".join(contexts) if contexts else ""

    def get_context_for_dependency_indices(
        self,
        subgraph_id: str,
        dep_indices: List[str]
    ) -> str:
        """
        Build context string from dependency indices within a subgraph.

        Args:
            subgraph_id: ID of the subgraph
            dep_indices: List of string indices (e.g., ['0', '1'])

        Returns:
            Formatted context string with all dependency results
        """
        contexts = []
        index_map = self._index_maps.get(subgraph_id, {})

        for dep_idx_str in dep_indices:
            try:
                dep_idx = int(dep_idx_str)
                task_id = index_map.get(dep_idx)
                if task_id:
                    result = self.get_result(task_id)
                    if result:
                        contexts.append(f"[Subtask {dep_idx}]: {result}")
            except (ValueError, TypeError):
                continue

        return "\n\n".join(contexts) if contexts else ""

    def clear_subgraph(self, task_ids: List[str]) -> None:
        """
        Clear results for specific tasks to free memory.

        Args:
            task_ids: List of task IDs to remove from store
        """
        for task_id in task_ids:
            self._store.pop(task_id, None)

    def get_all_contexts(self) -> Dict[str, str]:
        """
        Get all stored contexts for inspection/debugging.

        Returns:
            Dictionary mapping task_id to result
        """
        return dict(self._store)

    def get_context_summary(self) -> str:
        """
        Get human-readable summary of all stored contexts.

        Returns:
            Formatted string showing all task results
        """
        if not self._store:
            return "No contexts stored yet."

        lines = ["Context Store Summary:", "=" * 80]
        for task_id, result in self._store.items():
            lines.append(f"\nTask ID: {task_id[:8]}...")
            lines.append(f"Result: {result[:200]}{'...' if len(result) > 200 else ''}")
            lines.append("-" * 80)
        return "\n".join(lines)

    def get_task_index(self, subgraph_id: str, task_id: str) -> Optional[int]:
        """
        Get the index of a task within its subgraph.

        Args:
            subgraph_id: ID of the subgraph
            task_id: Task ID to look up

        Returns:
            Integer index or None if not found
        """
        index_map = self._index_maps.get(subgraph_id, {})
        for idx, tid in index_map.items():
            if tid == task_id:
                return idx
        return None


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
        self.context_store = ContextStore()

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
        # Retrieve context from dependencies if they exist
        context = None
        if task.dependencies:
            dep_ids = list(task.dependencies)
            # Build context with structured, LLM-friendly formatting
            context_parts = []

            for dep_id in dep_ids:
                result_str = self.context_store.get_result(dep_id)
                if result_str:
                    # Get the dependency task for its goal
                    dep_task = None
                    try:
                        dep_task, _ = dag.find_node(dep_id)
                    except ValueError:
                        pass

                    # Try to get index for cleaner display
                    dep_idx = None
                    if task.parent_id:
                        parent = dag.get_node(task.parent_id) if task.parent_id in dag.graph else None
                        if parent and parent.subgraph_id:
                            dep_idx = self.context_store.get_task_index(parent.subgraph_id, dep_id)

                    # Format as structured context
                    if dep_idx is not None:
                        context_entry = f"<subtask id=\"{dep_idx}\">"
                        if dep_task:
                            context_entry += f"\n  <goal>{dep_task.goal}</goal>"
                        context_entry += f"\n  <output>{result_str}</output>\n</subtask>"
                        context_parts.append(context_entry)
                    else:
                        # Fallback for non-indexed tasks
                        context_entry = f"<previous_task>\n  <output>{result_str}</output>\n</previous_task>"
                        context_parts.append(context_entry)

            if context_parts:
                context = "<context>\n" + "\n\n".join(context_parts) + "\n</context>"

        # Execute with context
        result, duration, token_metrics, messages = await self._async_execute_module(
            self.executor,
            task.goal,
            context=context if context else None
        )

        # Record with context metadata
        metadata = {}
        if context:
            metadata["context_received"] = context[:200] + "..." if len(context) > 200 else context
            metadata["dependency_ids"] = list(task.dependencies)

        task = self._record_module_result(
            task,
            "executor",
            task.goal,
            result.output,
            duration,
            metadata=metadata,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.with_result(result.output)
        dag.update_node(task)

        # Store result for future dependent tasks
        await self.context_store.store_result(task.task_id, result.output)

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

        # Retrieve context from dependencies if they exist
        context = None
        if task.dependencies:
            dep_ids = list(task.dependencies)
            # Build context with structured, LLM-friendly formatting
            context_parts = []

            for dep_id in dep_ids:
                result_str = self.context_store.get_result(dep_id)
                if result_str:
                    # Get the dependency task for its goal
                    dep_task = None
                    try:
                        dep_task, _ = dag.find_node(dep_id)
                    except ValueError:
                        pass

                    # Try to get index for cleaner display
                    dep_idx = None
                    if task.parent_id:
                        parent = dag.get_node(task.parent_id) if task.parent_id in dag.graph else None
                        if parent and parent.subgraph_id:
                            dep_idx = self.context_store.get_task_index(parent.subgraph_id, dep_id)

                    # Format as structured context
                    if dep_idx is not None:
                        context_entry = f"<subtask id=\"{dep_idx}\">"
                        if dep_task:
                            context_entry += f"\n  <goal>{dep_task.goal}</goal>"
                        context_entry += f"\n  <output>{result_str}</output>\n</subtask>"
                        context_parts.append(context_entry)
                    else:
                        # Fallback for non-indexed tasks
                        context_entry = f"<previous_task>\n  <output>{result_str}</output>\n</previous_task>"
                        context_parts.append(context_entry)

            if context_parts:
                context = "<context>\n" + "\n\n".join(context_parts) + "\n</context>"

        # Execute with context
        result, duration, token_metrics, messages = await self._async_execute_module(
            self.executor,
            task.goal,
            context=context if context else None
        )

        # Record with context metadata
        metadata = {"forced": True, "depth": task.depth}
        if context:
            metadata["context_received"] = context[:200] + "..." if len(context) > 200 else context
            metadata["dependency_ids"] = list(task.dependencies)

        task = self._record_module_result(
            task,
            "executor",
            task.goal,
            result.output,
            duration,
            metadata=metadata,
            token_metrics=token_metrics,
            messages=messages,
        )
        task = task.with_result(result.output)
        dag.update_node(task)

        # Store result for future dependent tasks
        await self.context_store.store_result(task.task_id, result.output)

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

        # Create TaskNodes for each subtask
        for idx, subtask in enumerate(planner_result.subtasks):
            subtask_node = TaskNode(
                goal=subtask.goal,
                parent_id=task.task_id,
                depth=task.depth + 1,
                max_depth=task.max_depth,
            )
            subtask_nodes.append(subtask_node)

        # Build index -> task_id mapping before creating subgraph
        index_to_task_id: Dict[str, str] = {}
        for idx, subtask_node in enumerate(subtask_nodes):
            index_to_task_id[str(idx)] = subtask_node.task_id

        # Convert index-based dependencies to task_id-based dependencies
        task_id_dependencies: Optional[Dict[str, List[str]]] = None
        if planner_result.dependencies_graph:
            task_id_dependencies = {}
            for subtask_idx_str, dep_indices in planner_result.dependencies_graph.items():
                # Validate that subtask_idx is valid
                try:
                    subtask_idx = int(subtask_idx_str)
                    if subtask_idx < 0 or subtask_idx >= len(subtask_nodes):
                        continue  # Skip invalid indices
                except (ValueError, TypeError):
                    continue  # Skip non-integer keys

                # Convert subtask index to task_id
                if subtask_idx_str in index_to_task_id:
                    subtask_task_id = index_to_task_id[subtask_idx_str]
                    # Convert dependency indices to task_ids
                    dep_task_ids = []
                    for dep_idx in dep_indices:
                        # Validate dependency index
                        try:
                            dep_idx_int = int(dep_idx)
                            # Prevent self-dependencies
                            if dep_idx_int == subtask_idx:
                                continue
                            # Validate dependency is within bounds
                            if dep_idx_int < 0 or dep_idx_int >= len(subtask_nodes):
                                continue
                            if dep_idx in index_to_task_id:
                                dep_task_ids.append(index_to_task_id[dep_idx])
                        except (ValueError, TypeError):
                            continue

                    if dep_task_ids:
                        task_id_dependencies[subtask_task_id] = dep_task_ids

        # Create the subgraph with converted dependencies
        dag.create_subgraph(task.task_id, subtask_nodes, task_id_dependencies)

        # Get the subgraph_id to register index mappings
        task = dag.get_node(task.task_id)
        subgraph_id = task.subgraph_id

        # Register index -> task_id mappings in the context store
        if subgraph_id:
            for idx, subtask_node in enumerate(subtask_nodes):
                self.context_store.register_index_mapping(
                    subgraph_id,
                    idx,
                    subtask_node.task_id
                )

        updated_metrics = task.metrics.model_copy()
        updated_metrics.subtasks_created = len(subtask_nodes)
        return task.model_copy(update={"metrics": updated_metrics})

    def _collect_subtask_results(self, subgraph: Optional[TaskDAG]) -> List[SubTask]:
        collected: List[SubTask] = []
        if subgraph:
            for node in subgraph.get_all_tasks(include_subgraphs=False):
                # Retrieve context that was used for this task
                context_input = None
                if node.dependencies:
                    dep_ids = list(node.dependencies)
                    context_input = self.context_store.get_context_for_dependencies(dep_ids)

                collected.append(
                    SubTask(
                        goal=node.goal,
                        task_type=node.task_type,
                        dependencies=[],
                        result=str(node.result) if node.result else "",
                        context_input=context_input,
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
