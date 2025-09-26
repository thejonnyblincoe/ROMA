"""
Recursive solver for hierarchical task decomposition with depth constraints.
"""

import asyncio
import logging
import warnings
from typing import Callable, Optional, Union, Tuple

import dspy

from src.roma_dspy.core.engine import TaskDAG
from src.roma_dspy.core.engine.event_loop import EventLoopController
from src.roma_dspy.core.engine.runtime import ModuleRuntime
from src.roma_dspy.core.modules import Aggregator, Atomizer, Executor, Planner, Verifier
from src.roma_dspy.core.signatures import TaskNode
from src.roma_dspy.types import TaskStatus, AgentType
from src.roma_dspy.types.checkpoint_types import CheckpointTrigger
from src.roma_dspy.types.checkpoint_models import CheckpointConfig
from src.roma_dspy.resilience.checkpoint_manager import CheckpointManager

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
        enable_logging: bool = False,
        enable_checkpoints: bool = True,
        checkpoint_config: Optional[CheckpointConfig] = None
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

        # Initialize checkpoint system - pass explicit config, no settings dependency
        self.checkpoint_enabled = enable_checkpoints
        checkpoint_cfg = checkpoint_config or CheckpointConfig()
        self.checkpoint_manager = CheckpointManager(checkpoint_cfg) if enable_checkpoints else None

        self.runtime = ModuleRuntime(
            atomizer=self.atomizer,
            planner=self.planner,
            executor=self.executor,
            aggregator=self.aggregator,
            verifier=self.verifier,
        )

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

        # Create initial checkpoint before execution
        checkpoint_id = None
        if self.checkpoint_manager:
            try:
                checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                    checkpoint_id=None,
                    dag=dag,
                    trigger=CheckpointTrigger.BEFORE_PLANNING,
                    current_depth=depth,
                    max_depth=self.max_depth,
                    solver_config={
                        'max_depth': self.max_depth,
                        'enable_logging': logger.level <= logging.DEBUG
                    }
                )
                logger.debug(f"Created initial checkpoint: {checkpoint_id}")
            except Exception as e:
                logger.warning(f"Failed to create initial checkpoint: {e}")

        try:
            # Execute based on current state
            task = await self._async_execute_state_machine(task, dag, checkpoint_id)

            logger.debug(f"Completed async_solve with status: {task.status}")
            return task
        except Exception as e:
            # Enhance error with task hierarchy context
            error_msg = f"Task '{task.task_id}' failed at depth {task.depth}: {str(e)}"
            if task.goal:
                error_msg += f"\nTask goal: {task.goal[:100]}..."

            # Add checkpoint recovery info
            if checkpoint_id and self.checkpoint_manager:
                error_msg += f"\nCheckpoint {checkpoint_id} available for recovery"
                logger.error(error_msg)
            else:
                logger.error(error_msg)

            # Re-raise with enhanced context but preserve original exception type
            enhanced_error = type(e)(error_msg)
            enhanced_error.__cause__ = e
            raise enhanced_error from e

    async def async_event_solve(
        self,
        task: Union[str, TaskNode],
        dag: Optional[TaskDAG] = None,
        depth: int = 0,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
        concurrency: int = 1,
    ) -> TaskNode:
        """Run the event-driven scheduler to solve the task graph."""

        logger.debug(
            "Starting async_event_solve for task: %s",
            task if isinstance(task, str) else task.goal,
        )

        task, dag = self._initialize_task_and_dag(task, dag, depth)

        # Pass checkpoint manager to event controller if available
        controller = EventLoopController(
            dag,
            self.runtime,
            priority_fn=priority_fn,
            checkpoint_manager=self.checkpoint_manager
        )
        await controller.run(max_concurrency=concurrency)

        updated_task = dag.get_node(task.task_id)

        logger.debug("Completed async_event_solve with status: %s", updated_task.status)
        return updated_task

    def event_solve(
        self,
        task: Union[str, TaskNode],
        dag: Optional[TaskDAG] = None,
        depth: int = 0,
        priority_fn: Optional[Callable[[TaskNode], int]] = None,
        concurrency: int = 1,
    ) -> TaskNode:
        """Synchronous wrapper around the event-driven scheduler."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("event_solve() cannot be called from a running event loop")

        return asyncio.run(
            self.async_event_solve(
                task=task,
                dag=dag,
                depth=depth,
                priority_fn=priority_fn,
                concurrency=concurrency,
            )
        )

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
            return self.runtime.force_execute(task, dag)

        # Process based on current state
        if task.status == TaskStatus.PENDING:
            logger.debug(f"Atomizing task: {task.goal[:50]}...")
            task = self.runtime.atomize(task, dag)

        if task.status == TaskStatus.ATOMIZING:
            task = self.runtime.transition_from_atomizing(task, dag)

        if task.status == TaskStatus.PLANNING:
            logger.debug(f"Planning task: {task.goal[:50]}...")
            task = self.runtime.plan(task, dag)

        if task.status == TaskStatus.EXECUTING:
            logger.debug(f"Executing task: {task.goal[:50]}...")
            task = self.runtime.execute(task, dag)
        elif task.status == TaskStatus.PLAN_DONE:
            task = self.runtime.process_subgraph(task, dag, self.solve)

        return task

    async def _async_execute_state_machine(self, task: TaskNode, dag: TaskDAG, checkpoint_id: Optional[str] = None) -> TaskNode:
        """Execute asynchronous state machine for task processing."""
        # Check for forced execution at max depth
        if task.should_force_execute():
            logger.debug(f"Force executing task at max depth: {task.depth}")
            return await self.runtime.force_execute_async(task, dag)

        # Process based on current state
        if task.status == TaskStatus.PENDING:
            logger.debug(f"Async atomizing task: {task.goal[:50]}...")
            task = await self.runtime.atomize_async(task, dag)

        if task.status == TaskStatus.ATOMIZING:
            task = self.runtime.transition_from_atomizing(task, dag)

        if task.status == TaskStatus.PLANNING:
            logger.debug(f"Async planning task: {task.goal[:50]}...")
            task = await self.runtime.plan_async(task, dag)

            # Create checkpoint after planning (expensive operation completed)
            if self.checkpoint_manager and task.status == TaskStatus.PLAN_DONE:
                try:
                    await self.checkpoint_manager.create_checkpoint(
                        checkpoint_id=f"{checkpoint_id}_after_plan" if checkpoint_id else None,
                        dag=dag,
                        trigger=CheckpointTrigger.AFTER_PLANNING,
                        current_depth=task.depth,
                        max_depth=self.max_depth
                    )
                except Exception as e:
                    logger.warning(f"Failed to create post-planning checkpoint: {e}")

        if task.status == TaskStatus.EXECUTING:
            logger.debug(f"Async executing task: {task.goal[:50]}...")
            task = await self.runtime.execute_async(task, dag)
        elif task.status == TaskStatus.PLAN_DONE:
            # Create checkpoint before aggregation (preserve completed subtasks)
            if self.checkpoint_manager:
                try:
                    await self.checkpoint_manager.create_checkpoint(
                        checkpoint_id=f"{checkpoint_id}_before_agg" if checkpoint_id else None,
                        dag=dag,
                        trigger=CheckpointTrigger.BEFORE_AGGREGATION,
                        current_depth=task.depth,
                        max_depth=self.max_depth
                    )
                except Exception as e:
                    logger.warning(f"Failed to create pre-aggregation checkpoint: {e}")

            task = await self.runtime.process_subgraph_async(task, dag, self.async_solve)

        return task

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


def event_solve(
    task: Union[str, TaskNode],
    max_depth: int = 2,
    priority_fn: Optional[Callable[[TaskNode], int]] = None,
    concurrency: int = 1,
    **kwargs,
) -> TaskNode:
    """Synchronously solve using the event-driven scheduler."""

    solver = RecursiveSolver(max_depth=max_depth, **kwargs)
    return solver.event_solve(task, priority_fn=priority_fn, concurrency=concurrency)


async def async_event_solve(
    task: Union[str, TaskNode],
    max_depth: int = 2,
    priority_fn: Optional[Callable[[TaskNode], int]] = None,
    concurrency: int = 1,
    **kwargs,
) -> TaskNode:
    """Asynchronously solve using the event-driven scheduler."""

    solver = RecursiveSolver(max_depth=max_depth, **kwargs)
    return await solver.async_event_solve(
        task,
        priority_fn=priority_fn,
        concurrency=concurrency,
    )
