"""
Recursive solver for hierarchical task decomposition with depth constraints.
"""

import asyncio
import logging
import warnings
from typing import Callable, Optional, Union, Tuple, TYPE_CHECKING

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

if TYPE_CHECKING:
    from src.roma_dspy.config.schemas.root import ROMAConfig

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
        config: Optional["ROMAConfig"] = None,
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
            config: ROMAConfig instance with complete configuration
            atomizer: Module for determining task atomicity (overrides config)
            planner: Module for task decomposition (overrides config)
            executor: Module for atomic task execution (overrides config)
            aggregator: Module for result synthesis (overrides config)
            verifier: Module for result validation (overrides config)
            max_depth: Maximum recursion depth (overrides config)
            lm: Language model to use (legacy parameter)
            enable_logging: Whether to enable debug logging
            checkpoint_config: Checkpoint configuration (overrides config)
        """
        # Initialize modules based on config or defaults
        if config is not None:
            self._init_from_config(config, atomizer, planner, executor, aggregator, verifier, max_depth)
        else:
            self._init_from_parameters(atomizer, planner, executor, aggregator, verifier, max_depth, lm)

        # Initialize checkpoint system
        self.checkpoint_enabled = enable_checkpoints
        checkpoint_cfg = checkpoint_config or CheckpointConfig()
        self.checkpoint_manager = CheckpointManager(checkpoint_cfg) if enable_checkpoints else None

        # Initialize runtime
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

        self.last_dag = None  # Store last DAG for visualization

    def _init_from_config(
        self,
        config: "ROMAConfig",
        atomizer: Optional[Atomizer],
        planner: Optional[Planner],
        executor: Optional[Executor],
        aggregator: Optional[Aggregator],
        verifier: Optional[Verifier],
        max_depth: Optional[int]
    ) -> None:
        """Initialize solver from ROMAConfig."""
        # Use provided modules or create from config
        self.atomizer = atomizer or Atomizer(config=config.agents.atomizer)
        self.planner = planner or Planner(config=config.agents.planner)
        self.executor = executor or Executor(config=config.agents.executor)
        self.aggregator = aggregator or Aggregator(config=config.agents.aggregator)
        self.verifier = verifier or (Verifier(config=config.agents.verifier) if config.agents.verifier.enabled else None)

        # Use runtime config
        self.max_depth = max_depth or config.runtime.max_concurrency

    def _init_from_parameters(
        self,
        atomizer: Optional[Atomizer],
        planner: Optional[Planner],
        executor: Optional[Executor],
        aggregator: Optional[Aggregator],
        verifier: Optional[Verifier],
        max_depth: int,
        lm: Optional[dspy.LM]
    ) -> None:
        """Initialize solver from individual parameters (legacy mode)."""
        # Initialize modules with defaults if not provided
        self.atomizer = atomizer or Atomizer(lm=lm)
        self.planner = planner or Planner(lm=lm)
        self.executor = executor or Executor(lm=lm)
        self.aggregator = aggregator or Aggregator(lm=lm)
        self.verifier = verifier  # Optional, not yet implemented

        self.max_depth = max_depth

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

        # Logging is now handled by TreeVisualizer when called by user

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

            # Logging is now handled by TreeVisualizer when called by user
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

        # Apply any pending state restorations from previous recovery operations
        if self.checkpoint_manager:
            await controller.apply_pending_restorations()

        await controller.run(max_concurrency=concurrency)

        updated_task = dag.get_node(task.task_id)

        # Logging is now handled by TreeVisualizer when called by user

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
        # Create DAG if not provided
        if dag is None:
            dag = TaskDAG()
            self.last_dag = dag  # Store for visualization

        # Convert string to TaskNode if needed
        if isinstance(task, str):
            task = TaskNode(goal=task, depth=depth, max_depth=self.max_depth, execution_id=dag.execution_id)

        # If task already exists but doesn't have execution_id, set it
        if task.execution_id is None:
            task = task.model_copy(update={'execution_id': dag.execution_id})

        # Add to DAG if not already present
        if task.task_id not in dag.graph:
            dag.add_node(task)

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

    # ==================== Unified Checkpoint Coordination ====================

    async def create_unified_checkpoint(
        self,
        trigger: CheckpointTrigger,
        dag: Optional[TaskDAG] = None,
        task_context: Optional[TaskNode] = None
    ) -> Optional[str]:
        """Create a unified checkpoint capturing all system components."""
        if not self.checkpoint_manager:
            logger.debug("Checkpoint manager not available, skipping unified checkpoint")
            return None

        try:
            logger.info(f"Creating unified system checkpoint for trigger: {trigger}")

            # Use provided DAG or create a minimal one
            target_dag = dag or TaskDAG("unified_checkpoint")
            if task_context and dag is None:
                target_dag.add_node(task_context)

            # Collect comprehensive system state
            solver_config = {
                "max_depth": self.max_depth,
                "enable_logging": self.enable_logging,
                "modules_enabled": {
                    "atomizer": self.atomizer is not None,
                    "planner": self.planner is not None,
                    "executor": self.executor is not None,
                    "aggregator": self.aggregator is not None,
                    "verifier": self.verifier is not None
                }
            }

            # Collect runtime state if available
            module_states = {}
            if hasattr(self, 'runtime') and self.runtime:
                module_states["runtime"] = {
                    "total_operations": getattr(self.runtime, '_operation_count', 0),
                    "last_activity": "unified_checkpoint_creation"
                }

            # Create the unified checkpoint
            checkpoint_id = await self.checkpoint_manager.create_checkpoint(
                checkpoint_id=None,  # Let manager generate ID
                dag=target_dag,
                trigger=trigger,
                current_depth=task_context.depth if task_context else 0,
                max_depth=self.max_depth,
                solver_config=solver_config,
                module_states=module_states
            )

            logger.info(f"Created unified checkpoint: {checkpoint_id}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create unified checkpoint: {e}")
            return None

    async def restore_from_unified_checkpoint(
        self,
        checkpoint_id: str,
        strategy: Optional[str] = None
    ) -> bool:
        """Restore system state from a unified checkpoint."""
        if not self.checkpoint_manager:
            logger.error("Checkpoint manager not available for restoration")
            return False

        try:
            logger.info(f"Restoring system from unified checkpoint: {checkpoint_id}")

            # Load checkpoint
            checkpoint_data = await self.checkpoint_manager.load_checkpoint(checkpoint_id)

            # Create recovery plan
            from src.roma_dspy.types.checkpoint_types import RecoveryStrategy
            recovery_strategy = RecoveryStrategy.PARTIAL
            if strategy == "full":
                recovery_strategy = RecoveryStrategy.FULL
            elif strategy == "selective":
                recovery_strategy = RecoveryStrategy.SELECTIVE

            recovery_plan = await self.checkpoint_manager.create_recovery_plan(
                checkpoint_data,
                strategy=recovery_strategy
            )

            # Enable module state restoration
            recovery_plan.restore_module_states = True

            # Create a temporary DAG for restoration
            temp_dag = TaskDAG("restoration_target")

            # Apply recovery plan
            restored_dag = await self.checkpoint_manager.apply_recovery_plan(recovery_plan, temp_dag)

            # Restore solver configuration if available
            if checkpoint_data.solver_config:
                solver_config = checkpoint_data.solver_config
                self.max_depth = solver_config.get("max_depth", self.max_depth)
                self.enable_logging = solver_config.get("enable_logging", self.enable_logging)

            logger.info(f"Successfully restored from unified checkpoint: {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore from unified checkpoint {checkpoint_id}: {e}")
            return False

    async def list_unified_checkpoints_async(self) -> list:
        """List all available unified checkpoints (async version)."""
        if not self.checkpoint_manager:
            return []

        try:
            return await self.checkpoint_manager.list_checkpoints()
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    def list_unified_checkpoints(self) -> list:
        """List all available unified checkpoints (sync version)."""
        try:
            import asyncio
            # Try to use existing event loop or create new one
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context, can't use run_until_complete
                logger.warning("list_unified_checkpoints called from async context. Use list_unified_checkpoints_async instead.")
                return []
            except RuntimeError:
                # No running loop, safe to create one
                return asyncio.run(self.list_unified_checkpoints_async())
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []

    async def auto_recover(self, max_attempts: int = 3) -> bool:
        """Simple recovery mechanism that attempts to restore from the latest checkpoint."""
        if not self.checkpoint_manager:
            logger.error("Cannot auto-recover: checkpoint manager not available")
            return False

        try:
            logger.info("Starting auto-recovery process...")

            # Get list of available checkpoints
            checkpoints = await self.checkpoint_manager.list_checkpoints()
            if not checkpoints:
                logger.warning("No checkpoints available for recovery")
                return False

            # Sort by creation time (most recent first)
            checkpoints.sort(key=lambda x: x["created_at"], reverse=True)

            # Try to recover from checkpoints, starting with the most recent
            for attempt, checkpoint in enumerate(checkpoints[:max_attempts], 1):
                checkpoint_id = checkpoint["checkpoint_id"]
                logger.info(f"Recovery attempt {attempt}/{max_attempts}: trying checkpoint {checkpoint_id}")

                try:
                    # Validate checkpoint first
                    is_valid = await self.checkpoint_manager.validate_checkpoint(checkpoint_id)
                    if not is_valid:
                        logger.warning(f"Checkpoint {checkpoint_id} is invalid, skipping")
                        continue

                    # Attempt restoration
                    success = await self.restore_from_unified_checkpoint(checkpoint_id, strategy="partial")

                    if success:
                        logger.info(f"Successfully recovered from checkpoint {checkpoint_id}")
                        return True
                    else:
                        logger.warning(f"Failed to restore from checkpoint {checkpoint_id}")

                except Exception as e:
                    logger.warning(f"Error during recovery attempt {attempt}: {e}")
                    continue

            logger.error(f"Auto-recovery failed after {max_attempts} attempts")
            return False

        except Exception as e:
            logger.error(f"Auto-recovery process failed: {e}")
            return False

    def get_system_health(self) -> dict:
        """Get overall system health status for recovery decisions."""
        health_status = {
            "checkpoint_system": {
                "enabled": self.checkpoint_manager is not None,
                "available": self.checkpoint_manager.config.enabled if self.checkpoint_manager else False
            },
            "modules": {
                "atomizer": self.atomizer is not None,
                "planner": self.planner is not None,
                "executor": self.executor is not None,
                "aggregator": self.aggregator is not None,
                "verifier": self.verifier is not None
            },
            "configuration": {
                "max_depth": self.max_depth,
                "logging_enabled": self.enable_logging
            }
        }

        # Add checkpoint storage stats if available (without async issues)
        if self.checkpoint_manager:
            try:
                import asyncio
                # Try to use existing event loop or create new one
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an async context, skip storage stats to avoid issues
                    health_status["checkpoint_storage"] = {"note": "Stats unavailable from async context. Use get_system_health_async()"}
                except RuntimeError:
                    # No running loop, safe to create one
                    storage_stats = asyncio.run(self.checkpoint_manager.get_storage_stats())
                    health_status["checkpoint_storage"] = storage_stats
            except Exception as e:
                health_status["checkpoint_storage"] = {"error": str(e)}

        return health_status

    async def get_system_health_async(self) -> dict:
        """Get overall system health status for recovery decisions (async version)."""
        health_status = {
            "checkpoint_system": {
                "enabled": self.checkpoint_manager is not None,
                "available": self.checkpoint_manager.config.enabled if self.checkpoint_manager else False
            },
            "modules": {
                "atomizer": self.atomizer is not None,
                "planner": self.planner is not None,
                "executor": self.executor is not None,
                "aggregator": self.aggregator is not None,
                "verifier": self.verifier is not None
            },
            "configuration": {
                "max_depth": self.max_depth,
                "logging_enabled": self.enable_logging
            }
        }

        # Add checkpoint storage stats if available
        if self.checkpoint_manager:
            try:
                storage_stats = await self.checkpoint_manager.get_storage_stats()
                health_status["checkpoint_storage"] = storage_stats
            except Exception as e:
                health_status["checkpoint_storage"] = {"error": str(e)}

        return health_status

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
