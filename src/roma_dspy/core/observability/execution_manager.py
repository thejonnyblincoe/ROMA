"""
Execution observability manager for ROMA-DSPy.

Handles all observability concerns for task execution:
- PostgreSQL initialization and lifecycle
- Execution record creation and updates
- DSPy settings configuration for tracing
- MLflow coordination
- Execution context setup for LM trace persistence

This manager follows SRP by centralizing all observability setup/teardown logic,
previously scattered across RecursiveSolver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Any

import dspy
from loguru import logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

if TYPE_CHECKING:
    from roma_dspy.core.engine.dag import TaskDAG
    from roma_dspy.core.signatures import TaskNode
    from roma_dspy.core.storage import PostgresStorage
    from roma_dspy.core.observability import MLflowManager
    from roma_dspy.config.schemas.root import ROMAConfig
    from roma_dspy.core.engine.runtime import ModuleRuntime
    from roma_dspy.types import TaskStatus, AgentType, ExecutionStatus


def add_span_attribute(task: "TaskNode", agent_type: "AgentType") -> None:
    """
    Add ROMA-specific attributes to current MLflow span (if available).

    Phase 3 (Layer 3): Rich metadata for filtering and analysis.
    This function is safe to call even if MLflow is not available or
    no span is active - it gracefully degrades.

    Args:
        task: Current task node
        agent_type: Type of agent being executed
    """
    if not MLFLOW_AVAILABLE:
        return

    try:
        span = mlflow.get_current_active_span()

        if span:
            # Build attributes dict
            attributes = {
                "roma.execution_id": task.execution_id or "unknown",
                "roma.task_id": task.task_id,
                "roma.parent_task_id": task.parent_id or "root",
                "roma.depth": task.depth,
                "roma.max_depth": task.max_depth,
                "roma.status": task.status.value,
                "roma.module": agent_type.value,
            }

            # Add optional attributes
            if task.task_type:
                attributes["roma.task_type"] = task.task_type.value
            if task.node_type:
                attributes["roma.node_type"] = task.node_type.value
            if hasattr(task, 'is_atomic'):
                attributes["roma.is_atomic"] = task.is_atomic

            span.set_attributes(attributes)
            logger.debug(f"Added ROMA attributes to span for task {task.task_id[:8]}")

    except (AttributeError, Exception) as e:
        # Non-fatal: attributes are optional enhancement
        logger.debug(f"Could not add ROMA span attributes: {e}")


class ObservabilityManager:
    """
    Manages all observability setup and teardown for task execution.

    Responsibilities:
    - Initialize and manage PostgreSQL storage lifecycle
    - Create and update execution records
    - Configure DSPy settings for distributed tracing
    - Coordinate MLflow tracing
    - Setup execution context for LM trace persistence

    This class enables clean separation of observability concerns from
    execution orchestration, making both easier to test and maintain.
    """

    def __init__(
        self,
        postgres_storage: Optional["PostgresStorage"] = None,
        mlflow_manager: Optional["MLflowManager"] = None,
        runtime: Optional["ModuleRuntime"] = None
    ):
        """
        Initialize observability manager.

        Args:
            postgres_storage: PostgreSQL storage for execution persistence
            mlflow_manager: MLflow manager for experiment tracking
            runtime: Module runtime for context store access
        """
        self.postgres_storage = postgres_storage
        self.mlflow_manager = mlflow_manager
        self.runtime = runtime

    async def setup_execution(
        self,
        task: "TaskNode",
        dag: "TaskDAG",
        config: "ROMAConfig",
        depth: int = 0,
        execution_mode: str = "recursive"
    ) -> None:
        """
        Setup all observability systems for execution.

        Performs comprehensive initialization:
        1. Initialize PostgreSQL storage if not already initialized
        2. Configure DSPy settings for distributed tracing
        3. Create execution record in PostgreSQL
        4. Setup execution context for LM trace persistence

        Args:
            task: Task being executed (TaskNode or string goal)
            dag: TaskDAG with execution_id
            config: ROMA configuration
            depth: Current recursion depth
            execution_mode: "recursive" or "event_driven"
        """
        # Initialize Postgres storage if available
        if self.postgres_storage and not self.postgres_storage._initialized:
            await self._initialize_postgres()

        # Configure DSPy settings with execution_id for trace correlation
        self._configure_dspy_tracing(dag.execution_id)

        # Create execution record in Postgres
        await self._create_execution_record(task, dag, config, depth, execution_mode)

        # Set execution context for LM trace persistence
        self._setup_trace_context(dag.execution_id)

    async def finalize_execution(
        self,
        dag: "TaskDAG",
        result: "TaskNode"
    ) -> None:
        """
        Finalize all observability data for completed execution.

        Updates execution status in PostgreSQL with:
        - Final status (completed/failed)
        - Task statistics (total, completed, failed)
        - DAG snapshot for replay/analysis

        Args:
            dag: TaskDAG with execution data
            result: Final task result with status
        """
        if not self.postgres_storage:
            return

        try:
            # Import types here to avoid circular dependency
            from roma_dspy.types import TaskStatus, ExecutionStatus

            # DAG snapshot now saved via checkpoints (see checkpoint_manager)
            await self.postgres_storage.update_execution(
                execution_id=dag.execution_id,
                status=ExecutionStatus.COMPLETED.value if result.status == TaskStatus.COMPLETED else ExecutionStatus.FAILED.value,
                total_tasks=len(dag.get_all_tasks()),
                completed_tasks=len(dag.completed_tasks),
                failed_tasks=len(dag.failed_tasks)
            )

            logger.debug(f"Updated execution status for {dag.execution_id}")

        except Exception as e:
            logger.warning(f"Failed to update execution in Postgres: {e}")

    async def _initialize_postgres(self) -> None:
        """Initialize PostgreSQL storage."""
        if not self.postgres_storage:
            return

        try:
            await self.postgres_storage.initialize()
            logger.debug("Initialized PostgreSQL storage")
        except Exception as e:
            logger.warning(f"Failed to initialize PostgreSQL: {e}")

    def _configure_dspy_tracing(self, execution_id: str) -> None:
        """
        Configure DSPy settings for distributed tracing.

        Sets execution_id on DSPy settings to enable trace correlation
        across distributed components.

        Args:
            execution_id: Unique execution identifier
        """
        if not hasattr(dspy.settings, '_roma_execution_id') or dspy.settings._roma_execution_id != execution_id:
            try:
                # Try to configure DSPy with tracing and token usage tracking enabled
                # BUG FIX: trace must be a list, not a boolean (DSPy calls len(trace))
                # See: https://github.com/stanfordnlp/dspy/issues/377
                # Enable track_usage to capture token metrics via get_lm_usage()
                if hasattr(dspy.settings, 'configure'):
                    dspy.settings.configure(trace=[], track_usage=True)

                # Set execution_id as custom attribute
                dspy.settings.execution_id = execution_id
                dspy.settings._roma_execution_id = execution_id

                logger.debug(f"Configured DSPy settings with execution_id: {execution_id}")

                # PHASE 1: Set session metadata for trace grouping
                # This groups ALL traces (including DSPy autolog) by execution_id
                if MLFLOW_AVAILABLE:
                    try:
                        # Check if there's an active trace before trying to update
                        if hasattr(mlflow, 'get_current_active_span') and mlflow.get_current_active_span():
                            mlflow.update_current_trace(metadata={
                                "mlflow.trace.session": execution_id,
                                "mlflow.trace.user": "roma-dspy",
                            })
                            logger.info(f"✓ Set MLflow session metadata for execution: {execution_id}")
                        else:
                            # No active trace yet - will be set when first span is created
                            logger.debug(f"No active MLflow trace yet for {execution_id}, metadata will be set later")
                    except AttributeError as e:
                        # MLflow version too old (< 3.0) - missing update_current_trace
                        logger.warning(f"MLflow session metadata not available (requires MLflow 3.0+): {e}")
                    except Exception as e:
                        # Non-fatal: session grouping is optional enhancement
                        logger.warning(f"Could not set MLflow session metadata: {e}")

            except (AttributeError, TypeError) as e:
                # DSPy API may not support custom kwargs - log warning but continue
                logger.warning(f"DSPy settings configuration partial: {e}. Continuing without full DSPy integration.")

                # Still set execution_id as attribute if possible
                try:
                    dspy.settings.execution_id = execution_id
                    dspy.settings._roma_execution_id = execution_id
                except Exception:
                    logger.warning("Could not set execution_id on dspy.settings")

            except Exception as e:
                # Unexpected error - log but don't fail execution
                logger.error(f"Unexpected error configuring DSPy settings: {e}. Continuing without DSPy integration.")

    async def _create_execution_record(
        self,
        task: Any,
        dag: "TaskDAG",
        config: "ROMAConfig",
        depth: int,
        execution_mode: str
    ) -> None:
        """
        Create execution record in PostgreSQL.

        Args:
            task: Task being executed
            dag: TaskDAG with execution_id
            config: ROMA configuration
            depth: Current recursion depth
            execution_mode: "recursive" or "event_driven"
        """
        if not self.postgres_storage:
            return

        try:
            # Extract goal from task
            from roma_dspy.core.signatures import TaskNode

            initial_goal = task.goal if isinstance(task, TaskNode) else str(task)

            # Serialize config using ROMAConfig.to_dict() method
            config_dict = config.to_dict() if config else {}

            await self.postgres_storage.create_execution(
                execution_id=dag.execution_id,
                initial_goal=initial_goal,
                max_depth=getattr(task, 'max_depth', 2) if isinstance(task, TaskNode) else 2,
                config=config_dict,
                metadata={
                    "solver_version": "0.1.0",
                    "depth": depth,
                    "execution_mode": execution_mode
                }
            )

            logger.debug(f"Created execution record: {dag.execution_id}")

        except Exception as e:
            logger.warning(f"Failed to create execution record in Postgres: {e}")

    def _setup_trace_context(self, execution_id: str) -> None:
        """
        Setup execution context for LM trace persistence.

        Args:
            execution_id: Unique execution identifier
        """
        if not self.postgres_storage or not self.runtime:
            return

        try:
            self.runtime.context_store.set_execution_context(
                execution_id=execution_id,
                postgres_storage=self.postgres_storage
            )
            logger.debug(f"Setup LM trace context for {execution_id}")
        except Exception as e:
            logger.warning(f"Failed to setup trace context: {e}")
