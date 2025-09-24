"""
Parallel Execution Engine for ROMA v2.0.

Pure concurrency engine that executes TaskNodes in parallel with semaphore control.
Single responsibility: concurrent execution only, no orchestration logic.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.node_action import NodeAction
from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.services.agent_service_registry import AgentServiceRegistry
from roma.application.services.context_builder_service import TaskContext
from roma.application.services.dependency_validator import DependencyValidator
from roma.domain.value_objects.execution_state import ExecutionState

logger = logging.getLogger(__name__)


@dataclass
class ParallelExecutionStats:
    """Statistics for parallel execution batch."""

    nodes_requested: int
    nodes_processed: int
    successful_nodes: int
    failed_nodes: int
    execution_time_seconds: float
    error_details: Optional[List[Dict[str, Any]]] = None


class ParallelExecutionEngine:
    """
    Parallel execution engine for concurrent TaskNode processing.

    Pure concurrency responsibility:
    - Execute multiple nodes in parallel with semaphore control
    - Handle individual node execution through TaskNodeProcessor
    - Manage state transitions via GraphStateManager
    - Return structured results without orchestration logic

    NOT responsible for:
    - Main execution loops (ExecutionOrchestrator)
    - Ready node detection (GraphStateManager)
    - Graph mutations (ExecutionOrchestrator)
    - Completion detection (ExecutionOrchestrator)
    """

    def __init__(
        self,
        state_manager: GraphStateManager,
        max_concurrent_tasks: int = 10,
        dependency_validator: Optional[DependencyValidator] = None,
        recovery_manager: Optional[Any] = None,
        context_builder: Optional[Any] = None
    ):
        """
        Initialize ParallelExecutionEngine.

        Args:
            state_manager: GraphStateManager for state coordination
            max_concurrent_tasks: Maximum number of concurrent task executions
            dependency_validator: Optional dependency validator for pre-execution validation
            recovery_manager: Optional recovery manager for dependency failure handling
            context_builder: Optional context builder for per-node context enrichment
        """
        self.state_manager = state_manager
        self.max_concurrent_tasks = max_concurrent_tasks
        self.context_builder = context_builder

        # Initialize dependency validator with recovery manager integration
        self.dependency_validator = dependency_validator or DependencyValidator(
            recovery_manager=recovery_manager
        )

        # Concurrency control
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._stats_lock = asyncio.Lock()  # Thread-safe statistics updates

        # Execution statistics (protected by _stats_lock)
        self._total_batches_processed = 0
        self._total_nodes_processed = 0
        self._total_execution_time = 0.0

        logger.info(f"ParallelExecutionEngine initialized with max_concurrent_tasks={max_concurrent_tasks}")

    async def execute_ready_nodes(
        self,
        ready_nodes: List[TaskNode],
        agent_service_registry: AgentServiceRegistry,
        context: TaskContext,
        execution_state: ExecutionState
    ) -> List[NodeResult]:
        """
        Execute multiple ready nodes in parallel with semaphore control.

        This is the core method that provides parallel execution capability
        without any orchestration logic.

        Args:
            ready_nodes: List of nodes ready for execution
            agent_service_registry: Registry providing access to agent services
            context: Base execution context (will be enriched per node)
            execution_id: Optional execution ID for session isolation

        Returns:
            List of NodeResult objects with execution outcomes

        Raises:
            Exception: If critical execution errors occur
        """
        if not ready_nodes:
            return []

        start_time = datetime.now(timezone.utc)
        logger.info(f"Executing {len(ready_nodes)} nodes in parallel (max concurrent: {self.max_concurrent_tasks})")

        # Pre-execution dependency validation
        executable_nodes = await self.dependency_validator.get_executable_nodes(ready_nodes, self.state_manager.graph)

        if len(executable_nodes) < len(ready_nodes):
            skipped_count = len(ready_nodes) - len(executable_nodes)
            logger.warning(f"Dependency validation filtered out {skipped_count} nodes from execution")

        if not executable_nodes:
            logger.info("No nodes passed dependency validation - returning empty results")
            return []

        # Create tasks for parallel execution with semaphore control
        execution_tasks = []

        for node in executable_nodes:
            task = self._execute_single_node_with_semaphore(node, agent_service_registry, context, execution_state)
            execution_tasks.append(task)

        # Execute all tasks in parallel
        node_results = await asyncio.gather(*execution_tasks, return_exceptions=True)

        # Process results and handle exceptions
        processed_results = []
        successful_count = 0
        failed_count = 0
        error_details = []

        for i, result in enumerate(node_results):
            node = executable_nodes[i]

            if isinstance(result, Exception):
                logger.error(f"Node execution failed for {node.task_id}: {result}")
                # Create failure NodeResult
                failure_result = NodeResult.failure(
                    task_id=node.task_id,
                    error=str(result),
                    agent_name="parallel_execution_engine",
                    metadata={"node_id": node.task_id, "exception_type": type(result).__name__}
                )
                processed_results.append(failure_result)
                failed_count += 1
                error_details.append({
                    "node_id": node.task_id,
                    "error": str(result),
                    "exception_type": type(result).__name__
                })
            else:
                processed_results.append(result)
                if result.is_successful:
                    successful_count += 1
                else:
                    failed_count += 1
                    if result.error:
                        error_details.append({
                            "node_id": node.task_id,
                            "error": result.error,
                            "action": result.action.value
                        })

        # Update statistics with thread safety
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        async with self._stats_lock:
            self._total_batches_processed += 1
            self._total_nodes_processed += len(executable_nodes)
            self._total_execution_time += execution_time

        logger.info(
            f"Parallel execution completed: {len(processed_results)} results "
            f"({successful_count} successful, {failed_count} failed) "
            f"in {execution_time:.2f}s"
        )

        return processed_results

    async def _execute_single_node_with_semaphore(
        self,
        node: TaskNode,
        agent_service_registry: AgentServiceRegistry,
        base_context: TaskContext,
        execution_state: ExecutionState
    ) -> NodeResult:
        """
        Execute a single node with semaphore control and state transitions.

        Args:
            node: TaskNode to execute
            agent_service_registry: Registry providing access to agent services
            base_context: Base context to use for this node
            execution_id: Optional execution ID for session isolation

        Returns:
            NodeResult with execution outcome
        """
        async with self._execution_semaphore:
            start_time = datetime.now(timezone.utc)
            node_id = node.task_id

            try:
                # Handle state transitions based on current node status
                current_status = node.status

                if current_status == TaskStatus.PENDING:
                    # Normal execution path: PENDING → READY → EXECUTING
                    logger.debug(f"Transitioning PENDING node {node_id} to EXECUTING")
                    await self.state_manager.transition_node_status(node_id, TaskStatus.READY)
                    await self.state_manager.transition_node_status(node_id, TaskStatus.EXECUTING)

                elif current_status == TaskStatus.READY:
                    # Already ready: READY → EXECUTING
                    logger.debug(f"Transitioning READY node {node_id} to EXECUTING")
                    await self.state_manager.transition_node_status(node_id, TaskStatus.EXECUTING)

                elif current_status == TaskStatus.WAITING_FOR_CHILDREN:
                    # Aggregation-ready nodes: do NOT change state, let handler manage transitions
                    logger.debug(f"Processing WAITING_FOR_CHILDREN node {node_id} without state change")
                    # No state transition - handler will manage AGGREGATING → COMPLETED

                elif current_status == TaskStatus.NEEDS_REPLAN:
                    # Replanning nodes: do NOT change state, let handler manage transitions
                    logger.debug(f"Processing NEEDS_REPLAN node {node_id} without state change")
                    # No state transition - handler will manage NEEDS_REPLAN → READY

                else:
                    # Unexpected status for execution
                    logger.warning(f"Node {node_id} has unexpected status {current_status} for execution")
                    # Try to process anyway, but this might fail

                logger.debug(f"Processing node {node_id} through agent pipeline")

                # Process node through agent services
                result = await self._process_node_with_agent_services(node, agent_service_registry, base_context, execution_state)

                # Ensure node_id metadata is present for downstream handlers
                try:
                    if (not hasattr(result, 'metadata')) or (result.metadata is None) or ("node_id" not in result.metadata):
                        new_meta = dict(result.metadata or {})
                        new_meta["node_id"] = node_id
                        result = result.model_copy(update={"metadata": new_meta})
                except Exception:
                    # Best-effort; do not fail execution for metadata enrichment
                    pass

                # Log execution metrics
                execution_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                logger.debug(f"Node {node_id} processed in {execution_time:.2f}ms: {result.action}")

                return result

            except Exception as e:
                # On exception, return NodeResult.failure instead of transitioning and re-raising
                # Let TaskNodeProcessor handle state transitions
                logger.error(f"Node {node_id} execution failed: {e}")

                return NodeResult.failure(
                    task_id=node_id,
                    error=str(e),
                    agent_name="parallel_execution_engine",
                    metadata={"node_id": node_id, "exception_type": type(e).__name__}
                )

    async def _process_node_with_agent_services(
        self,
        node: TaskNode,
        agent_service_registry: AgentServiceRegistry,
        context: TaskContext,
        execution_state: ExecutionState
    ) -> NodeResult:
        """
        Process a single node through the agent services pipeline.

        This routes nodes based on their status:
        - WAITING_FOR_CHILDREN nodes ready for aggregation → AggregatorService
        - Other nodes → Atomizer pipeline (Planner/Executor)

        Args:
            node: TaskNode to process
            agent_service_registry: Registry providing access to agent services
            context: Base execution context for the node

        Returns:
            NodeResult indicating next action and any results
        """
        try:
            logger.debug(f"Processing node {node.task_id} with agent services")

            # Build per-node enriched context
            enriched_context = await self._build_per_node_context(node, context, execution_state)

            # Check if this is a replanning node FIRST
            if node.status == TaskStatus.NEEDS_REPLAN:
                logger.debug(f"Node {node.task_id} needs replanning")

                # Get children for context (all should be terminal now)
                children = self.state_manager.get_children_nodes(node.task_id)
                failed_children = [c for c in children if c.status == TaskStatus.FAILED]
                completed_children = [c for c in children if c.status == TaskStatus.COMPLETED]

                logger.info(f"Replanning {node.task_id}: {len(failed_children)} failed, "
                           f"{len(completed_children)} completed")

                # Route to plan modifier with enriched context
                plan_modifier = agent_service_registry.get_plan_modifier_service()
                return await plan_modifier.run(
                    task=node,
                    context=enriched_context,
                    failed_children=failed_children,
                    failure_reason=f"{len(failed_children)} children failed"
                )

            # Check if this is an aggregation-ready parent
            if node.status == TaskStatus.WAITING_FOR_CHILDREN:
                logger.debug(f"Node {node.task_id} ready for aggregation")

                # Get child results for aggregation
                children_nodes = self.state_manager.get_children_nodes(node.task_id)
                child_envelopes = []

                for child_node in children_nodes:
                    if child_node.status == TaskStatus.COMPLETED:
                        child_result = await execution_state.get_cached_result(child_node.task_id)
                        if child_result:
                            child_envelopes.append(child_result)

                # Route to aggregator service with proper interface
                aggregator_service = agent_service_registry.get_aggregator_service()
                return await aggregator_service.run(
                    task=node,
                    context=enriched_context,
                    execution_id=execution_state.execution_id,
                    child_envelopes=child_envelopes,
                    children=children_nodes  # Pass children for threshold evaluation
                )

            # Phase 1: Atomizer Decision for non-aggregation nodes
            atomizer_service = agent_service_registry.get_atomizer_service()
            atomizer_result = await atomizer_service.run(
                task=node,
                context=enriched_context,
                execution_id=execution_state.execution_id
            )

            # Extract node type from atomizer result metadata
            if atomizer_result.action == NodeAction.NOOP:
                # Atomizer provided decision in metadata
                if atomizer_result.metadata and "node_type" in atomizer_result.metadata:
                    node_type = atomizer_result.metadata["node_type"]
                elif hasattr(atomizer_result, 'envelope') and atomizer_result.envelope:
                    atomizer_response = atomizer_result.envelope.result
                    if hasattr(atomizer_response, 'is_atomic'):
                        node_type = NodeType.EXECUTE if atomizer_response.is_atomic else NodeType.PLAN
                    else:
                        raise ValueError("Atomizer result missing is_atomic decision")
                else:
                    raise ValueError("Atomizer result missing node_type decision")
            else:
                # Atomizer returned error/retry action, return directly
                return atomizer_result

            logger.debug(f"Atomizer decision for {node.task_id}: {node_type}")

            # Phase 2: Execute based on atomizer decision
            if node_type == NodeType.EXECUTE:
                # Execute atomic task using executor service
                executor_service = agent_service_registry.get_executor_service()
                result = await executor_service.run(
                    task=node,
                    context=enriched_context,
                    execution_id=execution_state.execution_id
                )
            elif node_type == NodeType.PLAN:
                # Execute planning task using planner service
                planner_service = agent_service_registry.get_planner_service()
                result = await planner_service.run(
                    task=node,
                    context=enriched_context,
                    execution_id=execution_state.execution_id
                )
            else:
                raise ValueError(f"Unknown node type from atomizer: {node_type}")

            logger.debug(f"Node {node.task_id} processed successfully: {result.action}")
            return result

        except Exception as e:
            logger.error(f"Node processing failed for {node.task_id}: {e}")
            # Re-raise exception to be handled by caller at higher level
            raise e

    async def _build_per_node_context(
        self,
        node: TaskNode,
        base_context: TaskContext,
        execution_state: ExecutionState
    ) -> TaskContext:
        """
        Build a per-node TaskContext using the configured ContextBuilderService.

        Falls back to the provided base_context if no builder is available.
        """
        # If no context builder configured, return base context
        if not getattr(self, "context_builder", None):
            return base_context

        # Prepare lineage-derived results
        parent_results = []
        sibling_results = []
        child_results = []

        # Parent
        try:
            if node.parent_id:
                parent_env = await execution_state.get_cached_result(node.parent_id)
                if parent_env is not None:
                    try:
                        parent_results.append(parent_env.extract_primary_output())
                    except Exception:
                        parent_results.append(str(parent_env))
        except Exception as e:
            logger.debug(f"Parent context build failed for {node.task_id}: {e}")

        # Siblings
        try:
            siblings = []
            if hasattr(self.state_manager, "graph") and self.state_manager.graph:
                siblings = self.state_manager.graph.get_siblings(node.task_id)
            for sid in siblings:
                env = await execution_state.get_cached_result(str(sid))
                if env is not None:
                    try:
                        sibling_results.append(env.extract_primary_output())
                    except Exception:
                        sibling_results.append(str(env))
        except Exception as e:
            logger.debug(f"Sibling context build failed for {node.task_id}: {e}")

        # Children (completed)
        try:
            children_nodes = self.state_manager.get_children_nodes(node.task_id)
            for c in children_nodes:
                if execution_state.is_node_completed(c.task_id):
                    env = await execution_state.get_cached_result(c.task_id)
                    if env is not None:
                        try:
                            child_results.append(env.extract_primary_output())
                        except Exception:
                            child_results.append(str(env))
        except Exception as e:
            logger.debug(f"Child context build failed for {node.task_id}: {e}")

        # Execution metadata
        exec_meta = dict(getattr(base_context, "execution_metadata", {}) or {})
        try:
            exec_meta.setdefault("execution_id", execution_state.execution_id)
        except Exception:
            pass
        from datetime import datetime, timezone
        exec_meta["execution_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Build final context via service
        return await self.context_builder.build_context(
            task=node,
            overall_objective=base_context.overall_objective,
            parent_results=parent_results or None,
            sibling_results=sibling_results or None,
            child_results=child_results or None,
            execution_metadata=exec_meta
        )

    async def get_execution_stats(self) -> ParallelExecutionStats:
        """Get execution statistics with thread safety."""
        async with self._stats_lock:
            return ParallelExecutionStats(
                nodes_requested=0,  # Only available during active execution
                nodes_processed=self._total_nodes_processed,
                successful_nodes=0,  # Only available during active execution
                failed_nodes=0,  # Only available during active execution
                execution_time_seconds=self._total_execution_time
            )

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics with thread safety."""
        async with self._stats_lock:
            return {
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "total_batches_processed": self._total_batches_processed,
                "total_nodes_processed": self._total_nodes_processed,
                "total_execution_time_seconds": self._total_execution_time,
                "average_nodes_per_batch": (
                    self._total_nodes_processed / self._total_batches_processed
                    if self._total_batches_processed > 0 else 0
                ),
                "average_batch_time_seconds": (
                    self._total_execution_time / self._total_batches_processed
                    if self._total_batches_processed > 0 else 0
                )
            }

    async def reset_stats(self) -> None:
        """Reset execution statistics with thread safety."""
        async with self._stats_lock:
            self._total_batches_processed = 0
            self._total_nodes_processed = 0
            self._total_execution_time = 0.0
        logger.info("Execution statistics reset")
