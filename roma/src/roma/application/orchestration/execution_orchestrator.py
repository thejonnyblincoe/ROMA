"""
Execution Orchestrator for ROMA v2.0.

Main execution coordinator that handles:
- Main execution loop until completion
- Graph mutations (adding subtasks, triggering aggregation)
- Node result handling and state transitions
- Completion detection and deadlock handling
- Recovery and retry coordination

This is where all the orchestration logic lives - ParallelExecutionEngine
only handles concurrent execution.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
from collections import defaultdict, deque
from enum import Enum

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.child_evaluation_result import ChildEvaluationResult
from roma.domain.value_objects.config.execution_config import ExecutionConfig
from roma.domain.value_objects.result_envelope import AnyResultEnvelope
from roma.domain.value_objects.execution_result import ExecutionResult
from roma.domain.value_objects.execution_state import ExecutionState
from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine, ParallelExecutionStats
from roma.application.services.agent_service_registry import AgentServiceRegistry
from roma.application.services.context_builder_service import ContextBuilderService, TaskContext
from roma.application.services.recovery_manager import RecoveryManager
from roma.application.services.event_publisher import EventPublisher

logger = logging.getLogger(__name__)




class ExecutionOrchestrator:
    """
    Main execution orchestrator responsible for coordinating all aspects
    of task graph execution.

    Responsibilities:
    - Main execution loop until completion or limits reached
    - Handle NodeResults from ParallelExecutionEngine
    - Add subtasks to graph when planning results are received
    - Trigger aggregation when children complete
    - Manage result cache and context building
    - Handle deadlocks and recovery
    - Apply execution limits and timeouts
    """

    def __init__(
        self,
        graph_state_manager: GraphStateManager,
        parallel_engine: ParallelExecutionEngine,
        agent_service_registry: AgentServiceRegistry,
        context_builder: ContextBuilderService,
        recovery_manager: RecoveryManager,
        event_publisher: EventPublisher,
        execution_config: ExecutionConfig,
        knowledge_store: Optional[Any] = None
    ):
        """
        Initialize ExecutionOrchestrator.

        Args:
            graph_state_manager: State management for task graph
            parallel_engine: Engine for parallel node execution
            agent_service_registry: Registry managing all agent services
            context_builder: Service for building execution contexts
            recovery_manager: Service for error handling and recovery
            event_publisher: Event publisher for execution tracing
            execution_config: Configuration for execution limits and behavior
            knowledge_store: Optional KnowledgeStoreService for persisting results
        """
        self.graph_state_manager = graph_state_manager
        self.parallel_engine = parallel_engine
        self.agent_service_registry = agent_service_registry
        self.context_builder = context_builder
        self.recovery_manager = recovery_manager
        self.event_publisher = event_publisher
        self.execution_config = execution_config
        self.knowledge_store = knowledge_store

        # Per-execution state tracking
        self.execution_states: Dict[str, ExecutionState] = {}

        logger.info("ExecutionOrchestrator initialized")

        # Setup service callbacks after initialization
        self._setup_aggregator_service_callbacks()
        self._setup_plan_modifier_service_callbacks()

    def _setup_aggregator_service_callbacks(self) -> None:
        """Setup callbacks for AggregatorService to communicate with orchestrator."""
        aggregator_service = self.agent_service_registry.get_aggregator_service()
        aggregator_service.set_orchestrator_callbacks(
            get_parent=self.graph_state_manager.get_node_by_id,
            get_children=self.graph_state_manager.get_children_nodes,
            get_result=self._get_cached_result,
            transition_status=self.graph_state_manager.transition_node_status,
            handle_result=self._handle_aggregator_result
        )

    def _setup_plan_modifier_service_callbacks(self) -> None:
        """Setup callbacks for PlanModifierService to communicate with orchestrator."""
        plan_modifier_service = self.agent_service_registry.get_plan_modifier_service()
        plan_modifier_service.set_orchestrator_callbacks(
            get_all_nodes=self.graph_state_manager.get_all_nodes,
            get_children=self.graph_state_manager.get_children_nodes,
            remove_node=self.graph_state_manager.remove_node,
            transition_status=self.graph_state_manager.transition_node_status,
            handle_replan_result=self._handle_replan_result,
            context_builder=self.context_builder
        )

    async def execute(self, root_task: TaskNode, overall_objective: str, execution_id: str) -> ExecutionResult:
        """
        Execute the task graph starting from root task until completion.

        Args:
            root_task: Root task node to start execution
            overall_objective: Overall objective for context building
            execution_id: Unique execution identifier that propagates through the entire execution

        Returns:
            ExecutionResult with execution statistics and final result

        Raises:
            Exception: If critical execution errors occur
        """
        # Initialize per-execution state
        self.execution_states[execution_id] = ExecutionState(
            execution_id=execution_id,
            overall_objective=overall_objective,
            root_task=root_task
        )

        logger.info(f"Starting execution orchestration [{execution_id}] for root task: {root_task.task_id}")

        try:
            # Add root task to graph
            await self.graph_state_manager.add_node(root_task)

            # Check for cycles before starting execution
            if self.graph_state_manager.has_cycles():
                raise ValueError(f"Dependency cycle detected in task graph for execution {execution_id}. Cannot execute.")

            # Build initial context
            initial_context = await self.context_builder.build_context(
                task=root_task,
                overall_objective=overall_objective,
                execution_metadata={"execution_id": execution_id}
            )

            # Main execution loop
            state = self.execution_states[execution_id]
            while not self._is_execution_complete():
                # Check iteration limit
                if state.iterations >= self.execution_config.max_iterations:
                    logger.warning(f"Execution stopped: max iterations ({self.execution_config.max_iterations}) reached")
                    break

                # Check timeout
                if self._is_timeout_exceeded(state):
                    logger.warning(f"Execution stopped: timeout ({self.execution_config.total_timeout}s) exceeded")
                    break

                # Handle nodes that need replanning first
                plan_modifier_service = self.agent_service_registry.get_plan_modifier_service()
                await plan_modifier_service.process_replanning_nodes(initial_context)

                # Get ready nodes
                ready_nodes = self.graph_state_manager.get_ready_nodes()

                if not ready_nodes:
                    # No ready nodes - check for deadlock or pending aggregations
                    aggregator_service = self.agent_service_registry.get_aggregator_service()
                    queue_status = aggregator_service.get_queue_status()
                    if queue_status["pending_aggregations"] > 0:
                        await aggregator_service.process_aggregation_queue(initial_context)
                        continue
                    else:
                        logger.warning("No ready nodes found - possible deadlock")
                        break

                # Apply node limits
                if len(ready_nodes) > self.execution_config.max_tasks_per_level:
                    ready_nodes = ready_nodes[:self.execution_config.max_tasks_per_level]
                    logger.warning(f"Limited ready nodes to {self.execution_config.max_tasks_per_level}")

                # Execute ready nodes in parallel
                logger.info(f"Iteration {state.iterations + 1}: Processing {len(ready_nodes)} ready nodes")

                node_results = await self.parallel_engine.execute_ready_nodes(
                    ready_nodes, self.agent_service_registry, initial_context, execution_id
                )

                # Handle results
                await self._handle_node_results(node_results, overall_objective, execution_id)

                # Process any triggered aggregations
                aggregator_service = self.agent_service_registry.get_aggregator_service()
                await aggregator_service.process_aggregation_queue(initial_context)

                await state.increment_iteration()
                await state.add_processed_nodes(len(ready_nodes))

                logger.debug(f"Iteration {state.iterations} completed")

            # Calculate final results
            return await self._create_execution_result(root_task.task_id, execution_id)

        except Exception as e:
            logger.error(f"Execution orchestration failed: {e}")
            return await self._create_execution_result(root_task.task_id, execution_id, error=str(e))
        finally:
            # Clean up execution state
            await self.cleanup_execution(execution_id)

    async def _handle_node_results(
        self,
        node_results: List[NodeResult],
        overall_objective: str,
        execution_id: str
    ) -> None:
        """
        Handle results from parallel node execution.

        Args:
            node_results: List of NodeResult objects from parallel execution
            overall_objective: Overall objective for context building
            execution_id: ID of the execution being handled
        """
        for result in node_results:
            try:
                await self._handle_single_node_result(result, overall_objective, execution_id)
            except Exception as e:
                logger.error(f"Failed to handle node result {result}: {e}")
                # Continue processing other results

    async def _handle_single_node_result(
        self,
        result: NodeResult,
        overall_objective: str,
        execution_id: str
    ) -> None:
        """
        Handle a single node result based on its action.

        Args:
            result: NodeResult to handle
            overall_objective: Overall objective for context
            execution_id: ID of the execution being handled
        """
        if result.action == NodeAction.ADD_SUBTASKS:
            await self._handle_add_subtasks(result, overall_objective, execution_id)

        elif result.action == NodeAction.COMPLETE:
            await self._handle_node_completion(result, execution_id)

        elif result.action == NodeAction.AGGREGATE:
            await self._handle_aggregation_complete(result, execution_id)

        elif result.action == NodeAction.FAIL:
            await self._handle_node_failure(result, execution_id)

        elif result.action == NodeAction.RETRY:
            await self._handle_node_retry(result)

        elif result.action == NodeAction.REPLAN:
            await self._handle_replan(result, overall_objective, execution_id)

        else:
            logger.warning(f"Unhandled node action: {result.action}")

    async def _handle_add_subtasks(self, result: NodeResult, overall_objective: str, execution_id: str) -> None:
        """Handle ADD_SUBTASKS action by adding subtasks to graph."""
        if not result.new_nodes:
            logger.warning("ADD_SUBTASKS action but no new_nodes provided")
            return

        # Apply subtask limits
        subtasks = result.new_nodes
        if len(subtasks) > self.execution_config.max_subtasks_per_node:
            logger.warning(f"Limiting subtasks from {len(subtasks)} to {self.execution_config.max_subtasks_per_node}")
            subtasks = subtasks[:self.execution_config.max_subtasks_per_node]

        parent_node_id = None

        # Add subtasks to graph and track parent-child relationships
        for subtask in subtasks:
            await self.graph_state_manager.add_node(subtask)

            if subtask.parent_id:
                parent_node_id = subtask.parent_id

        # Cache planner result envelope if available
        if result.envelope and parent_node_id:
            # Get execution_id from result metadata or find by node
            execution_id = result.metadata.get("execution_id")
            if execution_id and execution_id in self.execution_states:
                state = self.execution_states[execution_id]
                await state.cache_result(parent_node_id, result.envelope)
            else:
                logger.warning(f"Could not find execution state for caching result of {parent_node_id}")

        logger.info(f"Added {len(subtasks)} subtasks for parent {parent_node_id}")

    async def _get_cached_result(self, node_id: str) -> Optional[AnyResultEnvelope]:
        """Get cached result for a node from any active execution."""
        for state in self.execution_states.values():
            result = await state.get_cached_result(node_id)
            if result:
                return result
        return None

    async def _handle_aggregator_result(self, result: NodeResult) -> None:
        """Handle result from aggregator service."""
        # Find execution_id from result metadata or use first available
        execution_id = result.metadata.get("execution_id")
        if not execution_id and self.execution_states:
            execution_id = next(iter(self.execution_states.keys()))

        if execution_id:
            await self._handle_single_node_result(result, "Aggregation completed", execution_id)
        else:
            logger.warning("Could not determine execution_id for aggregator result")

    async def _handle_node_completion(self, result: NodeResult, execution_id: str) -> None:
        """Handle COMPLETE action by caching result and checking for aggregation."""
        node_id = result.metadata.get("node_id")
        if not node_id:
            logger.warning("COMPLETE action but no node_id in metadata")
            return

        # Mark node as completed and transition status
        await self.graph_state_manager.transition_node_status(node_id, TaskStatus.COMPLETED)

        # Update execution state
        if execution_id in self.execution_states:
            state = self.execution_states[execution_id]
            await state.mark_node_completed(node_id)

            # Cache result envelope
            if result.envelope:
                await state.cache_result(node_id, result.envelope)
        else:
            logger.warning(f"Could not find execution state {execution_id} for node completion {node_id}")

        # Persist to KnowledgeStore if available
        if self.knowledge_store and result.envelope:
            completed_node = self.graph_state_manager.get_node_by_id(node_id)
            if completed_node:
                try:
                    record = await self.knowledge_store.add_or_update_record(completed_node, result.envelope)
                    logger.debug(f"Persisted node {node_id} result to KnowledgeStore: {record.task_id}")
                except Exception as e:
                    logger.error(f"Failed to persist node {node_id} to KnowledgeStore: {e}")

        # Check if parent needs aggregation using graph
        completed_node = self.graph_state_manager.get_node_by_id(node_id)
        if completed_node and completed_node.parent_id:
            parent_id = completed_node.parent_id
            aggregator_service = self.agent_service_registry.get_aggregator_service()
            await aggregator_service.notify_child_completion(parent_id)

        logger.debug(f"Node {node_id} completed successfully")

    async def _handle_aggregation_complete(self, result: NodeResult, execution_id: str) -> None:
        """Handle AGGREGATE action by marking parent as completed."""
        parent_id = result.metadata.get("node_id")
        if not parent_id:
            logger.warning("AGGREGATE action but no node_id in metadata")
            return

        # Mark parent as completed
        await self.graph_state_manager.transition_node_status(parent_id, TaskStatus.COMPLETED)

        # Update execution state
        if execution_id in self.execution_states:
            state = self.execution_states[execution_id]
            await state.mark_node_completed(parent_id)

            # Cache aggregated result
            if result.envelope:
                await state.cache_result(parent_id, result.envelope)
        else:
            logger.warning(f"Could not find execution state {execution_id} for aggregation completion {parent_id}")

        # Persist aggregated result to KnowledgeStore if available
        if self.knowledge_store and result.envelope:
            parent_node = self.graph_state_manager.get_node_by_id(parent_id)
            if parent_node:
                try:
                    record = await self.knowledge_store.add_or_update_record(parent_node, result.envelope)
                    logger.debug(f"Persisted aggregated result for parent {parent_id} to KnowledgeStore: {record.task_id}")
                except Exception as e:
                    logger.error(f"Failed to persist aggregated result for parent {parent_id} to KnowledgeStore: {e}")

        logger.info(f"Parent {parent_id} aggregation completed")

    async def _handle_node_failure(self, result: NodeResult, execution_id: str) -> None:
        """Handle FAIL action by marking node as failed."""
        node_id = result.metadata.get("node_id")
        if not node_id:
            logger.warning("FAIL action but no node_id in metadata")
            return

        # Check current status to avoid double-fail transitions
        current_node = self.graph_state_manager.get_node_by_id(node_id)
        if current_node and current_node.status == TaskStatus.FAILED:
            logger.debug(f"Node {node_id} already FAILED, skipping transition")
            if execution_id in self.execution_states:
                state = self.execution_states[execution_id]
                await state.mark_node_failed(node_id)
            return

        await self.graph_state_manager.transition_node_status(node_id, TaskStatus.FAILED)

        # Update execution state
        if execution_id in self.execution_states:
            state = self.execution_states[execution_id]
            await state.mark_node_failed(node_id)
        else:
            logger.warning(f"Could not find execution state {execution_id} for node failure {node_id}")

        logger.error(f"Node {node_id} failed: {result.error}")

    async def _handle_node_retry(self, result: NodeResult) -> None:
        """Handle RETRY action by resetting node to pending."""
        node_id = result.metadata.get("node_id")
        if not node_id:
            logger.warning("RETRY action but no node_id in metadata")
            return

        await self.graph_state_manager.transition_node_status(node_id, TaskStatus.PENDING)
        logger.info(f"Node {node_id} scheduled for retry")

    async def _handle_replan(self, result: NodeResult, overall_objective: str, execution_id: str) -> None:
        """Handle REPLAN action by replacing existing plan."""
        parent_id = result.metadata.get("node_id")
        if not parent_id:
            logger.warning("REPLAN action but no node_id in metadata")
            return

        # Remove old failed children first (defensive - should be done by PlanModifierService)
        existing_children = self.graph_state_manager.graph.get_children_nodes(parent_id)
        failed_children = [child for child in existing_children if child.status == TaskStatus.FAILED]

        if failed_children:
            logger.info(f"Cleaning up {len(failed_children)} failed children during replan for {parent_id}")
            for failed_child in failed_children:
                await self.graph_state_manager.graph.remove_node(failed_child.task_id)

        # Add new subtasks from replan
        await self._handle_add_subtasks(result, overall_objective, execution_id)

        # Update parent's replan count in metadata using thread-safe method
        parent_node = self.graph_state_manager.get_node_by_id(parent_id)
        if parent_node:
            replan_count = parent_node.metadata.get("replan_count", 0) + 1
            await self.graph_state_manager.graph.update_node_metadata(
                parent_id, {"replan_count": replan_count}
            )

        logger.info(f"Node {parent_id} replanned with {len(result.new_nodes or [])} new subtasks")

    async def _handle_replan_result(self, result: NodeResult) -> None:
        """Handle replan result from plan modifier service."""
        # Find execution_id from result metadata or use first available
        execution_id = result.metadata.get("execution_id")
        if not execution_id and self.execution_states:
            execution_id = next(iter(self.execution_states.keys()))

        if execution_id:
            await self._handle_replan(result, "Replan completed", execution_id)
        else:
            logger.warning("Could not determine execution_id for replan result")

    def _is_execution_complete(self) -> bool:
        """Check if execution is complete."""
        all_nodes = self.graph_state_manager.get_all_nodes()
        if not all_nodes:
            return True

        # Execution is complete when all nodes are terminal (COMPLETED or FAILED)
        for node in all_nodes:
            if node.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED}:
                return False

        return True

    def _is_timeout_exceeded(self, state: ExecutionState) -> bool:
        """Check if total execution timeout is exceeded."""
        elapsed = state.get_execution_time()
        return elapsed > self.execution_config.total_timeout

    async def _create_execution_result(
        self,
        root_task_id: str,
        execution_id: str,
        error: Optional[str] = None
    ) -> ExecutionResult:
        """Create final ExecutionResult."""
        # Get execution state
        state = self.execution_states.get(execution_id)
        if not state:
            logger.error(f"No execution state found for {execution_id}")
            return ExecutionResult(
                success=False,
                total_nodes=0,
                completed_nodes=0,
                failed_nodes=0,
                execution_time_seconds=0.0,
                iterations=0,
                final_result=None,
                error_details=[{"type": "execution_state_error", "message": f"No state found for {execution_id}"}]
            )

        execution_time = state.get_execution_time()
        stats = state.get_completion_stats()

        all_nodes = self.graph_state_manager.get_all_nodes()
        total_nodes = len(all_nodes)
        completed_nodes = stats["completed"]
        failed_nodes = stats["failed"]

        success = (failed_nodes == 0 and completed_nodes > 0 and error is None)

        # Get final result from root task if available
        final_result = await state.get_cached_result(root_task_id)

        error_details = []
        if error:
            error_details.append({"type": "orchestration_error", "message": error})

        logger.info(
            f"Execution completed: {completed_nodes}/{total_nodes} nodes completed, "
            f"{failed_nodes} failed, {stats['iterations']} iterations, "
            f"{execution_time:.2f}s total time"
        )

        return ExecutionResult(
            success=success,
            total_nodes=total_nodes,
            completed_nodes=completed_nodes,
            failed_nodes=failed_nodes,
            execution_time_seconds=execution_time,
            iterations=stats["iterations"],
            final_result=final_result,
            error_details=error_details
        )

    def get_orchestration_metrics(self, execution_id: str) -> Dict[str, Any]:
        """Get detailed orchestration metrics for a specific execution."""
        state = self.execution_states.get(execution_id)
        if not state:
            return {"error": f"No execution state found for {execution_id}"}

        stats = state.get_completion_stats()
        return {
            "execution_id": execution_id,
            "iterations": stats["iterations"],
            "total_nodes_processed": stats["total_processed"],
            "result_cache_size": stats["cached_results"],
            "completed_nodes": stats["completed"],
            "failed_nodes": stats["failed"],
            "graph_nodes": len(self.graph_state_manager.get_all_nodes()),
            "pending_aggregations": self.agent_service_registry.get_aggregator_service().get_queue_status()["pending_aggregations"],
            "execution_time_seconds": state.get_execution_time()
        }

    async def clear_execution_cache(self, execution_id: str) -> None:
        """Clear result cache and reset state for a specific execution."""
        state = self.execution_states.get(execution_id)
        if state:
            await state.clear_cache()
            logger.info(f"ExecutionOrchestrator cache cleared for execution {execution_id}")
        else:
            logger.warning(f"No execution state found for {execution_id} to clear")

    async def cleanup_execution(self, execution_id: str) -> None:
        """
        Clean up resources for a specific execution.

        Args:
            execution_id: ID of the execution to clean up
        """
        try:
            # Remove execution state
            if execution_id in self.execution_states:
                del self.execution_states[execution_id]

            logger.debug(f"ExecutionOrchestrator cleanup completed for execution {execution_id}")

        except Exception as e:
            logger.warning(f"Error during ExecutionOrchestrator cleanup for {execution_id}: {e}")

    async def cleanup_all(self) -> None:
        """Clean up all execution resources."""
        try:
            self.execution_states.clear()
            logger.debug("ExecutionOrchestrator cleanup completed for all executions")
        except Exception as e:
            logger.warning(f"Error during ExecutionOrchestrator full cleanup: {e}")