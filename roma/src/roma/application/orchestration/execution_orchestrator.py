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
from roma.application.orchestration.task_node_processor import TaskNodeProcessor
from roma.application.services.agent_service_registry import AgentServiceRegistry
from roma.application.services.context_builder_service import ContextBuilderService, TaskContext
from roma.application.services.recovery_manager import RecoveryManager
from roma.application.services.event_publisher import EventPublisher
from roma.application.services.deadlock_detector import DeadlockDetector

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
        deadlock_detector: DeadlockDetector,
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
            deadlock_detector: Deadlock detector service
            knowledge_store: Optional KnowledgeStoreService for persisting results
        """
        self.graph_state_manager = graph_state_manager
        self.parallel_engine = parallel_engine
        self.agent_service_registry = agent_service_registry
        self.context_builder = context_builder
        self.recovery_manager = recovery_manager
        self.event_publisher = event_publisher
        self.execution_config = execution_config
        self.deadlock_detector = deadlock_detector
        self.knowledge_store = knowledge_store

        # Single execution state (set up in execute() method)
        self.execution_state: Optional[ExecutionState] = None

        # Initialize task node processor for state management
        # Note: We'll initialize it per execution with the actual execution state

        logger.info("ExecutionOrchestrator initialized")


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
        # Initialize execution state (single execution per orchestrator)
        self.execution_state = ExecutionState(
            execution_id=execution_id,
            overall_objective=overall_objective,
            root_task=root_task
        )

        # Create task node processor for this execution's state management
        processor = TaskNodeProcessor(
            graph_manager=self.graph_state_manager,
            execution_state=self.execution_state
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
                execution_metadata={"execution_id": self.execution_state.execution_id}
            )

            # Start deadlock monitoring
            self.deadlock_detector.start_monitoring()

            # Main execution loop
            state = self.execution_state
            while not self._is_execution_complete():
                # Check iteration limit
                if state.iterations >= self.execution_config.max_iterations:
                    logger.warning(f"Execution stopped: max iterations ({self.execution_config.max_iterations}) reached")
                    break

                # Check timeout
                if self._is_timeout_exceeded(self.execution_state):
                    logger.warning(f"Execution stopped: timeout ({self.execution_config.total_timeout}s) exceeded")
                    break


                # Replanning is now handled by ParallelExecutionEngine routing
                # NEEDS_REPLAN nodes are included in ready_nodes and routed automatically

                # Get ready nodes (now async for thread safety)
                ready_nodes = await self.graph_state_manager.get_ready_nodes()

                if not ready_nodes:
                    # No ready nodes - check if this is actually a deadlock or just temporary waiting
                    logger.debug("No ready nodes found - analyzing execution state for deadlocks")

                    # Use deadlock detector to analyze the situation
                    deadlock_reports = self.deadlock_detector.analyze_execution_state()

                    if deadlock_reports:
                        # True deadlock detected
                        logger.error(f"Deadlock detected: {len(deadlock_reports)} issues found")
                        for report in deadlock_reports:
                            logger.error(f"- {report.deadlock_type.value}: {report.description}")
                        break
                    else:
                        # Check if there are any executing nodes that might complete
                        executing_nodes = [n for n in self.graph_state_manager.get_all_nodes()
                                         if n.status == TaskStatus.EXECUTING]

                        if executing_nodes:
                            # Still have executing nodes - wait for them to complete
                            logger.debug(f"Waiting for {len(executing_nodes)} executing nodes to complete")
                            await asyncio.sleep(0.1)  # Brief pause before next iteration
                            continue
                        else:
                            # No executing nodes and no ready nodes - true stall
                            logger.warning("No ready or executing nodes - execution stalled")
                            break

                # Apply node limits
                if len(ready_nodes) > self.execution_config.max_tasks_per_level:
                    ready_nodes = ready_nodes[:self.execution_config.max_tasks_per_level]
                    logger.warning(f"Limited ready nodes to {self.execution_config.max_tasks_per_level}")

                # Execute ready nodes in parallel
                logger.info(f"Iteration {state.iterations + 1}: Processing {len(ready_nodes)} ready nodes")

                node_results = await self.parallel_engine.execute_ready_nodes(
                    ready_nodes, self.agent_service_registry, initial_context, state
                )

                # Handle results using processor
                await self._handle_node_results_with_processor(node_results, processor)

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



    async def _handle_node_results_with_processor(
        self,
        node_results: List[NodeResult],
        processor: TaskNodeProcessor
    ) -> None:
        """
        Handle results from parallel node execution using TaskNodeProcessor.

        Args:
            node_results: List of node results to handle
            processor: TaskNodeProcessor for state management (contains execution_state)
        """
        for result in node_results:
            try:
                # Use processor for state management
                await processor.process_result(result)

                # Handle any special orchestration logic (like aggregation notifications)
                await self._handle_orchestration_logic(result)

            except Exception as e:
                logger.error(f"Failed to handle result for {result.task_id}: {e}")
                # Continue processing other results despite individual failures

    async def _handle_orchestration_logic(self, result: NodeResult) -> None:
        """
        Handle orchestration-specific logic after state management.

        Args:
            result: NodeResult to handle
        """
        # Handle knowledge store persistence
        if result.action in [NodeAction.COMPLETE, NodeAction.AGGREGATE] and result.envelope:
            if self.knowledge_store:
                try:
                    node = self.graph_state_manager.get_node_by_id(result.task_id)
                    if node:
                        record = await self.knowledge_store.add_or_update_record(node, result.envelope)
                        logger.debug(f"Persisted node {result.task_id} result to KnowledgeStore: {record.task_id}")
                except Exception as e:
                    logger.error(f"Failed to persist node {result.task_id} to KnowledgeStore: {e}")

        # Handle execution history persistence
        await self._persist_execution_history(result)

    async def _persist_execution_history(self, result: NodeResult) -> None:
        """
        Persist execution history for the node result.

        Args:
            result: NodeResult to persist
        """
        try:
            execution_history_repo = self.agent_service_registry.get_execution_history_repository()
            if not execution_history_repo:
                return

            node = self.graph_state_manager.get_node_by_id(result.task_id)
            if not node:
                logger.warning(f"Node {result.task_id} not found for execution history persistence")
                return

            # Update execution status based on result action
            status_mapping = {
                NodeAction.COMPLETE: TaskStatus.COMPLETED,
                NodeAction.FAIL: TaskStatus.FAILED,
                NodeAction.AGGREGATE: TaskStatus.COMPLETED
            }

            status = status_mapping.get(result.action, node.status)

            # Extract result data and error information
            result_data = None
            error_info = None
            execution_duration_ms = None

            if result.envelope:
                result_data = {
                    "primary_output": result.envelope.extract_primary_output(),
                    "metadata": result.envelope.metadata,
                    "timestamp": result.envelope.timestamp
                }

                if result.envelope.execution_metrics:
                    # Convert seconds to milliseconds for persistence
                    execution_duration_ms = int(result.envelope.execution_metrics.execution_time * 1000)

            if result.action == NodeAction.FAIL and result.error:
                error_info = {
                    "error_message": str(result.error),
                    "error_type": type(result.error).__name__,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }

            await execution_history_repo.update_execution_status(
                task_id=result.task_id,
                status=status,
                result=result_data,
                error_info=error_info,
                execution_duration_ms=execution_duration_ms
            )

            logger.debug(f"Persisted execution history for task {result.task_id} with status {status}")

        except Exception as e:
            logger.error(f"Failed to persist execution history for task {result.task_id}: {e}")


    async def _get_cached_result(self, node_id: str) -> Optional[AnyResultEnvelope]:
        """Get cached result for a node from the current execution."""
        if self.execution_state:
            return await self.execution_state.get_cached_result(node_id)
        return None








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
        if not self.execution_state:
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

        execution_time = self.execution_state.get_execution_time()
        stats = self.execution_state.get_completion_stats()

        all_nodes = self.graph_state_manager.get_all_nodes()
        total_nodes = len(all_nodes)
        completed_nodes = stats["completed"]
        failed_nodes = stats["failed"]

        success = (failed_nodes == 0 and completed_nodes > 0 and error is None)

        # Get final result from root task if available
        final_result = await self.execution_state.get_cached_result(root_task_id)

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
        if not self.execution_state:
            return {"error": f"No execution state found for {execution_id}"}

        stats = self.execution_state.get_completion_stats()
        return {
            "execution_id": execution_id,
            "iterations": stats["iterations"],
            "total_nodes_processed": stats["total_processed"],
            "result_cache_size": stats["cached_results"],
            "completed_nodes": stats["completed"],
            "failed_nodes": stats["failed"],
            "graph_nodes": len(self.graph_state_manager.get_all_nodes()),
            "execution_time_seconds": self.execution_state.get_execution_time()
        }

    async def clear_execution_cache(self, execution_id: str) -> None:
        """Clear result cache and reset state for a specific execution."""
        if self.execution_state:
            await self.execution_state.clear_cache()
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
            # Clear execution state
            self.execution_state = None

            logger.debug(f"ExecutionOrchestrator cleanup completed for execution {execution_id}")

        except Exception as e:
            logger.warning(f"Error during ExecutionOrchestrator cleanup for {execution_id}: {e}")

    async def cleanup_all(self) -> None:
        """Clean up all execution resources."""
        try:
            self.execution_state = None
            logger.debug("ExecutionOrchestrator cleanup completed for all executions")
        except Exception as e:
            logger.warning(f"Error during ExecutionOrchestrator full cleanup: {e}")
