"""
Aggregator Service Implementation.

Handles aggregator agent execution to combine results from child tasks.
Supports both full and partial aggregation based on failure thresholds.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable
from collections import deque

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.child_evaluation_result import ChildEvaluationResult
from roma.domain.value_objects.agent_responses import AggregatorResult
from roma.domain.value_objects.result_envelope import AnyResultEnvelope, ExecutionMetrics, AggregatorEnvelope
from roma.domain.interfaces.agent_service import AggregatorServiceInterface
from roma.application.services.context_builder_service import TaskContext
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager, RecoveryAction

logger = logging.getLogger(__name__)


class AggregatorService(AggregatorServiceInterface):
    """
    Implementation of Aggregator service for result synthesis.

    Manages aggregation queue and evaluation logic internally for better
    separation of concerns from the orchestrator.
    """

    def __init__(
        self,
        agent_runtime_service: AgentRuntimeService,
        recovery_manager: RecoveryManager
    ):
        """Initialize with required dependencies."""
        self.agent_runtime_service = agent_runtime_service
        self.recovery_manager = recovery_manager

        # Aggregation queue management
        self._aggregation_queue: deque[tuple[str, bool]] = deque()  # (parent_id, is_partial)

        # Callbacks for orchestrator communication
        self._get_parent_callback: Optional[Callable[[str], Optional[TaskNode]]] = None
        self._get_children_callback: Optional[Callable[[str], List[TaskNode]]] = None
        self._get_result_callback: Optional[Callable[[str], Optional[AnyResultEnvelope]]] = None
        self._transition_status_callback: Optional[Callable[[str, TaskStatus], Awaitable[None]]] = None
        self._handle_result_callback: Optional[Callable[[NodeResult], Awaitable[None]]] = None

    async def run(
        self,
        task: TaskNode,
        context: TaskContext,
        execution_id: Optional[str] = None,
        child_envelopes: List[AnyResultEnvelope] = None,
        is_partial: bool = False,
        **kwargs
    ) -> NodeResult:
        """
        Run aggregator to combine child results.

        Args:
            task: Parent task being aggregated
            context: Task context
            child_envelopes: Child result envelopes (only completed ones for partial)
            is_partial: Whether this is partial aggregation
            **kwargs: Additional parameters

        Returns:
            NodeResult with AGGREGATE action and AggregatorEnvelope
        """
        try:
            if not child_envelopes:
                raise ValueError("No child results provided for aggregation")

            # Get aggregator agent for task type
            aggregator_agent = await self.agent_runtime_service.get_agent(
                task.task_type, AgentType.AGGREGATOR
            )

            # Enhance context with aggregation metadata
            enhanced_context = self._enhance_context_for_aggregation(
                context, child_envelopes, is_partial
            )

            # Execute aggregator
            logger.info(
                f"Running aggregator for task {task.task_id} ({task.task_type}) - "
                f"{'partial' if is_partial else 'full'} aggregation with {len(child_envelopes)} results"
            )

            envelope = await self.agent_runtime_service.execute_agent(
                aggregator_agent, task, enhanced_context, AgentType.AGGREGATOR, execution_id
            )

            # Extract aggregator result
            if not envelope or not hasattr(envelope, 'result'):
                raise ValueError("Aggregator returned invalid envelope")

            aggregator_result: AggregatorResult = envelope.result
            if not isinstance(aggregator_result, AggregatorResult):
                raise ValueError(f"Expected AggregatorResult, got {type(aggregator_result)}")

            # Create execution metrics
            execution_time = context.execution_metadata.get("execution_time", 0.0)
            total_child_tokens = sum(
                getattr(child_env.execution_metrics, 'tokens_used', 0)
                for child_env in child_envelopes
            )

            metrics = ExecutionMetrics(
                execution_time=execution_time,
                tokens_used=getattr(envelope, 'tokens_used', 0) + total_child_tokens,
                cost_estimate=getattr(envelope, 'cost_estimate', 0.0),
                model_calls=1
            )

            # Create typed envelope
            aggregator_envelope = AggregatorEnvelope.create_success(
                result=aggregator_result,
                task_id=task.task_id,
                execution_id=context.execution_metadata.get("execution_id", "unknown"),
                agent_type=AgentType.AGGREGATOR,
                execution_metrics=metrics,
                output_text=aggregator_result.synthesized_result
            )

            # Record success with recovery manager
            await self.recovery_manager.record_success(task.task_id)

            return NodeResult.aggregation_result(
                envelope=aggregator_envelope,
                agent_name=getattr(aggregator_agent, 'name', 'AggregatorAgent'),
                processing_time_ms=execution_time * 1000,
                metadata={
                    "is_partial_aggregation": is_partial,
                    "child_results_count": len(child_envelopes),
                    "confidence": aggregator_result.confidence,
                    "quality_score": aggregator_result.quality_score,
                    "sources_used": aggregator_result.sources_used,
                    "gaps_identified": aggregator_result.gaps_identified
                }
            )

        except Exception as e:
            logger.error(f"Aggregator failed for task {task.task_id}: {e}")

            # Handle failure through recovery manager
            recovery_result = await self.recovery_manager.handle_failure(task, e)

            if recovery_result.action == RecoveryAction.RETRY:
                return NodeResult.retry(
                    error=str(e),
                    agent_name="AggregatorAgent",
                    agent_type=AgentType.AGGREGATOR.value,
                    metadata={
                        "retry_count": recovery_result.metadata.get("retry_attempt", 1),
                        "is_partial_aggregation": is_partial
                    }
                )
            else:
                return NodeResult.failure(
                    error=str(e),
                    agent_name="AggregatorAgent",
                    agent_type=AgentType.AGGREGATOR.value,
                    metadata={
                        "recovery_action": recovery_result.action.value,
                        "is_partial_aggregation": is_partial
                    }
                )

    def _enhance_context_for_aggregation(
        self,
        context: TaskContext,
        child_envelopes: List[AnyResultEnvelope],
        is_partial: bool
    ) -> TaskContext:
        """
        Enhance context with aggregation-specific information.

        Args:
            context: Original task context
            child_envelopes: Child result envelopes
            is_partial: Whether this is partial aggregation

        Returns:
            Enhanced TaskContext with aggregation metadata
        """
        # Extract child results for aggregator
        child_results = []
        failed_children = []

        for envelope in child_envelopes:
            if envelope.success:
                child_results.append({
                    "task_id": envelope.task_id,
                    "result": envelope.result,
                    "output_text": envelope.extract_primary_output(),
                    "confidence": getattr(envelope.result, 'confidence', 1.0),
                    "metadata": envelope.metadata
                })
            else:
                failed_children.append({
                    "task_id": envelope.task_id,
                    "error": envelope.error_message,
                    "metadata": envelope.metadata
                })

        # Create enhanced metadata
        enhanced_metadata = {
            **context.execution_metadata,
            "aggregation_type": "partial" if is_partial else "full",
            "child_results": child_results,
            "successful_children_count": len(child_results),
            "total_children_count": len(child_envelopes),
            "aggregation_timestamp": context.execution_metadata.get("execution_timestamp")
        }

        # Add failed children info for partial aggregation
        if is_partial and failed_children:
            enhanced_metadata["failed_children"] = failed_children
            enhanced_metadata["failed_children_count"] = len(failed_children)

            logger.info(
                f"Partial aggregation for {context.task.task_id}: "
                f"{len(child_results)} successful, {len(failed_children)} failed"
            )

        # Create enhanced context
        return TaskContext(
            task=context.task,
            overall_objective=context.overall_objective,
            execution_metadata=enhanced_metadata
        )

    def set_orchestrator_callbacks(
        self,
        get_parent: Callable[[str], Optional[TaskNode]],
        get_children: Callable[[str], List[TaskNode]],
        get_result: Callable[[str], Optional[AnyResultEnvelope]],
        transition_status: Callable[[str, TaskStatus], Awaitable[None]],
        handle_result: Callable[[NodeResult], Awaitable[None]]
    ) -> None:
        """
        Set callbacks for communicating with orchestrator.

        Args:
            get_parent: Callback to get parent node by ID
            get_children: Callback to get children nodes by parent ID
            get_result: Callback to get result envelope by node ID
            transition_status: Callback to transition node status
            handle_result: Callback to handle aggregation result
        """
        self._get_parent_callback = get_parent
        self._get_children_callback = get_children
        self._get_result_callback = get_result
        self._transition_status_callback = transition_status
        self._handle_result_callback = handle_result

    async def notify_child_completion(self, parent_id: str) -> None:
        """
        Notify aggregator service that a child has completed.

        Evaluates if the parent is ready for aggregation and adds to queue if needed.

        Args:
            parent_id: ID of the parent whose child completed
        """
        if not self._get_parent_callback or not self._get_children_callback:
            logger.error("Orchestrator callbacks not set - cannot evaluate parent for aggregation")
            return

        # Get parent and children nodes
        parent_node = self._get_parent_callback(parent_id)
        if not parent_node:
            logger.warning(f"Parent node {parent_id} not found for aggregation check")
            return

        child_nodes = self._get_children_callback(parent_id)
        if not child_nodes:
            logger.debug(f"Parent {parent_id} has no children")
            return

        # Use RecoveryManager to evaluate children
        evaluation_result = self.recovery_manager.evaluate_terminal_children(parent_node, child_nodes)

        if evaluation_result == ChildEvaluationResult.AGGREGATE_ALL:
            # Check if all children are actually terminal before adding to queue
            all_terminal = all(
                child.status in {TaskStatus.COMPLETED, TaskStatus.FAILED}
                for child in child_nodes
            )
            if all_terminal:
                # Check for duplicates before adding
                if (parent_id, False) not in self._aggregation_queue:
                    logger.info(f"Parent {parent_id}: Adding to aggregation queue (full aggregation)")
                    self._aggregation_queue.append((parent_id, False))  # False = not partial
                else:
                    logger.debug(f"Parent {parent_id}: Already in aggregation queue (full)")

        elif evaluation_result == ChildEvaluationResult.AGGREGATE_PARTIAL:
            # Check for duplicates before adding to aggregation queue with partial flag
            if (parent_id, True) not in self._aggregation_queue:
                logger.info(f"Parent {parent_id}: Adding to aggregation queue (partial aggregation)")
                self._aggregation_queue.append((parent_id, True))  # True = partial
            else:
                logger.debug(f"Parent {parent_id}: Already in aggregation queue (partial)")

        elif evaluation_result == ChildEvaluationResult.REPLAN:
            # Trigger replanning by transitioning parent to NEEDS_REPLAN
            logger.warning(f"Parent {parent_id}: Triggering replan due to failure threshold exceeded")
            if self._transition_status_callback:
                await self._transition_status_callback(parent_id, TaskStatus.NEEDS_REPLAN)

    async def process_aggregation_queue(self, base_context: TaskContext) -> None:
        """Process pending aggregations in the queue."""
        if not self._get_parent_callback or not self._get_children_callback or not self._get_result_callback:
            logger.error("Orchestrator callbacks not set - cannot process aggregation queue")
            return

        while self._aggregation_queue:
            parent_id, is_partial = self._aggregation_queue.popleft()

            try:
                await self._trigger_aggregation(parent_id, is_partial, base_context)
            except Exception as e:
                logger.error(f"Aggregation failed for parent {parent_id}: {e}")
                if self._transition_status_callback:
                    await self._transition_status_callback(parent_id, TaskStatus.FAILED)

    async def _trigger_aggregation(self, parent_id: str, is_partial: bool, base_context: TaskContext) -> None:
        """Trigger aggregation for a parent with completed children."""
        # Get parent node
        parent_node = self._get_parent_callback(parent_id)
        if not parent_node:
            logger.error(f"Parent node {parent_id} not found for aggregation")
            return

        # Get child result envelopes
        child_nodes = self._get_children_callback(parent_id)
        child_envelopes = []

        for child_node in child_nodes:
            result_envelope = self._get_result_callback(child_node.task_id)
            if result_envelope:
                child_envelopes.append(result_envelope)

        if not child_envelopes:
            logger.warning(f"No child results found for aggregation of parent {parent_id}")
            return

        # Filter to only successful children for partial aggregation
        if is_partial:
            child_envelopes = [env for env in child_envelopes if env.success]

        # Transition parent to AGGREGATING
        if self._transition_status_callback:
            await self._transition_status_callback(parent_id, TaskStatus.AGGREGATING)

        # Trigger aggregation via aggregator service
        logger.info(f"Triggering {'partial' if is_partial else 'full'} aggregation for parent {parent_id} with {len(child_envelopes)} child results")

        aggregation_result = await self.run(
            parent_node, base_context, child_envelopes=child_envelopes, is_partial=is_partial
        )

        # Handle aggregation result
        aggregation_result.metadata["node_id"] = parent_id
        if self._handle_result_callback:
            await self._handle_result_callback(aggregation_result)

    def get_queue_status(self) -> Dict[str, Any]:
        """Get aggregation queue status."""
        return {
            "pending_aggregations": len(self._aggregation_queue),
            "queue_items": [
                {"parent_id": parent_id, "is_partial": is_partial}
                for parent_id, is_partial in self._aggregation_queue
            ]
        }

    def clear_queue(self) -> None:
        """Clear the aggregation queue."""
        cleared_count = len(self._aggregation_queue)
        self._aggregation_queue.clear()
        logger.info(f"Cleared {cleared_count} items from aggregation queue")

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregator service statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "aggregations_performed": getattr(self, '_aggregations_performed', 0),
            "partial_aggregations": getattr(self, '_partial_aggregations', 0),
            "full_aggregations": getattr(self, '_full_aggregations', 0),
            "average_child_results": getattr(self, '_avg_child_results', 0.0)
        }