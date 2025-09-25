"""
Aggregator Service Implementation.

Handles aggregator agent execution to combine results from child tasks.
Supports both full and partial aggregation based on failure thresholds.
"""

import logging
from collections import deque
from typing import Any

from roma.domain.context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.interfaces.agent_runtime_service import IAgentRuntimeService
from roma.domain.interfaces.agent_service import AggregatorServiceInterface
from roma.domain.interfaces.recovery_manager import IRecoveryManager
from roma.domain.value_objects.agent_responses import AggregatorResult
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.child_evaluation_result import ChildEvaluationResult
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.recovery_action import RecoveryAction
from roma.domain.value_objects.result_envelope import (
    AggregatorEnvelope,
    AnyResultEnvelope,
    ExecutionMetrics,
)

logger = logging.getLogger(__name__)


class AggregatorService(AggregatorServiceInterface):
    """
    Implementation of Aggregator service for result synthesis.

    Manages aggregation queue and evaluation logic internally for better
    separation of concerns from the orchestrator.
    """

    def __init__(
        self, agent_runtime_service: IAgentRuntimeService, recovery_manager: IRecoveryManager
    ):
        """Initialize with required dependencies."""
        self.agent_runtime_service = agent_runtime_service
        self.recovery_manager = recovery_manager

        # Aggregation queue management - parent_id entries for aggregation
        self._aggregation_queue: deque[str] = deque()

        # Statistics
        self._aggregations_performed = 0
        self._partial_aggregations = 0
        self._full_aggregations = 0

    async def run(
        self,
        task: TaskNode,
        context: TaskContext,
        child_envelopes: list[AnyResultEnvelope] = None,
        is_partial: bool = False,
        **kwargs,
    ) -> NodeResult:
        """
        Run aggregator to combine child results.

        Args:
            task: Parent task being aggregated
            context: Task context (contains execution_id)
            child_envelopes: Child result envelopes (only completed ones for partial)
            is_partial: Whether this is partial aggregation
            **kwargs: Additional parameters

        Returns:
            NodeResult with AGGREGATE action and AggregatorEnvelope
        """
        try:
            if not child_envelopes:
                raise ValueError("No child results provided for aggregation")

            # Check for children info in kwargs to evaluate threshold
            children = kwargs.get("children", [])
            if children:
                evaluation = self.recovery_manager.evaluate_terminal_children(task, children)

                if evaluation == ChildEvaluationResult.REPLAN:
                    # Don't aggregate - signal replanning needed
                    logger.info(
                        f"Failure threshold exceeded for {task.task_id}, returning replan signal"
                    )

                    return NodeResult.replan(
                        task_id=task.task_id,
                        reason="threshold_exceeded",
                        agent_name="AggregatorAgent",
                        agent_type=AgentType.AGGREGATOR.value,
                    )

                elif evaluation == ChildEvaluationResult.AGGREGATE_PARTIAL:
                    # Set partial aggregation flag
                    is_partial = True
                    logger.info(
                        f"Partial aggregation enabled for {task.task_id} due to some failed children"
                    )

                # Threshold OK - proceed with aggregation
                logger.info(
                    f"Threshold check passed for {task.task_id}, proceeding with {'partial' if is_partial else 'full'} aggregation"
                )

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
                aggregator_agent, task, enhanced_context, AgentType.AGGREGATOR, context.execution_id
            )

            # Extract aggregator result
            if not envelope or not hasattr(envelope, "result"):
                raise ValueError("Aggregator returned invalid envelope")

            aggregator_result: AggregatorResult = envelope.result
            if not isinstance(aggregator_result, AggregatorResult):
                raise ValueError(f"Expected AggregatorResult, got {type(aggregator_result)}")

            # Create execution metrics
            execution_time = context.execution_metadata.get("execution_time", 0.0)
            total_child_tokens = sum(
                getattr(child_env.execution_metrics, "tokens_used", 0)
                for child_env in child_envelopes
            )

            metrics = ExecutionMetrics(
                execution_time=execution_time,
                tokens_used=getattr(envelope, "tokens_used", 0) + total_child_tokens,
                cost_estimate=getattr(envelope, "cost_estimate", 0.0),
                model_calls=1,
            )

            # Create typed envelope
            aggregator_envelope = AggregatorEnvelope.create_success(
                result=aggregator_result,
                task_id=task.task_id,
                execution_id=context.execution_id,
                agent_type=AgentType.AGGREGATOR,
                execution_metrics=metrics,
                output_text=aggregator_result.synthesized_result,
            )

            # Record success with recovery manager
            await self.recovery_manager.record_success(task.task_id)

            # Update statistics
            self._aggregations_performed += 1
            if is_partial:
                self._partial_aggregations += 1
            else:
                self._full_aggregations += 1

            return NodeResult.aggregation_result(
                task_id=task.task_id,
                envelope=aggregator_envelope,
                agent_name=getattr(aggregator_agent, "name", "AggregatorAgent"),
                processing_time_ms=execution_time * 1000,
                metadata={
                    "is_partial_aggregation": is_partial,
                    "child_results_count": len(child_envelopes),
                    "confidence": aggregator_result.confidence,
                    "quality_score": aggregator_result.quality_score,
                    "sources_used": aggregator_result.sources_used,
                    "gaps_identified": aggregator_result.gaps_identified,
                },
            )

        except Exception as e:
            logger.error(f"Aggregator failed for task {task.task_id}: {e}")

            # Handle failure through recovery manager
            recovery_result = await self.recovery_manager.handle_failure(task, e)

            if recovery_result.action == RecoveryAction.RETRY:
                return NodeResult.retry(
                    task_id=task.task_id,
                    error=str(e),
                    agent_name="AggregatorAgent",
                    agent_type=AgentType.AGGREGATOR.value,
                    metadata={
                        "retry_count": recovery_result.metadata.get("retry_attempt", 1),
                        "is_partial_aggregation": is_partial,
                    },
                )
            elif recovery_result.action == RecoveryAction.REPLAN:
                return NodeResult.replan(
                    task_id=task.task_id,
                    parent_id=recovery_result.metadata.get("parent_id"),
                    reason="critical_failure_exhausted_retries",
                    agent_name="AggregatorAgent",
                    agent_type=AgentType.AGGREGATOR.value,
                )
            else:
                return NodeResult.failure(
                    task_id=task.task_id,
                    error=str(e),
                    agent_name="AggregatorAgent",
                    agent_type=AgentType.AGGREGATOR.value,
                    metadata={"is_partial_aggregation": is_partial},
                )

    def _enhance_context_for_aggregation(
        self, context: TaskContext, child_envelopes: list[AnyResultEnvelope], is_partial: bool
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
                child_results.append(
                    {
                        "task_id": envelope.task_id,
                        "result": envelope.result,
                        "output_text": envelope.extract_primary_output(),
                        "confidence": getattr(envelope.result, "confidence", 1.0),
                        "metadata": envelope.metadata,
                    }
                )
            else:
                failed_children.append(
                    {
                        "task_id": envelope.task_id,
                        "error": envelope.error_message,
                        "metadata": envelope.metadata,
                    }
                )

        # Create enhanced metadata
        enhanced_metadata = {
            **context.execution_metadata,
            "aggregation_type": "partial" if is_partial else "full",
            "child_results": child_results,
            "successful_children_count": len(child_results),
            "total_children_count": len(child_envelopes),
            "aggregation_timestamp": context.execution_metadata.get("execution_timestamp"),
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
            execution_id=context.execution_id,
            execution_metadata=enhanced_metadata,
        )

    def queue_aggregation(self, parent_id: str) -> None:
        """
        Queue a parent node for aggregation.

        Args:
            parent_id: ID of the parent node to aggregate
        """
        if parent_id not in self._aggregation_queue:
            logger.info(f"Queuing parent {parent_id} for aggregation")
            self._aggregation_queue.append(parent_id)
        else:
            logger.debug(f"Parent {parent_id} already queued for aggregation")

    def get_queued_parents(self) -> list[str]:
        """Get list of parent IDs queued for aggregation."""
        return list(self._aggregation_queue)

    def clear_aggregation_queue(self) -> int:
        """Clear aggregation queue and return number of cleared items."""
        cleared_count = len(self._aggregation_queue)
        self._aggregation_queue.clear()
        logger.info(f"Cleared {cleared_count} items from aggregation queue")
        return cleared_count

    def get_queue_status(self) -> dict[str, Any]:
        """Get aggregation queue status."""
        return {
            "pending_aggregations": len(self._aggregation_queue),
            "queue_items": [{"parent_id": parent_id} for parent_id in self._aggregation_queue],
        }

    def get_stats(self) -> dict[str, Any]:
        """Get aggregator service statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "aggregations_performed": self._aggregations_performed,
            "partial_aggregations": self._partial_aggregations,
            "full_aggregations": self._full_aggregations,
            "queue_size": len(self._aggregation_queue),
        }
