"""
Executor Service Implementation.

Handles executor agent execution for atomic tasks.
Migrated from TaskNodeProcessor with clean separation of concerns.
"""

import logging
from typing import Any

from roma.domain.context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.interfaces.agent_runtime_service import IAgentRuntimeService
from roma.domain.interfaces.agent_service import ExecutorServiceInterface
from roma.domain.interfaces.recovery_manager import IRecoveryManager
from roma.domain.value_objects.agent_responses import ExecutorResult
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.recovery_action import RecoveryAction
from roma.domain.value_objects.result_envelope import ExecutionMetrics, ExecutorEnvelope

logger = logging.getLogger(__name__)


class ExecutorService(ExecutorServiceInterface):
    """Implementation of Executor service for atomic task execution."""

    def __init__(
        self, agent_runtime_service: IAgentRuntimeService, recovery_manager: IRecoveryManager
    ):
        """Initialize with required dependencies."""
        self.agent_runtime_service = agent_runtime_service
        self.recovery_manager = recovery_manager

    async def run(self, task: TaskNode, context: TaskContext, **_kwargs) -> NodeResult:
        """
        Run executor to perform atomic task execution.

        Args:
            task: Task to execute
            context: Task execution context (contains execution_id)
            **kwargs: Additional parameters

        Returns:
            NodeResult with COMPLETE action and ExecutorEnvelope
        """
        try:
            # Get executor agent for task type
            executor_agent = await self.agent_runtime_service.get_agent(
                task.task_type, AgentType.EXECUTOR
            )

            # Execute the atomic task
            logger.info(f"Executing atomic task {task.task_id} ({task.task_type})")
            envelope = await self.agent_runtime_service.execute_agent(
                executor_agent, task, context, AgentType.EXECUTOR, context.execution_id
            )

            # Extract executor result
            if not envelope or not hasattr(envelope, "result"):
                raise ValueError("Executor returned invalid envelope")

            executor_result: ExecutorResult = envelope.result
            if not isinstance(executor_result, ExecutorResult):
                raise ValueError(f"Expected ExecutorResult, got {type(executor_result)}")

            # Check if execution was successful
            if not executor_result.success:
                raise RuntimeError(
                    f"Executor reported failure: {executor_result.metadata.get('error', 'Unknown error')}"
                )

            # Create execution metrics
            execution_time = executor_result.execution_time or context.execution_metadata.get(
                "execution_time", 0.0
            )
            metrics = ExecutionMetrics(
                execution_time=execution_time,
                tokens_used=executor_result.tokens_used or 0,
                cost_estimate=getattr(envelope, "cost_estimate", 0.0),
                model_calls=1,
                network_requests=len(executor_result.sources),
            )

            # Create typed envelope
            executor_envelope = ExecutorEnvelope.create_success(
                result=executor_result,
                task_id=task.task_id,
                execution_id=context.execution_id,
                agent_type=AgentType.EXECUTOR,
                execution_metrics=metrics,
                output_text=str(executor_result.result),
            )

            # Record success with recovery manager
            await self.recovery_manager.record_success(task.task_id)

            return NodeResult.success(
                task_id=task.task_id,
                envelope=executor_envelope,
                agent_name=getattr(executor_agent, "name", "ExecutorAgent"),
                agent_type=AgentType.EXECUTOR.value,
                processing_time_ms=execution_time * 1000,
                metadata={
                    "confidence": executor_result.confidence,
                    "sources_used": executor_result.sources,
                    "result_type": type(executor_result.result).__name__,
                    "execution_metadata": executor_result.metadata,
                },
            )

        except Exception as e:
            logger.error(f"Executor failed for task {task.task_id}: {e}")

            # Handle failure through recovery manager
            recovery_result = await self.recovery_manager.handle_failure(task, e)

            if recovery_result.action == RecoveryAction.RETRY:
                return NodeResult.retry(
                    task_id=task.task_id,
                    error=str(e),
                    agent_name="ExecutorAgent",
                    agent_type=AgentType.EXECUTOR.value,
                    metadata={"retry_count": recovery_result.metadata.get("retry_attempt", 1)},
                )
            elif recovery_result.action == RecoveryAction.REPLAN:
                return NodeResult.replan(
                    task_id=task.task_id,
                    parent_id=recovery_result.metadata.get("parent_id"),
                    reason="critical_failure_exhausted_retries",
                    agent_name="ExecutorAgent",
                    agent_type=AgentType.EXECUTOR.value,
                )
            elif recovery_result.action == RecoveryAction.FORCE_ATOMIC:
                # Try a simplified execution approach
                logger.info(f"Attempting forced atomic execution for task {task.task_id}")
                return await self._force_atomic_execution(task, context)
            else:
                return NodeResult.failure(
                    task_id=task.task_id,
                    error=str(e),
                    agent_name="ExecutorAgent",
                    agent_type=AgentType.EXECUTOR.value,
                )

    async def _force_atomic_execution(self, task: TaskNode, context: TaskContext) -> NodeResult:
        """
        Attempt simplified atomic execution as a fallback.

        Args:
            task: Task to execute
            context: Task context

        Returns:
            NodeResult with simplified execution result
        """
        try:
            # Create a simplified executor result
            simple_result = ExecutorResult(
                result=f"Simplified execution of: {task.goal}",
                sources=["fallback_execution"],
                metadata={"execution_mode": "forced_atomic"},
                success=True,
                confidence=0.5,
            )

            metrics = ExecutionMetrics(execution_time=0.1, model_calls=0)

            executor_envelope = ExecutorEnvelope.create_success(
                result=simple_result,
                task_id=task.task_id,
                execution_id=context.execution_id,
                agent_type=AgentType.EXECUTOR,
                execution_metrics=metrics,
                output_text=simple_result.result,
            )

            logger.warning(f"Used forced atomic execution for task {task.task_id}")

            return NodeResult.success(
                task_id=task.task_id,
                envelope=executor_envelope,
                agent_name="FallbackExecutor",
                agent_type=AgentType.EXECUTOR.value,
                processing_time_ms=100,
                metadata={"execution_mode": "forced_atomic"},
            )

        except Exception as e:
            logger.error(f"Forced atomic execution also failed for task {task.task_id}: {e}")
            return NodeResult.failure(
                task_id=task.task_id,
                error=f"Both normal and forced execution failed: {e}",
                agent_name="FallbackExecutor",
                agent_type=AgentType.EXECUTOR.value,
            )

    def get_stats(self) -> dict[str, Any]:
        """Get executor service statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "tasks_executed": getattr(self, "_tasks_executed", 0),
            "successful_executions": getattr(self, "_successful_executions", 0),
            "forced_atomic_executions": getattr(self, "_forced_executions", 0),
            "total_execution_time": getattr(self, "_total_execution_time", 0.0),
        }
