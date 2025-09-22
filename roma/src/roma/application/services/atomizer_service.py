"""
Atomizer Service Implementation.

Handles atomizer agent execution to determine if tasks need decomposition.
Migrated from TaskNodeProcessor with clean separation of concerns.
"""

import logging
from typing import Any, Dict, Optional

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.agent_responses import AtomizerResult
from roma.domain.value_objects.result_envelope import ExecutionMetrics, AtomizerEnvelope
from roma.domain.interfaces.agent_service import AtomizerServiceInterface
from roma.application.services.context_builder_service import TaskContext
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager, RecoveryAction

logger = logging.getLogger(__name__)


class AtomizerService(AtomizerServiceInterface):
    """Implementation of Atomizer service for task evaluation."""

    def __init__(
        self,
        agent_runtime_service: AgentRuntimeService,
        recovery_manager: RecoveryManager
    ):
        """Initialize with required dependencies."""
        self.agent_runtime_service = agent_runtime_service
        self.recovery_manager = recovery_manager

    async def run(
        self,
        task: TaskNode,
        context: TaskContext,
        execution_id: Optional[str] = None,
        **kwargs
    ) -> NodeResult:
        """
        Run atomizer evaluation to determine if task needs decomposition.

        Args:
            task: Task to evaluate
            context: Task execution context
            execution_id: Optional execution ID for session isolation
            **kwargs: Additional parameters

        Returns:
            NodeResult with atomizer decision
        """
        try:
            start_time = context.execution_metadata.get("start_time", 0.0)

            # Get atomizer agent for task type
            atomizer_agent = await self.agent_runtime_service.get_agent(
                task.task_type, AgentType.ATOMIZER
            )

            # Execute atomizer
            logger.info(f"Running atomizer for task {task.task_id} ({task.task_type})")
            envelope = await self.agent_runtime_service.execute_agent(
                atomizer_agent, task, context, AgentType.ATOMIZER, execution_id
            )

            # Extract atomizer result
            if not envelope or not hasattr(envelope, 'result'):
                raise ValueError("Atomizer returned invalid envelope")

            atomizer_result: AtomizerResult = envelope.result
            if not isinstance(atomizer_result, AtomizerResult):
                raise ValueError(f"Expected AtomizerResult, got {type(atomizer_result)}")

            # Create execution metrics
            execution_time = context.execution_metadata.get("execution_time", 0.0)
            metrics = ExecutionMetrics(
                execution_time=execution_time,
                tokens_used=getattr(envelope, 'tokens_used', 0),
                cost_estimate=getattr(envelope, 'cost_estimate', 0.0),
                model_calls=1
            )

            # Create typed envelope
            atomizer_envelope = AtomizerEnvelope.create_success(
                result=atomizer_result,
                task_id=task.task_id,
                execution_id=context.execution_metadata.get("execution_id", "unknown"),
                agent_type=AgentType.ATOMIZER,
                execution_metrics=metrics,
                output_text=atomizer_result.reasoning
            )

            # Record success with recovery manager
            await self.recovery_manager.record_success(task.task_id)

            # Return atomizer decision without NodeAction - let execution engine decide routing
            return NodeResult(
                action=NodeAction.NOOP,  # Atomizer doesn't take action, just provides decision
                envelope=atomizer_envelope,
                agent_name=getattr(atomizer_agent, 'name', 'AtomizerAgent'),
                agent_type=AgentType.ATOMIZER.value,
                processing_time_ms=execution_time * 1000,
                metadata={
                    "atomizer_decision": atomizer_result.is_atomic,
                    "confidence": atomizer_result.confidence,
                    "reasoning": atomizer_result.reasoning,
                    "node_type": NodeType.EXECUTE if atomizer_result.is_atomic else NodeType.PLAN
                }
            )

        except Exception as e:
            logger.error(f"Atomizer failed for task {task.task_id}: {e}")

            # Handle failure through recovery manager
            recovery_result = await self.recovery_manager.handle_failure(task, e)

            if recovery_result.action == RecoveryAction.RETRY:
                return NodeResult.retry(
                    error=str(e),
                    agent_name="AtomizerAgent",
                    agent_type=AgentType.ATOMIZER.value,
                    metadata={"retry_count": recovery_result.metadata.get("retry_attempt", 1)}
                )
            else:
                return NodeResult.failure(
                    error=str(e),
                    agent_name="AtomizerAgent",
                    agent_type=AgentType.ATOMIZER.value,
                    metadata={"recovery_action": recovery_result.action.value}
                )

    def get_stats(self) -> Dict[str, Any]:
        """Get atomizer service statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "decisions_made": getattr(self, '_decisions_made', 0),
            "atomic_decisions": getattr(self, '_atomic_decisions', 0),
            "planning_decisions": getattr(self, '_planning_decisions', 0)
        }