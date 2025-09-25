"""
Plan Modifier Service Implementation.

Handles plan modifier agent execution to adjust plans based on failures or feedback.
Used for replanning scenarios when failure thresholds are exceeded.
"""

import logging
from typing import Any

from roma.application.services.hitl_service import HITLService
from roma.domain.context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.interfaces.agent_runtime_service import IAgentRuntimeService
from roma.domain.interfaces.agent_service import PlanModifierServiceInterface
from roma.domain.interfaces.recovery_manager import IRecoveryManager
from roma.domain.value_objects.agent_responses import PlanModifierResult
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.recovery_action import RecoveryAction
from roma.domain.value_objects.result_envelope import ExecutionMetrics, PlanModifierEnvelope

logger = logging.getLogger(__name__)


class PlanModifierService(PlanModifierServiceInterface):
    """
    Implementation of Plan Modifier service for replanning scenarios.

    Manages replanning workflow internally for better separation of concerns
    from the orchestrator.
    """

    def __init__(
        self,
        agent_runtime_service: IAgentRuntimeService,
        recovery_manager: IRecoveryManager,
        hitl_service: HITLService | None = None,
    ):
        """Initialize with required dependencies."""
        self.agent_runtime_service = agent_runtime_service
        self.recovery_manager = recovery_manager
        self.hitl_service = hitl_service

        # Statistics
        self._replans_performed = 0
        self._successful_replans = 0
        self._failed_replans = 0

    async def run(
        self,
        task: TaskNode,
        context: TaskContext,
        failed_children: list[TaskNode] = None,
        failure_reason: str = None,
        **_kwargs,
    ) -> NodeResult:
        """
        Run plan modifier to adjust plans based on failures.

        Args:
            task: Original parent task
            context: Task context (contains execution_id)
            failed_children: Failed child tasks
            failure_reason: Reason for replanning
            **kwargs: Additional parameters

        Returns:
            NodeResult with REPLAN action and new_nodes populated
        """
        try:
            if not failed_children:
                failed_children = []

            if not failure_reason:
                failure_reason = "Replanning triggered due to child task failures"

            # Get plan modifier agent for task type
            plan_modifier_agent = await self.agent_runtime_service.get_agent(
                task.task_type, AgentType.PLAN_MODIFIER
            )

            # Enhance context with failure information
            enhanced_context = self._enhance_context_for_replanning(
                context, failed_children, failure_reason
            )

            # Execute plan modifier
            logger.info(
                f"Running plan modifier for task {task.task_id} ({task.task_type}) - "
                f"reason: {failure_reason}, failed children: {len(failed_children)}"
            )

            envelope = await self.agent_runtime_service.execute_agent(
                plan_modifier_agent,
                task,
                enhanced_context,
                AgentType.PLAN_MODIFIER,
                context.execution_id,
            )

            # Extract plan modifier result
            if not envelope or not hasattr(envelope, "result"):
                raise ValueError("Plan modifier returned invalid envelope")

            plan_modifier_result: PlanModifierResult = envelope.result
            if not isinstance(plan_modifier_result, PlanModifierResult):
                raise ValueError(f"Expected PlanModifierResult, got {type(plan_modifier_result)}")

            # Convert modified plan to TaskNodes
            new_subtask_nodes = await self._convert_modified_plan_to_nodes(
                plan_modifier_result, task
            )

            # Create execution metrics
            execution_time = context.execution_metadata.get("execution_time", 0.0)
            metrics = ExecutionMetrics(
                execution_time=execution_time,
                tokens_used=getattr(envelope, "tokens_used", 0),
                cost_estimate=getattr(envelope, "cost_estimate", 0.0),
                model_calls=1,
            )

            # Create typed envelope
            plan_modifier_envelope = PlanModifierEnvelope.create_success(
                result=plan_modifier_result,
                task_id=task.task_id,
                execution_id=context.execution_id,
                agent_type=AgentType.PLAN_MODIFIER,
                execution_metrics=metrics,
                output_text=f"Modified plan with {len(new_subtask_nodes)} subtasks",
            )

            # Record success with recovery manager
            await self.recovery_manager.record_success(task.task_id)

            return NodeResult(
                task_id=task.task_id,
                action=NodeAction.ADD_SUBTASKS,  # Changed from REPLAN
                envelope=plan_modifier_envelope,
                new_nodes=new_subtask_nodes,
                agent_name=getattr(plan_modifier_agent, "name", "PlanModifierAgent"),
                agent_type=AgentType.PLAN_MODIFIER.value,
                processing_time_ms=execution_time * 1000,
                metadata={
                    "failure_reason": failure_reason,
                    "failed_children_count": len(failed_children),
                    "new_subtask_count": len(new_subtask_nodes),
                    "changes_made": plan_modifier_result.changes_made,
                    "impact_assessment": plan_modifier_result.impact_assessment,
                },
            )

        except Exception as e:
            logger.error(f"Plan modifier failed for task {task.task_id}: {e}")

            # Handle failure through recovery manager
            recovery_result = await self.recovery_manager.handle_failure(task, e)

            if recovery_result.action == RecoveryAction.RETRY:
                return NodeResult.retry(
                    task_id=task.task_id,
                    error=str(e),
                    agent_name="PlanModifierAgent",
                    agent_type=AgentType.PLAN_MODIFIER.value,
                    metadata={
                        "retry_count": recovery_result.metadata.get("retry_attempt", 1),
                        "failure_reason": failure_reason,
                    },
                )
            else:
                return NodeResult.failure(
                    task_id=task.task_id,
                    error=str(e),
                    agent_name="PlanModifierAgent",
                    agent_type=AgentType.PLAN_MODIFIER.value,
                    metadata={
                        "recovery_action": recovery_result.action.value,
                        "failure_reason": failure_reason,
                    },
                )

    def _enhance_context_for_replanning(
        self, context: TaskContext, failed_children: list[TaskNode], failure_reason: str
    ) -> TaskContext:
        """
        Enhance context with replanning-specific information.

        Args:
            context: Original task context
            failed_children: Failed child tasks
            failure_reason: Reason for replanning

        Returns:
            Enhanced TaskContext with replanning metadata
        """
        # Extract failure information
        failure_info = []
        for failed_child in failed_children:
            failure_info.append(
                {
                    "task_id": failed_child.task_id,
                    "goal": failed_child.goal,
                    "task_type": failed_child.task_type.value,
                    "failure_count": failed_child.retry_count,
                    "metadata": failed_child.metadata,
                }
            )

        # Create enhanced metadata
        enhanced_metadata = {
            **context.execution_metadata,
            "replanning_reason": failure_reason,
            "failed_children": failure_info,
            "failed_children_count": len(failed_children),
            "replanning_timestamp": context.execution_metadata.get("execution_timestamp"),
            "original_task_id": context.task.task_id,
        }

        logger.info(
            f"Replanning context for {context.task.task_id}: "
            f"{len(failed_children)} failed children, reason: {failure_reason}"
        )

        # Create enhanced context
        return TaskContext(
            task=context.task,
            overall_objective=context.overall_objective,
            execution_id=context.execution_id,
            execution_metadata=enhanced_metadata,
        )

    async def _convert_modified_plan_to_nodes(
        self, plan_modifier_result: PlanModifierResult, parent: TaskNode
    ) -> list[TaskNode]:
        """
        Convert modified plan to TaskNode subtasks.

        Args:
            plan_modifier_result: Result from plan modifier agent
            parent: Parent task node

        Returns:
            List of TaskNode subtasks with resolved dependencies
        """
        if not plan_modifier_result.modified_subtasks:
            raise ValueError("Plan modifier returned empty subtask list")

        subtasks = plan_modifier_result.modified_subtasks
        task_nodes = []
        subtask_id_mapping = {}

        # First pass: Create all nodes without dependencies
        for i, subtask in enumerate(subtasks):
            # Generate new subtask ID (with replan suffix)
            subtask_id = f"{parent.task_id}_replan_sub_{i}"
            subtask_id_mapping[f"sub_{i}"] = subtask_id
            subtask_id_mapping[str(i)] = subtask_id

            # Create TaskNode
            task_node = TaskNode(
                task_id=subtask_id,
                goal=subtask.goal,
                task_type=subtask.task_type,
                parent_id=parent.task_id,
                metadata={
                    "priority": subtask.priority,
                    "original_dependencies": subtask.dependencies,
                    "estimated_effort": subtask.estimated_effort,
                    "subtask_metadata": subtask.metadata or {},
                    "is_replanned": True,
                    "replan_iteration": parent.metadata.get("replan_count", 0) + 1,
                },
            )
            task_nodes.append(task_node)

        # Second pass: Resolve dependencies using new dependency graph if provided
        dependency_graph = plan_modifier_result.new_dependencies or {}

        for i, subtask in enumerate(subtasks):
            resolved_deps = set()

            # Use new dependency graph if available
            subtask_key = f"sub_{i}"
            if subtask_key in dependency_graph:
                for dep_label in dependency_graph[subtask_key]:
                    if dep_label in subtask_id_mapping:
                        resolved_deps.add(subtask_id_mapping[dep_label])
            else:
                # Fall back to original dependencies
                for dep_label in subtask.dependencies:
                    if dep_label in subtask_id_mapping:
                        resolved_deps.add(subtask_id_mapping[dep_label])

            if resolved_deps:
                # Update node with resolved dependencies
                task_nodes[i] = task_nodes[i].model_copy(
                    update={"dependencies": frozenset(resolved_deps)}
                )

        logger.info(
            f"Converted {len(subtasks)} modified subtasks to TaskNodes for parent {parent.task_id}"
        )

        return task_nodes

    def get_stats(self) -> dict[str, Any]:
        """Get plan modifier service statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "replans_performed": getattr(self, "_replans_performed", 0),
            "successful_replans": getattr(self, "_successful_replans", 0),
            "average_failed_children": getattr(self, "_avg_failed_children", 0.0),
        }
