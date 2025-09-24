"""
Planner Service Implementation.

Handles planner agent execution to decompose complex tasks into subtasks.
Migrated from TaskNodeProcessor with dependency resolution logic.
"""

import logging
from typing import List, Dict, Any, Optional

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.agent_responses import PlannerResult, SubTask
from roma.domain.value_objects.result_envelope import ExecutionMetrics, PlannerEnvelope
from roma.domain.interfaces.agent_service import PlannerServiceInterface
from roma.domain.context import TaskContext
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager, RecoveryAction

logger = logging.getLogger(__name__)


class PlannerService(PlannerServiceInterface):
    """Implementation of Planner service for task decomposition."""

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
        **kwargs
    ) -> NodeResult:
        """
        Run planner to decompose task into subtasks.

        Args:
            task: Task to decompose
            context: Task execution context (contains execution_id)
            **kwargs: Additional parameters

        Returns:
            NodeResult with ADD_SUBTASKS action and new_nodes populated
        """
        try:
            # Get planner agent for task type
            planner_agent = await self.agent_runtime_service.get_agent(
                task.task_type, AgentType.PLANNER
            )

            # Execute planner
            logger.info(f"Running planner for task {task.task_id} ({task.task_type})")
            envelope = await self.agent_runtime_service.execute_agent(
                planner_agent, task, context, AgentType.PLANNER, context.execution_id
            )

            # Extract planner result
            if not envelope or not hasattr(envelope, 'result'):
                raise ValueError("Planner returned invalid envelope")

            planner_result: PlannerResult = envelope.result
            if not isinstance(planner_result, PlannerResult):
                raise ValueError(f"Expected PlannerResult, got {type(planner_result)}")

            # Convert plan to TaskNodes with dependency resolution
            subtask_nodes = await self._convert_plan_to_nodes(planner_result, task)

            # Create execution metrics
            execution_time = context.execution_metadata.get("execution_time", 0.0)
            metrics = ExecutionMetrics(
                execution_time=execution_time,
                tokens_used=getattr(envelope, 'tokens_used', 0),
                cost_estimate=getattr(envelope, 'cost_estimate', 0.0),
                model_calls=1
            )

            # Create typed envelope
            planner_envelope = PlannerEnvelope.create_success(
                result=planner_result,
                task_id=task.task_id,
                execution_id=context.execution_id,
                agent_type=AgentType.PLANNER,
                execution_metrics=metrics,
                output_text=f"Created {len(subtask_nodes)} subtasks"
            )

            # Record success with recovery manager
            await self.recovery_manager.record_success(task.task_id)

            return NodeResult.planning_result(task_id=task.task_id, 
                subtasks=subtask_nodes,
                envelope=planner_envelope,
                agent_name=getattr(planner_agent, 'name', 'PlannerAgent'),
                processing_time_ms=execution_time * 1000,
                metadata={
                    "subtask_count": len(subtask_nodes),
                    "reasoning": planner_result.reasoning,
                    "estimated_effort": planner_result.estimated_total_effort
                }
            )

        except Exception as e:
            logger.error(f"Planner failed for task {task.task_id}: {e}")

            # Handle failure through recovery manager
            recovery_result = await self.recovery_manager.handle_failure(task, e)

            if recovery_result.action == RecoveryAction.RETRY:
                return NodeResult.retry(task_id=task.task_id,
                    error=str(e),
                    agent_name="PlannerAgent",
                    agent_type=AgentType.PLANNER.value,
                    metadata={"retry_count": recovery_result.metadata.get("retry_attempt", 1)}
                )
            elif recovery_result.action == RecoveryAction.REPLAN:
                return NodeResult.replan(
                    task_id=task.task_id,
                    parent_id=recovery_result.metadata.get("parent_id"),
                    reason="critical_failure_exhausted_retries",
                    agent_name="PlannerAgent",
                    agent_type=AgentType.PLANNER.value
                )
            else:
                return NodeResult.failure(task_id=task.task_id,
                    error=str(e),
                    agent_name="PlannerAgent",
                    agent_type=AgentType.PLANNER.value
                )

    async def _convert_plan_to_nodes(self, planner_result: PlannerResult, parent: TaskNode) -> List[TaskNode]:
        """
        Convert PlannerResult to TaskNode subtasks with dependency resolution.

        Args:
            planner_result: Result from planner agent
            parent: Parent task node

        Returns:
            List of TaskNode subtasks with resolved dependencies
        """
        if not planner_result.subtasks:
            raise ValueError("Planner returned empty subtask list")

        subtasks = planner_result.subtasks
        task_nodes = []
        subtask_id_mapping = {}

        # First pass: Create all nodes without dependencies
        for i, subtask in enumerate(subtasks):
            # Generate subtask ID
            subtask_id = f"{parent.task_id}_sub_{i}"
            subtask_id_mapping[f"sub_{i}"] = subtask_id
            subtask_id_mapping[str(i)] = subtask_id  # Support numeric references

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
                    "subtask_metadata": subtask.metadata or {}
                }
            )
            task_nodes.append(task_node)

        # Second pass: Resolve dependencies and update nodes
        for i, subtask in enumerate(subtasks):
            if subtask.dependencies:
                resolved_deps = set()
                for dep_label in subtask.dependencies:
                    if dep_label in subtask_id_mapping:
                        resolved_deps.add(subtask_id_mapping[dep_label])
                    else:
                        logger.warning(
                            f"Could not resolve dependency '{dep_label}' for subtask {i} "
                            f"in task {parent.task_id}"
                        )

                if resolved_deps:
                    # Update node with resolved dependencies
                    task_nodes[i] = task_nodes[i].model_copy(
                        update={"dependencies": frozenset(resolved_deps)}
                    )

        logger.info(
            f"Converted {len(subtasks)} subtasks to TaskNodes for parent {parent.task_id}"
        )

        return task_nodes

    def get_stats(self) -> Dict[str, Any]:
        """Get planner service statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "plans_created": getattr(self, '_plans_created', 0),
            "total_subtasks_generated": getattr(self, '_total_subtasks', 0),
            "average_subtasks_per_plan": getattr(self, '_avg_subtasks', 0.0)
        }