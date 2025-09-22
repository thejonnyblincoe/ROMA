"""
Plan Modifier Service Implementation.

Handles plan modifier agent execution to adjust plans based on failures or feedback.
Used for replanning scenarios when failure thresholds are exceeded.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Awaitable

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.agent_responses import PlanModifierResult
from roma.domain.value_objects.result_envelope import ExecutionMetrics, PlanModifierEnvelope
from roma.domain.interfaces.agent_service import PlanModifierServiceInterface
from roma.application.services.context_builder_service import TaskContext, ContextBuilderService
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager, RecoveryAction
from roma.application.services.hitl_service import HITLService
from roma.domain.value_objects.hitl_request import HITLRequestStatus

logger = logging.getLogger(__name__)


class PlanModifierService(PlanModifierServiceInterface):
    """
    Implementation of Plan Modifier service for replanning scenarios.

    Manages replanning workflow internally for better separation of concerns
    from the orchestrator.
    """

    def __init__(
        self,
        agent_runtime_service: AgentRuntimeService,
        recovery_manager: RecoveryManager,
        hitl_service: Optional[HITLService] = None
    ):
        """Initialize with required dependencies."""
        self.agent_runtime_service = agent_runtime_service
        self.recovery_manager = recovery_manager
        self.hitl_service = hitl_service

        # Callbacks for orchestrator communication
        self._get_all_nodes_callback: Optional[Callable[[], List[TaskNode]]] = None
        self._get_children_callback: Optional[Callable[[str], List[TaskNode]]] = None
        self._remove_node_callback: Optional[Callable[[str], Awaitable[None]]] = None
        self._transition_status_callback: Optional[Callable[[str, TaskStatus], Awaitable[None]]] = None
        self._handle_replan_result_callback: Optional[Callable[[NodeResult, str], Awaitable[None]]] = None
        self._context_builder: Optional[ContextBuilderService] = None
        self._hitl_enabled: bool = False

    async def run(
        self,
        task: TaskNode,
        context: TaskContext,
        execution_id: Optional[str] = None,
        failed_children: List[TaskNode] = None,
        failure_reason: str = None,
        **kwargs
    ) -> NodeResult:
        """
        Run plan modifier to adjust plans based on failures.

        Args:
            task: Original parent task
            context: Task context
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
                plan_modifier_agent, task, enhanced_context, AgentType.PLAN_MODIFIER, execution_id
            )

            # Extract plan modifier result
            if not envelope or not hasattr(envelope, 'result'):
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
                tokens_used=getattr(envelope, 'tokens_used', 0),
                cost_estimate=getattr(envelope, 'cost_estimate', 0.0),
                model_calls=1
            )

            # Create typed envelope
            plan_modifier_envelope = PlanModifierEnvelope.create_success(
                result=plan_modifier_result,
                task_id=task.task_id,
                execution_id=context.execution_metadata.get("execution_id", "unknown"),
                agent_type=AgentType.PLAN_MODIFIER,
                execution_metrics=metrics,
                output_text=f"Modified plan with {len(new_subtask_nodes)} subtasks"
            )

            # Record success with recovery manager
            await self.recovery_manager.record_success(task.task_id)

            return NodeResult(
                action=NodeAction.REPLAN,
                envelope=plan_modifier_envelope,
                new_nodes=new_subtask_nodes,
                agent_name=getattr(plan_modifier_agent, 'name', 'PlanModifierAgent'),
                agent_type=AgentType.PLAN_MODIFIER.value,
                processing_time_ms=execution_time * 1000,
                metadata={
                    "failure_reason": failure_reason,
                    "failed_children_count": len(failed_children),
                    "new_subtask_count": len(new_subtask_nodes),
                    "changes_made": plan_modifier_result.changes_made,
                    "impact_assessment": plan_modifier_result.impact_assessment
                }
            )

        except Exception as e:
            logger.error(f"Plan modifier failed for task {task.task_id}: {e}")

            # Handle failure through recovery manager
            recovery_result = await self.recovery_manager.handle_failure(task, e)

            if recovery_result.action == RecoveryAction.RETRY:
                return NodeResult.retry(
                    error=str(e),
                    agent_name="PlanModifierAgent",
                    agent_type=AgentType.PLAN_MODIFIER.value,
                    metadata={
                        "retry_count": recovery_result.metadata.get("retry_attempt", 1),
                        "failure_reason": failure_reason
                    }
                )
            else:
                return NodeResult.failure(
                    error=str(e),
                    agent_name="PlanModifierAgent",
                    agent_type=AgentType.PLAN_MODIFIER.value,
                    metadata={
                        "recovery_action": recovery_result.action.value,
                        "failure_reason": failure_reason
                    }
                )

    def _enhance_context_for_replanning(
        self,
        context: TaskContext,
        failed_children: List[TaskNode],
        failure_reason: str
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
            failure_info.append({
                "task_id": failed_child.task_id,
                "goal": failed_child.goal,
                "task_type": failed_child.task_type.value,
                "failure_count": failed_child.retry_count,
                "metadata": failed_child.metadata
            })

        # Create enhanced metadata
        enhanced_metadata = {
            **context.execution_metadata,
            "replanning_reason": failure_reason,
            "failed_children": failure_info,
            "failed_children_count": len(failed_children),
            "replanning_timestamp": context.execution_metadata.get("execution_timestamp"),
            "original_task_id": context.task.task_id
        }

        logger.info(
            f"Replanning context for {context.task.task_id}: "
            f"{len(failed_children)} failed children, reason: {failure_reason}"
        )

        # Create enhanced context
        return TaskContext(
            task=context.task,
            overall_objective=context.overall_objective,
            execution_metadata=enhanced_metadata
        )

    async def _convert_modified_plan_to_nodes(
        self,
        plan_modifier_result: PlanModifierResult,
        parent: TaskNode
    ) -> List[TaskNode]:
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
                priority=subtask.priority,
                metadata={
                    "original_dependencies": subtask.dependencies,
                    "estimated_effort": subtask.estimated_effort,
                    "subtask_metadata": subtask.metadata or {},
                    "is_replanned": True,
                    "replan_iteration": parent.metadata.get("replan_count", 0) + 1
                }
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

    def set_orchestrator_callbacks(
        self,
        get_all_nodes: Callable[[], List[TaskNode]],
        get_children: Callable[[str], List[TaskNode]],
        remove_node: Callable[[str], Awaitable[None]],
        transition_status: Callable[[str, TaskStatus], Awaitable[None]],
        handle_replan_result: Callable[[NodeResult, str], Awaitable[None]],
        context_builder: ContextBuilderService,
        hitl_enabled: bool = False
    ) -> None:
        """
        Set callbacks for communicating with orchestrator.

        Args:
            get_all_nodes: Callback to get all nodes in graph
            get_children: Callback to get children nodes by parent ID
            remove_node: Callback to remove node from graph
            transition_status: Callback to transition node status
            handle_replan_result: Callback to handle replan result
            context_builder: Context builder service
            hitl_enabled: Whether HITL replanning is enabled
        """
        self._get_all_nodes_callback = get_all_nodes
        self._get_children_callback = get_children
        self._remove_node_callback = remove_node
        self._transition_status_callback = transition_status
        self._handle_replan_result_callback = handle_replan_result
        self._context_builder = context_builder
        self._hitl_enabled = hitl_enabled

    async def process_replanning_nodes(self, base_context: TaskContext) -> None:
        """Process all nodes that need replanning (NEEDS_REPLAN status)."""
        if not self._get_all_nodes_callback or not self._get_children_callback:
            logger.error("Orchestrator callbacks not set - cannot process replanning nodes")
            return

        # Get all nodes with NEEDS_REPLAN status
        all_nodes = self._get_all_nodes_callback()
        replanning_nodes = [node for node in all_nodes if node.status == TaskStatus.NEEDS_REPLAN]

        if not replanning_nodes:
            return

        logger.info(f"Found {len(replanning_nodes)} nodes that need replanning")

        for node in replanning_nodes:
            await self._handle_single_replan(node, base_context)

    async def _handle_single_replan(self, node: TaskNode, base_context: TaskContext) -> None:
        """Handle replanning for a single node."""
        try:
            # Get failed children for context
            child_nodes = self._get_children_callback(node.task_id)
            failed_children = [child for child in child_nodes if child.status == TaskStatus.FAILED]

            # Build context for replanning
            replan_context = await self._context_builder.build_context(
                task=node,
                overall_objective=base_context.overall_objective,
                execution_metadata={
                    **base_context.execution_metadata,
                    "replanning_reason": f"Child failure threshold exceeded - {len(failed_children)} failed children",
                    "failed_children": [child.task_id for child in failed_children]
                }
            )

            # Check if HITL approval is required
            if self._hitl_enabled and self.hitl_service:
                logger.info(f"Requesting HITL approval for replanning node {node.task_id}")

                hitl_response = await self.hitl_service.request_replanning_approval(
                    node=node,
                    context=replan_context,
                    failed_children=failed_children,
                    failure_reason="Child failure threshold exceeded"
                )

                # Handle HITL response
                if hitl_response:
                    if hitl_response.status == HITLRequestStatus.REJECTED:
                        logger.warning(f"HITL rejected replanning for node {node.task_id}, proceeding with automatic replanning as fallback")
                        # Continue with automatic replanning as fallback strategy

                    elif hitl_response.status == HITLRequestStatus.MODIFIED:
                        logger.info(f"HITL modified replanning request for node {node.task_id}")
                        # Use modified context if provided
                        if hitl_response.modified_context:
                            replan_context = TaskContext(
                                task=replan_context.task,
                                overall_objective=replan_context.overall_objective,
                                execution_metadata={
                                    **replan_context.execution_metadata,
                                    **hitl_response.modified_context,
                                    "hitl_modified": True
                                }
                            )

                    elif hitl_response.status == HITLRequestStatus.APPROVED:
                        logger.info(f"HITL approved replanning for node {node.task_id}")

                    elif hitl_response.status == HITLRequestStatus.TIMEOUT:
                        logger.warning(f"HITL request timed out for node {node.task_id}, proceeding with automatic replanning")

            # Execute plan modifier
            modifier_result = await self.run(
                node,
                replan_context,
                failed_children=failed_children,
                failure_reason="Child failure threshold exceeded"
            )

            # Handle the modification result
            if modifier_result.action == NodeAction.REPLAN and modifier_result.new_nodes:
                # Remove old failed children from graph
                for failed_child in failed_children:
                    if self._remove_node_callback:
                        await self._remove_node_callback(failed_child.task_id)

                # Handle replan result through orchestrator
                if self._handle_replan_result_callback:
                    await self._handle_replan_result_callback(modifier_result, base_context.overall_objective)

                # Transition the parent back to READY for re-execution
                if self._transition_status_callback:
                    await self._transition_status_callback(node.task_id, TaskStatus.READY)

                logger.info(f"Node {node.task_id} successfully replanned with {len(modifier_result.new_nodes)} new subtasks")
            else:
                # Replanning failed - transition to FAILED
                if self._transition_status_callback:
                    await self._transition_status_callback(node.task_id, TaskStatus.FAILED)
                logger.error(f"Replanning failed for node {node.task_id}")

        except Exception as e:
            logger.error(f"Error during replanning for node {node.task_id}: {e}")
            # On error, transition to FAILED
            if self._transition_status_callback:
                await self._transition_status_callback(node.task_id, TaskStatus.FAILED)

    def get_stats(self) -> Dict[str, Any]:
        """Get plan modifier service statistics."""
        base_stats = super().get_stats()
        return {
            **base_stats,
            "replans_performed": getattr(self, '_replans_performed', 0),
            "successful_replans": getattr(self, '_successful_replans', 0),
            "average_failed_children": getattr(self, '_avg_failed_children', 0.0)
        }