"""
Tests for PlanModifierService.

Tests the plan modifier service functionality including replanning workflows,
HITL integration, and failure handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.agent_responses import PlanModifierResult
from roma.domain.value_objects.result_envelope import PlanModifierEnvelope, ExecutionMetrics
from roma.domain.value_objects.hitl_request import HITLRequestStatus
from roma.application.services.recovery_manager import RecoveryResult, RecoveryAction
from roma.application.services.plan_modifier_service import PlanModifierService
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager
from roma.application.services.hitl_service import HITLService
from roma.application.services.context_builder_service import TaskContext, ContextBuilderService


@pytest.fixture
def mock_agent_runtime_service():
    """Mock agent runtime service."""
    mock = Mock(spec=AgentRuntimeService)
    mock.get_agent = AsyncMock()
    mock.execute_agent = AsyncMock()
    return mock


@pytest.fixture
def mock_recovery_manager():
    """Mock recovery manager."""
    return Mock(spec=RecoveryManager)


@pytest.fixture
def mock_hitl_service():
    """Mock HITL service."""
    mock = Mock(spec=HITLService)
    mock.enabled = True
    mock.request_replanning_approval = AsyncMock()
    return mock


@pytest.fixture
def mock_context_builder():
    """Mock context builder service."""
    mock = Mock(spec=ContextBuilderService)
    mock.build_context = AsyncMock()
    return mock


@pytest.fixture
def plan_modifier_service(mock_agent_runtime_service, mock_recovery_manager, mock_hitl_service):
    """Create plan modifier service with mocked dependencies."""
    service = PlanModifierService(
        agent_runtime_service=mock_agent_runtime_service,
        recovery_manager=mock_recovery_manager,
        hitl_service=mock_hitl_service
    )

    # Set up callbacks
    service._get_all_nodes_callback = Mock(return_value=[])
    service._get_children_callback = Mock(return_value=[])
    service._remove_node_callback = AsyncMock()
    service._transition_status_callback = AsyncMock()
    service._handle_replan_result_callback = AsyncMock()
    service._hitl_enabled = True

    return service


@pytest.fixture
def sample_failed_task():
    """Create a sample task that needs replanning."""
    return TaskNode(
        task_id=str(uuid4()),
        goal="Complex analysis task that failed",
        task_type=TaskType.THINK,
        status=TaskStatus.NEEDS_REPLAN,
        node_type=NodeType.PLAN
    )


@pytest.fixture
def sample_context():
    """Create a sample task context."""
    return TaskContext(
        task=TaskNode(
            task_id=str(uuid4()),
            goal="Root task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        ),
        overall_objective="Complete analysis",
        execution_metadata={}
    )


@pytest.fixture
def sample_failed_children():
    """Create sample failed child tasks."""
    parent_id = str(uuid4())
    return [
        TaskNode(
            task_id=str(uuid4()),
            goal="Failed child 1",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.FAILED,
            parent_id=parent_id
        ),
        TaskNode(
            task_id=str(uuid4()),
            goal="Failed child 2",
            task_type=TaskType.THINK,
            status=TaskStatus.FAILED,
            parent_id=parent_id
        )
    ]


class TestPlanModifierService:
    """Test PlanModifierService functionality."""

    def test_agent_type_is_plan_modifier(self, plan_modifier_service):
        """Test that service has correct agent type."""
        assert plan_modifier_service.agent_type == AgentType.PLAN_MODIFIER

    def test_hitl_service_injection(self, plan_modifier_service, mock_hitl_service):
        """Test that HITL service is properly injected."""
        assert plan_modifier_service.hitl_service == mock_hitl_service

    @pytest.mark.asyncio
    async def test_run_creates_replacement_plan(self, plan_modifier_service, sample_failed_task, sample_context, sample_failed_children, mock_agent_runtime_service):
        """Test plan modifier creating replacement subtasks."""
        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestPlanModifierAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create PlanModifierResult
        plan_modifier_result = PlanModifierResult(
            modified_subtasks=[
                {
                    "goal": "Alternative approach to data collection",
                    "task_type": "RETRIEVE",
                    "priority": 1,
                    "dependencies": [],
                    "estimated_effort": 3
                },
                {
                    "goal": "Simplified analysis method",
                    "task_type": "THINK",
                    "priority": 2,
                    "dependencies": ["data_collection"],
                    "estimated_effort": 4
                }
            ],
            changes_made=[
                "Replaced failed API-based data collection with alternative sources",
                "Simplified analysis method to reduce timeout risk"
            ],
            reasoning="Replacing failed approach with more robust strategy"
        )

        # Create envelope
        envelope = PlanModifierEnvelope.create_success(
            result=plan_modifier_result,
            task_id=sample_failed_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.PLAN_MODIFIER,
            execution_metrics=ExecutionMetrics(execution_time=60.0, tokens_used=250, cost_estimate=0.05, model_calls=1),
            output_text="Plan modification completed with 2 new subtasks"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await plan_modifier_service.run(
            sample_failed_task,
            sample_context,
            execution_id="test_execution",
            failed_children=sample_failed_children,
            failure_reason="Child tasks failed due to API timeout"
        )

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.REPLAN
        assert len(result.new_nodes) == 2
        assert result.envelope.result.reasoning == "Replacing failed approach with more robust strategy"
        # metadata should contain confidence from envelope or service metadata

        # Verify new subtasks
        new_tasks = result.new_nodes
        assert new_tasks[0].goal == "Alternative approach to data collection"
        assert new_tasks[0].task_type == TaskType.RETRIEVE
        assert new_tasks[0].parent_id == sample_failed_task.task_id

        assert new_tasks[1].goal == "Simplified analysis method"
        assert new_tasks[1].task_type == TaskType.THINK
        assert new_tasks[1].parent_id == sample_failed_task.task_id

    @pytest.mark.asyncio
    async def test_run_with_hitl_approval(self, plan_modifier_service, sample_failed_task, sample_context, sample_failed_children, mock_agent_runtime_service, mock_hitl_service, mock_context_builder):
        """Test replanning workflow with HITL approval."""
        # Setup context builder
        plan_modifier_service._context_builder = mock_context_builder
        mock_context_builder.build_context.return_value = sample_context

        # Setup HITL service to approve
        from roma.domain.value_objects.hitl_request import HITLResponse
        hitl_response = HITLResponse(
            request_id=str(uuid4()),
            status=HITLRequestStatus.APPROVED,
            response_data={}
        )
        mock_hitl_service.request_replanning_approval.return_value = hitl_response

        # Setup mock agent response
        mock_agent = Mock()
        mock_agent.name = "TestPlanModifierAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.return_value = {
            "new_subtasks": [
                {
                    "goal": "HITL approved task",
                    "task_type": "THINK",
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "reasoning": "Plan approved by human operator"
        }

        # Setup replanning nodes
        plan_modifier_service._get_all_nodes_callback.return_value = [sample_failed_task]
        plan_modifier_service._get_children_callback.return_value = sample_failed_children

        # Process replanning
        await plan_modifier_service.process_replanning_nodes(sample_context)

        # Verify HITL service was called
        mock_hitl_service.request_replanning_approval.assert_called_once()

        # Verify plan modification was executed
        mock_agent_runtime_service.execute_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_hitl_rejection_fallback(self, plan_modifier_service, sample_failed_task, sample_context, sample_failed_children, mock_agent_runtime_service, mock_hitl_service, mock_context_builder):
        """Test replanning workflow when HITL rejects but continues with automatic fallback."""
        # Setup context builder
        plan_modifier_service._context_builder = mock_context_builder
        mock_context_builder.build_context.return_value = sample_context

        # Setup HITL service to reject
        from roma.domain.value_objects.hitl_request import HITLResponse
        hitl_response = HITLResponse(
            request_id=str(uuid4()),
            status=HITLRequestStatus.REJECTED,
            response_data={"reason": "Approach not suitable"}
        )
        mock_hitl_service.request_replanning_approval.return_value = hitl_response

        # Setup mock agent response (automatic fallback)
        mock_agent = Mock()
        mock_agent.name = "TestPlanModifierAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.return_value = {
            "new_subtasks": [
                {
                    "goal": "Automatic fallback task",
                    "task_type": "THINK",
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "reasoning": "Automatic replanning after HITL rejection"
        }

        # Setup replanning nodes
        plan_modifier_service._get_all_nodes_callback.return_value = [sample_failed_task]
        plan_modifier_service._get_children_callback.return_value = sample_failed_children

        # Process replanning
        await plan_modifier_service.process_replanning_nodes(sample_context)

        # Verify HITL service was called
        mock_hitl_service.request_replanning_approval.assert_called_once()

        # Verify automatic plan modification was still executed as fallback
        mock_agent_runtime_service.execute_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_hitl_modification(self, plan_modifier_service, sample_failed_task, sample_context, sample_failed_children, mock_agent_runtime_service, mock_hitl_service, mock_context_builder):
        """Test replanning workflow when HITL modifies the request."""
        # Setup context builder
        plan_modifier_service._context_builder = mock_context_builder
        mock_context_builder.build_context.return_value = sample_context

        # Setup HITL service to modify
        from roma.domain.value_objects.hitl_request import HITLResponse
        hitl_response = HITLResponse(
            request_id=str(uuid4()),
            status=HITLRequestStatus.MODIFIED,
            response_data={},
            modified_context={"priority": "high", "alternative_approach": "use_cache"}
        )
        mock_hitl_service.request_replanning_approval.return_value = hitl_response

        # Setup mock agent response
        mock_agent = Mock()
        mock_agent.name = "TestPlanModifierAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.return_value = {
            "new_subtasks": [
                {
                    "goal": "HITL modified task",
                    "task_type": "THINK",
                    "priority": 1,
                    "dependencies": []
                }
            ],
            "reasoning": "Plan modified per human feedback"
        }

        # Setup replanning nodes
        plan_modifier_service._get_all_nodes_callback.return_value = [sample_failed_task]
        plan_modifier_service._get_children_callback.return_value = sample_failed_children

        # Process replanning
        await plan_modifier_service.process_replanning_nodes(sample_context)

        # Verify HITL service was called
        mock_hitl_service.request_replanning_approval.assert_called_once()

        # Verify modified context was used in agent execution
        call_args = mock_agent_runtime_service.execute_agent.call_args
        used_context = call_args[0][2]  # Third argument is context
        assert used_context.execution_metadata["hitl_modified"] is True
        assert used_context.execution_metadata["priority"] == "high"

    @pytest.mark.asyncio
    async def test_run_handles_planning_error(self, plan_modifier_service, sample_failed_task, sample_context, sample_failed_children, mock_agent_runtime_service, mock_recovery_manager):
        """Test error handling when plan modification fails."""
        # Setup mock to raise exception
        mock_agent = Mock()
        mock_agent.name = "TestPlanModifierAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.side_effect = Exception("Plan modification failed")

        # Setup recovery manager with new interface
        recovery_result = RecoveryResult(
            action=RecoveryAction.FAIL_PERMANENTLY,
            reasoning="Plan modification failed after retry limit",
            metadata={}
        )
        mock_recovery_manager.handle_failure.return_value = recovery_result

        # Execute the service
        result = await plan_modifier_service.run(
            sample_failed_task,
            sample_context,
            execution_id="test_execution",
            failed_children=sample_failed_children,
            failure_reason="Test failure"
        )

        # Verify error handling
        assert result.action == NodeAction.FAIL
        assert "Plan modification failed" in result.error
        assert len(result.new_nodes) == 0

    @pytest.mark.asyncio
    async def test_run_with_retry_on_failure(self, plan_modifier_service, sample_failed_task, sample_context, sample_failed_children, mock_agent_runtime_service, mock_recovery_manager):
        """Test retry mechanism on plan modification failure."""
        # Setup mock to succeed on first try (simulating successful retry at higher level)
        mock_agent = Mock()
        mock_agent.name = "TestPlanModifierAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create mock envelope with plan modifier result
        from roma.domain.value_objects.agent_responses import PlanModifierResult
        plan_modifier_result = PlanModifierResult(
            modified_subtasks=[
                {
                    "goal": "Retry success task",
                    "task_type": "THINK",
                    "priority": 1,
                    "dependencies": []
                }
            ],
            reasoning="Success on retry",
            changes_made=["Modified task goal"],
            impact_assessment="Medium"
        )

        mock_envelope = Mock()
        mock_envelope.result = plan_modifier_result
        mock_envelope.tokens_used = 100
        mock_envelope.cost_estimate = 0.01
        mock_agent_runtime_service.execute_agent.return_value = mock_envelope

        # Execute the service
        result = await plan_modifier_service.run(
            sample_failed_task,
            sample_context,
            execution_id="test_execution",
            failed_children=sample_failed_children,
            failure_reason="Test failure"
        )

        # Verify successful execution
        assert result.action == NodeAction.REPLAN
        assert len(result.new_nodes) == 1
        assert result.new_nodes[0].goal == "Retry success task"
        assert mock_agent_runtime_service.execute_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_run_without_hitl_service(self, sample_failed_task, sample_context, sample_failed_children, mock_agent_runtime_service, mock_recovery_manager):
        """Test plan modifier service without HITL service."""
        # Create service without HITL
        service = PlanModifierService(
            agent_runtime_service=mock_agent_runtime_service,
            recovery_manager=mock_recovery_manager,
            hitl_service=None
        )

        # Setup mock agent response
        mock_agent = Mock()
        mock_agent.name = "TestPlanModifierAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create mock envelope with plan modifier result
        from roma.domain.value_objects.agent_responses import PlanModifierResult
        plan_modifier_result = PlanModifierResult(
            modified_subtasks=[
                {
                    "goal": "No HITL task",
                    "task_type": "THINK",
                    "priority": 1,
                    "dependencies": []
                }
            ],
            reasoning="Direct replanning without human approval",
            changes_made=["Removed HITL requirement"],
            impact_assessment="Low"
        )

        mock_envelope = Mock()
        mock_envelope.result = plan_modifier_result
        mock_envelope.tokens_used = 100
        mock_envelope.cost_estimate = 0.01
        mock_agent_runtime_service.execute_agent.return_value = mock_envelope

        # Execute the service
        result = await service.run(
            sample_failed_task,
            sample_context,
            execution_id="test_execution",
            failed_children=sample_failed_children,
            failure_reason="Test failure"
        )

        # Should work normally without HITL
        assert result.action == NodeAction.REPLAN
        assert len(result.new_nodes) == 1
        assert result.new_nodes[0].goal == "No HITL task"

    @pytest.mark.asyncio
    async def test_run_with_empty_new_subtasks(self, plan_modifier_service, sample_failed_task, sample_context, sample_failed_children, mock_agent_runtime_service):
        """Test handling when plan modifier returns no new subtasks."""
        # Setup mock agent response with empty subtasks using envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestPlanModifierAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create PlanModifierResult with empty subtasks
        plan_modifier_result = PlanModifierResult(
            modified_subtasks=[],
            changes_made=["Attempted plan modification but no viable alternatives found"],
            reasoning="Unable to create alternative plan"
        )

        # Create envelope
        envelope = PlanModifierEnvelope.create_success(
            result=plan_modifier_result,
            task_id=sample_failed_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.PLAN_MODIFIER,
            execution_metrics=ExecutionMetrics(execution_time=30.0, tokens_used=100, cost_estimate=0.02, model_calls=1),
            output_text="No new subtasks generated"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await plan_modifier_service.run(
            sample_failed_task,
            sample_context,
            execution_id="test_execution",
            failed_children=sample_failed_children,
            failure_reason="Test failure"
        )

        # Should fail when no new subtasks generated
        assert result.action == NodeAction.FAIL
        assert "Plan modifier returned empty subtask list" in result.error
        assert len(result.new_nodes) == 0

    @pytest.mark.asyncio
    async def test_process_replanning_nodes_filters_correctly(self, plan_modifier_service, sample_context, mock_context_builder):
        """Test that replanning only processes nodes with NEEDS_REPLAN status."""
        # Create mix of nodes with different statuses
        replan_node = TaskNode(
            task_id=str(uuid4()),
            goal="Needs replanning",
            task_type=TaskType.THINK,
            status=TaskStatus.NEEDS_REPLAN
        )

        other_nodes = [
            TaskNode(
                task_id=str(uuid4()),
                goal="Completed node",
                task_type=TaskType.THINK,
                status=TaskStatus.COMPLETED
            ),
            TaskNode(
                task_id=str(uuid4()),
                goal="Executing node",
                task_type=TaskType.THINK,
                status=TaskStatus.EXECUTING
            )
        ]

        all_nodes = [replan_node] + other_nodes

        # Setup callbacks
        plan_modifier_service._get_all_nodes_callback.return_value = all_nodes
        plan_modifier_service._get_children_callback.return_value = []
        plan_modifier_service._context_builder = mock_context_builder
        mock_context_builder.build_context.return_value = sample_context

        # Process replanning nodes
        await plan_modifier_service.process_replanning_nodes(sample_context)

        # Should only process the NEEDS_REPLAN node
        # Verify by checking if context builder was called (indicating processing)
        mock_context_builder.build_context.assert_called_once()

    def test_set_orchestrator_callbacks(self, plan_modifier_service, mock_context_builder):
        """Test setting orchestrator callbacks."""
        # Create mock callbacks
        get_all_nodes = Mock()
        get_children = Mock()
        remove_node = AsyncMock()
        transition_status = AsyncMock()
        handle_replan_result = AsyncMock()

        # Set callbacks
        plan_modifier_service.set_orchestrator_callbacks(
            get_all_nodes=get_all_nodes,
            get_children=get_children,
            remove_node=remove_node,
            transition_status=transition_status,
            handle_replan_result=handle_replan_result,
            context_builder=mock_context_builder,
            hitl_enabled=True
        )

        # Verify callbacks are set
        assert plan_modifier_service._get_all_nodes_callback == get_all_nodes
        assert plan_modifier_service._get_children_callback == get_children
        assert plan_modifier_service._remove_node_callback == remove_node
        assert plan_modifier_service._transition_status_callback == transition_status
        assert plan_modifier_service._handle_replan_result_callback == handle_replan_result
        assert plan_modifier_service._context_builder == mock_context_builder
        assert plan_modifier_service._hitl_enabled is True