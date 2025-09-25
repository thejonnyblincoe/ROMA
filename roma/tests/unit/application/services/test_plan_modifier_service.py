"""
Tests for PlanModifierService.

Tests the plan modifier service functionality including replanning workflows,
HITL integration, and failure handling.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.context_builder_service import ContextBuilderService
from roma.application.services.hitl_service import HITLService
from roma.application.services.plan_modifier_service import PlanModifierService
from roma.application.services.recovery_manager import (
    RecoveryAction,
    RecoveryManager,
    RecoveryResult,
)
from roma.domain.context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_responses import PlanModifierResult
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.result_envelope import ExecutionMetrics, PlanModifierEnvelope
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


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
        execution_id="test_execution",
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
        assert result.action == NodeAction.ADD_SUBTASKS
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
        assert result.action == NodeAction.ADD_SUBTASKS
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
        assert result.action == NodeAction.ADD_SUBTASKS
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

