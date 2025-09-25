"""
Tests for PlannerService.

Tests the planner service functionality including task decomposition and subtask creation.
"""

from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.planner_service import PlannerService
from roma.application.services.recovery_manager import RecoveryManager
from roma.domain.context import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_responses import PlannerResult, SubTask
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.result_envelope import ExecutionMetrics, PlannerEnvelope
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
def planner_service(mock_agent_runtime_service, mock_recovery_manager):
    """Create planner service with mocked dependencies."""
    return PlannerService(
        agent_runtime_service=mock_agent_runtime_service,
        recovery_manager=mock_recovery_manager
    )


@pytest.fixture
def sample_task():
    """Create a sample task node for planning."""
    return TaskNode(
        task_id=str(uuid4()),
        goal="Complete comprehensive market analysis for Q4 2024",
        task_type=TaskType.THINK,
        status=TaskStatus.READY,
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
        overall_objective="Complete market analysis",
        execution_id="test-planner-execution-id",
        execution_metadata={}
    )


class TestPlannerService:
    """Test PlannerService functionality."""

    def test_agent_type_is_planner(self, planner_service):
        """Test that service has correct agent type."""
        assert planner_service.agent_type == AgentType.PLANNER

    @pytest.mark.asyncio
    async def test_run_creates_subtasks(self, planner_service, sample_task, sample_context, mock_agent_runtime_service):
        """Test planner creating subtasks from a complex task."""
        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestPlannerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create SubTasks
        subtasks = [
            SubTask(
                goal="Research current market trends",
                task_type=TaskType.RETRIEVE,
                priority=1,
                dependencies=[]
            ),
            SubTask(
                goal="Analyze competitor strategies",
                task_type=TaskType.THINK,
                priority=2,
                dependencies=[]
            ),
            SubTask(
                goal="Create summary report",
                task_type=TaskType.WRITE,
                priority=3,
                dependencies=["task_0", "task_1"]  # String dependencies, not indices
            )
        ]

        # Create PlannerResult
        planner_result = PlannerResult(
            subtasks=subtasks,
            reasoning="Breaking down complex analysis into research, analysis, and reporting phases",
            strategy="sequential"
        )

        # Create envelope
        envelope = PlannerEnvelope.create_success(
            result=planner_result,
            task_id=sample_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.PLANNER,
            execution_metrics=ExecutionMetrics(execution_time=0.2, tokens_used=200, cost_estimate=0.02, model_calls=1),
            output_text="Breaking down complex analysis into research, analysis, and reporting phases"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await planner_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.ADD_SUBTASKS  # Planner adds subtasks, not PLAN
        assert len(result.new_nodes) == 3

        # Verify subtask properties
        subtasks = result.new_nodes
        assert subtasks[0].goal == "Research current market trends"
        assert subtasks[0].task_type == TaskType.RETRIEVE
        assert subtasks[0].parent_id == sample_task.task_id

        assert subtasks[1].goal == "Analyze competitor strategies"
        assert subtasks[1].task_type == TaskType.THINK
        assert subtasks[1].parent_id == sample_task.task_id

        assert subtasks[2].goal == "Create summary report"
        assert subtasks[2].task_type == TaskType.WRITE
        assert subtasks[2].parent_id == sample_task.task_id

        # Verify agent interaction
        mock_agent_runtime_service.get_agent.assert_called_once_with(
            sample_task.task_type, AgentType.PLANNER
        )

    @pytest.mark.asyncio
    async def test_run_handles_dependencies(self, planner_service, sample_task, sample_context, mock_agent_runtime_service):
        """Test planner correctly handling task dependencies."""
        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestPlannerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create SubTasks with dependencies
        subtasks = [
            SubTask(
                goal="Gather raw data",
                task_type=TaskType.RETRIEVE,
                priority=1,
                dependencies=[]
            ),
            SubTask(
                goal="Process and clean data",
                task_type=TaskType.CODE_INTERPRET,
                priority=2,
                dependencies=["subtask_0"]  # String dependency reference
            ),
            SubTask(
                goal="Generate visualizations",
                task_type=TaskType.IMAGE_GENERATION,
                priority=3,
                dependencies=["subtask_1"]  # String dependency reference
            )
        ]

        # Create PlannerResult
        planner_result = PlannerResult(
            subtasks=subtasks,
            reasoning="Sequential data pipeline with dependencies",
            strategy="sequential"
        )

        # Create envelope
        envelope = PlannerEnvelope.create_success(
            result=planner_result,
            task_id=sample_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.PLANNER,
            execution_metrics=ExecutionMetrics(execution_time=0.2, tokens_used=200, cost_estimate=0.02, model_calls=1),
            output_text="Sequential data pipeline with dependencies"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await planner_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify dependencies are tracked
        assert len(result.new_nodes) == 3
        subtasks = result.new_nodes

        # All subtasks should have parent_id set
        for subtask in subtasks:
            assert subtask.parent_id == sample_task.task_id

        # Verify task types
        assert subtasks[0].task_type == TaskType.RETRIEVE
        assert subtasks[1].task_type == TaskType.CODE_INTERPRET
        assert subtasks[2].task_type == TaskType.IMAGE_GENERATION

    @pytest.mark.asyncio
    async def test_run_handles_planning_error(self, planner_service, sample_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test error handling when planning fails."""
        # Setup mock to raise exception
        mock_agent = Mock()
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.side_effect = Exception("Planning agent failed")

        # Setup recovery manager
        from roma.application.services.recovery_manager import RecoveryAction, RecoveryResult
        recovery_result = RecoveryResult(action=RecoveryAction.FAIL_PERMANENTLY, reasoning="Planning failed")
        mock_recovery_manager.handle_failure = AsyncMock(return_value=recovery_result)

        # Execute the service
        result = await planner_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify fallback to execution
        assert result.action == NodeAction.FAIL
        assert result.error == "Planning agent failed"
        assert len(result.new_nodes) == 0

    @pytest.mark.asyncio
    async def test_run_with_empty_subtasks(self, planner_service, sample_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test handling when agent returns invalid envelope."""
        # Setup mock agent response with None envelope
        mock_agent = Mock()
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.return_value = None  # Invalid envelope

        # Setup recovery manager
        from roma.application.services.recovery_manager import RecoveryAction, RecoveryResult
        recovery_result = RecoveryResult(action=RecoveryAction.FAIL_PERMANENTLY, reasoning="Invalid response")
        mock_recovery_manager.handle_failure = AsyncMock(return_value=recovery_result)

        # Execute the service
        result = await planner_service.run(sample_task, sample_context, execution_id="test_execution")

        # Should fail when invalid envelope
        assert result.action == NodeAction.FAIL
        assert len(result.new_nodes) == 0
        assert result.error == "Planner returned invalid envelope"

    @pytest.mark.asyncio
    async def test_run_with_invalid_subtask_format(self, planner_service, sample_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test handling of invalid subtask format from agent."""
        # Setup mock agent with malformed subtasks
        mock_agent = Mock()
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.return_value = {
            "subtasks": [
                {
                    "goal": "Valid subtask",
                    "task_type": "THINK"
                    # Missing priority and dependencies
                },
                {
                    # Missing goal field
                    "task_type": "WRITE",
                    "priority": 1
                }
            ],
            "reasoning": "Malformed planning response"
        }

        # Setup recovery manager
        from roma.application.services.recovery_manager import RecoveryAction, RecoveryResult
        recovery_result = RecoveryResult(action=RecoveryAction.FAIL_PERMANENTLY, reasoning="Planning failed")
        mock_recovery_manager.handle_failure = AsyncMock(return_value=recovery_result)

        # Execute the service
        result = await planner_service.run(sample_task, sample_context, execution_id="test_execution")

        # Should handle gracefully and fallback
        assert result.action == NodeAction.FAIL
        assert result.error == "Planner returned invalid envelope"

    @pytest.mark.asyncio
    async def test_run_validates_subtask_types(self, planner_service, sample_task, sample_context, mock_agent_runtime_service):
        """Test validation of subtask types."""
        # Setup mock agent response with valid task types
        mock_agent = Mock()
        mock_agent.name = "TestPlannerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create proper PlannerResult with all task types
        from roma.domain.value_objects.agent_responses import PlannerResult
        planner_result = PlannerResult(
            subtasks=[
                {
                    "goal": "Research task",
                    "task_type": "RETRIEVE",
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "goal": "Analysis task",
                    "task_type": "THINK",
                    "priority": 2,
                    "dependencies": []
                },
                {
                    "goal": "Writing task",
                    "task_type": "WRITE",
                    "priority": 3,
                    "dependencies": []
                },
                {
                    "goal": "Coding task",
                    "task_type": "CODE_INTERPRET",
                    "priority": 4,
                    "dependencies": []
                },
                {
                    "goal": "Image task",
                    "task_type": "IMAGE_GENERATION",
                    "priority": 5,
                    "dependencies": []
                }
            ],
            reasoning="Comprehensive plan with all task types",
            estimated_total_effort=100
        )

        # Create mock envelope
        mock_envelope = Mock()
        mock_envelope.result = planner_result
        mock_envelope.tokens_used = 150
        mock_envelope.cost_estimate = 0.02
        mock_agent_runtime_service.execute_agent.return_value = mock_envelope

        # Execute the service
        result = await planner_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify all subtasks are created with correct types
        assert len(result.new_nodes) == 5
        subtasks = result.new_nodes

        expected_types = [
            TaskType.RETRIEVE,
            TaskType.THINK,
            TaskType.WRITE,
            TaskType.CODE_INTERPRET,
            TaskType.IMAGE_GENERATION
        ]

        for i, expected_type in enumerate(expected_types):
            assert subtasks[i].task_type == expected_type

    @pytest.mark.asyncio
    async def test_run_with_retry_on_failure(self, planner_service, sample_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test retry mechanism on planning failure."""
        # Setup mock to fail (testing recovery manager integration)
        mock_agent = Mock()
        mock_agent.name = "TestPlannerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.side_effect = Exception("First failure")

        # Setup recovery manager to retry once
        from roma.application.services.recovery_manager import RecoveryAction, RecoveryResult
        recovery_result = RecoveryResult(
            action=RecoveryAction.RETRY,
            reasoning="Should retry planning",
            metadata={"retry_attempt": 1}
        )
        mock_recovery_manager.handle_failure = AsyncMock(return_value=recovery_result)

        # Execute the service
        result = await planner_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify retry is requested (service doesn't handle retries internally)
        assert result.action == NodeAction.RETRY
        assert result.error == "First failure"
        assert mock_agent_runtime_service.execute_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_run_sets_node_metadata(self, planner_service, sample_task, sample_context, mock_agent_runtime_service):
        """Test that planner sets appropriate metadata on subtasks."""
        # Setup mock agent response
        mock_agent = Mock()
        mock_agent.name = "TestPlannerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create proper PlannerResult
        from roma.domain.value_objects.agent_responses import PlannerResult
        planner_result = PlannerResult(
            subtasks=[
                {
                    "goal": "First subtask",
                    "task_type": "THINK",
                    "priority": 1,
                    "dependencies": [],
                    "estimated_duration": 300,
                    "complexity": "medium"
                }
            ],
            reasoning="Single subtask with metadata",
            estimated_total_effort=100
        )

        # Create mock envelope
        mock_envelope = Mock()
        mock_envelope.result = planner_result
        mock_envelope.tokens_used = 75
        mock_envelope.cost_estimate = 0.015
        mock_agent_runtime_service.execute_agent.return_value = mock_envelope

        # Execute the service
        result = await planner_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify metadata is set correctly
        assert result.metadata["subtask_count"] == 1
        assert result.metadata["reasoning"] == "Single subtask with metadata"
        assert result.metadata["estimated_effort"] == 100
        assert len(result.new_nodes) == 1

        subtask = result.new_nodes[0]
        assert subtask.node_type is None  # Subtasks start with None, determined by atomizer
        assert subtask.status == TaskStatus.PENDING
        assert subtask.parent_id == sample_task.task_id
