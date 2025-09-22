"""
Tests for AtomizerService.

Tests the atomizer service functionality including task evaluation and decision making.
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
from roma.domain.value_objects.agent_responses import AtomizerResult
from roma.domain.value_objects.result_envelope import AtomizerEnvelope, ExecutionMetrics
from roma.application.services.atomizer_service import AtomizerService
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager
from roma.application.services.context_builder_service import TaskContext


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
def atomizer_service(mock_agent_runtime_service, mock_recovery_manager):
    """Create atomizer service with mocked dependencies."""
    return AtomizerService(
        agent_runtime_service=mock_agent_runtime_service,
        recovery_manager=mock_recovery_manager
    )


@pytest.fixture
def sample_task():
    """Create a sample task node."""
    return TaskNode(
        task_id=str(uuid4()),
        goal="Analyze the market trends for cryptocurrency",
        task_type=TaskType.THINK,
        status=TaskStatus.PENDING
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
        execution_metadata={}
    )


class TestAtomizerService:
    """Test AtomizerService functionality."""

    def test_agent_type_is_atomizer(self, atomizer_service):
        """Test that service has correct agent type."""
        assert atomizer_service.agent_type == AgentType.ATOMIZER

    @pytest.mark.asyncio
    async def test_run_returns_execute_action(self, atomizer_service, sample_task, sample_context, mock_agent_runtime_service):
        """Test atomizer returning NOOP action with EXECUTE decision for atomic task."""
        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestAtomizerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create AtomizerResult
        atomizer_result = AtomizerResult(
            is_atomic=True,
            node_type=NodeType.EXECUTE,
            reasoning="Task is atomic and can be executed directly",
            confidence=0.9
        )

        # Create envelope
        envelope = AtomizerEnvelope.create_success(
            result=atomizer_result,
            task_id=sample_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.ATOMIZER,
            execution_metrics=ExecutionMetrics(execution_time=0.1, tokens_used=100, cost_estimate=0.01, model_calls=1),
            output_text="Task is atomic and can be executed directly"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await atomizer_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.NOOP  # Atomizer returns NOOP, not EXECUTE
        assert result.metadata["atomizer_decision"] == True
        assert result.metadata["confidence"] == 0.9
        assert result.metadata["node_type"] == NodeType.EXECUTE

        # Verify agent interaction
        mock_agent_runtime_service.get_agent.assert_called_once_with(
            sample_task.task_type, AgentType.ATOMIZER
        )
        mock_agent_runtime_service.execute_agent.assert_called_once_with(
            mock_agent, sample_task, sample_context, AgentType.ATOMIZER, "test_execution"
        )

    @pytest.mark.asyncio
    async def test_run_returns_plan_action(self, atomizer_service, sample_task, sample_context, mock_agent_runtime_service):
        """Test atomizer returning NOOP action with PLAN decision for complex task."""
        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestAtomizerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create AtomizerResult
        atomizer_result = AtomizerResult(
            is_atomic=False,
            node_type=NodeType.PLAN,
            reasoning="Task is complex and needs decomposition",
            confidence=0.8,
            metadata={"subtask_count": 3}
        )

        # Create envelope
        envelope = AtomizerEnvelope.create_success(
            result=atomizer_result,
            task_id=sample_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.ATOMIZER,
            execution_metrics=ExecutionMetrics(execution_time=0.1, tokens_used=100, cost_estimate=0.01, model_calls=1),
            output_text="Task is complex and needs decomposition"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await atomizer_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.NOOP  # Atomizer returns NOOP, not PLAN
        assert result.metadata["atomizer_decision"] == False
        assert result.metadata["confidence"] == 0.8
        assert result.metadata["node_type"] == NodeType.PLAN

    @pytest.mark.asyncio
    async def test_run_with_retrieve_task_type(self, mock_agent_runtime_service, mock_recovery_manager, sample_context):
        """Test that RETRIEVE task type can be planned or executed like any other task."""
        # Create a RETRIEVE task
        retrieve_task = TaskNode(
            task_id=str(uuid4()),
            goal="Search for information about quantum computing",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.PENDING
        )

        atomizer_service = AtomizerService(
            agent_runtime_service=mock_agent_runtime_service,
            recovery_manager=mock_recovery_manager
        )

        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestAtomizerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create AtomizerResult for RETRIEVE (can be planned)
        atomizer_result = AtomizerResult(
            is_atomic=False,
            node_type=NodeType.PLAN,
            reasoning="RETRIEVE task needs decomposition into specific search queries",
            confidence=0.8
        )

        # Create envelope
        envelope = AtomizerEnvelope.create_success(
            result=atomizer_result,
            task_id=retrieve_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.ATOMIZER,
            execution_metrics=ExecutionMetrics(execution_time=0.1, tokens_used=100, cost_estimate=0.01, model_calls=1),
            output_text="RETRIEVE task needs decomposition into specific search queries"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await atomizer_service.run(retrieve_task, sample_context, execution_id="test_execution")

        # Verify RETRIEVE can be planned like any other task
        assert result.action == NodeAction.NOOP
        assert result.metadata["node_type"] == NodeType.PLAN
        assert result.metadata["atomizer_decision"] == False

    @pytest.mark.asyncio
    async def test_run_handles_agent_execution_error(self, atomizer_service, sample_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test error handling when agent execution fails."""
        # Setup mock to raise exception
        mock_agent = Mock()
        mock_agent.name = "TestAtomizerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.side_effect = Exception("Agent execution failed")

        # Setup recovery manager to return failure
        from roma.application.services.recovery_manager import RecoveryResult, RecoveryAction
        recovery_result = RecoveryResult(
            action=RecoveryAction.FAIL_PERMANENTLY,
            reasoning="Atomizer failed permanently",
            metadata={"retry_attempt": 1}
        )
        mock_recovery_manager.handle_failure = AsyncMock(return_value=recovery_result)

        # Execute the service
        result = await atomizer_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify failure result
        assert result.action == NodeAction.FAIL
        assert result.error == "Agent execution failed"

    @pytest.mark.asyncio
    async def test_run_with_retry_on_failure(self, atomizer_service, sample_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test retry mechanism on agent failure."""
        # Setup mock to raise exception
        mock_agent = Mock()
        mock_agent.name = "TestAtomizerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.side_effect = Exception("Agent execution failed")

        # Setup recovery manager to return retry
        from roma.application.services.recovery_manager import RecoveryResult, RecoveryAction
        recovery_result = RecoveryResult(
            action=RecoveryAction.RETRY,
            reasoning="Should retry this operation",
            metadata={"retry_attempt": 1}
        )
        mock_recovery_manager.handle_failure = AsyncMock(return_value=recovery_result)

        # Execute the service
        result = await atomizer_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify retry result
        assert result.action == NodeAction.RETRY
        assert result.error == "Agent execution failed"
        assert result.metadata["retry_count"] == 1
        # Atomizer service doesn't retry internally - it returns RETRY action for orchestrator to handle
        assert mock_agent_runtime_service.execute_agent.call_count == 1

    @pytest.mark.asyncio
    async def test_run_with_invalid_agent_response(self, atomizer_service, sample_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test handling of invalid agent response."""
        # Setup mock agent with invalid response (missing envelope structure)
        mock_agent = Mock()
        mock_agent.name = "TestAtomizerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.return_value = {
            "invalid_field": "should not be here",
            # Missing required envelope structure
        }

        # Setup recovery manager to return failure
        from roma.application.services.recovery_manager import RecoveryResult, RecoveryAction
        recovery_result = RecoveryResult(
            action=RecoveryAction.FAIL_PERMANENTLY,
            reasoning="Invalid agent response",
            metadata={"error_type": "invalid_response"}
        )
        mock_recovery_manager.handle_failure = AsyncMock(return_value=recovery_result)

        # Execute the service
        result = await atomizer_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify failure handling
        assert result.action == NodeAction.FAIL
        assert result.error == "Atomizer returned invalid envelope"

    @pytest.mark.asyncio
    async def test_run_with_different_task_types(self, atomizer_service, sample_context, mock_agent_runtime_service):
        """Test atomizer behavior with different task types."""
        task_types = [TaskType.THINK, TaskType.WRITE, TaskType.CODE_INTERPRET, TaskType.IMAGE_GENERATION]

        for task_type in task_types:
            # Create task with specific type
            task = TaskNode(
                task_id=str(uuid4()),
                goal=f"Test task for {task_type.value}",
                task_type=task_type,
                status=TaskStatus.PENDING
            )

            # Setup mock response with proper envelope
            mock_agent = Mock()
            mock_agent.name = "TestAtomizerAgent"
            mock_agent_runtime_service.get_agent.return_value = mock_agent

            # Create AtomizerResult
            atomizer_result = AtomizerResult(
                is_atomic=True,
                node_type=NodeType.EXECUTE,
                reasoning=f"Atomic task for {task_type.value}",
                confidence=0.8
            )

            # Create envelope
            envelope = AtomizerEnvelope.create_success(
                result=atomizer_result,
                task_id=task.task_id,
                execution_id="test_execution",
                agent_type=AgentType.ATOMIZER,
                execution_metrics=ExecutionMetrics(execution_time=0.1, tokens_used=100, cost_estimate=0.01, model_calls=1),
                output_text=f"Atomic task for {task_type.value}"
            )

            mock_agent_runtime_service.execute_agent.return_value = envelope

            # Execute and verify
            result = await atomizer_service.run(task, sample_context, execution_id="test_execution")
            assert result.action == NodeAction.NOOP

            # Verify correct agent type requested
            mock_agent_runtime_service.get_agent.assert_called_with(
                task_type, AgentType.ATOMIZER
            )

    @pytest.mark.asyncio
    async def test_run_preserves_metadata(self, atomizer_service, sample_task, sample_context, mock_agent_runtime_service):
        """Test that atomizer preserves all metadata from agent response."""
        # Setup mock agent with rich metadata
        mock_agent = Mock()
        mock_agent.name = "TestAtomizerAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create AtomizerResult with metadata
        atomizer_result = AtomizerResult(
            is_atomic=True,
            node_type=NodeType.EXECUTE,
            reasoning="Detailed reasoning",
            confidence=0.95,
            metadata={
                "complexity_score": 3,
                "estimated_time": 120,
                "required_resources": ["web_search", "data_analysis"]
            }
        )

        # Create envelope
        envelope = AtomizerEnvelope.create_success(
            result=atomizer_result,
            task_id=sample_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.ATOMIZER,
            execution_metrics=ExecutionMetrics(execution_time=0.1, tokens_used=100, cost_estimate=0.01, model_calls=1),
            output_text="Detailed reasoning"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await atomizer_service.run(sample_task, sample_context, execution_id="test_execution")

        # Verify standard atomizer metadata is preserved
        assert result.metadata["confidence"] == 0.95
        assert result.metadata["atomizer_decision"] == True
        assert result.metadata["reasoning"] == "Detailed reasoning"
        assert result.metadata["node_type"] == NodeType.EXECUTE
        # Note: Additional metadata from AtomizerResult.metadata is not currently passed through