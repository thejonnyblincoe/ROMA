"""
Tests for ExecutorService.

Tests the executor service functionality including atomic task execution.
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
from roma.domain.value_objects.agent_responses import ExecutorResult
from roma.domain.value_objects.result_envelope import ExecutorEnvelope, ExecutionMetrics
from roma.application.services.recovery_manager import RecoveryResult, RecoveryAction
from roma.application.services.executor_service import ExecutorService
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager
from roma.domain.context import TaskContext


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
def executor_service(mock_agent_runtime_service, mock_recovery_manager):
    """Create executor service with mocked dependencies."""
    return ExecutorService(
        agent_runtime_service=mock_agent_runtime_service,
        recovery_manager=mock_recovery_manager
    )


@pytest.fixture
def sample_execute_task():
    """Create a sample task node for execution."""
    return TaskNode(
        task_id=str(uuid4()),
        goal="Search for latest AI research papers",
        task_type=TaskType.RETRIEVE,
        status=TaskStatus.READY,
        node_type=NodeType.EXECUTE
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
        overall_objective="Research AI trends",
        execution_id="test-executor-execution-id",
        execution_metadata={}
    )


class TestExecutorService:
    """Test ExecutorService functionality."""

    def test_agent_type_is_executor(self, executor_service):
        """Test that service has correct agent type."""
        assert executor_service.agent_type == AgentType.EXECUTOR

    @pytest.mark.asyncio
    async def test_run_execute_retrieve_task(self, executor_service, sample_context, mock_agent_runtime_service):
        """Test executor handling RETRIEVE task."""
        # Create RETRIEVE task
        retrieve_task = TaskNode(
            task_id=str(uuid4()),
            goal="Find information about quantum computing",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.READY,
            node_type=NodeType.EXECUTE
        )

        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create ExecutorResult
        executor_result = ExecutorResult(
            result="Found 15 research papers on quantum computing",
            sources=["arxiv.org", "ieee.org", "nature.com"],
            confidence=0.9,
            execution_time=45.2,
            success=True
        )

        # Create envelope
        envelope = ExecutorEnvelope.create_success(
            result=executor_result,
            task_id=retrieve_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=45.2, tokens_used=150, cost_estimate=0.03, model_calls=1),
            output_text="Found 15 research papers on quantum computing"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await executor_service.run(retrieve_task, sample_context, execution_id="test_execution")

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.COMPLETE
        # Access result from envelope, not directly from NodeResult
        assert result.envelope.result.result == "Found 15 research papers on quantum computing"
        assert result.envelope.result.sources == ["arxiv.org", "ieee.org", "nature.com"]
        assert result.envelope.result.confidence == 0.9
        assert result.metadata["confidence"] == 0.9

        # Verify agent interaction
        mock_agent_runtime_service.get_agent.assert_called_once_with(
            TaskType.RETRIEVE, AgentType.EXECUTOR
        )

    @pytest.mark.asyncio
    async def test_run_execute_think_task(self, executor_service, sample_context, mock_agent_runtime_service):
        """Test executor handling THINK task."""
        # Create THINK task
        think_task = TaskNode(
            task_id=str(uuid4()),
            goal="Analyze market trends and provide insights",
            task_type=TaskType.THINK,
            status=TaskStatus.READY,
            node_type=NodeType.EXECUTE
        )

        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create ExecutorResult
        executor_result = ExecutorResult(
            result="Market shows upward trend with 15% growth potential",
            confidence=0.85,
            execution_time=30.0,
            success=True,
            metadata={
                "reasoning": "Based on Q3 data and historical patterns",
                "key_factors": ["inflation decrease", "tech sector growth"]
            }
        )

        # Create envelope
        envelope = ExecutorEnvelope.create_success(
            result=executor_result,
            task_id=think_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=30.0, tokens_used=100, cost_estimate=0.02, model_calls=1),
            output_text="Market shows upward trend with 15% growth potential"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await executor_service.run(think_task, sample_context, execution_id="test_execution")

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.COMPLETE
        # Access result from envelope, not directly from NodeResult
        assert result.envelope.result.result == "Market shows upward trend with 15% growth potential"
        assert result.envelope.result.confidence == 0.85
        assert result.metadata["confidence"] == 0.85
        assert result.envelope.result.metadata["reasoning"] == "Based on Q3 data and historical patterns"
        assert result.envelope.result.metadata["key_factors"] == ["inflation decrease", "tech sector growth"]

    @pytest.mark.asyncio
    async def test_run_execute_write_task(self, executor_service, sample_context, mock_agent_runtime_service):
        """Test executor handling WRITE task."""
        # Create WRITE task
        write_task = TaskNode(
            task_id=str(uuid4()),
            goal="Write a summary report on AI trends",
            task_type=TaskType.WRITE,
            status=TaskStatus.READY,
            node_type=NodeType.EXECUTE
        )

        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create ExecutorResult
        executor_result = ExecutorResult(
            result="# AI Trends Report 2024\n\nKey findings include...",
            execution_time=90.0,
            success=True,
            metadata={
                "word_count": 1250,
                "format": "markdown",
                "sections": ["introduction", "trends", "conclusion"]
            }
        )

        # Create envelope
        envelope = ExecutorEnvelope.create_success(
            result=executor_result,
            task_id=write_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=90.0, tokens_used=300, cost_estimate=0.05, model_calls=1),
            output_text="# AI Trends Report 2024\n\nKey findings include..."
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await executor_service.run(write_task, sample_context, execution_id="test_execution")

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.COMPLETE
        # Access result from envelope, not directly from NodeResult
        assert result.envelope.result.result == "# AI Trends Report 2024\n\nKey findings include..."
        assert result.metadata["confidence"] == 1.0  # Default confidence
        assert result.envelope.result.metadata["word_count"] == 1250
        assert result.envelope.result.metadata["format"] == "markdown"
        assert result.envelope.result.metadata["sections"] == ["introduction", "trends", "conclusion"]

    @pytest.mark.asyncio
    async def test_run_execute_code_interpret_task(self, executor_service, sample_context, mock_agent_runtime_service):
        """Test executor handling CODE_INTERPRET task."""
        # Create CODE_INTERPRET task
        code_task = TaskNode(
            task_id=str(uuid4()),
            goal="Analyze dataset and generate statistics",
            task_type=TaskType.CODE_INTERPRET,
            status=TaskStatus.READY,
            node_type=NodeType.EXECUTE
        )

        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create ExecutorResult
        executor_result = ExecutorResult(
            result="Analysis complete: Mean=42.5, Std=12.3, Correlation=0.78",
            execution_time=75.0,
            success=True,
            metadata={
                "code_executed": "import pandas as pd\ndf.describe()",
                "execution_output": "{'mean': 42.5, 'std': 12.3}",
                "files_generated": ["analysis_output.csv"]
            }
        )

        # Create envelope
        envelope = ExecutorEnvelope.create_success(
            result=executor_result,
            task_id=code_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=75.0, tokens_used=120, cost_estimate=0.03, model_calls=1),
            output_text="Analysis complete: Mean=42.5, Std=12.3, Correlation=0.78"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await executor_service.run(code_task, sample_context, execution_id="test_execution")

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.COMPLETE
        # Access result from envelope, not directly from NodeResult
        assert result.envelope.result.result == "Analysis complete: Mean=42.5, Std=12.3, Correlation=0.78"
        assert result.metadata["confidence"] == 1.0  # Default confidence
        assert result.envelope.result.metadata["code_executed"] == "import pandas as pd\ndf.describe()"
        assert result.envelope.result.metadata["files_generated"] == ["analysis_output.csv"]

    @pytest.mark.asyncio
    async def test_run_execute_image_generation_task(self, executor_service, sample_context, mock_agent_runtime_service):
        """Test executor handling IMAGE_GENERATION task."""
        # Create IMAGE_GENERATION task
        image_task = TaskNode(
            task_id=str(uuid4()),
            goal="Generate visualization of data trends",
            task_type=TaskType.IMAGE_GENERATION,
            status=TaskStatus.READY,
            node_type=NodeType.EXECUTE
        )

        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create ExecutorResult
        executor_result = ExecutorResult(
            result="Generated trend visualization chart",
            execution_time=120.0,
            success=True,
            metadata={
                "image_path": "/tmp/trend_chart.png",
                "image_format": "PNG",
                "dimensions": {"width": 800, "height": 600}
            }
        )

        # Create envelope
        envelope = ExecutorEnvelope.create_success(
            result=executor_result,
            task_id=image_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=120.0, tokens_used=80, cost_estimate=0.04, model_calls=1),
            output_text="Generated trend visualization chart"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await executor_service.run(image_task, sample_context, execution_id="test_execution")

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.COMPLETE
        # Access result from envelope, not directly from NodeResult
        assert result.envelope.result.result == "Generated trend visualization chart"
        assert result.metadata["confidence"] == 1.0  # Default confidence
        assert result.envelope.result.metadata["image_path"] == "/tmp/trend_chart.png"
        assert result.envelope.result.metadata["image_format"] == "PNG"
        assert result.envelope.result.metadata["dimensions"] == {"width": 800, "height": 600}

    @pytest.mark.asyncio
    async def test_run_handles_execution_error(self, executor_service, sample_execute_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test error handling when execution fails."""
        # Setup mock to raise exception
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.side_effect = Exception("Execution failed")

        # Setup recovery manager to return failure action
        recovery_result = RecoveryResult(
            action=RecoveryAction.FAIL_PERMANENTLY,
            reasoning="Task exceeded retry limit",
            metadata={}
        )
        mock_recovery_manager.handle_failure.return_value = recovery_result

        # Execute the service
        result = await executor_service.run(sample_execute_task, sample_context, execution_id="test_execution")

        # Verify error handling
        assert result.action == NodeAction.FAIL
        assert "Execution failed" in result.error
        assert result.envelope is None

    @pytest.mark.asyncio
    async def test_run_with_retry_on_failure(self, executor_service, sample_execute_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test retry mechanism on execution failure."""
        # Setup mock to return retry action
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.side_effect = Exception("First failure")

        # Setup recovery manager to retry once
        recovery_result = RecoveryResult(
            action=RecoveryAction.RETRY,
            reasoning="Will retry task execution",
            metadata={"retry_attempt": 1}
        )
        mock_recovery_manager.handle_failure.return_value = recovery_result

        # Execute the service
        result = await executor_service.run(sample_execute_task, sample_context, execution_id="test_execution")

        # Verify retry response
        assert result.action == NodeAction.RETRY
        assert "First failure" in result.error
        assert result.metadata["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_run_with_invalid_agent_response(self, executor_service, sample_execute_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test handling of invalid agent response."""
        # Setup mock agent with invalid response
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.return_value = {
            "invalid_field": "should not be here",
            # Missing required "result" field
        }

        # Setup recovery manager to handle the failure
        recovery_result = RecoveryResult(
            action=RecoveryAction.FAIL_PERMANENTLY,
            reasoning="Invalid agent response format",
            metadata={}
        )
        mock_recovery_manager.handle_failure.return_value = recovery_result

        # Execute the service
        result = await executor_service.run(sample_execute_task, sample_context, execution_id="test_execution")

        # Verify error handling
        assert result.action == NodeAction.FAIL
        assert "Executor returned invalid envelope" in result.error

    @pytest.mark.asyncio
    async def test_run_preserves_all_metadata(self, executor_service, sample_execute_task, sample_context, mock_agent_runtime_service):
        """Test that executor preserves all metadata from agent response."""
        # Setup mock agent with rich metadata
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create ExecutorResult with rich metadata
        executor_result = ExecutorResult(
            result="Task completed successfully",
            confidence=0.95,
            execution_time=120.5,
            tokens_used=1500,
            success=True,
            metadata={
                "cost": 0.05,
                "tools_used": ["web_search", "calculator"],
                "api_calls": 3,
                "cache_hits": 2
            }
        )

        # Create envelope
        envelope = ExecutorEnvelope.create_success(
            result=executor_result,
            task_id=sample_execute_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=120.5, tokens_used=1500, cost_estimate=0.05, model_calls=1),
            output_text="Task completed successfully"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await executor_service.run(sample_execute_task, sample_context, execution_id="test_execution")

        # Verify all metadata is preserved
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.COMPLETE
        assert result.envelope.result.result == "Task completed successfully"
        assert result.metadata["confidence"] == 0.95
        assert result.envelope.result.metadata["cost"] == 0.05
        assert result.envelope.result.metadata["tools_used"] == ["web_search", "calculator"]
        assert result.envelope.result.metadata["api_calls"] == 3
        assert result.envelope.result.metadata["cache_hits"] == 2

    @pytest.mark.asyncio
    async def test_run_with_different_task_types(self, executor_service, sample_context, mock_agent_runtime_service):
        """Test executor behavior with all task types."""
        task_types = [
            TaskType.RETRIEVE,
            TaskType.THINK,
            TaskType.WRITE,
            TaskType.CODE_INTERPRET,
            TaskType.IMAGE_GENERATION
        ]

        for task_type in task_types:
            # Create task with specific type
            task = TaskNode(
                task_id=str(uuid4()),
                goal=f"Execute {task_type.value} task",
                task_type=task_type,
                status=TaskStatus.READY,
                node_type=NodeType.EXECUTE
            )

            # Setup mock response with proper envelope structure
            mock_agent = Mock()
            mock_agent.name = "TestExecutorAgent"
            mock_agent_runtime_service.get_agent.return_value = mock_agent

            # Create ExecutorResult
            executor_result = ExecutorResult(
                result=f"Completed {task_type.value} task",
                success=True,
                metadata={"task_type": task_type.value}
            )

            # Create envelope
            envelope = ExecutorEnvelope.create_success(
                result=executor_result,
                task_id=task.task_id,
                execution_id="test_execution",
                agent_type=AgentType.EXECUTOR,
                execution_metrics=ExecutionMetrics(execution_time=30.0, tokens_used=50, cost_estimate=0.01, model_calls=1),
                output_text=f"Completed {task_type.value} task"
            )

            mock_agent_runtime_service.execute_agent.return_value = envelope

            # Execute and verify
            result = await executor_service.run(task, sample_context, execution_id="test_execution")
            assert result.action == NodeAction.COMPLETE
            assert result.envelope.result.result == f"Completed {task_type.value} task"

            # Verify correct agent type requested
            mock_agent_runtime_service.get_agent.assert_called_with(
                task_type, AgentType.EXECUTOR
            )

    @pytest.mark.asyncio
    async def test_run_with_empty_result(self, executor_service, sample_execute_task, sample_context, mock_agent_runtime_service):
        """Test handling when agent returns empty result."""
        # Setup mock agent response with empty result
        mock_agent = Mock()
        mock_agent.name = "TestExecutorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create ExecutorResult with empty result
        executor_result = ExecutorResult(
            result="",
            success=True,
            metadata={"reasoning": "No output generated"}
        )

        # Create envelope
        envelope = ExecutorEnvelope.create_success(
            result=executor_result,
            task_id=sample_execute_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=10.0, tokens_used=20, cost_estimate=0.001, model_calls=1),
            output_text=""
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service
        result = await executor_service.run(sample_execute_task, sample_context, execution_id="test_execution")

        # Should still complete but with empty output
        assert result.action == NodeAction.COMPLETE
        assert result.envelope.result.result == ""
        assert result.envelope.result.metadata["reasoning"] == "No output generated"