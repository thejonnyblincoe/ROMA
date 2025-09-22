"""
Tests for AggregatorService.

Tests the aggregator service functionality including child result aggregation,
queue management, and partial aggregation logic.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from uuid import uuid4
from collections import deque

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.agent_responses import AggregatorResult, ExecutorResult
from roma.domain.value_objects.result_envelope import AggregatorEnvelope, ExecutorEnvelope, ExecutionMetrics
from roma.domain.value_objects.child_evaluation_result import ChildEvaluationResult
from roma.application.services.recovery_manager import RecoveryResult, RecoveryAction
from roma.application.services.aggregator_service import AggregatorService
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
def aggregator_service(mock_agent_runtime_service, mock_recovery_manager):
    """Create aggregator service with mocked dependencies."""
    service = AggregatorService(
        agent_runtime_service=mock_agent_runtime_service,
        recovery_manager=mock_recovery_manager
    )

    # Create mock callbacks
    def mock_get_parent(parent_id: str):
        # Return a mock parent node
        return TaskNode(
            task_id=parent_id,
            goal="Parent task",
            task_type=TaskType.THINK,
            status=TaskStatus.AGGREGATING,
            node_type=NodeType.PLAN
        )

    def mock_get_children(parent_id: str):
        # Return some mock child nodes
        return [
            TaskNode(
                task_id="child_1",
                goal="Child task 1",
                task_type=TaskType.THINK,
                status=TaskStatus.COMPLETED,
                node_type=NodeType.EXECUTE
            ),
            TaskNode(
                task_id="child_2",
                goal="Child task 2",
                task_type=TaskType.THINK,
                status=TaskStatus.COMPLETED,
                node_type=NodeType.EXECUTE
            )
        ]

    def mock_get_result(task_id: str):
        # Return a mock result envelope
        return None

    async def mock_transition_status(task_id: str, status: TaskStatus):
        pass

    async def mock_handle_result(result: NodeResult):
        pass

    # Set up orchestrator callbacks properly
    service.set_orchestrator_callbacks(
        get_parent=mock_get_parent,
        get_children=mock_get_children,
        get_result=mock_get_result,
        transition_status=mock_transition_status,
        handle_result=mock_handle_result
    )

    # Mock the recovery manager to return AGGREGATE_ALL for terminal children evaluation
    mock_recovery_manager.evaluate_terminal_children.return_value = ChildEvaluationResult.AGGREGATE_ALL

    return service


@pytest.fixture
def sample_parent_task():
    """Create a sample parent task node."""
    return TaskNode(
        task_id=str(uuid4()),
        goal="Comprehensive market analysis",
        task_type=TaskType.THINK,
        status=TaskStatus.AGGREGATING,
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


class TestAggregatorService:
    """Test AggregatorService functionality."""

    def test_agent_type_is_aggregator(self, aggregator_service):
        """Test that service has correct agent type."""
        assert aggregator_service.agent_type == AgentType.AGGREGATOR

    def test_aggregation_queue_initialization(self, aggregator_service):
        """Test that aggregation queue is properly initialized."""
        assert isinstance(aggregator_service._aggregation_queue, deque)
        assert len(aggregator_service._aggregation_queue) == 0

    @pytest.mark.asyncio
    async def test_run_aggregates_child_results(self, aggregator_service, sample_parent_task, sample_context, mock_agent_runtime_service):
        """Test aggregator combining child task results."""
        # Create completed child tasks
        child1 = TaskNode(
            task_id=str(uuid4()),
            goal="Research market trends",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.COMPLETED,
            parent_id=sample_parent_task.task_id,
            result={"output": "Market shows 15% growth", "confidence": 0.9}
        )

        child2 = TaskNode(
            task_id=str(uuid4()),
            goal="Analyze competitors",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED,
            parent_id=sample_parent_task.task_id,
            result={"output": "3 main competitors identified", "confidence": 0.8}
        )

        # Create child result envelopes
        child1_envelope = ExecutorEnvelope.create_success(
            result=ExecutorResult(
                result="Market shows 15% growth",
                confidence=0.9,
                execution_time=30.0,
                success=True
            ),
            task_id=child1.task_id,
            execution_id="child1_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=30.0, tokens_used=100, cost_estimate=0.02, model_calls=1),
            output_text="Market shows 15% growth"
        )

        child2_envelope = ExecutorEnvelope.create_success(
            result=ExecutorResult(
                result="3 main competitors identified",
                confidence=0.8,
                execution_time=25.0,
                success=True
            ),
            task_id=child2.task_id,
            execution_id="child2_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=25.0, tokens_used=80, cost_estimate=0.015, model_calls=1),
            output_text="3 main competitors identified"
        )

        child_envelopes = [child1_envelope, child2_envelope]

        # Setup mock agent response with proper envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestAggregatorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create AggregatorResult
        aggregator_result = AggregatorResult(
            synthesized_result="Combined analysis: Market growth of 15% with 3 key competitors",
            summary="Market shows 15% growth with 3 main competitors identified",
            confidence=0.85,
            sources_used=[child1.task_id, child2.task_id],
            quality_score=0.9
        )

        # Create envelope
        envelope = AggregatorEnvelope.create_success(
            result=aggregator_result,
            task_id=sample_parent_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.AGGREGATOR,
            execution_metrics=ExecutionMetrics(execution_time=45.0, tokens_used=200, cost_estimate=0.04, model_calls=1),
            output_text="Combined analysis: Market growth of 15% with 3 key competitors"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute the service with child envelopes
        result = await aggregator_service.run(sample_parent_task, sample_context, execution_id="test_execution", child_envelopes=child_envelopes)

        # Verify the result
        assert isinstance(result, NodeResult)
        assert result.action == NodeAction.AGGREGATE
        assert result.envelope.result.synthesized_result == "Combined analysis: Market growth of 15% with 3 key competitors"
        assert result.metadata["confidence"] == 0.85
        assert result.envelope.result.sources_used == [child1.task_id, child2.task_id]
        assert result.envelope.result.quality_score == 0.9

        # Verify agent interaction
        mock_agent_runtime_service.get_agent.assert_called_once_with(
            sample_parent_task.task_type, AgentType.AGGREGATOR
        )

    @pytest.mark.asyncio
    async def test_notify_child_completion_adds_to_queue(self, aggregator_service):
        """Test that child completion notification adds parent to queue."""
        parent_id = str(uuid4())

        # Notify child completion
        await aggregator_service.notify_child_completion(parent_id)

        # Verify parent is in queue
        assert len(aggregator_service._aggregation_queue) == 1
        assert aggregator_service._aggregation_queue[0] == (parent_id, False)

    @pytest.mark.asyncio
    async def test_notify_child_completion_no_duplicates(self, aggregator_service):
        """Test that duplicate notifications don't create multiple queue entries."""
        parent_id = str(uuid4())

        # Notify same parent multiple times
        await aggregator_service.notify_child_completion(parent_id)
        await aggregator_service.notify_child_completion(parent_id)
        await aggregator_service.notify_child_completion(parent_id)

        # Should only have one entry
        assert len(aggregator_service._aggregation_queue) == 1
        assert aggregator_service._aggregation_queue[0] == (parent_id, False)

    @pytest.mark.asyncio
    async def test_process_aggregation_queue_handles_ready_parents(self, aggregator_service, sample_context):
        """Test processing aggregation queue for ready parents."""
        parent_id = str(uuid4())
        child_id = str(uuid4())

        # Create parent with completed children
        parent_task = TaskNode(
            task_id=parent_id,
            goal="Parent task",
            task_type=TaskType.THINK,
            status=TaskStatus.EXECUTING,
            node_type=NodeType.PLAN
        )

        completed_child = TaskNode(
            task_id=child_id,
            goal="Child task",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED,
            parent_id=parent_id,
            result={"output": "Child result"}
        )

        # Create a successful result envelope for the child
        from roma.domain.value_objects.result_envelope import ResultEnvelope, ExecutionMetrics
        execution_metrics = ExecutionMetrics(
            execution_time=1.0,
            tokens_used=100,
            model_calls=1,
            cost_estimate=0.01
        )
        child_envelope = ResultEnvelope.create_success(
            result={"output": "Child result"},
            task_id=child_id,
            execution_id="test_exec",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=execution_metrics
        )

        # Setup mock callbacks with proper mocks
        from unittest.mock import MagicMock, AsyncMock

        mock_get_parent = MagicMock(return_value=parent_task)
        mock_get_children = MagicMock(return_value=[completed_child])
        mock_get_result = MagicMock(return_value=child_envelope)
        mock_transition_status = AsyncMock()
        mock_handle_result = AsyncMock()

        aggregator_service.set_orchestrator_callbacks(
            get_parent=mock_get_parent,
            get_children=mock_get_children,
            get_result=mock_get_result,
            transition_status=mock_transition_status,
            handle_result=mock_handle_result
        )

        # Add to queue and process
        await aggregator_service.notify_child_completion(parent_id)
        await aggregator_service.process_aggregation_queue(sample_context)

        # Verify parent status was transitioned to AGGREGATING
        mock_transition_status.assert_called_with(parent_id, TaskStatus.AGGREGATING)



    @pytest.mark.asyncio
    async def test_run_with_partial_aggregation(self, aggregator_service, sample_parent_task, sample_context, mock_agent_runtime_service):
        """Test aggregator handling partial aggregation scenario."""
        # Create mix of completed and failed children
        completed_child = TaskNode(
            task_id=str(uuid4()),
            goal="Successful child",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED,
            parent_id=sample_parent_task.task_id,
            result={"output": "Success result", "confidence": 0.9}
        )

        failed_child = TaskNode(
            task_id=str(uuid4()),
            goal="Failed child",
            task_type=TaskType.THINK,
            status=TaskStatus.FAILED,
            parent_id=sample_parent_task.task_id
        )

        # Create envelope for only the successful child (partial aggregation)
        completed_child_envelope = ExecutorEnvelope.create_success(
            result=ExecutorResult(
                result="Success result",
                confidence=0.9,
                execution_time=40.0,
                success=True
            ),
            task_id=completed_child.task_id,
            execution_id="completed_child_execution",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=ExecutionMetrics(execution_time=40.0, tokens_used=90, cost_estimate=0.018, model_calls=1),
            output_text="Success result"
        )

        # Only include successful child envelope for partial aggregation
        child_envelopes = [completed_child_envelope]

        # Setup mock agent response for partial aggregation with envelope structure
        mock_agent = Mock()
        mock_agent.name = "TestAggregatorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent

        # Create AggregatorResult for partial aggregation
        aggregator_result = AggregatorResult(
            synthesized_result="Partial aggregation: Only successful results included",
            summary="Aggregated results from 1 of 2 completed child tasks",
            confidence=0.7,
            sources_used=[completed_child.task_id],
            gaps_identified=["Failed child task results unavailable"],
            quality_score=0.5
        )

        # Create envelope
        envelope = AggregatorEnvelope.create_success(
            result=aggregator_result,
            task_id=sample_parent_task.task_id,
            execution_id="test_execution",
            agent_type=AgentType.AGGREGATOR,
            execution_metrics=ExecutionMetrics(execution_time=35.0, tokens_used=180, cost_estimate=0.035, model_calls=1),
            output_text="Partial aggregation: Only successful results included"
        )

        mock_agent_runtime_service.execute_agent.return_value = envelope

        # Execute with partial aggregation - pass child envelopes and is_partial flag
        result = await aggregator_service.run(sample_parent_task, sample_context, execution_id="test_execution", child_envelopes=child_envelopes, is_partial=True)

        # Verify partial aggregation result
        assert result.action == NodeAction.AGGREGATE
        assert result.envelope.result.synthesized_result == "Partial aggregation: Only successful results included"
        assert result.envelope.result.gaps_identified == ["Failed child task results unavailable"]
        assert result.envelope.result.quality_score == 0.5

    @pytest.mark.asyncio
    async def test_run_handles_aggregation_error(self, aggregator_service, sample_parent_task, sample_context, mock_agent_runtime_service, mock_recovery_manager):
        """Test error handling when aggregation fails."""
        # Setup children
        child_id = str(uuid4())
        child = TaskNode(
            task_id=child_id,
            goal="Child task",
            task_type=TaskType.THINK,
            status=TaskStatus.COMPLETED,
            parent_id=sample_parent_task.task_id,
            result={"output": "Child result"}
        )

        # Create child envelope for aggregation
        from roma.domain.value_objects.result_envelope import ResultEnvelope, ExecutionMetrics
        execution_metrics = ExecutionMetrics(
            execution_time=1.0,
            tokens_used=100,
            model_calls=1,
            cost_estimate=0.01
        )
        child_envelope = ResultEnvelope.create_success(
            result={"output": "Child result"},
            task_id=child_id,
            execution_id="test_exec",
            agent_type=AgentType.EXECUTOR,
            execution_metrics=execution_metrics
        )

        # Setup mock to raise exception
        mock_agent = Mock()
        mock_agent.name = "TestAggregatorAgent"
        mock_agent_runtime_service.get_agent.return_value = mock_agent
        mock_agent_runtime_service.execute_agent.side_effect = Exception("Aggregation failed")

        # Setup recovery manager with new interface
        recovery_result = RecoveryResult(
            action=RecoveryAction.FAIL_PERMANENTLY,
            reasoning="Aggregation failed after retry limit",
            metadata={}
        )
        mock_recovery_manager.handle_failure.return_value = recovery_result

        # Execute the service with child_envelopes to trigger aggregation
        result = await aggregator_service.run(
            sample_parent_task,
            sample_context,
            execution_id="test_execution",
            child_envelopes=[child_envelope]
        )

        # Verify error handling
        assert result.action == NodeAction.FAIL
        assert "Aggregation failed" in result.error
        assert result.envelope is None

    @pytest.mark.asyncio
    async def test_run_with_no_children(self, aggregator_service, sample_parent_task, sample_context):
        """Test aggregator behavior when no children exist."""
        # Setup no children
        aggregator_service._get_children_callback.return_value = []

        # Execute the service
        result = await aggregator_service.run(sample_parent_task, sample_context, execution_id="test_execution")

        # Should handle gracefully
        assert result.action == NodeAction.FAIL
        assert "No child results provided for aggregation" in result.error
        assert result.envelope is None

    @pytest.mark.asyncio
    async def test_run_with_all_failed_children(self, aggregator_service, sample_parent_task, sample_context):
        """Test aggregator behavior when all children failed."""
        # Create all failed children
        failed_children = [
            TaskNode(
                task_id=str(uuid4()),
                goal=f"Failed child {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.FAILED,
                parent_id=sample_parent_task.task_id
            )
            for i in range(3)
        ]

        aggregator_service._get_children_callback.return_value = failed_children

        # Execute the service
        result = await aggregator_service.run(sample_parent_task, sample_context, execution_id="test_execution")

        # Should fail when all children failed and partial aggregation disabled
        assert result.action == NodeAction.FAIL
        assert "No child results provided for aggregation" in result.error

    @pytest.mark.skip(reason="Method get_aggregation_stats not implemented in service")
    async def test_get_aggregation_stats(self, aggregator_service):
        """Test getting aggregation service statistics."""
        # Add some items to queue
        await aggregator_service.notify_child_completion("parent1")
        await aggregator_service.notify_child_completion("parent2")

        stats = aggregator_service.get_aggregation_stats()

        assert "queue_size" in stats
        assert "partial_aggregation_enabled" in stats
        assert stats["queue_size"] == 2

    @pytest.mark.skip(reason="Method clear_aggregation_queue not implemented in service")
    async def test_clear_aggregation_queue(self, aggregator_service):
        """Test clearing aggregation queue."""
        # Add items to queue
        await aggregator_service.notify_child_completion("parent1")
        await aggregator_service.notify_child_completion("parent2")

        assert len(aggregator_service._aggregation_queue) == 2

        # Clear queue
        aggregator_service.clear_aggregation_queue()

        assert len(aggregator_service._aggregation_queue) == 0