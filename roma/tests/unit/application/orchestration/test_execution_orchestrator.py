"""
Tests for ExecutionOrchestrator - Main Orchestration Logic.

Tests the core orchestration loop, graph mutations, result handling,
and aggregation management.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from roma.application.orchestration.execution_orchestrator import ExecutionOrchestrator
from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
from roma.application.orchestration.task_node_processor import TaskNodeProcessor
from roma.application.services.context_builder_service import ContextBuilderService, TaskContext
from roma.application.services.recovery_manager import RecoveryManager
from roma.application.services.event_store import InMemoryEventStore
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.execution_result import ExecutionResult
from roma.domain.value_objects.config.execution_config import ExecutionConfig
from roma.domain.value_objects.result_envelope import ResultEnvelope
from roma.domain.value_objects.result_envelope import ExecutionMetrics
from roma.domain.value_objects.agent_type import AgentType


@pytest.fixture
def mock_graph_state_manager():
    """Mock GraphStateManager."""
    mock = AsyncMock(spec=GraphStateManager)
    return mock


@pytest.fixture
def mock_parallel_engine():
    """Mock ParallelExecutionEngine."""
    mock = AsyncMock(spec=ParallelExecutionEngine)
    return mock


@pytest.fixture
def mock_node_processor():
    """Mock TaskNodeProcessor."""
    mock = AsyncMock(spec=TaskNodeProcessor)
    return mock


@pytest.fixture
def mock_context_builder():
    """Mock ContextBuilderService."""
    mock = AsyncMock(spec=ContextBuilderService)
    return mock


@pytest.fixture
def mock_recovery_manager():
    """Mock RecoveryManager."""
    mock = AsyncMock(spec=RecoveryManager)
    return mock


@pytest.fixture
def mock_event_store():
    """Mock InMemoryEventStore."""
    return AsyncMock(spec=InMemoryEventStore)


@pytest.fixture
def execution_config():
    """Test ExecutionConfig."""
    return ExecutionConfig(
        max_concurrent_tasks=5,
        max_iterations=10,
        max_subtasks_per_node=3,
        max_tasks_per_level=5,
        total_timeout=60
    )


@pytest.fixture
def orchestrator(
    mock_graph_state_manager,
    mock_parallel_engine,
    mock_node_processor,
    mock_context_builder,
    mock_recovery_manager,
    mock_event_store,
    execution_config
):
    """ExecutionOrchestrator instance with mocked dependencies."""
    return ExecutionOrchestrator(
        graph_state_manager=mock_graph_state_manager,
        parallel_engine=mock_parallel_engine,
        node_processor=mock_node_processor,
        context_builder=mock_context_builder,
        recovery_manager=mock_recovery_manager,
        event_store=mock_event_store,
        execution_config=execution_config
    )


@pytest.fixture
def sample_task():
    """Sample TaskNode for testing."""
    return TaskNode(
        task_id="test_task",
        goal="Test goal",
        task_type=TaskType.THINK,
        status=TaskStatus.PENDING
    )


@pytest.fixture
def sample_result_envelope():
    """Sample ResultEnvelope for testing."""
    return ResultEnvelope.create_success(
        result={"output": "test result"},
        task_id="test_task",
        execution_id="test_exec",
        agent_type=AgentType.EXECUTOR,
        execution_metrics=ExecutionMetrics(execution_time=1.0, tokens_used=100),
        artifacts=[],
        output_text="Test output"
    )


class TestExecutionOrchestrator:
    """Test cases for ExecutionOrchestrator."""

    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator, execution_config):
        """Test orchestrator initialization."""
        assert orchestrator.execution_config == execution_config
        assert orchestrator.iterations == 0
        assert orchestrator.total_nodes_processed == 0
        assert len(orchestrator.result_cache) == 0
        assert len(orchestrator._completed_node_ids) == 0
        assert len(orchestrator._failed_node_ids) == 0

    @pytest.mark.asyncio
    async def test_execute_simple_task(self, orchestrator, sample_task, mock_graph_state_manager,
                                       mock_context_builder, mock_parallel_engine, sample_result_envelope):
        """Test executing a simple task that completes in one iteration."""
        # Setup mocks
        ready_task = sample_task.model_copy(update={"status": TaskStatus.READY})
        completed_task = sample_task.model_copy(update={"status": TaskStatus.COMPLETED})

        mock_graph_state_manager.get_ready_nodes.side_effect = [
            [ready_task],  # First iteration: task is ready
            []  # Second iteration: no more tasks
        ]
        # get_all_nodes needs to return PENDING task initially, then COMPLETED after processing
        mock_graph_state_manager.get_all_nodes.side_effect = [
            [sample_task],  # Initially: task is PENDING
            [ready_task],   # During iteration: task is READY
            [completed_task],  # After processing: task is COMPLETED
            [completed_task],  # For result calculation
        ]
        # Mock state transitions
        mock_graph_state_manager.transition_node_status.return_value = None

        mock_context_builder.build_context.return_value = TaskContext(
            task=sample_task,
            overall_objective="test objective",
            context_items=[],
            execution_metadata={}
        )

        # Mock parallel execution returning a complete result
        complete_result = NodeResult.success(
            envelope=sample_result_envelope,
            agent_name="test_agent",
            agent_type="executor",
            metadata={"node_id": sample_task.task_id}
        )
        mock_parallel_engine.execute_ready_nodes.return_value = [complete_result]

        # Execute
        result = await orchestrator.execute(sample_task, "test objective")

        # Verify basic execution occurred
        assert isinstance(result, ExecutionResult)
        assert orchestrator.iterations >= 1
        assert result.total_nodes == 1

        # Verify graph operations
        mock_graph_state_manager.add_node.assert_called_once_with(sample_task)

    @pytest.mark.asyncio
    async def test_execute_with_subtasks(self, orchestrator, sample_task, mock_graph_state_manager,
                                         mock_context_builder, mock_parallel_engine, sample_result_envelope):
        """Test executing a task that creates subtasks."""
        # Create subtasks
        subtask1 = TaskNode(task_id="sub1", goal="Subtask 1", task_type=TaskType.THINK, parent_id=sample_task.task_id)
        subtask2 = TaskNode(task_id="sub2", goal="Subtask 2", task_type=TaskType.THINK, parent_id=sample_task.task_id)

        # Setup mocks for multi-iteration execution
        ready_task = sample_task.model_copy(update={"status": TaskStatus.READY})
        completed_task = sample_task.model_copy(update={"status": TaskStatus.COMPLETED})
        ready_subtask1 = subtask1.model_copy(update={"status": TaskStatus.READY})
        ready_subtask2 = subtask2.model_copy(update={"status": TaskStatus.READY})

        mock_graph_state_manager.get_ready_nodes.side_effect = [
            [ready_task],        # Iteration 1: parent task ready
            [ready_subtask1, ready_subtask2], # Iteration 2: subtasks ready
            []                    # Iteration 3: no more tasks
        ]
        # get_all_nodes mock for completion checking
        mock_graph_state_manager.get_all_nodes.side_effect = [
            [sample_task],  # Initially: parent task is PENDING
            [ready_task],   # During iteration 1: parent task is READY
            [sample_task, subtask1, subtask2],  # After subtask creation: parent + children
            [sample_task, ready_subtask1, ready_subtask2],  # During iteration 2: subtasks READY
            [completed_task, subtask1.model_copy(update={"status": TaskStatus.COMPLETED}), subtask2.model_copy(update={"status": TaskStatus.COMPLETED})],  # All completed
            [completed_task, subtask1.model_copy(update={"status": TaskStatus.COMPLETED}), subtask2.model_copy(update={"status": TaskStatus.COMPLETED})],  # For result calculation
        ]
        # Mock state transitions
        mock_graph_state_manager.transition_node_status.return_value = None
        mock_graph_state_manager.add_node.return_value = None

        mock_context_builder.build_context.return_value = TaskContext(
            task=sample_task,
            overall_objective="test objective",
            context_items=[],
            execution_metadata={}
        )

        # Mock first iteration - parent returns ADD_SUBTASKS
        add_subtasks_result = NodeResult(
            action=NodeAction.ADD_SUBTASKS,
            envelope=sample_result_envelope,
            new_nodes=[subtask1, subtask2],
            metadata={"node_id": sample_task.task_id}
        )

        # Mock second iteration - subtasks complete
        subtask1_result = NodeResult.success(
            envelope=sample_result_envelope,
            agent_name="test_agent",
            metadata={"node_id": subtask1.task_id}
        )
        subtask2_result = NodeResult.success(
            envelope=sample_result_envelope,
            agent_name="test_agent",
            metadata={"node_id": subtask2.task_id}
        )

        mock_parallel_engine.execute_ready_nodes.side_effect = [
            [add_subtasks_result],
            [subtask1_result, subtask2_result]
        ]

        # Execute
        result = await orchestrator.execute(sample_task, "test objective")

        # Verify
        assert isinstance(result, ExecutionResult)
        assert orchestrator.iterations >= 2

        # Verify task was added to graph - should have been called multiple times (root + subtasks)
        assert mock_graph_state_manager.add_node.call_count >= 2  # Root task + subtasks
        # Check that root task was added first
        first_call = mock_graph_state_manager.add_node.call_args_list[0]
        assert first_call[0][0] == sample_task

    @pytest.mark.asyncio
    async def test_execution_limits(self, orchestrator, sample_task, mock_graph_state_manager,
                                    mock_context_builder, mock_parallel_engine, sample_result_envelope):
        """Test that execution limits are enforced."""
        # Setup infinite loop scenario - task never completes
        ready_task = sample_task.model_copy(update={"status": TaskStatus.READY})
        mock_graph_state_manager.get_ready_nodes.return_value = [ready_task]
        # Mock get_all_nodes to always return a PENDING task (infinite loop)
        mock_graph_state_manager.get_all_nodes.return_value = [sample_task]
        # Mock state transitions
        mock_graph_state_manager.transition_node_status.return_value = None

        mock_context_builder.build_context.return_value = TaskContext(
            task=sample_task,
            overall_objective="test objective",
            context_items=[],
            execution_metadata={}
        )

        # Mock parallel execution that returns retry (infinite loop)
        retry_result = NodeResult.retry(
            error="Test retry",
            agent_name="test_agent",
            metadata={"node_id": sample_task.task_id}
        )
        mock_parallel_engine.execute_ready_nodes.return_value = [retry_result]

        # Execute - should hit iteration limit
        result = await orchestrator.execute(sample_task, "test objective")

        # Verify limits were enforced
        assert orchestrator.iterations == orchestrator.execution_config.max_iterations

    @pytest.mark.asyncio
    async def test_aggregation_handling(self, orchestrator, sample_task, mock_graph_state_manager,
                                        mock_context_builder, mock_parallel_engine, mock_node_processor,
                                        sample_result_envelope):
        """Test parent-child aggregation setup."""
        # This test verifies the aggregation infrastructure is in place
        # Actual aggregation logic would be tested in integration tests

        # Setup parent with completed children scenario
        child1 = TaskNode(task_id="child1", goal="Child 1", task_type=TaskType.THINK, parent_id=sample_task.task_id)
        child2 = TaskNode(task_id="child2", goal="Child 2", task_type=TaskType.THINK, parent_id=sample_task.task_id)

        # Add nodes to orchestrator tracking
        orchestrator._parent_to_children[sample_task.task_id] = {child1.task_id, child2.task_id}
        orchestrator._child_to_parent[child1.task_id] = sample_task.task_id
        orchestrator._child_to_parent[child2.task_id] = sample_task.task_id

        # Mark children as completed
        orchestrator._completed_node_ids.add(child1.task_id)
        orchestrator._completed_node_ids.add(child2.task_id)
        orchestrator.result_cache[child1.task_id] = sample_result_envelope
        orchestrator.result_cache[child2.task_id] = sample_result_envelope

        # Setup mocks for simple completion
        mock_graph_state_manager.get_ready_nodes.return_value = []
        mock_context_builder.build_context.return_value = TaskContext(
            task=sample_task,
            overall_objective="test objective",
            context_items=[],
            execution_metadata={}
        )

        # Execute
        result = await orchestrator.execute(sample_task, "test objective")

        # Verify aggregation infrastructure exists
        assert hasattr(orchestrator, '_parent_to_children')
        assert hasattr(orchestrator, '_child_to_parent')
        assert hasattr(orchestrator, '_aggregation_queue')
        assert sample_task.task_id in orchestrator._parent_to_children

    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator, sample_task, mock_graph_state_manager,
                                  mock_context_builder, mock_parallel_engine):
        """Test error handling during execution."""
        # Setup mocks to raise exception
        mock_graph_state_manager.add_node.side_effect = Exception("Test error")
        mock_context_builder.build_context.return_value = AsyncMock()

        # Execute
        result = await orchestrator.execute(sample_task, "test objective")

        # Verify error handling
        assert isinstance(result, ExecutionResult)
        assert result.success is False
        assert len(result.error_details) > 0
        assert "Test error" in result.error_details[0]["message"]

    @pytest.mark.asyncio
    async def test_get_orchestration_metrics(self, orchestrator, sample_task):
        """Test orchestration metrics collection."""
        # Set some state
        orchestrator.iterations = 5
        orchestrator.total_nodes_processed = 10
        orchestrator._completed_node_ids.add("node1")
        orchestrator._completed_node_ids.add("node2")
        orchestrator._failed_node_ids.add("node3")
        orchestrator.result_cache["node1"] = "result1"
        orchestrator.start_time = datetime.now(timezone.utc)

        # Get metrics
        metrics = orchestrator.get_orchestration_metrics()

        # Verify
        assert metrics["iterations"] == 5
        assert metrics["total_nodes_processed"] == 10
        assert metrics["completed_nodes"] == 2
        assert metrics["failed_nodes"] == 1
        assert metrics["result_cache_size"] == 1
        assert "execution_time_seconds" in metrics

    @pytest.mark.asyncio
    async def test_clear_cache(self, orchestrator):
        """Test cache clearing functionality."""
        # Set some state
        orchestrator.iterations = 5
        orchestrator._completed_node_ids.add("node1")
        orchestrator._failed_node_ids.add("node2")
        orchestrator.result_cache["node1"] = "result1"

        # Clear cache
        orchestrator.clear_cache()

        # Verify everything is cleared
        assert orchestrator.iterations == 0
        assert len(orchestrator._completed_node_ids) == 0
        assert len(orchestrator._failed_node_ids) == 0
        assert len(orchestrator.result_cache) == 0
        assert len(orchestrator._parent_to_children) == 0
        assert len(orchestrator._child_to_parent) == 0