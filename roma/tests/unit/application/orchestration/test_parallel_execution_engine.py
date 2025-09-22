"""
Tests for ParallelExecutionEngine - Concurrent Execution Logic.

Tests the pure concurrency engine, semaphore control, error handling,
and performance metrics collection.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from roma.application.orchestration.parallel_execution_engine import (
    ParallelExecutionEngine,
    ParallelExecutionStats
)
from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.orchestration.task_node_processor import TaskNodeProcessor
from roma.application.services.context_builder_service import TaskContext
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.result_envelope import ResultEnvelope
from roma.domain.value_objects.result_envelope import ExecutionMetrics
from roma.domain.value_objects.agent_type import AgentType


@pytest.fixture
def mock_state_manager():
    """Mock GraphStateManager."""
    return AsyncMock(spec=GraphStateManager)


@pytest.fixture
def mock_task_processor():
    """Mock TaskNodeProcessor."""
    return AsyncMock(spec=TaskNodeProcessor)


@pytest.fixture
def sample_context():
    """Sample TaskContext for testing."""
    return TaskContext(
        task=None,  # Will be set per node
        overall_objective="Test objective",
        context_items=[],
        execution_metadata={}
    )


@pytest.fixture
def parallel_engine(mock_state_manager):
    """ParallelExecutionEngine instance with mocked dependencies."""
    return ParallelExecutionEngine(
        state_manager=mock_state_manager,
        max_concurrent_tasks=3  # Small for testing
    )


@pytest.fixture
def sample_nodes():
    """Sample TaskNodes for testing."""
    return [
        TaskNode(
            task_id="node_1",
            goal="Task 1",
            task_type=TaskType.THINK,
            status=TaskStatus.READY
        ),
        TaskNode(
            task_id="node_2",
            goal="Task 2",
            task_type=TaskType.RETRIEVE,
            status=TaskStatus.READY
        ),
        TaskNode(
            task_id="node_3",
            goal="Task 3",
            task_type=TaskType.WRITE,
            status=TaskStatus.READY
        )
    ]


def create_success_result(node_id: str, action: NodeAction = NodeAction.COMPLETE) -> NodeResult:
    """Helper to create successful NodeResult."""
    envelope = ResultEnvelope.create_success(
        result={"output": f"Result for {node_id}"},
        task_id=node_id,
        execution_id=f"exec_{node_id}",
        agent_type=AgentType.EXECUTOR,
        execution_metrics=ExecutionMetrics(execution_time=1.0, tokens_used=50),
        artifacts=[],
        output_text="Test output"
    )

    return NodeResult(
        action=action,
        envelope=envelope,
        agent_name="test_agent",
        agent_type="executor",
        metadata={"node_id": node_id}
    )


class TestParallelExecutionEngine:
    """Test cases for ParallelExecutionEngine."""

    def test_initialization(self, mock_state_manager):
        """Test engine initialization."""
        engine = ParallelExecutionEngine(
            state_manager=mock_state_manager,
            max_concurrent_tasks=5
        )

        assert engine.state_manager == mock_state_manager
        assert engine.max_concurrent_tasks == 5
        assert engine._execution_semaphore._value == 5
        assert engine._total_batches_processed == 0
        assert engine._total_nodes_processed == 0
        assert engine._total_execution_time == 0.0

    @pytest.mark.asyncio
    async def test_execute_ready_nodes_empty_list(self, parallel_engine, mock_task_processor, sample_context):
        """Test executing empty node list returns empty results."""
        results = await parallel_engine.execute_ready_nodes([], mock_task_processor, sample_context, "test_execution")

        assert results == []
        assert mock_task_processor.process_node.call_count == 0

    @pytest.mark.asyncio
    async def test_execute_ready_nodes_successful(self, parallel_engine, mock_state_manager,
                                                   mock_task_processor, sample_context, sample_nodes):
        """Test successful parallel execution of multiple nodes."""
        # Setup successful node processing
        mock_task_processor.process_node.side_effect = [
            create_success_result("node_1"),
            create_success_result("node_2"),
            create_success_result("node_3")
        ]

        # Execute
        results = await parallel_engine.execute_ready_nodes(
            sample_nodes, mock_task_processor, sample_context, "test_execution"
        )

        # Verify results
        assert len(results) == 3

        for i, result in enumerate(results):
            assert isinstance(result, NodeResult)
            assert result.action == NodeAction.COMPLETE
            assert result.is_successful is True
            assert result.metadata["node_id"] == f"node_{i+1}"

        # Verify all nodes were processed
        assert mock_task_processor.process_node.call_count == 3

        # Verify state transitions were called for each node
        expected_transitions = []
        for node in sample_nodes:
            expected_transitions.extend([
                (node.task_id, TaskStatus.READY),
                (node.task_id, TaskStatus.EXECUTING)
            ])

        actual_transitions = [(call[0][0], call[0][1]) for call in mock_state_manager.transition_node_status.call_args_list]
        assert len(actual_transitions) == 6  # 2 transitions per node

        # Verify statistics updated
        assert parallel_engine._total_batches_processed == 1
        assert parallel_engine._total_nodes_processed == 3
        assert parallel_engine._total_execution_time > 0

    @pytest.mark.asyncio
    async def test_execute_with_processing_failures(self, parallel_engine, mock_state_manager,
                                                     mock_task_processor, sample_context, sample_nodes):
        """Test handling of node processing failures."""
        # Setup mixed success/failure
        mock_task_processor.process_node.side_effect = [
            create_success_result("node_1"),
            NodeResult.failure(
                error="Processing failed",
                agent_name="test_agent",
                metadata={"node_id": "node_2"}
            ),
            create_success_result("node_3")
        ]

        # Execute
        results = await parallel_engine.execute_ready_nodes(
            sample_nodes, mock_task_processor, sample_context, "test_execution"
        )

        # Verify results
        assert len(results) == 3

        # First node successful
        assert results[0].is_successful is True
        assert results[0].metadata["node_id"] == "node_1"

        # Second node failed
        assert results[1].is_successful is False
        assert results[1].error == "Processing failed"
        assert results[1].metadata["node_id"] == "node_2"

        # Third node successful
        assert results[2].is_successful is True
        assert results[2].metadata["node_id"] == "node_3"

    @pytest.mark.asyncio
    async def test_execute_with_exceptions(self, parallel_engine, mock_state_manager,
                                           mock_task_processor, sample_context, sample_nodes):
        """Test handling of exceptions during node processing."""
        # Setup exception in processing
        mock_task_processor.process_node.side_effect = [
            create_success_result("node_1"),
            Exception("Critical error"),
            create_success_result("node_3")
        ]

        # Execute
        results = await parallel_engine.execute_ready_nodes(
            sample_nodes, mock_task_processor, sample_context, "test_execution"
        )

        # Verify results
        assert len(results) == 3

        # First node successful
        assert results[0].is_successful is True

        # Second node created failure result from exception
        assert results[1].is_successful is False
        assert "Critical error" in results[1].error
        assert results[1].agent_name == "parallel_execution_engine"
        assert results[1].metadata["exception_type"] == "Exception"

        # Third node successful
        assert results[2].is_successful is True

        # Verify failed node was transitioned to FAILED state
        failed_transitions = [call for call in mock_state_manager.transition_node_status.call_args_list
                             if call[0][1] == TaskStatus.FAILED]
        assert len(failed_transitions) == 1
        assert failed_transitions[0][0][0] == "node_2"

    @pytest.mark.asyncio
    async def test_semaphore_concurrency_control(self, mock_state_manager):
        """Test that semaphore correctly limits concurrent execution."""
        # Create engine with max 2 concurrent tasks
        engine = ParallelExecutionEngine(state_manager=mock_state_manager, max_concurrent_tasks=2)

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def slow_processor(node, context):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)

            # Simulate some work
            await asyncio.sleep(0.1)

            concurrent_count -= 1
            return create_success_result(node.task_id)

        mock_processor = AsyncMock()
        mock_processor.process_node.side_effect = slow_processor

        # Create 5 nodes to test semaphore limiting to 2 concurrent
        nodes = [
            TaskNode(task_id=f"node_{i}", goal=f"Task {i}", task_type=TaskType.THINK, status=TaskStatus.READY)
            for i in range(5)
        ]

        context = TaskContext(task=None, overall_objective="Test", context_items=[], execution_metadata={})

        # Execute
        results = await engine.execute_ready_nodes(nodes, mock_processor, context, "test_execution")

        # Verify all completed
        assert len(results) == 5
        assert all(r.is_successful for r in results)

        # Verify concurrency was limited to 2
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_metadata_enrichment(self, parallel_engine, mock_state_manager,
                                       mock_task_processor, sample_context, sample_nodes):
        """Test that node_id metadata is properly enriched."""
        # Create result without node_id in metadata
        result_without_metadata = NodeResult(
            action=NodeAction.COMPLETE,
            envelope=None,  # Minimal result
            agent_name="test_agent",
            metadata={}  # No node_id
        )

        mock_task_processor.process_node.return_value = result_without_metadata

        # Execute single node
        results = await parallel_engine.execute_ready_nodes(
            [sample_nodes[0]], mock_task_processor, sample_context
        )

        # Verify node_id was added to metadata
        assert len(results) == 1
        assert "node_id" in results[0].metadata
        assert results[0].metadata["node_id"] == "node_1"

    @pytest.mark.asyncio
    async def test_state_transition_failure_handling(self, parallel_engine, mock_state_manager,
                                                      mock_task_processor, sample_context, sample_nodes):
        """Test handling when state transitions fail."""
        # Setup state transition to fail
        mock_state_manager.transition_node_status.side_effect = Exception("State transition failed")

        # Setup successful processing
        mock_task_processor.process_node.return_value = create_success_result("node_1")

        # Execute - should handle state transition failure gracefully
        results = await parallel_engine.execute_ready_nodes(
            [sample_nodes[0]], mock_task_processor, sample_context
        )

        # Should create failure result due to state transition exception
        assert len(results) == 1
        assert results[0].is_successful is False
        assert "State transition failed" in results[0].error

    def test_get_execution_stats(self, parallel_engine):
        """Test execution statistics retrieval."""
        # Set some internal stats
        parallel_engine._total_nodes_processed = 100
        parallel_engine._total_execution_time = 50.5

        stats = parallel_engine.get_execution_stats()

        assert isinstance(stats, ParallelExecutionStats)
        assert stats.nodes_processed == 100
        assert stats.execution_time_seconds == 50.5

    def test_get_performance_metrics(self, parallel_engine):
        """Test performance metrics calculation."""
        # Set some stats
        parallel_engine._total_batches_processed = 10
        parallel_engine._total_nodes_processed = 50
        parallel_engine._total_execution_time = 25.0

        metrics = parallel_engine.get_performance_metrics()

        assert metrics["max_concurrent_tasks"] == 3
        assert metrics["total_batches_processed"] == 10
        assert metrics["total_nodes_processed"] == 50
        assert metrics["total_execution_time_seconds"] == 25.0
        assert metrics["average_nodes_per_batch"] == 5.0
        assert metrics["average_batch_time_seconds"] == 2.5

    def test_reset_stats(self, parallel_engine):
        """Test statistics reset functionality."""
        # Set some stats
        parallel_engine._total_batches_processed = 5
        parallel_engine._total_nodes_processed = 25
        parallel_engine._total_execution_time = 10.5

        # Reset
        parallel_engine.reset_stats()

        # Verify reset
        assert parallel_engine._total_batches_processed == 0
        assert parallel_engine._total_nodes_processed == 0
        assert parallel_engine._total_execution_time == 0.0

    @pytest.mark.asyncio
    async def test_concurrent_execution_ordering(self, mock_state_manager):
        """Test that concurrent execution doesn't affect result ordering."""
        engine = ParallelExecutionEngine(state_manager=mock_state_manager, max_concurrent_tasks=10)

        # Create processor that returns node_id in result
        async def ordered_processor(node, context):
            # Add small delay to test ordering
            await asyncio.sleep(0.01)
            return create_success_result(node.task_id)

        mock_processor = AsyncMock()
        mock_processor.process_node.side_effect = ordered_processor

        # Create ordered nodes
        nodes = [
            TaskNode(task_id=f"node_{i:03d}", goal=f"Task {i}", task_type=TaskType.THINK, status=TaskStatus.READY)
            for i in range(10)
        ]

        context = TaskContext(task=None, overall_objective="Test", context_items=[], execution_metadata={})

        # Execute
        results = await engine.execute_ready_nodes(nodes, mock_processor, context, "test_execution")

        # Verify results are in same order as input nodes
        assert len(results) == 10
        for i, result in enumerate(results):
            expected_node_id = f"node_{i:03d}"
            assert result.metadata["node_id"] == expected_node_id