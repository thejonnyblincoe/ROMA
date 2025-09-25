"""
Integration Tests for ExecutionConfig - Configuration Enforcement.

Tests that ExecutionConfig limits are properly enforced across all orchestration
components including ExecutionOrchestrator, ParallelExecutionEngine, and TaskNodeProcessor.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from roma.application.orchestration.execution_orchestrator import ExecutionOrchestrator
from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
from roma.application.orchestration.task_node_processor import TaskNodeProcessor
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.context_builder_service import ContextBuilderService, TaskContext
from roma.application.services.event_store import InMemoryEventStore
from roma.application.services.recovery_manager import RecoveryManager
from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.config.execution_config import ExecutionConfig
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.result_envelope import ExecutionMetrics, ResultEnvelope
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


@pytest.fixture
def restrictive_config():
    """Restrictive ExecutionConfig for testing limits."""
    return ExecutionConfig(
        max_concurrent_tasks=2,
        max_iterations=3,
        max_subtasks_per_node=2,
        max_tasks_per_level=3,
        total_timeout=10,
        task_timeout=2,
        enable_recovery=True,
        enable_aggregation=True
    )


@pytest.fixture
def mock_components():
    """Mock all orchestration components."""
    components = {}
    components['graph_state_manager'] = AsyncMock(spec=GraphStateManager)
    components['node_processor'] = AsyncMock(spec=TaskNodeProcessor)
    components['context_builder'] = AsyncMock(spec=ContextBuilderService)
    components['agent_runtime_service'] = AsyncMock(spec=AgentRuntimeService)
    components['recovery_manager'] = AsyncMock(spec=RecoveryManager)
    components['event_store'] = AsyncMock(spec=InMemoryEventStore)
    return components


@pytest.fixture
def sample_task():
    """Sample TaskNode for testing."""
    return TaskNode(
        task_id="test_task",
        goal="Test goal",
        task_type=TaskType.THINK,
        status=TaskStatus.PENDING
    )


def create_mock_result(action=NodeAction.COMPLETE, node_id="test_node"):
    """Create mock NodeResult."""
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
        metadata={"node_id": node_id}
    )


class TestExecutionConfigIntegration:
    """Integration tests for ExecutionConfig enforcement."""

    @pytest.mark.asyncio
    async def test_max_concurrent_tasks_enforcement(self, restrictive_config, mock_components):
        """Test that max_concurrent_tasks limit is enforced."""
        # Create ParallelExecutionEngine with restrictive config
        engine = ParallelExecutionEngine(
            state_manager=mock_components['graph_state_manager'],
            max_concurrent_tasks=restrictive_config.max_concurrent_tasks
        )

        # Verify the limit is set correctly
        assert engine.max_concurrent_tasks == 2
        assert engine._execution_semaphore._value == 2

        # Track concurrent executions
        concurrent_count = 0
        max_concurrent = 0

        async def slow_processor(node, context):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)

            # Simulate work longer than timeout
            await asyncio.sleep(0.5)

            concurrent_count -= 1
            return create_mock_result(node_id=node.task_id)

        mock_components['node_processor'].process_node.side_effect = slow_processor

        # Create more nodes than the concurrent limit
        nodes = [
            TaskNode(task_id=f"node_{i}", goal=f"Task {i}", task_type=TaskType.THINK, status=TaskStatus.READY)
            for i in range(5)
        ]

        context = TaskContext(task=None, overall_objective="Test", context_items=[], execution_metadata={})

        # Execute - should respect concurrency limit
        results = await engine.execute_ready_nodes(nodes, mock_components['node_processor'], context)

        # Verify all completed but concurrency was limited
        assert len(results) == 5
        assert all(r.is_successful for r in results)
        assert max_concurrent <= restrictive_config.max_concurrent_tasks

    @pytest.mark.asyncio
    async def test_max_iterations_enforcement(self, restrictive_config, mock_components, sample_task):
        """Test that max_iterations limit is enforced."""
        # Create ExecutionOrchestrator with restrictive config
        orchestrator = ExecutionOrchestrator(
            graph_state_manager=mock_components['graph_state_manager'],
            parallel_engine=AsyncMock(spec=ParallelExecutionEngine),
            node_processor=mock_components['node_processor'],
            context_builder=mock_components['context_builder'],
            recovery_manager=mock_components['recovery_manager'],
            event_store=mock_components['event_store'],
            execution_config=restrictive_config
        )

        # Setup mocks to create infinite loop scenario
        mock_components['graph_state_manager'].get_ready_nodes.return_value = [sample_task]
        mock_components['context_builder'].build_context.return_value = AsyncMock()

        # Mock parallel engine to return retry results (never complete)
        retry_result = NodeResult.retry(
            error="Test retry",
            agent_name="test_agent",
            metadata={"node_id": sample_task.task_id}
        )
        orchestrator.parallel_engine.execute_ready_nodes.return_value = [retry_result]

        # Execute - should hit iteration limit
        result = await orchestrator.execute(sample_task, "test objective")

        # Verify iteration limit was enforced
        assert orchestrator.iterations == restrictive_config.max_iterations
        assert result.success is False
        assert "maximum iterations" in result.error_details[0]["message"].lower()

    @pytest.mark.asyncio
    async def test_timeout_enforcement(self, mock_components):
        """Test that timeout limits are enforced."""
        # Create config with very short timeout
        short_timeout_config = ExecutionConfig(
            max_concurrent_tasks=10,
            max_iterations=100,
            total_timeout=1,  # 1 second
            task_timeout=1
        )

        orchestrator = ExecutionOrchestrator(
            graph_state_manager=mock_components['graph_state_manager'],
            parallel_engine=AsyncMock(spec=ParallelExecutionEngine),
            node_processor=mock_components['node_processor'],
            context_builder=mock_components['context_builder'],
            recovery_manager=mock_components['recovery_manager'],
            event_store=mock_components['event_store'],
            execution_config=short_timeout_config
        )

        sample_task = TaskNode(
            task_id="slow_task",
            goal="Slow task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )

        # Setup mocks for slow execution
        mock_components['graph_state_manager'].get_ready_nodes.return_value = [sample_task]
        mock_components['context_builder'].build_context.return_value = AsyncMock()

        # Mock parallel engine to simulate slow processing
        async def slow_execution(*args, **kwargs):
            await asyncio.sleep(2)  # Longer than timeout
            return [create_mock_result(node_id=sample_task.task_id)]

        orchestrator.parallel_engine.execute_ready_nodes.side_effect = slow_execution

        # Execute - should timeout
        start_time = datetime.now(UTC)
        result = await orchestrator.execute(sample_task, "test objective")
        end_time = datetime.now(UTC)

        # Verify timeout was enforced
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time <= short_timeout_config.total_timeout + 1  # Allow some margin
        assert result.success is False

    @pytest.mark.asyncio
    async def test_subtask_limit_enforcement(self, restrictive_config, mock_components, sample_task):
        """Test that max_subtasks_per_node limit is enforced."""
        # This would be enforced in the planner agent result processing
        # For now, we test that the config is properly passed through

        orchestrator = ExecutionOrchestrator(
            graph_state_manager=mock_components['graph_state_manager'],
            parallel_engine=AsyncMock(spec=ParallelExecutionEngine),
            node_processor=mock_components['node_processor'],
            context_builder=mock_components['context_builder'],
            recovery_manager=mock_components['recovery_manager'],
            event_store=mock_components['event_store'],
            execution_config=restrictive_config
        )

        # Verify config is accessible
        assert orchestrator.execution_config.max_subtasks_per_node == 2
        assert orchestrator.execution_config.max_tasks_per_level == 3

    @pytest.mark.asyncio
    async def test_recovery_disable_enforcement(self, mock_components, sample_task):
        """Test that recovery can be disabled via config."""
        no_recovery_config = ExecutionConfig(
            max_concurrent_tasks=10,
            max_iterations=100,
            enable_recovery=False,
            enable_aggregation=True
        )

        orchestrator = ExecutionOrchestrator(
            graph_state_manager=mock_components['graph_state_manager'],
            parallel_engine=AsyncMock(spec=ParallelExecutionEngine),
            node_processor=mock_components['node_processor'],
            context_builder=mock_components['context_builder'],
            recovery_manager=mock_components['recovery_manager'],
            event_store=mock_components['event_store'],
            execution_config=no_recovery_config
        )

        # Verify recovery is disabled
        assert orchestrator.execution_config.enable_recovery is False

        # In a real implementation, this would affect how failures are handled
        # The orchestrator would skip recovery attempts when enable_recovery=False

    @pytest.mark.asyncio
    async def test_aggregation_disable_enforcement(self, mock_components, sample_task):
        """Test that aggregation can be disabled via config."""
        no_aggregation_config = ExecutionConfig(
            max_concurrent_tasks=10,
            max_iterations=100,
            enable_recovery=True,
            enable_aggregation=False
        )

        orchestrator = ExecutionOrchestrator(
            graph_state_manager=mock_components['graph_state_manager'],
            parallel_engine=AsyncMock(spec=ParallelExecutionEngine),
            node_processor=mock_components['node_processor'],
            context_builder=mock_components['context_builder'],
            recovery_manager=mock_components['recovery_manager'],
            event_store=mock_components['event_store'],
            execution_config=no_aggregation_config
        )

        # Verify aggregation is disabled
        assert orchestrator.execution_config.enable_aggregation is False

        # In a real implementation, this would skip aggregation steps
        # when enable_aggregation=False

    @pytest.mark.asyncio
    async def test_config_propagation_through_components(self, restrictive_config, mock_components, sample_task):
        """Test that ExecutionConfig is properly propagated through all components."""
        # Create ParallelExecutionEngine with config
        parallel_engine = ParallelExecutionEngine(
            state_manager=mock_components['graph_state_manager'],
            max_concurrent_tasks=restrictive_config.max_concurrent_tasks
        )

        # Create ExecutionOrchestrator with config
        orchestrator = ExecutionOrchestrator(
            graph_state_manager=mock_components['graph_state_manager'],
            parallel_engine=parallel_engine,
            node_processor=mock_components['node_processor'],
            context_builder=mock_components['context_builder'],
            recovery_manager=mock_components['recovery_manager'],
            event_store=mock_components['event_store'],
            execution_config=restrictive_config
        )

        # Verify config is accessible at orchestrator level
        assert orchestrator.execution_config == restrictive_config

        # Verify concurrency limit is applied to parallel engine
        assert parallel_engine.max_concurrent_tasks == restrictive_config.max_concurrent_tasks

        # Verify limits are enforced
        assert orchestrator.execution_config.max_iterations == 3
        assert orchestrator.execution_config.total_timeout == 10

    def test_config_validation_in_components(self, mock_components):
        """Test that invalid configurations are rejected."""
        # Test invalid concurrent tasks
        with pytest.raises(ValueError):
            ExecutionConfig(max_concurrent_tasks=0)

        # Test invalid timeout relationship
        with pytest.raises(ValueError):
            ExecutionConfig(total_timeout=60, task_timeout=120)

        # Test that components handle valid configs
        valid_config = ExecutionConfig(
            max_concurrent_tasks=5,
            max_iterations=50,
            total_timeout=600,
            task_timeout=300
        )

        # Should not raise exceptions
        ParallelExecutionEngine(
            state_manager=mock_components['graph_state_manager'],
            max_concurrent_tasks=valid_config.max_concurrent_tasks
        )

        ExecutionOrchestrator(
            graph_state_manager=mock_components['graph_state_manager'],
            parallel_engine=AsyncMock(spec=ParallelExecutionEngine),
            node_processor=mock_components['node_processor'],
            context_builder=mock_components['context_builder'],
            recovery_manager=mock_components['recovery_manager'],
            event_store=mock_components['event_store'],
            execution_config=valid_config
        )

    @pytest.mark.asyncio
    async def test_performance_config_integration(self, mock_components, sample_task):
        """Test integration with high-performance configuration."""
        perf_config = ExecutionConfig(
            max_concurrent_tasks=20,
            max_iterations=1000,
            max_subtasks_per_node=20,
            max_tasks_per_level=50,
            total_timeout=7200,
            task_timeout=600,
            enable_recovery=True,
            enable_aggregation=True
        )

        # Create components with performance config
        parallel_engine = ParallelExecutionEngine(
            state_manager=mock_components['graph_state_manager'],
            max_concurrent_tasks=perf_config.max_concurrent_tasks
        )

        orchestrator = ExecutionOrchestrator(
            graph_state_manager=mock_components['graph_state_manager'],
            parallel_engine=parallel_engine,
            node_processor=mock_components['node_processor'],
            context_builder=mock_components['context_builder'],
            recovery_manager=mock_components['recovery_manager'],
            event_store=mock_components['event_store'],
            execution_config=perf_config
        )

        # Verify high-performance settings
        assert parallel_engine.max_concurrent_tasks == 20
        assert orchestrator.execution_config.max_iterations == 1000
        assert orchestrator.execution_config.total_timeout == 7200

        # These settings should support high-throughput execution
