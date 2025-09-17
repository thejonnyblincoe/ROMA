"""
Unit tests for ParallelExecutionEngine.

Following TDD principles - these tests define the interface and expected behavior
for the parallel execution engine with modified Kahn's algorithm.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock

from src.roma.domain.entities.task_node import TaskNode
from src.roma.domain.value_objects.task_type import TaskType
from src.roma.domain.value_objects.task_status import TaskStatus
from src.roma.domain.value_objects.node_type import NodeType
from src.roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from src.roma.application.orchestration.graph_state_manager import GraphStateManager
from src.roma.application.services.event_store import InMemoryEventStore

from .test_utils import (
    TestGraphFactory,
    MockExecutorFactory,
    TestStateManagerFactory,
    SampleNodeFactory
)


class TestParallelExecutionEngine:
    """Test ParallelExecutionEngine core functionality."""

    @pytest.fixture
    def sample_task_nodes(self):
        """Create sample task nodes for testing."""
        return SampleNodeFactory.create_sample_nodes(4)

    @pytest.fixture
    def mock_task_executor(self):
        """Create mock task executor."""
        return MockExecutorFactory.create_successful_executor()


    def test_parallel_execution_engine_init(self, mock_task_executor):
        """Test ParallelExecutionEngine initialization."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph()
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=10
        )
        
        assert engine.state_manager == state_manager
        assert engine.task_executor == mock_task_executor
        assert engine.max_concurrent_tasks == 10
        assert engine.is_running is False

    @pytest.mark.asyncio
    async def test_execute_graph_empty(self, mock_task_executor):
        """Test executing empty graph."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph()
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        result = await engine.execute_graph()
        
        assert result.total_nodes == 0
        assert result.completed_nodes == 0
        assert result.failed_nodes == 0
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_graph_single_node(self, sample_task_nodes, mock_task_executor):
        """Test executing graph with single node."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Mock successful task execution
        mock_task_executor.execute_task.return_value = {"status": "completed", "result": "success"}
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        result = await engine.execute_graph()
        
        assert result.total_nodes == 1
        assert result.completed_nodes == 1
        assert result.failed_nodes == 0
        assert result.success is True
        
        # Verify task executor was called
        mock_task_executor.execute_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_graph_parallel_execution(self, sample_task_nodes, mock_task_executor):
        """Test executing graph with parallel nodes."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        graph = await TestGraphFactory.create_parallel_graph(sample_task_nodes)
        state_manager = TestStateManagerFactory.create_state_manager(graph)
        
        # Use delayed mock executor to test parallelism  
        mock_task_executor = MockExecutorFactory.create_delay_executor(0.1)
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        import time
        start_time = time.time()
        result = await engine.execute_graph()
        end_time = time.time()
        
        # Should complete all 4 tasks
        assert result.total_nodes == 4
        assert result.completed_nodes == 4
        assert result.failed_nodes == 0
        assert result.success is True
        
        # Should take advantage of parallelism (less than 4 * 0.1 seconds)
        execution_time = end_time - start_time
        assert execution_time < 0.35  # Should be around 0.2s (2 batches)
        
        # Should have called execute_task 4 times
        assert mock_task_executor.execute_task.call_count == 4

    @pytest.mark.asyncio
    async def test_execute_graph_linear_execution(self, sample_task_nodes, mock_task_executor):
        """Test executing linear graph (sequential execution)."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        graph = await TestGraphFactory.create_linear_graph(sample_task_nodes)
        state_manager = TestStateManagerFactory.create_state_manager(graph)
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        result = await engine.execute_graph()
        
        # Should complete all 4 tasks sequentially
        assert result.total_nodes == 4
        assert result.completed_nodes == 4
        assert result.failed_nodes == 0
        assert result.success is True
        
        # Should have called execute_task 4 times
        assert mock_task_executor.execute_task.call_count == 4

    @pytest.mark.asyncio
    async def test_execute_graph_with_failures(self, sample_task_nodes, mock_task_executor):
        """Test executing graph with task failures."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        graph = await TestGraphFactory.create_parallel_graph(sample_task_nodes)
        state_manager = TestStateManagerFactory.create_state_manager(graph)
        
        # Use selective failure executor - first task succeeds, others fail
        mock_task_executor = MockExecutorFactory.create_selective_failure_executor([1, 2, 3])
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        result = await engine.execute_graph()
        
        # Should have 1 completed, 3 failed
        assert result.total_nodes == 4
        assert result.completed_nodes == 1
        assert result.failed_nodes == 3
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_graph_concurrency_limit(self, sample_task_nodes, mock_task_executor):
        """Test that concurrency limit is respected."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        graph = await TestGraphFactory.create_parallel_graph(sample_task_nodes)
        state_manager = TestStateManagerFactory.create_state_manager(graph)
        
        # Track concurrent executions with custom mock
        concurrent_count = 0
        max_concurrent_seen = 0
        
        async def mock_execute_with_tracking(node):
            nonlocal concurrent_count, max_concurrent_seen
            concurrent_count += 1
            max_concurrent_seen = max(max_concurrent_seen, concurrent_count)
            
            await asyncio.sleep(0.1)  # Simulate work
            
            concurrent_count -= 1
            return {"status": "completed", "result": "success"}
        
        mock_task_executor.execute_task.side_effect = mock_execute_with_tracking
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=2  # Limit to 2 concurrent tasks
        )
        
        result = await engine.execute_graph()
        
        # Should complete successfully
        assert result.success is True
        assert result.completed_nodes == 4
        
        # Should not exceed concurrency limit
        assert max_concurrent_seen <= 2

    @pytest.mark.asyncio
    async def test_execute_graph_dynamic_node_addition(self, sample_task_nodes, mock_task_executor):
        """Test executing graph with nodes added during execution."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Mock executor that adds new node during first task execution
        execution_count = 0
        
        async def mock_execute_with_node_addition(node):
            nonlocal execution_count
            execution_count += 1
            
            # During first execution, add a new node
            if execution_count == 1:
                new_node = TaskNode(
                    goal="Dynamically added task",
                    task_type=TaskType.WRITE,
                    node_type=NodeType.EXECUTE,
                    status=TaskStatus.PENDING,
                    parent_id=node.task_id
                )
                await state_manager.add_node(new_node)
            
            return {"status": "completed", "result": "success"}
        
        mock_task_executor.execute_task.side_effect = mock_execute_with_node_addition
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        result = await engine.execute_graph()
        
        # Should complete both original and dynamically added node
        assert result.total_nodes == 2
        assert result.completed_nodes == 2
        assert result.success is True

    def test_get_ready_nodes_batch(self, sample_task_nodes):
        """Test getting batch of ready nodes for execution."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        mock_task_executor = AsyncMock()
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        ready_batch = engine._get_ready_nodes_batch()
        
        # Should return the root node
        assert len(ready_batch) == 1
        assert ready_batch[0].task_id == sample_task_nodes[0].task_id

    def test_is_execution_complete_empty_graph(self):
        """Test completion check on empty graph."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph()
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        mock_task_executor = AsyncMock()
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        assert engine._is_execution_complete() is True

    @pytest.mark.asyncio
    async def test_is_execution_complete_with_pending_nodes(self, sample_task_nodes):
        """Test completion check with pending nodes."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph(root_node=sample_task_nodes[0])
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        mock_task_executor = AsyncMock()
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_task_executor,
            max_concurrent_tasks=5
        )
        
        assert engine._is_execution_complete() is False
        
        # Complete the node
        await state_manager.transition_node_status(sample_task_nodes[0].task_id, TaskStatus.READY)
        await state_manager.transition_node_status(sample_task_nodes[0].task_id, TaskStatus.EXECUTING)
        await state_manager.transition_node_status(sample_task_nodes[0].task_id, TaskStatus.COMPLETED)
        
        assert engine._is_execution_complete() is True


class TestParallelExecutionEnginePerformance:
    """Test ParallelExecutionEngine performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_parallel_graph_performance(self):
        """Test performance with large parallel graph."""
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        # Create graph with 1 root + 100 parallel children
        event_store = InMemoryEventStore()
        root_node = TaskNode(
            goal="Root task",
            task_type=TaskType.THINK,
            node_type=NodeType.PLAN,
            status=TaskStatus.PENDING
        )
        
        graph = DynamicTaskGraph(root_node=root_node)
        
        # Add 100 parallel child nodes
        child_nodes = []
        for i in range(100):
            child_node = TaskNode(
                goal=f"Parallel task {i}",
                task_type=TaskType.WRITE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id=root_node.task_id
            )
            child_nodes.append(child_node)
            await graph.add_node(child_node)
        
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Mock fast execution
        mock_executor = AsyncMock()
        mock_executor.execute_task.return_value = {"status": "completed", "result": "success"}
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_executor,
            max_concurrent_tasks=50  # High concurrency
        )
        
        import time
        start_time = time.time()
        
        result = await engine.execute_graph()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete all 101 tasks quickly (< 5 seconds)
        assert execution_time < 5.0
        assert result.total_nodes == 101
        assert result.completed_nodes == 101
        assert result.success is True
        
        # Should have called execute_task 101 times
        assert mock_executor.execute_task.call_count == 101