"""
Bug-specific test scenarios to expose critical issues found in ultrathink analysis.

These tests are designed to FAIL first (RED phase) to expose bugs, 
then we'll fix the implementation (GREEN phase).
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
from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
from src.roma.application.services.event_store import InMemoryEventStore

from .test_utils import (
    TestGraphFactory,
    MockExecutorFactory,
    TestStateManagerFactory,
    SampleNodeFactory
)


class TestCriticalBugScenarios:
    """Test critical bugs found in ultrathink analysis."""

    @pytest.fixture
    def sample_node(self) -> TaskNode:
        """Create single sample node for testing."""
        return SampleNodeFactory.create_single_node(
            goal="Test task for bug scenarios",
            task_type=TaskType.THINK,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )

    @pytest.mark.asyncio
    async def test_silent_exception_swallowing_bug(self, sample_node):
        """
        BUG TEST: Silent exception swallowing in ParallelExecutionEngine.
        
        Current implementation silently swallows exceptions in state transition
        failures, which could hide critical errors.
        
        This test should FAIL initially to expose the bug.
        """
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        # Create test environment
        graph = DynamicTaskGraph(root_node=sample_node)
        state_manager = TestStateManagerFactory.create_state_manager(graph)
        
        # Mock the transition_node_status to fail intermittently
        original_transition = state_manager.transition_node_status
        
        async def failing_transition(task_id: str, status: TaskStatus):
            if status == TaskStatus.FAILED:
                # Simulate state manager failure when trying to mark as failed
                raise Exception("State manager database connection lost")
            return await original_transition(task_id, status)
        
        state_manager.transition_node_status = failing_transition
        
        # Mock task executor that always fails
        mock_executor = AsyncMock()
        mock_executor.execute_task.side_effect = Exception("Task execution failed")
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_executor,
            max_concurrent_tasks=1
        )
        
        # Execute the graph - this should expose the silent exception bug
        result = await engine.execute_graph()
        
        # BUG: Currently, the implementation silently swallows the state transition
        # failure, leading to inconsistent statistics and hidden errors
        
        # This assertion should FAIL initially because the bug hides the real issue
        assert result.failed_nodes == 1, "Should properly track failed nodes even when state transition fails"
        
        # This assertion should FAIL because statistics become inconsistent
        stats = engine.get_execution_statistics()
        assert stats["total_nodes_processed"] == 1, "Should maintain consistent statistics"
        
        # We should be able to access error details
        assert result.error_details is not None, "Should capture error details when state transitions fail"

    @pytest.mark.asyncio
    async def test_statistics_inconsistency_bug(self):
        """
        BUG TEST: Statistics inconsistency in ParallelExecutionEngine.
        
        When state transitions fail in exception handlers, node counts 
        become inconsistent with actual state.
        
        This test should FAIL initially to expose the bug.
        """
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        # Create multiple nodes
        nodes = []
        for i in range(3):
            node = TaskNode(
                goal=f"Task {i}",
                task_type=TaskType.WRITE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            )
            nodes.append(node)
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph(root_node=nodes[0])
        for node in nodes[1:]:
            await graph.add_node(node)
        
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Mock executor that fails on 2nd task
        call_count = 0
        async def selective_failure(node):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Simulate a failure that might cause state transition issues
                raise RuntimeError("Critical system failure during task 2")
            return {"status": "completed", "result": "success"}
        
        mock_executor = AsyncMock()
        mock_executor.execute_task.side_effect = selective_failure
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_executor,
            max_concurrent_tasks=1
        )
        
        result = await engine.execute_graph()
        
        # Get final statistics
        final_stats = engine.get_execution_statistics()
        
        # BUG: These assertions should FAIL due to inconsistent statistics
        total_accounted = final_stats["completed_nodes"] + final_stats["failed_nodes"]
        
        assert result.total_nodes == result.completed_nodes + result.failed_nodes, \
            "Total nodes should equal completed + failed"
        
        assert final_stats["total_nodes_processed"] == total_accounted, \
            "Processed count should match completed + failed"
        
        # Verify actual graph state matches statistics
        all_nodes = state_manager.get_all_nodes()
        actual_completed = sum(1 for n in all_nodes if n.status == TaskStatus.COMPLETED)
        actual_failed = sum(1 for n in all_nodes if n.status == TaskStatus.FAILED)
        
        assert result.completed_nodes == actual_completed, \
            "Reported completed should match actual completed"
        assert result.failed_nodes == actual_failed, \
            "Reported failed should match actual failed"

    @pytest.mark.asyncio 
    async def test_race_condition_version_inconsistency_bug(self, sample_node):
        """
        BUG TEST: Race condition between state update and version increment.
        
        If event storage fails after graph state update but before version increment,
        we have inconsistent state.
        
        This test should FAIL initially to expose the race condition bug.
        """
        from src.roma.application.orchestration.graph_state_manager import GraphStateManager
        
        # Create event store that fails intermittently
        failing_event_store = AsyncMock()
        failing_event_store.append = AsyncMock(side_effect=Exception("Event store connection lost"))
        
        graph = DynamicTaskGraph(root_node=sample_node)
        state_manager = GraphStateManager(graph=graph, event_store=failing_event_store)
        
        # Record initial version
        initial_version = state_manager.version
        
        # Try to transition node status - this should fail during event storage
        with pytest.raises(Exception, match="Event store connection lost"):
            await state_manager.transition_node_status(sample_node.task_id, TaskStatus.READY)
        
        # BUG: These assertions should FAIL due to race condition
        # Version should not increment if the operation failed
        assert state_manager.version == initial_version, \
            "Version should not increment if event storage fails"
        
        # Graph state should be consistent with version
        node_in_graph = state_manager.get_node_by_id(sample_node.task_id)
        if state_manager.version == initial_version:
            assert node_in_graph.status == TaskStatus.PENDING, \
                "Node state should not change if version didn't increment"

    @pytest.mark.asyncio
    async def test_concurrent_state_modification_race_condition(self):
        """
        BUG TEST: Race condition in concurrent state modifications.
        
        Multiple concurrent state changes might interfere with each other,
        causing state inconsistency.
        
        This test should FAIL initially to expose concurrency bugs.
        """
        from src.roma.application.orchestration.graph_state_manager import GraphStateManager
        
        event_store = InMemoryEventStore()
        
        # Create node
        node = TaskNode(
            goal="Concurrent test task",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )
        
        graph = DynamicTaskGraph(root_node=node)
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Create multiple concurrent transition tasks
        async def transition_to_ready():
            return await state_manager.transition_node_status(node.task_id, TaskStatus.READY)
        
        async def transition_to_failed():
            return await state_manager.transition_node_status(node.task_id, TaskStatus.FAILED)
        
        # Run concurrent transitions
        results = await asyncio.gather(
            transition_to_ready(),
            transition_to_failed(),
            return_exceptions=True
        )
        
        # Analyze results - exactly one should succeed, one should fail
        successes = [r for r in results if isinstance(r, TaskNode)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        # BUG: These assertions might FAIL due to race conditions
        assert len(successes) == 1, \
            "Exactly one concurrent transition should succeed"
        assert len(failures) == 1, \
            "Exactly one concurrent transition should fail"
        
        # Final state should be consistent
        final_node = state_manager.get_node_by_id(node.task_id)
        success_result = successes[0]
        
        assert final_node.status == success_result.status, \
            "Final node state should match successful transition"
        assert final_node.version == success_result.version, \
            "Final node version should match successful transition"

    @pytest.mark.asyncio
    async def test_deadlock_detection_accuracy_bug(self):
        """
        BUG TEST: Deadlock detection might miss complex deadlock scenarios.
        
        Current implementation might not detect all types of deadlocks,
        especially with dynamic node addition.
        
        This test should FAIL initially to expose deadlock detection bugs.
        """
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        # Create a scenario that could lead to undetected deadlock
        event_store = InMemoryEventStore()
        
        # Create nodes with circular dependency potential
        node_a = TaskNode(
            goal="Task A",
            task_type=TaskType.THINK,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )
        
        graph = DynamicTaskGraph(root_node=node_a)
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Mock executor that creates dependent nodes during execution
        async def dynamic_node_creator(node):
            if node.goal == "Task A":
                # During execution of A, create B that depends on A
                node_b = TaskNode(
                    goal="Task B", 
                    task_type=TaskType.WRITE,
                    node_type=NodeType.EXECUTE,
                    status=TaskStatus.PENDING,
                    parent_id=node.task_id  # B depends on A
                )
                await state_manager.add_node(node_b)
                
                # Simulate partial completion that might create deadlock conditions
                raise Exception("Task A failed after creating dependency")
            
            return {"status": "completed", "result": "success"}
        
        mock_executor = AsyncMock()
        mock_executor.execute_task.side_effect = dynamic_node_creator
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_executor,
            max_concurrent_tasks=2
        )
        
        # Execute and check for proper deadlock detection
        result = await engine.execute_graph()
        
        # BUG: These assertions might FAIL if deadlock detection is inadequate
        if not result.success:
            # If execution failed, deadlock detection should be accurate
            is_deadlocked = engine.is_deadlocked()
            
            # Should properly detect the deadlock condition
            assert is_deadlocked or result.failed_nodes > 0, \
                "Should detect deadlock or properly handle failures"
            
            # Should have consistent statistics even in deadlock scenarios
            all_nodes = state_manager.get_all_nodes()
            total_terminal = sum(1 for n in all_nodes 
                               if n.status in {TaskStatus.COMPLETED, TaskStatus.FAILED})
            
            assert result.completed_nodes + result.failed_nodes == total_terminal, \
                "Statistics should be consistent even in deadlock scenarios"


class TestEdgeCaseCoverage:
    """Test edge cases not covered in original implementation."""

    @pytest.mark.asyncio
    async def test_empty_graph_with_concurrent_node_addition(self):
        """
        MISSING TEST: Adding nodes to empty graph during execution.
        
        This edge case is not covered in original tests.
        """
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph()  # Start with empty graph
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Mock executor that adds nodes during "execution" of empty graph
        nodes_created = []
        
        async def node_creating_executor(node):
            # This shouldn't be called since graph is empty
            nodes_created.append(node)
            return {"status": "completed", "result": "success"}
        
        mock_executor = AsyncMock()
        mock_executor.execute_task.side_effect = node_creating_executor
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_executor,
            max_concurrent_tasks=1
        )
        
        # Start execution in background
        execution_task = asyncio.create_task(engine.execute_graph())
        
        # Add node during execution
        new_node = TaskNode(
            goal="Dynamically added to empty graph",
            task_type=TaskType.RETRIEVE,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )
        
        await state_manager.add_node(new_node)
        
        # Wait for execution to complete
        result = await execution_task
        
        # Should handle dynamic addition to empty graph
        assert result.total_nodes == 1
        assert result.completed_nodes == 1
        assert len(nodes_created) == 1

    @pytest.mark.asyncio
    async def test_graph_with_only_failed_nodes(self):
        """
        MISSING TEST: Graph where all nodes fail.
        
        This edge case tests error propagation and final state consistency.
        """
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        # Create graph with multiple independent nodes
        nodes = []
        for i in range(3):
            node = TaskNode(
                goal=f"Failing task {i}",
                task_type=TaskType.WRITE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING
            )
            nodes.append(node)
        
        event_store = InMemoryEventStore()
        graph = DynamicTaskGraph(root_node=nodes[0])
        for node in nodes[1:]:
            await graph.add_node(node)
        
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Mock executor that always fails
        mock_executor = AsyncMock()
        mock_executor.execute_task.side_effect = Exception("All tasks fail")
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_executor,
            max_concurrent_tasks=2
        )
        
        result = await engine.execute_graph()
        
        # Should handle all-failure scenario gracefully
        assert result.success is False
        assert result.completed_nodes == 0
        assert result.failed_nodes == 3
        assert result.total_nodes == 3
        
        # All nodes should be in failed state
        all_nodes = state_manager.get_all_nodes()
        failed_count = sum(1 for n in all_nodes if n.status == TaskStatus.FAILED)
        assert failed_count == 3

    @pytest.mark.asyncio
    async def test_very_deep_linear_graph_performance(self):
        """
        MISSING TEST: Performance test for very deep linear dependencies.
        
        Tests stack overflow and performance with deep recursion-like execution.
        """
        from src.roma.application.orchestration.parallel_execution_engine import ParallelExecutionEngine
        
        # Create very deep linear chain (100 nodes)
        nodes = []
        event_store = InMemoryEventStore()
        
        # Create root
        root = TaskNode(
            goal="Root of deep chain",
            task_type=TaskType.THINK,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )
        nodes.append(root)
        graph = DynamicTaskGraph(root_node=root)
        
        # Create 99 more nodes in linear chain
        previous = root
        for i in range(1, 100):
            node = TaskNode(
                goal=f"Deep chain task {i}",
                task_type=TaskType.WRITE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id=previous.task_id
            )
            nodes.append(node)
            await graph.add_node(node)
            previous = node
        
        state_manager = GraphStateManager(graph=graph, event_store=event_store)
        
        # Mock fast executor
        mock_executor = AsyncMock()
        mock_executor.execute_task.return_value = {"status": "completed", "result": "success"}
        
        engine = ParallelExecutionEngine(
            state_manager=state_manager,
            task_executor=mock_executor,
            max_concurrent_tasks=1  # Force sequential execution
        )
        
        import time
        start_time = time.time()
        
        result = await engine.execute_graph()
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should handle deep linear graph without issues
        assert result.success is True
        assert result.total_nodes == 100
        assert result.completed_nodes == 100
        
        # Should complete in reasonable time (< 10 seconds)
        assert execution_time < 10.0, f"Deep linear graph took too long: {execution_time}s"
        
        # Should maintain consistent final state
        all_nodes = state_manager.get_all_nodes()
        completed_count = sum(1 for n in all_nodes if n.status == TaskStatus.COMPLETED)
        assert completed_count == 100