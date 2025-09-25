"""
Test utilities for orchestration tests.

Provides common test helpers to eliminate code duplication across test files
and improve maintainability following DRY principles.
"""

import asyncio
from unittest.mock import AsyncMock

from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.services.event_store import InMemoryEventStore
from roma.domain.entities.task_node import TaskNode
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.task_type import TaskType


class TestGraphFactory:
    """Factory for creating test graphs with common patterns."""

    @staticmethod
    async def create_parallel_graph(nodes: list[TaskNode]) -> DynamicTaskGraph:
        """
        Create parallel graph: root -> [child1, child2, child3, ...].
        
        Args:
            nodes: List of TaskNode objects, first is root, rest are children
            
        Returns:
            DynamicTaskGraph with parallel structure
        """
        if not nodes:
            raise ValueError("At least one node required")

        graph = DynamicTaskGraph(root_node=nodes[0])

        # All other nodes depend on root
        root_id = nodes[0].task_id
        for node in nodes[1:]:
            child_node = node.model_copy(update={"parent_id": root_id})
            await graph.add_node(child_node)

        return graph

    @staticmethod
    async def create_linear_graph(nodes: list[TaskNode]) -> DynamicTaskGraph:
        """
        Create linear dependency graph: node1 -> node2 -> node3 -> ...
        
        Args:
            nodes: List of TaskNode objects in dependency order
            
        Returns:
            DynamicTaskGraph with linear structure
        """
        if not nodes:
            raise ValueError("At least one node required")

        graph = DynamicTaskGraph(root_node=nodes[0])

        # Create linear chain
        previous_node = nodes[0]
        for node in nodes[1:]:
            child_node = node.model_copy(update={"parent_id": previous_node.task_id})
            await graph.add_node(child_node)
            previous_node = child_node

        return graph

    @staticmethod
    async def create_deep_chain_graph(depth: int, goal_prefix: str = "Task") -> DynamicTaskGraph:
        """
        Create very deep linear chain for performance testing.
        
        Args:
            depth: Number of nodes in the chain
            goal_prefix: Prefix for task goals
            
        Returns:
            DynamicTaskGraph with deep linear structure
        """
        if depth < 1:
            raise ValueError("Depth must be at least 1")

        # Create root
        root = TaskNode(
            goal=f"{goal_prefix} 0 (root)",
            task_type=TaskType.THINK,
            node_type=NodeType.EXECUTE,
            status=TaskStatus.PENDING
        )

        graph = DynamicTaskGraph(root_node=root)

        # Create chain of specified depth
        previous = root
        for i in range(1, depth):
            node = TaskNode(
                goal=f"{goal_prefix} {i}",
                task_type=TaskType.WRITE,
                node_type=NodeType.EXECUTE,
                status=TaskStatus.PENDING,
                parent_id=previous.task_id
            )
            await graph.add_node(node)
            previous = node

        return graph


class MockExecutorFactory:
    """Factory for creating mock executors with common patterns."""

    @staticmethod
    def create_successful_executor() -> AsyncMock:
        """Create mock executor that always succeeds."""
        executor = AsyncMock()
        executor.execute_task.return_value = {"status": "completed", "result": "success"}
        return executor

    @staticmethod
    def create_failing_executor(error_message: str = "Task execution failed") -> AsyncMock:
        """Create mock executor that always fails."""
        executor = AsyncMock()
        executor.execute_task.side_effect = Exception(error_message)
        return executor

    @staticmethod
    def create_selective_failure_executor(failure_indices: list[int]) -> AsyncMock:
        """
        Create mock executor that fails on specific call indices.
        
        Args:
            failure_indices: List of call indices (0-based) that should fail
            
        Returns:
            AsyncMock that fails on specified calls
        """
        executor = AsyncMock()
        call_count = 0

        async def selective_failure(node):
            nonlocal call_count
            call_count += 1
            if (call_count - 1) in failure_indices:  # Convert to 0-based
                raise Exception(f"Task {node.task_id} failed on call {call_count}")
            return {"status": "completed", "result": "success"}

        executor.execute_task.side_effect = selective_failure
        return executor

    @staticmethod
    def create_delay_executor(delay_seconds: float = 0.1) -> AsyncMock:
        """
        Create mock executor with artificial delay for testing parallelism.
        
        Args:
            delay_seconds: Delay in seconds for each task execution
            
        Returns:
            AsyncMock with delay
        """
        executor = AsyncMock()

        async def delayed_execution(node):
            await asyncio.sleep(delay_seconds)
            return {"status": "completed", "result": f"completed_{node.task_id}"}

        executor.execute_task.side_effect = delayed_execution
        return executor


class TestStateManagerFactory:
    """Factory for creating test state managers with common configurations."""

    @staticmethod
    def create_state_manager(graph: DynamicTaskGraph) -> GraphStateManager:
        """
        Create state manager with in-memory event store.
        
        Args:
            graph: DynamicTaskGraph to manage
            
        Returns:
            GraphStateManager with in-memory event store
        """
        event_store = InMemoryEventStore()
        return GraphStateManager(graph=graph, event_store=event_store)

    @staticmethod
    def create_failing_event_store_state_manager(
        graph: DynamicTaskGraph,
        error_message: str = "Event store connection lost"
    ) -> GraphStateManager:
        """
        Create state manager with failing event store for testing error scenarios.
        
        Args:
            graph: DynamicTaskGraph to manage
            error_message: Error message for event store failures
            
        Returns:
            GraphStateManager with failing event store
        """
        failing_event_store = AsyncMock()
        failing_event_store.append = AsyncMock(side_effect=Exception(error_message))
        return GraphStateManager(graph=graph, event_store=failing_event_store)


class SampleNodeFactory:
    """Factory for creating sample nodes with common configurations."""

    @staticmethod
    def create_sample_nodes(count: int = 4) -> list[TaskNode]:
        """
        Create list of sample task nodes for testing.
        
        Args:
            count: Number of nodes to create
            
        Returns:
            List of TaskNode objects with varied types
        """
        nodes = []
        task_types = [TaskType.THINK, TaskType.RETRIEVE, TaskType.THINK, TaskType.WRITE]
        node_types = [NodeType.PLAN, NodeType.EXECUTE, NodeType.EXECUTE, NodeType.EXECUTE]
        goals = [
            "Root task - research cryptocurrency",
            "Retrieve Bitcoin price data",
            "Analyze market trends",
            "Write analysis report"
        ]

        for i in range(count):
            node = TaskNode(
                goal=goals[i % len(goals)] + f" ({i})" if i >= len(goals) else goals[i],
                task_type=task_types[i % len(task_types)],
                node_type=node_types[i % len(node_types)],
                status=TaskStatus.PENDING
            )
            nodes.append(node)

        return nodes

    @staticmethod
    def create_single_node(
        goal: str = "Test task",
        task_type: TaskType = TaskType.THINK,
        node_type: NodeType = NodeType.EXECUTE,
        status: TaskStatus = TaskStatus.PENDING
    ) -> TaskNode:
        """
        Create single task node with specified parameters.
        
        Args:
            goal: Task objective
            task_type: Type of task (RETRIEVE, WRITE, THINK)
            node_type: Node type (PLAN, EXECUTE)
            status: Initial status
            
        Returns:
            TaskNode with specified configuration
        """
        return TaskNode(
            goal=goal,
            task_type=task_type,
            node_type=node_type,
            status=status
        )


# Common test fixtures that can be imported
def create_test_environment():
    """Create common test environment with state manager and sample nodes."""
    nodes = SampleNodeFactory.create_sample_nodes(4)
    event_store = InMemoryEventStore()
    graph = DynamicTaskGraph(root_node=nodes[0])
    state_manager = GraphStateManager(graph=graph, event_store=event_store)

    return {
        'nodes': nodes,
        'event_store': event_store,
        'graph': graph,
        'state_manager': state_manager
    }
