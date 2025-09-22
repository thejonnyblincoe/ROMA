"""
Integration tests for dependency execution flow.

Tests the complete flow from SubTask.dependencies to execution ordering.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from typing import List, Dict, Any

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.agent_type import AgentType
from roma.domain.value_objects.agent_responses import SubTask, PlannerResult, AtomizerResult
from roma.domain.value_objects.node_result import NodeResult
from roma.domain.value_objects.node_action import NodeAction
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.application.orchestration.graph_state_manager import GraphStateManager
from roma.application.orchestration.execution_orchestrator import ExecutionOrchestrator
from roma.application.orchestration.task_node_processor import TaskNodeProcessor
from roma.application.services.event_store import InMemoryEventStore
from roma.application.services.context_builder_service import ContextBuilderService, TaskContext
from roma.application.services.agent_runtime_service import AgentRuntimeService
from roma.application.services.recovery_manager import RecoveryManager
from roma.domain.value_objects.config.execution_config import ExecutionConfig


class TestDependencyExecutionFlow:
    """Integration tests for dependency execution flow."""

    @pytest.fixture
    def execution_config(self) -> ExecutionConfig:
        """Create execution config."""
        return ExecutionConfig(
            max_concurrent_nodes=2,
            max_iterations=10,
            max_subtasks_per_node=10
        )

    @pytest.fixture
    def mock_agent_runtime(self):
        """Create mock agent runtime service."""
        service = AsyncMock(spec=AgentRuntimeService)

        # Mock atomizer to always return PLAN
        atomizer_result = AtomizerResult(
            is_atomic=False,
            node_type=NodeType.PLAN,
            reasoning="Task needs decomposition for testing"
        )
        atomizer_envelope = Mock()
        atomizer_envelope.result = atomizer_result

        # Mock planner to return subtasks with dependencies
        planner_result = PlannerResult(subtasks=[
            SubTask(
                goal="First step: gather data",
                task_type=TaskType.RETRIEVE,
                dependencies=[]
            ),
            SubTask(
                goal="Second step: analyze data",
                task_type=TaskType.THINK,
                dependencies=["sub_0"]
            ),
            SubTask(
                goal="Third step: write report",
                task_type=TaskType.WRITE,
                dependencies=["sub_0", "sub_1"]
            )
        ])
        planner_envelope = Mock()
        planner_envelope.result = planner_result

        async def mock_execute_agent(agent, task, context, agent_type):
            if agent_type == AgentType.ATOMIZER:
                return atomizer_envelope
            elif agent_type == AgentType.PLANNER:
                return planner_envelope
            elif agent_type == AgentType.EXECUTOR:
                # Return simple success envelope
                executor_envelope = Mock()
                executor_envelope.result = f"Executed: {task.goal}"
                return executor_envelope
            else:
                return Mock()

        service.execute_agent = mock_execute_agent
        service.get_agent = AsyncMock(return_value=Mock())
        return service

    @pytest.fixture
    def mock_context_builder(self):
        """Create mock context builder."""
        builder = AsyncMock(spec=ContextBuilderService)

        async def mock_build_context(task, overall_objective, execution_metadata=None):
            return TaskContext(
                task=task,
                overall_objective=overall_objective,
                execution_metadata=execution_metadata or {}
            )

        builder.build_context = mock_build_context
        return builder

    @pytest.fixture
    def mock_recovery_manager(self):
        """Create mock recovery manager."""
        return AsyncMock(spec=RecoveryManager)

    @pytest.fixture
    def orchestrator(
        self,
        execution_config: ExecutionConfig,
        mock_agent_runtime: AgentRuntimeService,
        mock_context_builder: ContextBuilderService,
        mock_recovery_manager: RecoveryManager
    ):
        """Create ExecutionOrchestrator with all dependencies."""
        graph = DynamicTaskGraph()
        event_store = InMemoryEventStore()
        graph_state_manager = GraphStateManager(graph, event_store)

        task_node_processor = TaskNodeProcessor(
            mock_agent_runtime,
            mock_context_builder,
            mock_recovery_manager
        )

        orchestrator = ExecutionOrchestrator(
            graph_state_manager=graph_state_manager,
            node_processor=task_node_processor,
            execution_config=execution_config
        )

        return orchestrator

    @pytest.mark.asyncio
    async def test_dependency_execution_order(self, orchestrator: ExecutionOrchestrator):
        """Test that tasks execute in dependency order."""
        # Create root task that will be decomposed
        root_task = TaskNode(
            task_id="root",
            goal="Complex task requiring ordered execution",
            task_type=TaskType.THINK
        )

        execution_result = await orchestrator.execute_task(root_task, "Test dependency ordering")

        # Verify execution was successful
        assert execution_result.success

        # Get all nodes from graph
        all_nodes = orchestrator.graph_state_manager.get_all_nodes()
        subtasks = [node for node in all_nodes if node.parent_id == "root"]

        # Should have created 3 subtasks
        assert len(subtasks) == 3

        # Find each subtask by goal content
        retrieve_task = next(node for node in subtasks if "gather data" in node.goal)
        think_task = next(node for node in subtasks if "analyze data" in node.goal)
        write_task = next(node for node in subtasks if "write report" in node.goal)

        # Verify dependencies were set correctly
        assert len(retrieve_task.dependencies) == 0
        assert retrieve_task.task_id in think_task.dependencies
        assert retrieve_task.task_id in write_task.dependencies
        assert think_task.task_id in write_task.dependencies

        # All should be completed by end of execution
        assert all(node.status == TaskStatus.COMPLETED for node in subtasks)

    @pytest.mark.asyncio
    async def test_parallel_execution_with_dependencies(self, orchestrator: ExecutionOrchestrator):
        """Test that independent tasks can execute in parallel while respecting dependencies."""
        # Create a more complex dependency structure for parallel testing
        # This would need a more sophisticated mock setup but demonstrates the concept

        root_task = TaskNode(
            task_id="parallel_root",
            goal="Task with parallel branches",
            task_type=TaskType.THINK
        )

        # Add the root task
        await orchestrator.graph_state_manager.add_node(root_task)

        # Manually create a diamond dependency pattern: A -> B,C -> D
        task_a = TaskNode(task_id="A", goal="Start", task_type=TaskType.RETRIEVE)
        task_b = TaskNode(task_id="B", goal="Branch 1", task_type=TaskType.THINK, dependencies=frozenset(["A"]))
        task_c = TaskNode(task_id="C", goal="Branch 2", task_type=TaskType.THINK, dependencies=frozenset(["A"]))
        task_d = TaskNode(task_id="D", goal="End", task_type=TaskType.WRITE, dependencies=frozenset(["B", "C"]))

        for task in [task_a, task_b, task_c, task_d]:
            await orchestrator.graph_state_manager.add_node(task)

        # Execute one iteration to see readiness
        ready_nodes = orchestrator.graph_state_manager.get_ready_nodes()

        # Initially only A should be ready
        ready_ids = {node.task_id for node in ready_nodes}
        assert ready_ids == {"A"}

        # Complete A
        await orchestrator.graph_state_manager.transition_node_status("A", TaskStatus.COMPLETED)

        # Now B and C should both be ready (can execute in parallel)
        ready_nodes = orchestrator.graph_state_manager.get_ready_nodes()
        ready_ids = {node.task_id for node in ready_nodes}
        assert ready_ids == {"B", "C"}

        # Complete both B and C
        await orchestrator.graph_state_manager.transition_node_status("B", TaskStatus.COMPLETED)
        await orchestrator.graph_state_manager.transition_node_status("C", TaskStatus.COMPLETED)

        # Now D should be ready
        ready_nodes = orchestrator.graph_state_manager.get_ready_nodes()
        ready_ids = {node.task_id for node in ready_nodes}
        assert ready_ids == {"D"}

    @pytest.mark.asyncio
    async def test_failure_handling_with_dependencies(self, orchestrator: ExecutionOrchestrator):
        """Test that dependency failures are handled correctly."""
        # Create tasks with dependencies
        task_a = TaskNode(task_id="fail_a", goal="Will fail", task_type=TaskType.RETRIEVE)
        task_b = TaskNode(task_id="fail_b", goal="Depends on A", task_type=TaskType.THINK,
                         dependencies=frozenset(["fail_a"]))

        await orchestrator.graph_state_manager.add_node(task_a)
        await orchestrator.graph_state_manager.add_node(task_b)

        # Fail task A
        await orchestrator.graph_state_manager.transition_node_status("fail_a", TaskStatus.FAILED)

        # Task B should still be pending (not ready) since its dependency failed
        ready_nodes = orchestrator.graph_state_manager.get_ready_nodes()
        ready_ids = {node.task_id for node in ready_nodes}
        assert "fail_b" not in ready_ids

        # B should remain pending since its dependency is failed, not completed
        task_b_current = orchestrator.graph_state_manager.get_node_by_id("fail_b")
        assert task_b_current.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_no_infinite_waiting_on_failed_dependencies(self, orchestrator: ExecutionOrchestrator):
        """Test that execution doesn't hang when dependencies fail."""
        # This test ensures that the execution completes even when some dependencies fail
        # The exact behavior may depend on the recovery strategy

        task_a = TaskNode(task_id="hang_a", goal="Will fail", task_type=TaskType.RETRIEVE)
        task_b = TaskNode(task_id="hang_b", goal="Depends on failed A", task_type=TaskType.THINK,
                         dependencies=frozenset(["hang_a"]))

        await orchestrator.graph_state_manager.add_node(task_a)
        await orchestrator.graph_state_manager.add_node(task_b)

        # Start with both pending
        ready_nodes = orchestrator.graph_state_manager.get_ready_nodes()
        assert len([n for n in ready_nodes if n.task_id == "hang_a"]) == 1
        assert len([n for n in ready_nodes if n.task_id == "hang_b"]) == 0

        # Fail A
        await orchestrator.graph_state_manager.transition_node_status("hang_a", TaskStatus.FAILED)

        # No nodes should be ready now (A failed, B can't proceed)
        ready_nodes = orchestrator.graph_state_manager.get_ready_nodes()
        assert len(ready_nodes) == 0

        # This represents a scenario where execution should terminate
        # since no progress can be made

    @pytest.mark.asyncio
    async def test_aggregation_waits_for_all_children(self, orchestrator: ExecutionOrchestrator):
        """Test that parent aggregation waits for all children to complete."""
        # Add a parent task
        parent = TaskNode(task_id="agg_parent", goal="Parent task", task_type=TaskType.THINK)
        await orchestrator.graph_state_manager.add_node(parent)

        # Add child tasks (simulating what would happen after planning)
        child1 = TaskNode(task_id="agg_child1", goal="Child 1", task_type=TaskType.RETRIEVE, parent_id="agg_parent")
        child2 = TaskNode(task_id="agg_child2", goal="Child 2", task_type=TaskType.THINK, parent_id="agg_parent")
        child3 = TaskNode(task_id="agg_child3", goal="Child 3", task_type=TaskType.WRITE, parent_id="agg_parent")

        for child in [child1, child2, child3]:
            await orchestrator.graph_state_manager.add_node(child)

        # Complete only some children
        await orchestrator.graph_state_manager.transition_node_status("agg_child1", TaskStatus.COMPLETED)
        await orchestrator.graph_state_manager.transition_node_status("agg_child2", TaskStatus.COMPLETED)

        # Parent should not be ready for aggregation yet
        assert not orchestrator._all_children_completed("agg_parent")

        # Complete the last child
        await orchestrator.graph_state_manager.transition_node_status("agg_child3", TaskStatus.COMPLETED)

        # Now parent should be ready for aggregation
        assert orchestrator._all_children_completed("agg_parent")