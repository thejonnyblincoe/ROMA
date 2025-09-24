"""
Tests for DeadlockDetector.

Tests the deadlock detection service functionality including cycle detection,
deadlock analysis, and monitoring.
"""

import pytest
from unittest.mock import Mock
from uuid import uuid4

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.value_objects.node_type import NodeType
from roma.domain.value_objects.deadlock_analysis import (
    DeadlockReport, DeadlockType, DeadlockSeverity
)
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.application.services.deadlock_detector import DeadlockDetector


@pytest.fixture
def task_graph():
    """Create a task graph for testing."""
    return DynamicTaskGraph()


@pytest.fixture
def deadlock_detector(task_graph):
    """Create deadlock detector with task graph."""
    return DeadlockDetector(graph=task_graph)


@pytest.fixture
def sample_tasks():
    """Create sample tasks for testing."""
    tasks = []
    for i in range(5):
        task = TaskNode(
            task_id=f"task_{i}",
            goal=f"Test task {i}",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING
        )
        tasks.append(task)
    return tasks


@pytest.mark.skip(reason="DeadlockDetector interface has changed - tests need rewrite")
class TestDeadlockDetector:
    """Test DeadlockDetector functionality."""

    def test_initialization(self, deadlock_detector, task_graph):
        """Test deadlock detector initialization."""
        assert deadlock_detector.task_graph == task_graph
        assert deadlock_detector.deadlock_check_interval == 30.0

    @pytest.mark.asyncio
    async def test_check_for_deadlocks_no_deadlock(self, deadlock_detector, task_graph, sample_tasks):
        """Test deadlock check when no deadlock exists."""
        # Add tasks to graph without cycles
        for task in sample_tasks[:3]:
            await task_graph.add_node(task)

        # Add dependencies: task_0 -> task_1 -> task_2 (linear, no cycle)
        await task_graph.add_dependency(sample_tasks[1].task_id, sample_tasks[0].task_id)
        await task_graph.add_dependency(sample_tasks[2].task_id, sample_tasks[1].task_id)

        # Check for deadlocks
        report = await deadlock_detector.check_for_deadlocks()

        assert report.deadlock_type == DeadlockType.NO_DEADLOCK
        assert report.severity == DeadlockSeverity.NONE
        assert len(report.affected_nodes) == 0

    @pytest.mark.asyncio
    async def test_check_for_deadlocks_circular_dependency(self, deadlock_detector, task_graph, sample_tasks):
        """Test deadlock detection with circular dependencies."""
        # Add tasks to graph
        for task in sample_tasks[:3]:
            await task_graph.add_node(task)

        # Create circular dependency: task_0 -> task_1 -> task_2 -> task_0
        await task_graph.add_dependency(sample_tasks[1].task_id, sample_tasks[0].task_id)
        await task_graph.add_dependency(sample_tasks[2].task_id, sample_tasks[1].task_id)
        await task_graph.add_dependency(sample_tasks[0].task_id, sample_tasks[2].task_id)

        # Check for deadlocks
        report = await deadlock_detector.check_for_deadlocks()

        assert report.deadlock_type == DeadlockType.CIRCULAR_DEPENDENCY
        assert report.severity in [DeadlockSeverity.MEDIUM, DeadlockSeverity.HIGH]
        assert len(report.affected_nodes) == 3
        assert all(task.task_id in report.affected_nodes for task in sample_tasks[:3])

    @pytest.mark.asyncio
    async def test_check_for_deadlocks_waiting_chain(self, deadlock_detector, task_graph):
        """Test deadlock detection with long waiting chains."""
        # Create tasks in WAITING status forming a long chain
        waiting_tasks = []
        for i in range(10):
            task = TaskNode(
                task_id=f"waiting_task_{i}",
                goal=f"Waiting task {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.WAITING
            )
            waiting_tasks.append(task)
            await task_graph.add_node(task)

        # Create long dependency chain
        for i in range(9):
            await task_graph.add_dependency(
                waiting_tasks[i + 1].task_id,
                waiting_tasks[i].task_id
            )

        # Check for deadlocks
        report = await deadlock_detector.check_for_deadlocks()

        assert report.deadlock_type == DeadlockType.WAITING_CHAIN
        assert report.severity in [DeadlockSeverity.MEDIUM, DeadlockSeverity.HIGH]
        assert len(report.affected_nodes) == 10

    @pytest.mark.asyncio
    async def test_check_for_deadlocks_resource_contention(self, deadlock_detector, task_graph):
        """Test deadlock detection with resource contention."""
        # Create many tasks competing for the same resource (EXECUTING status)
        executing_tasks = []
        for i in range(15):  # More than typical max_concurrent_tasks
            task = TaskNode(
                task_id=f"executing_task_{i}",
                goal=f"Executing task {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.EXECUTING
            )
            executing_tasks.append(task)
            await task_graph.add_node(task)

        # Check for deadlocks
        report = await deadlock_detector.check_for_deadlocks()

        # Should detect potential resource contention
        assert report.deadlock_type == DeadlockType.RESOURCE_CONTENTION
        assert report.severity in [DeadlockSeverity.LOW, DeadlockSeverity.MEDIUM]
        assert len(report.affected_nodes) == 15

    @pytest.mark.asyncio
    async def test_detect_circular_dependencies(self, deadlock_detector, task_graph, sample_tasks):
        """Test specific circular dependency detection method."""
        # Add tasks to graph
        for task in sample_tasks[:4]:
            await task_graph.add_node(task)

        # Create multiple cycles
        # Cycle 1: task_0 -> task_1 -> task_0
        await task_graph.add_dependency(sample_tasks[1].task_id, sample_tasks[0].task_id)
        await task_graph.add_dependency(sample_tasks[0].task_id, sample_tasks[1].task_id)

        # Cycle 2: task_2 -> task_3 -> task_2
        await task_graph.add_dependency(sample_tasks[3].task_id, sample_tasks[2].task_id)
        await task_graph.add_dependency(sample_tasks[2].task_id, sample_tasks[3].task_id)

        cycles = deadlock_detector._detect_circular_dependencies()

        assert len(cycles) >= 2  # At least 2 cycles detected
        # Verify cycles contain expected nodes
        all_cycle_nodes = set()
        for cycle in cycles:
            all_cycle_nodes.update(cycle)

        expected_nodes = {task.task_id for task in sample_tasks[:4]}
        assert all_cycle_nodes == expected_nodes

    @pytest.mark.asyncio
    async def test_detect_waiting_chains(self, deadlock_detector, task_graph):
        """Test waiting chain detection method."""
        # Create chain of waiting tasks
        chain_tasks = []
        for i in range(8):
            task = TaskNode(
                task_id=f"chain_task_{i}",
                goal=f"Chain task {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.WAITING
            )
            chain_tasks.append(task)
            await task_graph.add_node(task)

        # Create dependency chain
        for i in range(7):
            await task_graph.add_dependency(
                chain_tasks[i + 1].task_id,
                chain_tasks[i].task_id
            )

        # Add some non-waiting tasks
        normal_task = TaskNode(
            task_id="normal_task",
            goal="Normal task",
            task_type=TaskType.THINK,
            status=TaskStatus.READY
        )
        await task_graph.add_node(normal_task)

        chains = deadlock_detector._detect_waiting_chains()

        assert len(chains) >= 1
        # Should find the chain of 8 waiting tasks
        longest_chain = max(chains, key=len)
        assert len(longest_chain) == 8

    @pytest.mark.asyncio
    async def test_detect_resource_contention(self, deadlock_detector, task_graph):
        """Test resource contention detection method."""
        # Create many executing tasks
        executing_tasks = []
        for i in range(20):
            task = TaskNode(
                task_id=f"exec_task_{i}",
                goal=f"Executing task {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.EXECUTING
            )
            executing_tasks.append(task)
            await task_graph.add_node(task)

        # Add some ready tasks waiting for execution slots
        ready_tasks = []
        for i in range(10):
            task = TaskNode(
                task_id=f"ready_task_{i}",
                goal=f"Ready task {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.READY
            )
            ready_tasks.append(task)
            await task_graph.add_node(task)

        contention = deadlock_detector._detect_resource_contention()

        assert contention is not None
        assert "executing_count" in contention
        assert "ready_count" in contention
        assert contention["executing_count"] == 20
        assert contention["ready_count"] == 10

    def test_classify_deadlock_severity_high(self, deadlock_detector):
        """Test high severity deadlock classification."""
        # Large circular dependency affecting many nodes
        large_cycle = [f"task_{i}" for i in range(15)]

        severity = deadlock_detector._classify_deadlock_severity(
            deadlock_type=DeadlockType.CIRCULAR_DEPENDENCY,
            affected_nodes=large_cycle
        )

        assert severity == DeadlockSeverity.HIGH

    def test_classify_deadlock_severity_medium(self, deadlock_detector):
        """Test medium severity deadlock classification."""
        # Medium-sized waiting chain
        medium_chain = [f"task_{i}" for i in range(8)]

        severity = deadlock_detector._classify_deadlock_severity(
            deadlock_type=DeadlockType.WAITING_CHAIN,
            affected_nodes=medium_chain
        )

        assert severity == DeadlockSeverity.MEDIUM

    def test_classify_deadlock_severity_low(self, deadlock_detector):
        """Test low severity deadlock classification."""
        # Small resource contention
        few_tasks = [f"task_{i}" for i in range(3)]

        severity = deadlock_detector._classify_deadlock_severity(
            deadlock_type=DeadlockType.RESOURCE_CONTENTION,
            affected_nodes=few_tasks
        )

        assert severity == DeadlockSeverity.LOW

    @pytest.mark.asyncio
    async def test_resolve_deadlock_circular_dependency(self, deadlock_detector, task_graph, sample_tasks):
        """Test deadlock resolution for circular dependencies."""
        # Create circular dependency
        for task in sample_tasks[:3]:
            await task_graph.add_node(task)

        await task_graph.add_dependency(sample_tasks[1].task_id, sample_tasks[0].task_id)
        await task_graph.add_dependency(sample_tasks[2].task_id, sample_tasks[1].task_id)
        await task_graph.add_dependency(sample_tasks[0].task_id, sample_tasks[2].task_id)

        # Create deadlock report
        report = DeadlockReport(
            deadlock_type=DeadlockType.CIRCULAR_DEPENDENCY,
            affected_nodes=[task.task_id for task in sample_tasks[:3]],
            severity=DeadlockSeverity.HIGH,
            details={"cycles": [[task.task_id for task in sample_tasks[:3]]]}
        )

        # Resolve deadlock
        resolution_plan = await deadlock_detector.resolve_deadlock(report)

        assert resolution_plan["strategy"] == "break_circular_dependencies"
        assert "dependencies_to_remove" in resolution_plan
        assert len(resolution_plan["dependencies_to_remove"]) >= 1

    @pytest.mark.asyncio
    async def test_resolve_deadlock_waiting_chain(self, deadlock_detector, task_graph):
        """Test deadlock resolution for waiting chains."""
        # Create waiting chain
        chain_tasks = []
        for i in range(10):
            task = TaskNode(
                task_id=f"chain_task_{i}",
                goal=f"Chain task {i}",
                task_type=TaskType.THINK,
                status=TaskStatus.WAITING
            )
            chain_tasks.append(task)
            await task_graph.add_node(task)

        # Create deadlock report
        report = DeadlockReport(
            deadlock_type=DeadlockType.WAITING_CHAIN,
            affected_nodes=[task.task_id for task in chain_tasks],
            severity=DeadlockSeverity.HIGH,
            details={"chain_length": 10}
        )

        # Resolve deadlock
        resolution_plan = await deadlock_detector.resolve_deadlock(report)

        assert resolution_plan["strategy"] == "parallelize_waiting_chain"
        assert "nodes_to_parallelize" in resolution_plan

    @pytest.mark.asyncio
    async def test_get_deadlock_statistics(self, deadlock_detector, task_graph, sample_tasks):
        """Test getting deadlock statistics."""
        # Add various tasks to create different patterns
        for task in sample_tasks:
            await task_graph.add_node(task)

        # Create some dependencies
        await task_graph.add_dependency(sample_tasks[1].task_id, sample_tasks[0].task_id)
        await task_graph.add_dependency(sample_tasks[2].task_id, sample_tasks[1].task_id)

        stats = await deadlock_detector.get_deadlock_statistics()

        assert "total_nodes" in stats
        assert "dependency_count" in stats
        assert "status_distribution" in stats
        assert "potential_bottlenecks" in stats
        assert stats["total_nodes"] == len(sample_tasks)
        assert stats["dependency_count"] >= 2

    @pytest.mark.asyncio
    async def test_enable_disable_monitoring(self, deadlock_detector):
        """Test enabling and disabling deadlock monitoring."""
        # Initially enabled by default
        assert deadlock_detector.monitoring_enabled is True

        # Disable monitoring
        deadlock_detector.disable_monitoring()
        assert deadlock_detector.monitoring_enabled is False

        # Enable monitoring
        deadlock_detector.enable_monitoring()
        assert deadlock_detector.monitoring_enabled is True

    @pytest.mark.asyncio
    async def test_check_for_deadlocks_monitoring_disabled(self, deadlock_detector, task_graph, sample_tasks):
        """Test that deadlock check returns early when monitoring is disabled."""
        # Add tasks with circular dependency
        for task in sample_tasks[:3]:
            await task_graph.add_node(task)

        await task_graph.add_dependency(sample_tasks[1].task_id, sample_tasks[0].task_id)
        await task_graph.add_dependency(sample_tasks[0].task_id, sample_tasks[1].task_id)

        # Disable monitoring
        deadlock_detector.disable_monitoring()

        # Check for deadlocks - should return no deadlock even though one exists
        report = await deadlock_detector.check_for_deadlocks()

        assert report.deadlock_type == DeadlockType.NO_DEADLOCK
        assert report.severity == DeadlockSeverity.NONE