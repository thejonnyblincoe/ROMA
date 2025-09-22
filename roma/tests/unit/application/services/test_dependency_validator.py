"""
Test Dependency Validator Service.

Tests the pre-execution dependency validation functionality.
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from roma.domain.entities.task_node import TaskNode
from roma.domain.value_objects.task_type import TaskType
from roma.domain.value_objects.task_status import TaskStatus
from roma.domain.graph.dynamic_task_graph import DynamicTaskGraph
from roma.application.services.dependency_validator import (
    DependencyValidator,
    DependencyValidationResult,
    DependencyValidationError
)
from roma.domain.value_objects.dependency_status import DependencyStatus
from roma.application.services.recovery_manager import RecoveryManager


class TestDependencyValidator:
    """Test dependency validation service."""

    @pytest.fixture
    def validator(self) -> DependencyValidator:
        """Create dependency validator."""
        return DependencyValidator()

    @pytest.fixture
    def permissive_validator(self) -> DependencyValidator:
        """Create validator that allows pending dependencies."""
        return DependencyValidator(allow_pending_dependencies=True)

    @pytest.fixture
    def graph(self) -> DynamicTaskGraph:
        """Create empty task graph."""
        return DynamicTaskGraph()

    @pytest_asyncio.fixture
    async def populated_graph(self) -> DynamicTaskGraph:
        """Create graph with test nodes."""
        graph = DynamicTaskGraph()

        # Create test nodes
        node_a = TaskNode(task_id="A", goal="First task", task_type=TaskType.THINK, status=TaskStatus.COMPLETED)
        node_b = TaskNode(task_id="B", goal="Second task", task_type=TaskType.THINK, status=TaskStatus.FAILED)
        node_c = TaskNode(task_id="C", goal="Third task", task_type=TaskType.THINK, status=TaskStatus.PENDING)
        node_d = TaskNode(task_id="D", goal="Fourth task", task_type=TaskType.THINK, status=TaskStatus.EXECUTING)

        # Node with dependencies
        node_e = TaskNode(
            task_id="E",
            goal="Depends on A and B",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING,
            dependencies=frozenset(["A", "B"])
        )

        # Node with missing dependency
        node_f = TaskNode(
            task_id="F",
            goal="Depends on missing node",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING,
            dependencies=frozenset(["MISSING"])
        )

        # Add all nodes
        for node in [node_a, node_b, node_c, node_d, node_e, node_f]:
            await graph.add_node(node)

        return graph

    def test_validator_initialization(self):
        """Test validator initialization with different settings."""
        # Default validator
        validator = DependencyValidator()
        assert not validator.allow_pending_dependencies

        # Permissive validator
        permissive = DependencyValidator(allow_pending_dependencies=True)
        assert permissive.allow_pending_dependencies

    def test_validation_result_creation(self):
        """Test DependencyValidationResult creation and properties."""
        # Valid result
        result = DependencyValidationResult("test_node", is_valid=True)
        assert result.node_id == "test_node"
        assert result.is_valid
        assert not result.has_issues
        assert isinstance(result.validated_at, datetime)

        # Invalid result with issues
        result = DependencyValidationResult(
            "test_node",
            is_valid=False,
            missing_dependencies={"missing1"},
            failed_dependencies={"failed1"},
            pending_dependencies={"pending1"}
        )
        assert not result.is_valid
        assert result.has_issues
        assert "missing1" in result.missing_dependencies
        assert "failed1" in result.failed_dependencies
        assert "pending1" in result.pending_dependencies

    def test_validation_result_to_dict(self):
        """Test conversion of validation result to dictionary."""
        result = DependencyValidationResult(
            "test_node",
            is_valid=False,
            missing_dependencies={"missing1"},
            validation_message="Test error"
        )

        result_dict = result.to_dict()
        assert result_dict["node_id"] == "test_node"
        assert not result_dict["is_valid"]
        assert result_dict["missing_dependencies"] == ["missing1"]
        assert result_dict["validation_message"] == "Test error"
        assert "validated_at" in result_dict

    def test_validate_node_no_dependencies(self, validator: DependencyValidator, graph: DynamicTaskGraph):
        """Test validation of node with no dependencies."""
        node = TaskNode(task_id="simple", goal="Simple task", task_type=TaskType.THINK)

        result = validator.validate_node_dependencies(node, graph)

        assert result.is_valid
        assert result.node_id == "simple"
        assert not result.has_issues
        assert "No dependencies" in result.validation_message

    @pytest.mark.asyncio
    async def test_validate_node_with_completed_dependencies(
        self,
        validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test validation of node with completed dependencies."""
        # Node that depends only on completed node A
        node = TaskNode(
            task_id="depends_on_completed",
            goal="Depends on A",
            task_type=TaskType.THINK,
            dependencies=frozenset(["A"])
        )

        result = validator.validate_node_dependencies(node, populated_graph)

        assert result.is_valid
        assert not result.has_issues
        assert "All dependencies satisfied" in result.validation_message

    @pytest.mark.asyncio
    async def test_validate_node_with_failed_dependencies(
        self,
        validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test validation of node with failed dependencies."""
        # Node that depends on failed node B
        node = TaskNode(
            task_id="depends_on_failed",
            goal="Depends on B",
            task_type=TaskType.THINK,
            dependencies=frozenset(["B"])
        )

        result = validator.validate_node_dependencies(node, populated_graph)

        assert not result.is_valid
        assert result.has_issues
        assert "B" in result.failed_dependencies
        assert "failed dependencies" in result.validation_message

    @pytest.mark.asyncio
    async def test_validate_node_with_missing_dependencies(
        self,
        validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test validation of node with missing dependencies."""
        # Node F already has missing dependency in populated_graph
        node_f = populated_graph.get_node("F")

        result = validator.validate_node_dependencies(node_f, populated_graph)

        assert not result.is_valid
        assert result.has_issues
        assert "MISSING" in result.missing_dependencies
        assert "missing dependencies" in result.validation_message

    @pytest.mark.asyncio
    async def test_validate_node_with_pending_dependencies_strict(
        self,
        validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test validation of node with pending dependencies (strict mode)."""
        # Node that depends on pending node C
        node = TaskNode(
            task_id="depends_on_pending",
            goal="Depends on C",
            task_type=TaskType.THINK,
            dependencies=frozenset(["C"])
        )

        result = validator.validate_node_dependencies(node, populated_graph)

        assert not result.is_valid
        assert result.has_issues
        assert "C" in result.pending_dependencies
        assert "incomplete dependencies" in result.validation_message

    @pytest.mark.asyncio
    async def test_validate_node_with_pending_dependencies_permissive(
        self,
        permissive_validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test validation of node with pending dependencies (permissive mode)."""
        # Node that depends on pending node C
        node = TaskNode(
            task_id="depends_on_pending",
            goal="Depends on C",
            task_type=TaskType.THINK,
            dependencies=frozenset(["C"])
        )

        result = permissive_validator.validate_node_dependencies(node, populated_graph)

        assert result.is_valid  # Should pass in permissive mode
        assert "C" in result.pending_dependencies
        assert "All dependencies satisfied" in result.validation_message

    @pytest.mark.asyncio
    async def test_validate_node_with_mixed_dependencies(
        self,
        validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test validation of node with mixed dependency states."""
        # Node E already has mixed dependencies (A=completed, B=failed)
        node_e = populated_graph.get_node("E")

        result = validator.validate_node_dependencies(node_e, populated_graph)

        assert not result.is_valid
        assert result.has_issues
        assert "B" in result.failed_dependencies  # Failed dependency should be flagged
        # A should not be in any problem sets since it's completed

    @pytest.mark.asyncio
    async def test_validate_ready_nodes(
        self,
        validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test validation of multiple ready nodes."""
        # Create nodes with different dependency states
        nodes = [
            TaskNode(task_id="no_deps", goal="No deps", task_type=TaskType.THINK),
            TaskNode(task_id="good_deps", goal="Good deps", task_type=TaskType.THINK, dependencies=frozenset(["A"])),
            TaskNode(task_id="bad_deps", goal="Bad deps", task_type=TaskType.THINK, dependencies=frozenset(["B"])),
        ]

        results = validator.validate_ready_nodes(nodes, populated_graph)

        assert len(results) == 3
        assert results[0].is_valid  # No dependencies
        assert results[1].is_valid  # Depends on completed A
        assert not results[2].is_valid  # Depends on failed B

    @pytest.mark.asyncio
    async def test_get_executable_nodes(
        self,
        validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test filtering of ready nodes to executable nodes."""
        # Create mix of valid and invalid nodes
        nodes = [
            TaskNode(task_id="executable1", goal="No deps", task_type=TaskType.THINK),
            TaskNode(task_id="executable2", goal="Good deps", task_type=TaskType.THINK, dependencies=frozenset(["A"])),
            TaskNode(task_id="not_executable", goal="Bad deps", task_type=TaskType.THINK, dependencies=frozenset(["B"])),
        ]

        executable = await validator.get_executable_nodes(nodes, populated_graph)

        assert len(executable) == 2
        executable_ids = {node.task_id for node in executable}
        assert "executable1" in executable_ids
        assert "executable2" in executable_ids
        assert "not_executable" not in executable_ids

    @pytest.mark.asyncio
    async def test_validate_graph_integrity_healthy(self, validator: DependencyValidator):
        """Test graph integrity validation for healthy graph."""
        graph = DynamicTaskGraph()

        # Create simple healthy graph
        node_a = TaskNode(task_id="A", goal="First", task_type=TaskType.THINK, status=TaskStatus.COMPLETED)
        node_b = TaskNode(task_id="B", goal="Second", task_type=TaskType.THINK, status=TaskStatus.COMPLETED, dependencies=frozenset(["A"]))

        await graph.add_node(node_a)
        await graph.add_node(node_b)

        integrity = validator.validate_graph_integrity(graph)

        assert integrity["is_healthy"]
        assert integrity["status"] == "healthy"
        assert len(integrity["issues"]) == 0
        assert len(integrity["warnings"]) == 0

    @pytest.mark.asyncio
    async def test_validate_graph_integrity_with_cycles(self, validator: DependencyValidator):
        """Test that dependency validator works with cycle prevention."""
        graph = DynamicTaskGraph()

        # Create nodes
        node_a = TaskNode(task_id="A", goal="First", task_type=TaskType.THINK)
        node_b = TaskNode(task_id="B", goal="Second", task_type=TaskType.THINK)

        await graph.add_node(node_a)
        await graph.add_node(node_b)

        # Create first dependency
        await graph.add_dependency_edge("A", "B")

        # Test that cycle creation is prevented (this is good behavior)
        with pytest.raises(ValueError, match="would create a cycle"):
            await graph.add_dependency_edge("B", "A")

        # Validate graph integrity (should be healthy since no cycles exist)
        integrity = validator.validate_graph_integrity(graph)

        assert integrity["is_healthy"]
        assert integrity["status"] == "healthy"
        assert "Circular dependencies" not in str(integrity["issues"])

    @pytest.mark.asyncio
    async def test_validate_graph_integrity_with_orphans(self, validator: DependencyValidator):
        """Test graph integrity validation with orphaned dependencies."""
        graph = DynamicTaskGraph()

        # Create node with dependency on non-existent node
        node_with_orphan = TaskNode(
            task_id="orphan_user",
            goal="Uses orphan",
            task_type=TaskType.THINK,
            dependencies=frozenset(["MISSING_NODE"])
        )

        await graph.add_node(node_with_orphan)

        integrity = validator.validate_graph_integrity(graph)

        assert len(integrity["warnings"]) > 0
        assert len(integrity["orphaned_dependencies"]) > 0
        assert "MISSING_NODE" in integrity["orphaned_dependencies"]

    @pytest.mark.asyncio
    async def test_validate_graph_integrity_with_failed_dependency_chains(self, validator: DependencyValidator):
        """Test graph integrity validation with failed dependency chains."""
        graph = DynamicTaskGraph()

        # Create failed node and dependent pending node
        failed_node = TaskNode(task_id="failed", goal="Failed task", task_type=TaskType.THINK, status=TaskStatus.FAILED)
        waiting_node = TaskNode(
            task_id="waiting",
            goal="Waiting task",
            task_type=TaskType.THINK,
            status=TaskStatus.PENDING,
            dependencies=frozenset(["failed"])
        )

        await graph.add_node(failed_node)
        await graph.add_node(waiting_node)

        integrity = validator.validate_graph_integrity(graph)

        assert len(integrity["warnings"]) > 0
        assert len(integrity["nodes_with_failed_deps"]) > 0
        assert "waiting" in integrity["nodes_with_failed_deps"]
        assert "recovery strategies" in str(integrity["recommendations"])

    # Enhanced Recovery Integration Tests

    @pytest.fixture
    def recovery_manager(self) -> RecoveryManager:
        """Create recovery manager for testing."""
        return RecoveryManager()

    @pytest.fixture
    def validator_with_recovery(self, recovery_manager: RecoveryManager) -> DependencyValidator:
        """Create validator with recovery manager."""
        return DependencyValidator(recovery_manager=recovery_manager)

    def test_dependency_status_enum(self):
        """Test DependencyStatus enum functionality."""
        # Test status properties
        assert DependencyStatus.COMPLETED.is_satisfied
        assert not DependencyStatus.COMPLETED.is_blocking
        assert not DependencyStatus.COMPLETED.is_pending

        assert not DependencyStatus.FAILED.is_satisfied
        assert DependencyStatus.FAILED.is_blocking
        assert not DependencyStatus.FAILED.is_pending

        assert not DependencyStatus.PENDING.is_satisfied
        assert not DependencyStatus.PENDING.is_blocking
        assert DependencyStatus.PENDING.is_pending

        assert not DependencyStatus.MISSING.is_satisfied
        assert DependencyStatus.MISSING.is_blocking
        assert not DependencyStatus.MISSING.is_pending

    def test_dependency_status_from_task_status(self):
        """Test conversion from TaskStatus to DependencyStatus."""
        assert DependencyStatus.from_task_status(TaskStatus.COMPLETED) == DependencyStatus.COMPLETED
        assert DependencyStatus.from_task_status(TaskStatus.FAILED) == DependencyStatus.FAILED
        assert DependencyStatus.from_task_status(TaskStatus.PENDING) == DependencyStatus.PENDING
        assert DependencyStatus.from_task_status(TaskStatus.EXECUTING) == DependencyStatus.EXECUTING
        assert DependencyStatus.from_task_status(TaskStatus.READY) == DependencyStatus.READY

    def test_validator_with_recovery_manager_initialization(self, validator_with_recovery: DependencyValidator):
        """Test validator initialization with recovery manager."""
        assert validator_with_recovery.recovery_manager is not None
        assert isinstance(validator_with_recovery.recovery_manager, RecoveryManager)

    @pytest.mark.asyncio
    async def test_dependency_failure_recovery_integration(
        self,
        validator_with_recovery: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test dependency failure recovery through RecoveryManager integration."""
        # Create nodes with failed dependencies
        nodes = [
            TaskNode(task_id="blocked1", goal="Blocked by B", task_type=TaskType.THINK, dependencies=frozenset(["B"])),
            TaskNode(task_id="blocked2", goal="Blocked by B", task_type=TaskType.THINK, dependencies=frozenset(["B"])),
        ]

        # This should trigger recovery handling for failed dependency B
        executable = await validator_with_recovery.get_executable_nodes(nodes, populated_graph)

        # Both nodes should be filtered out due to failed dependency
        assert len(executable) == 0

        # Verify that recovery manager was invoked (we can't easily test the side effects without mocks)
        # But we can verify the functionality doesn't crash and properly filters

    @pytest.mark.asyncio
    async def test_mixed_dependency_states_with_recovery(
        self,
        validator_with_recovery: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test handling of mixed dependency states with recovery integration."""
        nodes = [
            # Valid node (no dependencies)
            TaskNode(task_id="valid", goal="Valid", task_type=TaskType.THINK),
            # Valid node (completed dependency)
            TaskNode(task_id="valid_dep", goal="Valid dep", task_type=TaskType.THINK, dependencies=frozenset(["A"])),
            # Invalid node (failed dependency)
            TaskNode(task_id="invalid_dep", goal="Invalid dep", task_type=TaskType.THINK, dependencies=frozenset(["B"])),
            # Invalid node (missing dependency)
            TaskNode(task_id="missing_dep", goal="Missing dep", task_type=TaskType.THINK, dependencies=frozenset(["MISSING"])),
        ]

        executable = await validator_with_recovery.get_executable_nodes(nodes, populated_graph)

        # Only valid nodes should be executable
        assert len(executable) == 2
        executable_ids = {node.task_id for node in executable}
        assert "valid" in executable_ids
        assert "valid_dep" in executable_ids
        assert "invalid_dep" not in executable_ids
        assert "missing_dep" not in executable_ids

    @pytest.mark.asyncio
    async def test_get_executable_nodes_async_signature(
        self,
        validator: DependencyValidator,
        populated_graph: DynamicTaskGraph
    ):
        """Test that get_executable_nodes properly handles async signature."""
        nodes = [
            TaskNode(task_id="test", goal="Test", task_type=TaskType.THINK, dependencies=frozenset(["A"]))
        ]

        # This should work with await
        executable = await validator.get_executable_nodes(nodes, populated_graph)
        assert len(executable) == 1

    @pytest.mark.asyncio
    async def test_recovery_integration_with_permissive_mode(
        self,
        recovery_manager: RecoveryManager
    ):
        """Test recovery integration with permissive dependency validation."""
        validator = DependencyValidator(
            allow_pending_dependencies=True,
            recovery_manager=recovery_manager
        )

        graph = DynamicTaskGraph()

        # Create pending dependency
        pending_node = TaskNode(task_id="pending", goal="Pending", task_type=TaskType.THINK, status=TaskStatus.PENDING)
        dependent_node = TaskNode(task_id="dependent", goal="Depends on pending", task_type=TaskType.THINK, dependencies=frozenset(["pending"]))

        await graph.add_node(pending_node)

        # In permissive mode, should allow execution with pending dependencies
        executable = await validator.get_executable_nodes([dependent_node], graph)
        assert len(executable) == 1  # Should be executable in permissive mode

    def test_dependency_validation_error_handling(self):
        """Test DependencyValidationError functionality."""
        validation_details = {"failed_deps": ["dep1", "dep2"], "reason": "Dependencies failed"}
        error = DependencyValidationError("Validation failed", validation_details)

        assert str(error) == "Validation failed"
        assert error.validation_details == validation_details
        assert error.validation_details["failed_deps"] == ["dep1", "dep2"]

    @pytest.mark.asyncio
    async def test_large_dependency_graph_performance(self, validator: DependencyValidator):
        """Test validator performance with large dependency graphs."""
        graph = DynamicTaskGraph()

        # Create a large graph with many dependencies
        nodes = []
        for i in range(100):
            if i == 0:
                node = TaskNode(task_id=f"node_{i}", goal=f"Task {i}", task_type=TaskType.THINK, status=TaskStatus.COMPLETED)
            else:
                # Each node depends on the previous one
                deps = frozenset([f"node_{i-1}"])
                node = TaskNode(task_id=f"node_{i}", goal=f"Task {i}", task_type=TaskType.THINK, dependencies=deps)
            nodes.append(node)
            await graph.add_node(node)

        # Validate all nodes - should be fast
        import time
        start_time = time.time()

        validation_results = validator.validate_ready_nodes(nodes, graph)

        end_time = time.time()
        validation_time = end_time - start_time

        # Should complete in reasonable time (less than 1 second for 100 nodes)
        assert validation_time < 1.0
        assert len(validation_results) == 100

    @pytest.mark.asyncio
    async def test_concurrent_validation_safety(self, validator: DependencyValidator):
        """Test that validator handles concurrent operations safely."""
        graph = DynamicTaskGraph()

        # Create test nodes
        completed_node = TaskNode(task_id="completed", goal="Completed", task_type=TaskType.THINK, status=TaskStatus.COMPLETED)
        await graph.add_node(completed_node)

        # Create multiple nodes that depend on the completed node
        dependent_nodes = []
        for i in range(10):
            node = TaskNode(
                task_id=f"dep_{i}",
                goal=f"Dependent {i}",
                task_type=TaskType.THINK,
                dependencies=frozenset(["completed"])
            )
            dependent_nodes.append(node)

        # Run multiple validations concurrently
        import asyncio
        tasks = [validator.get_executable_nodes([node], graph) for node in dependent_nodes]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 10
        for result in results:
            assert len(result) == 1  # Each should have one executable node